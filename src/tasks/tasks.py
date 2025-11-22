"""Celery task definitions."""

import logging
import time
from pathlib import Path

import torch
from celery import Task

from src.tasks.worker import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="src.tasks.tasks.generate_pattern",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
)
def generate_pattern_task(
    self: Task,
    producer_name: str = None,
    style_profile: dict = None,
    producer_style: str = None,  # Legacy parameter for backward compatibility
    bars: int = 4,
    tempo: int = 120,
    time_signature: tuple = (4, 4),
    humanize: bool = True,
    pattern_type: str = "verse",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict:
    """
    Generate drum pattern using ML model with dynamic style transfer.

    WEEK 2 UPDATE: Now supports dynamic producer names with automatic
    style research and transfer. Legacy producer_style parameter still
    supported for backward compatibility.

    Complete pipeline:
    1. Validates style profile or legacy style name
    2. Loads model (with LRU caching and GPU fallback)
    3. Generates tokens autoregressively
    4. Converts tokens to MIDI notes
    5. Applies dynamic style transfer using researched parameters
    6. Exports to MIDI file with metadata
    7. Cleans up GPU memory

    Args:
        producer_name: Producer name (new, e.g., "Timbaland", "Aphex Twin")
        style_profile: Complete style profile from ProducerResearchAgent
        producer_style: Legacy style name (e.g., "J Dilla") - deprecated
        bars: Number of bars to generate (1-32)
        tempo: Tempo in BPM (40-300)
        time_signature: Time signature as (numerator, denominator)
        humanize: Apply humanization/style transfer algorithms
        pattern_type: Pattern type (intro, verse, chorus, etc.)
        temperature: Sampling temperature (0.5-1.5)
        top_k: Top-k sampling parameter (0-100)
        top_p: Nucleus sampling parameter (0.0-1.0)

    Returns:
        Dict with:
            - midi_file: Absolute path to generated MIDI file
            - duration_seconds: Total generation time
            - tokens_generated: Number of tokens
            - producer_name: Producer name used
            - style_description: Human-readable style description
            - bars: Number of bars
            - tempo: BPM
            - device: Device used (cuda/cpu)
            - gpu_memory_mb: GPU memory usage (if applicable)

    Raises:
        Exception: Re-raised after max retries for Celery to handle
    """
    start_time = time.time()
    device = None
    model = None

    try:
        # Update task state: Starting
        self.update_state(
            state="PROGRESS", meta={"progress": 5, "status": "Validating parameters..."}
        )

        # Determine producer name and style profile
        use_dynamic_style = style_profile is not None

        if use_dynamic_style:
            # Week 2: Dynamic producer with researched style
            actual_producer_name = producer_name or style_profile.get("producer_name", "Unknown")
            logger.info(
                f"Task {self.request.id}: Generating {bars} bars of {actual_producer_name} "
                f"@ {tempo} BPM using dynamic style transfer "
                f"(quality: {style_profile.get('research_quality', 'unknown')})"
            )
        else:
            # Legacy: Fixed style
            actual_producer_name = producer_style or producer_name or "J Dilla"
            logger.info(
                f"Task {self.request.id}: Generating {bars} bars of {actual_producer_name} "
                f"@ {tempo} BPM using legacy fixed style"
            )

        # For legacy mode or fallback, normalize style name
        if not use_dynamic_style or not style_profile:
            from src.models.styles import (
                StyleNotFoundError,
                get_model_path,
                get_numeric_style_id,
                normalize_style_name,
                validate_tempo_for_style,
            )

            try:
                normalized_style = normalize_style_name(actual_producer_name)
                logger.debug(f"Normalized style: {actual_producer_name} -> {normalized_style}")
            except StyleNotFoundError as e:
                logger.warning(f"Style not in legacy registry, using default: {e}")
                normalized_style = "J Dilla"  # Fallback

            # Validate tempo (warns if outside preferred range)
            validate_tempo_for_style(normalized_style, tempo, warn_only=True)
        else:
            # Dynamic mode: use style_profile
            normalized_style = actual_producer_name

        # Update task state: Loading model
        self.update_state(
            state="PROGRESS",
            meta={"progress": 15, "status": f"Loading {normalized_style} model..."},
        )

        # Get model path and numeric style ID
        style_id = get_numeric_style_id(normalized_style)
        model_path = get_model_path(normalized_style)

        logger.debug(f"Model path: {model_path}, style_id: {style_id}")

        # Load model with GPU/CPU fallback
        from src.inference.model_loader import (
            clear_gpu_cache,
            detect_device,
            get_gpu_memory_info,
            load_model,
        )

        try:
            # Auto-detect device
            device = detect_device()

            # Try loading real model
            if model_path.exists():
                logger.info(f"Loading trained model from {model_path}")
                model, metadata = load_model(model_path, device=device)
                device = metadata["device"]  # May have fallen back to CPU
            else:
                # Model doesn't exist yet - use mock model
                logger.warning(
                    f"Model checkpoint not found at {model_path}. "
                    "Using MockDrumModel for testing. "
                    "Trained models will be available after Phase 6 (Training)."
                )
                from src.inference.mock import MockDrumModel

                model = MockDrumModel()
                model.to(device)
                model.eval()
                metadata = {"vocab_size": 500, "device": device, "model_type": "mock"}

        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM during model loading, falling back to CPU")
            clear_gpu_cache()
            device = "cpu"

            # Retry loading on CPU
            if model_path.exists():
                model, metadata = load_model(model_path, device="cpu")
            else:
                from src.inference.mock import MockDrumModel

                model = MockDrumModel()
                model.to("cpu")
                model.eval()
                metadata = {"vocab_size": 500, "device": "cpu", "model_type": "mock"}

        logger.info(f"Model loaded on {device}")

        # Log GPU memory if available
        if device == "cuda":
            gpu_info = get_gpu_memory_info()
            logger.info(
                f"GPU: {gpu_info.get('device_name', 'Unknown')}, "
                f"{gpu_info.get('allocated_gb', 0):.2f}GB allocated"
            )

        # Update task state: Generating
        self.update_state(
            state="PROGRESS",
            meta={"progress": 40, "status": f"Generating {bars}-bar pattern...", "device": device},
        )

        # Generate pattern tokens
        from src.inference.generate import generate_pattern

        try:
            # Calculate max tokens needed (rough estimate)
            tokens_per_bar = 64  # Approximate for 16th note resolution
            max_length = min(bars * tokens_per_bar, 512)

            logger.debug(
                f"Generating with: style_id={style_id}, max_length={max_length}, "
                f"device={device}"
            )

            tokens = generate_pattern(
                model=model,
                tokenizer=None,  # Tokenizer integration pending
                num_bars=bars,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=max_length,
                device=device,
                style_id=style_id,
            )

            logger.info(f"Generated {len(tokens)} tokens successfully")

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM during generation, falling back to CPU")
            clear_gpu_cache()

            # Move model to CPU and retry
            model.to("cpu")
            device = "cpu"

            tokens = generate_pattern(
                model=model,
                tokenizer=None,
                num_bars=bars,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_length=max_length,
                device="cpu",
                style_id=style_id,
            )

            logger.info(f"Generated {len(tokens)} tokens on CPU (GPU fallback)")

        # Update task state: Converting to MIDI
        self.update_state(
            state="PROGRESS", meta={"progress": 65, "status": "Converting to MIDI notes..."}
        )

        # TODO: When tokenizer is implemented, use:
        # from src.midi.export import detokenize_to_notes
        # notes = detokenize_to_notes(tokens, tokenizer)

        # TEMPORARY: Create mock pattern for testing
        # This will be replaced with actual token→note conversion
        notes = []
        ticks_per_beat = 480
        beats_per_bar = time_signature[0]

        for bar in range(bars):
            base_tick = bar * beats_per_bar * ticks_per_beat

            # Kick on beats 1 and 3
            notes.append({"pitch": 36, "velocity": 100, "time": base_tick})
            if beats_per_bar >= 3:
                notes.append({"pitch": 36, "velocity": 95, "time": base_tick + 2 * ticks_per_beat})

            # Snare on beats 2 and 4
            if beats_per_bar >= 2:
                notes.append({"pitch": 38, "velocity": 90, "time": base_tick + ticks_per_beat})
            if beats_per_bar >= 4:
                notes.append({"pitch": 38, "velocity": 92, "time": base_tick + 3 * ticks_per_beat})

            # Hi-hats on 8th notes
            eighths_per_bar = beats_per_bar * 2
            for eighth in range(eighths_per_bar):
                velocity = 70 if eighth % 2 == 0 else 60
                notes.append(
                    {
                        "pitch": 42,
                        "velocity": velocity,
                        "time": base_tick + eighth * (ticks_per_beat // 2),
                    }
                )

        logger.debug(f"Created {len(notes)} MIDI notes")

        # Apply humanization/style transfer if enabled
        if humanize:
            self.update_state(
                state="PROGRESS", meta={"progress": 80, "status": "Applying style transfer..."}
            )

            if use_dynamic_style and style_profile:
                # Week 2: Dynamic style transfer with researched parameters
                from src.midi.style_transfer import apply_producer_style, get_style_description

                notes = apply_producer_style(
                    notes=notes,
                    style_profile=style_profile,
                    tempo=tempo,
                    ticks_per_beat=ticks_per_beat,
                )

                style_desc = get_style_description(style_profile)
                logger.info(f"Applied dynamic style transfer: {style_desc}")
            else:
                # Legacy: Fixed style humanization
                from src.midi.humanize import apply_style_humanization

                notes = apply_style_humanization(
                    notes=notes, style=normalized_style, tempo=tempo, ticks_per_beat=ticks_per_beat
                )

                logger.info(f"Applied {normalized_style}-style humanization (legacy mode)")

        # Update task state: Exporting
        self.update_state(
            state="PROGRESS", meta={"progress": 90, "status": "Exporting MIDI file..."}
        )

        # Export to MIDI file
        from src.midi.export import export_pattern

        # Create unique filename with task ID
        task_id = self.request.id
        safe_producer = actual_producer_name.lower().replace(" ", "_").replace(".", "")
        filename = f"{task_id}_{safe_producer}_{bars}bars_{tempo}bpm.mid"
        output_path = Path("output/patterns") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Exporting to {output_path}")

        midi_path = export_pattern(
            notes=notes,
            output_path=output_path,
            tempo=tempo,
            time_signature=time_signature,
            humanize=False,  # Already humanized/styled above
            style_name=actual_producer_name,
            ticks_per_beat=ticks_per_beat,
        )

        # Calculate final metrics
        duration = time.time() - start_time

        # Get GPU memory usage if on CUDA
        gpu_memory_mb = None
        if device == "cuda":
            gpu_info = get_gpu_memory_info()
            gpu_memory_mb = gpu_info.get("allocated_gb", 0) * 1024

        # Clean up GPU cache
        if device == "cuda":
            clear_gpu_cache()

        # Generate style description for result
        style_description = "Legacy fixed style"
        if use_dynamic_style and style_profile:
            from src.midi.style_transfer import get_style_description

            style_description = get_style_description(style_profile)

        logger.info(
            f"✓ Pattern generated in {duration:.2f}s: {midi_path} "
            f"({len(tokens)} tokens, {len(notes)} notes, device={device})"
        )

        # Return success result
        return {
            "midi_file": str(midi_path.resolve()),
            "duration_seconds": round(duration, 3),
            "tokens_generated": len(tokens),
            "notes_count": len(notes),
            "producer_name": actual_producer_name,
            "style_description": style_description,
            "bars": bars,
            "tempo": tempo,
            "time_signature": time_signature,
            "device": device,
            "gpu_memory_mb": gpu_memory_mb,
            "humanized": humanize,
            "style_transfer_mode": "dynamic" if use_dynamic_style else "legacy",
            "model_type": metadata.get("model_type", "unknown"),
        }

    except Exception as exc:
        duration = time.time() - start_time
        error_msg = f"Pattern generation failed after {duration:.2f}s: {str(exc)}"
        logger.error(error_msg, exc_info=True)

        # Clean up GPU if we allocated it
        if device == "cuda":
            from src.inference.model_loader import clear_gpu_cache

            clear_gpu_cache()

        # Update task state to failure
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "progress": 0,
                "status": "Failed",
                "duration_seconds": round(duration, 3),
            },
        )

        # Re-raise for Celery retry mechanism
        raise


@celery_app.task(
    bind=True,
    name="src.tasks.tasks.tokenize_midi",
    max_retries=3,
    default_retry_delay=30,
)
def tokenize_midi(
    self: Task,
    midi_path: str,
    output_dir: str = "data/tokenized",
    style_name: str = "Unknown",
) -> dict:
    """
    Tokenize MIDI file for training data preparation.

    Loads MIDI file, tokenizes using MidiTok REMI tokenizer, and saves
    tokens to disk for use in training pipeline.

    Args:
        midi_path: Path to MIDI file to tokenize
        output_dir: Directory to save tokenized data (default: "data/tokenized")
        style_name: Producer style label for this MIDI (default: "Unknown")

    Returns:
        Dict with:
            - token_file: Path to saved token file
            - num_tokens: Number of tokens generated
            - midi_file: Original MIDI path
            - style: Producer style label
            - duration_seconds: Processing time
            - metadata: Additional MIDI metadata (tempo, time signature, etc.)

    Raises:
        FileNotFoundError: If MIDI file doesn't exist
        ValueError: If MIDI file is invalid or empty
    """
    start_time = time.time()

    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading MIDI file..."})

        logger.info(f"Task {self.request.id}: Tokenizing MIDI: {midi_path}")

        # Validate MIDI file exists
        midi_path = Path(midi_path)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        # Load MIDI file
        import mido

        from src.midi.io import read_midi_file

        try:
            mid = read_midi_file(midi_path)
            logger.debug(f"Loaded MIDI: {len(mid.tracks)} tracks, {mid.ticks_per_beat} ticks/beat")
        except Exception as e:
            raise ValueError(f"Failed to load MIDI file: {e}") from e

        # Extract metadata from MIDI
        self.update_state(
            state="PROGRESS", meta={"progress": 25, "status": "Extracting metadata..."}
        )

        metadata = {
            "ticks_per_beat": mid.ticks_per_beat,
            "num_tracks": len(mid.tracks),
            "type": mid.type,
        }

        # Extract tempo and time signature from first track
        tempo_bpm = 120  # Default
        time_signature = (4, 4)  # Default

        for msg in mid.tracks[0]:
            if msg.type == "set_tempo":
                tempo_bpm = mido.tempo2bpm(msg.tempo)
                metadata["tempo"] = tempo_bpm
            elif msg.type == "time_signature":
                time_signature = (msg.numerator, msg.denominator)
                metadata["time_signature"] = time_signature

        logger.debug(f"MIDI metadata: {tempo_bpm} BPM, {time_signature[0]}/{time_signature[1]}")

        # Count total notes
        total_notes = 0
        for track in mid.tracks:
            total_notes += sum(1 for msg in track if msg.type == "note_on" and msg.velocity > 0)

        metadata["total_notes"] = total_notes

        if total_notes == 0:
            raise ValueError("MIDI file contains no notes")

        logger.debug(f"Found {total_notes} notes in MIDI")

        # Tokenize MIDI
        self.update_state(state="PROGRESS", meta={"progress": 50, "status": "Tokenizing MIDI..."})

        # TODO: Integrate MidiTok REMI tokenizer when available
        # For now, create a placeholder token representation
        logger.warning(
            "MidiTok integration pending - creating placeholder tokenization. "
            "This will be replaced with actual REMI tokenization in Phase 6."
        )

        # Placeholder: Create simple token representation
        # In real implementation, this will use:
        # from miditok import REMI
        # tokenizer = REMI()
        # tokens = tokenizer(mid)

        # For now, create mock tokens based on note count
        tokens = list(range(1, min(total_notes * 3, 512)))  # Placeholder
        num_tokens = len(tokens)

        logger.debug(f"Generated {num_tokens} tokens (placeholder)")

        # Save tokens to disk
        self.update_state(state="PROGRESS", meta={"progress": 75, "status": "Saving tokens..."})

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename from style and MIDI filename
        safe_style = style_name.lower().replace(" ", "_")
        midi_stem = midi_path.stem
        token_filename = f"{safe_style}_{midi_stem}_tokens.pt"
        token_path = output_dir / token_filename

        # Save tokens as PyTorch tensor
        import torch

        torch.save(torch.tensor(tokens, dtype=torch.long), token_path)

        logger.debug(f"Saved tokens to {token_path}")

        # Also save metadata JSON
        metadata_path = token_path.with_suffix(".json")
        import json

        metadata_full = {
            **metadata,
            "style": style_name,
            "midi_file": str(midi_path),
            "num_tokens": num_tokens,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata_full, f, indent=2)

        logger.debug(f"Saved metadata to {metadata_path}")

        # Calculate duration
        duration = time.time() - start_time

        logger.info(
            f"✓ Tokenized {midi_path.name} in {duration:.2f}s: "
            f"{num_tokens} tokens, {total_notes} notes"
        )

        return {
            "token_file": str(token_path.resolve()),
            "metadata_file": str(metadata_path.resolve()),
            "num_tokens": num_tokens,
            "midi_file": str(midi_path),
            "style": style_name,
            "duration_seconds": round(duration, 3),
            "metadata": metadata_full,
        }

    except FileNotFoundError as e:
        logger.error(f"MIDI file not found: {e}")
        self.update_state(
            state="FAILURE",
            meta={
                "error": str(e),
                "error_type": "FileNotFoundError",
                "status": "Failed - file not found",
            },
        )
        raise

    except ValueError as e:
        logger.error(f"Invalid MIDI file: {e}")
        self.update_state(
            state="FAILURE",
            meta={"error": str(e), "error_type": "ValueError", "status": "Failed - invalid MIDI"},
        )
        raise

    except Exception as exc:
        duration = time.time() - start_time
        logger.error(f"Tokenization failed after {duration:.2f}s: {exc}", exc_info=True)

        self.update_state(
            state="FAILURE",
            meta={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "status": "Failed",
                "duration_seconds": round(duration, 3),
            },
        )

        raise


@celery_app.task(
    bind=True,
    name="src.tasks.tasks.tokenize_midi_batch",
)
def tokenize_midi_batch(
    self: Task,
    midi_files: list,
    output_dir: str = "data/tokenized",
    style_name: str = "Unknown",
) -> dict:
    """
    Batch tokenize multiple MIDI files.

    Useful for preprocessing training datasets. Processes files sequentially
    with progress updates.

    Args:
        midi_files: List of MIDI file paths
        output_dir: Directory to save tokenized data
        style_name: Producer style label for all files

    Returns:
        Dict with:
            - total_files: Total number of files
            - successful: Number of successfully tokenized files
            - failed: Number of failed files
            - token_files: List of generated token file paths
            - errors: List of errors encountered
            - duration_seconds: Total processing time
    """
    start_time = time.time()
    total_files = len(midi_files)
    successful = 0
    failed = 0
    token_files = []
    errors = []

    logger.info(f"Task {self.request.id}: Batch tokenizing {total_files} MIDI files")

    for idx, midi_file in enumerate(midi_files):
        try:
            # Update progress
            progress = int((idx / total_files) * 100)
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "status": f"Processing file {idx+1}/{total_files}",
                    "current_file": midi_file,
                    "successful": successful,
                    "failed": failed,
                },
            )

            # Tokenize this file
            result = tokenize_midi(midi_file, output_dir, style_name)
            token_files.append(result["token_file"])
            successful += 1

            logger.debug(f"✓ Tokenized {idx+1}/{total_files}: {midi_file}")

        except Exception as e:
            failed += 1
            error_msg = f"{midi_file}: {str(e)}"
            errors.append(error_msg)
            logger.warning(f"✗ Failed {idx+1}/{total_files}: {error_msg}")

    duration = time.time() - start_time

    logger.info(
        f"✓ Batch tokenization complete: {successful}/{total_files} successful, "
        f"{failed} failed, {duration:.2f}s"
    )

    return {
        "total_files": total_files,
        "successful": successful,
        "failed": failed,
        "token_files": token_files,
        "errors": errors,
        "duration_seconds": round(duration, 3),
    }


@celery_app.task(
    bind=True,
    name="src.tasks.tasks.train_model",
)
def train_model(self: Task, config_path: str) -> dict:
    """
    Train model (Phase 6 implementation).

    Placeholder for Phase 6 training pipeline. Will implement:
    - Load tokenized dataset
    - Initialize or resume model
    - Training loop with validation
    - Checkpoint saving
    - Wandb logging

    Args:
        config_path: Path to training configuration YAML

    Returns:
        Dictionary with training results

    Note:
        This is a placeholder. Full implementation in Phase 6.
    """
    logger.info(f"Training task received with config: {config_path}")
    logger.warning(
        "Model training not yet implemented (Phase 6). "
        "This task will handle full training pipeline including: "
        "dataset loading, model initialization, training loop, "
        "validation, checkpointing, and logging."
    )

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 0,
            "status": "Training not implemented yet (Phase 6)",
            "config": config_path,
        },
    )

    return {
        "status": "not_implemented",
        "message": "Training task will be implemented in Phase 6",
        "config": config_path,
    }
