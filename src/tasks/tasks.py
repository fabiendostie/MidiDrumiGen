"""Celery task definitions."""

from celery import Task
from src.tasks.worker import celery_app
import logging
from pathlib import Path
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='src.tasks.tasks.generate_pattern')
def generate_pattern_task(
    self: Task,
    producer_style: str,
    bars: int,
    tempo: int,
    time_signature: tuple,
    humanize: bool = True,
    pattern_type: str = "verse",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict:
    """
    Generate drum pattern using ML model and export to MIDI.

    This is a Celery task that:
    1. Loads the appropriate model (with caching)
    2. Generates pattern tokens using the model
    3. Converts tokens to MIDI notes (placeholder until tokenizer ready)
    4. Applies humanization based on producer style
    5. Exports to MIDI file
    6. Returns file path and metadata

    Args:
        producer_style: Producer style name (e.g., "J Dilla")
        bars: Number of bars to generate
        tempo: Tempo in BPM
        time_signature: Time signature as (numerator, denominator)
        humanize: Apply humanization
        pattern_type: Pattern type (intro, verse, etc.)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter

    Returns:
        Dict with:
            - midi_file: Path to generated MIDI file
            - duration_seconds: Generation time
            - tokens_generated: Number of tokens
            - style: Producer style
            - bars: Number of bars
            - tempo: BPM
    """
    start_time = time.time()

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting generation...'})

        logger.info(f"Generating pattern: {producer_style}, {bars} bars @ {tempo} BPM")

        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 25, 'status': 'Loading model...'})

        # Load model with error handling
        try:
            from src.inference.model_loader import load_model
            from src.models.styles import get_style_id

            style_id = get_style_id(producer_style)
            model_path = f"models/checkpoints/{producer_style.lower().replace(' ', '_')}_v1.pt"

            # Try to load model, fallback to mock if not available
            try:
                model, metadata = load_model(model_path)
                logger.info(f"Loaded model from {model_path}")
            except FileNotFoundError:
                logger.warning(f"Model not found at {model_path}, using mock model")
                from src.inference.mock import MockDrumModel
                model = MockDrumModel()
                metadata = {"vocab_size": 1000}
        except Exception as e:
            logger.error(f"Model loading failed: {e}, using mock model")
            from src.inference.mock import MockDrumModel
            model = MockDrumModel()
            metadata = {"vocab_size": 1000}

        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Generating pattern...'})

        # Generate pattern
        from src.inference.generate import generate_pattern
        from src.models.styles import get_numeric_style_id

        try:
            style_id = get_numeric_style_id(producer_style)
        except:
            style_id = 0  # Default style

        tokens = generate_pattern(
            model=model,
            tokenizer=None,  # Placeholder until Phase 5
            num_bars=bars,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device="cuda",
            style_id=style_id
        )

        logger.info(f"Generated {len(tokens)} tokens")

        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Converting to MIDI...'})

        # Convert to MIDI notes (placeholder until tokenizer is implemented)
        # For now, create a simple test pattern
        notes = []
        ticks_per_beat = 480
        beats_per_bar = time_signature[0]
        total_ticks = ticks_per_beat * beats_per_bar * bars

        # Create a basic drum pattern
        for bar in range(bars):
            base_tick = bar * beats_per_bar * ticks_per_beat

            # Kick on beats 1 and 3
            notes.append({'pitch': 36, 'velocity': 100, 'time': base_tick})
            notes.append({'pitch': 36, 'velocity': 95, 'time': base_tick + 2 * ticks_per_beat})

            # Snare on beats 2 and 4
            notes.append({'pitch': 38, 'velocity': 90, 'time': base_tick + ticks_per_beat})
            notes.append({'pitch': 38, 'velocity': 92, 'time': base_tick + 3 * ticks_per_beat})

            # Hi-hats on 8th notes
            for eighth in range(8):
                velocity = 70 if eighth % 2 == 0 else 60  # Accented on downbeats
                notes.append({
                    'pitch': 42,
                    'velocity': velocity,
                    'time': base_tick + eighth * (ticks_per_beat // 2)
                })

        # Apply humanization
        if humanize:
            self.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Applying humanization...'})

            from src.midi.humanize import apply_style_humanization
            notes = apply_style_humanization(notes, producer_style, tempo)
            logger.info(f"Applied {producer_style} humanization")

        # Export to MIDI
        self.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Exporting MIDI file...'})

        from src.midi.export import export_pattern

        # Create unique filename
        task_id = self.request.id
        filename = f"{task_id}_{producer_style.lower().replace(' ', '_')}_{bars}bars.mid"
        output_path = Path("output/patterns") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = export_pattern(
            notes=notes,
            output_path=output_path,
            tempo=tempo,
            time_signature=time_signature
        )

        if not success:
            raise Exception("MIDI export failed")

        duration = time.time() - start_time

        logger.info(f"Pattern generated successfully in {duration:.2f}s: {output_path}")

        # Return result
        return {
            'midi_file': str(output_path),
            'duration_seconds': duration,
            'tokens_generated': len(tokens),
            'style': producer_style,
            'bars': bars,
            'tempo': tempo,
        }

    except Exception as exc:
        logger.error(f"Generation failed: {exc}", exc_info=True)
        # Update state to failure
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(exc),
                'progress': 0,
                'status': 'Failed'
            }
        )
        raise


@celery_app.task
def tokenize_midi(midi_path: str) -> dict:
    """
    Tokenize MIDI file.

    Args:
        midi_path: Path to MIDI file

    Returns:
        Dictionary with tokenized data
    """
    # TODO: Implement tokenization in Phase 5
    logger.info(f"Tokenization task received for: {midi_path}")
    return {"status": "complete", "tokens": []}


@celery_app.task
def train_model(config_path: str) -> dict:
    """
    Train model.

    Args:
        config_path: Path to training configuration

    Returns:
        Dictionary with training results
    """
    # TODO: Implement training in Phase 6
    logger.info(f"Training task received with config: {config_path}")
    return {"status": "complete"}
