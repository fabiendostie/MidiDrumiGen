"""Tokenize MIDI dataset using MidiTok for training."""

import argparse
from pathlib import Path
import logging
import json
import torch
from typing import Dict, Any, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miditok import REMI, TokenizerConfig
from miditok.constants import CHORD_MAPS
import mido

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_tokenizer(
    vocab_size: int = 1000,
    beat_resolution: int = 8,
    num_velocities: int = 32,
) -> REMI:
    """
    Create REMI tokenizer for drum patterns.

    REMI (Revamped MIDI) is well-suited for drum patterns as it captures:
    - Note on/off events
    - Timing (position, duration)
    - Velocity levels

    Args:
        vocab_size: Target vocabulary size (actual may vary slightly)
        beat_resolution: Ticks per beat (8 = 16th notes, 16 = 32nd notes)
        num_velocities: Number of velocity bins

    Returns:
        Configured REMI tokenizer

    Example:
        >>> tokenizer = create_tokenizer()
        >>> print(f"Vocab size: {len(tokenizer)}")
    """
    # Configure tokenizer
    # REMI params: https://miditok.readthedocs.io/en/latest/tokenizations.html#remi
    config = TokenizerConfig(
        use_chords=False,  # Drums don't use chords
        use_programs=False,  # Single program (drums)
        use_rests=True,  # Important for capturing silence
        use_tempos=True,  # Capture tempo changes
        use_time_signatures=True,  # Capture time signature
        beat_res={(0, 4): beat_resolution},  # 16th note resolution
        num_velocities=num_velocities,  # Velocity bins
        special_tokens=["PAD", "BOS", "EOS"],  # Padding, begin, end tokens
        delete_equal_successive_tempo_changes=True,
        delete_equal_successive_time_sig_changes=True,
    )

    tokenizer = REMI(config)
    logger.info(f"Created REMI tokenizer with config: {config}")

    return tokenizer


def tokenize_midi_file(
    midi_path: Path,
    tokenizer: REMI,
    style_name: str,
    output_dir: Path,
) -> bool:
    """
    Tokenize a single MIDI file and save tokens.

    Args:
        midi_path: Path to MIDI file
        tokenizer: MidiTok tokenizer
        style_name: Producer style name (for labeling)
        output_dir: Directory to save tokenized output

    Returns:
        True if successful, False otherwise

    Saves:
        - {output_dir}/{style}_{number}.pt: Token tensor
        - {output_dir}/{style}_{number}.json: Metadata
    """
    try:
        # Load MIDI file
        midi = mido.MidiFile(midi_path)

        # Validate MIDI has content
        has_notes = False
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    has_notes = True
                    break
            if has_notes:
                break

        if not has_notes:
            logger.warning(f"Skipping {midi_path.name}: No note events found")
            return False

        # Tokenize
        tokens = tokenizer(str(midi_path))

        # Extract IDs from MidiTok TokSequence
        if isinstance(tokens, list):
            if len(tokens) == 0:
                logger.warning(f"Skipping {midi_path.name}: Empty token sequence")
                return False
            # Get first track (tokens[0]) and extract IDs
            tokens_ids = tokens[0].ids if hasattr(tokens[0], 'ids') else tokens[0]
        else:
            # Single TokSequence object
            tokens_ids = tokens.ids if hasattr(tokens, 'ids') else tokens

        # Convert to tensor
        if isinstance(tokens_ids, list):
            tokens_tensor = torch.tensor(tokens_ids, dtype=torch.long)
        else:
            tokens_tensor = tokens_ids

        # Validate tokens
        if len(tokens_tensor) == 0:
            logger.warning(f"Skipping {midi_path.name}: Empty token sequence after conversion")
            return False

        if len(tokens_tensor) < 10:  # Too short to be useful
            logger.warning(f"Skipping {midi_path.name}: Sequence too short ({len(tokens_tensor)} tokens)")
            return False

        # Extract metadata
        tempo = 120  # Default tempo
        time_signature = [4, 4]  # Default time signature

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
                    time_signature = [msg.numerator, msg.denominator]

        metadata = {
            'style': style_name,
            'tempo': tempo,
            'time_signature': time_signature,
            'num_tokens': len(tokens_tensor),
            'source_midi': str(midi_path.name),
        }

        # Generate output filename
        # Format: {style_name}_{index}.pt
        existing_files = list(output_dir.glob(f"{style_name}_*.pt"))
        file_index = len(existing_files) + 1
        output_stem = f"{style_name}_{file_index:04d}"

        # Save tokens
        tokens_path = output_dir / f"{output_stem}.pt"
        torch.save(tokens_tensor, tokens_path)

        # Save metadata
        metadata_path = output_dir / f"{output_stem}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Tokenized {midi_path.name} -> {tokens_path.name} ({len(tokens_tensor)} tokens)")
        return True

    except Exception as e:
        logger.error(f"Error tokenizing {midi_path.name}: {e}")
        return False


def tokenize_dataset(
    input_dir: Path,
    output_dir: Path,
    style_name: str,
    tokenizer: REMI,
) -> Dict[str, int]:
    """
    Tokenize all MIDI files in a directory (searches recursively).

    Args:
        input_dir: Directory containing MIDI files
        output_dir: Directory to save tokenized files
        style_name: Producer style name for this dataset
        tokenizer: MidiTok tokenizer

    Returns:
        Dict with statistics (total files, successful, failed)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all MIDI files (search recursively with **)
    midi_files = []
    for ext in ['*.mid', '*.midi']:
        midi_files.extend(input_dir.rglob(ext))

    if len(midi_files) == 0:
        logger.warning(f"No MIDI files found in {input_dir}")
        return {'total': 0, 'successful': 0, 'failed': 0}

    logger.info(f"Found {len(midi_files)} MIDI files in {input_dir}")

    # Tokenize each file
    successful = 0
    failed = 0

    for midi_file in midi_files:
        success = tokenize_midi_file(
            midi_path=midi_file,
            tokenizer=tokenizer,
            style_name=style_name,
            output_dir=output_dir,
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Log statistics
    logger.info(f"Tokenization complete: {successful} successful, {failed} failed")

    return {
        'total': len(midi_files),
        'successful': successful,
        'failed': failed,
    }


def save_tokenizer(tokenizer: REMI, output_path: Path) -> None:
    """
    Save tokenizer configuration.

    Args:
        tokenizer: MidiTok tokenizer to save
        output_path: Path to save tokenizer
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer config as JSON
    config_dict = {
        'vocab_size': len(tokenizer),
        'beat_resolution': tokenizer.config.beat_res,
        'num_velocities': tokenizer.config.num_velocities,
        'special_tokens': tokenizer.config.special_tokens,
        'use_rests': tokenizer.config.use_rests,
        'use_tempos': tokenizer.config.use_tempos,
        'use_time_signatures': tokenizer.config.use_time_signatures,
    }
    
    config_path = output_path.with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved tokenizer config to {config_path}")


def main():
    """Main tokenization script."""
    parser = argparse.ArgumentParser(description="Tokenize MIDI dataset for training")

    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing MIDI files to tokenize'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to save tokenized files'
    )

    parser.add_argument(
        '--style',
        type=str,
        required=True,
        help='Producer style name (e.g., "j_dilla", "metro_boomin")'
    )

    parser.add_argument(
        '--vocab-size',
        type=int,
        default=1000,
        help='Target vocabulary size (default: 1000)'
    )

    parser.add_argument(
        '--beat-resolution',
        type=int,
        default=8,
        help='Beat resolution: 8=16th notes, 16=32nd notes (default: 8)'
    )

    parser.add_argument(
        '--num-velocities',
        type=int,
        default=32,
        help='Number of velocity bins (default: 32)'
    )

    parser.add_argument(
        '--save-tokenizer',
        type=Path,
        default=None,
        help='Path to save tokenizer config (optional)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Normalize style name (lowercase with underscores)
    style_name = args.style.lower().replace(' ', '_').replace('-', '_')

    # Create tokenizer
    logger.info("Creating tokenizer...")
    tokenizer = create_tokenizer(
        vocab_size=args.vocab_size,
        beat_resolution=args.beat_resolution,
        num_velocities=args.num_velocities,
    )
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Tokenize dataset
    logger.info(f"Tokenizing MIDI files from {args.input_dir}")
    logger.info(f"Style: {style_name}")

    stats = tokenize_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        style_name=style_name,
        tokenizer=tokenizer,
    )

    # Save tokenizer if requested
    if args.save_tokenizer:
        save_tokenizer(tokenizer, args.save_tokenizer)

    # Print summary
    print("\n" + "=" * 80)
    print("TOKENIZATION SUMMARY")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Style name: {style_name}")
    print(f"Total MIDI files: {stats['total']}")
    print(f"Successfully tokenized: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print("=" * 80)

    if stats['successful'] == 0:
        logger.error("No files were successfully tokenized!")
        sys.exit(1)


if __name__ == "__main__":
    main()
