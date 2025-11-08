"""MIDI pattern export pipeline using mido."""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from .io import create_midi_file, save_midi_file
from .validate import validate_drum_pattern
from .humanize import apply_style_humanization
from .constants import DEFAULT_TICKS_PER_BEAT

logger = logging.getLogger(__name__)


def export_pattern(
    notes: List[Dict],
    output_path: Path,
    tempo: int = 120,
    time_signature: Tuple[int, int] = (4, 4),
    humanize: bool = True,
    style_name: str = "Unknown",
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
) -> Path:
    """
    Export drum pattern to MIDI file with optional humanization.

    Complete pipeline:
    1. Validate pattern for musical correctness
    2. Create MIDI file with metadata
    3. Apply humanization if enabled
    4. Add notes to MIDI track
    5. Save MIDI file

    Args:
        notes: List of note dicts with 'pitch', 'velocity', 'time' keys
               - pitch: MIDI note number (35-81 for drums)
               - velocity: Note velocity (1-127)
               - time: Absolute time in ticks
        output_path: Path to save MIDI file
        tempo: BPM (default: 120)
        time_signature: (numerator, denominator) tuple (default: 4/4)
        humanize: Apply humanization algorithms (default: True)
        style_name: Producer style name for metadata (default: "Unknown")
        ticks_per_beat: MIDI resolution (default: 480)

    Returns:
        Path to saved MIDI file

    Raises:
        ValueError: If pattern validation fails

    Example:
        >>> notes = [
        ...     {'pitch': 36, 'velocity': 100, 'time': 0},     # Kick on beat 1
        ...     {'pitch': 42, 'velocity': 80, 'time': 240},    # Hi-hat on 1.5
        ...     {'pitch': 38, 'velocity': 90, 'time': 480},    # Snare on beat 2
        ... ]
        >>> export_pattern(notes, Path("output.mid"), tempo=120)
    """
    logger.info(f"Exporting pattern: {len(notes)} notes, tempo={tempo}, humanize={humanize}")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Validate pattern
    is_valid, errors = validate_drum_pattern(notes)
    if not is_valid:
        error_msg = f"Invalid drum pattern: {'; '.join(errors)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("Pattern validation passed")

    # 2. Apply humanization if enabled
    processed_notes = notes.copy()
    if humanize:
        logger.debug(f"Applying humanization with style: {style_name}")
        processed_notes = apply_style_humanization(
            processed_notes,
            style=style_name,
            tempo=tempo,
            ticks_per_beat=ticks_per_beat
        )

    # 3. Create MIDI file with metadata
    mid, track = create_midi_file(
        tempo=tempo,
        time_signature=time_signature,
        ticks_per_beat=ticks_per_beat,
        track_name=f"Drums - {style_name}"
    )

    # 4. Create MIDI events with absolute times
    # We need to create all events (note on + note off) first,
    # then sort by time, then convert to delta times

    midi_events = []
    NOTE_DURATION = 10  # Short duration for drum hits

    for note in processed_notes:
        note_time = note['time']
        pitch = note['pitch']
        velocity = note['velocity']

        # Create note on event with absolute time
        midi_events.append({
            'type': 'note_on',
            'note': pitch,
            'velocity': velocity,
            'time': note_time,
            'channel': 9
        })

        # Create note off event with absolute time
        midi_events.append({
            'type': 'note_off',
            'note': pitch,
            'velocity': 0,
            'time': note_time + NOTE_DURATION,
            'channel': 9
        })

    # Sort all events by time
    midi_events.sort(key=lambda e: e['time'])

    # Convert absolute times to delta times and add to track
    last_time = 0
    for event in midi_events:
        # Calculate delta time
        delta_time = event['time'] - last_time

        # Ensure delta time is non-negative
        if delta_time < 0:
            logger.warning(f"Negative delta time detected: {delta_time}, clamping to 0")
            delta_time = 0

        # Add event to track
        track.append(Message(
            event['type'],
            note=event['note'],
            velocity=event['velocity'],
            time=delta_time,
            channel=event['channel']
        ))

        # Update last time
        last_time = event['time']

    # 5. Add end of track marker
    track.append(MetaMessage('end_of_track', time=0))

    # 6. Save MIDI file
    save_midi_file(mid, output_path)

    logger.info(f"Successfully exported MIDI to {output_path}")
    return output_path


def detokenize_to_notes(
    tokens: List[int],
    tokenizer,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT
) -> List[Dict]:
    """
    Convert tokenized pattern to note list.

    This is a placeholder for tokenizer integration.
    Will be completed when MidiTok tokenizer is set up.

    Args:
        tokens: List of token IDs from model
        tokenizer: MidiTok tokenizer instance
        ticks_per_beat: MIDI resolution

    Returns:
        List of note dicts with 'pitch', 'velocity', 'time' keys

    TODO: Implement when tokenizer is ready
    """
    # Placeholder implementation
    # In real implementation, this will use tokenizer.decode()
    logger.warning("detokenize_to_notes is a placeholder - tokenizer not implemented yet")

    # Example return format:
    # return [
    #     {'pitch': 36, 'velocity': 100, 'time': 0},
    #     {'pitch': 42, 'velocity': 80, 'time': 240},
    # ]

    raise NotImplementedError("Tokenizer integration pending - use export_pattern with notes directly")


def export_from_tokens(
    tokens: List[int],
    tokenizer,
    output_path: Path,
    **kwargs
) -> Path:
    """
    High-level export function that handles tokenization.

    Convenience wrapper around detokenize_to_notes + export_pattern.

    Args:
        tokens: Token IDs from model generation
        tokenizer: MidiTok tokenizer instance
        output_path: Where to save MIDI file
        **kwargs: Additional arguments passed to export_pattern

    Returns:
        Path to saved MIDI file

    TODO: Enable when tokenizer is implemented
    """
    # Detokenize tokens to notes
    notes = detokenize_to_notes(tokens, tokenizer)

    # Export notes to MIDI
    return export_pattern(notes, output_path, **kwargs)
