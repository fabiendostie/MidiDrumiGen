"""Humanization algorithms for drum patterns."""

import random
from typing import List, Dict, Optional
import logging

from .constants import DEFAULT_TICKS_PER_BEAT

logger = logging.getLogger(__name__)


def apply_swing(
    note_time: int,
    swing_percentage: float,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT
) -> int:
    """
    Apply swing timing to offbeat notes.

    Swing delays offbeat notes (8th note upbeats) to create a laid-back feel.

    Args:
        note_time: Original note time in ticks
        swing_percentage: 50-75 range
                         - 50 = straight (no swing)
                         - 58 = slight swing (humanized)
                         - 66 = triplet swing (jazz/hip-hop)
                         - 75 = extreme swing
        ticks_per_beat: Ticks per quarter note (default: 480)

    Returns:
        Adjusted note time in ticks

    Example:
        >>> # Apply J Dilla style swing (62%)
        >>> swung_time = apply_swing(240, 62, 480)  # Offbeat 8th note
    """
    eighth_note = ticks_per_beat / 2

    # Check if note is on an offbeat (odd multiple of eighth notes)
    position_in_eighths = note_time / eighth_note
    is_offbeat = position_in_eighths % 2 == 1

    if is_offbeat:
        # Calculate swing offset
        # At 50%, no offset; at 66%, offset by 1/6 of eighth note (triplet)
        swing_offset = ((swing_percentage - 50) / 50) * (eighth_note / 4)
        return int(note_time + swing_offset)

    return note_time


def apply_micro_timing(
    note_time: int,
    max_offset_ms: float,
    tempo: int,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT
) -> int:
    """
    Add subtle random timing variations.

    Simulates human imprecision by randomly shifting note times.

    Args:
        note_time: Original time in ticks
        max_offset_ms: Maximum offset in milliseconds (±15ms typical)
        tempo: BPM
        ticks_per_beat: MIDI resolution (default: 480)

    Returns:
        Humanized time in ticks

    Example:
        >>> # Add up to ±15ms variation
        >>> humanized = apply_micro_timing(480, 15.0, 120, 480)
    """
    # Convert milliseconds to ticks
    # Formula: ms_per_tick = (60000 ms/min) / (BPM * ticks_per_beat)
    ms_per_tick = (60000 / tempo) / ticks_per_beat
    max_offset_ticks = int(max_offset_ms / ms_per_tick)

    # Add random offset (uniform distribution)
    offset = random.randint(-max_offset_ticks, max_offset_ticks)

    # Ensure time doesn't go negative
    return max(0, note_time + offset)


def apply_velocity_variation(
    base_velocity: int,
    variation: float = 0.1,
    min_velocity: int = 1,
    max_velocity: int = 127
) -> int:
    """
    Add random velocity variation for dynamics.

    Args:
        base_velocity: Original velocity (1-127)
        variation: Percentage variation (0.1 = ±10%, 0.15 = ±15%)
        min_velocity: Minimum allowed velocity (default: 1)
        max_velocity: Maximum allowed velocity (default: 127)

    Returns:
        Humanized velocity (clamped to valid range)

    Example:
        >>> # Add ±10% velocity variation
        >>> varied = apply_velocity_variation(100, variation=0.1)
        >>> # Result will be in range [90, 110]
    """
    # Calculate random offset
    offset = int(base_velocity * variation * random.uniform(-1, 1))
    new_velocity = base_velocity + offset

    # Clamp to valid MIDI range
    return max(min_velocity, min(max_velocity, new_velocity))


def apply_accent_pattern(
    notes: List[Dict],
    accent_positions: List[int],
    accent_boost: int = 20,
    accent_reduction: int = 5
) -> List[Dict]:
    """
    Apply accent pattern to emphasize specific beats.

    Boosts velocity of accented notes and reduces adjacent notes.

    Args:
        notes: List of note dicts with 'velocity' key
        accent_positions: Indices of notes to accent (e.g., [0, 2] for beats 1 and 3)
        accent_boost: Velocity increase for accented notes (default: 20)
        accent_reduction: Velocity decrease for adjacent notes (default: 5)

    Returns:
        Modified notes list with accent pattern applied

    Example:
        >>> notes = [
        ...     {'pitch': 36, 'velocity': 80, 'time': 0},
        ...     {'pitch': 42, 'velocity': 70, 'time': 240},
        ...     {'pitch': 38, 'velocity': 80, 'time': 480},
        ... ]
        >>> accented = apply_accent_pattern(notes, accent_positions=[0, 2])
        >>> # notes[0] and notes[2] will have +20 velocity
    """
    result = [note.copy() for note in notes]

    for i in accent_positions:
        if i < len(result):
            # Boost accented note
            result[i]['velocity'] = min(127, result[i]['velocity'] + accent_boost)

            # Reduce adjacent notes slightly for contrast
            if i > 0:
                result[i-1]['velocity'] = max(1, result[i-1]['velocity'] - accent_reduction)
            if i < len(result) - 1:
                result[i+1]['velocity'] = max(1, result[i+1]['velocity'] - accent_reduction)

    return result


def add_ghost_notes(
    notes: List[Dict],
    probability: float = 0.3,
    ghost_velocity: int = 30,
    ghost_note: int = 38,  # Snare
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT
) -> List[Dict]:
    """
    Add subtle ghost notes between main hits.

    Ghost notes are quiet, short notes that add groove (common in funk/hip-hop).

    Args:
        notes: List of note dicts (will not be modified)
        probability: Chance of adding ghost note between consecutive notes (0-1)
        ghost_velocity: Velocity of ghost notes (20-40 typical)
        ghost_note: MIDI note number for ghosts (default: 38 = snare)
        ticks_per_beat: MIDI resolution (default: 480)

    Returns:
        New notes list with ghost notes added

    Example:
        >>> notes = [
        ...     {'pitch': 36, 'velocity': 100, 'time': 0},
        ...     {'pitch': 38, 'velocity': 90, 'time': 480},
        ... ]
        >>> with_ghosts = add_ghost_notes(notes, probability=0.5)
    """
    result = notes.copy()
    new_ghosts = []

    # Sort by time to ensure correct ordering
    sorted_notes = sorted(notes, key=lambda n: n['time'])

    for i in range(len(sorted_notes) - 1):
        if random.random() < probability:
            curr_time = sorted_notes[i]['time']
            next_time = sorted_notes[i + 1]['time']

            # Only add ghost if there's sufficient space (at least 16th note)
            min_spacing = ticks_per_beat / 4
            if next_time - curr_time >= min_spacing * 2:
                # Add ghost note between main notes
                ghost_time = (curr_time + next_time) // 2

                new_ghosts.append({
                    'pitch': ghost_note,
                    'velocity': ghost_velocity,
                    'time': ghost_time
                })

    # Combine original notes with ghost notes
    result.extend(new_ghosts)

    logger.debug(f"Added {len(new_ghosts)} ghost notes")

    return result


# Producer-specific style parameters
PRODUCER_STYLES = {
    'j_dilla': {
        'swing': 62.0,  # Signature Dilla swing
        'micro_timing_ms': 20.0,  # More variation
        'ghost_note_prob': 0.4,
        'velocity_variation': 0.15,
        'preferred_tempo_range': (85, 95),
    },
    'J Dilla': {  # Alias with proper casing
        'swing': 62.0,
        'micro_timing_ms': 20.0,
        'ghost_note_prob': 0.4,
        'velocity_variation': 0.15,
        'preferred_tempo_range': (85, 95),
    },
    'metro_boomin': {
        'swing': 52.0,  # Straighter timing
        'micro_timing_ms': 5.0,  # Tighter quantization
        'ghost_note_prob': 0.1,
        'velocity_variation': 0.08,
        'preferred_tempo_range': (130, 150),
    },
    'Metro Boomin': {  # Alias
        'swing': 52.0,
        'micro_timing_ms': 5.0,
        'ghost_note_prob': 0.1,
        'velocity_variation': 0.08,
        'preferred_tempo_range': (130, 150),
    },
    'questlove': {
        'swing': 58.0,
        'micro_timing_ms': 12.0,
        'ghost_note_prob': 0.5,  # Lots of ghost notes
        'velocity_variation': 0.20,  # Very dynamic
        'preferred_tempo_range': (90, 110),
    },
    'Questlove': {  # Alias
        'swing': 58.0,
        'micro_timing_ms': 12.0,
        'ghost_note_prob': 0.5,
        'velocity_variation': 0.20,
        'preferred_tempo_range': (90, 110),
    },
    'unknown': {
        'swing': 54.0,  # Slight humanization
        'micro_timing_ms': 10.0,
        'ghost_note_prob': 0.2,
        'velocity_variation': 0.10,
        'preferred_tempo_range': (90, 130),
    },
    'Unknown': {  # Alias
        'swing': 54.0,
        'micro_timing_ms': 10.0,
        'ghost_note_prob': 0.2,
        'velocity_variation': 0.10,
        'preferred_tempo_range': (90, 130),
    },
}


def apply_style_humanization(
    notes: List[Dict],
    style: str,
    tempo: int,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT
) -> List[Dict]:
    """
    Apply producer-specific humanization to pattern.

    Combines swing, micro-timing, velocity variation, and ghost notes
    according to style parameters.

    Args:
        notes: List of note dicts with 'pitch', 'velocity', 'time' keys
        style: Producer style name (e.g., "J Dilla", "Metro Boomin")
        tempo: BPM
        ticks_per_beat: MIDI resolution (default: 480)

    Returns:
        Humanized notes list

    Example:
        >>> notes = [{'pitch': 36, 'velocity': 100, 'time': 0}]
        >>> humanized = apply_style_humanization(notes, "J Dilla", 95)
    """
    # Get style parameters (default to 'unknown' if style not found)
    style_key = style.lower().replace(' ', '_')
    if style_key not in PRODUCER_STYLES and style not in PRODUCER_STYLES:
        logger.warning(f"Unknown style '{style}', using default humanization")
        style_params = PRODUCER_STYLES['unknown']
    else:
        style_params = PRODUCER_STYLES.get(style, PRODUCER_STYLES.get(style_key, PRODUCER_STYLES['unknown']))

    logger.debug(f"Applying {style} humanization: {style_params}")

    # Apply humanization to each note
    humanized = []
    for note in notes:
        humanized_note = note.copy()

        # Apply swing timing
        if 'swing' in style_params:
            humanized_note['time'] = apply_swing(
                humanized_note['time'],
                style_params['swing'],
                ticks_per_beat
            )

        # Apply micro-timing variation
        if 'micro_timing_ms' in style_params:
            humanized_note['time'] = apply_micro_timing(
                humanized_note['time'],
                style_params['micro_timing_ms'],
                tempo,
                ticks_per_beat
            )

        # Apply velocity variation
        if 'velocity_variation' in style_params:
            humanized_note['velocity'] = apply_velocity_variation(
                humanized_note['velocity'],
                style_params['velocity_variation']
            )

        humanized.append(humanized_note)

    # Add ghost notes if configured
    if 'ghost_note_prob' in style_params and style_params['ghost_note_prob'] > 0:
        humanized = add_ghost_notes(
            humanized,
            probability=style_params['ghost_note_prob'],
            ticks_per_beat=ticks_per_beat
        )

    return humanized
