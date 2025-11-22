"""Pattern validation for drum patterns."""

import logging

from .constants import DEFAULT_TICKS_PER_BEAT, MAX_DRUM_NOTE, MIN_DRUM_NOTE

logger = logging.getLogger(__name__)


# Physically impossible simultaneous drum hits
EXCLUSIVE_PAIRS = [
    (42, 46),  # Closed Hi-Hat + Open Hi-Hat (same instrument)
    (44, 42),  # Pedal Hi-Hat + Closed Hi-Hat (same instrument)
    (44, 46),  # Pedal Hi-Hat + Open Hi-Hat (same instrument)
]


def validate_drum_pattern(
    notes: list[dict], ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT, allow_empty: bool = False
) -> tuple[bool, list[str]]:
    """
    Validate drum pattern for musical correctness and physical constraints.

    Checks:
    1. Note range (35-81 for GM drums)
    2. Velocity range (1-127)
    3. Pattern density (not too sparse/dense)
    4. No impossible simultaneous hits (e.g., closed/open hi-hat)
    5. Maximum simultaneous hits limit (realistic playing)

    Args:
        notes: List of note dicts with 'pitch', 'velocity', 'time' keys
        ticks_per_beat: MIDI resolution (default: 480)
        allow_empty: Whether to allow empty patterns (default: False)

    Returns:
        (is_valid, error_messages) tuple
        - is_valid: True if pattern passes all checks
        - error_messages: List of validation errors (empty if valid)

    Example:
        >>> notes = [{'pitch': 36, 'velocity': 100, 'time': 0}]
        >>> is_valid, errors = validate_drum_pattern(notes)
        >>> if not is_valid:
        ...     print(f"Validation failed: {errors}")
    """
    errors = []

    # 0. Check for empty pattern
    if not notes:
        if not allow_empty:
            errors.append("Pattern is empty (no notes)")
        return len(errors) == 0, errors

    # 1. Validate note range (GM drums: 35-81)
    for i, note in enumerate(notes):
        pitch = note.get("pitch")
        if pitch is None:
            errors.append(f"Note {i}: missing 'pitch' key")
            continue

        if not (MIN_DRUM_NOTE <= pitch <= MAX_DRUM_NOTE):
            errors.append(
                f"Note {i}: invalid drum note {pitch} " f"(must be {MIN_DRUM_NOTE}-{MAX_DRUM_NOTE})"
            )

    # 2. Validate velocity range (1-127, 0 is reserved for note off)
    for i, note in enumerate(notes):
        velocity = note.get("velocity")
        if velocity is None:
            errors.append(f"Note {i}: missing 'velocity' key")
            continue

        if not (1 <= velocity <= 127):
            errors.append(f"Note {i}: invalid velocity {velocity} (must be 1-127)")

    # 3. Validate time values (must be non-negative)
    for i, note in enumerate(notes):
        time = note.get("time")
        if time is None:
            errors.append(f"Note {i}: missing 'time' key")
            continue

        if time < 0:
            errors.append(f"Note {i}: negative time value {time}")

    # Stop here if basic structure is invalid
    if errors:
        return False, errors

    # 4. Check pattern density (notes per beat)
    if notes:
        # Find pattern duration
        max_time = max(n["time"] for n in notes)
        duration_in_beats = max_time / ticks_per_beat

        if duration_in_beats > 0:
            density = len(notes) / duration_in_beats

            # Reasonable density limits
            min_density = 0.5  # At least 0.5 notes per beat
            max_density = 16  # Max 16 notes per beat (64th notes)

            if density < min_density:
                errors.append(
                    f"Pattern too sparse ({density:.2f} notes/beat, " f"minimum {min_density})"
                )
            elif density > max_density:
                errors.append(
                    f"Pattern too dense ({density:.2f} notes/beat, " f"maximum {max_density})"
                )

    # 5. Group notes by time for simultaneous hit checking
    time_groups: dict[int, list[int]] = {}
    for note in notes:
        time = note["time"]
        pitch = note["pitch"]
        time_groups.setdefault(time, []).append(pitch)

    # 6. Check for impossible simultaneous hits
    for time, pitches in time_groups.items():
        # Check exclusive pairs (physically impossible)
        for p1, p2 in EXCLUSIVE_PAIRS:
            if p1 in pitches and p2 in pitches:
                errors.append(
                    f"Impossible simultaneous notes at time {time}: "
                    f"{p1} and {p2} (same physical instrument)"
                )

    # 7. Check maximum simultaneous hits (realistic limit)
    max_simultaneous = 4  # Human has 4 limbs
    for time, pitches in time_groups.items():
        if len(pitches) > max_simultaneous:
            errors.append(
                f"Too many simultaneous hits at time {time}: "
                f"{len(pitches)} notes (maximum {max_simultaneous})"
            )

    # 8. Check for duplicate notes at same time (same pitch multiple times)
    for time, pitches in time_groups.items():
        unique_pitches = set(pitches)
        if len(unique_pitches) < len(pitches):
            duplicates = [p for p in pitches if pitches.count(p) > 1]
            errors.append(f"Duplicate notes at time {time}: {set(duplicates)}")

    is_valid = len(errors) == 0

    if is_valid:
        logger.debug(f"Pattern validation passed: {len(notes)} notes")
    else:
        logger.warning(f"Pattern validation failed: {len(errors)} errors")

    return is_valid, errors


def validate_pattern_structure(notes: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate that note dictionaries have required structure.

    Basic structural validation before musical validation.

    Args:
        notes: List of note dicts

    Returns:
        (is_valid, error_messages) tuple

    Example:
        >>> notes = [{'pitch': 36}]  # Missing velocity and time
        >>> is_valid, errors = validate_pattern_structure(notes)
        >>> print(errors)  # ['Note 0: missing velocity', 'Note 0: missing time']
    """
    errors = []
    required_keys = ["pitch", "velocity", "time"]

    for i, note in enumerate(notes):
        if not isinstance(note, dict):
            errors.append(f"Note {i}: not a dictionary")
            continue

        for key in required_keys:
            if key not in note:
                errors.append(f"Note {i}: missing '{key}' key")

    return len(errors) == 0, errors


def get_pattern_statistics(notes: list[dict], ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT) -> dict:
    """
    Calculate pattern statistics for debugging/analysis.

    Args:
        notes: List of note dicts
        ticks_per_beat: MIDI resolution (default: 480)

    Returns:
        Dictionary with pattern statistics

    Example:
        >>> notes = [{'pitch': 36, 'velocity': 100, 'time': 0}]
        >>> stats = get_pattern_statistics(notes)
        >>> print(stats['total_notes'])  # 1
    """
    if not notes:
        return {
            "total_notes": 0,
            "unique_pitches": 0,
            "duration_beats": 0,
            "density": 0,
            "velocity_range": (0, 0),
            "time_range": (0, 0),
        }

    pitches = [n["pitch"] for n in notes]
    velocities = [n["velocity"] for n in notes]
    times = [n["time"] for n in notes]

    duration_beats = (max(times) - min(times)) / ticks_per_beat if times else 0
    density = len(notes) / duration_beats if duration_beats > 0 else 0

    return {
        "total_notes": len(notes),
        "unique_pitches": len(set(pitches)),
        "duration_beats": duration_beats,
        "density": density,
        "velocity_range": (min(velocities), max(velocities)),
        "time_range": (min(times), max(times)),
        "pitch_counts": {p: pitches.count(p) for p in set(pitches)},
    }
