"""
Dynamic style transfer module - applies producer-specific transformations to MIDI patterns.

This module replaces the hardcoded humanization parameters with dynamic parameters
from the ProducerResearchAgent, enabling ANY-producer drum generation.
"""

import logging
import random
from typing import Any

from .constants import DEFAULT_TICKS_PER_BEAT
from .humanize import (
    add_ghost_notes,
    apply_micro_timing,
    apply_swing,
    apply_velocity_variation,
)

logger = logging.getLogger(__name__)


def remove_duplicate_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate notes (same pitch at same time).

    Keeps the first occurrence of each note.
    """
    seen = set()
    unique_notes = []

    for note in notes:
        key = (note["time"], note["pitch"])
        if key not in seen:
            seen.add(key)
            unique_notes.append(note)

    return unique_notes


def apply_producer_style(
    notes: list[dict[str, Any]],
    style_profile: dict[str, Any],
    tempo: int,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
) -> list[dict[str, Any]]:
    """
    Apply producer-specific style transformations to MIDI notes.

    This is the main entry point for style transfer. It extracts parameters
    from the producer profile (researched by ProducerResearchAgent) and applies
    all relevant transformations.

    Args:
        notes: List of MIDI note events with 'pitch', 'velocity', 'time' keys
        style_profile: Producer profile from ProducerResearchAgent
        tempo: BPM
        ticks_per_beat: MIDI resolution (default: 480)

    Returns:
        Styled note events with transformations applied

    Example:
        >>> from src.research.producer_agent import quick_research
        >>> profile = await quick_research("Timbaland")
        >>> styled_notes = apply_producer_style(notes, profile, tempo=100)
    """
    if not notes:
        logger.warning("Empty notes list provided, returning empty list")
        return []

    style_params = style_profile.get("style_params", {})
    producer_name = style_profile.get("producer_name", "Unknown")

    logger.info(
        f"Applying {producer_name} style transfer: "
        f"swing={style_params.get('swing_percentage', 'N/A')}%, "
        f"micro_timing={style_params.get('micro_timing_ms', 'N/A')}ms, "
        f"complexity={style_params.get('complexity_level', 'N/A')}"
    )

    # Copy notes to avoid modifying original
    styled_notes = [note.copy() for note in notes]

    # 1. Apply swing timing
    styled_notes = apply_style_swing(styled_notes, style_params, ticks_per_beat)

    # 2. Apply micro-timing variations
    styled_notes = apply_style_micro_timing(styled_notes, style_params, tempo, ticks_per_beat)

    # 3. Apply velocity shaping
    styled_notes = apply_style_velocity(styled_notes, style_params)

    # 4. Add ghost notes
    styled_notes = apply_style_ghost_notes(styled_notes, style_params, ticks_per_beat)

    # 5. Apply quantization grid adjustments
    styled_notes = apply_quantization_grid(styled_notes, style_params, ticks_per_beat)

    # 6. Apply signature techniques (producer-specific modifications)
    styled_notes = apply_signature_techniques(styled_notes, style_params, tempo, ticks_per_beat)

    # 7. Remove any duplicate notes (same pitch at same time)
    notes_before_dedup = len(styled_notes)
    styled_notes = remove_duplicate_notes(styled_notes)
    duplicates_removed = notes_before_dedup - len(styled_notes)

    if duplicates_removed > 0:
        logger.debug(f"Removed {duplicates_removed} duplicate notes")

    logger.info(
        f"Style transfer complete: {len(notes)} → {len(styled_notes)} notes "
        f"({len(styled_notes) - len(notes):+d} added, {duplicates_removed} duplicates removed)"
    )

    return styled_notes


def apply_style_swing(
    notes: list[dict[str, Any]],
    style_params: dict[str, Any],
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Apply swing timing from style parameters.

    Args:
        notes: List of note events
        style_params: Style parameters dict with 'swing_percentage' key
        ticks_per_beat: MIDI resolution

    Returns:
        Notes with swing applied
    """
    swing_percentage = style_params.get("swing_percentage", 54.0)

    if swing_percentage <= 50.0:
        logger.debug("Swing disabled (≤50%), skipping swing transformation")
        return notes

    logger.debug(f"Applying {swing_percentage}% swing")

    for note in notes:
        note["time"] = apply_swing(note["time"], swing_percentage, ticks_per_beat)

    return notes


def apply_style_micro_timing(
    notes: list[dict[str, Any]],
    style_params: dict[str, Any],
    tempo: int,
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Apply micro-timing variations from style parameters.

    Args:
        notes: List of note events
        style_params: Style parameters dict with 'micro_timing_ms' key
        tempo: BPM
        ticks_per_beat: MIDI resolution

    Returns:
        Notes with micro-timing applied
    """
    micro_timing_ms = style_params.get("micro_timing_ms", 0.0)

    if micro_timing_ms <= 0.0:
        logger.debug("Micro-timing disabled (≤0ms), skipping")
        return notes

    logger.debug(f"Applying ±{micro_timing_ms}ms micro-timing")

    for note in notes:
        note["time"] = apply_micro_timing(note["time"], micro_timing_ms, tempo, ticks_per_beat)

    return notes


def apply_style_velocity(
    notes: list[dict[str, Any]],
    style_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Apply velocity variation from style parameters.

    Args:
        notes: List of note events
        style_params: Style parameters dict with 'velocity_variation' key

    Returns:
        Notes with velocity variation applied
    """
    velocity_variation = style_params.get("velocity_variation", 0.1)

    if velocity_variation <= 0.0:
        logger.debug("Velocity variation disabled (≤0), skipping")
        return notes

    logger.debug(f"Applying {velocity_variation*100:.1f}% velocity variation")

    for note in notes:
        note["velocity"] = apply_velocity_variation(note["velocity"], variation=velocity_variation)

    return notes


def apply_style_ghost_notes(
    notes: list[dict[str, Any]],
    style_params: dict[str, Any],
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Add ghost notes based on style parameters.

    Args:
        notes: List of note events
        style_params: Style parameters dict with 'ghost_note_prob' key
        ticks_per_beat: MIDI resolution

    Returns:
        Notes with ghost notes added
    """
    ghost_note_prob = style_params.get("ghost_note_prob", 0.0)

    if ghost_note_prob <= 0.0:
        logger.debug("Ghost notes disabled (prob≤0), skipping")
        return notes

    logger.debug(f"Adding ghost notes with {ghost_note_prob*100:.1f}% probability")

    # Add ghost notes using existing function
    notes_with_ghosts = add_ghost_notes(
        notes,
        probability=ghost_note_prob,
        ghost_velocity=30,
        ghost_note=38,  # Snare
        ticks_per_beat=ticks_per_beat,
    )

    return notes_with_ghosts


def apply_quantization_grid(
    notes: list[dict[str, Any]],
    style_params: dict[str, Any],
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Apply quantization grid adjustments.

    Different producers use different rhythmic grids:
    - "16th": Standard 16th note grid (most common)
    - "32nd": 32nd note grid (more detailed, complex patterns)
    - "triplet": Triplet-based grid (jazz, hip-hop)

    Args:
        notes: List of note events
        style_params: Style parameters dict with 'quantization_grid' key
        ticks_per_beat: MIDI resolution

    Returns:
        Notes with quantization adjustments applied
    """
    quantization_grid = style_params.get("quantization_grid", "16th")

    logger.debug(f"Applying {quantization_grid} quantization grid")

    if quantization_grid == "32nd":
        # 32nd note grid: tighter quantization
        grid_size = ticks_per_beat / 8
    elif quantization_grid == "triplet":
        # Triplet grid: divide by 6 (triplets per beat)
        grid_size = ticks_per_beat / 6
    else:
        # Default 16th note grid
        grid_size = ticks_per_beat / 4

    # Quantize note times to grid (light quantization, not full snap)
    for note in notes:
        # Calculate nearest grid position
        grid_position = round(note["time"] / grid_size) * grid_size

        # Apply 50% quantization (blend between original and grid)
        note["time"] = int((note["time"] + grid_position) / 2)

    return notes


def apply_signature_techniques(
    notes: list[dict[str, Any]],
    style_params: dict[str, Any],
    tempo: int,
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Apply producer-specific signature techniques.

    This function interprets the 'signature_techniques' from the LLM
    and applies appropriate transformations. Currently implements
    rule-based heuristics based on technique names.

    Args:
        notes: List of note events
        style_params: Style parameters dict with 'signature_techniques' key
        tempo: BPM
        ticks_per_beat: MIDI resolution

    Returns:
        Notes with signature techniques applied
    """
    techniques = style_params.get("signature_techniques", [])

    if not techniques:
        logger.debug("No signature techniques specified")
        return notes

    logger.debug(f"Applying signature techniques: {techniques[:3]}...")

    # Convert to lowercase for case-insensitive matching
    techniques_lower = [t.lower() for t in techniques]

    # Apply technique-specific transformations
    if any("stutter" in t for t in techniques_lower):
        notes = apply_stuttering_effect(notes, ticks_per_beat)

    if any("syncopat" in t for t in techniques_lower):
        notes = apply_syncopation(notes, ticks_per_beat)

    if any("sparse" in t for t in techniques_lower):
        notes = apply_sparse_pattern(notes)

    if any("complex" in t or "polyrhythm" in t for t in techniques_lower):
        notes = apply_complexity_boost(notes, ticks_per_beat)

    if any("shuffle" in t or "2-step" in t or "2 step" in t for t in techniques_lower):
        notes = apply_shuffle_timing(notes, ticks_per_beat)

    return notes


def apply_stuttering_effect(
    notes: list[dict[str, Any]],
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Apply stuttering/repetition effect (Timbaland signature).

    Adds quick repetitions of certain notes for a stuttering effect.
    """
    logger.debug("Applying stuttering effect")

    new_notes = []
    stutter_probability = 0.15  # 15% chance per note

    for note in notes:
        new_notes.append(note.copy())

        # Randomly add stutter
        if random.random() < stutter_probability:
            # Add 1-2 quick repetitions
            num_stutters = random.randint(1, 2)
            stutter_gap = ticks_per_beat // 16  # 64th note

            for i in range(1, num_stutters + 1):
                stutter_note = note.copy()
                stutter_note["time"] = note["time"] + (i * stutter_gap)
                stutter_note["velocity"] = max(1, note["velocity"] - (i * 10))  # Decay
                new_notes.append(stutter_note)

    return new_notes


def apply_syncopation(
    notes: list[dict[str, Any]],
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Enhance syncopation by shifting some notes off the beat.
    """
    logger.debug("Enhancing syncopation")

    for note in notes:
        # Check if note is on a strong beat (downbeat)
        position = note["time"] % ticks_per_beat
        is_downbeat = position == 0

        # 20% chance to shift downbeat notes slightly off-beat
        if is_downbeat and random.random() < 0.2:
            offset = ticks_per_beat // 16  # Shift by 64th note
            note["time"] += offset

    return notes


def apply_sparse_pattern(
    notes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create sparser patterns by randomly removing notes (subtle).
    """
    logger.debug("Applying sparse pattern effect")

    # Remove 5-10% of notes randomly (excluding first and last)
    if len(notes) <= 2:
        return notes

    num_to_remove = max(0, int(len(notes) * 0.05))  # Remove ~5%

    if num_to_remove == 0:
        return notes

    # Sort by time to identify candidates
    sorted_notes = sorted(enumerate(notes), key=lambda x: x[1]["time"])

    # Randomly select notes to remove (exclude first and last)
    candidates = sorted_notes[1:-1]
    if len(candidates) <= num_to_remove:
        return notes

    to_remove_indices = random.sample(range(len(candidates)), num_to_remove)
    to_remove_original_indices = [candidates[i][0] for i in to_remove_indices]

    # Filter out removed notes
    filtered_notes = [note for i, note in enumerate(notes) if i not in to_remove_original_indices]

    return filtered_notes


def apply_complexity_boost(
    notes: list[dict[str, Any]],
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Add complexity through additional hi-hat patterns or fills.
    """
    logger.debug("Boosting pattern complexity")

    new_notes = notes.copy()

    # Add occasional hi-hat hits between existing notes
    hi_hat_closed = 42
    fill_probability = 0.1  # 10% chance per gap

    sorted_notes = sorted(notes, key=lambda n: n["time"])

    for i in range(len(sorted_notes) - 1):
        gap = sorted_notes[i + 1]["time"] - sorted_notes[i]["time"]

        # If gap is large enough and random check passes, consider adding fill
        if gap >= ticks_per_beat / 4 and random.random() < fill_probability:
            # Add hi-hat in the middle of the gap
            fill_time = (sorted_notes[i]["time"] + sorted_notes[i + 1]["time"]) // 2
            new_notes.append(
                {
                    "pitch": hi_hat_closed,
                    "velocity": random.randint(40, 60),
                    "time": fill_time,
                }
            )

    return new_notes


def apply_shuffle_timing(
    notes: list[dict[str, Any]],
    ticks_per_beat: int,
) -> list[dict[str, Any]]:
    """
    Apply shuffle timing (Burial, garage, UK style).

    Similar to swing but with more irregular timing shifts.
    """
    logger.debug("Applying shuffle timing")

    for note in notes:
        # Apply irregular timing shifts (more dramatic than swing)
        position = note["time"] % ticks_per_beat
        sixteenth_position = position / (ticks_per_beat / 4)

        # Shift every other 16th note
        if int(sixteenth_position) % 2 == 1:
            # Random shift forward (shuffle feel)
            shift = random.randint(0, int(ticks_per_beat / 16))
            note["time"] += shift

    return notes


def get_style_description(style_profile: dict[str, Any]) -> str:
    """
    Generate human-readable description of applied style.

    Args:
        style_profile: Producer profile from research agent

    Returns:
        Formatted string describing the style characteristics

    Example:
        >>> description = get_style_description(profile)
        >>> print(description)
        Timbaland style: stuttering rhythms, syncopated hi-hats (Swing: 54%, Complexity: 0.65)
    """
    style_params = style_profile.get("style_params", {})
    producer_name = style_profile.get("producer_name", "Unknown")

    techniques = style_params.get("signature_techniques", [])
    swing = style_params.get("swing_percentage", "N/A")
    complexity = style_params.get("complexity_level", "N/A")

    # Format techniques (first 3)
    techniques_str = ", ".join(techniques[:3]) if techniques else "standard patterns"

    description = (
        f"{producer_name} style: {techniques_str} " f"(Swing: {swing}%, Complexity: {complexity})"
    )

    return description


def validate_style_profile(style_profile: dict[str, Any]) -> bool:
    """
    Validate that style profile has required keys.

    Args:
        style_profile: Producer profile to validate

    Returns:
        True if valid, False otherwise (logs warnings)
    """
    required_keys = ["producer_name", "style_params"]
    required_style_keys = [
        "tempo_range",
        "swing_percentage",
        "micro_timing_ms",
        "ghost_note_prob",
        "velocity_variation",
    ]

    # Check top-level keys
    for key in required_keys:
        if key not in style_profile:
            logger.warning(f"Style profile missing required key: {key}")
            return False

    # Check style_params keys
    style_params = style_profile.get("style_params", {})
    for key in required_style_keys:
        if key not in style_params:
            logger.warning(f"Style params missing required key: {key}")
            return False

    return True
