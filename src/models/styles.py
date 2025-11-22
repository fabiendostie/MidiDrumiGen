"""Producer style registry with model mappings and humanization parameters."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Producer style registry mapping style names to model IDs and parameters
PRODUCER_STYLES: dict[str, dict[str, Any]] = {
    "J Dilla": {
        "style_id": 0,  # Numeric ID for model conditioning
        "model_id": "j_dilla_v1",
        "model_path": "models/checkpoints/j_dilla_v1.pt",
        "description": "Signature swing and soulful groove",
        "preferred_tempo_range": (85, 95),
        "humanization": {
            "swing": 62.0,  # Signature Dilla swing
            "micro_timing_ms": 20.0,  # More timing variation
            "ghost_note_prob": 0.4,
            "velocity_variation": 0.15,
        },
    },
    "Metro Boomin": {
        "style_id": 1,  # Numeric ID for model conditioning
        "model_id": "metro_boomin_v1",
        "model_path": "models/checkpoints/metro_boomin_v1.pt",
        "description": "Tight trap drums with rolls",
        "preferred_tempo_range": (130, 150),
        "humanization": {
            "swing": 52.0,  # Straighter timing
            "micro_timing_ms": 5.0,  # Tighter quantization
            "ghost_note_prob": 0.1,
            "velocity_variation": 0.08,
        },
    },
    "Questlove": {
        "style_id": 2,  # Numeric ID for model conditioning
        "model_id": "questlove_v1",
        "model_path": "models/checkpoints/questlove_v1.pt",
        "description": "Dynamic funk drumming with ghost notes",
        "preferred_tempo_range": (90, 110),
        "humanization": {
            "swing": 58.0,
            "micro_timing_ms": 12.0,
            "ghost_note_prob": 0.5,  # Lots of ghost notes
            "velocity_variation": 0.20,  # Very dynamic
        },
    },
    "Timbaland": {
        "style_id": 3,  # Numeric ID for model conditioning
        "model_id": "timbaland_v1",
        "model_path": "models/checkpoints/timbaland_v1.pt",
        "description": "Syncopated funk with experimental rhythms",
        "preferred_tempo_range": (95, 115),
        "humanization": {
            "swing": 54.0,
            "micro_timing_ms": 10.0,
            "ghost_note_prob": 0.35,
            "velocity_variation": 0.12,
        },
    },
}


# Style name aliases for case-insensitive lookup
STYLE_ALIASES: dict[str, str] = {
    "j dilla": "J Dilla",
    "j_dilla": "J Dilla",
    "jdilla": "J Dilla",
    "dilla": "J Dilla",
    "metro boomin": "Metro Boomin",
    "metro_boomin": "Metro Boomin",
    "metroboomin": "Metro Boomin",
    "metro": "Metro Boomin",
    "questlove": "Questlove",
    "quest": "Questlove",
    "timbaland": "Timbaland",
    "timbo": "Timbaland",
}


class StyleNotFoundError(Exception):
    """Raised when a producer style is not found in the registry."""

    pass


def normalize_style_name(style_name: str) -> str:
    """
    Normalize style name for lookup.

    Handles case-insensitive lookup and common aliases.

    Args:
        style_name: Style name (e.g., "j dilla", "J Dilla", "jdilla")

    Returns:
        Normalized style name from PRODUCER_STYLES

    Raises:
        StyleNotFoundError: If style not found

    Example:
        >>> normalize_style_name("j dilla")
        'J Dilla'
        >>> normalize_style_name("metro")
        'Metro Boomin'
    """
    # Try exact match first
    if style_name in PRODUCER_STYLES:
        return style_name

    # Try lowercase lookup
    lower_name = style_name.lower()
    if lower_name in STYLE_ALIASES:
        return STYLE_ALIASES[lower_name]

    # Try case-insensitive search in registry
    for key in PRODUCER_STYLES:
        if key.lower() == lower_name:
            return key

    raise StyleNotFoundError(
        f"Style '{style_name}' not found. Available styles: {list_available_styles()}"
    )


def get_style_id(style_name: str) -> str:
    """
    Get model ID from style name.

    Args:
        style_name: Producer style name (e.g., "J Dilla")

    Returns:
        Model ID string (e.g., "j_dilla_v1")

    Raises:
        StyleNotFoundError: If style not found

    Example:
        >>> get_style_id("J Dilla")
        'j_dilla_v1'
    """
    normalized_name = normalize_style_name(style_name)
    return PRODUCER_STYLES[normalized_name]["model_id"]


def get_numeric_style_id(style_name: str) -> int:
    """
    Get numeric style ID from style name for model conditioning.

    Args:
        style_name: Producer style name (e.g., "J Dilla")

    Returns:
        Numeric style ID (e.g., 0 for J Dilla, 1 for Metro Boomin, 2 for Questlove)

    Raises:
        StyleNotFoundError: If style not found

    Example:
        >>> get_numeric_style_id("J Dilla")
        0
    """
    normalized_name = normalize_style_name(style_name)
    return PRODUCER_STYLES[normalized_name]["style_id"]


def get_style_params(style_name: str) -> dict[str, Any]:
    """
    Get all parameters for a producer style.

    Returns the complete style configuration including humanization parameters.

    Args:
        style_name: Producer style name

    Returns:
        Dict with all style parameters (humanization, tempo, description, etc.)

    Raises:
        StyleNotFoundError: If style not found

    Example:
        >>> params = get_style_params("J Dilla")
        >>> print(params['humanization']['swing'])
        62.0
    """
    normalized_name = normalize_style_name(style_name)
    return PRODUCER_STYLES[normalized_name]


def get_humanization_params(style_name: str) -> dict[str, Any]:
    """
    Get humanization parameters for a style.

    Args:
        style_name: Producer style name

    Returns:
        Dict with humanization params (swing, micro_timing_ms, etc.)

    Raises:
        StyleNotFoundError: If style not found

    Example:
        >>> params = get_humanization_params("Metro Boomin")
        >>> params['swing']
        52.0
    """
    normalized_name = normalize_style_name(style_name)
    return PRODUCER_STYLES[normalized_name]["humanization"]


def get_model_path(style_name: str, base_dir: Path | None = None) -> Path:
    """
    Resolve model checkpoint path for a style.

    Args:
        style_name: Producer style name
        base_dir: Base directory for models (default: current directory)

    Returns:
        Absolute path to model checkpoint

    Raises:
        StyleNotFoundError: If style not found

    Example:
        >>> path = get_model_path("J Dilla")
        >>> print(path)
        C:\\...\\models\\checkpoints\\j_dilla_v1.pt
    """
    normalized_name = normalize_style_name(style_name)
    relative_path = PRODUCER_STYLES[normalized_name]["model_path"]

    if base_dir is None:
        base_dir = Path.cwd()

    model_path = base_dir / relative_path
    return model_path.resolve()


def list_available_styles() -> list[str]:
    """
    Return list of all available producer styles.

    Returns:
        List of style names in PRODUCER_STYLES

    Example:
        >>> styles = list_available_styles()
        >>> print(styles)
        ['J Dilla', 'Metro Boomin', 'Questlove']
    """
    return list(PRODUCER_STYLES.keys())


def get_preferred_tempo_range(style_name: str) -> tuple[int, int]:
    """
    Get preferred tempo range for a style.

    Args:
        style_name: Producer style name

    Returns:
        Tuple of (min_bpm, max_bpm)

    Raises:
        StyleNotFoundError: If style not found

    Example:
        >>> min_bpm, max_bpm = get_preferred_tempo_range("J Dilla")
        >>> print(f"Preferred tempo: {min_bpm}-{max_bpm} BPM")
        Preferred tempo: 85-95 BPM
    """
    normalized_name = normalize_style_name(style_name)
    return PRODUCER_STYLES[normalized_name]["preferred_tempo_range"]


def validate_tempo_for_style(style_name: str, tempo: int, warn_only: bool = True) -> bool:
    """
    Check if tempo is within preferred range for a style.

    Args:
        style_name: Producer style name
        tempo: BPM to validate
        warn_only: If True, log warning instead of raising error

    Returns:
        True if tempo is in range, False otherwise

    Example:
        >>> validate_tempo_for_style("J Dilla", 120)  # Returns False, logs warning
        >>> validate_tempo_for_style("Metro Boomin", 140)  # Returns True
    """
    try:
        min_bpm, max_bpm = get_preferred_tempo_range(style_name)

        if min_bpm <= tempo <= max_bpm:
            return True

        message = (
            f"Tempo {tempo} BPM is outside preferred range for {style_name} "
            f"({min_bpm}-{max_bpm} BPM). Results may not match typical style."
        )

        if warn_only:
            logger.warning(message)
            return False
        else:
            raise ValueError(message)

    except StyleNotFoundError:
        # If style not found, validation passes (will be caught elsewhere)
        return True


def get_style_description(style_name: str) -> str:
    """
    Get human-readable description of a style.

    Args:
        style_name: Producer style name

    Returns:
        Description string

    Raises:
        StyleNotFoundError: If style not found

    Example:
        >>> desc = get_style_description("J Dilla")
        >>> print(desc)
        Signature swing and soulful groove
    """
    normalized_name = normalize_style_name(style_name)
    return PRODUCER_STYLES[normalized_name]["description"]


def get_all_styles_info() -> dict[str, dict[str, Any]]:
    """
    Get complete information about all available styles.

    Useful for API endpoints that need to return style catalog.

    Returns:
        Dict mapping style names to their full configurations

    Example:
        >>> info = get_all_styles_info()
        >>> for style, params in info.items():
        ...     print(f"{style}: {params['description']}")
    """
    return PRODUCER_STYLES.copy()
