"""Model loading with LRU caching for efficient inference."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch

from src.models.transformer import DrumPatternTransformer

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


def detect_device() -> str:
    """
    Auto-detect best available device (CUDA or CPU).

    Returns:
        Device string: "cuda" if available, "cpu" otherwise
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"CUDA device detected: {device_name} ({vram_gb:.1f}GB VRAM)")
    else:
        device = "cpu"
        logger.info("No CUDA device detected, using CPU")

    return device


@lru_cache(maxsize=4)
def load_model(
    model_path: Path,
    device: Optional[str] = None
) -> Tuple[DrumPatternTransformer, Dict[str, Any]]:
    """
    Load PyTorch model with caching.

    Uses LRU cache to keep up to 4 models in memory for fast reuse.
    Automatically handles device placement and gracefully falls back to CPU on GPU OOM.

    Args:
        model_path: Path to model checkpoint (.pt or .pth)
        device: Target device ("cuda", "cpu", or None for auto-detect)

    Returns:
        Tuple of (model, metadata) where:
            - model: Loaded DrumPatternTransformer in eval mode
            - metadata: Dict with model info (vocab_size, n_styles, etc.)

    Raises:
        ModelLoadError: If model file doesn't exist or loading fails

    Example:
        >>> model, metadata = load_model(Path("models/checkpoints/j_dilla_v1.pt"))
        >>> print(f"Loaded model with vocab_size={metadata['vocab_size']}")
    """
    logger.info(f"Loading model from {model_path}")

    # Validate model path
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    if not model_path.exists():
        raise ModelLoadError(
            f"Model checkpoint not found: {model_path}\n"
            "This model will be created during Phase 6 (Training).\n"
            "For testing, consider using the MockDrumModel instead."
        )

    # Auto-detect device if not specified
    if device is None:
        device = detect_device()

    try:
        # Load checkpoint with device mapping
        logger.debug(f"Loading checkpoint to {device}")
        checkpoint = torch.load(model_path, map_location=device)

        # Extract metadata from checkpoint
        metadata = checkpoint.get('metadata', {})

        # Get model configuration from metadata
        vocab_size = metadata.get('vocab_size', 1000)
        n_styles = metadata.get('n_styles', 50)
        n_positions = metadata.get('n_positions', 2048)
        n_embd = metadata.get('n_embd', 768)
        n_layer = metadata.get('n_layer', 12)
        n_head = metadata.get('n_head', 12)
        dropout = metadata.get('dropout', 0.1)

        logger.debug(
            f"Model config: vocab_size={vocab_size}, n_styles={n_styles}, "
            f"n_embd={n_embd}, n_layer={n_layer}"
        )

        # Initialize model with checkpoint configuration
        model = DrumPatternTransformer(
            vocab_size=vocab_size,
            n_styles=n_styles,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            dropout=dropout,
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Move to device
        model.to(device)

        # Set to evaluation mode
        model.eval()

        # Add device to metadata
        metadata['device'] = device
        metadata['model_path'] = str(model_path)

        logger.info(f"Model loaded successfully on {device}")

        # Log GPU memory usage if on CUDA
        if device == "cuda":
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")

        return model, metadata

    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory, attempting to load on CPU")

        if device == "cuda":
            # Retry on CPU
            torch.cuda.empty_cache()
            return load_model(model_path, device="cpu")
        else:
            raise ModelLoadError("Failed to load model: Out of memory on CPU")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise ModelLoadError(f"Model loading failed: {str(e)}")


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory usage statistics.

    Returns:
        Dict with memory info (empty if no CUDA available):
            - allocated_gb: Currently allocated memory
            - reserved_gb: Reserved memory by PyTorch
            - max_allocated_gb: Peak memory usage
            - device_name: GPU device name
            - device_count: Number of GPUs

    Example:
        >>> info = get_gpu_memory_info()
        >>> if info:
        >>>     print(f"GPU: {info['device_name']}, {info['allocated_gb']:.2f}GB used")
    """
    if not torch.cuda.is_available():
        return {}

    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        'device_name': torch.cuda.get_device_name(0),
        'device_count': torch.cuda.device_count(),
    }


def clear_gpu_cache():
    """
    Clear GPU cache to free memory.

    Useful between generation requests to free up unused GPU memory.
    Only has effect if CUDA is available.

    Example:
        >>> clear_gpu_cache()
        >>> # Memory freed, ready for next generation
    """
    if torch.cuda.is_available():
        logger.debug("Clearing GPU cache")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU cache cleared")
    else:
        logger.debug("No CUDA device, skipping cache clear")


def clear_model_cache():
    """
    Clear the LRU model cache.

    Forces all cached models to be unloaded from memory.
    Useful when switching between many different models or to free memory.

    Example:
        >>> clear_model_cache()
        >>> # All cached models removed from memory
    """
    logger.info("Clearing model cache")
    load_model.cache_clear()
    clear_gpu_cache()
    logger.info("Model cache cleared")
