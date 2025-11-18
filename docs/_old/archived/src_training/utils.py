"""Training utilities and helper functions."""

import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Dict with 'total' and 'trainable' parameter counts

    Example:
        >>> from src.models.transformer import DrumPatternTransformer
        >>> model = DrumPatternTransformer(vocab_size=1000, n_styles=50)
        >>> params = count_parameters(model)
        >>> print(f"Total: {params['total']:,}, Trainable: {params['trainable']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save training checkpoint with model state and metadata.

    Checkpoint format matches what load_model() expects in src/inference/model_loader.py

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch number
        train_loss: Average training loss
        val_loss: Validation loss
        checkpoint_path: Path to save checkpoint
        metadata: Additional metadata (vocab_size, n_styles, etc.)

    Example:
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     epoch=10,
        ...     train_loss=1.234,
        ...     val_loss=1.456,
        ...     checkpoint_path=Path("models/checkpoints/best_model.pt"),
        ...     metadata={'vocab_size': 1000, 'n_styles': 50}
        ... )
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Add metadata (required for model loading)
    if metadata is not None:
        checkpoint['metadata'] = metadata

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to

    Returns:
        Dict with checkpoint metadata (epoch, losses, etc.)

    Example:
        >>> info = load_checkpoint(
        ...     checkpoint_path=Path("models/checkpoints/best_model.pt"),
        ...     model=model,
        ...     optimizer=optimizer,
        ...     device="cuda"
        ... )
        >>> print(f"Resuming from epoch {info['epoch']}")
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model weights from {checkpoint_path}")

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded optimizer state")

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Loaded scheduler state")

    # Return metadata
    return {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', 0.0),
        'val_loss': checkpoint.get('val_loss', 0.0),
        'metadata': checkpoint.get('metadata', {}),
    }


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.

    Perplexity = exp(loss)
    Lower is better. Perplexity < 10 is good, < 5 is excellent.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value

    Example:
        >>> loss = 2.3
        >>> perplexity = calculate_perplexity(loss)
        >>> print(f"Perplexity: {perplexity:.2f}")
        Perplexity: 9.97
    """
    return torch.exp(torch.tensor(loss)).item()


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate

    Example:
        >>> lr = get_learning_rate(optimizer)
        >>> print(f"Current LR: {lr:.2e}")
    """
    return optimizer.param_groups[0]['lr']


def cleanup_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int = 3,
    keep_best: bool = True,
    best_checkpoint_name: str = "best_model.pt",
) -> None:
    """
    Clean up old checkpoints, keeping only the most recent N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of most recent checkpoints to keep
        keep_best: If True, always keep the best checkpoint
        best_checkpoint_name: Name of the best model checkpoint

    Example:
        >>> cleanup_checkpoints(
        ...     checkpoint_dir=Path("models/checkpoints"),
        ...     keep_last_n=3,
        ...     keep_best=True
        ... )
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return

    # Get all checkpoint files
    checkpoints = list(checkpoint_dir.glob("*.pt"))

    # Separate best checkpoint from others
    best_checkpoint = checkpoint_dir / best_checkpoint_name
    other_checkpoints = [c for c in checkpoints if c != best_checkpoint]

    # Sort by modification time (newest first)
    other_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Determine which to delete
    to_delete = other_checkpoints[keep_last_n:]

    # Delete old checkpoints
    for checkpoint in to_delete:
        logger.info(f"Removing old checkpoint: {checkpoint.name}")
        checkpoint.unlink()

    if to_delete:
        logger.info(f"Cleaned up {len(to_delete)} old checkpoints")


def save_training_config(
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save training configuration to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config

    Example:
        >>> config = {
        ...     'model': {'vocab_size': 1000, 'n_styles': 50},
        ...     'training': {'batch_size': 32, 'learning_rate': 5e-5}
        ... }
        >>> save_training_config(config, Path("runs/experiment_1/config.json"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved training config to {output_path}")


def format_time(seconds: float) -> str:
    """
    Format seconds as human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (HH:MM:SS)

    Example:
        >>> format_time(3661)
        '01:01:01'
        >>> format_time(125)
        '00:02:05'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_gpu_memory() -> Dict[str, float]:
    """
    Log current GPU memory usage.

    Returns:
        Dict with memory statistics in GB

    Example:
        >>> if torch.cuda.is_available():
        ...     memory = log_gpu_memory()
        ...     print(f"GPU Memory: {memory['allocated_gb']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    memory_info = {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
        'device_name': torch.cuda.get_device_name(0),
    }

    logger.debug(
        f"GPU Memory - Allocated: {allocated:.2f}GB, "
        f"Reserved: {reserved:.2f}GB, "
        f"Max: {max_allocated:.2f}GB"
    )

    return memory_info


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: For full reproducibility, also set:
    # - np.random.seed(seed)
    # - random.seed(seed)
    # - torch.backends.cudnn.deterministic = True
    # - torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")
