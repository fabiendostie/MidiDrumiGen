"""Training script for DrumPatternTransformer."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import logging
import time
from typing import Optional, Dict, Any
import sys

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transformer import DrumPatternTransformer
from src.training.dataset import DrumPatternDataset, create_dataloaders
from src.training.utils import (
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    calculate_perplexity,
    get_learning_rate,
    cleanup_checkpoints,
    save_training_config,
    format_time,
    log_gpu_memory,
    set_seed,
)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available, experiment tracking disabled")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for DrumPatternTransformer with mixed precision and checkpointing.

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: DrumPatternTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        checkpoint_dir: Path = Path("models/checkpoints"),
        device: str = "cuda",
        use_wandb: bool = True,
        wandb_project: str = "MidiDrumiGen",
        resume_from: Optional[Path] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: DrumPatternTransformer instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration dict
            checkpoint_dir: Directory to save checkpoints
            device: Device to train on ("cuda" or "cpu")
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            resume_from: Path to checkpoint to resume from (optional)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Extract training hyperparameters
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.max_grad_norm = config['training']['max_grad_norm']
        self.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler (warmup + cosine decay)
        warmup_ratio = config['training']['warmup_ratio']
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        # Mixed precision training
        self.use_amp = config['optimization']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing config
        self.save_interval = config['checkpointing']['save_interval']
        self.keep_last_n = config['checkpointing']['keep_last_n']

        # Logging config
        self.log_interval = config['logging']['log_interval']

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Weights & Biases logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config=config,
                name=f"train_{time.strftime('%Y%m%d_%H%M%S')}",
            )
            wandb.watch(self.model, log='all', log_freq=100)

        # Resume from checkpoint if provided
        if resume_from is not None:
            self._resume_from_checkpoint(resume_from)

        # Log model info
        param_counts = count_parameters(self.model)
        logger.info(f"Model parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    def _resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume training from a checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint_info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        self.start_epoch = checkpoint_info['epoch'] + 1
        self.best_val_loss = checkpoint_info['val_loss']

        logger.info(f"Resumed from epoch {checkpoint_info['epoch']}, val_loss={self.best_val_loss:.4f}")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            style_ids = batch['style_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        style_ids=style_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    style_ids=style_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Track loss
            total_loss += loss.item() * self.gradient_accumulation_steps

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                current_lr = get_learning_rate(self.optimizer)
                avg_loss = total_loss / (batch_idx + 1)
                perplexity = calculate_perplexity(avg_loss)

                logger.info(
                    f"Batch [{batch_idx + 1}/{num_batches}] - "
                    f"Loss: {avg_loss:.4f}, "
                    f"Perplexity: {perplexity:.2f}, "
                    f"LR: {current_lr:.2e}"
                )

                if self.use_wandb:
                    wandb.log({
                        'train/batch_loss': loss.item() * self.gradient_accumulation_steps,
                        'train/learning_rate': current_lr,
                        'train/step': self.global_step,
                    })

        avg_train_loss = total_loss / num_batches
        return avg_train_loss

    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation loop.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            style_ids = batch['style_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        style_ids=style_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    style_ids=style_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()

        avg_val_loss = total_loss / num_batches
        return avg_val_loss

    def save_model_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        # Prepare metadata
        metadata = {
            'vocab_size': self.config['model']['vocab_size'],
            'n_styles': self.config['model']['n_styles'],
            'n_positions': self.config['model']['n_positions'],
            'n_embd': self.config['model']['n_embd'],
            'n_layer': self.config['model']['n_layer'],
            'n_head': self.config['model']['n_head'],
            'dropout': self.config['model']['dropout'],
        }

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=best_path,
                metadata=metadata,
            )
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % self.save_interval == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=epoch_path,
                metadata=metadata,
            )
            logger.info(f"Saved periodic checkpoint: epoch_{epoch + 1:03d}.pt")

            # Cleanup old checkpoints
            cleanup_checkpoints(
                checkpoint_dir=self.checkpoint_dir,
                keep_last_n=self.keep_last_n,
                keep_best=True,
            )

    def train(self) -> None:
        """Run full training loop."""
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)

        # Log GPU memory before training
        if torch.cuda.is_available():
            log_gpu_memory()

        training_start_time = time.time()

        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()

            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logger.info("-" * 80)

            # Training
            train_loss = self.train_epoch()
            train_perplexity = calculate_perplexity(train_loss)

            # Validation
            val_loss = self.validate()
            val_perplexity = calculate_perplexity(val_loss)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1} complete - "
                f"Train Loss: {train_loss:.4f} (PPL: {train_perplexity:.2f}), "
                f"Val Loss: {val_loss:.4f} (PPL: {val_perplexity:.2f}), "
                f"Time: {format_time(epoch_time)}"
            )

            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/perplexity': train_perplexity,
                    'val/loss': val_loss,
                    'val/perplexity': val_perplexity,
                    'epoch_time': epoch_time,
                })

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_model_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                is_best=is_best,
            )

            # Log GPU memory
            if torch.cuda.is_available():
                memory_info = log_gpu_memory()
                if self.use_wandb and memory_info:
                    wandb.log({'gpu/memory_allocated_gb': memory_info['allocated_gb']})

        # Training complete
        total_training_time = time.time() - training_start_time
        logger.info("=" * 80)
        logger.info(f"Training complete! Total time: {format_time(total_training_time)}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation perplexity: {calculate_perplexity(self.best_val_loss):.2f}")
        logger.info("=" * 80)

        if self.use_wandb:
            wandb.finish()


def main():
    """Main training function."""
    import yaml
    from src.models.styles import PRODUCER_STYLES

    # Load configuration
    config_path = Path("configs/base.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")

    # Set random seed
    set_seed(42)

    # Create style mapping from producer styles
    style_mapping = {
        style.lower().replace(' ', '_'): params['style_id']
        for style, params in PRODUCER_STYLES.items()
    }
    logger.info(f"Style mapping: {style_mapping}")

    # Create dataloaders
    tokens_dir = Path("data/tokenized")
    if not tokens_dir.exists():
        logger.error(f"Tokenized data directory not found: {tokens_dir}")
        logger.error("Please run scripts/tokenize_dataset.py first to prepare training data")
        sys.exit(1)

    train_loader, val_loader = create_dataloaders(
        tokens_dir=tokens_dir,
        style_mapping=style_mapping,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        max_length=config['data']['max_length'],
        shuffle=config['data']['shuffle'],
    )

    # Initialize model
    model = DrumPatternTransformer(
        vocab_size=config['model']['vocab_size'],
        n_styles=config['model']['n_styles'],
        n_positions=config['model']['n_positions'],
        n_embd=config['model']['n_embd'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        dropout=config['model']['dropout'],
    )

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=Path("models/checkpoints"),
        device=device,
        use_wandb=config['logging']['use_wandb'],
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
