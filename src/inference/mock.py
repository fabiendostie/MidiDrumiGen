"""Mock model for testing the inference pipeline without trained weights."""

import logging
from typing import Optional, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MockDrumModel(nn.Module):
    """
    Mock model for testing pipeline without trained weights.

    Generates simple, deterministic drum patterns:
    - Kick drum on beats 1 and 3
    - Snare on beats 2 and 4
    - Closed hi-hat on all 8th notes

    This allows testing the full inference pipeline (model loading, generation,
    MIDI export) without needing actual trained model checkpoints.
    """

    # Token ID constants (matching expected tokenizer format)
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    BAR_TOKEN_ID = 10
    POSITION_TOKEN_BASE = 20  # Position 0 = 20, Position 1 = 21, etc.
    NOTE_ON_TOKEN_BASE = 100  # Note events start at 100
    VELOCITY_TOKEN_BASE = 200  # Velocity values start at 200

    # MIDI note numbers (General MIDI drum mapping)
    KICK = 36
    SNARE = 38
    CLOSED_HIHAT = 42

    def __init__(
        self,
        vocab_size: int = 500,
        n_styles: int = 50,
        n_positions: int = 2048,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
    ):
        """
        Initialize mock model with same signature as DrumPatternTransformer.

        Args:
            vocab_size: Size of token vocabulary
            n_styles: Number of producer styles
            n_positions: Maximum sequence length
            n_embd: Embedding dimension (unused in mock)
            n_layer: Number of transformer layers (unused in mock)
            n_head: Number of attention heads (unused in mock)
            dropout: Dropout rate (unused in mock)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.n_styles = n_styles
        self.n_positions = n_positions

        # Dummy parameters to satisfy model checks
        self.dummy_param = nn.Parameter(torch.zeros(1))

        logger.info(f"Initialized MockDrumModel (vocab_size={vocab_size})")

    def forward(
        self,
        input_ids: torch.Tensor,
        style_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Mock forward pass.

        Returns random logits with proper shape for testing.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            style_ids: Style IDs (batch_size,)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target tokens for training (batch_size, seq_len)

        Returns:
            SimpleNamespace with 'logits' and optional 'loss'
        """
        batch_size, seq_len = input_ids.shape

        # Generate random logits with proper shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size, device=input_ids.device)

        # Add slight bias toward BOS/EOS tokens to make generation more stable
        logits[:, :, self.BOS_TOKEN_ID] += 0.1
        logits[:, :, self.EOS_TOKEN_ID] += 0.1

        # Mock loss if labels provided
        loss = None
        if labels is not None:
            loss = torch.tensor(0.5, device=input_ids.device)

        # Return namespace mimicking HuggingFace model output
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        style_id: int,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Generate deterministic drum pattern for testing.

        Creates a simple 4-bar pattern:
        - Bar structure with position markers
        - Kick on beats 1 and 3 (quarter notes)
        - Snare on beats 2 and 4 (quarter notes)
        - Closed hi-hat on all 8th notes

        Args:
            style_id: Producer style ID (affects pattern slightly)
            max_length: Maximum sequence length (default: 512)
            temperature: Sampling temperature (unused in mock)
            top_k: Top-k sampling (unused in mock)
            top_p: Nucleus sampling (unused in mock)
            device: Device to generate on

        Returns:
            Generated token IDs as 1D tensor
        """
        logger.info(f"Mock model generating pattern (style_id={style_id})")

        pattern_tokens = [self.BOS_TOKEN_ID]

        # Generate 4 bars
        num_bars = 4
        positions_per_bar = 16  # 16th note resolution

        for bar in range(num_bars):
            # Add bar token
            pattern_tokens.append(self.BAR_TOKEN_ID + bar)

            for pos in range(positions_per_bar):
                # Add position token
                pattern_tokens.append(self.POSITION_TOKEN_BASE + pos)

                # Kick on beats 1 and 3 (positions 0 and 8)
                if pos == 0 or pos == 8:
                    pattern_tokens.append(self.NOTE_ON_TOKEN_BASE + self.KICK)
                    pattern_tokens.append(self.VELOCITY_TOKEN_BASE + 100)  # Velocity 100

                # Snare on beats 2 and 4 (positions 4 and 12)
                if pos == 4 or pos == 12:
                    pattern_tokens.append(self.NOTE_ON_TOKEN_BASE + self.SNARE)
                    pattern_tokens.append(self.VELOCITY_TOKEN_BASE + 90)  # Velocity 90

                # Hi-hat on all 8th notes (even positions: 0, 2, 4, 6, 8, 10, 12, 14)
                if pos % 2 == 0:
                    pattern_tokens.append(self.NOTE_ON_TOKEN_BASE + self.CLOSED_HIHAT)
                    # Vary velocity slightly based on style_id
                    velocity = 70 + (style_id * 5) % 20
                    pattern_tokens.append(self.VELOCITY_TOKEN_BASE + velocity)

        # Add EOS token
        pattern_tokens.append(self.EOS_TOKEN_ID)

        # Truncate if exceeds max_length
        if len(pattern_tokens) > max_length:
            pattern_tokens = pattern_tokens[:max_length]
            logger.warning(f"Pattern truncated to max_length={max_length}")

        logger.info(f"Mock pattern generated: {len(pattern_tokens)} tokens")

        return torch.tensor(pattern_tokens, dtype=torch.long, device=device)

    def to(self, device):
        """Move model to device."""
        super().to(device)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        return self


def create_mock_checkpoint(
    save_path: str,
    vocab_size: int = 500,
    n_styles: int = 50,
) -> None:
    """
    Create a mock model checkpoint file for testing.

    Saves a checkpoint with the same structure as a real model checkpoint,
    allowing the model loader to be tested without actual trained models.

    Args:
        save_path: Path where checkpoint should be saved
        vocab_size: Vocabulary size
        n_styles: Number of styles

    Example:
        >>> create_mock_checkpoint("models/checkpoints/test_mock.pt")
        >>> # Can now be loaded with load_model()
    """
    logger.info(f"Creating mock checkpoint at {save_path}")

    # Create mock model
    model = MockDrumModel(vocab_size=vocab_size, n_styles=n_styles)

    # Create checkpoint dict matching real checkpoint structure
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': {
            'vocab_size': vocab_size,
            'n_styles': n_styles,
            'n_positions': 2048,
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
            'dropout': 0.1,
            'model_type': 'mock',
            'description': 'Mock model for testing',
        },
        'epoch': 0,
        'train_loss': 0.0,
        'val_loss': 0.0,
    }

    # Save checkpoint
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)

    logger.info(f"Mock checkpoint saved: {save_path}")


def get_mock_tokens(num_bars: int = 4) -> List[int]:
    """
    Get list of mock tokens without instantiating model.

    Useful for testing tokenizer operations.

    Args:
        num_bars: Number of bars to generate tokens for

    Returns:
        List of token IDs representing a simple drum pattern
    """
    pattern_tokens = [MockDrumModel.BOS_TOKEN_ID]

    positions_per_bar = 16

    for bar in range(num_bars):
        pattern_tokens.append(MockDrumModel.BAR_TOKEN_ID + bar)

        for pos in range(positions_per_bar):
            pattern_tokens.append(MockDrumModel.POSITION_TOKEN_BASE + pos)

            # Simplified pattern: just kick and snare
            if pos == 0 or pos == 8:  # Kick on 1 and 3
                pattern_tokens.append(MockDrumModel.NOTE_ON_TOKEN_BASE + MockDrumModel.KICK)
                pattern_tokens.append(MockDrumModel.VELOCITY_TOKEN_BASE + 100)

            if pos == 4 or pos == 12:  # Snare on 2 and 4
                pattern_tokens.append(MockDrumModel.NOTE_ON_TOKEN_BASE + MockDrumModel.SNARE)
                pattern_tokens.append(MockDrumModel.VELOCITY_TOKEN_BASE + 90)

    pattern_tokens.append(MockDrumModel.EOS_TOKEN_ID)

    return pattern_tokens
