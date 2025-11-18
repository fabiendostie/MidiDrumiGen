"""Dataset class for tokenized drum patterns."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DrumPatternDataset(Dataset):
    """
    Dataset for tokenized drum patterns with style conditioning.

    Loads pre-tokenized MIDI sequences from disk along with style labels.
    Handles padding/truncation to max_length and creates attention masks.

    Example:
        >>> dataset = DrumPatternDataset(
        ...     tokens_dir="data/tokenized",
        ...     style_mapping={"j_dilla": 0, "metro_boomin": 1},
        ...     max_length=512
        ... )
        >>> batch = dataset[0]
        >>> print(batch.keys())
        dict_keys(['input_ids', 'labels', 'style_ids', 'attention_mask'])
    """

    def __init__(
        self,
        tokens_dir: str | Path,
        style_mapping: Dict[str, int],
        max_length: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        validate_data: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            tokens_dir: Directory containing tokenized .pt files
            style_mapping: Dict mapping style names to numeric IDs (e.g., {"j_dilla": 0})
            max_length: Maximum sequence length (sequences truncated/padded to this)
            pad_token_id: Token ID for padding (default: 0)
            bos_token_id: Token ID for beginning of sequence (default: 1)
            eos_token_id: Token ID for end of sequence (default: 2)
            validate_data: If True, validate all token files on init
        """
        self.tokens_dir = Path(tokens_dir)
        self.style_mapping = style_mapping
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        if not self.tokens_dir.exists():
            raise FileNotFoundError(f"Tokens directory not found: {self.tokens_dir}")

        # Load all token files
        self.token_files = sorted(list(self.tokens_dir.glob("*.pt")))

        if len(self.token_files) == 0:
            raise ValueError(f"No .pt files found in {self.tokens_dir}")

        logger.info(f"Found {len(self.token_files)} tokenized files")

        # Validate data if requested
        if validate_data:
            self._validate_data()

    def _validate_data(self):
        """Validate all token files can be loaded and have correct format."""
        logger.info("Validating dataset...")

        invalid_files = []
        for token_file in self.token_files:
            try:
                # Try to load
                tokens = torch.load(token_file)

                # Check it's a tensor
                if not isinstance(tokens, torch.Tensor):
                    invalid_files.append((token_file, "Not a tensor"))
                    continue

                # Check it's 1D
                if tokens.dim() != 1:
                    invalid_files.append((token_file, f"Wrong dimensions: {tokens.dim()}"))
                    continue

                # Check length
                if len(tokens) == 0:
                    invalid_files.append((token_file, "Empty sequence"))
                    continue

                # Check style can be extracted
                style_name = self._extract_style_from_filename(token_file.stem)
                if style_name not in self.style_mapping:
                    invalid_files.append((token_file, f"Unknown style: {style_name}"))
                    continue

            except Exception as e:
                invalid_files.append((token_file, str(e)))

        if invalid_files:
            logger.warning(f"Found {len(invalid_files)} invalid files:")
            for file, reason in invalid_files[:5]:  # Show first 5
                logger.warning(f"  {file.name}: {reason}")

            # Remove invalid files from dataset
            self.token_files = [
                f for f in self.token_files
                if not any(f == invalid[0] for invalid in invalid_files)
            ]

            logger.info(f"Remaining valid files: {len(self.token_files)}")
        else:
            logger.info("All files validated successfully")

    def _extract_style_from_filename(self, filename: str) -> str:
        """
        Extract style name from filename.

        Expected format: "{style_name}_{number}.pt"
        Example: "j_dilla_001.pt" -> "j_dilla"

        Args:
            filename: File stem (without extension)

        Returns:
            Style name string
        """
        # Split by underscore and take all but last part (which is the number)
        parts = filename.split('_')

        # Handle various naming conventions:
        # "j_dilla_001" -> "j_dilla"
        # "metro_boomin_001" -> "metro_boomin"
        # "questlove_001" -> "questlove"

        # Try removing last part if it's a number
        if parts[-1].isdigit():
            style_name = '_'.join(parts[:-1])
        else:
            style_name = '_'.join(parts)

        return style_name.lower()

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.token_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of sample to retrieve

        Returns:
            Dict with keys:
                - input_ids: Input token sequence (max_length-1,)
                - labels: Target token sequence (max_length-1,)
                - style_ids: Style ID tensor (scalar)
                - attention_mask: Attention mask (max_length-1,)
        """
        # Load tokens
        token_path = self.token_files[idx]
        tokens = torch.load(token_path)

        # Ensure tokens are LongTensor
        if tokens.dtype != torch.long:
            tokens = tokens.long()

        # Extract style from filename
        style_name = self._extract_style_from_filename(token_path.stem)
        style_id = self.style_mapping.get(style_name, 0)  # Default to 0 if not found

        # Add BOS and EOS tokens if not already present
        if tokens[0] != self.bos_token_id:
            tokens = torch.cat([torch.tensor([self.bos_token_id]), tokens])
        if tokens[-1] != self.eos_token_id:
            tokens = torch.cat([tokens, torch.tensor([self.eos_token_id])])

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            padding_length = self.max_length - len(tokens)
            padding = torch.full(
                (padding_length,),
                self.pad_token_id,
                dtype=torch.long
            )
            tokens = torch.cat([tokens, padding])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (tokens != self.pad_token_id).long()

        # For autoregressive training:
        # input_ids = tokens[:-1]  (all but last)
        # labels = tokens[1:]      (all but first)
        # This creates the "predict next token" training setup
        input_ids = tokens[:-1]
        labels = tokens[1:]
        attention_mask = attention_mask[:-1]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'style_ids': torch.tensor(style_id, dtype=torch.long),
            'attention_mask': attention_mask,
        }

    def get_style_distribution(self) -> Dict[str, int]:
        """
        Get distribution of styles in dataset.

        Returns:
            Dict mapping style names to counts
        """
        style_counts = {}

        for token_file in self.token_files:
            style_name = self._extract_style_from_filename(token_file.stem)
            style_counts[style_name] = style_counts.get(style_name, 0) + 1

        return style_counts

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict with statistics about the dataset
        """
        total_tokens = 0
        sequence_lengths = []

        for token_file in self.token_files[:100]:  # Sample first 100 for speed
            tokens = torch.load(token_file)
            sequence_lengths.append(len(tokens))
            total_tokens += len(tokens)

        return {
            'num_samples': len(self.token_files),
            'num_styles': len(set(self.style_mapping.values())),
            'avg_sequence_length': sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
            'max_sequence_length': max(sequence_lengths) if sequence_lengths else 0,
            'min_sequence_length': min(sequence_lengths) if sequence_lengths else 0,
            'style_distribution': self.get_style_distribution(),
        }


def create_dataloaders(
    tokens_dir: str | Path,
    style_mapping: Dict[str, int],
    batch_size: int = 32,
    train_split: float = 0.9,
    max_length: int = 512,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        tokens_dir: Directory with tokenized files
        style_mapping: Dict mapping style names to IDs
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest is validation)
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle training data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> style_mapping = {"j_dilla": 0, "metro_boomin": 1, "questlove": 2}
        >>> train_loader, val_loader = create_dataloaders(
        ...     tokens_dir="data/tokenized",
        ...     style_mapping=style_mapping,
        ...     batch_size=32
        ... )
    """
    # Create full dataset
    dataset = DrumPatternDataset(
        tokens_dir=tokens_dir,
        style_mapping=style_mapping,
        max_length=max_length,
    )

    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    logger.info(f"Dataset split: {train_size} train, {val_size} validation")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
