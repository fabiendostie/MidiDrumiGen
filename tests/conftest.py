"""Pytest configuration and shared fixtures."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Add project root to path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_checkpoint_path(temp_dir):
    """Create a mock model checkpoint file."""
    checkpoint_path = temp_dir / "test_model.pt"

    # Create a simple checkpoint
    checkpoint = {
        "model_state_dict": {},
        "metadata": {
            "vocab_size": 500,
            "n_styles": 50,
            "n_positions": 2048,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "dropout": 0.1,
        },
        "epoch": 0,
        "train_loss": 0.5,
        "val_loss": 0.4,
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_style_params():
    """Sample producer style parameters."""
    return {
        "model_id": "test_style_v1",
        "model_path": "models/checkpoints/test_style_v1.pt",
        "description": "Test style for unit testing",
        "preferred_tempo_range": (90, 120),
        "humanization": {
            "swing": 55.0,
            "micro_timing_ms": 10.0,
            "ghost_note_prob": 0.3,
            "velocity_variation": 0.12,
        },
    }


# v1.x fixtures removed in v2.0 migration (PyTorch-based testing)
# @pytest.fixture
# def mock_model():
#     """Create a mock model instance."""
#     from src.inference.mock import MockDrumModel
#     return MockDrumModel(vocab_size=500, n_styles=50)
#
# @pytest.fixture(autouse=True)
# def reset_lru_cache():
#     """Reset LRU cache before each test."""
#     from src.inference.model_loader import load_model
#     yield
#     load_model.cache_clear()
