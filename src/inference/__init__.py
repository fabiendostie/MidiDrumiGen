"""Inference module for pattern generation.

Note: model_loader.py and mock.py have been archived in v2.0 migration.
v2.0 uses LLM-based generation instead of PyTorch models.
"""

from .generate import generate_pattern

__all__ = [
    "generate_pattern",
]
