"""
Database module for MidiDrumiGen v2.0.

Provides database models, manager, and exceptions.
"""

from .manager import ArtistNotFoundError, DatabaseError, DatabaseManager
from .models import Artist, GenerationHistory, ResearchSource, StyleProfile

__all__ = [
    "DatabaseManager",
    "ArtistNotFoundError",
    "DatabaseError",
    "Artist",
    "StyleProfile",
    "ResearchSource",
    "GenerationHistory",
]
