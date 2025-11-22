"""
Research Collectors Package for MidiDrumiGen v2.0

This package contains all research collectors for gathering data
about artist drumming styles from various sources.

Collectors:
    - ScholarPaperCollector: Academic papers from Semantic Scholar, arXiv, CrossRef
    - WebArticleCollector: Web articles (E1.S2)
    - AudioAnalysisCollector: Audio analysis (E1.S3)
    - MidiDatabaseCollector: MIDI databases (E1.S4)
"""

from .articles import WebArticleCollector
from .base import (
    BaseCollector,
    CollectorError,
    CollectorTimeoutError,
    InsufficientDataError,
    ResearchSource,
)
from .papers import ScholarPaperCollector

__all__ = [
    "BaseCollector",
    "CollectorError",
    "CollectorTimeoutError",
    "InsufficientDataError",
    "ResearchSource",
    "ScholarPaperCollector",
    "WebArticleCollector",
]
