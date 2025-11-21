"""
Base Collector Class for MidiDrumiGen v2.0 Research Pipeline

This module provides the abstract base class for all research collectors.
Each collector type (papers, articles, audio, MIDI) inherits from BaseCollector.

Part of: Epic 1 - Research Pipeline
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResearchSource:
    """
    Data class representing a single research source.

    Attributes:
        source_type: Type of source (paper, article, audio, midi)
        title: Title or name of the source
        url: URL to the source (if applicable)
        file_path: Local file path (for downloaded content)
        raw_content: Raw text content from source
        extracted_data: Structured data extracted from source
        confidence: Confidence score (0.0-1.0) of source quality
        collected_at: Timestamp when source was collected
        metadata: Additional metadata about the source
    """

    source_type: str  # paper, article, audio, midi
    title: str
    url: str | None = None
    file_path: str | None = None
    raw_content: str | None = None
    extracted_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    collected_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate confidence score is between 0 and 1."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


class BaseCollector(ABC):
    """
    Abstract base class for all research collectors.

    All collectors must implement the collect() method to gather
    research data for a given artist name.

    Subclasses:
        - ScholarPaperCollector (E1.S1)
        - WebArticleCollector (E1.S2)
        - AudioAnalysisCollector (E1.S3)
        - MidiDatabaseCollector (E1.S4)
    """

    def __init__(self, timeout: int = 300):
        """
        Initialize base collector.

        Args:
            timeout: Maximum time in seconds for collection (default: 5 minutes)
        """
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def collect(self, artist_name: str) -> list[ResearchSource]:
        """
        Collect research sources for the given artist.

        This is the main method that must be implemented by all collectors.

        Args:
            artist_name: Name of the artist to research

        Returns:
            List of ResearchSource objects containing collected data

        Raises:
            CollectorError: If collection fails critically
            TimeoutError: If collection exceeds timeout
        """
        pass

    def _calculate_confidence(
        self,
        citation_count: int | None = None,
        relevance_score: float | None = None,
        source_quality: float | None = None,
    ) -> float:
        """
        Calculate confidence score for a research source.

        Args:
            citation_count: Number of citations (for papers)
            relevance_score: How relevant the source is to drumming (0-1)
            source_quality: Quality of the source (0-1)

        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.5

        # Citation count bonus (papers)
        if citation_count is not None:
            citation_bonus = min(citation_count / 100, 0.3)  # Max 0.3 bonus
            base_confidence += citation_bonus

        # Relevance score bonus
        if relevance_score is not None:
            base_confidence += relevance_score * 0.2

        # Source quality bonus
        if source_quality is not None:
            base_confidence += source_quality * 0.2

        return min(base_confidence, 1.0)

    async def _retry_with_backoff(self, func, max_retries: int = 3, initial_delay: float = 1.0):
        """
        Retry a function with exponential backoff.

        Args:
            func: Async function to retry
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds

        Returns:
            Result from successful function call

        Raises:
            Exception: The last exception if all retries fail
        """
        import asyncio

        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff

        raise last_exception


class CollectorError(Exception):
    """Base exception for collector errors."""

    pass


class CollectorTimeoutError(CollectorError):
    """Raised when collector exceeds timeout."""

    pass


class InsufficientDataError(CollectorError):
    """Raised when collector cannot find enough data."""

    pass
