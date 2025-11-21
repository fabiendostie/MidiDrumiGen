"""
Unit Tests for Scholar Paper Collector

Story: E1.S1 - Scholar Paper Collection
Test Coverage: Tempo extraction, API parsing, confidence calculation
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.research.collectors.base import ResearchSource
from src.research.collectors.papers import ScholarPaperCollector

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def collector():
    """Create ScholarPaperCollector instance for testing."""
    return ScholarPaperCollector(timeout=60, min_papers=3)


@pytest.fixture
def mock_semantic_scholar_response():
    """Mock response from Semantic Scholar API."""
    return {
        "data": [
            {
                "title": "Rhythmic Analysis of John Bonham Drumming Style",
                "abstract": "Study of John Bonham typical tempo of 120 BPM with powerful groove",
                "url": "https://example.com/paper1",
                "authors": [{"name": "Smith, J."}, {"name": "Jones, A."}],
                "citationCount": 45,
                "year": 2020,
            },
            {
                "title": "Heavy Rock Drumming Techniques",
                "abstract": "Analysis of drumming at 110-140 beats per minute",
                "url": "https://example.com/paper2",
                "authors": [{"name": "Doe, J."}],
                "citationCount": 12,
                "year": 2019,
            },
        ]
    }


# =============================================================================
# Test Tempo Extraction
# =============================================================================


class TestTempoExtraction:
    """Test tempo mention extraction from text."""

    def test_extract_single_bpm(self, collector):
        """Should extract single BPM mention."""
        text = "The drummer plays at 120 BPM"
        tempos = collector._extract_tempo_mentions(text)
        assert 120 in tempos

    def test_extract_multiple_bpm(self, collector):
        """Should extract multiple BPM mentions."""
        text = "Songs range from 90 BPM to 140 BPM"
        tempos = collector._extract_tempo_mentions(text)
        assert 90 in tempos
        assert 140 in tempos

    def test_extract_bpm_range(self, collector):
        """Should extract both values from BPM range."""
        text = "Typical tempo range of 110-130 bpm"
        tempos = collector._extract_tempo_mentions(text)
        assert 110 in tempos
        assert 130 in tempos

    def test_extract_beats_per_minute(self, collector):
        """Should extract 'beats per minute' format."""
        text = "Playing at 95 beats per minute"
        tempos = collector._extract_tempo_mentions(text)
        assert 95 in tempos

    def test_filter_unrealistic_tempos(self, collector):
        """Should filter out tempos outside 20-300 BPM range."""
        text = "Values: 5 BPM, 120 BPM, 500 BPM"
        tempos = collector._extract_tempo_mentions(text)
        assert 5 not in tempos
        assert 120 in tempos
        assert 500 not in tempos

    def test_remove_duplicates(self, collector):
        """Should remove duplicate tempo values."""
        text = "120 BPM and 120 beats per minute"
        tempos = collector._extract_tempo_mentions(text)
        assert tempos.count(120) == 1

    def test_empty_text(self, collector):
        """Should return empty list for empty text."""
        assert collector._extract_tempo_mentions("") == []
        assert collector._extract_tempo_mentions(None) == []


# =============================================================================
# Test Confidence Calculation
# =============================================================================


class TestConfidenceCalculation:
    """Test confidence score calculation logic."""

    def test_base_confidence(self, collector):
        """Should return base confidence when no factors provided."""
        confidence = collector._calculate_confidence()
        assert confidence == 0.5

    def test_citation_count_bonus(self, collector):
        """Should add bonus for citation count."""
        confidence = collector._calculate_confidence(citation_count=100)
        assert confidence > 0.5
        assert confidence <= 1.0

    def test_relevance_score_bonus(self, collector):
        """Should add bonus for relevance score."""
        confidence = collector._calculate_confidence(relevance_score=0.8)
        assert confidence > 0.5

    def test_source_quality_bonus(self, collector):
        """Should add bonus for source quality."""
        confidence = collector._calculate_confidence(source_quality=0.9)
        assert confidence > 0.5

    def test_confidence_capped_at_one(self, collector):
        """Should cap confidence at 1.0."""
        confidence = collector._calculate_confidence(
            citation_count=1000, relevance_score=1.0, source_quality=1.0
        )
        assert confidence == 1.0


# =============================================================================
# Test Semantic Scholar Integration
# =============================================================================


@pytest.mark.asyncio
class TestSemanticScholarSearch:
    """Test Semantic Scholar API integration."""

    async def test_search_success(self, collector, mock_semantic_scholar_response):
        """Should successfully search Semantic Scholar."""
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_semantic_scholar_response)

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            results = await collector._search_semantic_scholar("John Bonham")

            assert len(results) == 2
            assert all(isinstance(r, ResearchSource) for r in results)
            assert results[0].source_type == "paper"
            assert results[0].title == "Rhythmic Analysis of John Bonham Drumming Style"
            assert 120 in results[0].extracted_data["tempo_mentions"]

    async def test_search_404(self, collector):
        """Should handle 404 gracefully."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            results = await collector._search_semantic_scholar("Unknown Artist")
            assert results == []

    async def test_search_api_error(self, collector):
        """Should handle API errors gracefully."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = Exception(
                "API Error"
            )

            results = await collector._search_semantic_scholar("John Bonham")
            assert results == []


# =============================================================================
# Test Main Collect Method
# =============================================================================


@pytest.mark.asyncio
class TestCollect:
    """Test main collect() method."""

    async def test_collect_success(self, collector):
        """Should successfully collect papers from all sources."""
        with (
            patch.object(
                collector,
                "_search_semantic_scholar",
                return_value=[
                    ResearchSource(source_type="paper", title="Test Paper", confidence=0.8)
                ],
            ),
            patch.object(collector, "_search_arxiv", return_value=[]),
            patch.object(collector, "_search_crossref", return_value=[]),
        ):
            results = await collector.collect("John Bonham")
            assert len(results) >= 1
            assert all(r.source_type == "paper" for r in results)

    async def test_collect_warns_few_papers(self, collector, caplog):
        """Should warn when finding fewer than min_papers."""
        with (
            patch.object(
                collector,
                "_search_semantic_scholar",
                return_value=[ResearchSource(source_type="paper", title="Test", confidence=0.5)],
            ),
            patch.object(collector, "_search_arxiv", return_value=[]),
            patch.object(collector, "_search_crossref", return_value=[]),
        ):
            results = await collector.collect("Obscure Artist")
            assert len(results) < collector.min_papers
            assert "Only found" in caplog.text


# =============================================================================
# Integration Tests (Slow - Marked)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
class TestRealAPIIntegration:
    """Integration tests with real APIs (requires network)."""

    async def test_real_semantic_scholar_search(self, collector):
        """
        Test real Semantic Scholar API call.

        Note: This test requires network access and may be slow.
        It may also fail if the API is unavailable or rate-limited.
        """
        results = await collector._search_semantic_scholar("John Bonham")
        # Just verify it doesn't crash, results may vary
        assert isinstance(results, list)


# TODO: Add more tests
# - Test arXiv XML parsing
# - Test CrossRef API integration
# - Test rate limiting with exponential backoff
# - Test timeout handling
# - Test parallel execution of all searches
