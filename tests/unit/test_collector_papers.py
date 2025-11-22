"""
Unit Tests for Scholar Paper Collector

Story: E1.S1 - Scholar Paper Collection
Test Coverage: Tempo extraction, API parsing, confidence calculation
"""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import ClientResponseError

from src.research.collectors.base import ResearchSource
from src.research.collectors.papers import ScholarPaperCollector

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def collector():
    """Create ScholarPaperCollector instance for testing."""
    return ScholarPaperCollector(timeout=60, min_papers=3)


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
        confidence = collector._calculate_paper_confidence({})
        assert confidence == 0.5

    def test_citation_count_bonus(self, collector):
        """Should add bonus for citation count."""
        confidence = collector._calculate_paper_confidence({"citationCount": 100})
        assert confidence > 0.5
        assert confidence <= 1.0

    def test_recency_bonus(self, collector):
        """Should add bonus for recent papers."""
        confidence_2022 = collector._calculate_paper_confidence({"year": 2022})
        confidence_2018 = collector._calculate_paper_confidence({"year": 2018})
        confidence_2010 = collector._calculate_paper_confidence({"year": 2010})
        assert confidence_2022 > confidence_2018 > confidence_2010

    def test_relevance_bonus(self, collector):
        """Should add bonus for tempo mentions."""
        confidence = collector._calculate_paper_confidence({"abstract": "tempo of 120 BPM"})
        assert confidence > 0.5

    def test_confidence_capped_at_one(self, collector):
        """Should cap confidence at 1.0."""
        paper = {
            "citationCount": 10000,
            "year": 2023,
            "abstract": "120 BPM 130 BPM 140 BPM 150 BPM",
        }
        confidence = collector._calculate_paper_confidence(paper)
        assert confidence == 1.0


# =============================================================================
# Test API Integrations
# =============================================================================


@pytest.mark.asyncio
class TestApiSearches:
    """Test API integration methods."""

    @patch("aiohttp.ClientSession.get")
    async def test_semantic_scholar_success(
        self, mock_get, collector, mock_semantic_scholar_response
    ):
        """Should successfully parse Semantic Scholar response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_semantic_scholar_response

        async def __aenter__(self):
            return mock_response

        mock_response.__aenter__ = __aenter__
        mock_get.return_value = mock_response

        results = await collector._search_semantic_scholar("John Bonham")

        assert len(results) == 2
        assert results[0].title == "Rhythmic Analysis of John Bonham Drumming Style"
        assert 120 in results[0].extracted_data["tempo_mentions"]

    @patch("aiohttp.ClientSession.get")
    async def test_arxiv_success(self, mock_get, collector, mock_arxiv_response):
        """Should successfully parse arXiv XML response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = mock_arxiv_response

        async def __aenter__(self):
            return mock_response

        mock_response.__aenter__ = __aenter__
        mock_get.return_value = mock_response

        results = await collector._search_arxiv("Test Artist")
        assert len(results) == 1
        assert results[0].title == "Drum Pattern Analysis Using Deep Learning"
        assert 95 in results[0].extracted_data["tempo_mentions"]

    @patch("aiohttp.ClientSession.get")
    async def test_crossref_success(self, mock_get, collector, mock_crossref_response):
        """Should successfully parse CrossRef JSON response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_crossref_response

        async def __aenter__(self):
            return mock_response

        mock_response.__aenter__ = __aenter__
        mock_get.return_value = mock_response

        results = await collector._search_crossref("Test Artist")
        assert len(results) == 1
        assert results[0].title == "A Study of Rock Drumming"
        assert 140 in results[0].extracted_data["tempo_mentions"]

    @patch("aiohttp.ClientSession.get")
    async def test_api_error_graceful_handling(self, mock_get, collector):
        """Should handle API errors gracefully and return an empty list."""
        mock_get.side_effect = ClientResponseError(
            history=(), request_info=AsyncMock(), status=500, message="Server Error"
        )

        results = await collector._search_semantic_scholar("Error Artist")
        assert results == []

    @patch("aiohttp.ClientSession.get")
    async def test_search_unexpected_exception(self, mock_get, collector):
        """Should handle unexpected exceptions gracefully and return an empty list."""
        mock_get.side_effect = Exception("Unexpected network issue")

        results = await collector._search_semantic_scholar("Exceptional Artist")
        assert results == []
        results = await collector._search_arxiv("Exceptional Artist")
        assert results == []
        results = await collector._search_crossref("Exceptional Artist")
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
            patch.object(collector, "_search_semantic_scholar", new_callable=AsyncMock) as mock_ss,
            patch.object(collector, "_search_arxiv", new_callable=AsyncMock) as mock_arxiv,
            patch.object(collector, "_search_crossref", new_callable=AsyncMock) as mock_cr,
        ):
            mock_ss.return_value = [
                ResearchSource(
                    source_type="paper",
                    title="SS Paper",
                    confidence=0.8,
                    raw_content="",
                    extracted_data={},
                    metadata={},
                )
            ]
            mock_arxiv.return_value = [
                ResearchSource(
                    source_type="paper",
                    title="ArXiv Paper",
                    confidence=0.7,
                    raw_content="",
                    extracted_data={},
                    metadata={},
                )
            ]
            mock_cr.return_value = []

            results = await collector.collect("John Bonham")
            assert len(results) == 2
            assert all(isinstance(r, ResearchSource) for r in results)

    async def test_collect_handles_exceptions(self, collector):
        """Should continue collection if one source fails."""
        with (
            patch.object(collector, "_search_semantic_scholar", new_callable=AsyncMock) as mock_ss,
            patch.object(collector, "_search_arxiv", new_callable=AsyncMock) as mock_arxiv,
            patch.object(collector, "_search_crossref", new_callable=AsyncMock) as mock_cr,
        ):
            mock_ss.side_effect = Exception("Semantic Scholar API down")
            mock_arxiv.return_value = [
                ResearchSource(
                    source_type="paper",
                    title="ArXiv Paper",
                    confidence=0.7,
                    raw_content="",
                    extracted_data={},
                    metadata={},
                )
            ]
            mock_cr.return_value = []

            results = await collector.collect("Test Artist")
            assert len(results) == 1
            assert results[0].title == "ArXiv Paper"

    async def test_collect_warns_few_papers(self, collector, caplog):
        """Should warn when finding fewer than min_papers."""
        with (
            patch.object(collector, "_search_semantic_scholar", new_callable=AsyncMock) as mock_ss,
            patch.object(collector, "_search_arxiv", new_callable=AsyncMock) as mock_arxiv,
            patch.object(collector, "_search_crossref", new_callable=AsyncMock) as mock_cr,
        ):
            mock_ss.return_value = [
                ResearchSource(
                    source_type="paper",
                    title="Test",
                    confidence=0.5,
                    raw_content="",
                    extracted_data={},
                    metadata={},
                )
            ]
            mock_arxiv.return_value = []
            mock_cr.return_value = []

            collector.min_papers = 2
            results = await collector.collect("Obscure Artist")
            assert len(results) < collector.min_papers
            assert "Only found" in caplog.text


# =============================================================================
# Test Rate Limiting
# =============================================================================


@pytest.mark.asyncio
class TestRateLimiting:
    """Test the rate limiting logic."""

    async def test_wait_for_rate_limit_semantic_scholar(self, collector):
        """Should wait if rate limit is exceeded."""
        collector._SS_RATE_LIMIT = 2
        collector._SS_RATE_WINDOW = 10

        with (
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch("time.time", side_effect=[100, 100.1, 101, 101.1, 102, 102.1, 108, 108.1]),
        ):
            await collector._wait_for_rate_limit_semantic_scholar()  # call 1, time=100
            await collector._wait_for_rate_limit_semantic_scholar()  # call 2, time=101
            await collector._wait_for_rate_limit_semantic_scholar()  # call 3, time=102, should wait

            mock_sleep.assert_called_once()
            # The third call at t=102 should wait for the first call at t=100 to expire.
            # Window is 10s. Expiry is 100 + 10 = 110.
            # Wait time = 110 - 102 = 8
            assert mock_sleep.call_args[0][0] == pytest.approx(8.1)


# =============================================================================
# Integration Tests (Slow - Marked)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
class TestRealAPIIntegration:
    """Integration tests with real APIs (requires network)."""

    async def test_real_semantic_scholar_search(self, collector):
        """Test real Semantic Scholar API call."""
        results = await collector._search_semantic_scholar("John Bonham")
        assert isinstance(results, list)
        # We can't assert a specific number, but we expect some results
        if results:
            assert isinstance(results[0], ResearchSource)

    async def test_real_arxiv_search(self, collector):
        """Test real arXiv API call."""
        results = await collector._search_arxiv("drumming analysis")
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], ResearchSource)

    async def test_real_crossref_search(self, collector):
        """Test real CrossRef API call."""
        results = await collector._search_crossref("drumming rhythm")
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], ResearchSource)
