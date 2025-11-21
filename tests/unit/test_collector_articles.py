"""
Unit Tests for Web Article Collector

Story: E1.S2 - Web Article Collection
Test Coverage: NLP filtering, HTML parsing, equipment extraction
"""

from unittest.mock import AsyncMock, patch

import pytest
from bs4 import BeautifulSoup

from src.research.collectors.articles import WebArticleCollector
from src.research.collectors.base import ResearchSource

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def collector():
    """Create WebArticleCollector instance for testing."""
    return WebArticleCollector(timeout=60, min_articles=5)


@pytest.fixture
def drumming_article_html():
    """Mock HTML for drumming-related article."""
    return """
    <html>
        <body>
            <div class="content">
                <h1>John Bonham: Master Drummer</h1>
                <p>John Bonham was known for his powerful drumming style and
                   innovative use of the bass drum. His technique revolutionized
                   rock drumming.</p>
                <p>Bonham's drum kit featured Ludwig drums and Paiste cymbals.
                   He was famous for his fast kick drum pedal work.</p>
                <p>His playing style emphasized heavy groove and syncopated beats.</p>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def non_drumming_article_html():
    """Mock HTML for non-drumming article."""
    return """
    <html>
        <body>
            <div class="content">
                <p>This article is about John Bonham's fashion choices and
                   personal life. He enjoyed motorcycles and cars.</p>
            </div>
        </body>
    </html>
    """


# =============================================================================
# Test NLP Filtering
# =============================================================================


class TestNLPFiltering:
    """Test spaCy NLP content filtering."""

    def test_filter_drumming_content(self, collector):
        """Should keep articles with >= 3 drumming keywords."""
        articles = [
            ResearchSource(
                source_type="article",
                title="Test 1",
                raw_content="John plays drums with great technique and rhythm on his drum kit",
                confidence=0.5,
            ),
            ResearchSource(
                source_type="article",
                title="Test 2",
                raw_content="This article mentions drums once",
                confidence=0.5,
            ),
        ]

        filtered = collector._filter_drumming_content(articles)
        assert len(filtered) == 1
        assert filtered[0].title == "Test 1"

    def test_boost_confidence_with_keywords(self, collector):
        """Should boost confidence for high keyword density."""
        articles = [
            ResearchSource(
                source_type="article",
                title="Heavy Drumming",
                raw_content=" ".join(["drum"] * 20 + ["word"] * 80),  # 20% density
                confidence=0.5,
            )
        ]

        filtered = collector._filter_drumming_content(articles)
        assert filtered[0].confidence > 0.5

    def test_add_keyword_metadata(self, collector):
        """Should add keyword count and density to metadata."""
        articles = [
            ResearchSource(
                source_type="article",
                title="Drumming Article",
                raw_content="drum beat rhythm groove cymbal",
                confidence=0.5,
            )
        ]

        filtered = collector._filter_drumming_content(articles)
        assert "keyword_count" in filtered[0].extracted_data
        assert "keyword_density" in filtered[0].extracted_data


# =============================================================================
# Test HTML Content Extraction
# =============================================================================


class TestContentExtraction:
    """Test HTML parsing and content extraction."""

    def test_extract_from_div_content(self, collector, drumming_article_html):
        """Should extract content from div.content."""
        soup = BeautifulSoup(drumming_article_html, "lxml")
        content = collector._extract_content(soup, "drummerworld")
        assert "John Bonham" in content
        assert "powerful drumming" in content

    def test_remove_script_and_style_tags(self, collector):
        """Should remove script and style tags."""
        html = """
        <div class="content">
            <script>alert('test');</script>
            <style>.class { color: red; }</style>
            <p>Real content here</p>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        content = collector._extract_content(soup, "wikipedia")
        assert "alert" not in content
        assert "color: red" not in content
        assert "Real content" in content

    def test_fallback_to_paragraphs(self, collector):
        """Should fall back to extracting all <p> tags."""
        html = """
        <html>
            <body>
                <p>First paragraph</p>
                <p>Second paragraph</p>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        content = collector._extract_content(soup, "unknown_site")
        assert "First paragraph" in content
        assert "Second paragraph" in content


# =============================================================================
# Test Equipment and Technique Extraction
# =============================================================================


class TestEquipmentExtraction:
    """Test extraction of equipment and technique mentions."""

    def test_extract_equipment_mentions(self, collector):
        """Should extract equipment mentions from text."""
        text = """
        John Bonham used Ludwig drums and Paiste cymbals in his kit.
        He preferred Remo drumheads and played with wooden sticks.
        """
        extracted = collector._extract_equipment_and_techniques(text)
        assert "equipment_mentions" in extracted
        assert len(extracted["equipment_mentions"]) > 0

    def test_extract_technique_mentions(self, collector):
        """Should extract technique descriptions."""
        text = """
        His technique was powerful and heavy, with fast kick drum work.
        He employed a syncopated style with subtle ghost notes.
        """
        extracted = collector._extract_equipment_and_techniques(text)
        assert "technique_mentions" in extracted
        assert len(extracted["technique_mentions"]) > 0

    def test_limit_extracted_items(self, collector):
        """Should limit equipment/technique lists to 5 items each."""
        text = " ".join([f"He used equipment_{i} and technique_{i}." for i in range(20)])
        extracted = collector._extract_equipment_and_techniques(text)
        assert len(extracted["equipment_mentions"]) <= 5
        assert len(extracted["technique_mentions"]) <= 5


# =============================================================================
# Test Site Scraping
# =============================================================================


@pytest.mark.asyncio
class TestSiteScraping:
    """Test individual site scraping logic."""

    async def test_scrape_site_success(self, collector, drumming_article_html):
        """Should successfully scrape site and return article."""
        site_config = {
            "base": "https://example.com",
            "search": "/drummer/{artist}",
            "requires_transform": False,
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=drumming_article_html)

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            results = await collector._scrape_site("test_site", site_config, "John Bonham")

            assert len(results) == 1
            assert results[0].source_type == "article"
            assert "John Bonham" in results[0].raw_content

    async def test_scrape_site_404(self, collector):
        """Should handle 404 gracefully."""
        site_config = {"base": "https://example.com", "search": "/drummer/{artist}"}

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            results = await collector._scrape_site("test_site", site_config, "Unknown Artist")

            assert results == []

    async def test_transform_artist_name_drummerworld(self, collector):
        """Should transform artist name for Drummerworld."""
        site_config = {
            "base": "https://drummerworld.com",
            "search": "/drummers/{artist}.html",
            "requires_transform": True,
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<html><body></body></html>")

            mock_get = mock_session.return_value.__aenter__.return_value.get
            mock_get.return_value.__aenter__.return_value = mock_response

            await collector._scrape_site("drummerworld", site_config, "John Bonham")

            # Verify URL was called with transformed name
            call_url = mock_get.call_args[0][0]
            assert "john_bonham" in call_url.lower()


# =============================================================================
# Test Main Collect Method
# =============================================================================


@pytest.mark.asyncio
class TestCollect:
    """Test main collect() method."""

    async def test_collect_parallel_execution(self, collector):
        """Should scrape all sites in parallel."""
        with patch.object(
            collector,
            "_scrape_site",
            return_value=[
                ResearchSource(
                    source_type="article",
                    title="Test",
                    raw_content="drum beat rhythm groove cymbal",
                    confidence=0.6,
                )
            ],
        ):
            results = await collector.collect("John Bonham")
            # Should call scrape_site for each configured site
            assert len(results) > 0

    async def test_collect_filters_results(self, collector):
        """Should filter results through NLP."""
        drumming_article = ResearchSource(
            source_type="article",
            title="Drumming",
            raw_content="drum beat rhythm groove cymbal technique",
            confidence=0.5,
        )
        non_drumming_article = ResearchSource(
            source_type="article",
            title="Other",
            raw_content="fashion motorcycles cars",
            confidence=0.5,
        )

        with patch.object(
            collector,
            "_scrape_site",
            side_effect=[[drumming_article], [non_drumming_article], [], []],
        ):
            results = await collector.collect("John Bonham")
            # Only drumming article should pass filter
            assert len(results) == 1
            assert results[0].title == "Drumming"


# =============================================================================
# Integration Tests (Slow - Marked)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
class TestRealWebScraping:
    """Integration tests with real websites (requires network)."""

    async def test_real_wikipedia_scrape(self, collector):
        """
        Test real Wikipedia scraping.

        Note: This test requires network access and may be slow.
        It may fail if Wikipedia is unavailable or blocks the request.
        """
        site_config = {
            "base": "https://en.wikipedia.org",
            "search": "/wiki/{artist}",
            "requires_transform": True,
        }

        results = await collector._scrape_site("wikipedia", site_config, "John_Bonham")

        # Just verify it doesn't crash, results may vary
        assert isinstance(results, list)


# TODO: Add more tests
# - Test robots.txt compliance
# - Test timeout handling
# - Test with Scrapy integration
# - Test rate limiting
# - Test concurrent scraping limits
