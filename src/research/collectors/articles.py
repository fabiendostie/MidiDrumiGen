"""
Web Article Collector for MidiDrumiGen v2.0

Scrapes music journalism sites for artist style descriptions using
BeautifulSoup4, Scrapy, and spaCy NLP for content filtering.

Story: E1.S2 - Web Article Collection
Epic: E1 - Research Pipeline
Priority: HIGH
Story Points: 3
"""

import asyncio
from typing import Any
from urllib.parse import quote, urljoin
from urllib.robotparser import RobotFileParser

import aiohttp
import spacy
from bs4 import BeautifulSoup

from .base import BaseCollector, CollectorError, ResearchSource


class WebArticleCollector(BaseCollector):
    """
    Collects web articles about artist drumming styles.

    Sources:
        - Drummerworld
        - Wikipedia
        - Pitchfork
        - Rolling Stone
        - Sound On Sound
        - Music Connection
        - Music Business Worldwide
        - The Pro Audio Files
        - Produce Like a Pro
        - Songstuff
        - Songwriter Universe
        - Renegade Producer

    Acceptance Criteria:
        - Scrapes 4+ configured sites
        - Uses spaCy NLP to filter drumming-related content
        - Extracts equipment and technique mentions
        - Handles HTTP errors gracefully
        - Respects robots.txt
    """

    # Target websites with search patterns
    SITES = {
        "drummerworld": {
            "base": "https://www.drummerworld.com",
            "search": "/drummers/{artist}.html",
            "requires_transform": True,  # Need to convert name to lowercase
        },
        "wikipedia": {
            "base": "https://en.wikipedia.org",
            "search": "/wiki/{artist}",
            "requires_transform": True,  # Need to replace spaces with underscores
        },
        "pitchfork": {"base": "https://pitchfork.com", "search": "/search/?query={artist}"},
        "rolling_stone": {"base": "https://www.rollingstone.com", "search": "/search/{artist}"},
    }

    # Drumming-related keywords for NLP filtering
    DRUMMING_KEYWORDS = [
        "drum",
        "drummer",
        "drumming",
        "percussion",
        "percussionist",
        "beat",
        "beats",
        "rhythm",
        "rhythmic",
        "groove",
        "tempo",
        "kit",
        "drum kit",
        "cymbal",
        "cymbals",
        "snare",
        "kick",
        "hi-hat",
        "hi hat",
        "hihat",
        "tom",
        "toms",
        "ride",
        "crash",
        "splash",
        "china",
        "bell",
        "cowbell",
        "stick",
        "sticks",
        "drumstick",
        "brush",
        "mallet",
        "fill",
        "fills",
        "rudiment",
        "paradiddle",
        "flam",
        "timing",
        "syncopation",
        "polyrhythm",
        "shuffle",
        "technique",
        "playing style",
        "approach",
    ]

    # Request delay in seconds (polite rate limiting)
    REQUEST_DELAY = 2.0

    # User-Agent for polite scraping
    USER_AGENT = "MidiDrumiGen/2.0 (Research Bot; +https://github.com/fabiendostie/MidiDrumiGen)"

    # Source authority weights for confidence scoring
    SOURCE_WEIGHTS = {
        "drummerworld": 0.9,  # Specialized drummer resource
        "wikipedia": 0.8,  # Reliable, comprehensive
        "pitchfork": 0.6,  # Music journalism
        "rolling_stone": 0.6,  # Music journalism
    }

    def __init__(self, timeout: int = 300, min_articles: int = 5):
        """
        Initialize Web Article Collector.

        Args:
            timeout: Maximum time in seconds for collection (default: 5 min)
            min_articles: Minimum number of articles to collect
        """
        super().__init__(timeout)
        self.min_articles = min_articles
        self._robots_cache: dict[str, RobotFileParser] = {}
        self._last_request_time: dict[str, float] = {}

        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            raise

    async def collect(self, artist_name: str) -> list[ResearchSource]:
        """
        Collect web articles about the artist's drumming style.

        Args:
            artist_name: Name of the artist/drummer to research

        Returns:
            List of ResearchSource objects with source_type='article'

        Raises:
            CollectorError: If critical error occurs
        """
        self.logger.info(f"Starting article collection for artist: {artist_name}")

        articles = []

        try:
            # Scrape all sites in parallel
            tasks = [
                self._scrape_site(site_name, config, artist_name)
                for site_name, config in self.SITES.items()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results, filtering out exceptions
            for result in results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Site scraping failed: {result}")
                else:
                    articles.extend(result)

            # Filter articles using NLP
            articles = self._filter_drumming_content(articles)

            self.logger.info(
                f"Collected {len(articles)} drumming-related articles " f"for {artist_name}"
            )

            if len(articles) < self.min_articles:
                self.logger.warning(
                    f"Only found {len(articles)} articles, "
                    f"minimum {self.min_articles} recommended"
                )

            return articles

        except Exception as e:
            raise CollectorError(f"Article collection failed: {e}") from e

    async def _check_robots_txt(self, base_url: str, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.

        Args:
            base_url: Base URL of the site
            url: Full URL to check

        Returns:
            True if allowed, False if disallowed
        """
        if base_url in self._robots_cache:
            rp = self._robots_cache[base_url]
        else:
            try:
                rp = RobotFileParser()
                robots_url = urljoin(base_url, "/robots.txt")
                rp.set_url(robots_url)
                rp.read()
                self._robots_cache[base_url] = rp
            except Exception as e:
                self.logger.debug(f"Could not read robots.txt for {base_url}: {e}")
                return True  # Allow if we can't read robots.txt

        return rp.can_fetch(self.USER_AGENT, url)

    async def _enforce_rate_limit(self, site_name: str) -> None:
        """
        Enforce rate limiting with 2-second delay between requests to same site.

        Args:
            site_name: Name of the site
        """
        import time

        current_time = time.time()
        if site_name in self._last_request_time:
            elapsed = current_time - self._last_request_time[site_name]
            if elapsed < self.REQUEST_DELAY:
                await asyncio.sleep(self.REQUEST_DELAY - elapsed)

        self._last_request_time[site_name] = time.time()

    def _calculate_article_confidence(self, article: dict[str, Any]) -> float:
        """
        Calculate confidence score for an article based on Dev Notes formula.

        Args:
            article: Dictionary with article data

        Returns:
            Confidence score between 0 and 1
        """
        site_name = article.get("site", "unknown")
        base_confidence = self.SOURCE_WEIGHTS.get(site_name, 0.5)

        # Content length boost (more content = more data)
        word_count = article.get("word_count", 0)
        length_boost = min(0.15, word_count / 5000)

        # Drumming keyword density boost
        keyword_count = article.get("keyword_count", 0)
        keyword_boost = min(0.15, keyword_count * 0.02)

        # Equipment mentions boost
        equipment_count = len(article.get("equipment", []))
        equipment_boost = min(0.1, equipment_count * 0.03)

        return min(1.0, base_confidence + length_boost + keyword_boost + equipment_boost)

    async def _scrape_site(
        self, site_name: str, config: dict[str, Any], artist_name: str
    ) -> list[ResearchSource]:
        """
        Scrape a single website for articles about artist.

        Args:
            site_name: Name of the site (for logging)
            config: Site configuration dict with base URL and search pattern
            artist_name: Name of artist to search

        Returns:
            List of ResearchSource objects from this site
        """
        articles = []

        try:
            # Transform artist name if needed
            if config.get("requires_transform"):
                if site_name == "drummerworld":
                    search_artist = artist_name.lower().replace(" ", "_")
                elif site_name == "wikipedia":
                    search_artist = artist_name.replace(" ", "_")
                else:
                    search_artist = artist_name
            else:
                search_artist = quote(artist_name)

            # Build URL
            search_path = config["search"].format(artist=search_artist)
            url = urljoin(config["base"], search_path)

            # Check robots.txt
            if not await self._check_robots_txt(config["base"], url):
                self.logger.debug(f"{site_name}: URL disallowed by robots.txt")
                return articles

            # Enforce rate limiting
            await self._enforce_rate_limit(site_name)

            self.logger.debug(f"Scraping {site_name}: {url}")

            # Fetch page with retry logic
            async def fetch_page():
                headers = {"User-Agent": self.USER_AGENT}
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=30),
                        headers=headers,
                    ) as resp,
                ):
                    if resp.status == 404:
                        self.logger.debug(f"{site_name}: Artist not found (404)")
                        return None, 404

                    if resp.status == 429:
                        # Rate limited - raise to trigger retry
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=429,
                            message="Rate limited",
                        )

                    if resp.status in (500, 503):
                        # Server error - raise to trigger retry
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=f"Server error {resp.status}",
                        )

                    if resp.status != 200:
                        self.logger.warning(f"{site_name} returned {resp.status}")
                        return None, resp.status

                    html = await resp.text()
                    return html, resp.status

            # Use retry with exponential backoff for transient errors
            try:
                result = await self._retry_with_backoff(
                    fetch_page, max_retries=3, initial_delay=1.0
                )
                html, status = result
            except Exception as e:
                self.logger.warning(f"{site_name}: All retries failed: {e}")
                return articles

            if html is None:
                return articles

            # Parse HTML
            soup = BeautifulSoup(html, "lxml")

            # Extract text content based on site structure
            content = self._extract_content(soup, site_name)

            if content:
                # Extract equipment and techniques
                extracted = self._extract_equipment_and_techniques(content)
                word_count = len(content.split())

                # Calculate confidence using proper formula
                article_data = {
                    "site": site_name,
                    "word_count": word_count,
                    "keyword_count": 0,  # Will be set after NLP filtering
                    "equipment": extracted.get("equipment_mentions", []),
                }
                confidence = self._calculate_article_confidence(article_data)

                articles.append(
                    ResearchSource(
                        source_type="article",
                        title=f"{artist_name} - {site_name.replace('_', ' ').title()}",
                        url=url,
                        raw_content=content,
                        extracted_data=extracted,
                        confidence=confidence,
                        metadata={"site": site_name, "word_count": word_count},
                    )
                )

        except TimeoutError:
            self.logger.warning(f"{site_name}: Request timed out")
        except Exception as e:
            self.logger.error(f"{site_name} scraping failed: {e}")

        return articles

    def _extract_content(self, soup: BeautifulSoup, site_name: str) -> str:
        """
        Extract main text content from HTML based on site structure.

        Args:
            soup: BeautifulSoup parsed HTML
            site_name: Name of site (to apply site-specific extraction)

        Returns:
            Extracted text content
        """
        # Site-specific selectors
        selectors = {
            "drummerworld": ["div.content", "div.main"],
            "wikipedia": ["div.mw-parser-output"],
            "pitchfork": ["article", "div.review-detail"],
            "rolling_stone": ["article", "div.article-content"],
        }

        # Try site-specific selectors first
        if site_name in selectors:
            for selector in selectors[site_name]:
                element = soup.select_one(selector)
                if element:
                    # Remove script and style tags
                    for tag in element(["script", "style", "nav", "footer"]):
                        tag.decompose()
                    return element.get_text(strip=True, separator=" ")

        # Fallback: get all <p> tags
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text(strip=True) for p in paragraphs)

    def _filter_drumming_content(self, articles: list[ResearchSource]) -> list[ResearchSource]:
        """
        Filter articles to only include drumming-related content using spaCy NLP.

        Args:
            articles: List of articles to filter

        Returns:
            Filtered list of articles with >= 3 drumming keyword mentions
        """
        filtered = []

        for article in articles:
            if not article.raw_content:
                continue

            # Process with spaCy
            doc = self.nlp(article.raw_content.lower())

            # Count keyword mentions
            keyword_count = sum(
                1
                for token in doc
                if any(keyword in token.text for keyword in self.DRUMMING_KEYWORDS)
            )

            if keyword_count >= 3:  # Threshold
                # Recalculate confidence with keyword count
                article_data = {
                    "site": article.metadata.get("site", "unknown"),
                    "word_count": article.metadata.get("word_count", 0),
                    "keyword_count": keyword_count,
                    "equipment": article.extracted_data.get("equipment_mentions", []),
                }
                article.confidence = self._calculate_article_confidence(article_data)

                keyword_density = keyword_count / len(doc) if len(doc) > 0 else 0
                article.extracted_data["keyword_count"] = keyword_count
                article.extracted_data["keyword_density"] = keyword_density

                filtered.append(article)
            else:
                self.logger.debug(
                    f"Filtered out article (only {keyword_count} keywords): " f"{article.title}"
                )

        return filtered

    def _extract_equipment_and_techniques(self, text: str) -> dict[str, Any]:
        """
        Extract equipment mentions and technique descriptions from text.

        Args:
            text: Article text content

        Returns:
            Dictionary with extracted equipment and techniques
        """
        # Process with spaCy
        doc = self.nlp(text)

        # Equipment keywords
        equipment_keywords = [
            "cymbal",
            "snare",
            "kick",
            "tom",
            "hi-hat",
            "ride",
            "crash",
            "drum kit",
            "drum set",
            "drums",
            "stick",
            "brush",
            "mallet",
        ]

        # Technique keywords
        technique_keywords = [
            "technique",
            "style",
            "approach",
            "method",
            "playing",
            "fast",
            "slow",
            "heavy",
            "light",
            "powerful",
            "subtle",
            "syncopated",
            "straight",
            "swing",
            "shuffle",
            "groove",
        ]

        equipment = []
        techniques = []

        # Extract sentences containing keywords
        for sent in doc.sents:
            sent_text = sent.text.lower()

            # Check for equipment
            for eq in equipment_keywords:
                if eq in sent_text:
                    equipment.append(sent.text.strip())
                    break

            # Check for techniques
            for tech in technique_keywords:
                if tech in sent_text and len(sent.text) < 200:
                    techniques.append(sent.text.strip())
                    break

        return {
            "equipment_mentions": list(set(equipment))[:5],  # Max 5
            "technique_mentions": list(set(techniques))[:5],  # Max 5
        }


# TODO: Implement unit tests
# - Test NLP filtering with mock articles
# - Test equipment/technique extraction
# - Test HTML content extraction
# - Test URL transformation
# - Test robots.txt compliance
# - Test with BeautifulSoup mock responses
