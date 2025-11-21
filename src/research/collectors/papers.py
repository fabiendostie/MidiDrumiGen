"""
Scholar Paper Collector for MidiDrumiGen v2.0

Searches academic databases (Semantic Scholar, arXiv, CrossRef) for papers
analyzing artist drumming styles.

Story: E1.S1 - Scholar Paper Collection
Epic: E1 - Research Pipeline
Priority: HIGH
Story Points: 3
"""

import asyncio
import re

import aiohttp

from .base import BaseCollector, CollectorError, CollectorTimeoutError, ResearchSource


class ScholarPaperCollector(BaseCollector):
    """
    Collects academic papers about artist drumming styles.

    Sources:
        - Semantic Scholar API (100 req/5min)
        - arXiv API (unlimited)
        - CrossRef API (50 req/sec)

    Acceptance Criteria:
        - Searches all 3 configured APIs
        - Extracts tempo mentions using regex
        - Assigns confidence scores based on citation count
        - Handles rate limiting with exponential backoff
        - Returns List[ResearchSource] with source_type='paper'
    """

    # API Endpoints
    SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
    ARXIV_BASE = "http://export.arxiv.org/api/query"
    CROSSREF_BASE = "https://api.crossref.org/works"

    def __init__(self, timeout: int = 300, min_papers: int = 3):
        """
        Initialize Scholar Paper Collector.

        Args:
            timeout: Maximum time in seconds for collection (default: 5 min)
            min_papers: Minimum number of papers to collect
        """
        super().__init__(timeout)
        self.min_papers = min_papers

    async def collect(self, artist_name: str) -> list[ResearchSource]:
        """
        Collect academic papers about the artist's drumming style.

        Args:
            artist_name: Name of the artist/drummer to research

        Returns:
            List of ResearchSource objects with source_type='paper'

        Raises:
            CollectorTimeoutError: If collection exceeds timeout
            CollectorError: If critical error occurs
        """
        self.logger.info(f"Starting paper collection for artist: {artist_name}")

        papers = []

        try:
            # Run all searches in parallel
            results = await asyncio.gather(
                self._search_semantic_scholar(artist_name),
                self._search_arxiv(artist_name),
                self._search_crossref(artist_name),
                return_exceptions=True,
            )

            # Combine results, filtering out exceptions
            for result in results:
                if isinstance(result, Exception):
                    self.logger.warning(f"Search failed: {result}")
                else:
                    papers.extend(result)

            self.logger.info(f"Collected {len(papers)} papers for {artist_name}")

            if len(papers) < self.min_papers:
                self.logger.warning(
                    f"Only found {len(papers)} papers, " f"minimum {self.min_papers} recommended"
                )

            return papers

        except TimeoutError:
            raise CollectorTimeoutError(f"Paper collection timed out after {self.timeout}s")
        except Exception as e:
            raise CollectorError(f"Paper collection failed: {e}")

    async def _search_semantic_scholar(self, artist_name: str) -> list[ResearchSource]:
        """
        Search Semantic Scholar for papers about artist.

        API Docs: https://api.semanticscholar.org/api-docs/graph

        Args:
            artist_name: Name of artist to search

        Returns:
            List of ResearchSource objects from Semantic Scholar
        """
        papers = []
        query = f"{artist_name} drumming rhythm analysis"

        url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/search"
        params = {
            "query": query,
            "fields": "title,abstract,authors,citationCount,year,url",
            "limit": 10,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Semantic Scholar returned {resp.status}")
                        return papers

                    data = await resp.json()

                    for paper in data.get("data", []):
                        # Extract tempo mentions from abstract
                        tempo_mentions = self._extract_tempo_mentions(paper.get("abstract", ""))

                        # Calculate confidence based on citations
                        confidence = self._calculate_confidence(
                            citation_count=paper.get("citationCount", 0),
                            relevance_score=0.7,  # Assume relevant if returned
                            source_quality=0.9,  # High quality (academic)
                        )

                        papers.append(
                            ResearchSource(
                                source_type="paper",
                                title=paper.get("title", "Untitled"),
                                url=paper.get("url"),
                                raw_content=paper.get("abstract", ""),
                                extracted_data={
                                    "authors": [a.get("name") for a in paper.get("authors", [])],
                                    "citation_count": paper.get("citationCount", 0),
                                    "year": paper.get("year"),
                                    "tempo_mentions": tempo_mentions,
                                    "source": "semantic_scholar",
                                },
                                confidence=confidence,
                                metadata={"database": "Semantic Scholar", "query": query},
                            )
                        )

        except Exception as e:
            self.logger.error(f"Semantic Scholar search failed: {e}")

        return papers

    async def _search_arxiv(self, artist_name: str) -> list[ResearchSource]:
        """
        Search arXiv for papers about artist.

        API Docs: https://info.arxiv.org/help/api/index.html

        Args:
            artist_name: Name of artist to search

        Returns:
            List of ResearchSource objects from arXiv
        """
        papers = []
        query = f"all:{artist_name} AND (drumming OR percussion OR rhythm)"

        params = {
            "search_query": query,
            "start": 0,
            "max_results": 10,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.ARXIV_BASE, params=params) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"arXiv returned {resp.status}")
                        return papers

                    # Parse XML response (arXiv returns Atom XML)
                    xml_content = await resp.text()
                    # TODO: Implement XML parsing
                    # For now, return empty list
                    self.logger.info("arXiv search completed (XML parsing TODO)")

        except Exception as e:
            self.logger.error(f"arXiv search failed: {e}")

        return papers

    async def _search_crossref(self, artist_name: str) -> list[ResearchSource]:
        """
        Search CrossRef for papers about artist.

        API Docs: https://github.com/CrossRef/rest-api-doc

        Args:
            artist_name: Name of artist to search

        Returns:
            List of ResearchSource objects from CrossRef
        """
        papers = []
        query = f"{artist_name} drumming"

        url = self.CROSSREF_BASE
        params = {"query": query, "rows": 10, "sort": "relevance", "order": "desc"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"CrossRef returned {resp.status}")
                        return papers

                    data = await resp.json()

                    for item in data.get("message", {}).get("items", []):
                        # Extract abstract if available
                        abstract = item.get("abstract", "")

                        # Extract tempo mentions
                        tempo_mentions = self._extract_tempo_mentions(abstract)

                        # Calculate confidence
                        confidence = self._calculate_confidence(
                            citation_count=item.get("is-referenced-by-count", 0),
                            relevance_score=0.6,
                            source_quality=0.8,
                        )

                        papers.append(
                            ResearchSource(
                                source_type="paper",
                                title=item.get("title", ["Untitled"])[0],
                                url=item.get("URL"),
                                raw_content=abstract,
                                extracted_data={
                                    "authors": [
                                        f"{a.get('given', '')} {a.get('family', '')}"
                                        for a in item.get("author", [])
                                    ],
                                    "year": item.get("published-print", {}).get(
                                        "date-parts", [[None]]
                                    )[0][0],
                                    "doi": item.get("DOI"),
                                    "tempo_mentions": tempo_mentions,
                                    "source": "crossref",
                                },
                                confidence=confidence,
                                metadata={"database": "CrossRef", "query": query},
                            )
                        )

        except Exception as e:
            self.logger.error(f"CrossRef search failed: {e}")

        return papers

    def _extract_tempo_mentions(self, text: str) -> list[int]:
        """
        Extract tempo mentions (BPM) from text using regex.

        Args:
            text: Text to search for tempo mentions

        Returns:
            List of tempo values (integers) found in text

        Examples:
            >>> _extract_tempo_mentions("typical tempo of 120 BPM")
            [120]
            >>> _extract_tempo_mentions("ranging from 90-140 beats per minute")
            [90, 140]
        """
        if not text:
            return []

        # Regex patterns for tempo mentions
        patterns = [
            r"(\d+)\s*(?:BPM|bpm)",  # "120 BPM"
            r"(\d+)\s*beats?\s*per\s*minute",  # "120 beats per minute"
            r"tempo\s*of\s*(\d+)",  # "tempo of 120"
            r"(\d+)\s*(?:to|-)\s*(\d+)\s*(?:BPM|bpm)",  # "90-120 BPM" (range)
        ]

        tempos = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Range pattern
                    tempos.extend([int(t) for t in match if t.isdigit()])
                else:
                    # Single value
                    if match.isdigit():
                        tempos.append(int(match))

        # Filter out unrealistic tempos (20-300 BPM)
        tempos = [t for t in tempos if 20 <= t <= 300]

        return list(set(tempos))  # Remove duplicates


# TODO: Implement unit tests
# - Test tempo extraction regex
# - Test confidence calculation
# - Test API response parsing
# - Test rate limiting handling
# - Test with mock API responses
