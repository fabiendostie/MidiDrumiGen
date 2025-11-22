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
import time
import xml.etree.ElementTree as ET
from math import log10

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

        # Rate limiting tracking
        self._semantic_scholar_calls: list[float] = []
        self._arxiv_last_call: float = 0
        self._crossref_calls: list[float] = []

        # Rate limit constants
        self._SS_RATE_LIMIT = 100  # requests per 5 minutes
        self._SS_RATE_WINDOW = 300  # 5 minutes in seconds
        self._ARXIV_MIN_INTERVAL = 3  # seconds between requests
        self._CROSSREF_RATE_LIMIT = 50  # requests per second

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

        except TimeoutError as e:
            raise CollectorTimeoutError(f"Paper collection timed out after {self.timeout}s") from e
        except Exception as e:
            raise CollectorError(f"Paper collection failed: {e}") from e

    async def _wait_for_rate_limit_semantic_scholar(self):
        """Wait if necessary to comply with Semantic Scholar rate limits."""
        current_time = time.time()
        # Remove calls older than the rate window
        self._semantic_scholar_calls = [
            t for t in self._semantic_scholar_calls if current_time - t < self._SS_RATE_WINDOW
        ]

        if len(self._semantic_scholar_calls) >= self._SS_RATE_LIMIT:
            # Wait until oldest call expires
            wait_time = self._SS_RATE_WINDOW - (current_time - self._semantic_scholar_calls[0])
            if wait_time > 0:
                self.logger.info(f"Rate limiting: waiting {wait_time:.1f}s for Semantic Scholar")
                await asyncio.sleep(wait_time)

        self._semantic_scholar_calls.append(time.time())

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
        query = f'"{artist_name}" drumming style rhythm analysis'
        self.logger.debug(f"Querying Semantic Scholar with: {query}")

        url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/search"
        params = {
            "query": query,
            "fields": "title,abstract,authors,citationCount,year,url,paperId",
            "limit": 10,
        }

        try:
            async with aiohttp.ClientSession() as session:
                await self._wait_for_rate_limit_semantic_scholar()
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    for paper in data.get("data", []):
                        abstract = paper.get("abstract", "") or ""
                        tempo_mentions = self._extract_tempo_mentions(abstract)
                        confidence = self._calculate_paper_confidence(paper)

                        papers.append(
                            ResearchSource(
                                source_type="paper",
                                title=paper.get("title", "Untitled"),
                                url=paper.get("url"),
                                raw_content=abstract,
                                extracted_data={
                                    "authors": [a.get("name") for a in paper.get("authors", [])],
                                    "citation_count": paper.get("citationCount", 0),
                                    "year": paper.get("year"),
                                    "tempo_mentions": tempo_mentions,
                                    "source": "semantic_scholar",
                                    "paper_id": paper.get("paperId"),
                                },
                                confidence=confidence,
                                metadata={"database": "Semantic Scholar", "query": query},
                            )
                        )
        except aiohttp.ClientResponseError as e:
            self.logger.error(
                f"Semantic Scholar API request failed with status {e.status}: {e.message}"
            )
        except TimeoutError:
            self.logger.error("Semantic Scholar request timed out after 30 seconds.")
        except aiohttp.ClientError as e:
            self.logger.error(f"Semantic Scholar request failed due to a client error: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during Semantic Scholar search: {e}", exc_info=True
            )

        self.logger.info(f"Found {len(papers)} papers from Semantic Scholar for '{artist_name}'.")
        return papers

    def _calculate_paper_confidence(self, paper: dict) -> float:
        """
        Calculate confidence score for a paper using tech spec formula.

        Args:
            paper: Paper data dict with citationCount, year, abstract

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.5

        # Citation boost (log scale to prevent runaway values)
        citations = paper.get("citationCount", 0) or 0
        citation_boost = min(0.3, 0.1 * log10(citations + 1))

        # Recency boost
        year = paper.get("year") or 2000
        if year >= 2020:
            recency_boost = 0.1
        elif year >= 2015:
            recency_boost = 0.05
        else:
            recency_boost = 0

        # Abstract relevance (has tempo mentions)
        abstract = paper.get("abstract", "") or ""
        tempo_mentions = len(re.findall(r"\d+\s*BPM", abstract, re.IGNORECASE))
        relevance_boost = min(0.2, 0.05 * tempo_mentions)

        return min(1.0, base_confidence + citation_boost + recency_boost + relevance_boost)

    async def _wait_for_rate_limit_arxiv(self):
        """Wait if necessary to comply with arXiv rate limits (1 req per 3 seconds)."""
        current_time = time.time()
        time_since_last = current_time - self._arxiv_last_call

        if time_since_last < self._ARXIV_MIN_INTERVAL:
            wait_time = self._ARXIV_MIN_INTERVAL - time_since_last
            self.logger.info(f"Rate limiting: waiting {wait_time:.1f}s for arXiv")
            await asyncio.sleep(wait_time)

        self._arxiv_last_call = time.time()

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
        query = f'all:"{artist_name}" AND (drumming OR percussion OR rhythm)'
        self.logger.debug(f"Querying arXiv with: {query}")

        params = {
            "search_query": query,
            "start": 0,
            "max_results": 10,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            await self._wait_for_rate_limit_arxiv()
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    self.ARXIV_BASE, params=params, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp,
            ):
                resp.raise_for_status()
                xml_content = await resp.text()

            if not xml_content:
                return papers

            root = ET.fromstring(xml_content)
            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            for entry in root.findall("atom:entry", namespaces):
                title_elem = entry.find("atom:title", namespaces)
                summary_elem = entry.find("atom:summary", namespaces)
                id_elem = entry.find("atom:id", namespaces)
                published_elem = entry.find("atom:published", namespaces)

                title = (
                    title_elem.text.strip()
                    if title_elem is not None and title_elem.text
                    else "Untitled"
                )
                abstract = (
                    summary_elem.text.strip()
                    if summary_elem is not None and summary_elem.text
                    else ""
                )
                arxiv_id = id_elem.text if id_elem is not None and id_elem.text else ""
                published = (
                    published_elem.text
                    if published_elem is not None and published_elem.text
                    else ""
                )

                authors = [
                    author.find("atom:name", namespaces).text
                    for author in entry.findall("atom:author", namespaces)
                    if author.find("atom:name", namespaces) is not None
                ]

                year = int(published[:4]) if published else None
                tempo_mentions = self._extract_tempo_mentions(abstract)
                paper_data = {"citationCount": 0, "year": year, "abstract": abstract}
                confidence = self._calculate_paper_confidence(paper_data)

                papers.append(
                    ResearchSource(
                        source_type="paper",
                        title=title,
                        url=arxiv_id,
                        raw_content=abstract,
                        extracted_data={
                            "authors": authors,
                            "year": year,
                            "tempo_mentions": tempo_mentions,
                            "source": "arxiv",
                            "arxiv_id": arxiv_id.split("/")[-1] if arxiv_id else None,
                        },
                        confidence=confidence,
                        metadata={"database": "arXiv", "query": query},
                    )
                )
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"arXiv API request failed with status {e.status}: {e.message}")
        except TimeoutError:
            self.logger.error("arXiv request timed out after 30 seconds.")
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse arXiv XML response: {e}")
        except aiohttp.ClientError as e:
            self.logger.error(f"arXiv request failed due to a client error: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during arXiv search: {e}", exc_info=True
            )

        self.logger.info(f"Found {len(papers)} papers from arXiv for '{artist_name}'.")
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
        query = f'"{artist_name}" drumming rhythm'
        self.logger.debug(f"Querying CrossRef with: {query}")

        url = self.CROSSREF_BASE
        params = {
            "query": query,
            "rows": 10,
            "sort": "relevance",
            "order": "desc",
            "select": "DOI,title,abstract,is-referenced-by-count,author,published-print,URL,created",
        }
        headers = {
            "User-Agent": "MidiDrumiGen/2.0 (mailto:fabz@mididrumigen.com)",
        }

        try:
            async with (
                aiohttp.ClientSession(headers=headers) as session,
                session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp,
            ):
                resp.raise_for_status()
                data = await resp.json()

            for item in data.get("message", {}).get("items", []):
                abstract = item.get("abstract", "") or ""
                tempo_mentions = self._extract_tempo_mentions(abstract)

                year = None
                for date_field in ["published-print", "published-online", "created"]:
                    if date_field in item and item[date_field].get("date-parts"):
                        year = item[date_field]["date-parts"][0][0]
                        break

                paper_data = {
                    "citationCount": item.get("is-referenced-by-count", 0),
                    "year": year,
                    "abstract": abstract,
                }
                confidence = self._calculate_paper_confidence(paper_data)
                title = item.get("title", ["Untitled"])[0]

                papers.append(
                    ResearchSource(
                        source_type="paper",
                        title=title,
                        url=item.get("URL"),
                        raw_content=abstract,
                        extracted_data={
                            "authors": [
                                f"{a.get('given', '')} {a.get('family', '')}".strip()
                                for a in item.get("author", [])
                            ],
                            "year": year,
                            "doi": item.get("DOI"),
                            "citation_count": item.get("is-referenced-by-count", 0),
                            "tempo_mentions": tempo_mentions,
                            "source": "crossref",
                        },
                        confidence=confidence,
                        metadata={"database": "CrossRef", "query": query},
                    )
                )
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"CrossRef API request failed with status {e.status}: {e.message}")
        except TimeoutError:
            self.logger.error("CrossRef request timed out after 30 seconds.")
        except aiohttp.ClientError as e:
            self.logger.error(f"CrossRef request failed due to a client error: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during CrossRef search: {e}", exc_info=True
            )

        self.logger.info(f"Found {len(papers)} papers from CrossRef for '{artist_name}'.")
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
