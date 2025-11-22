"""Web scraping for producer information and MIDI sources."""

import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ProducerWebScraper:
    """
    Scrape producer information from multiple web sources.

    Sources:
    - Wikipedia: Genre, biography, discography
    - YouTube: Tutorial transcripts (requires API)
    - MIDI repositories: Free MIDI packs

    Example:
        >>> scraper = ProducerWebScraper()
        >>> data = await scraper.scrape_all("Timbaland")
        >>> print(data['wikipedia']['genre'])
        ['hip hop', 'R&B', 'pop']
    """

    def __init__(self, user_agent: str = "MidiDrumiGen/1.0"):
        """
        Initialize web scraper.

        Args:
            user_agent: User agent string for requests
        """
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def scrape_wikipedia(self, producer_name: str) -> dict[str, Any]:
        """
        Scrape producer info from Wikipedia.

        Args:
            producer_name: Producer name to search

        Returns:
            Dict with Wikipedia data (genre, birth year, bio, etc.)

        Example:
            >>> scraper = ProducerWebScraper()
            >>> data = scraper.scrape_wikipedia("Timbaland")
            >>> print(data['genres'])
            ['Hip hop', 'R&B', 'Pop']
        """
        logger.info(f"Scraping Wikipedia for: {producer_name}")

        try:
            # Search Wikipedia
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": producer_name + " producer",
                "format": "json",
                "srlimit": 1,
            }

            response = self.session.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            search_results = response.json()

            if not search_results.get("query", {}).get("search"):
                logger.warning(f"No Wikipedia results for {producer_name}")
                return {"found": False}

            # Get page title
            page_title = search_results["query"]["search"][0]["title"]

            # Fetch page content
            content_url = "https://en.wikipedia.org/w/api.php"
            content_params = {
                "action": "query",
                "titles": page_title,
                "prop": "extracts|categories",
                "exintro": True,
                "explaintext": True,
                "format": "json",
            }

            response = self.session.get(content_url, params=content_params, timeout=10)
            response.raise_for_status()
            page_data = response.json()

            # Extract page content
            pages = page_data["query"]["pages"]
            page_id = list(pages.keys())[0]
            page = pages[page_id]

            extract = page.get("extract", "")
            categories = page.get("categories", [])

            # Extract genres from categories
            genres = []
            genre_keywords = [
                "hip hop",
                "r&b",
                "jazz",
                "rock",
                "electronic",
                "pop",
                "trap",
                "boom bap",
            ]
            for cat in categories:
                cat_title = cat.get("title", "").lower()
                for keyword in genre_keywords:
                    if keyword in cat_title and keyword not in genres:
                        genres.append(keyword)

            # Extract genres from intro text
            if not genres:
                intro_lower = extract.lower()
                for keyword in genre_keywords:
                    if keyword in intro_lower and keyword not in genres:
                        genres.append(keyword)

            # Extract birth year (if mentioned)
            birth_year = None
            birth_match = re.search(r"born.*?(\d{4})", extract)
            if birth_match:
                birth_year = int(birth_match.group(1))

            return {
                "found": True,
                "page_title": page_title,
                "page_url": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                "intro": extract[:500] + "..." if len(extract) > 500 else extract,
                "genres": genres,
                "birth_year": birth_year,
                "categories": [cat["title"] for cat in categories[:10]],
            }

        except Exception as e:
            logger.error(f"Wikipedia scraping error for {producer_name}: {e}")
            return {"found": False, "error": str(e)}

    def search_youtube_tutorials(
        self, producer_name: str, api_key: str | None = None, max_results: int = 3
    ) -> dict[str, Any]:
        """
        Search for YouTube tutorials about producer's drumming style.

        Requires YouTube Data API v3 key.

        Args:
            producer_name: Producer name
            api_key: YouTube API key (optional)
            max_results: Maximum number of videos to return

        Returns:
            Dict with video metadata (titles, URLs, descriptions)

        Example:
            >>> scraper = ProducerWebScraper()
            >>> data = scraper.search_youtube_tutorials("Timbaland", api_key="...")
            >>> for video in data['videos']:
            ...     print(video['title'])
        """
        logger.info(f"Searching YouTube for: {producer_name}")

        if not api_key:
            logger.warning("No YouTube API key provided, skipping YouTube search")
            return {"found": False, "reason": "no_api_key"}

        try:
            from googleapiclient.discovery import build

            youtube = build("youtube", "v3", developerKey=api_key)

            # Search for tutorials
            search_query = f"{producer_name} drum programming tutorial"
            search_response = (
                youtube.search()
                .list(
                    q=search_query,
                    part="id,snippet",
                    maxResults=max_results,
                    type="video",
                    relevanceLanguage="en",
                )
                .execute()
            )

            videos = []
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]

                videos.append(
                    {
                        "video_id": video_id,
                        "title": snippet["title"],
                        "description": snippet["description"],
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "channel": snippet["channelTitle"],
                        "published_at": snippet["publishedAt"],
                    }
                )

            return {
                "found": True,
                "query": search_query,
                "videos": videos,
                "count": len(videos),
            }

        except ImportError:
            logger.warning("google-api-python-client not installed")
            return {"found": False, "reason": "missing_library"}
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return {"found": False, "error": str(e)}

    def get_youtube_transcript(self, video_id: str) -> str | None:
        """
        Get transcript for a YouTube video.

        Requires youtube-transcript-api package.

        Args:
            video_id: YouTube video ID

        Returns:
            Full transcript text or None if unavailable

        Example:
            >>> scraper = ProducerWebScraper()
            >>> transcript = scraper.get_youtube_transcript("dQw4w9WgXcQ")
            >>> print(transcript[:100])
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_transcript = " ".join([t["text"] for t in transcript_list])

            logger.info(f"Retrieved transcript for video {video_id} ({len(full_transcript)} chars)")
            return full_transcript

        except ImportError:
            logger.warning("youtube-transcript-api not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to get transcript for {video_id}: {e}")
            return None

    def search_midi_repositories(self, producer_name: str) -> dict[str, Any]:
        """
        Search for producer-specific MIDI files in online repositories.

        Checks:
        - FreeMIDI.org (if available)
        - MIDI World
        - Kit Maker (for specific packs)

        Args:
            producer_name: Producer name

        Returns:
            Dict with MIDI file URLs and metadata

        Example:
            >>> scraper = ProducerWebScraper()
            >>> data = scraper.search_midi_repositories("J Dilla")
            >>> for midi in data['midi_files']:
            ...     print(midi['url'])
        """
        logger.info(f"Searching MIDI repositories for: {producer_name}")

        # Search FreeMIDI (example - actual API/scraping would be needed)
        try:
            # Note: This is a placeholder - actual implementation would need
            # to use the specific site's API or scraping logic
            search_query = producer_name.lower().replace(" ", "+")

            # FreeMIDI search (placeholder)
            # In reality, you'd implement proper scraping here
            logger.debug(f"Would search FreeMIDI for: {search_query}")

            # For now, return empty result
            return {
                "found": False,
                "reason": "not_implemented",
                "note": "MIDI repository scraping requires site-specific implementation",
                "suggested_sources": [
                    "https://freemidi.org",
                    "https://www.midiworld.com",
                    "https://kitmaker.com",
                ],
            }

        except Exception as e:
            logger.error(f"MIDI repository search error: {e}")
            return {"found": False, "error": str(e)}

    async def scrape_all(
        self,
        producer_name: str,
        youtube_api_key: str | None = None,
        include_transcripts: bool = True,
    ) -> dict[str, Any]:
        """
        Scrape all available sources for producer information.

        Args:
            producer_name: Producer name
            youtube_api_key: YouTube API key (optional)
            include_transcripts: Whether to fetch video transcripts

        Returns:
            Aggregated data from all sources

        Example:
            >>> scraper = ProducerWebScraper()
            >>> data = await scraper.scrape_all("Timbaland")
            >>> print(data.keys())
            dict_keys(['wikipedia', 'youtube', 'midi_repositories'])
        """
        logger.info(f"Scraping all sources for: {producer_name}")

        results = {}

        # Wikipedia
        results["wikipedia"] = self.scrape_wikipedia(producer_name)
        time.sleep(1)  # Be respectful with rate limiting

        # YouTube
        youtube_data = self.search_youtube_tutorials(producer_name, youtube_api_key)
        if youtube_data["found"] and include_transcripts:
            # Fetch transcripts for videos
            for video in youtube_data.get("videos", []):
                transcript = self.get_youtube_transcript(video["video_id"])
                video["transcript"] = transcript
                time.sleep(1)  # Rate limiting

        results["youtube"] = youtube_data

        # MIDI repositories
        results["midi_repositories"] = self.search_midi_repositories(producer_name)

        return results

    def close(self):
        """Close the requests session."""
        self.session.close()
