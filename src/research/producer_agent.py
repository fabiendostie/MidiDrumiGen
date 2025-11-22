"""Main producer research agent - orchestrates web scraping and LLM synthesis."""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.cache import ProducerStyleCache
from src.research.llm_synthesizer import StyleSynthesizer
from src.research.web_scraper import ProducerWebScraper

logger = logging.getLogger(__name__)


class ProducerResearchAgent:
    """
    Automatically research ANY producer's drum programming style.

    Workflow:
    1. Check cache for existing profile
    2. If not cached:
       a. Scrape web sources (Wikipedia, YouTube, MIDI repos)
       b. Synthesize style parameters using LLM (Claude/GPT-4)
       c. Cache result for 30 days
    3. Return style profile

    Example:
        >>> agent = ProducerResearchAgent(claude_api_key="...")
        >>> profile = await agent.research_producer("Timbaland")
        >>> print(profile['style_params']['tempo_range'])
        [95, 140]
    """

    def __init__(
        self,
        claude_api_key: str | None = None,
        openai_api_key: str | None = None,
        youtube_api_key: str | None = None,
        redis_url: str | None = None,
        cache_dir: Path | str = "data/producer_cache",
    ):
        """
        Initialize producer research agent.

        Args:
            claude_api_key: Anthropic API key (optional, reads from env)
            openai_api_key: OpenAI API key for GPT-4 backup (optional)
            youtube_api_key: YouTube Data API v3 key (optional)
            redis_url: Redis connection URL (optional, e.g., "redis://localhost:6379/0")
            cache_dir: Directory for file-based cache fallback
        """
        # Initialize cache
        self.cache = ProducerStyleCache(redis_url=redis_url, cache_dir=cache_dir)

        # Initialize web scraper
        self.scraper = ProducerWebScraper()

        # Initialize LLM synthesizer
        self.synthesizer = StyleSynthesizer(
            claude_api_key=claude_api_key,
            openai_api_key=openai_api_key,
        )

        # Store API keys
        self.youtube_api_key = youtube_api_key

        logger.info("ProducerResearchAgent initialized")

    async def research_producer(
        self,
        producer_name: str,
        force_refresh: bool = False,
        include_youtube: bool = True,
        include_midi_search: bool = False,
    ) -> dict[str, Any]:
        """
        Research producer style from multiple sources.

        Returns cached profile if available, otherwise performs full research.

        Args:
            producer_name: Producer name (any format, e.g., "Timbaland", "J. Dilla")
            force_refresh: If True, bypass cache and research from scratch
            include_youtube: Whether to search YouTube tutorials
            include_midi_search: Whether to search MIDI repositories

        Returns:
            Complete producer profile with:
                - producer_name: Normalized name
                - cached: Whether result was from cache
                - style_params: Style parameters dict
                - data_sources: Sources used for research
                - cached_at: Timestamp

        Example:
            >>> agent = ProducerResearchAgent()
            >>> profile = await agent.research_producer("Timbaland")
            >>> print(profile['style_params']['swing_percentage'])
            54.0
        """
        logger.info(f"Researching producer: {producer_name}")

        # 1. Check cache (unless force_refresh)
        if not force_refresh:
            cached_profile = self.cache.get(producer_name)
            if cached_profile:
                logger.info(f"Found cached profile for {producer_name}")
                cached_profile["cached"] = True
                return cached_profile

        logger.info(f"No cache found, performing fresh research for {producer_name}")

        # 2. Scrape web sources
        scraped_data = await self.scraper.scrape_all(
            producer_name=producer_name,
            youtube_api_key=self.youtube_api_key if include_youtube else None,
            include_transcripts=True,
        )

        # 3. Synthesize style parameters using LLM
        try:
            style_params = await self.synthesizer.synthesize_style(
                producer_name=producer_name,
                scraped_data=scraped_data,
            )
        except Exception as e:
            logger.error(f"Style synthesis failed: {e}, using defaults")
            style_params = self.synthesizer._create_default_params(producer_name, scraped_data)

        # 4. Build complete profile
        profile = {
            "producer_name": producer_name,
            "normalized_name": self.cache._normalize_name(producer_name),
            "cached": False,
            "cached_at": datetime.now(UTC).isoformat() + "Z",
            "style_params": style_params,
            "data_sources": {
                "wikipedia": {
                    "found": scraped_data["wikipedia"]["found"],
                    "url": scraped_data["wikipedia"].get("page_url"),
                },
                "youtube": {
                    "found": scraped_data["youtube"]["found"],
                    "video_count": scraped_data["youtube"].get("count", 0),
                },
                "midi_repositories": {
                    "found": scraped_data["midi_repositories"]["found"],
                },
            },
            "model_adapter_path": None,  # Populated after training adapter
            "research_quality": self._assess_research_quality(scraped_data, style_params),
        }

        # 5. Cache result
        self.cache.set(producer_name, profile)

        logger.info(
            f"Research complete for {producer_name}, quality: {profile['research_quality']}"
        )

        return profile

    def _assess_research_quality(
        self, scraped_data: dict[str, Any], style_params: dict[str, Any]
    ) -> str:
        """
        Assess quality of research based on data sources available.

        Args:
            scraped_data: Scraped web data
            style_params: Synthesized style parameters

        Returns:
            Quality rating: "high", "medium", "low"
        """
        score = 0

        # Wikipedia found
        if scraped_data["wikipedia"]["found"]:
            score += 2

        # YouTube tutorials found
        if scraped_data["youtube"]["found"] and scraped_data["youtube"].get("count", 0) > 0:
            score += 2

        # Transcripts available
        youtube_videos = scraped_data["youtube"].get("videos", [])
        if any(v.get("transcript") for v in youtube_videos):
            score += 2

        # LLM synthesis (not just defaults)
        if style_params.get("synthesized_with") != "default_heuristics":
            score += 2

        # MIDI files found
        if scraped_data["midi_repositories"]["found"]:
            score += 2

        # Determine quality
        if score >= 6:
            return "high"
        elif score >= 4:
            return "medium"
        else:
            return "low"

    def get_cached_producer(self, producer_name: str) -> dict[str, Any] | None:
        """
        Get cached producer profile without research.

        Args:
            producer_name: Producer name

        Returns:
            Cached profile or None
        """
        return self.cache.get(producer_name)

    def clear_cache(self, producer_name: str | None = None) -> None:
        """
        Clear cache for specific producer or all producers.

        Args:
            producer_name: Producer name to clear, or None to clear all
        """
        if producer_name:
            self.cache.delete(producer_name)
            logger.info(f"Cleared cache for {producer_name}")
        else:
            self.cache.clear_all()
            logger.info("Cleared all producer cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        return self.cache.get_stats()

    async def batch_research(
        self, producer_names: list[str], max_concurrent: int = 3
    ) -> dict[str, dict[str, Any]]:
        """
        Research multiple producers concurrently.

        Args:
            producer_names: List of producer names to research
            max_concurrent: Maximum concurrent research operations

        Returns:
            Dict mapping producer names to profiles

        Example:
            >>> agent = ProducerResearchAgent()
            >>> producers = ["Timbaland", "J Dilla", "Metro Boomin"]
            >>> profiles = await agent.batch_research(producers)
            >>> for name, profile in profiles.items():
            ...     print(f"{name}: {profile['style_params']['swing_percentage']}")
        """
        logger.info(f"Batch researching {len(producer_names)} producers")

        async def research_with_semaphore(sem, name):
            async with sem:
                return name, await self.research_producer(name)

        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [research_with_semaphore(semaphore, name) for name in producer_names]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dict
        profiles = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch research error: {result}")
                continue

            name, profile = result
            profiles[name] = profile

        logger.info(f"Batch research complete: {len(profiles)}/{len(producer_names)} succeeded")

        return profiles

    def close(self):
        """Clean up resources."""
        self.scraper.close()
        logger.info("ProducerResearchAgent closed")


# Convenience function for quick testing
async def quick_research(producer_name: str, claude_api_key: str | None = None) -> dict[str, Any]:
    """
    Quick research function for testing.

    Example:
        >>> from src.research.producer_agent import quick_research
        >>> profile = await quick_research("Timbaland")
        >>> print(profile['style_params'])
    """
    agent = ProducerResearchAgent(claude_api_key=claude_api_key)
    try:
        return await agent.research_producer(producer_name)
    finally:
        agent.close()
