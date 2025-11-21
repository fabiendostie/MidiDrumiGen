"""Test script for producer research agent."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from src.research.producer_agent import ProducerResearchAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_single_producer(producer_name: str):
    """Test researching a single producer."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing producer research: {producer_name}")
    logger.info(f"{'='*80}\n")

    # Initialize agent (API keys from environment variables)
    agent = ProducerResearchAgent()

    try:
        # Research producer
        profile = await agent.research_producer(producer_name)

        # Print results
        print(f"\n{producer_name} Research Results:")
        print(f"{'-'*80}")
        print(f"Cached: {profile['cached']}")
        print(f"Research Quality: {profile['research_quality']}")
        print("\nStyle Parameters:")
        print(json.dumps(profile["style_params"], indent=2))
        print("\nData Sources:")
        print(json.dumps(profile["data_sources"], indent=2))
        print(f"{'-'*80}\n")

        return profile

    finally:
        agent.close()


async def test_batch_research():
    """Test researching multiple producers at once."""
    logger.info(f"\n{'='*80}")
    logger.info("Testing batch producer research")
    logger.info(f"{'='*80}\n")

    # Test with 5 known producers
    producers = [
        "Timbaland",
        "J Dilla",
        "Metro Boomin",
        "Questlove",
        "Travis Barker",
    ]

    agent = ProducerResearchAgent()

    try:
        # Batch research
        profiles = await agent.batch_research(producers, max_concurrent=3)

        # Print summary
        print("\nBatch Research Summary:")
        print(f"{'-'*80}")
        print(f"{'Producer':<20} {'Quality':<10} {'Tempo Range':<15} {'Swing %':<10} {'Genres'}")
        print(f"{'-'*80}")

        for name, profile in profiles.items():
            style = profile["style_params"]
            tempo = f"{style['tempo_range'][0]}-{style['tempo_range'][1]}"
            swing = f"{style['swing_percentage']:.1f}"
            genres = ", ".join(style["genre_tags"][:2])

            print(f"{name:<20} {profile['research_quality']:<10} {tempo:<15} {swing:<10} {genres}")

        print(f"{'-'*80}\n")

        # Show cache stats
        stats = agent.get_cache_stats()
        print("Cache Statistics:")
        print(json.dumps(stats, indent=2))

        return profiles

    finally:
        agent.close()


async def test_cache_functionality():
    """Test cache hit/miss behavior."""
    logger.info(f"\n{'='*80}")
    logger.info("Testing cache functionality")
    logger.info(f"{'='*80}\n")

    agent = ProducerResearchAgent()

    try:
        # First request (should miss cache)
        print("First request (should be MISS)...")
        profile1 = await agent.research_producer("Timbaland")
        print(f"  Cached: {profile1['cached']}")
        print(f"  Quality: {profile1['research_quality']}")

        # Second request (should hit cache)
        print("\nSecond request (should be HIT)...")
        profile2 = await agent.research_producer("Timbaland")
        print(f"  Cached: {profile2['cached']}")
        print(f"  Timestamp: {profile2['cached_at']}")

        # Verify they match
        assert profile1["style_params"] == profile2["style_params"]
        print("\n[OK] Cache working correctly!")

    finally:
        agent.close()


async def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("PRODUCER RESEARCH AGENT - TEST SUITE")
    print("=" * 80)

    # Check for API keys
    import os

    has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_youtube = bool(os.getenv("YOUTUBE_API_KEY"))

    print("\nAPI Key Status:")
    print(f"  Claude API: {'OK' if has_claude else 'MISSING (will use defaults)'}")
    print(f"  OpenAI API: {'OK' if has_openai else 'MISSING (optional)'}")
    print(f"  YouTube API: {'OK' if has_youtube else 'MISSING (optional)'}")

    if not has_claude:
        print("\nWARNING: No Claude API key found in environment.")
        print("Set ANTHROPIC_API_KEY to enable LLM-powered style synthesis.")
        print("Continuing with default heuristics...\n")

    # Run tests
    try:
        # Test 1: Single producer
        await test_single_producer("Timbaland")

        # Test 2: Batch research
        await test_batch_research()

        # Test 3: Cache functionality
        await test_cache_functionality()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print("\n" + "=" * 80)
        print("TESTS FAILED!")
        print("=" * 80 + "\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
