"""Test Gemini API integration."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.research.producer_agent import ProducerResearchAgent

load_dotenv()


async def main():
    print("=" * 80)
    print("TESTING GEMINI API INTEGRATION")
    print("=" * 80)

    # Initialize agent (will use Gemini as backup since Claude has no credits)
    agent = ProducerResearchAgent()

    try:
        # Clear cache to force fresh research
        agent.clear_cache("Flying Lotus")

        print("\nResearching: Flying Lotus (Electronic/Hip-Hop Producer)")
        print("This will use Gemini API since Claude has no credits...")
        print()

        # Research producer (should fall back to Gemini)
        profile = await agent.research_producer("Flying Lotus")

        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        print(f"\nProducer: {profile['producer_name']}")
        print(f"Research Quality: {profile['research_quality']}")
        print("\nStyle Parameters:")
        print(json.dumps(profile["style_params"], indent=2))

        print("\n" + "=" * 80)
        if (
            "synthesized_with" in profile["style_params"]
            and profile["style_params"]["synthesized_with"] != "default_heuristics"
        ):
            print("SUCCESS! Gemini API is working!")
        else:
            print("Note: Using default heuristics (Gemini may not have been called)")
        print("=" * 80)

    finally:
        agent.close()


if __name__ == "__main__":
    asyncio.run(main())
