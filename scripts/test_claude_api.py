"""Quick test of Claude API integration."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.research.producer_agent import ProducerResearchAgent

# Load environment variables
load_dotenv()


async def test_claude_api():
    """Test Claude API with a fresh producer research."""

    print("=" * 80)
    print("TESTING CLAUDE API INTEGRATION")
    print("=" * 80)

    # Check if API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not found in .env file!")
        return

    print(f"\nClaude API Key: {api_key[:20]}...{api_key[-4:]}")
    print("Initializing producer research agent with Claude API...\n")

    # Initialize agent with Claude API
    agent = ProducerResearchAgent(claude_api_key=api_key)

    try:
        # Clear cache for this producer to force fresh research
        agent.clear_cache("Burial")

        print("Researching producer: Burial (UK dubstep producer)")
        print("This will use Claude API to synthesize style parameters...\n")

        # Research producer (should use Claude API)
        profile = await agent.research_producer("Burial")

        # Display results
        print("\n" + "=" * 80)
        print("RESEARCH RESULTS")
        print("=" * 80)

        print(f"\nProducer: {profile['producer_name']}")
        print(f"Research Quality: {profile['research_quality']}")
        print(f"Synthesized With: {profile['style_params'].get('synthesized_with', 'claude_api')}")

        print("\n--- Style Parameters ---")
        print(json.dumps(profile["style_params"], indent=2))

        print("\n--- Data Sources ---")
        print(json.dumps(profile["data_sources"], indent=2))

        print("\n" + "=" * 80)
        if profile["research_quality"] == "high":
            print("✓ SUCCESS! Claude API is working perfectly!")
        elif profile["research_quality"] == "medium":
            print("✓ SUCCESS! Claude API is working (some data sources missing)")
        else:
            print("⚠ WARNING: Quality is low - Claude API may not have been used")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        agent.close()


if __name__ == "__main__":
    asyncio.run(test_claude_api())
