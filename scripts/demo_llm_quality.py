"""Demonstrate LLM-powered producer research quality."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.research.producer_agent import ProducerResearchAgent

load_dotenv()


async def main():
    print("=" * 80)
    print("LLM-POWERED PRODUCER RESEARCH DEMO")
    print("=" * 80)

    producers = [
        "Timbaland",
        "Aphex Twin",
        "Anderson .Paak",
        "Burial",
    ]

    agent = ProducerResearchAgent()

    try:
        for producer in producers:
            # Clear cache to get fresh LLM research
            agent.clear_cache(producer)

            print(f"\n{'='*80}")
            print(f"RESEARCHING: {producer}")
            print(f"{'='*80}\n")

            profile = await agent.research_producer(producer)

            style = profile["style_params"]

            print(f"Tempo Range: {style['tempo_range'][0]}-{style['tempo_range'][1]} BPM")
            print(f"Swing: {style['swing_percentage']}%")
            print(f"Complexity: {style['complexity_level']}")
            print("\nSignature Techniques:")
            for tech in style["signature_techniques"][:5]:
                print(f"  - {tech}")

            print(f"\nGenres: {', '.join(style['genre_tags'][:4])}")
            print(f"\nDescription:\n  {style['description'][:200]}...")

            # Wait a bit between requests (Gemini rate limits)
            if producer != producers[-1]:
                print("\nWaiting 5 seconds for rate limits...")
                await asyncio.sleep(5)

        print(f"\n{'='*80}")
        print("DEMO COMPLETE - LLM synthesis is working beautifully!")
        print(f"{'='*80}")

    finally:
        agent.close()


if __name__ == "__main__":
    asyncio.run(main())
