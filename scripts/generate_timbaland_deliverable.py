"""
Week 2 Deliverable: Generate Timbaland-style drum patterns.

This script demonstrates the complete Week 2 pipeline:
1. Research Timbaland's style (cached)
2. Generate base drum pattern
3. Apply dynamic style transfer
4. Export to MIDI
5. Validate and report characteristics
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.midi.constants import DEFAULT_TICKS_PER_BEAT
from src.midi.export import export_pattern
from src.midi.style_transfer import apply_producer_style, get_style_description
from src.research.producer_agent import ProducerResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_base_pattern(bars: int = 4, ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT):
    """
    Create a basic drum pattern to apply Timbaland style to.

    This creates a standard 4/4 pattern with kick, snare, and hi-hats.
    The style transfer will transform this into a Timbaland-style pattern.

    Args:
        bars: Number of bars to generate
        ticks_per_beat: MIDI resolution

    Returns:
        List of note dictionaries
    """
    notes = []
    beats_per_bar = 4

    logger.info(f"Creating base {bars}-bar pattern...")

    for bar in range(bars):
        base_tick = bar * beats_per_bar * ticks_per_beat

        # Kick drum on beats 1 and 3
        notes.append({"pitch": 36, "velocity": 100, "time": base_tick})  # Kick
        notes.append({"pitch": 36, "velocity": 95, "time": base_tick + 2 * ticks_per_beat})  # Kick

        # Snare on beats 2 and 4
        notes.append({"pitch": 38, "velocity": 90, "time": base_tick + ticks_per_beat})  # Snare
        notes.append({"pitch": 38, "velocity": 92, "time": base_tick + 3 * ticks_per_beat})  # Snare

        # Hi-hat pattern (8th notes)
        for eighth in range(beats_per_bar * 2):
            velocity = 70 if eighth % 2 == 0 else 60
            notes.append(
                {
                    "pitch": 42,  # Closed hi-hat
                    "velocity": velocity,
                    "time": base_tick + eighth * (ticks_per_beat // 2),
                }
            )

    logger.info(f"Created base pattern: {len(notes)} notes")
    return notes


def analyze_pattern_characteristics(notes, style_profile):
    """
    Analyze and report pattern characteristics.

    Args:
        notes: List of note dictionaries
        style_profile: Producer profile from research agent

    Returns:
        Dict with analysis results
    """
    style_params = style_profile.get("style_params", {})

    # Count notes by type
    kicks = sum(1 for n in notes if n["pitch"] == 36)
    snares = sum(1 for n in notes if n["pitch"] == 38)
    hihats = sum(1 for n in notes if n["pitch"] in [42, 44, 46])

    # Velocity analysis
    velocities = [n["velocity"] for n in notes]
    avg_velocity = sum(velocities) / len(velocities) if velocities else 0
    velocity_range = (min(velocities), max(velocities)) if velocities else (0, 0)

    # Timing analysis
    times = sorted(n["time"] for n in notes)
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    analysis = {
        "total_notes": len(notes),
        "kicks": kicks,
        "snares": snares,
        "hihats": hihats,
        "avg_velocity": round(avg_velocity, 1),
        "velocity_range": velocity_range,
        "avg_note_gap_ticks": round(avg_gap, 1),
        "style_params": {
            "swing_percentage": style_params.get("swing_percentage", "N/A"),
            "micro_timing_ms": style_params.get("micro_timing_ms", "N/A"),
            "ghost_note_prob": style_params.get("ghost_note_prob", "N/A"),
            "velocity_variation": style_params.get("velocity_variation", "N/A"),
            "complexity_level": style_params.get("complexity_level", "N/A"),
        },
    }

    return analysis


async def generate_timbaland_pattern(
    bars: int = 4, tempo: int = 100, variation_num: int = 1, output_dir: Path = None
):
    """
    Generate a single Timbaland-style drum pattern.

    Args:
        bars: Number of bars
        tempo: BPM
        variation_num: Variation number (for filename)
        output_dir: Output directory

    Returns:
        Path to generated MIDI file
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating Timbaland Pattern Variation #{variation_num}")
    logger.info(f"{'='*60}")

    # Step 1: Research Timbaland style
    logger.info("Step 1: Researching Timbaland style...")
    agent = ProducerResearchAgent()
    profile = await agent.research_producer("Timbaland")

    logger.info(f"  ✓ Research complete (quality: {profile.get('research_quality', 'unknown')})")
    logger.info(f"  ✓ Cached: {profile.get('cached', False)}")

    # Display style parameters
    style_params = profile.get("style_params", {})
    logger.info("\n  Style Parameters:")
    logger.info(f"    - Tempo range: {style_params.get('tempo_range', 'N/A')} BPM")
    logger.info(f"    - Swing: {style_params.get('swing_percentage', 'N/A')}%")
    logger.info(f"    - Micro-timing: ±{style_params.get('micro_timing_ms', 'N/A')}ms")
    logger.info(
        f"    - Ghost notes: {style_params.get('ghost_note_prob', 'N/A')*100 if isinstance(style_params.get('ghost_note_prob'), float) else 'N/A'}%"
    )
    logger.info(f"    - Velocity variation: {style_params.get('velocity_variation', 'N/A')}")
    logger.info(f"    - Complexity: {style_params.get('complexity_level', 'N/A')}")

    # Display signature techniques
    techniques = style_params.get("signature_techniques", [])
    if techniques:
        logger.info("\n  Signature Techniques:")
        for tech in techniques[:5]:
            logger.info(f"    - {tech}")

    # Step 2: Generate base pattern
    logger.info("\nStep 2: Generating base drum pattern...")
    base_notes = create_base_pattern(bars=bars)
    logger.info(f"  ✓ Base pattern created: {len(base_notes)} notes")

    # Step 3: Apply Timbaland style transfer
    logger.info("\nStep 3: Applying Timbaland style transfer...")
    styled_notes = apply_producer_style(
        notes=base_notes, style_profile=profile, tempo=tempo, ticks_per_beat=DEFAULT_TICKS_PER_BEAT
    )

    notes_added = len(styled_notes) - len(base_notes)
    logger.info(
        f"  ✓ Style transfer complete: {len(styled_notes)} notes ({notes_added:+d} from transformations)"
    )

    # Step 4: Analyze pattern characteristics
    logger.info("\nStep 4: Analyzing pattern characteristics...")
    analysis = analyze_pattern_characteristics(styled_notes, profile)

    logger.info("  Pattern Analysis:")
    logger.info(f"    - Total notes: {analysis['total_notes']}")
    logger.info(
        f"    - Kicks: {analysis['kicks']}, Snares: {analysis['snares']}, Hi-hats: {analysis['hihats']}"
    )
    logger.info(
        f"    - Avg velocity: {analysis['avg_velocity']} (range: {analysis['velocity_range']})"
    )
    logger.info(f"    - Avg note spacing: {analysis['avg_note_gap_ticks']} ticks")

    # Step 5: Export to MIDI
    logger.info("\nStep 5: Exporting to MIDI...")

    if output_dir is None:
        output_dir = Path("output/timbaland_deliverables")

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"timbaland_week2_v{variation_num}_{bars}bars_{tempo}bpm.mid"
    output_path = output_dir / filename

    midi_path = export_pattern(
        notes=styled_notes,
        output_path=output_path,
        tempo=tempo,
        time_signature=(4, 4),
        humanize=False,  # Already styled
        style_name="Timbaland",
        ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
    )

    logger.info(f"  ✓ MIDI exported: {midi_path}")

    # Step 6: Generate style description
    style_desc = get_style_description(profile)
    logger.info("\n  Style Description:")
    logger.info(f"    {style_desc}")

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Variation #{variation_num} Complete!")
    logger.info(f"{'='*60}\n")

    return midi_path, analysis


async def main():
    """Main function to generate Timbaland deliverables."""
    print("\n" + "=" * 60)
    print("WEEK 2 DELIVERABLE: Timbaland Drum Pattern Generator")
    print("=" * 60 + "\n")

    # Configuration
    NUM_VARIATIONS = 3
    BARS = 4
    TEMPO = 100

    output_dir = Path("output/timbaland_deliverables")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Configuration:")
    logger.info(f"  - Number of variations: {NUM_VARIATIONS}")
    logger.info(f"  - Bars per pattern: {BARS}")
    logger.info(f"  - Tempo: {TEMPO} BPM")
    logger.info(f"  - Output directory: {output_dir}\n")

    # Generate multiple variations
    results = []

    for i in range(1, NUM_VARIATIONS + 1):
        try:
            midi_path, analysis = await generate_timbaland_pattern(
                bars=BARS, tempo=TEMPO, variation_num=i, output_dir=output_dir
            )

            results.append(
                {"variation": i, "midi_path": midi_path, "analysis": analysis, "success": True}
            )

        except Exception as e:
            logger.error(f"Failed to generate variation #{i}: {e}", exc_info=True)
            results.append({"variation": i, "success": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60 + "\n")

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"Total variations requested: {NUM_VARIATIONS}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}\n")

    if successful > 0:
        print("Generated Files:")
        for result in results:
            if result["success"]:
                midi_path = result["midi_path"]
                analysis = result["analysis"]
                print(f"\n  Variation #{result['variation']}:")
                print(f"    File: {midi_path.name}")
                print(f"    Size: {midi_path.stat().st_size} bytes")
                print(
                    f"    Notes: {analysis['total_notes']} (K:{analysis['kicks']}, S:{analysis['snares']}, HH:{analysis['hihats']})"
                )
                print(
                    f"    Velocity: {analysis['velocity_range'][0]}-{analysis['velocity_range'][1]} (avg: {analysis['avg_velocity']})"
                )

    print(f"\n{'='*60}")
    print("SUCCESS: WEEK 2 DELIVERABLE COMPLETE")
    print(f"{'='*60}\n")

    print(f"All files saved to: {output_dir.resolve()}\n")

    # Save summary JSON
    summary_path = (
        output_dir / f"generation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    import json

    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "num_variations": NUM_VARIATIONS,
            "bars": BARS,
            "tempo": TEMPO,
        },
        "results": [
            {
                "variation": r["variation"],
                "success": r["success"],
                "midi_file": str(r.get("midi_path", "")) if r["success"] else None,
                "analysis": r.get("analysis") if r["success"] else None,
                "error": r.get("error") if not r["success"] else None,
            }
            for r in results
        ],
        "summary": {
            "total": NUM_VARIATIONS,
            "successful": successful,
            "failed": failed,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary saved to: {summary_path}\n")

    return successful == NUM_VARIATIONS


if __name__ == "__main__":
    # Run async main
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
