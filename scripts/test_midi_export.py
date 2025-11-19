"""Test script for MIDI export pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.midi.export import export_pattern
from src.midi.humanize import apply_style_humanization
from src.midi.validate import get_pattern_statistics, validate_drum_pattern


def create_sample_pattern() -> list:
    """Create a simple 4-bar drum pattern."""
    # 4/4 time, 480 ticks per beat, 4 bars = 16 beats
    ticks_per_beat = 480

    notes = []

    # Simple drum pattern: kick, snare, hi-hat
    for bar in range(4):
        bar_offset = bar * 4 * ticks_per_beat

        # Kicks on beats 1 and 3
        notes.append({
            'pitch': 36,  # Kick
            'velocity': 100,
            'time': bar_offset
        })
        notes.append({
            'pitch': 36,  # Kick
            'velocity': 95,
            'time': bar_offset + 2 * ticks_per_beat
        })

        # Snare on beats 2 and 4
        notes.append({
            'pitch': 38,  # Snare
            'velocity': 90,
            'time': bar_offset + 1 * ticks_per_beat
        })
        notes.append({
            'pitch': 38,  # Snare
            'velocity': 92,
            'time': bar_offset + 3 * ticks_per_beat
        })

        # Hi-hats on every 8th note
        for eighth in range(8):
            notes.append({
                'pitch': 42,  # Closed hi-hat
                'velocity': 70 + (eighth % 2) * 10,  # Accent on downbeats
                'time': bar_offset + eighth * (ticks_per_beat // 2)
            })

    return notes


def test_validation():
    """Test pattern validation."""
    print("\n" + "="*60)
    print("TEST 1: Pattern Validation")
    print("="*60)

    # Create sample pattern
    notes = create_sample_pattern()
    print(f"\nCreated pattern with {len(notes)} notes")

    # Validate
    is_valid, errors = validate_drum_pattern(notes)

    if is_valid:
        print("[PASS] Pattern validation PASSED")
    else:
        print("[FAIL] Pattern validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Get statistics
    stats = get_pattern_statistics(notes)
    print("\nPattern Statistics:")
    print(f"  Total notes: {stats['total_notes']}")
    print(f"  Unique pitches: {stats['unique_pitches']}")
    print(f"  Duration: {stats['duration_beats']:.1f} beats")
    print(f"  Density: {stats['density']:.2f} notes/beat")
    print(f"  Velocity range: {stats['velocity_range']}")

    return True


def test_humanization():
    """Test humanization algorithms."""
    print("\n" + "="*60)
    print("TEST 2: Humanization")
    print("="*60)

    notes = create_sample_pattern()
    print(f"\nOriginal pattern: {len(notes)} notes")

    # Test different styles
    styles = ["J Dilla", "Metro Boomin", "Questlove"]

    for style in styles:
        humanized = apply_style_humanization(notes, style, tempo=95)
        print(f"\n{style} humanization: {len(humanized)} notes")
        print("  (Ghost notes may have been added)")

    print("\n[PASS] Humanization test PASSED")
    return True


def test_export():
    """Test MIDI export."""
    print("\n" + "="*60)
    print("TEST 3: MIDI Export")
    print("="*60)

    # Create output directory
    output_dir = Path("output/test_patterns")
    output_dir.mkdir(parents=True, exist_ok=True)

    notes = create_sample_pattern()

    # Test 1: Export without humanization
    print("\nExporting pattern without humanization...")
    output_path1 = output_dir / "test_pattern_no_humanize.mid"
    try:
        result_path1 = export_pattern(
            notes,
            output_path1,
            tempo=95,
            time_signature=(4, 4),
            humanize=False,
            style_name="Test"
        )
        print(f"[PASS] Exported to: {result_path1}")
        print(f"       File size: {result_path1.stat().st_size} bytes")
    except Exception as e:
        print(f"[FAIL] Export failed: {e}")
        return False

    # Test 2: Export with J Dilla style humanization
    print("\nExporting pattern with J Dilla humanization...")
    output_path2 = output_dir / "test_pattern_j_dilla.mid"
    try:
        result_path2 = export_pattern(
            notes,
            output_path2,
            tempo=95,
            time_signature=(4, 4),
            humanize=True,
            style_name="J Dilla"
        )
        print(f"[PASS] Exported to: {result_path2}")
        print(f"       File size: {result_path2.stat().st_size} bytes")
    except Exception as e:
        print(f"[FAIL] Export failed: {e}")
        return False

    # Test 3: Export with Metro Boomin style
    print("\nExporting pattern with Metro Boomin style...")
    output_path3 = output_dir / "test_pattern_metro_boomin.mid"
    try:
        result_path3 = export_pattern(
            notes,
            output_path3,
            tempo=140,
            time_signature=(4, 4),
            humanize=True,
            style_name="Metro Boomin"
        )
        print(f"[PASS] Exported to: {result_path3}")
        print(f"       File size: {result_path3.stat().st_size} bytes")
    except Exception as e:
        print(f"[FAIL] Export failed: {e}")
        return False

    print("\n[PASS] All exports successful!")
    return True


def test_invalid_patterns():
    """Test validation with invalid patterns."""
    print("\n" + "="*60)
    print("TEST 4: Invalid Pattern Detection")
    print("="*60)

    # Test 1: Invalid note range
    print("\nTest 4.1: Invalid note range")
    invalid_notes1 = [{'pitch': 200, 'velocity': 100, 'time': 0}]
    is_valid, errors = validate_drum_pattern(invalid_notes1)
    if not is_valid:
        print(f"[PASS] Correctly detected invalid note: {errors[0]}")
    else:
        print("[FAIL] Failed to detect invalid note")
        return False

    # Test 2: Invalid velocity
    print("\nTest 4.2: Invalid velocity")
    invalid_notes2 = [{'pitch': 36, 'velocity': 200, 'time': 0}]
    is_valid, errors = validate_drum_pattern(invalid_notes2)
    if not is_valid:
        print(f"[PASS] Correctly detected invalid velocity: {errors[0]}")
    else:
        print("[FAIL] Failed to detect invalid velocity")
        return False

    # Test 3: Impossible simultaneous hits
    print("\nTest 4.3: Impossible simultaneous hits (closed + open hi-hat)")
    invalid_notes3 = [
        {'pitch': 42, 'velocity': 100, 'time': 0},  # Closed hi-hat
        {'pitch': 46, 'velocity': 100, 'time': 0},  # Open hi-hat (same time)
    ]
    is_valid, errors = validate_drum_pattern(invalid_notes3)
    if not is_valid:
        print(f"[PASS] Correctly detected impossible hits: {errors[0]}")
    else:
        print("[FAIL] Failed to detect impossible simultaneous hits")
        return False

    print("\n[PASS] Invalid pattern detection PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MIDI EXPORT PIPELINE TEST SUITE")
    print("="*60)

    tests = [
        ("Validation", test_validation),
        ("Humanization", test_humanization),
        ("Export", test_export),
        ("Invalid Patterns", test_invalid_patterns),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] {test_name} test CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n*** ALL TESTS PASSED! Phase 2 Complete! ***")
        print("\nGenerated MIDI files are in: output/test_patterns/")
        print("You can now load these files in a DAW (Ableton, FL Studio, etc.)")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
