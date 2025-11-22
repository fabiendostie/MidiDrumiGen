"""
Test script for manual verification of Celery tasks.

This script allows testing Celery tasks without needing the full
API/Redis/Worker setup. It tests the task functions directly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from src.tasks.tasks import generate_pattern_task, tokenize_midi

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_generate_pattern():
    """Test pattern generation task directly."""
    logger.info("=" * 60)
    logger.info("TEST 1: Generate Pattern Task")
    logger.info("=" * 60)

    try:
        # Create a mock task instance
        class MockRequest:
            id = "test-task-123"

        class MockTask:
            request = MockRequest()

            def update_state(self, state, meta):
                logger.info(f"State update: {state} - {meta.get('status', 'N/A')}")

        # Test parameters
        params = {
            "producer_style": "J Dilla",
            "bars": 4,
            "tempo": 95,
            "time_signature": (4, 4),
            "humanize": True,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.9,
        }

        logger.info(f"Testing with params: {params}")

        # Call task function directly (Celery binds `self` automatically)
        result = generate_pattern_task.apply(kwargs=params).get()

        logger.info("✓ Task completed successfully!")
        logger.info(f"Result: {result}")
        logger.info(f"MIDI file: {result.get('midi_file')}")
        logger.info(f"Duration: {result.get('duration_seconds'):.2f}s")
        logger.info(f"Tokens: {result.get('tokens_generated')}")
        logger.info(f"Device: {result.get('device')}")

        # Verify MIDI file was created
        midi_path = Path(result["midi_file"])
        if midi_path.exists():
            logger.info(f"✓ MIDI file exists ({midi_path.stat().st_size} bytes)")
        else:
            logger.error(f"✗ MIDI file not found at {midi_path}")

        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_tokenize_midi():
    """Test MIDI tokenization task directly."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Tokenize MIDI Task")
    logger.info("=" * 60)

    try:
        # First, create a test MIDI file from previous test
        test_midi = Path("output/patterns")
        if not test_midi.exists():
            logger.warning("No MIDI files from previous test, creating mock pattern first...")
            if not test_generate_pattern():
                logger.error("Failed to create test MIDI file")
                return False

        # Find the most recent MIDI file
        midi_files = list(test_midi.glob("*.mid"))
        if not midi_files:
            logger.error("No MIDI files found to tokenize")
            return False

        test_file = max(midi_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using MIDI file: {test_file}")

        # Create a mock task instance
        class MockRequest:
            id = "test-tokenize-456"

        class MockTask:
            request = MockRequest()

            def update_state(self, state, meta):
                logger.info(f"State update: {state} - {meta.get('status', 'N/A')}")

        # Call tokenization task
        result = tokenize_midi.apply(
            kwargs={
                "midi_path": str(test_file),
                "output_dir": "output/test_tokens",
                "style_name": "J Dilla",
            }
        ).get()

        logger.info("✓ Tokenization completed successfully!")
        logger.info(f"Result: {result}")
        logger.info(f"Token file: {result.get('token_file')}")
        logger.info(f"Num tokens: {result.get('num_tokens')}")
        logger.info(f"Duration: {result.get('duration_seconds'):.2f}s")

        # Verify token file was created
        token_path = Path(result["token_file"])
        if token_path.exists():
            logger.info(f"✓ Token file exists ({token_path.stat().st_size} bytes)")
        else:
            logger.error(f"✗ Token file not found at {token_path}")

        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_multiple_styles():
    """Test generating patterns with different styles."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Multiple Producer Styles")
    logger.info("=" * 60)

    styles = ["J Dilla", "Metro Boomin", "Questlove"]
    results = {}

    class MockRequest:
        def __init__(self, task_id):
            self.id = task_id

    class MockTask:
        def __init__(self, task_id):
            self.request = MockRequest(task_id)

        def update_state(self, state, meta):
            pass

    for style in styles:
        try:
            logger.info(f"\nGenerating {style} pattern...")

            result = generate_pattern_task.apply(
                kwargs={
                    "producer_style": style,
                    "bars": 2,  # Shorter for faster testing
                    "tempo": 120,
                    "time_signature": (4, 4),
                    "humanize": True,
                }
            ).get()

            results[style] = result
            logger.info(f"✓ {style}: {result['midi_file']}")

        except Exception as e:
            logger.error(f"✗ {style} failed: {e}")
            results[style] = {"error": str(e)}

    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    for style, result in results.items():
        if "error" in result:
            logger.error(f"  ✗ {style}: {result['error']}")
        else:
            logger.info(
                f"  ✓ {style}: {result['duration_seconds']:.2f}s, "
                f"{result['tokens_generated']} tokens"
            )

    return len([r for r in results.values() if "error" not in r]) == len(styles)


def test_error_handling():
    """Test error handling in tasks."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Error Handling")
    logger.info("=" * 60)

    # Test 1: Invalid style
    logger.info("\nTest 4.1: Invalid producer style")
    try:

        class MockRequest:
            id = "test-error-1"

        class MockTask:
            request = MockRequest()

            def update_state(self, state, meta):
                pass

        # This should fail because the task will raise an exception
        generate_pattern_task.apply(
            kwargs={
                "producer_style": "Invalid Style Name",
                "bars": 4,
                "tempo": 120,
                "time_signature": (4, 4),
            }
        ).get()

        logger.error("✗ Should have raised ValueError for invalid style")
        return False

    except ValueError as e:
        logger.info(f"✓ Correctly raised ValueError: {e}")

    # Test 2: Invalid MIDI file for tokenization
    logger.info("\nTest 4.2: Invalid MIDI file path")
    try:

        class MockRequest:
            id = "test-error-2"

        class MockTask:
            request = MockRequest()

            def update_state(self, state, meta):
                pass

        # This should fail because the task will raise an exception
        tokenize_midi.apply(
            kwargs={"midi_path": "nonexistent_file.mid", "output_dir": "output/test_tokens"}
        ).get()

        logger.error("✗ Should have raised FileNotFoundError")
        return False

    except FileNotFoundError as e:
        logger.info(f"✓ Correctly raised FileNotFoundError: {e}")

    logger.info("\n✓ All error handling tests passed!")
    return True


def main():
    """Run all tests."""
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "Celery Tasks Manual Test Suite" + " " * 17 + "║")
    logger.info("╚" + "=" * 58 + "╝")

    tests = [
        ("Generate Pattern", test_generate_pattern),
        ("Tokenize MIDI", test_tokenize_midi),
        ("Multiple Styles", test_multiple_styles),
        ("Error Handling", test_error_handling),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n\nRunning: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}", exc_info=True)
            results[test_name] = False

    # Print summary
    logger.info("\n\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.error(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
