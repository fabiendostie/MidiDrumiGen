"""Test suite for Phase 3: Model Loading and Inference.

This script tests:
1. Device detection (CUDA/CPU)
2. Style registry operations
3. Model path resolution
4. Error handling for missing models
5. Mock model generation
"""

import logging
import sys
from pathlib import Path

import torch

try:
    from src.inference.mock import MockDrumModel, create_mock_checkpoint, get_mock_tokens
    from src.inference.model_loader import (
        ModelLoadError,
        clear_gpu_cache,
        detect_device,
        get_gpu_memory_info,
        load_model,
    )
    from src.models.styles import (
        StyleNotFoundError,
        get_humanization_params,
        get_model_path,
        get_preferred_tempo_range,
        get_style_id,
        get_style_params,
        list_available_styles,
        normalize_style_name,
        validate_tempo_for_style,
    )
except ModuleNotFoundError:
    fallback_root = Path(__file__).parent.parent
    fallback_root_str = str(fallback_root)
    if fallback_root_str not in sys.path:
        sys.path.insert(0, fallback_root_str)
    from src.inference.mock import MockDrumModel, create_mock_checkpoint, get_mock_tokens
    from src.inference.model_loader import (
        ModelLoadError,
        clear_gpu_cache,
        detect_device,
        get_gpu_memory_info,
        load_model,
    )
    from src.models.styles import (
        StyleNotFoundError,
        get_humanization_params,
        get_model_path,
        get_preferred_tempo_range,
        get_style_id,
        get_style_params,
        list_available_styles,
        normalize_style_name,
        validate_tempo_for_style,
    )

PROJECT_ROOT = Path(__file__).parent.parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_test_header(test_name: str):
    """Print formatted test header."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")


def print_result(test_name: str, passed: bool, message: str = ""):
    """Print test result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {test_name}")
    if message:
        print(f"      {message}")


def test_device_detection():
    """Test CUDA/CPU device detection."""
    print_test_header("Device Detection")

    try:
        device = detect_device()
        print(f"Detected device: {device}")

        if device == "cuda":
            print("CUDA available: True")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"Device count: {torch.cuda.device_count()}")

            # Test GPU memory info
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                print("GPU Memory Info:")
                print(f"  - Device: {gpu_info['device_name']}")
                print(f"  - Allocated: {gpu_info['allocated_gb']:.2f}GB")
                print(f"  - Reserved: {gpu_info['reserved_gb']:.2f}GB")
        else:
            print("CUDA available: False")

        print_result("Device detection", True, f"Device: {device}")
        return True

    except Exception as e:
        print_result("Device detection", False, str(e))
        return False


def test_style_registry():
    """Test style parameter retrieval."""
    print_test_header("Style Registry")

    all_passed = True

    try:
        # Test listing available styles
        styles = list_available_styles()
        print(f"Available styles: {styles}")
        print_result("List available styles", True, f"Found {len(styles)} styles")

        # Test each style
        for style in styles:
            try:
                # Get style ID
                style_id = get_style_id(style)
                print(f"\n{style}:")
                print(f"  - Model ID: {style_id}")

                # Get full parameters
                params = get_style_params(style)
                print(f"  - Description: {params['description']}")

                # Get humanization params
                humanization = get_humanization_params(style)
                print("  - Humanization:")
                print(f"    - Swing: {humanization['swing']}")
                print(f"    - Micro-timing: {humanization['micro_timing_ms']}ms")
                print(f"    - Ghost note prob: {humanization['ghost_note_prob']}")
                print(f"    - Velocity variation: {humanization['velocity_variation']}")

                # Get tempo range
                min_bpm, max_bpm = get_preferred_tempo_range(style)
                print(f"  - Preferred tempo: {min_bpm}-{max_bpm} BPM")

                print_result(f"Style: {style}", True)

            except Exception as e:
                print_result(f"Style: {style}", False, str(e))
                all_passed = False

        # Test style name normalization
        print("\nTesting style name normalization:")
        test_names = ["j dilla", "J Dilla", "jdilla", "metro", "questlove"]
        for test_name in test_names:
            try:
                normalized = normalize_style_name(test_name)
                print(f"  '{test_name}' -> '{normalized}'")
            except StyleNotFoundError:
                print(f"  '{test_name}' -> NOT FOUND")

        return all_passed

    except Exception as e:
        print_result("Style registry", False, str(e))
        return False


def test_model_path_resolution():
    """Test model path resolution for each style."""
    print_test_header("Model Path Resolution")

    all_passed = True

    try:
        styles = list_available_styles()

        for style in styles:
            try:
                model_path = get_model_path(style)
                print(f"{style}: {model_path}")

                # Check if path is absolute
                if not model_path.is_absolute():
                    print_result(
                        f"Path for {style}",
                        False,
                        "Path is not absolute"
                    )
                    all_passed = False
                else:
                    print_result(f"Path for {style}", True)

            except Exception as e:
                print_result(f"Path for {style}", False, str(e))
                all_passed = False

        return all_passed

    except Exception as e:
        print_result("Model path resolution", False, str(e))
        return False


def test_missing_model_handling():
    """Test graceful handling of missing model files."""
    print_test_header("Missing Model Handling")

    try:
        # Try to load non-existent model
        fake_path = Path("models/checkpoints/nonexistent_model.pt")

        try:
            model, metadata = load_model(fake_path)
            print_result(
                "Missing model error",
                False,
                "Should have raised ModelLoadError"
            )
            return False

        except ModelLoadError as e:
            print("Correctly raised ModelLoadError:")
            print(f"  {str(e)}")
            print_result("Missing model error", True, "Error handled correctly")
            return True

    except Exception as e:
        print_result("Missing model handling", False, str(e))
        return False


def test_mock_model():
    """Test mock model generation."""
    print_test_header("Mock Model Generation")

    all_passed = True

    try:
        # Create mock model
        print("Creating mock model...")
        model = MockDrumModel(vocab_size=500, n_styles=50)
        print(f"Mock model created: vocab_size={model.vocab_size}")

        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        style_ids = torch.tensor([0, 1])

        outputs = model(input_ids, style_ids)
        print(f"Forward pass output shape: {outputs.logits.shape}")

        expected_shape = (batch_size, seq_len, 500)
        if outputs.logits.shape == expected_shape:
            print_result("Forward pass", True, f"Output shape correct: {outputs.logits.shape}")
        else:
            print_result("Forward pass", False, f"Expected {expected_shape}, got {outputs.logits.shape}")
            all_passed = False

        # Test generation
        print("\nTesting pattern generation...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        generated = model.generate(style_id=0, max_length=512, device=device)
        print(f"Generated pattern: {len(generated)} tokens")
        print(f"First 20 tokens: {generated[:20].tolist()}")

        if len(generated) > 0:
            print_result("Pattern generation", True, f"Generated {len(generated)} tokens")
        else:
            print_result("Pattern generation", False, "No tokens generated")
            all_passed = False

        # Test mock tokens helper
        print("\nTesting mock tokens helper...")
        mock_tokens = get_mock_tokens(num_bars=4)
        print(f"Mock tokens: {len(mock_tokens)} tokens")
        print_result("Mock tokens helper", True, f"{len(mock_tokens)} tokens")

        # Test mock checkpoint creation
        print("\nTesting mock checkpoint creation...")
        checkpoint_path = PROJECT_ROOT / "output" / "test_mock_checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        create_mock_checkpoint(str(checkpoint_path), vocab_size=500, n_styles=50)

        if checkpoint_path.exists():
            print(f"Checkpoint created: {checkpoint_path}")
            print(f"File size: {checkpoint_path.stat().st_size / 1024:.1f} KB")

            # Try loading it
            print("Attempting to load mock checkpoint...")
            try:
                checkpoint = torch.load(checkpoint_path)
                print("Checkpoint loaded successfully")
                print(f"Metadata: {checkpoint['metadata']}")
                print_result("Mock checkpoint", True, "Created and loaded successfully")
            except Exception as e:
                print_result("Mock checkpoint", False, f"Failed to load: {e}")
                all_passed = False
        else:
            print_result("Mock checkpoint", False, "File not created")
            all_passed = False

        return all_passed

    except Exception as e:
        print_result("Mock model generation", False, str(e))
        return False


def test_tempo_validation():
    """Test tempo validation for styles."""
    print_test_header("Tempo Validation")

    all_passed = True

    try:
        # Test J Dilla (85-95 BPM)
        test_cases = [
            ("J Dilla", 90, True),   # In range
            ("J Dilla", 120, False), # Out of range
            ("Metro Boomin", 140, True),  # In range
            ("Metro Boomin", 90, False),  # Out of range
        ]

        for style, tempo, expected in test_cases:
            result = validate_tempo_for_style(style, tempo, warn_only=True)
            status = "VALID" if result else "INVALID"
            expected_status = "VALID" if expected else "INVALID"

            print(f"{style} @ {tempo} BPM: {status} (expected: {expected_status})")

            if result == expected:
                print_result(f"{style} @ {tempo}BPM", True)
            else:
                print_result(f"{style} @ {tempo}BPM", False, f"Expected {expected_status}, got {status}")
                all_passed = False

        return all_passed

    except Exception as e:
        print_result("Tempo validation", False, str(e))
        return False


def test_gpu_cache_clearing():
    """Test GPU cache clearing."""
    print_test_header("GPU Cache Clearing")

    try:
        if torch.cuda.is_available():
            # Get initial memory
            initial_memory = torch.cuda.memory_allocated()
            print(f"Initial GPU memory: {initial_memory / 1e9:.2f}GB")

            # Allocate some tensors
            tensors = [torch.randn(1000, 1000, device="cuda") for _ in range(10)]
            after_alloc = torch.cuda.memory_allocated()
            print(f"After allocation: {after_alloc / 1e9:.2f}GB")

            # Delete tensors and clear cache
            del tensors
            clear_gpu_cache()

            after_clear = torch.cuda.memory_allocated()
            print(f"After cache clear: {after_clear / 1e9:.2f}GB")

            print_result("GPU cache clearing", True, "Cache cleared successfully")
            return True
        else:
            print("No CUDA device, skipping GPU cache test")
            print_result("GPU cache clearing", True, "Skipped (no CUDA)")
            return True

    except Exception as e:
        print_result("GPU cache clearing", False, str(e))
        return False


def run_all_tests():
    """Run all Phase 3 tests."""
    print("\n" + "="*70)
    print("PHASE 3 TEST SUITE: Model Loading and Inference")
    print("="*70)

    results = {}

    # Run all tests
    results["Device Detection"] = test_device_detection()
    results["Style Registry"] = test_style_registry()
    results["Model Path Resolution"] = test_model_path_resolution()
    results["Missing Model Handling"] = test_missing_model_handling()
    results["Mock Model"] = test_mock_model()
    results["Tempo Validation"] = test_tempo_validation()
    results["GPU Cache Clearing"] = test_gpu_cache_clearing()

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All Phase 3 tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
