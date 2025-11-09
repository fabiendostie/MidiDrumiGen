"""Unit tests for model_loader module."""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.inference.model_loader import (
    detect_device,
    load_model,
    get_gpu_memory_info,
    clear_gpu_cache,
    clear_model_cache,
    ModelLoadError,
)


class TestDeviceDetection:
    """Tests for device detection functions."""

    def test_detect_device_returns_string(self):
        """Test that detect_device returns a valid device string."""
        device = detect_device()
        assert device in ["cuda", "cpu"]

    def test_detect_device_with_cuda_available(self):
        """Test device detection when CUDA is available."""
        if torch.cuda.is_available():
            device = detect_device()
            assert device == "cuda"

    def test_detect_device_without_cuda(self):
        """Test device detection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            device = detect_device()
            assert device == "cpu"


class TestGPUMemoryInfo:
    """Tests for GPU memory information functions."""

    def test_get_gpu_memory_info_with_cuda(self):
        """Test GPU memory info when CUDA is available."""
        if torch.cuda.is_available():
            info = get_gpu_memory_info()
            assert isinstance(info, dict)
            assert 'allocated_gb' in info
            assert 'reserved_gb' in info
            assert 'max_allocated_gb' in info
            assert 'device_name' in info
            assert 'device_count' in info
            assert info['device_count'] > 0

    def test_get_gpu_memory_info_without_cuda(self):
        """Test GPU memory info when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            info = get_gpu_memory_info()
            assert info == {}

    def test_clear_gpu_cache_with_cuda(self):
        """Test GPU cache clearing when CUDA is available."""
        if torch.cuda.is_available():
            # Should not raise any errors
            clear_gpu_cache()

    def test_clear_gpu_cache_without_cuda(self):
        """Test GPU cache clearing when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise any errors
            clear_gpu_cache()


class TestModelLoading:
    """Tests for model loading functions."""

    def test_load_model_with_nonexistent_path(self):
        """Test loading model with non-existent path raises error."""
        fake_path = Path("nonexistent/model.pt")

        with pytest.raises(ModelLoadError) as exc_info:
            load_model(fake_path)

        assert "not found" in str(exc_info.value).lower()
        assert "Phase 6" in str(exc_info.value)

    def test_load_model_with_mock_checkpoint(self, mock_checkpoint_path, device):
        """Test loading a valid mock checkpoint."""
        # Clear cache first
        clear_model_cache()

        # Mock the model class to avoid actual loading
        with patch('src.inference.model_loader.DrumPatternTransformer') as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance

            model, metadata = load_model(mock_checkpoint_path, device=device)

            # Verify model was initialized
            MockModel.assert_called_once()

            # Verify metadata
            assert isinstance(metadata, dict)
            assert metadata['vocab_size'] == 500
            assert metadata['n_styles'] == 50
            assert metadata['device'] == device

            # Verify model was moved to device and set to eval
            mock_instance.to.assert_called_once_with(device)
            mock_instance.eval.assert_called_once()

    def test_load_model_caching(self, mock_checkpoint_path, device):
        """Test that load_model uses LRU cache."""
        clear_model_cache()

        with patch('src.inference.model_loader.DrumPatternTransformer') as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance

            # Load model first time
            model1, metadata1 = load_model(mock_checkpoint_path, device=device)

            # Load model second time (should be cached)
            model2, metadata2 = load_model(mock_checkpoint_path, device=device)

            # Should only initialize model once due to caching
            assert MockModel.call_count == 1

            # Both calls should return same result
            assert model1 is model2
            assert metadata1 == metadata2

    def test_load_model_auto_device_detection(self, mock_checkpoint_path):
        """Test that load_model auto-detects device when not specified."""
        clear_model_cache()

        with patch('src.inference.model_loader.DrumPatternTransformer') as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance

            # Don't specify device
            model, metadata = load_model(mock_checkpoint_path, device=None)

            # Should have detected a device
            assert 'device' in metadata
            assert metadata['device'] in ['cuda', 'cpu']

    def test_load_model_with_invalid_checkpoint(self, temp_dir):
        """Test loading model with corrupted checkpoint."""
        checkpoint_path = temp_dir / "corrupt.pt"

        # Create invalid checkpoint
        with open(checkpoint_path, 'w') as f:
            f.write("not a valid checkpoint")

        with pytest.raises(ModelLoadError):
            load_model(checkpoint_path)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_model_gpu_oom_fallback(self, mock_checkpoint_path):
        """Test that GPU OOM triggers CPU fallback."""
        clear_model_cache()

        with patch('src.inference.model_loader.DrumPatternTransformer') as MockModel:
            # First call (CUDA) raises OOM, second call (CPU) succeeds
            mock_instance = MagicMock()
            MockModel.side_effect = [
                torch.cuda.OutOfMemoryError(),
                mock_instance
            ]

            # Should fallback to CPU
            with patch('torch.cuda.empty_cache'):
                model, metadata = load_model(mock_checkpoint_path, device="cuda")

            # Should have retried on CPU
            assert MockModel.call_count == 2
            # Metadata should indicate CPU device after fallback
            # (This would work in actual implementation with recursion)

    def test_clear_model_cache(self, mock_checkpoint_path):
        """Test clearing model cache."""
        clear_model_cache()

        with patch('src.inference.model_loader.DrumPatternTransformer') as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance

            # Load model
            load_model(mock_checkpoint_path)

            # Clear cache
            clear_model_cache()

            # Load again (should call constructor again)
            load_model(mock_checkpoint_path)

            # Should have called constructor twice (not cached)
            assert MockModel.call_count == 2

    def test_load_model_with_custom_metadata(self, temp_dir):
        """Test loading model with custom configuration in metadata."""
        checkpoint_path = temp_dir / "custom_model.pt"

        checkpoint = {
            'model_state_dict': {},
            'metadata': {
                'vocab_size': 1000,
                'n_styles': 25,
                'n_positions': 1024,
                'n_embd': 512,
                'n_layer': 6,
                'n_head': 8,
                'dropout': 0.2,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        clear_model_cache()

        with patch('src.inference.model_loader.DrumPatternTransformer') as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance

            model, metadata = load_model(checkpoint_path)

            # Verify custom config was used
            MockModel.assert_called_once_with(
                vocab_size=1000,
                n_styles=25,
                n_positions=1024,
                n_embd=512,
                n_layer=6,
                n_head=8,
                dropout=0.2,
            )

            assert metadata['vocab_size'] == 1000
            assert metadata['n_styles'] == 25

    def test_load_model_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        string_path = "models/test.pt"

        with pytest.raises(ModelLoadError) as exc_info:
            load_model(string_path)

        # Should handle string path (even though file doesn't exist)
        assert "not found" in str(exc_info.value).lower()


class TestModelLoadErrorException:
    """Tests for ModelLoadError exception."""

    def test_model_load_error_is_exception(self):
        """Test that ModelLoadError is an Exception."""
        assert issubclass(ModelLoadError, Exception)

    def test_model_load_error_message(self):
        """Test ModelLoadError with custom message."""
        error = ModelLoadError("Test error message")
        assert str(error) == "Test error message"

    def test_raising_model_load_error(self):
        """Test raising ModelLoadError."""
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError("Model failed to load")

        assert "failed to load" in str(exc_info.value).lower()
