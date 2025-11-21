"""Comprehensive unit tests for generate_pattern_task (Phase 5)."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from celery.exceptions import Retry


class TestGeneratePatternTaskSuccess:
    """Tests for successful pattern generation scenarios."""

    @patch("src.midi.export.export_pattern")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.validate_tempo_for_style")
    @patch("src.models.styles.get_model_path")
    @patch("src.models.styles.get_numeric_style_id")
    @patch("src.models.styles.normalize_style_name")
    @patch("src.inference.model_loader.load_model")
    @patch("src.inference.model_loader.detect_device", return_value="cpu")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_complete_generation_pipeline(
        self,
        mock_mkdir,
        mock_path_exists,
        mock_detect_device,
        mock_load_model,
        mock_normalize_style,
        mock_get_style_id,
        mock_get_model_path,
        mock_validate_tempo,
        mock_generate,
        mock_humanize,
        mock_export,
    ):
        """Test complete end-to-end pattern generation."""
        # Setup mocks
        mock_model = MagicMock()
        mock_normalize_style.return_value = "J Dilla"
        mock_get_style_id.return_value = 0
        mock_get_model_path.return_value = Path("models/checkpoints/j_dilla_v1.pt")
        mock_load_model.return_value = (
            mock_model,
            {"vocab_size": 1000, "device": "cpu", "model_type": "trained"},
        )
        mock_generate.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_humanize.return_value = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
        ]
        mock_export.return_value = Path("output/patterns/test.mid")

        # Import task
        from src.tasks.tasks import generate_pattern_task

        # Call task using .apply() to bypass queue
        result = generate_pattern_task.apply(
            kwargs={
                "producer_style": "J Dilla",
                "bars": 4,
                "tempo": 95,
                "time_signature": (4, 4),
                "humanize": True,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.9,
            }
        ).get()

        # Verify result structure
        assert isinstance(result, dict)
        assert "midi_file" in result
        assert "duration_seconds" in result
        assert "tokens_generated" in result
        assert "notes_count" in result
        assert "style" in result
        assert "bars" in result
        assert "tempo" in result
        assert "device" in result

        # Verify result values
        assert result["style"] == "J Dilla"
        assert result["bars"] == 4
        assert result["tempo"] == 95
        assert result["tokens_generated"] == 5
        assert result["device"] == "cpu"
        assert result["humanized"] is True

        # Verify mocks called correctly
        mock_normalize_style.assert_called_once_with("J Dilla")
        mock_load_model.assert_called_once()
        mock_generate.assert_called_once()
        mock_humanize.assert_called_once()
        mock_export.assert_called_once()

    @patch("src.midi.export.export_pattern")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.validate_tempo_for_style")
    @patch("src.models.styles.get_model_path")
    @patch("src.models.styles.get_numeric_style_id")
    @patch("src.models.styles.normalize_style_name")
    @patch("src.inference.model_loader.load_model")
    @patch("src.inference.model_loader.detect_device", return_value="cpu")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_generation_without_humanization(
        self,
        mock_mkdir,
        mock_path_exists,
        mock_detect_device,
        mock_load_model,
        mock_normalize_style,
        mock_get_style_id,
        mock_get_model_path,
        mock_validate_tempo,
        mock_generate,
        mock_export,
    ):
        """Test pattern generation with humanization disabled."""
        # Setup mocks
        mock_model = MagicMock()
        mock_normalize_style.return_value = "Metro Boomin"
        mock_get_style_id.return_value = 1
        mock_get_model_path.return_value = Path("models/checkpoints/metro_boomin_v1.pt")
        mock_load_model.return_value = (
            mock_model,
            {"vocab_size": 1000, "device": "cpu", "model_type": "trained"},
        )
        mock_generate.return_value = [1, 2, 3]
        mock_export.return_value = Path("output/patterns/test.mid")

        from src.tasks.tasks import generate_pattern_task

        result = generate_pattern_task.apply(
            kwargs={
                "producer_style": "Metro Boomin",
                "bars": 2,
                "tempo": 140,
                "time_signature": (4, 4),
                "humanize": False,
            }
        ).get()

        # Verify humanization was NOT applied
        assert result["humanized"] is False

        # Verify export was called with humanize=False
        export_call_args = mock_export.call_args
        assert export_call_args[1]["humanize"] is False


class TestGeneratePatternTaskGPUFallback:
    """Tests for GPU OOM fallback scenarios."""

    @patch("src.midi.export.export_pattern")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.validate_tempo_for_style")
    @patch("src.models.styles.get_model_path")
    @patch("src.models.styles.get_numeric_style_id")
    @patch("src.models.styles.normalize_style_name")
    @patch("src.inference.model_loader.clear_gpu_cache")
    @patch("src.inference.model_loader.load_model")
    @patch("src.inference.model_loader.detect_device", return_value="cuda")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_gpu_oom_during_model_loading(
        self,
        mock_mkdir,
        mock_path_exists,
        mock_detect_device,
        mock_load_model,
        mock_clear_cache,
        mock_normalize_style,
        mock_get_style_id,
        mock_get_model_path,
        mock_validate_tempo,
        mock_generate,
        mock_humanize,
        mock_export,
    ):
        """Test GPU OOM during model loading triggers CPU fallback."""
        # Setup mocks - first call OOM, second call succeeds on CPU
        mock_model = MagicMock()
        mock_normalize_style.return_value = "J Dilla"
        mock_get_style_id.return_value = 0
        mock_get_model_path.return_value = Path("models/checkpoints/j_dilla_v1.pt")

        # First call: GPU OOM, Second call: Success on CPU
        mock_load_model.side_effect = [
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            (mock_model, {"vocab_size": 1000, "device": "cpu", "model_type": "trained"}),
        ]

        mock_generate.return_value = [1, 2, 3]
        mock_humanize.return_value = []
        mock_export.return_value = Path("output/patterns/test.mid")

        from src.tasks.tasks import generate_pattern_task

        result = generate_pattern_task.apply(
            kwargs={"producer_style": "J Dilla", "bars": 4, "tempo": 95, "time_signature": (4, 4)}
        ).get()

        # Verify GPU cache was cleared
        mock_clear_cache.assert_called()

        # Verify load_model was called twice (GPU fail, CPU success)
        assert mock_load_model.call_count == 2

        # Verify final device is CPU
        assert result["device"] == "cpu"

    @patch("src.midi.export.export_pattern")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.validate_tempo_for_style")
    @patch("src.models.styles.get_model_path")
    @patch("src.models.styles.get_numeric_style_id")
    @patch("src.models.styles.normalize_style_name")
    @patch("src.inference.model_loader.clear_gpu_cache")
    @patch("src.inference.model_loader.load_model")
    @patch("src.inference.model_loader.detect_device", return_value="cuda")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_gpu_oom_during_generation(
        self,
        mock_mkdir,
        mock_path_exists,
        mock_detect_device,
        mock_load_model,
        mock_clear_cache,
        mock_normalize_style,
        mock_get_style_id,
        mock_get_model_path,
        mock_validate_tempo,
        mock_generate,
        mock_humanize,
        mock_export,
    ):
        """Test GPU OOM during generation triggers CPU fallback."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.to = Mock(return_value=mock_model)
        mock_normalize_style.return_value = "J Dilla"
        mock_get_style_id.return_value = 0
        mock_get_model_path.return_value = Path("models/checkpoints/j_dilla_v1.pt")
        mock_load_model.return_value = (
            mock_model,
            {"vocab_size": 1000, "device": "cuda", "model_type": "trained"},
        )

        # First generate call: GPU OOM, Second: Success on CPU
        mock_generate.side_effect = [torch.cuda.OutOfMemoryError("CUDA out of memory"), [1, 2, 3]]

        mock_humanize.return_value = []
        mock_export.return_value = Path("output/patterns/test.mid")

        from src.tasks.tasks import generate_pattern_task

        result = generate_pattern_task.apply(
            kwargs={"producer_style": "J Dilla", "bars": 4, "tempo": 95, "time_signature": (4, 4)}
        ).get()

        # Verify GPU cache was cleared
        assert mock_clear_cache.call_count >= 1

        # Verify model.to("cpu") was called
        mock_model.to.assert_called_with("cpu")

        # Verify generate was called twice
        assert mock_generate.call_count == 2

        # Verify task completed
        assert "midi_file" in result


class TestGeneratePatternTaskMockModel:
    """Tests for mock model fallback when no trained checkpoint exists."""

    @patch("src.midi.export.export_pattern")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.validate_tempo_for_style")
    @patch("src.models.styles.get_model_path")
    @patch("src.models.styles.get_numeric_style_id")
    @patch("src.models.styles.normalize_style_name")
    @patch("src.inference.mock.MockDrumModel")
    @patch("src.inference.model_loader.detect_device", return_value="cpu")
    @patch("pathlib.Path.exists", return_value=False)  # Model doesn't exist
    @patch("pathlib.Path.mkdir")
    def test_uses_mock_model_when_checkpoint_missing(
        self,
        mock_mkdir,
        mock_path_exists,
        mock_detect_device,
        mock_mock_drum_class,
        mock_normalize_style,
        mock_get_style_id,
        mock_get_model_path,
        mock_validate_tempo,
        mock_generate,
        mock_humanize,
        mock_export,
    ):
        """Test task uses MockDrumModel when real checkpoint doesn't exist."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model_instance.eval = Mock(return_value=mock_model_instance)
        mock_mock_drum_class.return_value = mock_model_instance

        mock_normalize_style.return_value = "J Dilla"
        mock_get_style_id.return_value = 0
        mock_get_model_path.return_value = Path("models/checkpoints/j_dilla_v1.pt")

        mock_generate.return_value = [1, 2, 3, 4]
        mock_humanize.return_value = []
        mock_export.return_value = Path("output/patterns/test.mid")

        from src.tasks.tasks import generate_pattern_task

        result = generate_pattern_task.apply(
            kwargs={"producer_style": "J Dilla", "bars": 4, "tempo": 95, "time_signature": (4, 4)}
        ).get()

        # Verify MockDrumModel was instantiated
        mock_mock_drum_class.assert_called_once()

        # Verify model was moved to device and set to eval
        mock_model_instance.to.assert_called()
        mock_model_instance.eval.assert_called()

        # Verify task completed with mock model
        assert result["model_type"] == "mock"
        assert "midi_file" in result


class TestGeneratePatternTaskValidation:
    """Tests for input validation and error handling."""

    @patch("src.models.styles.normalize_style_name")
    def test_invalid_style_raises_error(self, mock_normalize_style):
        """Test invalid producer style raises ValueError."""
        from src.models.styles import StyleNotFoundError

        # Mock raises StyleNotFoundError
        mock_normalize_style.side_effect = StyleNotFoundError("Style 'NonExistent' not found")

        from src.tasks.tasks import generate_pattern_task

        # Task will retry, so we catch the Retry exception
        with pytest.raises((ValueError, Retry)):
            generate_pattern_task.apply(
                kwargs={
                    "producer_style": "NonExistent",
                    "bars": 4,
                    "tempo": 120,
                    "time_signature": (4, 4),
                },
                throw=True,  # Raise exceptions instead of returning them
            ).get()

    @patch("src.midi.export.export_pattern")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.validate_tempo_for_style")
    @patch("src.models.styles.get_model_path")
    @patch("src.models.styles.get_numeric_style_id")
    @patch("src.models.styles.normalize_style_name")
    @patch("src.inference.model_loader.load_model")
    @patch("src.inference.model_loader.detect_device", return_value="cpu")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_creates_output_directory(
        self,
        mock_mkdir,
        mock_path_exists,
        mock_detect_device,
        mock_load_model,
        mock_normalize_style,
        mock_get_style_id,
        mock_get_model_path,
        mock_validate_tempo,
        mock_generate,
        mock_humanize,
        mock_export,
    ):
        """Test task creates output directory if it doesn't exist."""
        # Setup mocks
        mock_model = MagicMock()
        mock_normalize_style.return_value = "J Dilla"
        mock_get_style_id.return_value = 0
        mock_get_model_path.return_value = Path("models/checkpoints/j_dilla_v1.pt")
        mock_load_model.return_value = (
            mock_model,
            {"vocab_size": 1000, "device": "cpu", "model_type": "trained"},
        )
        mock_generate.return_value = [1, 2, 3]
        mock_humanize.return_value = []
        mock_export.return_value = Path("output/patterns/test.mid")

        from src.tasks.tasks import generate_pattern_task

        generate_pattern_task.apply(
            kwargs={"producer_style": "J Dilla", "bars": 4, "tempo": 95, "time_signature": (4, 4)}
        ).get()

        # Verify mkdir was called with correct arguments
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
