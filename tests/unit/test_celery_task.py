"""Unit tests for Celery tasks (Phase 4)."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path


class TestGeneratePatternTask:
    """Tests for generate_pattern_task."""

    @patch('src.midi.export.export_pattern_to_midi')
    @patch('src.midi.humanize.apply_style_humanization')
    @patch('src.inference.generate.generate_pattern')
    @patch('src.models.styles.get_style_id')
    @patch('src.inference.model_loader.load_model')
    def test_task_completes_successfully(
        self,
        mock_load_model,
        mock_get_style_id,
        mock_generate,
        mock_humanize,
        mock_export
    ):
        """Test task completes successfully with all steps."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = (mock_model, {"vocab_size": 1000})
        mock_get_style_id.return_value = 0
        mock_generate.return_value = [1, 2, 3, 4]  # Mock tokens
        mock_humanize.return_value = []  # Mock humanized notes
        mock_export.return_value = True

        # Import and call task
        from src.tasks.tasks import generate_pattern_task

        # Create mock task context
        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()

        result = generate_pattern_task(
            mock_self,
            producer_style="J Dilla",
            bars=4,
            tempo=95,
            time_signature=(4, 4),
            humanize=True
        )

        # Verify result structure
        assert "midi_file" in result
        assert "duration_seconds" in result
        assert "tokens_generated" in result
        assert "style" in result
        assert "bars" in result
        assert "tempo" in result

        # Verify result values
        assert result["style"] == "J Dilla"
        assert result["bars"] == 4
        assert result["tempo"] == 95

    @patch('src.midi.export.export_pattern_to_midi')
    @patch('src.midi.humanize.apply_style_humanization')
    @patch('src.inference.generate.generate_pattern')
    @patch('src.models.styles.get_style_id')
    @patch('src.inference.mock.MockDrumModel')
    @patch('src.inference.model_loader.load_model')
    def test_task_falls_back_to_mock_model(
        self,
        mock_load_model,
        mock_drum_model_class,
        mock_get_style_id,
        mock_generate,
        mock_humanize,
        mock_export
    ):
        """Test task falls back to mock model when real model not found."""
        # Setup mocks to simulate model not found
        mock_load_model.side_effect = FileNotFoundError("Model not found")
        mock_mock_model = Mock()
        mock_drum_model_class.return_value = mock_mock_model
        mock_get_style_id.return_value = 0
        mock_generate.return_value = [1, 2, 3, 4]
        mock_humanize.return_value = []
        mock_export.return_value = True

        # Import and call task
        from src.tasks.tasks import generate_pattern_task

        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()

        result = generate_pattern_task(
            mock_self,
            producer_style="J Dilla",
            bars=4,
            tempo=95,
            time_signature=(4, 4)
        )

        # Verify mock model was used
        mock_drum_model_class.assert_called()

        # Verify task still completed
        assert "midi_file" in result

    @patch('src.midi.export.export_pattern_to_midi')
    @patch('src.midi.humanize.apply_style_humanization')
    @patch('src.inference.generate.generate_pattern')
    @patch('src.models.styles.get_style_id')
    @patch('src.inference.model_loader.load_model')
    def test_task_updates_progress(
        self,
        mock_load_model,
        mock_get_style_id,
        mock_generate,
        mock_humanize,
        mock_export
    ):
        """Test task updates progress state during execution."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = (mock_model, {"vocab_size": 1000})
        mock_get_style_id.return_value = 0
        mock_generate.return_value = [1, 2, 3, 4]
        mock_humanize.return_value = []
        mock_export.return_value = True

        # Import and call task
        from src.tasks.tasks import generate_pattern_task

        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()

        generate_pattern_task(
            mock_self,
            producer_style="J Dilla",
            bars=4,
            tempo=95,
            time_signature=(4, 4)
        )

        # Verify update_state was called multiple times
        assert mock_self.update_state.call_count >= 4

        # Verify progress states
        calls = [call_args[1] for call_args in mock_self.update_state.call_args_list]
        progress_values = [c.get('meta', {}).get('progress') for c in calls]
        assert 10 in progress_values  # Starting
        assert 25 in progress_values  # Loading model
        assert 50 in progress_values  # Generating pattern

    @patch('src.midi.export.export_pattern_to_midi')
    @patch('src.midi.humanize.apply_style_humanization')
    @patch('src.inference.generate.generate_pattern')
    @patch('src.models.styles.get_style_id')
    @patch('src.inference.model_loader.load_model')
    def test_task_applies_humanization_when_enabled(
        self,
        mock_load_model,
        mock_get_style_id,
        mock_generate,
        mock_humanize,
        mock_export
    ):
        """Test task applies humanization when humanize=True."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = (mock_model, {"vocab_size": 1000})
        mock_get_style_id.return_value = 0
        mock_generate.return_value = [1, 2, 3, 4]
        mock_humanize.return_value = []
        mock_export.return_value = True

        # Import and call task
        from src.tasks.tasks import generate_pattern_task

        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()

        generate_pattern_task(
            mock_self,
            producer_style="J Dilla",
            bars=4,
            tempo=95,
            time_signature=(4, 4),
            humanize=True
        )

        # Verify humanization was called
        mock_humanize.assert_called_once()

    @patch('src.midi.export.export_pattern_to_midi')
    @patch('src.midi.humanize.apply_style_humanization')
    @patch('src.inference.generate.generate_pattern')
    @patch('src.models.styles.get_style_id')
    @patch('src.inference.model_loader.load_model')
    def test_task_skips_humanization_when_disabled(
        self,
        mock_load_model,
        mock_get_style_id,
        mock_generate,
        mock_humanize,
        mock_export
    ):
        """Test task skips humanization when humanize=False."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = (mock_model, {"vocab_size": 1000})
        mock_get_style_id.return_value = 0
        mock_generate.return_value = [1, 2, 3, 4]
        mock_export.return_value = True

        # Import and call task
        from src.tasks.tasks import generate_pattern_task

        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()

        generate_pattern_task(
            mock_self,
            producer_style="J Dilla",
            bars=4,
            tempo=95,
            time_signature=(4, 4),
            humanize=False
        )

        # Verify humanization was NOT called
        mock_humanize.assert_not_called()

    @patch('src.midi.export.export_pattern_to_midi')
    @patch('src.midi.humanize.apply_style_humanization')
    @patch('src.inference.generate.generate_pattern')
    @patch('src.models.styles.get_style_id')
    @patch('src.inference.model_loader.load_model')
    def test_task_creates_output_directory(
        self,
        mock_load_model,
        mock_get_style_id,
        mock_generate,
        mock_humanize,
        mock_export
    ):
        """Test task creates output directory if it doesn't exist."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = (mock_model, {"vocab_size": 1000})
        mock_get_style_id.return_value = 0
        mock_generate.return_value = [1, 2, 3, 4]
        mock_humanize.return_value = []
        mock_export.return_value = True

        # Import and call task
        from src.tasks.tasks import generate_pattern_task

        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()

        with patch('pathlib.Path.mkdir') as mock_mkdir:
            generate_pattern_task(
                mock_self,
                producer_style="J Dilla",
                bars=4,
                tempo=95,
                time_signature=(4, 4)
            )

            # Verify mkdir was called with correct arguments
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('src.midi.export.export_pattern_to_midi')
    @patch('src.midi.humanize.apply_style_humanization')
    @patch('src.inference.generate.generate_pattern')
    @patch('src.models.styles.get_style_id')
    @patch('src.inference.model_loader.load_model')
    def test_task_raises_on_export_failure(
        self,
        mock_load_model,
        mock_get_style_id,
        mock_generate,
        mock_humanize,
        mock_export
    ):
        """Test task raises exception when MIDI export fails."""
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = (mock_model, {"vocab_size": 1000})
        mock_get_style_id.return_value = 0
        mock_generate.return_value = [1, 2, 3, 4]
        mock_humanize.return_value = []
        mock_export.return_value = False  # Export failure

        # Import and call task
        from src.tasks.tasks import generate_pattern_task

        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()

        with pytest.raises(Exception) as exc_info:
            generate_pattern_task(
                mock_self,
                producer_style="J Dilla",
                bars=4,
                tempo=95,
                time_signature=(4, 4)
            )

        assert "MIDI export failed" in str(exc_info.value)
