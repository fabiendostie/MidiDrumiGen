"""Unit tests for Celery tasks (Phase 4)."""

from unittest.mock import MagicMock, Mock, patch

import mido
import pytest


class TestTokenizeMidiTask:
    """Tests for tokenize_midi task."""

    @patch("src.midi.io.read_midi_file")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("torch.save")
    @patch("builtins.open", new_callable=MagicMock)
    def test_tokenize_midi_completes_successfully(
        self, mock_open, mock_torch_save, mock_mkdir, mock_exists, mock_read_midi_file
    ):
        """Test tokenize_midi task completes successfully."""
        # Setup mock MidiFile
        mock_mid = MagicMock()
        mock_mid.tracks = [MagicMock()]
        mock_mid.tracks[0].messages = [
            MagicMock(type="set_tempo", tempo=mido.bpm2tempo(120)),
            MagicMock(type="time_signature", numerator=4, denominator=4),
        ]
        mock_mid.tracks[0].__iter__.return_value = [
            MagicMock(type="note_on", velocity=100),
            MagicMock(type="note_off", velocity=0),
        ]
        mock_read_midi_file.return_value = mock_mid

        # Import and call task

        from src.tasks.tasks import tokenize_midi

        mock_self = Mock()
        mock_self.request.id = "test-tokenize-id"
        mock_self.update_state = Mock()

        midi_path = "test_midi.mid"
        output_dir = "output/test_tokens"
        style_name = "Test Style"

        result = tokenize_midi(mock_self, midi_path, output_dir=output_dir, style_name=style_name)

        # Assertions
        assert "token_file" in result
        assert "num_tokens" in result
        assert result["num_tokens"] > 0
        assert "midi_file" in result
        assert result["style"] == style_name
        mock_torch_save.assert_called_once()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_self.update_state.assert_called()

    @patch("pathlib.Path.exists", return_value=False)
    def test_tokenize_midi_raises_file_not_found(self, mock_exists):
        """Test tokenize_midi raises FileNotFoundError for non-existent file."""
        from src.tasks.tasks import tokenize_midi

        mock_self = Mock()
        mock_self.request.id = "test-tokenize-id"
        mock_self.update_state = Mock()

        midi_path = "non_existent.mid"

        with pytest.raises(FileNotFoundError):
            tokenize_midi(mock_self, midi_path)

        mock_self.update_state.assert_called_with(
            state="FAILURE",
            meta=pytest.approx(
                {
                    "error": f"MIDI file not found: {midi_path}",
                    "error_type": "FileNotFoundError",
                    "status": "Failed - file not found",
                }
            ),
        )

    @patch("src.midi.io.read_midi_file")
    @patch("pathlib.Path.exists", return_value=True)
    def test_tokenize_midi_raises_value_error_for_empty_midi(
        self, mock_exists, mock_read_midi_file
    ):
        """Test tokenize_midi raises ValueError for MIDI file with no notes."""
        mock_mid = MagicMock()
        mock_mid.tracks = [MagicMock()]
        mock_mid.tracks[0].__iter__.return_value = []  # No notes
        mock_read_midi_file.return_value = mock_mid

        from src.tasks.tasks import tokenize_midi

        mock_self = Mock()
        mock_self.request.id = "test-tokenize-id"
        mock_self.update_state = Mock()

        midi_path = "empty.mid"

        with pytest.raises(ValueError, match="MIDI file contains no notes"):
            tokenize_midi(mock_self, midi_path)

        mock_self.update_state.assert_called_with(
            state="FAILURE",
            meta=pytest.approx(
                {
                    "error": "MIDI file contains no notes",
                    "error_type": "ValueError",
                    "status": "Failed - invalid MIDI",
                }
            ),
        )

    @patch("src.midi.io.read_midi_file")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("torch.save")
    @patch("builtins.open", new_callable=MagicMock)
    def test_tokenize_midi_updates_progress(
        self, mock_open, mock_torch_save, mock_mkdir, mock_exists, mock_read_midi_file
    ):
        """Test tokenize_midi updates progress state during execution."""
        mock_mid = MagicMock()
        mock_mid.tracks = [MagicMock()]
        mock_mid.tracks[0].messages = [
            MagicMock(type="set_tempo", tempo=mido.bpm2tempo(120)),
            MagicMock(type="time_signature", numerator=4, denominator=4),
        ]
        mock_mid.tracks[0].__iter__.return_value = [
            MagicMock(type="note_on", velocity=100),
            MagicMock(type="note_off", velocity=0),
        ]
        mock_read_midi_file.return_value = mock_mid

        from src.tasks.tasks import tokenize_midi

        mock_self = Mock()
        mock_self.request.id = "test-tokenize-id"
        mock_self.update_state = Mock()

        tokenize_midi(mock_self, "test.mid")

        # Verify update_state was called multiple times
        assert mock_self.update_state.call_count >= 3

        # Verify progress states
        calls = [call_args[1] for call_args in mock_self.update_state.call_args_list]
        progress_values = [c.get("meta", {}).get("progress") for c in calls]
        assert 10 in progress_values  # Loading MIDI file
        assert 25 in progress_values  # Extracting metadata
        assert 50 in progress_values  # Tokenizing MIDI


class TestGeneratePatternTask:
    @patch("src.midi.export.export_pattern")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.get_style_id")
    @patch("src.inference.model_loader.load_model")
    def test_task_completes_successfully(
        self, mock_load_model, mock_get_style_id, mock_generate, mock_humanize, mock_export
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
            humanize=True,
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

    @patch("src.midi.export.export_pattern_to_midi")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.get_style_id")
    @patch("src.inference.mock.MockDrumModel")
    @patch("src.inference.model_loader.load_model")
    def test_task_falls_back_to_mock_model(
        self,
        mock_load_model,
        mock_drum_model_class,
        mock_get_style_id,
        mock_generate,
        mock_humanize,
        mock_export,
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
            mock_self, producer_style="J Dilla", bars=4, tempo=95, time_signature=(4, 4)
        )

        # Verify mock model was used
        mock_drum_model_class.assert_called()

        # Verify task still completed
        assert "midi_file" in result

    @patch("src.midi.export.export_pattern_to_midi")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.get_style_id")
    @patch("src.inference.model_loader.load_model")
    def test_task_updates_progress(
        self, mock_load_model, mock_get_style_id, mock_generate, mock_humanize, mock_export
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
            mock_self, producer_style="J Dilla", bars=4, tempo=95, time_signature=(4, 4)
        )

        # Verify update_state was called multiple times
        assert mock_self.update_state.call_count >= 4

        # Verify progress states
        calls = [call_args[1] for call_args in mock_self.update_state.call_args_list]
        progress_values = [c.get("meta", {}).get("progress") for c in calls]
        assert 10 in progress_values  # Starting
        assert 25 in progress_values  # Loading model
        assert 50 in progress_values  # Generating pattern

    @patch("src.midi.export.export_pattern_to_midi")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.get_style_id")
    @patch("src.inference.model_loader.load_model")
    def test_task_applies_humanization_when_enabled(
        self, mock_load_model, mock_get_style_id, mock_generate, mock_humanize, mock_export
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
            humanize=True,
        )

        # Verify humanization was called
        mock_humanize.assert_called_once()

    @patch("src.midi.export.export_pattern_to_midi")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.get_style_id")
    @patch("src.inference.model_loader.load_model")
    def test_task_skips_humanization_when_disabled(
        self, mock_load_model, mock_get_style_id, mock_generate, mock_humanize, mock_export
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
            humanize=False,
        )

        # Verify humanization was NOT called
        mock_humanize.assert_not_called()

    @patch("src.midi.export.export_pattern_to_midi")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.get_style_id")
    @patch("src.inference.model_loader.load_model")
    def test_task_creates_output_directory(
        self, mock_load_model, mock_get_style_id, mock_generate, mock_humanize, mock_export
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

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            generate_pattern_task(
                mock_self, producer_style="J Dilla", bars=4, tempo=95, time_signature=(4, 4)
            )

            # Verify mkdir was called with correct arguments
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("src.midi.export.export_pattern_to_midi")
    @patch("src.midi.humanize.apply_style_humanization")
    @patch("src.inference.generate.generate_pattern")
    @patch("src.models.styles.get_style_id")
    @patch("src.inference.model_loader.load_model")
    def test_task_raises_on_export_failure(
        self, mock_load_model, mock_get_style_id, mock_generate, mock_humanize, mock_export
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

        with pytest.raises(RuntimeError):
            generate_pattern_task(
                mock_self, producer_style="J Dilla", bars=4, tempo=95, time_signature=(4, 4)
            )
