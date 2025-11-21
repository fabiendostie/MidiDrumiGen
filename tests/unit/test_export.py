"""Unit tests for MIDI export module."""

import contextlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.midi.constants import DEFAULT_TICKS_PER_BEAT
from src.midi.export import (
    detokenize_to_notes,
    export_from_tokens,
    export_pattern,
)


class TestExportPattern:
    """Tests for export_pattern function."""

    def test_export_pattern_basic(self, temp_dir):
        """Test basic pattern export."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
            {"pitch": 42, "velocity": 80, "time": 960},
        ]
        output_path = temp_dir / "test.mid"

        result = export_pattern(notes, output_path, tempo=120, humanize=False)

        assert result == output_path
        assert output_path.exists()

    def test_export_pattern_creates_directory(self, temp_dir):
        """Test that parent directory is created if it doesn't exist."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]
        output_path = temp_dir / "subdir" / "test.mid"

        export_pattern(notes, output_path, humanize=False)

        assert output_path.parent.exists()
        assert output_path.exists()

    def test_export_pattern_with_humanization(self, temp_dir):
        """Test export with humanization enabled."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
        ]
        output_path = temp_dir / "humanized.mid"

        export_pattern(notes, output_path, tempo=95, humanize=True, style_name="J Dilla")

        assert output_path.exists()

    def test_export_pattern_different_tempo(self, temp_dir):
        """Test export with different tempos."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        for tempo in [80, 120, 160]:
            output_path = temp_dir / f"tempo_{tempo}.mid"
            export_pattern(notes, output_path, tempo=tempo, humanize=False)
            assert output_path.exists()

    def test_export_pattern_different_time_signature(self, temp_dir):
        """Test export with different time signatures."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]
        output_path = temp_dir / "3_4.mid"

        export_pattern(notes, output_path, time_signature=(3, 4), humanize=False)

        assert output_path.exists()

    def test_export_pattern_custom_ticks_per_beat(self, temp_dir):
        """Test export with custom MIDI resolution."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]
        output_path = temp_dir / "test.mid"

        export_pattern(notes, output_path, ticks_per_beat=960, humanize=False)

        assert output_path.exists()

    def test_export_pattern_invalid_notes_raises_error(self, temp_dir):
        """Test that invalid pattern raises ValueError."""
        # Invalid velocity (0)
        notes = [{"pitch": 36, "velocity": 0, "time": 0}]
        output_path = temp_dir / "invalid.mid"

        with pytest.raises(ValueError) as exc_info:
            export_pattern(notes, output_path, humanize=False)

        assert "invalid" in str(exc_info.value).lower()

    def test_export_pattern_empty_notes_raises_error(self, temp_dir):
        """Test that empty pattern raises ValueError."""
        notes = []
        output_path = temp_dir / "empty.mid"

        with pytest.raises(ValueError) as exc_info:
            export_pattern(notes, output_path, humanize=False)

        assert "empty" in str(exc_info.value).lower()

    def test_export_pattern_with_all_drum_types(self, temp_dir):
        """Test export with various drum types."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},  # Kick
            {"pitch": 38, "velocity": 90, "time": 240},  # Snare
            {"pitch": 42, "velocity": 70, "time": 480},  # Closed hi-hat
            {"pitch": 46, "velocity": 80, "time": 720},  # Open hi-hat
            {"pitch": 49, "velocity": 95, "time": 960},  # Crash
            {"pitch": 51, "velocity": 75, "time": 1200},  # Ride
        ]
        output_path = temp_dir / "all_drums.mid"

        export_pattern(notes, output_path, humanize=False)

        assert output_path.exists()

    def test_export_pattern_simultaneous_notes(self, temp_dir):
        """Test export with simultaneous notes (kick + hi-hat)."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 42, "velocity": 70, "time": 0},  # Same time as kick
        ]
        output_path = temp_dir / "simultaneous.mid"

        export_pattern(notes, output_path, humanize=False)

        assert output_path.exists()

    def test_export_pattern_different_styles(self, temp_dir):
        """Test export with different producer styles."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        for style in ["J Dilla", "Metro Boomin", "Questlove"]:
            output_path = temp_dir / f"{style.lower().replace(' ', '_')}.mid"
            export_pattern(notes, output_path, humanize=True, style_name=style)
            assert output_path.exists()

    def test_export_pattern_long_pattern(self, temp_dir):
        """Test export with long pattern (many notes)."""
        # Create 100 notes
        notes = [
            {"pitch": 36 + (i % 6), "velocity": 80 + (i % 20), "time": i * 120} for i in range(100)
        ]
        output_path = temp_dir / "long.mid"

        export_pattern(notes, output_path, humanize=False)

        assert output_path.exists()

    def test_export_pattern_note_ordering(self, temp_dir):
        """Test that notes are properly ordered by time."""
        # Provide notes in random order
        notes = [
            {"pitch": 38, "velocity": 90, "time": 480},
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 42, "velocity": 80, "time": 240},
        ]
        output_path = temp_dir / "ordered.mid"

        export_pattern(notes, output_path, humanize=False)

        assert output_path.exists()


class TestDetokenizeToNotes:
    """Tests for detokenize_to_notes function."""

    def test_detokenize_not_implemented(self):
        """Test that function raises NotImplementedError."""
        tokens = [1, 2, 3, 4]
        tokenizer = MagicMock()

        with pytest.raises(NotImplementedError) as exc_info:
            detokenize_to_notes(tokens, tokenizer)

        assert "tokenizer integration pending" in str(exc_info.value).lower()

    def test_detokenize_placeholder_warning(self, caplog):
        """Test that placeholder function logs warning."""
        tokens = [1, 2, 3, 4]
        tokenizer = MagicMock()

        with contextlib.suppress(NotImplementedError):
            detokenize_to_notes(tokens, tokenizer)

        # Check warning was logged
        assert any("placeholder" in record.message.lower() for record in caplog.records)


class TestExportFromTokens:
    """Tests for export_from_tokens function."""

    def test_export_from_tokens_not_implemented(self, temp_dir):
        """Test that function raises NotImplementedError."""
        tokens = [1, 2, 3, 4]
        tokenizer = MagicMock()
        output_path = temp_dir / "test.mid"

        with pytest.raises(NotImplementedError):
            export_from_tokens(tokens, tokenizer, output_path)


class TestExportEdgeCases:
    """Tests for edge cases in export."""

    def test_export_pattern_with_very_high_velocity(self, temp_dir):
        """Test export with maximum velocity (127)."""
        notes = [{"pitch": 36, "velocity": 127, "time": 0}]
        output_path = temp_dir / "max_velocity.mid"

        export_pattern(notes, output_path, humanize=False)

        assert output_path.exists()

    def test_export_pattern_with_very_low_velocity(self, temp_dir):
        """Test export with minimum velocity (1)."""
        notes = [{"pitch": 36, "velocity": 1, "time": 0}]
        output_path = temp_dir / "min_velocity.mid"

        export_pattern(notes, output_path, humanize=False)

        assert output_path.exists()

    def test_export_pattern_with_extreme_tempo(self, temp_dir):
        """Test export with extreme tempo values."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        # Very slow
        output_path = temp_dir / "slow.mid"
        export_pattern(notes, output_path, tempo=40, humanize=False)
        assert output_path.exists()

        # Very fast
        output_path = temp_dir / "fast.mid"
        export_pattern(notes, output_path, tempo=240, humanize=False)
        assert output_path.exists()

    def test_export_pattern_with_zero_time(self, temp_dir):
        """Test export with note at time 0."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]
        output_path = temp_dir / "zero_time.mid"

        export_pattern(notes, output_path, humanize=False)

        assert output_path.exists()

    def test_export_pattern_path_as_string(self, temp_dir):
        """Test export with path as string instead of Path object."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]
        output_path = str(temp_dir / "string_path.mid")

        result = export_pattern(notes, output_path, humanize=False)

        assert Path(result).exists()

    def test_export_pattern_with_metadata(self, temp_dir):
        """Test that exported MIDI has proper metadata."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
        ]
        output_path = temp_dir / "metadata.mid"
        style_name = "Test Style"

        export_pattern(
            notes,
            output_path,
            tempo=120,
            time_signature=(4, 4),
            humanize=False,
            style_name=style_name,
        )

        assert output_path.exists()

        # Read back and verify it's a valid MIDI file
        import mido

        mid = mido.MidiFile(output_path)
        assert mid.ticks_per_beat == DEFAULT_TICKS_PER_BEAT
        assert len(mid.tracks) > 0
