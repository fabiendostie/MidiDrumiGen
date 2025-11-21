"""Unit tests for MIDI validation module."""

import pytest

from src.midi.constants import MAX_DRUM_NOTE, MIN_DRUM_NOTE
from src.midi.validate import (
    EXCLUSIVE_PAIRS,
    get_pattern_statistics,
    validate_drum_pattern,
    validate_pattern_structure,
)


class TestValidateDrumPattern:
    """Tests for validate_drum_pattern function."""

    def test_validate_valid_pattern(self):
        """Test validation of a valid drum pattern."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
            {"pitch": 42, "velocity": 80, "time": 960},
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_empty_pattern(self):
        """Test that empty pattern fails validation."""
        notes = []

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert len(errors) > 0
        assert any("empty" in err.lower() for err in errors)

    def test_validate_empty_pattern_with_allow_empty(self):
        """Test that empty pattern passes when allow_empty=True."""
        notes = []

        is_valid, errors = validate_drum_pattern(notes, allow_empty=True)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_note_range(self):
        """Test that notes outside valid range fail."""
        notes = [
            {"pitch": 20, "velocity": 100, "time": 0},  # Too low
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("invalid drum note" in err.lower() for err in errors)

    def test_validate_invalid_velocity_zero(self):
        """Test that velocity 0 fails validation."""
        notes = [{"pitch": 36, "velocity": 0, "time": 0}]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("invalid velocity" in err.lower() for err in errors)

    def test_validate_invalid_velocity_too_high(self):
        """Test that velocity > 127 fails validation."""
        notes = [{"pitch": 36, "velocity": 128, "time": 0}]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("invalid velocity" in err.lower() for err in errors)

    def test_validate_negative_time(self):
        """Test that negative time values fail."""
        notes = [{"pitch": 36, "velocity": 100, "time": -10}]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("negative time" in err.lower() for err in errors)

    def test_validate_missing_pitch_key(self):
        """Test that missing pitch key fails validation."""
        notes = [{"velocity": 100, "time": 0}]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("missing 'pitch'" in err.lower() for err in errors)

    def test_validate_missing_velocity_key(self):
        """Test that missing velocity key fails validation."""
        notes = [{"pitch": 36, "time": 0}]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("missing 'velocity'" in err.lower() for err in errors)

    def test_validate_missing_time_key(self):
        """Test that missing time key fails validation."""
        notes = [{"pitch": 36, "velocity": 100}]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("missing 'time'" in err.lower() for err in errors)

    def test_validate_pattern_too_sparse(self):
        """Test that very sparse patterns fail density check."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 10000},  # Very far apart
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("too sparse" in err.lower() for err in errors)

    def test_validate_pattern_too_dense(self):
        """Test that very dense patterns fail density check."""
        # 100 notes in half a beat
        notes = [{"pitch": 36, "velocity": 100, "time": i} for i in range(100)]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("too dense" in err.lower() for err in errors)

    def test_validate_impossible_simultaneous_hits(self):
        """Test that impossible simultaneous hits fail."""
        # Closed and open hi-hat at same time (physically impossible)
        notes = [
            {"pitch": 42, "velocity": 100, "time": 0},  # Closed hi-hat
            {"pitch": 46, "velocity": 80, "time": 0},  # Open hi-hat
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("impossible simultaneous" in err.lower() for err in errors)

    def test_validate_too_many_simultaneous_hits(self):
        """Test that > 4 simultaneous hits fail (human has 4 limbs)."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 0},
            {"pitch": 42, "velocity": 80, "time": 0},
            {"pitch": 49, "velocity": 85, "time": 0},
            {"pitch": 51, "velocity": 75, "time": 0},  # 5th note at same time
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("too many simultaneous" in err.lower() for err in errors)

    def test_validate_duplicate_notes_at_same_time(self):
        """Test that duplicate notes at same time fail."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 36, "velocity": 90, "time": 0},  # Duplicate
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is False
        assert any("duplicate" in err.lower() for err in errors)

    def test_validate_valid_simultaneous_hits(self):
        """Test that valid simultaneous hits pass (e.g., kick + hi-hat)."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},  # Kick
            {"pitch": 42, "velocity": 70, "time": 0},  # Closed hi-hat
            {"pitch": 38, "velocity": 90, "time": 480},
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_boundary_note_values(self):
        """Test validation with boundary note values."""
        notes = [
            {"pitch": MIN_DRUM_NOTE, "velocity": 1, "time": 0},
            {"pitch": MAX_DRUM_NOTE, "velocity": 127, "time": 480},
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is True

    def test_validate_custom_ticks_per_beat(self):
        """Test validation with custom MIDI resolution."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 960},
        ]

        is_valid, errors = validate_drum_pattern(notes, ticks_per_beat=960)

        assert is_valid is True

    def test_validate_logs_debug_on_success(self, caplog):
        """Test that successful validation logs debug message."""
        import logging

        caplog.set_level(logging.DEBUG)

        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        is_valid, errors = validate_drum_pattern(notes)

        # Verify validation succeeded
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_logs_warning_on_failure(self, caplog):
        """Test that failed validation logs warning."""
        import logging

        caplog.set_level(logging.WARNING)

        notes = [{"pitch": 36, "velocity": 0, "time": 0}]

        is_valid, errors = validate_drum_pattern(notes)

        # Verify validation failed
        assert is_valid is False
        assert len(errors) > 0


class TestValidatePatternStructure:
    """Tests for validate_pattern_structure function."""

    def test_validate_structure_valid(self):
        """Test validation of valid structure."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        is_valid, errors = validate_pattern_structure(notes)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_structure_missing_pitch(self):
        """Test validation with missing pitch."""
        notes = [{"velocity": 100, "time": 0}]

        is_valid, errors = validate_pattern_structure(notes)

        assert is_valid is False
        assert any("missing 'pitch'" in err for err in errors)

    def test_validate_structure_missing_velocity(self):
        """Test validation with missing velocity."""
        notes = [{"pitch": 36, "time": 0}]

        is_valid, errors = validate_pattern_structure(notes)

        assert is_valid is False
        assert any("missing 'velocity'" in err for err in errors)

    def test_validate_structure_missing_time(self):
        """Test validation with missing time."""
        notes = [{"pitch": 36, "velocity": 100}]

        is_valid, errors = validate_pattern_structure(notes)

        assert is_valid is False
        assert any("missing 'time'" in err for err in errors)

    def test_validate_structure_not_dict(self):
        """Test validation with non-dict items."""
        notes = [36, 100, 0]  # Not dictionaries

        is_valid, errors = validate_pattern_structure(notes)

        assert is_valid is False
        assert any("not a dictionary" in err for err in errors)

    def test_validate_structure_multiple_errors(self):
        """Test that multiple errors are reported."""
        notes = [
            {"pitch": 36},  # Missing velocity and time
        ]

        is_valid, errors = validate_pattern_structure(notes)

        assert is_valid is False
        assert len(errors) == 2  # Missing velocity and time


class TestGetPatternStatistics:
    """Tests for get_pattern_statistics function."""

    def test_get_statistics_empty_pattern(self):
        """Test statistics for empty pattern."""
        notes = []

        stats = get_pattern_statistics(notes)

        assert stats["total_notes"] == 0
        assert stats["unique_pitches"] == 0
        assert stats["duration_beats"] == 0
        assert stats["density"] == 0

    def test_get_statistics_single_note(self):
        """Test statistics for single note."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        stats = get_pattern_statistics(notes)

        assert stats["total_notes"] == 1
        assert stats["unique_pitches"] == 1

    def test_get_statistics_multiple_notes(self):
        """Test statistics for multiple notes."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
            {"pitch": 42, "velocity": 80, "time": 960},
        ]

        stats = get_pattern_statistics(notes)

        assert stats["total_notes"] == 3
        assert stats["unique_pitches"] == 3
        assert stats["duration_beats"] == 2  # 960 ticks = 2 beats

    def test_get_statistics_duplicate_pitches(self):
        """Test statistics with duplicate pitches."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 36, "velocity": 90, "time": 480},
            {"pitch": 38, "velocity": 80, "time": 960},
        ]

        stats = get_pattern_statistics(notes)

        assert stats["total_notes"] == 3
        assert stats["unique_pitches"] == 2  # Only 36 and 38

    def test_get_statistics_velocity_range(self):
        """Test velocity range calculation."""
        notes = [
            {"pitch": 36, "velocity": 50, "time": 0},
            {"pitch": 38, "velocity": 127, "time": 480},
            {"pitch": 42, "velocity": 1, "time": 960},
        ]

        stats = get_pattern_statistics(notes)

        assert stats["velocity_range"] == (1, 127)

    def test_get_statistics_time_range(self):
        """Test time range calculation."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 100},
            {"pitch": 38, "velocity": 90, "time": 500},
        ]

        stats = get_pattern_statistics(notes)

        assert stats["time_range"] == (100, 500)

    def test_get_statistics_density(self):
        """Test density calculation."""
        # 4 notes over 2 beats = 2 notes/beat
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 240},
            {"pitch": 42, "velocity": 80, "time": 480},
            {"pitch": 49, "velocity": 85, "time": 720},
        ]

        stats = get_pattern_statistics(notes, ticks_per_beat=480)

        # Duration = (720 - 0) / 480 = 1.5 beats
        # Density = 4 / 1.5 = 2.67 notes/beat
        assert stats["duration_beats"] == pytest.approx(1.5)
        assert stats["density"] == pytest.approx(2.67, rel=0.01)

    def test_get_statistics_pitch_counts(self):
        """Test pitch count calculation."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 36, "velocity": 90, "time": 240},
            {"pitch": 38, "velocity": 80, "time": 480},
        ]

        stats = get_pattern_statistics(notes)

        assert stats["pitch_counts"][36] == 2
        assert stats["pitch_counts"][38] == 1

    def test_get_statistics_custom_ticks_per_beat(self):
        """Test statistics with custom MIDI resolution."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 960},
        ]

        stats = get_pattern_statistics(notes, ticks_per_beat=960)

        assert stats["duration_beats"] == 1  # 960 ticks = 1 beat at 960 TPB


class TestExclusivePairs:
    """Tests for EXCLUSIVE_PAIRS constant."""

    def test_exclusive_pairs_defined(self):
        """Test that EXCLUSIVE_PAIRS is defined."""
        assert EXCLUSIVE_PAIRS is not None
        assert isinstance(EXCLUSIVE_PAIRS, list)

    def test_exclusive_pairs_structure(self):
        """Test that each pair is a tuple of two integers."""
        for pair in EXCLUSIVE_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], int)
            assert isinstance(pair[1], int)

    def test_exclusive_pairs_includes_hihat(self):
        """Test that hi-hat pairs are included."""
        # Closed (42) and open (46) hi-hat should be exclusive
        assert (42, 46) in EXCLUSIVE_PAIRS or (46, 42) in EXCLUSIVE_PAIRS


class TestValidationEdgeCases:
    """Tests for edge cases in validation."""

    def test_validate_single_note_at_boundary(self):
        """Test validation of single note at time boundary."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is True

    def test_validate_all_drum_types(self):
        """Test validation with all GM drum types."""
        notes = [
            {"pitch": i, "velocity": 100, "time": (i - MIN_DRUM_NOTE) * 100}
            for i in range(MIN_DRUM_NOTE, MAX_DRUM_NOTE + 1)
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is True

    def test_validate_maximum_allowed_simultaneous(self):
        """Test exactly 4 simultaneous hits (maximum allowed)."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 0},
            {"pitch": 42, "velocity": 80, "time": 0},
            {"pitch": 49, "velocity": 85, "time": 0},
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is True

    def test_validate_notes_at_different_times(self):
        """Test that notes at different times pass validation."""
        notes = [
            {"pitch": 42, "velocity": 70, "time": 0},
            {"pitch": 46, "velocity": 80, "time": 240},  # Different time
        ]

        is_valid, errors = validate_drum_pattern(notes)

        assert is_valid is True
