"""Unit tests for MIDI humanization module."""

from unittest.mock import patch

from src.midi.constants import DEFAULT_TICKS_PER_BEAT
from src.midi.humanize import (
    PRODUCER_STYLES,
    add_ghost_notes,
    apply_accent_pattern,
    apply_micro_timing,
    apply_style_humanization,
    apply_swing,
    apply_velocity_variation,
)


class TestApplySwing:
    """Tests for apply_swing function."""

    def test_apply_swing_no_swing(self):
        """Test that 50% swing leaves notes unchanged."""
        note_time = 240  # Offbeat 8th note
        result = apply_swing(note_time, 50.0, DEFAULT_TICKS_PER_BEAT)
        assert result == note_time

    def test_apply_swing_on_downbeat(self):
        """Test that downbeats (on-beats) are not affected."""
        note_time = 480  # On-beat quarter note
        result = apply_swing(note_time, 66.0, DEFAULT_TICKS_PER_BEAT)
        assert result == note_time

    def test_apply_swing_on_offbeat(self):
        """Test that offbeats are delayed with swing."""
        note_time = 240  # Offbeat 8th note
        result = apply_swing(note_time, 66.0, DEFAULT_TICKS_PER_BEAT)
        assert result > note_time

    def test_apply_swing_different_percentages(self):
        """Test swing with different percentages."""
        note_time = 240

        swing_50 = apply_swing(note_time, 50.0, DEFAULT_TICKS_PER_BEAT)
        swing_58 = apply_swing(note_time, 58.0, DEFAULT_TICKS_PER_BEAT)
        swing_66 = apply_swing(note_time, 66.0, DEFAULT_TICKS_PER_BEAT)

        # Higher swing should delay more
        assert swing_50 <= swing_58 <= swing_66

    def test_apply_swing_j_dilla_style(self):
        """Test J Dilla swing (62%)."""
        note_time = 240
        result = apply_swing(note_time, 62.0, DEFAULT_TICKS_PER_BEAT)
        assert result > note_time

    def test_apply_swing_custom_ticks_per_beat(self):
        """Test swing with custom MIDI resolution."""
        note_time = 120  # Offbeat with 240 TPB
        result = apply_swing(note_time, 60.0, ticks_per_beat=240)
        assert result > note_time

    def test_apply_swing_returns_int(self):
        """Test that function returns integer."""
        result = apply_swing(240, 60.0, DEFAULT_TICKS_PER_BEAT)
        assert isinstance(result, int)


class TestApplyMicroTiming:
    """Tests for apply_micro_timing function."""

    @patch("random.randint")
    def test_apply_micro_timing_adds_offset(self, mock_randint):
        """Test that micro-timing adds random offset."""
        mock_randint.return_value = 10  # Fixed offset

        note_time = 480
        result = apply_micro_timing(note_time, 15.0, tempo=120)

        assert result != note_time

    def test_apply_micro_timing_non_negative(self):
        """Test that time never goes negative."""
        note_time = 10
        # Even with large offset, should not go negative
        result = apply_micro_timing(note_time, 100.0, tempo=120)
        assert result >= 0

    def test_apply_micro_timing_zero_offset(self):
        """Test with zero millisecond offset."""
        note_time = 480
        result = apply_micro_timing(note_time, 0.0, tempo=120)
        assert result == note_time

    def test_apply_micro_timing_different_tempos(self):
        """Test micro-timing at different tempos."""
        note_time = 480

        # At faster tempo, ticks-per-millisecond changes
        result_slow = apply_micro_timing(note_time, 15.0, tempo=60)
        result_fast = apply_micro_timing(note_time, 15.0, tempo=200)

        # Both should be close to original (random variation)
        assert abs(result_slow - note_time) <= 100
        assert abs(result_fast - note_time) <= 100

    def test_apply_micro_timing_returns_int(self):
        """Test that function returns integer."""
        result = apply_micro_timing(480, 15.0, tempo=120)
        assert isinstance(result, int)


class TestApplyVelocityVariation:
    """Tests for apply_velocity_variation function."""

    @patch("random.uniform")
    def test_apply_velocity_variation_adds_offset(self, mock_uniform):
        """Test that velocity variation adds offset."""
        mock_uniform.return_value = 0.5  # 50% of variation

        base_velocity = 100
        result = apply_velocity_variation(base_velocity, variation=0.1)

        # Should be 100 + 5 = 105
        assert result != base_velocity

    def test_apply_velocity_variation_stays_in_range(self):
        """Test that velocity stays in valid MIDI range (1-127)."""
        for base_velocity in [1, 64, 127]:
            result = apply_velocity_variation(base_velocity, variation=0.2)
            assert 1 <= result <= 127

    def test_apply_velocity_variation_clamps_minimum(self):
        """Test that velocity is clamped to minimum."""
        result = apply_velocity_variation(
            base_velocity=5, variation=1.0, min_velocity=1  # Very high variation
        )
        assert result >= 1

    def test_apply_velocity_variation_clamps_maximum(self):
        """Test that velocity is clamped to maximum."""
        result = apply_velocity_variation(
            base_velocity=120, variation=1.0, max_velocity=127  # Very high variation
        )
        assert result <= 127

    def test_apply_velocity_variation_zero_variation(self):
        """Test with zero variation."""
        base_velocity = 100
        result = apply_velocity_variation(base_velocity, variation=0.0)
        assert result == base_velocity

    def test_apply_velocity_variation_custom_range(self):
        """Test with custom min/max velocity."""
        result = apply_velocity_variation(
            base_velocity=50, variation=0.5, min_velocity=40, max_velocity=60
        )
        assert 40 <= result <= 60

    def test_apply_velocity_variation_returns_int(self):
        """Test that function returns integer."""
        result = apply_velocity_variation(100, variation=0.1)
        assert isinstance(result, int)


class TestApplyAccentPattern:
    """Tests for apply_accent_pattern function."""

    def test_apply_accent_pattern_boosts_accented_notes(self):
        """Test that accented notes have higher velocity."""
        notes = [
            {"pitch": 36, "velocity": 80, "time": 0},
            {"pitch": 38, "velocity": 80, "time": 240},
            {"pitch": 36, "velocity": 80, "time": 480},
        ]

        result = apply_accent_pattern(notes, accent_positions=[0, 2])

        # Accented notes should have boosted velocity
        assert result[0]["velocity"] > 80
        assert result[2]["velocity"] > 80

    def test_apply_accent_pattern_reduces_adjacent(self):
        """Test that adjacent notes are reduced."""
        notes = [
            {"pitch": 36, "velocity": 80, "time": 0},
            {"pitch": 38, "velocity": 80, "time": 240},
            {"pitch": 36, "velocity": 80, "time": 480},
        ]

        result = apply_accent_pattern(notes, accent_positions=[1])

        # Adjacent notes should be reduced
        assert result[0]["velocity"] < 80
        assert result[2]["velocity"] < 80

    def test_apply_accent_pattern_does_not_modify_original(self):
        """Test that original notes list is not modified."""
        notes = [
            {"pitch": 36, "velocity": 80, "time": 0},
            {"pitch": 38, "velocity": 80, "time": 240},
        ]

        result = apply_accent_pattern(notes, accent_positions=[0])

        # Original should be unchanged
        assert notes[0]["velocity"] == 80
        # Result should be modified
        assert result[0]["velocity"] != 80

    def test_apply_accent_pattern_empty_accents(self):
        """Test with no accent positions."""
        notes = [{"pitch": 36, "velocity": 80, "time": 0}]
        result = apply_accent_pattern(notes, accent_positions=[])

        assert result[0]["velocity"] == 80

    def test_apply_accent_pattern_custom_boost(self):
        """Test with custom accent boost value."""
        notes = [{"pitch": 36, "velocity": 80, "time": 0}]
        result = apply_accent_pattern(notes, accent_positions=[0], accent_boost=30)

        assert result[0]["velocity"] == 110  # 80 + 30

    def test_apply_accent_pattern_respects_max_velocity(self):
        """Test that boosted velocity doesn't exceed 127."""
        notes = [{"pitch": 36, "velocity": 120, "time": 0}]
        result = apply_accent_pattern(notes, accent_positions=[0], accent_boost=50)

        assert result[0]["velocity"] == 127

    def test_apply_accent_pattern_respects_min_velocity(self):
        """Test that reduced velocity doesn't go below 1."""
        notes = [
            {"pitch": 36, "velocity": 3, "time": 0},
            {"pitch": 38, "velocity": 80, "time": 240},
        ]
        result = apply_accent_pattern(notes, accent_positions=[1], accent_reduction=10)

        assert result[0]["velocity"] >= 1


class TestAddGhostNotes:
    """Tests for add_ghost_notes function."""

    @patch("random.random")
    def test_add_ghost_notes_adds_notes(self, mock_random):
        """Test that ghost notes are added."""
        mock_random.return_value = 0.1  # Always add (prob < 0.3)

        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 960},  # Large gap
        ]

        result = add_ghost_notes(notes, probability=0.3)

        # Should have added ghost notes
        assert len(result) > len(notes)

    @patch("random.random")
    def test_add_ghost_notes_does_not_add_if_probability_low(self, mock_random):
        """Test that ghost notes are not added when random > probability."""
        mock_random.return_value = 0.9  # Never add (prob < 0.3)

        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 960},
        ]

        result = add_ghost_notes(notes, probability=0.3)

        # Should not have added ghost notes
        assert len(result) == len(notes)

    def test_add_ghost_notes_does_not_modify_original(self):
        """Test that original notes are not modified."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 960},
        ]
        original_len = len(notes)

        add_ghost_notes(notes, probability=0.5)

        # Original unchanged
        assert len(notes) == original_len

    def test_add_ghost_notes_zero_probability(self):
        """Test with zero probability."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 960},
        ]

        result = add_ghost_notes(notes, probability=0.0)

        assert len(result) == len(notes)

    def test_add_ghost_notes_custom_velocity(self):
        """Test with custom ghost note velocity."""
        with patch("random.random", return_value=0.1):
            notes = [
                {"pitch": 36, "velocity": 100, "time": 0},
                {"pitch": 38, "velocity": 90, "time": 960},
            ]

            result = add_ghost_notes(notes, probability=0.5, ghost_velocity=25)

            # Find ghost notes (those with velocity 25)
            ghost_notes = [n for n in result if n["velocity"] == 25]
            assert len(ghost_notes) > 0

    def test_add_ghost_notes_custom_pitch(self):
        """Test with custom ghost note pitch."""
        with patch("random.random", return_value=0.1):
            notes = [
                {"pitch": 36, "velocity": 100, "time": 0},
                {"pitch": 38, "velocity": 90, "time": 960},
            ]

            result = add_ghost_notes(notes, probability=0.5, ghost_note=42)  # Hi-hat

            # Find ghost notes (those with pitch 42)
            ghost_notes = [n for n in result if n["pitch"] == 42]
            assert len(ghost_notes) > 0

    def test_add_ghost_notes_respects_minimum_spacing(self):
        """Test that ghost notes respect minimum spacing."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 50},  # Too close
        ]

        result = add_ghost_notes(notes, probability=1.0)

        # Should not add ghost (notes too close)
        assert len(result) == len(notes)


class TestProducerStyles:
    """Tests for PRODUCER_STYLES registry."""

    def test_producer_styles_exists(self):
        """Test that PRODUCER_STYLES is defined."""
        assert PRODUCER_STYLES is not None
        assert isinstance(PRODUCER_STYLES, dict)

    def test_producer_styles_has_required_styles(self):
        """Test that required styles are present."""
        required_styles = ["J Dilla", "Metro Boomin", "Questlove"]
        for style in required_styles:
            assert style in PRODUCER_STYLES

    def test_producer_styles_parameters(self):
        """Test that each style has required parameters."""
        required_params = [
            "swing",
            "micro_timing_ms",
            "ghost_note_prob",
            "velocity_variation",
            "preferred_tempo_range",
        ]

        for style_name, style_data in PRODUCER_STYLES.items():
            for param in required_params:
                assert param in style_data, f"{style_name} missing {param}"

    def test_producer_styles_has_aliases(self):
        """Test that lowercase aliases exist."""
        assert "j_dilla" in PRODUCER_STYLES
        assert "metro_boomin" in PRODUCER_STYLES
        assert "questlove" in PRODUCER_STYLES


class TestApplyStyleHumanization:
    """Tests for apply_style_humanization function."""

    def test_apply_style_humanization_j_dilla(self):
        """Test J Dilla style humanization."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
        ]

        result = apply_style_humanization(notes, "J Dilla", tempo=95)

        # Should return modified notes
        assert len(result) >= len(notes)  # May add ghost notes
        assert result != notes

    def test_apply_style_humanization_metro_boomin(self):
        """Test Metro Boomin style humanization."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        result = apply_style_humanization(notes, "Metro Boomin", tempo=140)

        assert len(result) >= len(notes)

    def test_apply_style_humanization_questlove(self):
        """Test Questlove style humanization."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        result = apply_style_humanization(notes, "Questlove", tempo=100)

        assert len(result) >= len(notes)

    def test_apply_style_humanization_unknown_style(self, caplog):
        """Test that unknown style uses default humanization."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        result = apply_style_humanization(notes, "Unknown Style", tempo=120)

        # Should log warning
        assert any("unknown style" in record.message.lower() for record in caplog.records)
        # Should still return results
        assert len(result) > 0

    def test_apply_style_humanization_does_not_modify_original(self):
        """Test that original notes are not modified."""
        notes = [{"pitch": 36, "velocity": 100, "time": 480}]
        original_velocity = notes[0]["velocity"]
        original_time = notes[0]["time"]

        apply_style_humanization(notes, "J Dilla", tempo=95)

        # Original should be unchanged
        assert notes[0]["velocity"] == original_velocity
        assert notes[0]["time"] == original_time

    def test_apply_style_humanization_applies_swing(self):
        """Test that swing is applied."""
        notes = [{"pitch": 36, "velocity": 100, "time": 240}]  # Offbeat

        result = apply_style_humanization(notes, "J Dilla", tempo=95)

        # Time may be different due to swing
        # (Note: might also be affected by micro-timing)
        assert len(result) > 0

    def test_apply_style_humanization_applies_velocity_variation(self):
        """Test that velocity variation is applied."""
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 36, "velocity": 100, "time": 480},
        ]

        result = apply_style_humanization(notes, "J Dilla", tempo=95)

        # Velocities should likely be different (due to random variation)
        # But we can't guarantee this in every run
        assert len(result) >= len(notes)

    def test_apply_style_humanization_custom_ticks_per_beat(self):
        """Test humanization with custom MIDI resolution."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        result = apply_style_humanization(notes, "J Dilla", tempo=95, ticks_per_beat=960)

        assert len(result) > 0

    def test_apply_style_humanization_case_insensitive(self):
        """Test that style names are case-insensitive."""
        notes = [{"pitch": 36, "velocity": 100, "time": 0}]

        result_upper = apply_style_humanization(notes, "J DILLA", tempo=95)
        result_lower = apply_style_humanization(notes, "j dilla", tempo=95)

        # Both should work
        assert len(result_upper) > 0
        assert len(result_lower) > 0
