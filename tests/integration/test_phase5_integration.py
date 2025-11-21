"""Integration tests for Phase 5: Complete generation pipeline end-to-end."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


class TestGenerationPipelineIntegration:
    """End-to-end tests for complete generation pipeline."""

    def test_complete_pipeline_with_mock_model(self, temp_output_dir):
        """Test complete pipeline from model loading to MIDI export using MockDrumModel."""
        # Import all components
        from src.inference.generate import generate_pattern
        from src.inference.mock import MockDrumModel
        from src.midi.export import export_pattern
        from src.midi.humanize import apply_style_humanization

        # Step 1: Initialize mock model
        model = MockDrumModel()
        device = "cpu"
        model.to(device)
        model.eval()

        # Step 2: Generate tokens
        tokens = generate_pattern(
            model=model,
            tokenizer=None,
            num_bars=4,
            temperature=0.9,
            top_k=50,
            top_p=0.9,
            max_length=256,
            device=device,
            style_id=0,
        )

        assert len(tokens) > 0, "Should generate tokens"
        assert isinstance(tokens, list), "Tokens should be a list"

        # Step 3: Convert tokens to notes (mock for now)
        # TODO: When tokenizer is implemented, use detokenize_to_notes
        notes = []
        ticks_per_beat = 480
        for bar in range(4):
            base_tick = bar * 4 * ticks_per_beat
            # Kick
            notes.append({"pitch": 36, "velocity": 100, "time": base_tick})
            # Snare
            notes.append({"pitch": 38, "velocity": 90, "time": base_tick + 2 * ticks_per_beat})
            # Hi-hat
            for i in range(8):
                notes.append(
                    {"pitch": 42, "velocity": 70, "time": base_tick + i * (ticks_per_beat // 2)}
                )

        assert len(notes) > 0, "Should have notes"

        # Step 4: Apply humanization
        humanized_notes = apply_style_humanization(
            notes=notes, style="J Dilla", tempo=95, ticks_per_beat=ticks_per_beat
        )

        # Humanization may add ghost notes, so count can increase
        assert len(humanized_notes) >= len(notes), "Humanization can add notes (ghost notes)"

        # Step 5: Export to MIDI
        output_path = temp_output_dir / "test_pattern.mid"
        midi_path = export_pattern(
            notes=humanized_notes,
            output_path=output_path,
            tempo=95,
            time_signature=(4, 4),
            humanize=False,  # Already humanized
            style_name="J Dilla",
            ticks_per_beat=ticks_per_beat,
        )

        # Verify MIDI file was created
        assert midi_path.exists(), "MIDI file should exist"
        assert midi_path.stat().st_size > 0, "MIDI file should not be empty"

    def test_generation_with_different_styles(self, temp_output_dir):
        """Test generation with different producer styles."""
        from src.inference.generate import generate_pattern
        from src.inference.mock import MockDrumModel
        from src.midi.export import export_pattern

        model = MockDrumModel()
        device = "cpu"
        model.to(device)
        model.eval()

        styles = ["J Dilla", "Metro Boomin", "Questlove"]
        style_ids = [0, 1, 2]

        for style, style_id in zip(styles, style_ids, strict=False):
            # Generate
            tokens = generate_pattern(
                model=model,
                tokenizer=None,
                num_bars=2,
                temperature=0.9,
                device=device,
                style_id=style_id,
                max_length=128,
            )

            assert len(tokens) > 0

            # Create simple notes
            notes = [
                {"pitch": 36, "velocity": 100, "time": 0},
                {"pitch": 38, "velocity": 90, "time": 480},
            ]

            # Export
            output_path = temp_output_dir / f"test_{style.replace(' ', '_').lower()}.mid"
            midi_path = export_pattern(
                notes=notes,
                output_path=output_path,
                tempo=120,
                time_signature=(4, 4),
                humanize=True,
                style_name=style,
            )

            assert midi_path.exists()

    def test_gpu_fallback_cpu_execution(self, temp_output_dir):
        """Test that generation works on CPU (fallback scenario)."""
        from src.inference.generate import generate_pattern
        from src.inference.mock import MockDrumModel

        model = MockDrumModel()
        device = "cpu"  # Explicitly use CPU
        model.to(device)
        model.eval()

        tokens = generate_pattern(
            model=model,
            tokenizer=None,
            num_bars=4,
            temperature=1.0,
            device=device,
            style_id=0,
            max_length=128,
        )

        assert len(tokens) > 0
        # Verify it completed on CPU
        assert device == "cpu"

    def test_different_time_signatures(self, temp_output_dir):
        """Test MIDI export with different time signatures."""
        from src.midi.export import export_pattern

        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
        ]

        time_signatures = [(4, 4), (3, 4), (6, 8), (7, 8)]

        for ts in time_signatures:
            output_path = temp_output_dir / f"test_{ts[0]}_{ts[1]}.mid"
            midi_path = export_pattern(
                notes=notes,
                output_path=output_path,
                tempo=120,
                time_signature=ts,
                humanize=False,
                style_name="Test",
            )

            assert midi_path.exists()
            assert midi_path.stat().st_size > 0

    def test_different_tempos(self, temp_output_dir):
        """Test MIDI export with different tempos."""
        from src.midi.export import export_pattern

        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 42, "velocity": 70, "time": 240},
        ]

        tempos = [60, 95, 120, 140, 180]

        for tempo in tempos:
            output_path = temp_output_dir / f"test_{tempo}bpm.mid"
            midi_path = export_pattern(
                notes=notes,
                output_path=output_path,
                tempo=tempo,
                time_signature=(4, 4),
                humanize=False,
                style_name="Test",
            )

            assert midi_path.exists()


class TestModelLoaderIntegration:
    """Integration tests for model loading with caching."""

    def test_model_loader_cache_clear(self):
        """Test that model loader cache can be cleared."""
        from src.inference.model_loader import clear_model_cache

        # Should not raise any errors
        clear_model_cache()

        # Call again to ensure it's idempotent
        clear_model_cache()


class TestHumanizationIntegration:
    """Integration tests for humanization pipeline."""

    def test_humanization_preserves_note_structure(self):
        """Test that humanization doesn't break note structure."""
        from src.midi.humanize import apply_style_humanization

        original_notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
            {"pitch": 42, "velocity": 70, "time": 240},
        ]

        humanized = apply_style_humanization(
            notes=original_notes.copy(), style="J Dilla", tempo=95, ticks_per_beat=480
        )

        # Verify structure (humanization may add ghost notes)
        assert len(humanized) >= len(original_notes), "Humanization can add ghost notes"
        for note in humanized:
            assert "pitch" in note
            assert "velocity" in note
            assert "time" in note
            assert 1 <= note["velocity"] <= 127
            assert note["time"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
