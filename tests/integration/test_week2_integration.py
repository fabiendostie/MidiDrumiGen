"""
Integration tests for Week 2: Dynamic Producer Style Transfer Pipeline.

Tests the complete flow:
1. Producer research (with cache)
2. Style parameter extraction
3. Pattern generation
4. Style transfer application
5. MIDI export
"""

import pytest

# Marks all tests in this module as async
pytestmark = pytest.mark.asyncio


class TestProducerResearchIntegration:
    """Test producer research agent integration."""

    async def test_timbaland_research_cached(self):
        """Test Timbaland research (should be cached from Week 1)."""
        from src.research.producer_agent import ProducerResearchAgent

        agent = ProducerResearchAgent()
        profile = await agent.research_producer("Timbaland")

        assert profile is not None
        assert profile["producer_name"] == "Timbaland"
        assert "style_params" in profile
        assert profile["cached"] is True  # Should be cached from Week 1

        # Check required style parameters
        style_params = profile["style_params"]
        assert "tempo_range" in style_params
        assert "swing_percentage" in style_params
        assert "micro_timing_ms" in style_params
        assert "ghost_note_prob" in style_params
        assert "velocity_variation" in style_params

    async def test_multiple_producers_batch(self):
        """Test batch research of multiple producers."""
        from src.research.producer_agent import ProducerResearchAgent

        agent = ProducerResearchAgent()

        # These should all be cached from Week 1
        producers = ["Timbaland", "J Dilla", "Metro Boomin", "Aphex Twin"]

        profiles = await agent.batch_research(producers)

        assert len(profiles) == len(producers)

        for producer_name in producers:
            assert producer_name in profiles
            profile = profiles[producer_name]
            assert profile["producer_name"] == producer_name
            assert "style_params" in profile


class TestStyleTransferIntegration:
    """Test style transfer module integration."""

    async def test_apply_timbaland_style(self):
        """Test applying Timbaland style to mock pattern."""
        from src.midi.style_transfer import apply_producer_style
        from src.research.producer_agent import ProducerResearchAgent

        # Get Timbaland profile
        agent = ProducerResearchAgent()
        profile = await agent.research_producer("Timbaland")

        # Create mock note pattern
        notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 42, "velocity": 70, "time": 240},
            {"pitch": 38, "velocity": 90, "time": 480},
            {"pitch": 42, "velocity": 65, "time": 720},
        ]

        # Apply style transfer
        styled_notes = apply_producer_style(
            notes=notes, style_profile=profile, tempo=100, ticks_per_beat=480
        )

        # Verify transformations applied
        assert styled_notes is not None
        assert len(styled_notes) >= len(notes)  # May have ghost notes added

    async def test_style_transfer_preserves_pitch(self):
        """Test that style transfer doesn't change note pitches."""
        from src.midi.style_transfer import apply_producer_style
        from src.research.producer_agent import ProducerResearchAgent

        agent = ProducerResearchAgent()
        profile = await agent.research_producer("J Dilla")

        original_notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
        ]

        styled_notes = apply_producer_style(
            notes=original_notes.copy(), style_profile=profile, tempo=95, ticks_per_beat=480
        )

        # Original pitches should be preserved (excluding ghost notes)
        original_pitches = {n["pitch"] for n in original_notes}
        styled_pitches = {n["pitch"] for n in styled_notes}

        # All original pitches should still exist
        assert original_pitches.issubset(styled_pitches)

    async def test_style_validation(self):
        """Test style profile validation."""
        from src.midi.style_transfer import validate_style_profile

        # Valid profile
        valid_profile = {
            "producer_name": "Test Producer",
            "style_params": {
                "tempo_range": [90, 120],
                "swing_percentage": 55,
                "micro_timing_ms": 10,
                "ghost_note_prob": 0.2,
                "velocity_variation": 0.15,
            },
        }

        assert validate_style_profile(valid_profile) is True

        # Invalid profile (missing keys)
        invalid_profile = {
            "producer_name": "Test Producer",
            "style_params": {
                "tempo_range": [90, 120],
                # Missing other required keys
            },
        }

        assert validate_style_profile(invalid_profile) is False


class TestAPIIntegration:
    """Test API routes with dynamic producer names."""

    @pytest.mark.skip(reason="Requires running API server and Celery worker")
    async def test_api_dynamic_producer_endpoint(self):
        """Test API endpoint with dynamic producer name."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)

        # Test dynamic producer request
        response = client.post(
            "/api/v1/generate",
            json={"producer_name": "Timbaland", "bars": 4, "tempo": 100, "humanize": True},
        )

        assert response.status_code == 202  # Accepted
        data = response.json()
        assert "task_id" in data
        assert "status" in data
        assert data["status"] == "queued"

    @pytest.mark.skip(reason="Requires running API server and Celery worker")
    async def test_api_legacy_producer_style_endpoint(self):
        """Test API endpoint with legacy producer_style enum."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)

        # Test legacy producer_style request
        response = client.post(
            "/api/v1/generate",
            json={"producer_style": "J Dilla", "bars": 4, "tempo": 95, "humanize": True},
        )

        assert response.status_code == 202  # Accepted
        data = response.json()
        assert "task_id" in data


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    async def test_research_to_style_transfer_pipeline(self):
        """Test complete pipeline: research → style transfer → validation."""
        from src.midi.style_transfer import apply_producer_style
        from src.research.producer_agent import ProducerResearchAgent

        # Step 1: Research producer
        agent = ProducerResearchAgent()
        profile = await agent.research_producer("Burial")

        assert profile is not None
        assert "style_params" in profile

        # Step 2: Create mock pattern
        notes = []
        ticks_per_beat = 480
        for i in range(8):
            # Kick pattern
            notes.append({"pitch": 36, "velocity": 100, "time": i * ticks_per_beat})
            # Snare pattern
            if i % 2 == 1:
                notes.append({"pitch": 38, "velocity": 90, "time": i * ticks_per_beat})
            # Hi-hat pattern
            for j in range(4):
                notes.append(
                    {
                        "pitch": 42,
                        "velocity": 70 if j % 2 == 0 else 60,
                        "time": i * ticks_per_beat + j * (ticks_per_beat // 4),
                    }
                )

        original_count = len(notes)

        # Step 3: Apply style transfer
        styled_notes = apply_producer_style(
            notes=notes, style_profile=profile, tempo=140, ticks_per_beat=ticks_per_beat
        )

        # Step 4: Validate results
        assert styled_notes is not None
        assert len(styled_notes) >= original_count  # May have ghost notes

        # Verify notes have valid MIDI values
        for note in styled_notes:
            assert 1 <= note["velocity"] <= 127
            assert note["time"] >= 0
            assert 0 <= note["pitch"] <= 127

    async def test_multiple_styles_produce_different_results(self):
        """Test that different producer styles produce different results."""
        from src.midi.style_transfer import apply_producer_style
        from src.research.producer_agent import ProducerResearchAgent

        agent = ProducerResearchAgent()

        # Get two very different producers
        timbaland_profile = await agent.research_producer("Timbaland")
        aphex_twin_profile = await agent.research_producer("Aphex Twin")

        # Create same base pattern
        base_notes = [
            {"pitch": 36, "velocity": 100, "time": 0},
            {"pitch": 38, "velocity": 90, "time": 480},
            {"pitch": 36, "velocity": 100, "time": 960},
            {"pitch": 38, "velocity": 90, "time": 1440},
        ]

        # Apply Timbaland style
        timbaland_notes = apply_producer_style(
            notes=base_notes.copy(), style_profile=timbaland_profile, tempo=100, ticks_per_beat=480
        )

        # Apply Aphex Twin style
        aphex_notes = apply_producer_style(
            notes=base_notes.copy(), style_profile=aphex_twin_profile, tempo=140, ticks_per_beat=480
        )

        # Results should be different
        assert timbaland_notes != aphex_notes

        # Check swing differences (Aphex Twin should have higher complexity)
        timbaland_swing = timbaland_profile["style_params"]["swing_percentage"]
        aphex_swing = aphex_twin_profile["style_params"]["swing_percentage"]

        # These producers should have different swing settings
        assert (
            timbaland_swing != aphex_swing
            or timbaland_profile["style_params"]["complexity_level"]
            != aphex_twin_profile["style_params"]["complexity_level"]
        )


class TestCachedProducers:
    """Test with pre-cached producers from Week 1."""

    async def test_all_cached_producers_load(self):
        """Test that all Week 1 cached producers load successfully."""
        from src.research.cache import ProducerStyleCache

        cache = ProducerStyleCache()

        # These were cached in Week 1
        expected_producers = [
            "timbaland",
            "j_dilla",
            "metro_boomin",
            "questlove",
            "travis_barker",
            "flying_lotus",
            "aphex_twin",
            "anderson_paak",
            "burial",
        ]

        for producer in expected_producers:
            profile = cache.get(producer)
            if profile:  # May not all be present
                assert "style_params" in profile
                assert "producer_name" in profile

    async def test_cache_performance(self):
        """Test that cached lookups are fast."""
        import time

        from src.research.producer_agent import ProducerResearchAgent

        agent = ProducerResearchAgent()

        # First call (should be cached)
        start = time.time()
        profile = await agent.research_producer("Timbaland")
        duration = time.time() - start

        assert profile["cached"] is True
        assert duration < 0.5  # Should be < 500ms for cached lookup

    async def test_style_description_generation(self):
        """Test style description generation for cached producers."""
        from src.midi.style_transfer import get_style_description
        from src.research.producer_agent import ProducerResearchAgent

        agent = ProducerResearchAgent()

        producers = ["Timbaland", "J Dilla", "Aphex Twin"]

        for producer_name in producers:
            profile = await agent.research_producer(producer_name)
            description = get_style_description(profile)

            assert isinstance(description, str)
            assert len(description) > 0
            assert producer_name in description


# Fixtures


@pytest.fixture
def mock_notes():
    """Create mock MIDI note pattern for testing."""
    return [
        {"pitch": 36, "velocity": 100, "time": 0},
        {"pitch": 42, "velocity": 70, "time": 240},
        {"pitch": 38, "velocity": 90, "time": 480},
        {"pitch": 42, "velocity": 65, "time": 720},
        {"pitch": 36, "velocity": 95, "time": 960},
        {"pitch": 42, "velocity": 75, "time": 1200},
        {"pitch": 38, "velocity": 92, "time": 1440},
        {"pitch": 42, "velocity": 68, "time": 1680},
    ]


@pytest.fixture
async def timbaland_profile():
    """Get cached Timbaland profile."""
    from src.research.producer_agent import ProducerResearchAgent

    agent = ProducerResearchAgent()
    return await agent.research_producer("Timbaland")


# Run async tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
