"""
Unit Tests for Audio Analysis Collector

Story: E1.S3 - Audio Analysis Collection
Test Coverage: Audio analysis, swing ratio, syncopation, velocity estimation
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.research.collectors.audio import AudioAnalysisCollector

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def collector():
    """Create AudioAnalysisCollector instance for testing."""
    return AudioAnalysisCollector(timeout=60, max_videos=3)


@pytest.fixture
def mock_audio_signal():
    """Create a mock audio signal for testing."""
    # 1 second of audio at 22050 Hz
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple sine wave with some noise
    y = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    return y.astype(np.float32), sr


# =============================================================================
# Test Swing Ratio Calculation
# =============================================================================


class TestSwingRatioCalculation:
    """Test swing ratio calculation."""

    def test_straight_rhythm_50_percent(self, collector):
        """Should return ~50% for straight rhythm."""
        # Equal spacing = straight rhythm
        beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ratio = collector._calculate_swing_ratio(beat_times)
        assert 49.0 <= ratio <= 51.0

    def test_swung_rhythm_greater_than_50(self, collector):
        """Should return >50% for swung rhythm."""
        # Long-short pattern = swung rhythm
        beat_times = np.array([0.0, 0.6, 1.0, 1.6, 2.0, 2.6, 3.0, 3.6])
        ratio = collector._calculate_swing_ratio(beat_times)
        assert ratio > 50.0

    def test_insufficient_beats_default(self, collector):
        """Should return 50.0 for fewer than 4 beats."""
        beat_times = np.array([0.0, 0.5, 1.0])
        ratio = collector._calculate_swing_ratio(beat_times)
        assert ratio == 50.0

    def test_empty_beats_default(self, collector):
        """Should return 50.0 for empty beats array."""
        beat_times = np.array([])
        ratio = collector._calculate_swing_ratio(beat_times)
        assert ratio == 50.0


# =============================================================================
# Test Syncopation Index Calculation
# =============================================================================


class TestSyncopationIndexCalculation:
    """Test syncopation index calculation."""

    def test_on_beat_low_syncopation(self, collector):
        """Should return low syncopation for onsets on beats."""
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # All on beats
        index = collector._calculate_syncopation_index(onsets, beats)
        assert index < 0.1

    def test_off_beat_high_syncopation(self, collector):
        """Should return higher syncopation for off-beat onsets."""
        beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        onsets = np.array([0.25, 0.75, 1.25, 1.75])  # All off beats
        index = collector._calculate_syncopation_index(onsets, beats)
        assert index > 0.3

    def test_empty_arrays_zero(self, collector):
        """Should return 0.0 for empty arrays."""
        index = collector._calculate_syncopation_index(np.array([]), np.array([]))
        assert index == 0.0


# =============================================================================
# Test Velocity Distribution Estimation
# =============================================================================


class TestVelocityEstimation:
    """Test velocity distribution estimation."""

    def test_returns_mean_and_std(self, collector, mock_audio_signal):
        """Should return dictionary with mean and std."""
        y, sr = mock_audio_signal
        onset_frames = np.array([0, 100, 200, 300])
        result = collector._estimate_velocity_distribution(y, sr, onset_frames)
        assert "mean" in result
        assert "std" in result
        assert isinstance(result["mean"], int)
        assert isinstance(result["std"], int)

    def test_velocity_range(self, collector, mock_audio_signal):
        """Should return velocities in MIDI range."""
        y, sr = mock_audio_signal
        onset_frames = np.array([0, 100, 200])
        result = collector._estimate_velocity_distribution(y, sr, onset_frames)
        assert 1 <= result["mean"] <= 127

    def test_empty_onsets_default(self, collector, mock_audio_signal):
        """Should return default values for empty onsets."""
        y, sr = mock_audio_signal
        result = collector._estimate_velocity_distribution(y, sr, np.array([]))
        assert result["mean"] == 64
        assert result["std"] == 16


# =============================================================================
# Test Confidence Scoring
# =============================================================================


class TestConfidenceScoring:
    """Test confidence scoring algorithm."""

    def test_combined_method_boost(self, collector):
        """Should boost confidence for combined analysis."""
        analysis = {
            "analysis_method": "combined",
            "beat_positions": [0.0, 0.5, 1.0],
            "tempo_bpm": 120,
        }
        confidence = collector._calculate_confidence(analysis)
        assert confidence > 0.6

    def test_many_beats_boost(self, collector):
        """Should boost confidence for many detected beats."""
        analysis = {
            "analysis_method": "librosa",
            "beat_positions": list(range(25)),
            "tempo_bpm": 120,
        }
        confidence = collector._calculate_confidence(analysis)
        assert confidence >= 0.7

    def test_reasonable_tempo_boost(self, collector):
        """Should boost confidence for reasonable tempo."""
        analysis = {
            "analysis_method": "librosa",
            "beat_positions": [0.0, 0.5],
            "tempo_bpm": 120,
        }
        confidence = collector._calculate_confidence(analysis)

        analysis_bad_tempo = {
            "analysis_method": "librosa",
            "beat_positions": [0.0, 0.5],
            "tempo_bpm": 30,  # Too slow
        }
        confidence_bad = collector._calculate_confidence(analysis_bad_tempo)

        assert confidence > confidence_bad

    def test_confidence_capped_at_1(self, collector):
        """Should cap confidence at 1.0."""
        analysis = {
            "analysis_method": "combined",
            "beat_positions": list(range(50)),
            "tempo_bpm": 120,
        }
        confidence = collector._calculate_confidence(analysis)
        assert confidence <= 1.0


# =============================================================================
# Test Librosa Analysis
# =============================================================================


class TestLibrosaAnalysis:
    """Test Librosa audio analysis."""

    def test_analyze_with_librosa(self, collector, mock_audio_signal):
        """Should analyze audio and return expected keys."""
        y, sr = mock_audio_signal
        result = collector._analyze_with_librosa(y, sr)

        assert "tempo_bpm" in result
        assert "swing_ratio" in result
        assert "syncopation_index" in result
        assert "velocity_mean" in result
        assert "velocity_std" in result
        assert "beat_positions" in result
        assert "onset_positions" in result

    def test_tempo_in_valid_range(self, collector, mock_audio_signal):
        """Should return tempo in reasonable range."""
        y, sr = mock_audio_signal
        result = collector._analyze_with_librosa(y, sr)
        # Tempo should be positive
        assert result["tempo_bpm"] > 0


# =============================================================================
# Test YouTube Search
# =============================================================================


@pytest.mark.asyncio
class TestYouTubeSearch:
    """Test YouTube search functionality."""

    async def test_search_youtube_returns_urls(self, collector):
        """Should return list of URLs from search."""
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = {
                "entries": [
                    {"id": "video1"},
                    {"id": "video2"},
                    {"id": "video3"},
                ]
            }
            mock_ydl.return_value.__enter__.return_value = mock_instance

            urls = await collector._search_youtube("Test Artist")
            assert len(urls) == 3
            assert "youtube.com" in urls[0]

    async def test_search_youtube_no_results(self, collector):
        """Should return empty list when no results."""
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = {"entries": []}
            mock_ydl.return_value.__enter__.return_value = mock_instance

            urls = await collector._search_youtube("Unknown Artist")
            assert urls == []

    async def test_search_youtube_error_handling(self, collector):
        """Should handle errors gracefully."""
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.side_effect = Exception("Network error")

            urls = await collector._search_youtube("Test Artist")
            assert urls == []


# =============================================================================
# Test Main Collect Method
# =============================================================================


@pytest.mark.asyncio
class TestCollect:
    """Test main collect() method."""

    async def test_collect_returns_research_sources(self, collector):
        """Should return list of ResearchSource objects."""
        mock_analysis = {
            "tempo_bpm": 120,
            "swing_ratio": 55.0,
            "syncopation_index": 0.3,
            "velocity_mean": 80,
            "velocity_std": 15,
            "beat_positions": [0.0, 0.5, 1.0],
            "onset_positions": [0.0, 0.5, 1.0],
            "analysis_method": "librosa",
        }
        with (
            patch.object(collector, "_search_youtube", return_value=["http://youtube.com/video1"]),
            patch.object(collector, "_download_audio", return_value=True),
            patch.object(collector, "_analyze_audio", return_value=mock_analysis),
        ):
            results = await collector.collect("Test Artist")

            assert len(results) == 1
            assert results[0].source_type == "audio"
            assert "tempo_bpm" in results[0].extracted_data

    async def test_collect_no_videos_found(self, collector):
        """Should return empty list when no videos found."""
        with patch.object(collector, "_search_youtube", return_value=[]):
            results = await collector.collect("Unknown Artist")
            assert results == []

    async def test_collect_handles_download_failure(self, collector):
        """Should skip videos that fail to download."""
        mock_analysis = {
            "tempo_bpm": 120,
            "swing_ratio": 50.0,
            "syncopation_index": 0.2,
            "velocity_mean": 70,
            "velocity_std": 10,
            "beat_positions": [0.0, 0.5],
            "onset_positions": [0.0, 0.5],
            "analysis_method": "librosa",
        }
        with (
            patch.object(
                collector,
                "_search_youtube",
                return_value=["http://youtube.com/video1", "http://youtube.com/video2"],
            ),
            patch.object(collector, "_download_audio", side_effect=[False, True]),
            patch.object(collector, "_analyze_audio", return_value=mock_analysis),
        ):
            results = await collector.collect("Test Artist")
            # Only second video should succeed
            assert len(results) == 1

    async def test_collect_handles_analysis_failure(self, collector):
        """Should skip videos that fail analysis."""
        with (
            patch.object(collector, "_search_youtube", return_value=["http://youtube.com/video1"]),
            patch.object(collector, "_download_audio", return_value=True),
            patch.object(collector, "_analyze_audio", return_value=None),
        ):
            results = await collector.collect("Test Artist")
            assert results == []


# =============================================================================
# Test Error Handling
# =============================================================================


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_import_error_yt_dlp(self, collector):
        """Should handle missing yt-dlp gracefully."""
        with (
            patch.dict("sys.modules", {"yt_dlp": None}),
            patch("builtins.__import__", side_effect=ImportError("No module")),
        ):
            # The import happens inside the method, so we need to patch differently
            urls = await collector._search_youtube("Test Artist")
            # Should return empty list, not crash
            assert isinstance(urls, list)

    async def test_download_failure_returns_false(self, collector):
        """Should return False on download failure."""
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.download.side_effect = Exception("Download error")
            mock_ydl.return_value.__enter__.return_value = mock_instance

            result = await collector._download_audio(
                "http://youtube.com/test", Path("/tmp/test.wav")
            )
            assert result is False
