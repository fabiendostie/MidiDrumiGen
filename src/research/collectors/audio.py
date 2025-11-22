"""
Audio Analysis Collector for MidiDrumiGen v2.0

Downloads and analyzes audio recordings from YouTube to extract drumming
style parameters using Librosa and madmom.

Story: E1.S3 - Audio Analysis Collection
Epic: E1 - Research Pipeline
Priority: HIGH
Story Points: 5
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .base import BaseCollector, CollectorError, ResearchSource


class AudioAnalysisCollector(BaseCollector):
    """
    Collects and analyzes audio recordings to extract drumming parameters.

    Uses:
        - yt-dlp for YouTube audio download
        - Librosa for primary audio analysis
        - madmom for advanced RNN/DBN beat tracking

    Acceptance Criteria:
        - Downloads audio from YouTube using yt-dlp
        - Analyzes with Librosa and madmom
        - Extracts tempo, swing ratio, syncopation index, velocity distribution
        - Cleans up temporary files after analysis
        - Returns List[ResearchSource] with source_type='audio'
    """

    # yt-dlp configuration
    YDL_OPTS = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    # Maximum number of videos to analyze
    MAX_VIDEOS = 5

    # Sample rate for analysis
    SAMPLE_RATE = 22050

    def __init__(self, timeout: int = 600, max_videos: int = 5):
        """
        Initialize Audio Analysis Collector.

        Args:
            timeout: Maximum time in seconds for collection (default: 10 min)
            max_videos: Maximum number of videos to analyze
        """
        super().__init__(timeout)
        self.max_videos = max_videos

    async def collect(self, artist_name: str) -> list[ResearchSource]:
        """
        Collect and analyze audio recordings for the artist.

        Args:
            artist_name: Name of the artist/drummer to research

        Returns:
            List of ResearchSource objects with source_type='audio'

        Raises:
            CollectorError: If critical error occurs
        """
        self.logger.info(f"Starting audio analysis collection for artist: {artist_name}")

        results = []

        try:
            # Create temporary directory for audio files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Search for YouTube videos
                video_urls = await self._search_youtube(artist_name)

                if not video_urls:
                    self.logger.warning(f"No YouTube videos found for {artist_name}")
                    return results

                # Limit to max videos
                video_urls = video_urls[: self.max_videos]
                self.logger.info(f"Found {len(video_urls)} videos to analyze")

                # Download and analyze each video
                for i, url in enumerate(video_urls):
                    try:
                        audio_path = temp_path / f"audio_{i}.wav"

                        # Download audio
                        success = await self._download_audio(url, audio_path)
                        if not success:
                            self.logger.warning(f"Failed to download {url}, skipping")
                            continue

                        # Analyze audio
                        analysis = await asyncio.to_thread(self._analyze_audio, audio_path)

                        if analysis:
                            # Calculate confidence
                            confidence = self._calculate_confidence(analysis)

                            results.append(
                                ResearchSource(
                                    source_type="audio",
                                    title=f"{artist_name} - Audio Analysis {i + 1}",
                                    url=url,
                                    raw_content=None,
                                    extracted_data=analysis,
                                    confidence=confidence,
                                    metadata={
                                        "video_index": i,
                                        "analysis_method": analysis.get(
                                            "analysis_method", "librosa"
                                        ),
                                    },
                                )
                            )

                    except Exception as e:
                        self.logger.warning(f"Error analyzing video {url}: {e}")
                        continue

            self.logger.info(f"Completed audio analysis: {len(results)} sources collected")
            return results

        except Exception as e:
            raise CollectorError(f"Audio collection failed: {e}") from e

    async def _search_youtube(self, artist_name: str) -> list[str]:
        """
        Search YouTube for drum performance videos.

        Args:
            artist_name: Artist name to search for

        Returns:
            List of YouTube video URLs
        """
        try:
            import yt_dlp

            search_query = f"{artist_name} drummer drum cam live performance"

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "default_search": f"ytsearch{self.max_videos * 2}",
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = await asyncio.to_thread(ydl.extract_info, search_query, download=False)

                if result and "entries" in result:
                    urls = []
                    for entry in result["entries"]:
                        if entry and "url" in entry:
                            urls.append(entry["url"])
                        elif entry and "id" in entry:
                            urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
                    return urls

            return []

        except ImportError:
            self.logger.error("yt-dlp not installed. Run: pip install yt-dlp")
            return []
        except Exception as e:
            self.logger.error(f"YouTube search failed: {e}")
            return []

    async def _download_audio(self, url: str, output_path: Path) -> bool:
        """
        Download audio from YouTube video.

        Args:
            url: YouTube video URL
            output_path: Path to save the audio file

        Returns:
            True if download successful, False otherwise
        """
        try:
            import yt_dlp

            ydl_opts = {
                **self.YDL_OPTS,
                "outtmpl": str(output_path.with_suffix("")),
                # Limit to first 60 seconds
                "download_ranges": lambda info_dict, ydl: [{"start_time": 0, "end_time": 60}],
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])

            # Check if file was created (yt-dlp adds extension)
            wav_path = output_path.with_suffix(".wav")
            if wav_path.exists():
                if output_path != wav_path:
                    wav_path.rename(output_path)
                return True

            # Try without extension
            if output_path.exists():
                return True

            self.logger.warning(f"Audio file not found after download: {output_path}")
            return False

        except Exception as e:
            self.logger.error(f"Download failed for {url}: {e}")
            return False

    def _analyze_audio(self, audio_path: Path) -> dict[str, Any] | None:
        """
        Analyze audio file using Librosa and madmom.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary with analysis results, or None if analysis fails
        """
        try:
            # Load audio with Librosa
            y, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE)

            if len(y) == 0:
                self.logger.warning("Empty audio file")
                return None

            # Primary analysis with Librosa
            librosa_results = self._analyze_with_librosa(y, sr)

            # Try madmom for advanced analysis
            madmom_results = None
            try:
                madmom_results = self._analyze_with_madmom(audio_path)
            except Exception as e:
                self.logger.info(f"madmom analysis failed, using librosa only: {e}")

            # Combine results
            analysis = {**librosa_results}

            if madmom_results:
                # Average tempo if madmom provides one
                if "tempo_bpm" in madmom_results:
                    analysis["tempo_bpm"] = (
                        librosa_results["tempo_bpm"] + madmom_results["tempo_bpm"]
                    ) / 2
                analysis["analysis_method"] = "combined"
                analysis["madmom_tempo"] = madmom_results.get("tempo_bpm")
            else:
                analysis["analysis_method"] = "librosa"

            return analysis

        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
            return None

    def _analyze_with_librosa(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Analyze audio using Librosa.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Dictionary with Librosa analysis results
        """
        # Tempo and beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Handle numpy array tempo (newer librosa versions return array)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Calculate swing ratio
        swing_ratio = self._calculate_swing_ratio(beat_times)

        # Calculate syncopation index
        syncopation_index = self._calculate_syncopation_index(onset_times, beat_times)

        # Estimate velocity distribution
        velocity_stats = self._estimate_velocity_distribution(y, sr, onset_frames)

        return {
            "tempo_bpm": tempo,
            "tempo_confidence": 0.8,  # Default confidence
            "swing_ratio": swing_ratio,
            "syncopation_index": syncopation_index,
            "velocity_mean": velocity_stats["mean"],
            "velocity_std": velocity_stats["std"],
            "beat_positions": beat_times.tolist(),
            "onset_positions": onset_times.tolist(),
        }

    def _analyze_with_madmom(self, audio_path: Path) -> dict[str, Any] | None:
        """
        Analyze audio using madmom for advanced beat tracking.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary with madmom analysis results, or None if unavailable
        """
        try:
            from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

            # Process audio with RNN
            proc = RNNBeatProcessor()
            act = proc(str(audio_path))

            # Track beats with DBN
            dbn = DBNBeatTrackingProcessor(fps=100)
            beats = dbn(act)

            if len(beats) < 2:
                return None

            # Calculate tempo from beat intervals
            intervals = np.diff(beats)
            if len(intervals) > 0:
                avg_interval = np.mean(intervals)
                tempo = 60.0 / avg_interval if avg_interval > 0 else 120.0
            else:
                tempo = 120.0

            return {
                "tempo_bpm": tempo,
                "beat_positions": beats.tolist(),
            }

        except ImportError:
            self.logger.debug("madmom not available")
            return None
        except Exception as e:
            self.logger.debug(f"madmom analysis error: {e}")
            return None

    def _calculate_swing_ratio(self, beat_times: np.ndarray) -> float:
        """
        Calculate swing ratio from consecutive beat pairs.

        Straight 8ths: ratio = 50% (equal spacing)
        Swung 8ths: ratio > 50% (long-short pattern)

        Args:
            beat_times: Array of beat times in seconds

        Returns:
            Swing ratio as percentage (50.0 to 75.0 typical range)
        """
        if len(beat_times) < 4:
            return 50.0  # Default to straight

        # Get consecutive intervals
        intervals = np.diff(beat_times)
        ratios = []

        for i in range(0, len(intervals) - 1, 2):
            long_note = intervals[i]
            short_note = intervals[i + 1]
            total = long_note + short_note
            if total > 0:
                ratio = (long_note / total) * 100
                ratios.append(ratio)

        return float(np.mean(ratios)) if ratios else 50.0

    def _calculate_syncopation_index(self, onsets: np.ndarray, beats: np.ndarray) -> float:
        """
        Calculate syncopation index based on onset vs beat alignment.

        Higher values indicate more syncopated playing.

        Args:
            onsets: Array of onset times in seconds
            beats: Array of beat times in seconds

        Returns:
            Syncopation index (0.0 to 1.0)
        """
        if len(onsets) == 0 or len(beats) == 0:
            return 0.0

        # Calculate minimum distance from each onset to nearest beat
        deviations = []
        for onset in onsets:
            distances = np.abs(beats - onset)
            min_distance = np.min(distances)
            deviations.append(min_distance)

        # Average deviation normalized by beat interval
        avg_deviation = np.mean(deviations)
        if len(beats) >= 2:
            avg_beat_interval = np.mean(np.diff(beats))
            if avg_beat_interval > 0:
                # Normalize: 0.5 beat interval = max syncopation
                syncopation = min(1.0, avg_deviation / (avg_beat_interval * 0.5))
                return float(syncopation)

        return 0.0

    def _estimate_velocity_distribution(
        self, y: np.ndarray, sr: int, onset_frames: np.ndarray
    ) -> dict[str, int]:
        """
        Estimate velocity distribution from RMS energy at onsets.

        Args:
            y: Audio time series
            sr: Sample rate
            onset_frames: Array of onset frame indices

        Returns:
            Dictionary with mean and std of estimated MIDI velocities
        """
        if len(onset_frames) == 0:
            return {"mean": 64, "std": 16}

        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]

        # Get RMS values at onset positions
        onset_energies = []
        for frame in onset_frames:
            if frame < len(rms):
                onset_energies.append(rms[frame])

        if not onset_energies:
            return {"mean": 64, "std": 16}

        # Normalize to MIDI velocity range (0-127)
        energies = np.array(onset_energies)
        max_energy = np.max(energies) if np.max(energies) > 0 else 1.0
        normalized = energies / max_energy

        # Map to MIDI velocity
        velocities = (normalized * 100 + 27).astype(int)  # Range ~27-127
        velocities = np.clip(velocities, 1, 127)

        return {
            "mean": int(np.mean(velocities)),
            "std": int(np.std(velocities)),
        }

    def _calculate_confidence(self, analysis: dict[str, Any]) -> float:
        """
        Calculate confidence score for analysis results.

        Args:
            analysis: Dictionary with analysis results

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.6

        # Boost for combined analysis method
        if analysis.get("analysis_method") == "combined":
            base_confidence += 0.15

        # Boost for more beats detected (better quality audio)
        num_beats = len(analysis.get("beat_positions", []))
        if num_beats > 20:
            base_confidence += 0.1
        elif num_beats > 10:
            base_confidence += 0.05

        # Boost for reasonable tempo range (60-200 BPM)
        tempo = analysis.get("tempo_bpm", 0)
        if 60 <= tempo <= 200:
            base_confidence += 0.1

        return min(1.0, base_confidence)
