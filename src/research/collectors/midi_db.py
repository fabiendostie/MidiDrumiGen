"""
MIDI Database Collector for MidiDrumiGen v2.0

Downloads and analyzes MIDI files from online databases (BitMIDI, FreeMIDI)
to extract drum track patterns with timing and velocity data.

Story: E1.S4 - MIDI Database Collection
Epic: E1 - Research Pipeline
Priority: HIGH
Story Points: 5
"""

import asyncio
import re
import tempfile
from pathlib import Path
from urllib.parse import quote, urljoin

import aiohttp
import mido
from bs4 import BeautifulSoup

from .base import BaseCollector, CollectorError, ResearchSource

# General MIDI drum note mapping
GM_DRUMS = {
    35: "kick_acoustic",
    36: "kick_bass",
    38: "snare_acoustic",
    40: "snare_electric",
    42: "hihat_closed",
    44: "hihat_pedal",
    46: "hihat_open",
    49: "crash_1",
    51: "ride",
    57: "crash_2",
}

# Primary extraction targets
KICK_NOTES = [35, 36]
SNARE_NOTES = [38, 40]
HIHAT_NOTES = [42, 44, 46]

# Drum channel in 0-indexed MIDI (channel 10 in 1-indexed)
DRUM_CHANNEL = 9


class MidiDatabaseCollector(BaseCollector):
    """
    Collects and analyzes MIDI files from online databases.

    Uses:
        - BitMIDI for public MIDI files
        - FreeMIDI for user-uploaded MIDI content
        - mido 1.3.3 for MIDI parsing

    Acceptance Criteria:
        - Searches 2+ MIDI databases (BitMIDI, FreeMIDI)
        - Extracts drum track from MIDI files (channel 9)
        - Parses kick/snare/hihat patterns with timing and velocity
        - Stores MIDI file paths in ResearchSource.file_path
        - Returns List[ResearchSource] with source_type='midi'
    """

    # Maximum MIDI files to process per artist
    MAX_MIDI_FILES = 10

    # Rate limiting delay between requests (seconds)
    RATE_LIMIT_DELAY = 2.0

    # HTTP request timeout
    REQUEST_TIMEOUT = 30

    # User agent for requests
    USER_AGENT = "MidiDrumiGen/2.0 (Research Bot; +https://github.com/fabiendostie/MidiDrumiGen)"

    def __init__(self, timeout: int = 300, max_files: int = 10):
        """
        Initialize MIDI Database Collector.

        Args:
            timeout: Maximum time in seconds for collection (default: 5 min)
            max_files: Maximum number of MIDI files to process
        """
        super().__init__(timeout)
        self.max_files = min(max_files, self.MAX_MIDI_FILES)
        self._temp_dir = None

    async def collect(self, artist_name: str) -> list[ResearchSource]:
        """
        Collect and analyze MIDI files for the artist.

        Args:
            artist_name: Name of the artist/drummer to research

        Returns:
            List of ResearchSource objects with source_type='midi'

        Raises:
            CollectorError: If critical error occurs
        """
        self.logger.info(f"Starting MIDI database collection for artist: {artist_name}")

        results = []

        try:
            # Create temporary directory for MIDI files
            with tempfile.TemporaryDirectory() as temp_dir:
                self._temp_dir = Path(temp_dir)

                # Search all databases in parallel
                search_tasks = [
                    self._search_bitmidi(artist_name),
                    self._search_freemidi(artist_name),
                ]

                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                # Flatten results, ignoring errors
                midi_urls = []
                db_names = ["BitMIDI", "FreeMIDI"]
                for i, result in enumerate(search_results):
                    if isinstance(result, Exception):
                        self.logger.warning(f"Search failed for {db_names[i]}: {result}")
                        continue
                    midi_urls.extend(result)

                if not midi_urls:
                    self.logger.warning(f"No MIDI files found for {artist_name}")
                    return results

                # Limit to max files
                midi_urls = midi_urls[: self.max_files]
                self.logger.info(f"Found {len(midi_urls)} MIDI files to analyze")

                # Download and analyze each file
                for i, url in enumerate(midi_urls):
                    try:
                        # Rate limiting
                        if i > 0:
                            await asyncio.sleep(self.RATE_LIMIT_DELAY)

                        midi_path = await self._download_midi(url, i)
                        if not midi_path:
                            continue

                        drum_track = self._extract_drum_track(midi_path)
                        if not drum_track:
                            self.logger.info(f"No drum track found in {url}")
                            continue

                        patterns = self._analyze_patterns(drum_track, midi_path)
                        if not patterns:
                            continue

                        # Velocity and timing analysis
                        velocity_stats = self._analyze_velocity_distribution(patterns)
                        timing_stats = self._analyze_timing_patterns(patterns)

                        # Merge stats into patterns
                        patterns["velocity_stats"] = velocity_stats
                        patterns["timing_stats"] = timing_stats

                        confidence = self._calculate_midi_confidence(patterns)

                        results.append(
                            ResearchSource(
                                source_type="midi",
                                title=f"{artist_name} - MIDI Pattern {i + 1}",
                                url=url,
                                file_path=str(midi_path),
                                raw_content=None,  # MIDI is binary
                                extracted_data=patterns,
                                confidence=confidence,
                                metadata={
                                    "file_index": i,
                                    "database": self._get_database_name(url),
                                },
                            )
                        )

                    except Exception as e:
                        self.logger.warning(f"Failed to process {url}: {e}")
                        continue

            self.logger.info(
                f"Completed MIDI database collection: {len(results)} sources collected"
            )
            return results

        except Exception as e:
            raise CollectorError(f"MIDI collection failed: {e}") from e

    async def _search_bitmidi(self, artist_name: str) -> list[str]:
        """
        Search BitMIDI for MIDI files.

        Args:
            artist_name: Artist name to search for

        Returns:
            List of MIDI file download URLs
        """
        urls = []
        encoded_query = quote(artist_name)
        search_url = f"https://bitmidi.com/search?q={encoded_query}"

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": self.USER_AGENT}
                async with session.get(
                    search_url, headers=headers, timeout=self.REQUEST_TIMEOUT
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"BitMIDI search returned status {response.status}")
                        return urls

                    html = await response.text()
                    soup = BeautifulSoup(html, "lxml")

                    # Find MIDI file links
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        # BitMIDI links to MIDI pages like /midis/song-name
                        if "/midis/" in href and not href.endswith("/midis/"):
                            # Convert page URL to download URL
                            midi_page = urljoin("https://bitmidi.com", href)
                            urls.append(midi_page)

                            if len(urls) >= self.max_files:
                                break

        except TimeoutError:
            self.logger.warning("BitMIDI search timed out")
        except aiohttp.ClientError as e:
            self.logger.warning(f"BitMIDI search error: {e}")

        return urls

    async def _search_freemidi(self, artist_name: str) -> list[str]:
        """
        Search FreeMIDI for MIDI files.

        Args:
            artist_name: Artist name to search for

        Returns:
            List of MIDI file download URLs
        """
        urls = []
        encoded_query = quote(artist_name)
        search_url = f"https://freemidi.org/search?q={encoded_query}"

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": self.USER_AGENT}
                async with session.get(
                    search_url, headers=headers, timeout=self.REQUEST_TIMEOUT
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"FreeMIDI search returned status {response.status}")
                        return urls

                    html = await response.text()
                    soup = BeautifulSoup(html, "lxml")

                    # Find MIDI file links
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        # FreeMIDI links to getter pages
                        if "getter" in href or ".mid" in href.lower():
                            midi_url = urljoin("https://freemidi.org", href)
                            urls.append(midi_url)

                            if len(urls) >= self.max_files:
                                break

        except TimeoutError:
            self.logger.warning("FreeMIDI search timed out")
        except aiohttp.ClientError as e:
            self.logger.warning(f"FreeMIDI search error: {e}")

        return urls

    async def _download_midi(self, url: str, index: int) -> Path | None:
        """
        Download MIDI file from URL.

        Args:
            url: URL to download from
            index: File index for naming

        Returns:
            Path to downloaded file, or None if download failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": self.USER_AGENT}

                # For BitMIDI, we need to get the actual download link from the page
                if "bitmidi.com" in url and "/midis/" in url:
                    url = await self._get_bitmidi_download_url(session, url, headers)
                    if not url:
                        return None

                async with session.get(
                    url, headers=headers, timeout=self.REQUEST_TIMEOUT
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"Download failed with status {response.status}: {url}")
                        return None

                    content = await response.read()

                    # Verify it's a MIDI file
                    if not content.startswith(b"MThd"):
                        self.logger.warning(f"Not a valid MIDI file: {url}")
                        return None

                    # Save to temp directory
                    filename = f"midi_{index}.mid"
                    filepath = self._temp_dir / filename
                    filepath.write_bytes(content)

                    return filepath

        except TimeoutError:
            self.logger.warning(f"Download timed out: {url}")
        except aiohttp.ClientError as e:
            self.logger.warning(f"Download error: {e}")

        return None

    async def _get_bitmidi_download_url(
        self, session: aiohttp.ClientSession, page_url: str, headers: dict
    ) -> str | None:
        """
        Extract actual download URL from BitMIDI page.

        Args:
            session: aiohttp session
            page_url: BitMIDI MIDI page URL
            headers: Request headers

        Returns:
            Direct download URL or None
        """
        try:
            async with session.get(
                page_url, headers=headers, timeout=self.REQUEST_TIMEOUT
            ) as response:
                if response.status != 200:
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, "lxml")

                # Find download button/link
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if ".mid" in href.lower():
                        return urljoin("https://bitmidi.com", href)

                # Try to find in script data
                for script in soup.find_all("script"):
                    if script.string and ".mid" in script.string:
                        match = re.search(r'["\']([^"\']+\.mid[^"\']*)["\']', script.string)
                        if match:
                            return urljoin("https://bitmidi.com", match.group(1))

        except Exception as e:
            self.logger.warning(f"Failed to get BitMIDI download URL: {e}")

        return None

    def _extract_drum_track(self, midi_path: Path) -> mido.MidiTrack | None:
        """
        Extract drum track from MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            mido.MidiTrack with drum data, or None if not found
        """
        try:
            mid = mido.MidiFile(str(midi_path))
        except Exception as e:
            self.logger.warning(f"Failed to load MIDI file {midi_path}: {e}")
            return None

        # Method 1: Find track with channel 9 (GM drums)
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, "channel") and msg.channel == DRUM_CHANNEL:
                    return track

        # Method 2: Check track names for 'drum'
        for track in mid.tracks:
            if track.name and "drum" in track.name.lower():
                return track

        # Method 3: Check for program change to drum kit
        for track in mid.tracks:
            for msg in track:
                if msg.type == "program_change" and msg.channel == DRUM_CHANNEL:
                    return track

        return None

    def _analyze_patterns(self, track: mido.MidiTrack, midi_path: Path) -> dict | None:
        """
        Extract drum patterns from MIDI track.

        Args:
            track: mido.MidiTrack with drum data
            midi_path: Path to original MIDI file for metadata

        Returns:
            Dictionary with pattern data
        """
        patterns = {
            "kick": [],
            "snare": [],
            "hihat": [],
            "tempo": None,
            "time_signature": [4, 4],
            "ticks_per_beat": 480,
            "total_ticks": 0,
        }

        # Get ticks_per_beat from file
        try:
            mid = mido.MidiFile(str(midi_path))
            patterns["ticks_per_beat"] = mid.ticks_per_beat
        except Exception:
            pass

        current_time = 0
        for msg in track:
            current_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                note_data = {"time": current_time, "velocity": msg.velocity}

                if msg.note in KICK_NOTES:
                    patterns["kick"].append(note_data)
                elif msg.note in SNARE_NOTES:
                    patterns["snare"].append(note_data)
                elif msg.note in HIHAT_NOTES:
                    patterns["hihat"].append(note_data)

            elif msg.type == "set_tempo":
                # Convert microseconds per beat to BPM
                patterns["tempo"] = round(mido.tempo2bpm(msg.tempo), 1)

            elif msg.type == "time_signature":
                patterns["time_signature"] = [msg.numerator, msg.denominator]

        patterns["total_ticks"] = current_time

        # Check if we have any useful data
        total_notes = len(patterns["kick"]) + len(patterns["snare"]) + len(patterns["hihat"])
        if total_notes == 0:
            self.logger.warning("No drum notes found in track")
            return None

        return patterns

    def _analyze_velocity_distribution(self, patterns: dict) -> dict:
        """
        Analyze velocity distribution for each drum type.

        Args:
            patterns: Dictionary with kick/snare/hihat patterns

        Returns:
            Dictionary with velocity statistics
        """
        import statistics

        stats = {}

        for drum_type in ["kick", "snare", "hihat"]:
            velocities = [note["velocity"] for note in patterns.get(drum_type, [])]
            if velocities:
                stats[drum_type] = {
                    "mean": round(statistics.mean(velocities), 1),
                    "std": (round(statistics.stdev(velocities), 1) if len(velocities) > 1 else 0.0),
                    "min": min(velocities),
                    "max": max(velocities),
                    "count": len(velocities),
                }
            else:
                stats[drum_type] = {
                    "mean": 0,
                    "std": 0,
                    "min": 0,
                    "max": 0,
                    "count": 0,
                }

        return stats

    def _analyze_timing_patterns(self, patterns: dict) -> dict:
        """
        Analyze timing patterns and detect common rhythms.

        Args:
            patterns: Dictionary with kick/snare/hihat patterns

        Returns:
            Dictionary with timing analysis
        """
        ticks_per_beat = patterns.get("ticks_per_beat", 480)
        stats = {
            "pattern_type": "unknown",
            "swing_ratio": 50.0,
            "has_backbeat": False,
            "density": {},
        }

        # Calculate note density (notes per beat)
        total_ticks = patterns.get("total_ticks", 1)
        total_beats = total_ticks / ticks_per_beat if ticks_per_beat > 0 else 1

        for drum_type in ["kick", "snare", "hihat"]:
            count = len(patterns.get(drum_type, []))
            stats["density"][drum_type] = round(count / total_beats, 2) if total_beats > 0 else 0

        # Detect backbeat (snare on beats 2 and 4)
        snare_times = [note["time"] for note in patterns.get("snare", [])]
        if snare_times and ticks_per_beat > 0:
            # Check if snares fall on beats 2 and 4
            beat_positions = [t / ticks_per_beat for t in snare_times]
            on_2_4 = sum(
                1
                for pos in beat_positions
                if abs(pos % 4 - 2) < 0.1 or abs(pos % 4 - 4) < 0.1 or abs(pos % 4) < 0.1
            )
            stats["has_backbeat"] = on_2_4 >= len(snare_times) * 0.5

        # Calculate swing ratio from hihat timing
        hihat_times = [note["time"] for note in patterns.get("hihat", [])]
        if len(hihat_times) >= 4:
            intervals = []
            for i in range(len(hihat_times) - 1):
                intervals.append(hihat_times[i + 1] - hihat_times[i])

            # Swing ratio from consecutive pairs
            ratios = []
            for i in range(0, len(intervals) - 1, 2):
                total = intervals[i] + intervals[i + 1]
                if total > 0:
                    ratio = (intervals[i] / total) * 100
                    ratios.append(ratio)

            if ratios:
                import statistics

                stats["swing_ratio"] = round(statistics.mean(ratios), 1)

        # Detect pattern type
        kick_density = stats["density"].get("kick", 0)
        if kick_density >= 4:
            stats["pattern_type"] = "four_on_floor"
        elif kick_density >= 2 and stats["has_backbeat"]:
            stats["pattern_type"] = "rock_beat"
        elif kick_density < 2 and stats["has_backbeat"]:
            stats["pattern_type"] = "half_time"

        return stats

    def _calculate_midi_confidence(self, patterns: dict) -> float:
        """
        Calculate confidence score for MIDI analysis results.

        Args:
            patterns: Dictionary with pattern and analysis data

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.5

        # Boost for more notes (better pattern representation)
        total_notes = (
            len(patterns.get("kick", []))
            + len(patterns.get("snare", []))
            + len(patterns.get("hihat", []))
        )
        if total_notes > 100:
            base_confidence += 0.2
        elif total_notes > 50:
            base_confidence += 0.15
        elif total_notes > 20:
            base_confidence += 0.1

        # Boost for having all three drum types
        has_kick = len(patterns.get("kick", [])) > 0
        has_snare = len(patterns.get("snare", [])) > 0
        has_hihat = len(patterns.get("hihat", [])) > 0

        if has_kick and has_snare and has_hihat:
            base_confidence += 0.15
        elif (has_kick and has_snare) or (has_kick and has_hihat):
            base_confidence += 0.1

        # Boost for tempo information
        if patterns.get("tempo"):
            base_confidence += 0.05

        # Boost for velocity variation (indicates human playing)
        velocity_stats = patterns.get("velocity_stats", {})
        for drum_type in ["kick", "snare", "hihat"]:
            if velocity_stats.get(drum_type, {}).get("std", 0) > 5:
                base_confidence += 0.03

        return min(1.0, base_confidence)

    def _get_database_name(self, url: str) -> str:
        """
        Get database name from URL.

        Args:
            url: Source URL

        Returns:
            Database name string
        """
        if "bitmidi.com" in url:
            return "BitMIDI"
        elif "freemidi.org" in url:
            return "FreeMIDI"
        elif "musescore.com" in url:
            return "Musescore"
        return "Unknown"
