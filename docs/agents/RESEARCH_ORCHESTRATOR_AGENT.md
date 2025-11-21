# Research Orchestrator Agent
# Sub-Agent Specification Document

**Agent Name:** Research Orchestrator
**Version:** 2.0.0
**Date:** 2025-11-17
**Parent:** Main Orchestrator Agent

---

## Agent Overview

### Purpose
Coordinate all data collection activities for a given artist, aggregate results, and produce a StyleProfile suitable for MIDI generation.

### Responsibilities
1. Manage 4 collector sub-agents (Papers, Articles, Audio, MIDI)
2. Run collectors in parallel for optimal performance
3. Aggregate results from all sources
4. Delegate to Style Profile Builder
5. Store StyleProfile in database
6. Report progress to Main Orchestrator
7. Handle collector failures gracefully

---

## Architecture

### File Location
`src/research/orchestrator.py`

### Dependencies
```python
from typing import Dict, List, Optional
import asyncio
import logging
from dataclasses import dataclass

from src.research.collectors.papers import ScholarPaperCollector
from src.research.collectors.articles import WebArticleCollector
from src.research.collectors.audio import AudioAnalysisCollector
from src.research.collectors.midi_db import MidiDatabaseCollector
from src.research.profile_builder import StyleProfileBuilder
from src.database.manager import DatabaseManager
```

---

## Interface Definition

### Input Schema
```python
@dataclass
class ResearchRequest:
    artist_name: str
    depth: str = "full"  # "quick" or "full"
    priority: int = 5  # 1-10
    timeout_minutes: int = 20
```

### Output Schema
```python
@dataclass
class StyleProfile:
    artist_name: str
    text_description: str
    quantitative_params: dict
    midi_templates: List[Path]
    embedding: np.ndarray
    confidence_score: float
    sources_count: dict
    created_at: datetime
    updated_at: datetime
```

---

## Implementation

### Class Structure

```python
class ResearchOrchestrator:
    """
    Coordinates all research collectors and builds StyleProfile.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        config: dict
    ):
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize collectors
        self.collectors = {
            'papers': ScholarPaperCollector(config['research']),
            'articles': WebArticleCollector(config['research']),
            'audio': AudioAnalysisCollector(config['research']),
            'midi': MidiDatabaseCollector(config['research'])
        }

        # Initialize profile builder
        self.profile_builder = StyleProfileBuilder(config)

        # Progress tracking
        self.progress_callback = None

    async def research_artist(
        self,
        request: ResearchRequest,
        progress_callback: Optional[Callable] = None
    ) -> StyleProfile:
        """
        Main entry point for artist research.

        Args:
            request: Research request with artist name and parameters
            progress_callback: Optional callback for progress updates

        Returns:
            StyleProfile ready for generation

        Raises:
            ResearchError: If research fails or produces low-quality results
        """
        self.progress_callback = progress_callback
        self.logger.info(f"Starting research for: {request.artist_name}")

        try:
            # Phase 1: Data Collection (0-80%)
            self._update_progress(0, "Starting data collection...")
            sources = await self._collect_all_sources(request)

            # Phase 2: Profile Building (80-95%)
            self._update_progress(80, "Building style profile...")
            profile = await self.profile_builder.build(
                artist_name=request.artist_name,
                sources=sources
            )

            # Phase 3: Validation & Storage (95-100%)
            self._update_progress(95, "Validating and storing...")
            await self._validate_profile(profile)
            await self._store_profile(profile)

            self._update_progress(100, "Research complete!")
            self.logger.info(
                f"Research complete: {request.artist_name} "
                f"(confidence: {profile.confidence_score:.2f})"
            )

            return profile

        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            raise ResearchError(f"Failed to research {request.artist_name}: {e}")

    async def _collect_all_sources(
        self,
        request: ResearchRequest
    ) -> Dict[str, List[ResearchSource]]:
        """
        Run all collectors in parallel with timeout.
        """
        # Create tasks for each collector
        tasks = {
            collector_name: asyncio.create_task(
                self._run_collector_with_progress(
                    collector_name,
                    collector,
                    request.artist_name,
                    progress_weight
                )
            )
            for collector_name, collector, progress_weight in [
                ('papers', self.collectors['papers'], 0.20),
                ('articles', self.collectors['articles'], 0.20),
                ('audio', self.collectors['audio'], 0.25),
                ('midi', self.collectors['midi'], 0.15)
            ]
        }

        # Wait for all with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=request.timeout_minutes * 60
            )
        except asyncio.TimeoutError:
            self.logger.warning("Research timeout, using partial results")
            results = [
                task.result() if task.done() else []
                for task in tasks.values()
            ]

        # Map results back to collector names
        sources = {}
        for collector_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self.logger.warning(f"{collector_name} failed: {result}")
                sources[collector_name] = []
            else:
                sources[collector_name] = result

        return sources

    async def _run_collector_with_progress(
        self,
        name: str,
        collector: BaseCollector,
        artist_name: str,
        progress_weight: float
    ) -> List[ResearchSource]:
        """
        Run a single collector and update progress.
        """
        self._update_progress(
            base=0,
            message=f"Collecting {name}..."
        )

        try:
            sources = await collector.collect(artist_name)
            self.logger.info(f"{name}: Found {len(sources)} sources")
            return sources
        except Exception as e:
            self.logger.error(f"{name} failed: {e}")
            return []

    async def _validate_profile(self, profile: StyleProfile):
        """
        Validate StyleProfile meets minimum quality standards.
        """
        if profile.confidence_score < self.config['research']['quality_threshold']:
            self.logger.warning(
                f"Low confidence: {profile.confidence_score:.2f} "
                f"(threshold: {self.config['research']['quality_threshold']})"
            )

        total_sources = sum(profile.sources_count.values())
        if total_sources < self.config['research']['min_sources_total']:
            raise ResearchError(
                f"Insufficient sources: {total_sources} "
                f"(minimum: {self.config['research']['min_sources_total']})"
            )

    async def _store_profile(self, profile: StyleProfile):
        """
        Store StyleProfile in database.
        """
        await self.db.save_style_profile(profile)

        # Update artist record
        artist = await self.db.get_or_create_artist(profile.artist_name)
        artist.research_status = 'cached'
        artist.confidence_score = profile.confidence_score
        artist.sources_count = sum(profile.sources_count.values())
        artist.last_updated = datetime.now()
        await self.db.update_artist(artist)

    def _update_progress(self, percent: int, message: str):
        """
        Report progress to callback.
        """
        if self.progress_callback:
            self.progress_callback(percent, message)
```

---

## Collector Sub-Agents

### 1. Scholar Paper Collector
**File:** `src/research/collectors/papers.py`
**Purpose:** Search academic papers for drumming analysis
**APIs:** Semantic Scholar, arXiv, CrossRef
**Output:** Text descriptions, tempo data, style features

### 2. Web Article Collector
**File:** `src/research/collectors/articles.py`
**Purpose:** Scrape music journalism for interviews and reviews
**Sources:** Pitchfork, Rolling Stone, Drummerworld, Wikipedia
**Output:** Textual style descriptions, equipment mentions

### 3. Audio Analysis Collector
**File:** `src/research/collectors/audio.py`
**Purpose:** Analyze audio recordings for rhythm features
**Sources:** YouTube (via yt-dlp), SoundCloud
**Output:** Tempo, swing, syncopation, velocity distribution

### 4. MIDI Database Collector
**File:** `src/research/collectors/midi_db.py`
**Purpose:** Find existing MIDI files of artist's songs
**Sources:** BitMIDI, FreeMIDI, Musescore
**Output:** MIDI pattern templates

---

## Configuration

### Config File: `configs/generation.yaml`

```yaml
research:
  # Quality thresholds
  quality_threshold: 0.6  # Minimum confidence score
  min_sources_total: 8  # Minimum total sources
  min_sources_per_type: 2  # Minimum per collector

  # Timeouts
  timeout_minutes: 20
  collector_timeout_minutes: 5

  # Parallelism
  max_concurrent_collectors: 4

  # Retry behavior
  max_retries: 3
  retry_delay_seconds: 5
```

---

## Error Handling

### Error Types

```python
class ResearchError(Exception):
    """Base exception for research failures."""
    pass

class InsufficientDataError(ResearchError):
    """Not enough sources found."""
    pass

class CollectorTimeoutError(ResearchError):
    """Collector exceeded timeout."""
    pass

class LowConfidenceError(ResearchError):
    """Profile quality too low."""
    pass
```

### Error Scenarios

**Scenario 1: All Collectors Fail**
```python
if all(len(sources) == 0 for sources in sources.values()):
    raise ResearchError(f"No sources found for {artist_name}")
```

**Scenario 2: Partial Collector Failure**
```python
# Continue with available sources, log warnings
if len(sources['papers']) == 0:
    logger.warning("No papers found, using other sources")
```

**Scenario 3: Low Confidence**
```python
if profile.confidence_score < 0.6:
    raise LowConfidenceError(
        f"Confidence too low: {profile.confidence_score:.2f}"
    )
```

---

## Progress Reporting

### Progress Stages

```python
PROGRESS_STAGES = {
    0: "Starting research...",
    10: "Searching papers...",
    30: "Scraping articles...",
    50: "Analyzing audio...",
    65: "Searching MIDI databases...",
    80: "Building profile...",
    90: "Calculating confidence...",
    95: "Storing in database...",
    100: "Research complete!"
}
```

### Callback Interface

```python
def progress_callback(percent: int, message: str):
    """
    Called by Research Orchestrator to report progress.

    Args:
        percent: 0-100
        message: Human-readable status message
    """
    print(f"[{percent}%] {message}")
```

---

## Testing

### Unit Tests

```python
# tests/unit/test_research_orchestrator.py

@pytest.mark.asyncio
async def test_research_artist_success(mock_collectors, mock_db):
    """Test successful artist research."""
    orchestrator = ResearchOrchestrator(mock_db, TEST_CONFIG)

    request = ResearchRequest(
        artist_name="John Bonham",
        depth="full"
    )

    profile = await orchestrator.research_artist(request)

    assert profile.artist_name == "John Bonham"
    assert profile.confidence_score > 0.6
    assert sum(profile.sources_count.values()) >= 8

@pytest.mark.asyncio
async def test_research_artist_insufficient_data(mock_collectors, mock_db):
    """Test research fails with insufficient data."""
    # Mock collectors to return empty results
    for collector in mock_collectors.values():
        collector.collect = AsyncMock(return_value=[])

    orchestrator = ResearchOrchestrator(mock_db, TEST_CONFIG)
    request = ResearchRequest(artist_name="Unknown Artist")

    with pytest.raises(InsufficientDataError):
        await orchestrator.research_artist(request)

@pytest.mark.asyncio
async def test_research_artist_timeout(mock_collectors, mock_db):
    """Test research handles timeout gracefully."""
    # Mock one collector to hang
    mock_collectors['audio'].collect = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )

    orchestrator = ResearchOrchestrator(mock_db, TEST_CONFIG)
    request = ResearchRequest(
        artist_name="Travis Barker",
        timeout_minutes=1
    )

    # Should complete with partial results
    profile = await orchestrator.research_artist(request)

    assert profile.sources_count['audio'] == 0  # Failed collector
    assert profile.sources_count['papers'] > 0  # Other collectors succeeded
```

### Integration Tests

```python
# tests/integration/test_research_pipeline.py

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_research_pipeline(real_db):
    """Test full research pipeline with real APIs (slow)."""
    orchestrator = ResearchOrchestrator(real_db, PROD_CONFIG)

    request = ResearchRequest(
        artist_name="Travis Barker",
        depth="full"
    )

    profile = await orchestrator.research_artist(request)

    # Verify profile completeness
    assert profile.confidence_score > 0.0
    assert len(profile.text_description) > 100
    assert profile.quantitative_params['tempo_avg'] > 0
    assert len(profile.midi_templates) >= 0  # May be 0 if no MIDI found

    # Verify database storage
    stored_profile = await real_db.get_style_profile(request.artist_name)
    assert stored_profile.artist_name == profile.artist_name
```

---

## Performance Metrics

### Target Performance
- **Total Research Time:** < 20 minutes (80th percentile)
- **Collector Parallelism:** 4 concurrent
- **Minimum Sources:** 8 total, 2 per type
- **Confidence Score:** > 0.6 for usable profiles

### Monitoring

```python
# Log metrics for each research run
metrics = {
    'artist_name': profile.artist_name,
    'total_time_seconds': time_end - time_start,
    'sources_found': sum(profile.sources_count.values()),
    'confidence_score': profile.confidence_score,
    'collector_times': {
        'papers': papers_time,
        'articles': articles_time,
        'audio': audio_time,
        'midi': midi_time
    }
}

logger.info("Research metrics", extra=metrics)
```

---

## Augmentation Support

### Augment Existing Profile

```python
async def augment_artist(
    self,
    artist_name: str
) -> StyleProfile:
    """
    Add more sources to existing profile.

    Runs collectors again with broader search parameters.
    """
    # Load existing profile
    existing_profile = await self.db.get_style_profile(artist_name)

    if not existing_profile:
        raise ValueError(f"No existing profile for {artist_name}")

    # Collect additional sources
    self.logger.info(f"Augmenting {artist_name}")
    new_sources = await self._collect_all_sources(
        ResearchRequest(
            artist_name=artist_name,
            depth="full"  # Always use full depth for augmentation
        )
    )

    # Merge with existing sources
    all_sources = self._merge_sources(existing_profile, new_sources)

    # Rebuild profile
    updated_profile = await self.profile_builder.build(
        artist_name=artist_name,
        sources=all_sources
    )

    # Store updated profile
    await self._store_profile(updated_profile)

    self.logger.info(
        f"Augmentation complete: "
        f"{existing_profile.confidence_score:.2f} → {updated_profile.confidence_score:.2f}"
    )

    return updated_profile
```

---

## Communication Protocol

### Messages from Main Orchestrator

**Research Request:**
```json
{
  "command": "research_artist",
  "payload": {
    "artist_name": "John Bonham",
    "depth": "full",
    "timeout_minutes": 20
  },
  "task_id": "uuid-123",
  "sender": "main_orchestrator"
}
```

### Messages to Main Orchestrator

**Progress Update:**
```json
{
  "task_id": "uuid-123",
  "status": "in_progress",
  "progress": 45,
  "message": "Analyzing audio...",
  "sender": "research_orchestrator"
}
```

**Completion:**
```json
{
  "task_id": "uuid-123",
  "status": "completed",
  "progress": 100,
  "result": {
    "artist_id": 123,
    "confidence": 0.82,
    "sources_count": 24
  },
  "sender": "research_orchestrator"
}
```

**Error:**
```json
{
  "task_id": "uuid-123",
  "status": "failed",
  "error": "InsufficientDataError",
  "message": "Only 3 sources found, minimum 8 required",
  "sender": "research_orchestrator"
}
```

---

## Dependencies

### Python Packages
```
asyncio (stdlib)
aiohttp==3.9.3
beautifulsoup4==4.12.3
scrapy==2.11.1
librosa==0.10.1
madmom==0.16.1            # Advanced beat/tempo tracking (cross-platform)
yt-dlp==2024.3.10
numpy==1.26.4
scipy==1.12.0
spacy==3.7.4
sentence-transformers==2.3.1
```

### External APIs
- Semantic Scholar API (100 req/5 min)
- ArXiv API (no rate limit)
- CrossRef API (50 req/sec with token)
- YouTube (via yt-dlp, subject to ToS)

---

## Future Enhancements

1. **Smart Caching:** Cache partial results for faster retries
2. **Source Ranking:** Prioritize high-quality sources
3. **User Feedback:** Allow manual source submission
4. **Batch Research:** Research multiple artists in parallel
5. **Incremental Updates:** Daily background updates for cached artists

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Agent Status:** Specification Complete
**Next Step:** Implementation
