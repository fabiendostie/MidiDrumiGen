# Epic Technical Specification: Research Pipeline

Date: 2025-11-19
Author: Fabz
Epic ID: 1
Status: Draft

---

## Overview

Epic 1 implements the foundational Research Pipeline that enables MidiDrumiGen v2.0 to automatically collect and analyze drumming style information for any artist. This epic delivers a multi-source research system that aggregates data from academic papers, web articles, audio analysis, and MIDI databases to create comprehensive StyleProfiles. The research system is critical to v2.0's core value proposition: on-demand artist style generation without pre-trained models.

The Research Pipeline coordinates four specialized collector agents running in parallel, aggregates their findings using an LLM-enhanced StyleProfile builder, generates vector embeddings for similarity search, and persists all data in PostgreSQL with pgvector support. This eliminates the need for manual style research and enables unlimited artist coverage.

## Objectives and Scope

**In Scope:**
- Implementation of 4 data collectors: Scholar Paper, Web Article, Audio Analysis, and MIDI Database
- Research Orchestrator for parallel collector coordination
- StyleProfile Builder with LLM synthesis and vector embedding generation
- Integration with PostgreSQL database for profile storage
- Research API endpoints (POST /api/v1/research, GET /api/v1/research/{artist})
- Celery task queue integration for async research operations
- Augmentation feature to add sources to existing profiles
- Confidence scoring and quality validation

**Out of Scope:**
- MIDI pattern generation (covered in Epic 2: LLM Generation Engine)
- Max for Live UI integration (covered in Epic 6: Ableton Integration)
- Real-time research progress streaming (deferred to v2.1)
- Community-contributed research sources (Phase 3)
- Local LLM support for StyleProfile generation (Phase 3)

## System Architecture Alignment

This epic aligns with the **Orchestrator-Agent Architecture** pattern defined in ARCHITECTURE.md. The Research Orchestrator acts as the primary coordinator within the research layer, managing four specialized collector agents. All research operations are asynchronous (asyncio/aiohttp) and integrate with the existing Celery + Redis task queue infrastructure established in Epic 3.

**Architecture Dependencies:**
- Database Layer: Leverages Epic 3's PostgreSQL + pgvector setup for storing Artists, ResearchSources, and StyleProfiles
- Caching Layer: Uses Redis for task queue management and future response caching
- Main Orchestrator: Research Pipeline is invoked by the Main Orchestrator when cache misses occur

**Key Architectural Constraints:**
- All collectors must implement BaseCollector interface for consistency
- Research timeout: 20 minutes maximum (configurable via RESEARCH_TIMEOUT_MINUTES)
- Minimum data quality: 10+ total sources with confidence score ≥ 0.6
- Vector embeddings: 384 dimensions using sentence-transformers/all-MiniLM-L6-v2

## Detailed Design

### Services and Modules

| Module | Responsibility | Inputs | Outputs | Owner/Story |
|--------|---------------|--------|---------|-------------|
| **ResearchOrchestrator** | Coordinates all collectors, runs parallel async tasks, aggregates results, manages timeouts | artist_name: str, depth: ResearchDepth | StyleProfile object | E1.S6 |
| **ScholarPaperCollector** | Searches academic databases (Semantic Scholar, arXiv, CrossRef) for papers | artist_name: str | List[ResearchSource] | E1.S1 |
| **WebArticleCollector** | Scrapes music journalism sites (Pitchfork, Rolling Stone, Drummerworld, Wikipedia) | artist_name: str | List[ResearchSource] | E1.S2 |
| **AudioAnalysisCollector** | Downloads YouTube audio via yt-dlp, analyzes with Librosa + madmom | artist_name: str | List[ResearchSource] | E1.S3 |
| **MidiDatabaseCollector** | Searches MIDI databases (BitMIDI, FreeMIDI, Musescore) | artist_name: str | List[ResearchSource] | E1.S4 |
| **StyleProfileBuilder** | Aggregates research data, generates text description, creates embeddings | artist_name: str, List[List[ResearchSource]] | StyleProfile | E1.S5 |
| **ResearchAPIRouter** | FastAPI endpoints for research operations | HTTP requests | JSON responses | E1.S7 |

### Data Models and Contracts

**ResearchSource** (Pydantic model)
```python
class ResearchSource(BaseModel):
    source_type: Literal['paper', 'article', 'audio', 'midi']
    title: Optional[str]
    url: Optional[str]
    file_path: Optional[str]  # For MIDI files
    raw_content: str  # Text content or JSON
    extracted_data: dict  # Structured data (tempo, params, etc.)
    confidence: float  # 0.0-1.0
    collected_at: datetime
```

**StyleProfile** (SQLAlchemy model, PostgreSQL table: style_profiles)
```python
class StyleProfile(Base):
    __tablename__ = 'style_profiles'

    id: int (PRIMARY KEY)
    artist_id: int (FOREIGN KEY → artists.id, UNIQUE)
    text_description: str (TEXT, for LLM prompts)
    quantitative_params: dict (JSONB)
        - tempo_min, tempo_max, tempo_avg: int
        - swing_percent: float
        - ghost_note_prob: float
        - syncopation_level: float
        - velocity_mean, velocity_std: int
    midi_templates_json: dict (JSONB, array of file paths)
    embedding: vector(384) (pgvector, for similarity search)
    confidence_score: float (0.0-1.0, minimum 0.6 for usability)
    created_at, updated_at: datetime
```

**Artist** (existing from Epic 3, extended)
```python
class Artist(Base):
    __tablename__ = 'artists'

    id: int
    name: str (UNIQUE, indexed)
    research_status: str  # 'pending', 'researching', 'cached', 'failed'
    last_updated: datetime
    sources_count: int
    confidence_score: float
```

### APIs and Interfaces

**POST /api/v1/research**
- **Purpose:** Trigger research for new artist
- **Request Body:** `{"artist": str, "depth": "quick"|"full" (optional)}`
- **Response:** `202 Accepted {"task_id": uuid, "status": "researching", "estimated_time_minutes": int}`
- **Celery Task:** research_artist_task.apply_async(args=[artist_name, depth])
- **Timeout:** 20 minutes (RESEARCH_TIMEOUT_MINUTES)

**GET /api/v1/research/{artist}**
- **Purpose:** Check if artist is cached and get metadata
- **Response 200:** `{"exists": true, "confidence": float, "last_updated": datetime, "sources_count": {...}}`
- **Response 404:** `{"exists": false, "message": "Artist not found in cache"}`

**POST /api/v1/augment/{artist}**
- **Purpose:** Add more sources to existing StyleProfile
- **Response:** `202 Accepted {"task_id": uuid, "status": "augmenting", "current_sources": int, "target_sources": int}`
- **Celery Task:** augment_artist_task.apply_async(args=[artist_name])

**GET /api/v1/task/{task_id}**
- **Purpose:** Poll Celery task status and progress
- **Response:** `{"task_id": uuid, "status": "in_progress"|"completed"|"failed", "progress": int, "current_step": str, "result": {...}}`

**BaseCollector Interface** (Abstract Base Class)
```python
class BaseCollector(ABC):
    @abstractmethod
    async def collect(self, artist_name: str) -> List[ResearchSource]:
        """Collect research data for given artist."""
        pass

    def _calculate_confidence(self, data: dict) -> float:
        """Calculate confidence score for this source."""
        pass
```

### Workflows and Sequencing

**Research Flow (First-Time Artist)**
```
User Request (artist_name) → POST /api/v1/research
    ↓
FastAPI creates Celery task → research_artist_task.apply_async()
    ↓
ResearchOrchestrator.research_artist(artist_name)
    ↓
[Parallel Execution with asyncio.gather()]
    ├─→ ScholarPaperCollector.collect() → 3-5 min
    ├─→ WebArticleCollector.collect() → 3-5 min
    ├─→ AudioAnalysisCollector.collect() → 8-10 min
    └─→ MidiDatabaseCollector.collect() → 2-3 min
    ↓
All collectors return List[ResearchSource]
    ↓
StyleProfileBuilder.build(artist_name, all_sources)
    ├─→ Extract tempo range (from audio + papers + MIDI)
    ├─→ Extract swing percentage (from audio analysis)
    ├─→ Generate text description (from articles + papers)
    ├─→ Create vector embedding (SentenceTransformer)
    ├─→ Calculate confidence score (weighted by source types)
    └─→ Return StyleProfile object
    ↓
DatabaseManager.save_style_profile(profile)
    ├─→ Create/update Artist record (research_status='cached')
    ├─→ Insert StyleProfile
    └─→ Insert all ResearchSources
    ↓
Task completes → Result stored in Celery backend
    ↓
User polls GET /api/v1/task/{task_id} → "status": "completed"
```

**Augmentation Flow (Existing Artist)**
```
User Request (artist_name) → POST /api/v1/augment/{artist}
    ↓
Load existing StyleProfile from database
    ↓
Run collectors again (same parallel flow)
    ↓
Merge new ResearchSources with existing
    ↓
Rebuild StyleProfile with enhanced data
    ├─→ Confidence score typically increases
    ├─→ Text description enriched
    └─→ More MIDI templates added
    ↓
Update database (preserve old sources, add new)
    ↓
Return updated profile
```

**Sequence Diagram (Text Representation)**
```
User → API: POST /api/v1/research {"artist": "J Dilla"}
API → Celery: research_artist_task.apply_async()
API → User: 202 {"task_id": "abc-123"}

Celery Worker → ResearchOrchestrator: research_artist("J Dilla")
ResearchOrchestrator → [Collectors]: asyncio.gather(all_collectors)

[Parallel - 5-20 min]
  Scholar → SemanticScholar API: search papers
  Scholar → ResearchOrchestrator: [3 papers]

  Articles → Pitchfork/Wikipedia: scrape + NLP filter
  Articles → ResearchOrchestrator: [8 articles]

  Audio → YouTube (yt-dlp): download audio
  Audio → Librosa + madmom: analyze tempo/swing
  Audio → ResearchOrchestrator: [4 audio analyses]

  MIDI → BitMIDI: search + download
  MIDI → mido: parse patterns
  MIDI → ResearchOrchestrator: [2 MIDI files]

ResearchOrchestrator → StyleProfileBuilder: build(sources)
StyleProfileBuilder → SentenceTransformer: encode(description)
StyleProfileBuilder → ResearchOrchestrator: StyleProfile(confidence=0.82)

ResearchOrchestrator → DatabaseManager: save_style_profile()
DatabaseManager → PostgreSQL: INSERT INTO style_profiles, research_sources

Celery Worker → Celery Backend: store result

User → API: GET /api/v1/task/abc-123
API → Celery Backend: get task status
API → User: 200 {"status": "completed", "result": {...}}
```

## Non-Functional Requirements

### Performance

- **Research Completion Time:** < 20 minutes for 80th percentile of artists (target: 12 minutes average)
- **Parallel Collector Execution:** All 4 collectors run concurrently via asyncio.gather() to minimize total time
- **API Response Time:** Synchronous endpoints (GET /api/v1/research/{artist}) < 100ms
- **Database Query Performance:** StyleProfile lookup by artist name < 50ms (indexed on artists.name)
- **Vector Search Performance:** Similarity queries on 10,000+ profiles < 200ms (IVFFlat index)
- **Embedding Generation:** < 500ms per StyleProfile (sentence-transformers on CPU)
- **Memory Footprint:** < 2GB RAM per Celery worker during research operations
- **Concurrent Research Tasks:** Support 10 simultaneous artist research operations

### Security

- **API Authentication:** All research endpoints require valid API key (X-API-Key header) in production
- **Rate Limiting:** 100 requests per IP per hour for research endpoints (prevents abuse)
- **Input Sanitization:** Artist names validated with regex `^[a-zA-Z0-9\s\-'\.]+$` (max 100 chars)
- **External API Keys:** Stored in environment variables (SEMANTIC_SCHOLAR_API_KEY, etc.), never hardcoded
- **Database Access:** Research module uses read-write database user with least privilege
- **HTTPS Only:** All external API calls (Semantic Scholar, YouTube, etc.) use HTTPS
- **Data Scrubbing:** Remove potential PII from scraped web content before storage
- **SQL Injection Prevention:** All database queries use SQLAlchemy parameterized queries

### Reliability/Availability

- **Collector Fault Tolerance:** If 1-2 collectors fail, research continues with remaining sources (minimum 10 sources required)
- **Graceful Degradation:** Audio collector (slowest) can be skipped if timeout approaches 20 minutes
- **Retry Logic:** External API calls retry 3 times with exponential backoff (1s, 2s, 4s)
- **Timeout Handling:** Each collector has individual 5-minute timeout; total research timeout 20 minutes
- **Celery Task Persistence:** Research tasks stored in Redis, survive worker restarts
- **Database Connection Pooling:** SQLAlchemy pool_size=10, max_overflow=20 for connection resilience
- **Error Recovery:** Failed research tasks mark artist.research_status='failed' and log detailed errors
- **Idempotency:** Re-running research for same artist updates existing profile (no duplicates)

### Observability

- **Structured Logging:** All research operations logged with JSON format (loguru)
  - Log entries include: artist_name, collector_name, sources_found, confidence_score, duration_ms
- **Metrics Collection:** Prometheus metrics for:
  - `research_duration_seconds{collector}` (Histogram)
  - `research_sources_total{source_type}` (Counter)
  - `research_confidence_score{artist}` (Gauge)
  - `collector_failures_total{collector, error_type}` (Counter)
- **Progress Tracking:** Celery task progress updates every collector completion (25%, 50%, 75%, 100%)
- **Health Checks:** GET /health endpoint includes research pipeline status (collectors reachable)
- **Distributed Tracing:** OpenTelemetry spans for research flow (future enhancement)
- **Error Alerting:** Critical failures (all collectors fail) trigger alerts via logging

## Dependencies and Integrations

### Python Libraries (from requirements.txt)

**Research & Data Collection:**
- `beautifulsoup4==4.12.3` - HTML parsing for web scraping
- `lxml==5.3.0` - XML/HTML parser backend
- `scrapy==2.12.0` - Web crawling framework
- `newspaper3k==0.2.8` - Article content extraction
- `aiohttp==3.11.10` - Async HTTP client for parallel requests
- `httpx==0.28.1` - Modern HTTP client with async support

**Audio Analysis:**
- `librosa==0.10.2.post1` - Primary audio analysis (tempo, beat tracking, onset detection)
- `madmom==0.16.1` - Advanced beat tracking with RNN/DBN models
- `yt-dlp==2024.12.13` - YouTube audio download
- `soundfile==0.12.1` - Audio file I/O
- `audioread==3.0.1` - Multi-format audio reading

**NLP & Embeddings:**
- `spacy==3.8.2` - NLP for keyword extraction and content filtering
- `sentence-transformers==3.3.1` - Vector embeddings (all-MiniLM-L6-v2 model)
- `transformers==4.47.1` - HuggingFace library for embeddings

**MIDI Processing:**
- `mido==1.3.3` - MIDI file parsing and manipulation

**Database:**
- `sqlalchemy==2.0.36` - ORM for database operations
- `psycopg2-binary==2.9.10` - PostgreSQL adapter
- `asyncpg==0.30.0` - Async PostgreSQL driver
- `pgvector==0.3.6` - Vector similarity search extension

**Task Queue:**
- `celery==5.4.0` - Distributed task queue
- `redis==5.2.1` - Message broker and cache backend

**Scientific Computing:**
- `numpy==1.26.4` - Numerical arrays and operations
- `scipy==1.14.1` - Scientific algorithms (signal processing)
- `pandas==2.2.3` - Data manipulation for aggregation

### External APIs & Services

**Academic Paper Sources:**
- **Semantic Scholar API** - `https://api.semanticscholar.org/graph/v1/paper/search`
  - Rate limit: 100 requests per 5 minutes
  - Authentication: API key (optional, higher limits)
  - Free tier sufficient for MVP
- **ArXiv API** - `http://export.arxiv.org/api/query`
  - Rate limit: 1 request per 3 seconds (self-imposed)
  - No authentication required
  - Open access
- **CrossRef API** - `https://api.crossref.org/works`
  - Rate limit: 50 requests per second
  - Authentication: Optional (polite mode recommended)
  - Free for research

**Web Scraping Targets:**
- **Drummerworld** - `https://www.drummerworld.com/drummers/`
  - Comprehensive drummer biographies
  - Technique descriptions, equipment lists
  - Respect robots.txt, 2-second delay between requests
- **Wikipedia** - `https://en.wikipedia.org/wiki/`
  - Artist biographies, discographies
  - Technique mentions, style influences
  - Use Wikipedia API where possible
- **Pitchfork** - `https://pitchfork.com/search/`
  - Music journalism, album reviews
  - Artist interviews, style analysis
  - Rate limit: 1 request per 2 seconds
- **Rolling Stone** - `https://www.rollingstone.com/search/articles/`
  - Artist profiles, technique articles
  - Historical context
  - Rate limit: 1 request per 2 seconds

**Audio/Video Sources:**
- **YouTube** (via yt-dlp)
  - Live performances, drum cam videos
  - Download audio-only streams
  - Comply with YouTube Terms of Service (personal use)

**MIDI Databases:**
- **BitMIDI** - `https://bitmidi.com/search`
  - Free MIDI file repository
  - Artist-tagged drum patterns
- **FreeMIDI** - `https://freemidi.org/search/`
  - User-uploaded MIDI files
- **Musescore** - `https://musescore.com/sheetmusic`
  - Sheet music with MIDI export
  - Requires attribution

### System Integrations

**Epic 3 Dependencies (Database & Caching):**
- Requires Epic 3 completed (E3.S1, E3.S2, E3.S3)
- Uses database models: `Artist`, `StyleProfile`, `ResearchSource` (from E3.S1)
- Uses DatabaseManager CRUD operations (from E3.S2)
- Requires PostgreSQL with pgvector extension installed
- Requires Alembic migrations applied (from E3.S3)

**Future Epic Integration Points:**
- Epic 2 (LLM Generation): Consumes StyleProfile.text_description for prompt building
- Epic 3 (Vector Search): Uses StyleProfile.embedding for similarity queries (E3.S4)
- Epic 4 (API Layer): Research endpoints integrated into main FastAPI app
- Epic 5 (MIDI Export): May use MIDI templates from StyleProfile.midi_templates_json

### spaCy Model Dependency

**Required Download:**
```bash
python -m spacy download en_core_web_sm
```
- Model: en_core_web_sm (English, small, 12 MB)
- Used for: Named entity recognition, keyword extraction, content filtering
- Installed during setup, not in requirements.txt

## Acceptance Criteria (Authoritative)

### AC-1: Scholar Paper Collection (E1.S1)
1. ScholarPaperCollector class implemented in `src/research/collectors/papers.py`
2. Searches Semantic Scholar, arXiv, and CrossRef APIs for drumming style papers
3. Extracts tempo mentions using regex pattern `(\d+)\s*BPM`
4. Assigns confidence scores based on citation count (higher citations = higher confidence)
5. Implements rate limiting with exponential backoff for API protection
6. Returns List[ResearchSource] with source_type='paper'

### AC-2: Web Article Collection (E1.S2)
1. WebArticleCollector class implemented in `src/research/collectors/articles.py`
2. Scrapes 4+ configured sites: Drummerworld, Wikipedia, Pitchfork, Rolling Stone
3. Uses spaCy NLP to filter drumming-related content (min 3 drumming keywords)
4. Extracts equipment mentions and technique descriptions
5. Handles HTTP errors gracefully (404, 503) without crashing
6. Respects robots.txt and implements 2-second delays

### AC-3: Audio Analysis Collection (E1.S3)
1. AudioAnalysisCollector class implemented in `src/research/collectors/audio.py`
2. Downloads audio using yt-dlp from YouTube
3. Analyzes with Librosa (primary) and madmom (advanced beat tracking)
4. Extracts: tempo, swing ratio, syncopation index, velocity distribution
5. Cleans up temporary audio files after analysis (no storage bloat)

### AC-4: MIDI Database Collection (E1.S4)
1. MidiDatabaseCollector implemented in `src/research/collectors/midi_db.py`
2. Searches BitMIDI, FreeMIDI, and Musescore databases
3. Extracts drum track from MIDI files (channel 10 or drum program)
4. Parses kick/snare/hihat patterns with timing and velocity
5. Stores MIDI file paths in ResearchSource.file_path

### AC-5: Style Profile Builder (E1.S5)
1. StyleProfileBuilder class implemented in `src/research/profile_builder.py`
2. Aggregates data from all 4 collector types
3. Resolves conflicts in tempo data (filters outliers >2 std dev)
4. Generates text description suitable for LLM prompts
5. Creates 384-dim vector embedding using SentenceTransformer (all-MiniLM-L6-v2)
6. Calculates confidence score (0.0-1.0, minimum 0.6 for usability)
7. Returns complete StyleProfile object

### AC-6: Research Orchestrator (E1.S6)
1. ResearchOrchestrator class implemented in `src/research/orchestrator.py`
2. Coordinates all 4 collectors in parallel using asyncio.gather()
3. Implements 20-minute total timeout with per-collector 5-minute timeouts
4. Continues with partial results if 1-2 collectors fail (min 10 sources required)
5. Updates Celery task progress at each collector completion (25%, 50%, 75%, 100%)
6. Stores complete StyleProfile in database via DatabaseManager

### AC-7: Research API Endpoints (E1.S7)
1. POST /api/v1/research endpoint triggers Celery task and returns task_id
2. GET /api/v1/research/{artist} checks cache and returns metadata
3. POST /api/v1/augment/{artist} adds more sources to existing profile
4. GET /api/v1/task/{task_id} returns task status and progress
5. All endpoints validate inputs (artist name regex, max length 100 chars)
6. All endpoints return proper HTTP status codes (200, 202, 400, 404, 500)

### AC-8: Augmentation Feature (E1.S8)
1. Augmentation script `scripts/augment_style.py` implemented
2. Loads existing StyleProfile from database
3. Runs collectors again to gather 5+ additional sources
4. Merges new sources with existing (no duplicates)
5. Rebuilds StyleProfile with increased confidence score
6. Updates database while preserving old sources

## Traceability Mapping

| AC | PRD Section | Architecture Component | API/Module | Test Coverage |
|----|------------|------------------------|------------|---------------|
| AC-1 | FR-1.1 Scholar Papers | ScholarPaperCollector | src/research/collectors/papers.py | tests/unit/test_collector_papers.py |
| AC-2 | FR-1.2 Web Articles | WebArticleCollector | src/research/collectors/articles.py | tests/unit/test_collector_articles.py |
| AC-3 | FR-1.3 Audio Analysis | AudioAnalysisCollector | src/research/collectors/audio.py | tests/unit/test_collector_audio.py |
| AC-4 | FR-1.4 MIDI Database | MidiDatabaseCollector | src/research/collectors/midi_db.py | tests/unit/test_collector_midi.py |
| AC-5 | FR-1.5 StyleProfile Gen | StyleProfileBuilder | src/research/profile_builder.py | tests/unit/test_profile_builder.py |
| AC-6 | Architecture: Research Layer | ResearchOrchestrator | src/research/orchestrator.py | tests/integration/test_research_orchestrator.py |
| AC-7 | FR-5.1 Research Endpoints | ResearchAPIRouter | src/api/routes/research.py | tests/integration/test_api_research.py |
| AC-8 | US-002 Augmentation | Augmentation Script | scripts/augment_style.py | tests/integration/test_augmentation.py |

**PRD ↔ Architecture ↔ Implementation Mapping:**
- PRD User Story US-001 → Epic 1 → Stories E1.S1-E1.S8
- Architecture Section 3.2-3.4 (Collectors) → AC-1 through AC-4
- Architecture Section 4 (Profile Builder) → AC-5
- Architecture Section 2 (Research Orchestrator) → AC-6
- NFR-1 Performance (< 20 min research) → AC-6 timeout implementation
- NFR-3 Reliability → AC-6 fault tolerance

## Risks, Assumptions, Open Questions

### Risks
**HIGH:**
- **External API Availability:** Semantic Scholar, arXiv, and YouTube may have outages or rate limit changes
  - *Mitigation:* Implement robust retry logic, cache successful results, continue with partial data
- **YouTube Terms of Service:** yt-dlp usage may violate ToS for commercial applications
  - *Mitigation:* Use only for research/personal use initially, explore official YouTube Data API v3 for production
- **LLM API Rate Limits:** Research for popular artists may trigger rate limits during testing
  - *Mitigation:* Epic 1 doesn't use LLMs yet (deferred to Epic 2), only embeddings (local model)

**MEDIUM:**
- **Web Scraping Stability:** Sites like Pitchfork/Rolling Stone may change HTML structure
  - *Mitigation:* Abstract selectors into config files, implement fallback sources, unit test HTML parsing
- **Audio Analysis Accuracy:** Librosa/madmom may struggle with complex polyrhythms or poor audio quality
  - *Mitigation:* Set minimum confidence thresholds, require multiple audio sources for validation
- **MIDI Database Copyright:** Some MIDI files may be copyrighted
  - *Mitigation:* Use only as research references, don't redistribute, cite sources

**LOW:**
- **spaCy Model Download:** Users may forget to download en_core_web_sm
  - *Mitigation:* Add to setup script, document clearly in README, check on startup

### Assumptions
1. **Epic 3 Completion:** Assumes Epic 3 (Database & Caching) stories E3.S1-E3.S3 are fully completed
2. **Internet Connectivity:** Research pipeline requires stable internet for all collectors
3. **PostgreSQL Setup:** Assumes PostgreSQL 15+ with pgvector extension is installed and configured
4. **Redis Running:** Assumes Redis 7+ is running for Celery task queue
5. **Minimum Data Availability:** Assumes at least 10 total sources can be found for most artists
6. **English Language Bias:** Assumes most research sources are in English (non-English artists may have limited coverage)

### Open Questions
1. **Q: How to handle artists with minimal online presence (< 10 sources)?**
   - *Status:* RESOLVED - Return confidence score < 0.6, allow generation with warning to user
2. **Q: Should we cache negative results (artist not found)?**
   - *Status:* OPEN - Consider adding failed_research_cache table to avoid repeated 20-min timeouts
3. **Q: What confidence score threshold should block generation?**
   - *Status:* PROPOSED - Minimum 0.6, but allow 0.4-0.6 with user warning
4. **Q: How to handle duplicate artists (e.g., "Dave Grohl" vs "David Grohl")?**
   - *Status:* OPEN - Implement fuzzy artist name matching + user confirmation before research
5. **Q: Should audio files be cached for future re-analysis?**
   - *Status:* OPEN - Storage cost vs re-download cost tradeoff, defer to Phase 2

## Test Strategy Summary

### Unit Tests (tests/unit/)
**Collectors (E1.S1-E1.S4):**
- Mock external API responses (Semantic Scholar, arXiv, CrossRef)
- Test extraction logic (tempo regex, NLP filtering, MIDI parsing)
- Test error handling (404s, timeouts, malformed responses)
- Test confidence scoring algorithms
- **Coverage Target:** 85% for all collector modules

**StyleProfileBuilder (E1.S5):**
- Test tempo aggregation with outlier filtering
- Test text description generation with various source combinations
- Test embedding generation (mock SentenceTransformer)
- Test confidence calculation (weighted by source type and count)
- **Coverage Target:** 90%

**ResearchOrchestrator (E1.S6):**
- Test parallel collector execution with mocked collectors
- Test timeout handling (individual and total)
- Test partial failure scenarios (1-2 collectors fail, but min sources met)
- Test database persistence
- **Coverage Target:** 80%

### Integration Tests (tests/integration/)
**Slow Tests (marked with pytest.mark.slow):**
- Real API calls to Semantic Scholar, arXiv (use test artists with known results)
- Real web scraping (use static test pages or cached HTML)
- Real audio analysis (use 10-second test audio files)
- Real MIDI parsing (use test MIDI files in tests/fixtures/)
- **Run Frequency:** CI on merge to main, optional in local dev

**End-to-End Research Flow:**
- Test full research pipeline for test artist ("John Bonham" or "Dave Grohl")
- Verify database storage (Artist, StyleProfile, ResearchSource records)
- Verify Celery task creation and completion
- Verify API endpoints return correct data
- **Coverage Target:** Full user flow covered

### Performance Tests
**Research Completion Time:**
- Measure total time for 10 diverse artists
- Verify 80th percentile < 20 minutes
- Identify slowest collector (likely Audio)

**Database Query Performance:**
- Load 10,000 mock StyleProfiles
- Measure query time for artist lookup (target < 50ms)
- Measure vector similarity search (target < 200ms)

**Concurrent Research:**
- Run 10 simultaneous research tasks
- Verify no race conditions or deadlocks
- Measure memory usage per worker (target < 2GB)

### Test Data & Fixtures
**Test Artists (known drumming styles):**
- John Bonham (classic rock, powerful, triplet feel)
- Travis Barker (punk, fast, technical)
- J Dilla (hip-hop, loose swing, quantization)
- Questlove (funk, groove-oriented, syncopation)

**Mock Data:**
- tests/fixtures/mock_papers.json - Semantic Scholar responses
- tests/fixtures/mock_articles.html - Scraped article examples
- tests/fixtures/test_audio.wav - 10-second drum loop
- tests/fixtures/test_pattern.mid - Example MIDI drum pattern

### CI/CD Integration
**Pre-commit Hooks:**
- black formatting
- ruff linting
- mypy type checking

**GitHub Actions (or equivalent):**
- Run unit tests on every push
- Run integration tests (fast) on PR
- Run slow integration tests on merge to main
- Generate coverage report (fail if < 80%)
- Security scan (bandit, safety)

**Test Commands:**
```bash
# Unit tests (fast)
pytest tests/unit/ -v --cov=src.research --cov-report=html

# Integration tests (slow, requires network)
pytest tests/integration/ -v --slow

# Performance tests
pytest tests/performance/ -v --benchmark

# All tests with coverage
pytest tests/ --cov=src --cov-report=term-missing
```
