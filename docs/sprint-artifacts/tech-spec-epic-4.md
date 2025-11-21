# Epic Technical Specification: API Layer

Date: 2025-11-19
Author: Fabz
Epic ID: 4
Status: Draft

---

## Overview

Epic 4 implements the RESTful API layer for MidiDrumiGen v2.0 using FastAPI 0.115+, providing HTTP endpoints for all system operations. The API serves as the bridge between user interfaces (Max for Live device, future web dashboard) and backend services (Research, Generation, Database). It implements comprehensive request validation using Pydantic models, consistent error responses, rate limiting, and auto-generated OpenAPI documentation.

The API follows REST principles with JSON request/response bodies, proper HTTP status codes, and stateless operations. All endpoints support asynchronous processing for long-running tasks (research, generation) using Celery task queues with progress tracking via task IDs.

## Objectives and Scope

**In Scope:**
- FastAPI application initialization with middleware (CORS, logging, exception handling)
- Research endpoints (POST /research, GET /research/{artist}, POST /augment/{artist})
- Generation endpoints (POST /generate)
- Utility endpoints (GET /task/{task_id}, GET /artists, GET /similar/{artist}, GET /stats)
- Request validation using Pydantic v2 models
- Global error handling with consistent error response format
- Rate limiting (100 requests/user/hour)
- OpenAPI/Swagger documentation (auto-generated)
- Health check endpoint (GET /health)

**Out of Scope:**
- Authentication/authorization (future: API keys, JWT)
- WebSocket support for real-time updates (future)
- GraphQL endpoint (not planned)
- Admin panel (future web dashboard)
- API versioning beyond v1 (v2 endpoints in future releases)

**Success Criteria:**
- All endpoints return responses < 100ms (excluding async tasks)
- 99.9% uptime (excluding scheduled maintenance)
- Request validation catches 100% of malformed inputs
- Error messages are clear and actionable (user-friendly)
- OpenAPI docs generated automatically and accurate
- Rate limiting prevents abuse (max 100 req/hour per user)

## System Architecture Alignment

This epic aligns with the **API Gateway Layer** defined in ARCHITECTURE.md Section 2.1:

**Referenced Components:**
- FastAPI REST API Server (`src/api/main.py`)
- Route handlers (`src/api/routes/research.py`, `generate.py`, `utils.py`)
- Pydantic models (`src/api/models/`)
- Middleware (CORS, rate limiting, logging)

**Architectural Constraints:**
- All routes must be async (async def)
- Use Pydantic v2 models for validation
- Follow REST naming conventions (nouns, not verbs)
- HTTP status codes: 200 (OK), 201 (Created), 202 (Accepted), 400 (Bad Request), 404 (Not Found), 500 (Internal Error)
- JSON-only responses (no XML, HTML)

**Integration Points:**
- **Epic 1 (Research):** Triggers research tasks via Celery
- **Epic 2 (Generation):** Calls LLMProviderManager for pattern generation
- **Epic 3 (Database):** Queries/stores via DatabaseManager
- **Max for Live:** Receives HTTP requests from JavaScript bridge
- **Celery:** Submits long-running tasks, polls for completion

## Detailed Design

### Services and Modules

| Module | Responsibility | Input | Output | Owner |
|--------|---------------|-------|--------|-------|
| `src/api/main.py` | FastAPI app initialization, middleware setup | N/A | `app: FastAPI` | API |
| `src/api/routes/research.py` | Research endpoints (POST /research, GET /research, POST /augment) | HTTP requests | JSON responses | API |
| `src/api/routes/generate.py` | Generation endpoint (POST /generate) | HTTP requests | JSON responses | API |
| `src/api/routes/utils.py` | Utility endpoints (GET /task, /artists, /similar, /stats) | HTTP requests | JSON responses | API |
| `src/api/models/requests.py` | Pydantic request models | Raw JSON | Validated objects | API |
| `src/api/models/responses.py` | Pydantic response models | Python dicts | JSON responses | API |
| `src/api/middleware/rate_limit.py` | Rate limiting (100 req/hour) | HTTP headers | 429 or pass-through | API |
| `src/api/middleware/logging.py` | Request/response logging | HTTP requests | Logs | API |

### Data Models and Contracts

**Request Models (Pydantic v2):**

```python
from pydantic import BaseModel, Field, validator

class ResearchRequest(BaseModel):
    artist: str = Field(..., min_length=1, max_length=100, description="Artist name")
    depth: str = Field("full", regex="^(quick|full)$", description="Research depth")

    @validator('artist')
    def validate_artist(cls, v):
        if not v.strip():
            raise ValueError("Artist name cannot be empty")
        if not re.match(r'^[a-zA-Z0-9\s\-\']+$', v):
            raise ValueError("Artist name contains invalid characters")
        return v.strip()

class GenerateRequest(BaseModel):
    artist: str = Field(..., min_length=1, max_length=100)
    bars: int = Field(4, ge=1, le=16, description="Number of bars (1-16)")
    tempo: int = Field(120, ge=40, le=300, description="Tempo in BPM (40-300)")
    time_signature: tuple[int, int] = Field((4, 4), description="Time signature")
    variations: int = Field(4, ge=1, le=8, description="Number of variations (1-8)")
    provider: str = Field("auto", regex="^(auto|anthropic|google|openai)$")
    humanize: bool = Field(True, description="Apply humanization")

    @validator('time_signature')
    def validate_time_sig(cls, v):
        valid_sigs = [(4, 4), (3, 4), (5, 4), (6, 8), (7, 8)]
        if v not in valid_sigs:
            raise ValueError(f"Invalid time signature. Must be one of: {valid_sigs}")
        return v
```

**Response Models:**

```python
class ResearchResponse(BaseModel):
    task_id: str
    status: str = Field(..., regex="^(researching|augmenting)$")
    estimated_time_minutes: int

class ArtistInfoResponse(BaseModel):
    exists: bool
    confidence: float = Field(None, ge=0.0, le=1.0)
    last_updated: str = None
    sources_count: dict = None

class GenerateResponse(BaseModel):
    status: str = Field(..., regex="^(success|failed)$")
    generation_time_ms: int
    provider_used: str
    midi_files: list[str]
    confidence: float

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict = None
    timestamp: str
    request_id: str
```

### APIs and Interfaces

**Research Endpoints:**

```python
# POST /api/v1/research
@router.post("/research", response_model=ResearchResponse, status_code=202)
async def trigger_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db)
):
    """
    Trigger artist research (async task).

    Returns:
        202 Accepted with task_id for polling
    """
    # Check if already cached
    existing = await db.get_artist(request.artist)
    if existing and existing.research_status == "cached":
        raise HTTPException(
            status_code=400,
            detail=f"Artist '{request.artist}' already researched"
        )

    # Submit Celery task
    task = research_artist_task.delay(request.artist, request.depth)

    return ResearchResponse(
        task_id=task.id,
        status="researching",
        estimated_time_minutes=15 if request.depth == "full" else 5
    )

# GET /api/v1/research/{artist}
@router.get("/research/{artist}", response_model=ArtistInfoResponse)
async def get_artist_info(
    artist: str,
    db: DatabaseManager = Depends(get_db)
):
    """Check if artist is cached and get metadata."""
    artist_record = await db.get_artist(artist)

    if not artist_record or artist_record.research_status != "cached":
        return ArtistInfoResponse(exists=False)

    profile = await db.get_style_profile(artist)

    return ArtistInfoResponse(
        exists=True,
        confidence=profile.confidence_score,
        last_updated=artist_record.last_updated.isoformat(),
        sources_count=profile.sources_count
    )

# POST /api/v1/augment/{artist}
@router.post("/augment/{artist}", response_model=ResearchResponse, status_code=202)
async def augment_research(
    artist: str,
    db: DatabaseManager = Depends(get_db)
):
    """Add more sources to existing artist research."""
    existing = await db.get_artist(artist)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Artist '{artist}' not found")

    task = augment_artist_task.delay(artist)

    return ResearchResponse(
        task_id=task.id,
        status="augmenting",
        estimated_time_minutes=10
    )
```

**Generation Endpoint:**

```python
# POST /api/v1/generate
@router.post("/generate", response_model=GenerateResponse)
async def generate_pattern(
    request: GenerateRequest,
    db: DatabaseManager = Depends(get_db)
):
    """
    Generate MIDI drum patterns.

    Returns:
        200 OK with MIDI file paths
    """
    # Check if artist is cached
    profile = await db.get_style_profile(request.artist)
    if not profile:
        raise HTTPException(
            status_code=400,
            detail=f"Artist '{request.artist}' not researched. Call /research first."
        )

    # Generate patterns
    start_time = time.time()
    coordinator = HybridCoordinator()

    try:
        midi_files = await coordinator.generate(
            profile=profile,
            params=GenerationParams(
                bars=request.bars,
                tempo=request.tempo,
                time_signature=request.time_signature,
                variations=request.variations,
                provider=request.provider,
                humanize=request.humanize
            )
        )
    except GenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))

    generation_time_ms = int((time.time() - start_time) * 1000)

    # Log to database
    await db.log_generation(
        artist_id=profile.artist_id,
        provider_used=midi_files[0].provider_used,
        generation_time_ms=generation_time_ms,
        user_params=request.dict(),
        output_files=[str(f) for f in midi_files]
    )

    return GenerateResponse(
        status="success",
        generation_time_ms=generation_time_ms,
        provider_used=midi_files[0].provider_used,
        midi_files=[str(f) for f in midi_files],
        confidence=profile.confidence_score
    )
```

**Utility Endpoints:**

```python
# GET /api/v1/task/{task_id}
@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Poll Celery task status."""
    result = AsyncResult(task_id)

    if result.state == "PENDING":
        return {"task_id": task_id, "status": "in_progress", "progress": 0}
    elif result.state == "PROGRESS":
        return {
            "task_id": task_id,
            "status": "in_progress",
            "progress": result.info.get('current', 0),
            "current_step": result.info.get('status', '')
        }
    elif result.state == "SUCCESS":
        return {
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "result": result.result
        }
    elif result.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(result.info)
        }

# GET /api/v1/artists
@router.get("/artists")
async def list_artists(
    limit: int = Query(10, ge=1, le=100),
    db: DatabaseManager = Depends(get_db)
):
    """List cached artists."""
    artists = await db.get_all_artists(limit=limit)
    cached = [a for a in artists if a.research_status == "cached"]

    return {
        "total": len(artists),
        "cached": len(cached),
        "recent": [
            {
                "name": a.name,
                "confidence": a.confidence_score,
                "last_updated": a.last_updated.isoformat()
            }
            for a in cached[:limit]
        ]
    }

# GET /api/v1/similar/{artist}
@router.get("/similar/{artist}")
async def find_similar_artists(
    artist: str,
    limit: int = Query(5, ge=1, le=20),
    db: DatabaseManager = Depends(get_db)
):
    """Find artists with similar drumming styles."""
    similar = await db.find_similar_artists(artist, limit=limit)

    return {
        "artist": artist,
        "similar_artists": [
            {"name": s.name, "similarity": s.similarity_score}
            for s in similar
        ]
    }

# GET /api/v1/stats
@router.get("/stats")
async def get_stats(db: DatabaseManager = Depends(get_db)):
    """System usage statistics."""
    stats = await db.get_generation_stats()

    return {
        "total_generations": stats['total'],
        "avg_time_ms": stats['avg_time'],
        "total_cost_usd": stats['total_cost'],
        "provider_breakdown": stats['by_provider']
    }
```

### Workflows and Sequencing

**Research Request Flow:**
```
1. Client → POST /api/v1/research
   Body: {"artist": "John Bonham", "depth": "full"}

2. API validates request (Pydantic)
   → Check artist name format
   → Check depth value

3. API checks if artist already cached
   → Query database.get_artist()
   → If cached: Return 400 error

4. API submits Celery task
   → research_artist_task.delay(artist, depth)
   → Get task_id from Celery

5. API returns 202 Accepted
   Body: {"task_id": "uuid", "status": "researching", "estimated_time_minutes": 15}

6. Client polls GET /api/v1/task/{task_id} until complete
```

**Generation Request Flow:**
```
1. Client → POST /api/v1/generate
   Body: {
     "artist": "John Bonham",
     "bars": 4,
     "tempo": 120,
     "time_signature": [4, 4],
     "variations": 4
   }

2. API validates request (Pydantic)
   → Validate all parameter ranges

3. API checks if artist is cached
   → database.get_style_profile(artist)
   → If not found: Return 400 error

4. API calls HybridCoordinator.generate()
   → Synchronous (waits for completion)
   → Returns MIDI file paths

5. API logs generation to database
   → generation_history table

6. API returns 200 OK
   Body: {
     "status": "success",
     "generation_time_ms": 1847,
     "provider_used": "anthropic",
     "midi_files": ["path1.mid", "path2.mid", ...]
   }
```

## Non-Functional Requirements

### Performance

- **Response Time (non-async endpoints):** < 100ms (95th percentile)
  - GET /research/{artist}: < 50ms (database query)
  - GET /artists: < 100ms (database query with limit)
  - GET /similar/{artist}: < 200ms (vector similarity search)
  - GET /stats: < 150ms (aggregation query)

- **Async Task Submission:** < 50ms
  - POST /research: Submit task and return 202 immediately
  - POST /augment: Submit task and return 202 immediately

- **Concurrent Requests:** Support 100 simultaneous users
  - FastAPI async workers (4-8 workers)
  - Uvicorn with uvloop for performance

- **Throughput:** 1000+ requests/minute (excluding generation endpoint)

**Measurement:** Prometheus histogram metrics, CloudWatch performance insights

### Security

- **Input Validation:** 100% of malformed requests rejected at API layer
  - Pydantic validators catch invalid types, ranges, formats
  - Custom validators for artist names (alphanumeric + spaces, hyphens, apostrophes)

- **CORS Configuration:**
  - **Development:** Allow `http://localhost:*` (all ports)
  - **Production:** Whitelist specific origins only

- **Rate Limiting:** 100 requests per user per hour
  - Track by IP address (development) or API key (production)
  - Return 429 Too Many Requests with Retry-After header

- **SQL Injection Prevention:** Use SQLAlchemy ORM (parameterized queries)

- **API Keys (future):** X-API-Key header validation

### Reliability/Availability

- **Uptime Target:** 99.9% (excluding scheduled maintenance)
  - 43 minutes downtime/month allowed

- **Health Checks:**
  - GET /health endpoint checks:
    - Database connection
    - Redis connection
    - Celery workers alive
  - Return 503 if any dependency unhealthy

- **Graceful Degradation:**
  - If database read-only: Serve cached data, reject writes
  - If Celery down: Return 503 for research/augment, allow generation

- **Error Recovery:**
  - Retry database queries (max 3 retries)
  - Log all 500 errors for investigation

### Observability

- **Logging:**
  - Log all requests (method, path, status, latency)
  - Log validation errors (help debug bad clients)
  - Log 500 errors with full stack trace
  - Structured JSON logs (Loguru)

- **Metrics (Prometheus):**
  - `api_requests_total` (counter by method, endpoint, status)
  - `api_request_duration_seconds` (histogram by endpoint)
  - `api_active_connections` (gauge)
  - `api_rate_limit_exceeded_total` (counter)

- **Tracing:**
  - Request ID in all logs (UUID generated per request)
  - Trace requests across API → Celery → Database
  - Include request_id in error responses

- **Dashboards:**
  - Grafana dashboard: request rate, latency, error rate
  - Alert if error rate > 1% or latency > 500ms

## Dependencies and Integrations

**External Dependencies (requirements.txt):**
```python
# Web Framework
fastapi==0.115.4
uvicorn[standard]==0.32.1       # ASGI server
pydantic==2.10.3                # Validation
pydantic-settings==2.6.1
python-multipart==0.0.19        # File uploads (future)

# Middleware
fastapi-limiter==0.1.6          # Rate limiting
fastapi-cache2==0.2.1           # Response caching

# CORS
python-cors==1.0.0

# Async
aiohttp==3.11.10
httpx==0.28.1                   # For testing

# Celery integration
celery==5.4.0
redis==5.2.1

# Monitoring
prometheus-client==0.21.1
```

**Integration Points:**

1. **Epic 1 (Research Pipeline):**
   - POST /research → `research_artist_task.delay()`
   - POST /augment → `augment_artist_task.delay()`

2. **Epic 2 (Generation Engine):**
   - POST /generate → `HybridCoordinator.generate()`

3. **Epic 3 (Database):**
   - All endpoints query via `DatabaseManager`
   - GET /research → `get_artist()`, `get_style_profile()`
   - GET /similar → `find_similar_artists()`
   - GET /stats → `get_generation_stats()`

4. **Max for Live (Epic 6):**
   - JavaScript HTTP bridge calls all endpoints
   - Polling GET /task/{task_id} for progress

5. **Celery (Task Queue):**
   - Submit tasks via `.delay()`
   - Poll task status via `AsyncResult(task_id)`

**External APIs:** None (this epic is purely backend infrastructure)

## Acceptance Criteria (Authoritative)

**AC-1:** FastAPI app initialized with title "MidiDrumiGen API" v2.0.0
- App instance created in `main.py`
- OpenAPI docs available at `/docs`
- ReDoc available at `/redoc`

**AC-2:** CORS middleware allows localhost (development)
- Origins: `["http://localhost:*"]`
- Methods: `["*"]`
- Headers: `["*"]`
- Credentials: `True`

**AC-3:** All research endpoints return correct responses
- POST /research: 202 with task_id
- GET /research/{artist}: 200 or 404
- POST /augment/{artist}: 202 with task_id

**AC-4:** POST /generate validates all parameters
- Artist: required, 1-100 chars
- Bars: 1-16
- Tempo: 40-300
- Time signature: valid options only
- Variations: 1-8
- Provider: auto/anthropic/google/openai

**AC-5:** POST /generate returns 400 if artist not researched
- Error message: "Artist not researched. Call /research first."

**AC-6:** GET /task/{task_id} returns task status
- Pending: status="in_progress", progress=0
- Success: status="completed", progress=100, result
- Failure: status="failed", error

**AC-7:** GET /artists returns cached artists
- Total count
- Cached count
- Recent list with limit

**AC-8:** GET /similar/{artist} returns similar artists
- Uses vector similarity search
- Returns top N (default 5, max 20)

**AC-9:** Global exception handler catches all errors
- Returns 500 with error response format
- Includes request_id for tracing

**AC-10:** Rate limiting enforces 100 req/hour
- Returns 429 after limit exceeded
- Includes Retry-After header

**AC-11:** OpenAPI docs auto-generated and accurate
- All endpoints documented
- Request/response schemas included
- Example values provided

## Traceability Mapping

| AC | Spec Section | Component | Test Idea |
|----|-------------|-----------|-----------|
| AC-1 | Detailed Design | `main.py` | Unit: Verify app.title, app.version |
| AC-2 | Non-Functional Req | `main.py` CORS middleware | Unit: Test CORS headers in response |
| AC-3 | APIs & Interfaces | `routes/research.py` | Integration: Call endpoints, verify status codes |
| AC-4 | Data Models | `GenerateRequest` Pydantic model | Unit: Test validation with invalid params |
| AC-5 | APIs & Interfaces | `generate_pattern()` | Integration: Call without research, assert 400 |
| AC-6 | APIs & Interfaces | `get_task_status()` | Integration: Submit task, poll until complete |
| AC-7 | APIs & Interfaces | `list_artists()` | Integration: Query database, verify response |
| AC-8 | APIs & Interfaces | `find_similar_artists()` | Integration: Test vector similarity search |
| AC-9 | Non-Functional Req | Global exception handler | Unit: Raise exception, verify 500 response |
| AC-10 | Non-Functional Req | Rate limit middleware | Integration: Send 101 requests, verify 429 |
| AC-11 | Out of Scope (auto-generated) | FastAPI /docs | Manual: Verify docs render correctly |

## Risks, Assumptions, Open Questions

**Risks:**

1. **Rate Limiting Bypass (MEDIUM):**
   - **Risk:** Users bypass rate limits with multiple IPs
   - **Mitigation:** Implement API key authentication (future)
   - **Monitoring:** Track request patterns, ban suspicious IPs

2. **Celery Task Queue Overload (MEDIUM):**
   - **Risk:** Too many concurrent research tasks overwhelm workers
   - **Mitigation:** Task queue limits (max 10 concurrent research tasks)
   - **Monitoring:** Alert if queue depth > 50

3. **Database Connection Pool Exhaustion (LOW):**
   - **Risk:** High request volume depletes connection pool
   - **Mitigation:** Connection pooling (max 20 connections)
   - **Monitoring:** Track active connections

**Assumptions:**

1. Max for Live device is only client during MVP (no web dashboard)
2. All clients can handle async operations (submit task → poll status)
3. Users accept 15-20 minute wait for research (not instant)
4. CORS localhost-only is sufficient for development (no remote clients)

**Open Questions:**

1. **Q:** Should we implement API versioning (v1, v2) from the start?
   - **A:** Yes, prefix all endpoints with `/api/v1/` for future flexibility

2. **Q:** How to handle API breaking changes in future releases?
   - **A:** Maintain `/api/v1/` indefinitely, add `/api/v2/` for new features

3. **Q:** Should we support file uploads for custom MIDI templates?
   - **A:** Out of scope for v2.0 MVP, planned for v2.2.0

4. **Q:** Should we implement WebSocket for real-time progress updates?
   - **A:** Out of scope for v2.0 MVP, planned for v2.3.0 (real-time generation)

## Test Strategy Summary

**Unit Tests (pytest):**
- `test_app_initialization()`: Verify app.title, version, CORS config
- `test_request_validation()`: Test Pydantic validators with invalid inputs
- `test_exception_handler()`: Raise exception, verify 500 response format
- `test_rate_limit_middleware()`: Mock rate limiter, verify 429 response

**Integration Tests (pytest + httpx):**
- `test_research_endpoint()`: POST /research, verify 202 + task_id
- `test_get_artist_info()`: GET /research/{artist}, verify response
- `test_generate_endpoint()`: POST /generate, verify 200 + MIDI files
- `test_generate_uncached_artist()`: POST /generate for non-researched artist, verify 400
- `test_task_status_polling()`: Submit task, poll until complete
- `test_list_artists()`: GET /artists, verify response structure
- `test_similar_artists()`: GET /similar/{artist}, verify vector search
- `test_stats_endpoint()`: GET /stats, verify aggregation

**Performance Tests:**
- `test_response_time()`: Assert GET /artists < 100ms
- `test_concurrent_requests()`: Send 100 requests in parallel, verify all succeed
- `test_rate_limit_enforcement()`: Send 101 requests, verify 101st returns 429

**End-to-End Tests:**
- `test_full_research_flow()`: POST /research → poll /task → GET /research
- `test_full_generation_flow()`: POST /generate → verify MIDI files created

**Coverage Target:** 90%+ for all route handlers, 85%+ for middleware
