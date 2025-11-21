# API Contracts - MidiDrumiGen v2.0

**Generated:** 2025-11-17
**Base URL:** `http://localhost:8000`
**API Version:** v1
**Framework:** FastAPI 0.115.4

---

## Overview

MidiDrumiGen v2.0 provides a RESTful API for generating drum patterns in the style of any producer or artist. The API uses an asynchronous task queue (Celery) for long-running generation operations.

**Architecture Pattern:** Request → Queue → Poll

1. Client POSTs generation request
2. Server queues Celery task, returns `task_id`
3. Client polls `/status/{task_id}` for completion
4. Result includes MIDI file path

---

## Base Endpoints

### GET `/`
**Summary:** API root
**Authentication:** None

**Response:**
```json
{
  "name": "Drum Pattern Generator API",
  "version": "0.1.0",
  "status": "running",
  "docs": "/docs"
}
```

---

### GET `/health`
**Summary:** Health check with dependency status
**Authentication:** None

**Response:**
```json
{
  "status": "healthy",
  "redis": "connected"
}
```

**Status Values:**
- `healthy` - All systems operational
- `degraded` - Redis disconnected, task queue unavailable

---

## Generation API

### POST `/api/v1/generate`
**Summary:** Generate drum pattern
**Tags:** `generation`
**Authentication:** None
**Status Code:** `202 ACCEPTED` (task queued)

**Request Body** (`application/json`):
```json
{
  "producer_name": "Timbaland",  // NEW: Dynamic producer research
  "bars": 4,                     // Required: 1-16 bars
  "tempo": 100,                  // Required: 40-300 BPM
  "time_signature": [4, 4],      // Optional: default [4, 4]
  "humanize": true,              // Optional: default true
  "pattern_type": "main",        // Optional: "main", "fill", "intro", "outro"
  "temperature": 0.8,            // Optional: 0.0-1.0, default 0.8
  "top_k": 50,                   // Optional: sampling parameter
  "top_p": 0.9                   // Optional: nucleus sampling
}
```

**Legacy Request (Backward Compatible):**
```json
{
  "producer_style": "J Dilla",   // OLD: Pre-defined style name
  "bars": 4,
  "tempo": 95,
  "humanize": true
}
```

**Response:**
```json
{
  "task_id": "abc-123-def-456",
  "status": "queued",
  "message": "Pattern generation queued for Timbaland (medium quality)"
}
```

**Workflow:**
1. Extract producer name from `producer_name` or fallback to `producer_style`
2. Research producer style using `ProducerResearchAgent` (cached or fresh, ~3s first time, <100ms cached)
3. Validate tempo against producer's typical range (warning only)
4. Queue Celery task with complete style profile
5. Return `task_id` for polling

**Error Responses:**

| Status Code | Condition | Detail |
|------------|-----------|---------|
| `400 BAD REQUEST` | Missing producer name | "Either 'producer_name' or 'producer_style' must be provided" |
| `500 INTERNAL SERVER ERROR` | Research failed | "Failed to research producer '{name}': {error}" |
| `500 INTERNAL SERVER ERROR` | Task queue failed | "Failed to queue generation task: {error}" |

---

### GET `/api/v1/status/{task_id}`
**Summary:** Get task status and result
**Tags:** `status`
**Authentication:** None

**Path Parameters:**
- `task_id` (string, required) - Celery task ID from `/generate`

**Response:**
```json
{
  "task_id": "abc-123-def-456",
  "status": "completed",
  "progress": 100,
  "result": {
    "midi_file": "output/patterns/abc-123_timbaland_4bars.mid",
    "duration_seconds": 1.234,
    "style": "Timbaland"
  }
}
```

**Task States:**

| Status | Description | Progress |
|--------|-------------|----------|
| `pending` | Task queued, waiting to start | 0 |
| `processing` | Task running | 0-100 |
| `completed` | Task finished successfully | 100 |
| `failed` | Task encountered error | null |

**Response Examples:**

**Pending:**
```json
{
  "task_id": "abc-123",
  "status": "pending",
  "progress": 0
}
```

**Processing:**
```json
{
  "task_id": "abc-123",
  "status": "processing",
  "progress": 45
}
```

**Completed:**
```json
{
  "task_id": "abc-123",
  "status": "completed",
  "progress": 100,
  "result": {
    "midi_file": "output/patterns/abc-123_j_dilla_4bars.mid",
    "duration_seconds": 1.234,
    "style": "J Dilla",
    "humanization_applied": true,
    "actual_tempo": 95
  }
}
```

**Failed:**
```json
{
  "task_id": "abc-123",
  "status": "failed",
  "error": "Model file not found: models/j_dilla_v1.pth"
}
```

**Error Responses:**

| Status Code | Condition | Detail |
|------------|-----------|---------|
| `500 INTERNAL SERVER ERROR` | Task lookup failed | "Failed to retrieve task status: {error}" |

---

### GET `/api/v1/styles`
**Summary:** List available producer styles
**Tags:** `styles`
**Authentication:** None

**Response:**
```json
{
  "count": 3,
  "styles": [
    {
      "name": "J Dilla",
      "model_id": "j_dilla_v1",
      "description": "Signature swing and soulful groove",
      "preferred_tempo_range": [85, 95],
      "humanization": {
        "swing": 62.0,
        "micro_timing_ms": 20.0,
        "ghost_note_prob": 0.4,
        "velocity_variation": 0.15
      }
    },
    {
      "name": "Timbaland",
      "model_id": "timbaland_v1",
      "description": "Syncopated percussion with triplet feels",
      "preferred_tempo_range": [95, 110],
      "humanization": {
        "swing": 55.0,
        "micro_timing_ms": 15.0,
        "ghost_note_prob": 0.3,
        "velocity_variation": 0.12
      }
    }
  ]
}
```

**Error Responses:**

| Status Code | Condition | Detail |
|------------|-----------|---------|
| `500 INTERNAL SERVER ERROR` | Style catalog failed | "Failed to retrieve styles: {error}" |

---

## Request Models (Pydantic)

### `PatternGenerationRequest`
```python
class PatternGenerationRequest(BaseModel):
    # Producer identification (one required)
    producer_name: Optional[str] = None       # NEW: Dynamic research
    producer_style: Optional[str] = None      # LEGACY: Pre-defined styles

    # Pattern parameters
    bars: int = Field(ge=1, le=16)            # Required: 1-16
    tempo: int = Field(ge=40, le=300)         # Required: 40-300 BPM
    time_signature: Tuple[int, int] = (4, 4)  # Optional

    # Generation options
    humanize: bool = True                     # Optional
    pattern_type: str = "main"                # Optional: main/fill/intro/outro
    temperature: float = Field(ge=0.0, le=1.0, default=0.8)
    top_k: Optional[int] = Field(ge=1, le=100, default=50)
    top_p: Optional[float] = Field(ge=0.0, le=1.0, default=0.9)

    def get_producer_name(self) -> str:
        """Extract producer name (dynamic or legacy)."""
        if self.producer_name:
            return self.producer_name
        elif self.producer_style:
            return self.producer_style
        else:
            raise ValueError("Either producer_name or producer_style required")
```

### `TaskResponse`
```python
class TaskResponse(BaseModel):
    task_id: str                              # Celery task UUID
    status: str                               # "queued"
    message: str                              # Human-readable status
```

### `TaskStatusResponse`
```python
class TaskStatusResponse(BaseModel):
    task_id: str                              # Celery task UUID
    status: str                               # pending/processing/completed/failed
    progress: Optional[int] = None            # 0-100 or null
    result: Optional[Dict[str, Any]] = None   # Generation result
    error: Optional[str] = None               # Error message if failed
```

### `StyleInfo`
```python
class StyleInfo(BaseModel):
    name: str                                 # Producer name
    model_id: str                             # Model identifier
    description: str                          # Style description
    preferred_tempo_range: Tuple[int, int]    # [min, max] BPM
    humanization: Dict[str, float]            # Humanization parameters
```

### `StylesListResponse`
```python
class StylesListResponse(BaseModel):
    styles: List[StyleInfo]                   # All available styles
    count: int                                # Total count
```

---

## Middleware

### CORS
**Configuration:**
- `allow_origins`: `["*"]` (configure for production)
- `allow_credentials`: `true`
- `allow_methods`: `["*"]`
- `allow_headers`: `["*"]`

### Request Logging
**Logs:**
- All HTTP requests with method, path, status code, duration
- Format: `→ METHOD /path` (request), `← METHOD /path [CODE] (duration)` (response)

### Global Error Handler
**Catches:** Uncaught exceptions
**Response:**
```json
{
  "error": "Internal server error",
  "message": "Exception message",
  "path": "/api/v1/generate"
}
```

---

## WebSocket (Future)

**Planned:** Real-time progress updates via WebSocket
**Endpoint:** `ws://localhost:8000/ws/tasks/{task_id}`
**Status:** Not implemented (use polling for now)

---

## Interactive Documentation

**Swagger UI:** http://localhost:8000/docs
**ReDoc:** http://localhost:8000/redoc

FastAPI automatically generates OpenAPI 3.0 specification with interactive testing.

---

## Authentication (Future)

**Current:** No authentication (development only)
**Planned:**
- API key authentication (v0.2.0)
- Rate limiting per API key (100 req/hour)
- User accounts with usage tracking (v0.3.0)

---

## Rate Limiting (Future)

**Planned:**
- Public: 10 requests per minute
- Authenticated: 100 requests per hour
- Premium: 1000 requests per hour

---

## Example Usage

### cURL
```bash
# Generate pattern
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "producer_name": "Timbaland",
    "bars": 4,
    "tempo": 100,
    "humanize": true
  }'

# Response: {"task_id": "abc-123", "status": "queued", ...}

# Poll status
curl http://localhost:8000/api/v1/status/abc-123

# List styles
curl http://localhost:8000/api/v1/styles
```

### Python
```python
import requests
import time

# Generate pattern
resp = requests.post('http://localhost:8000/api/v1/generate', json={
    'producer_name': 'J Dilla',
    'bars': 4,
    'tempo': 95,
    'humanize': True
})

task_id = resp.json()['task_id']

# Poll until complete
while True:
    status_resp = requests.get(f'http://localhost:8000/api/v1/status/{task_id}')
    status = status_resp.json()

    if status['status'] == 'completed':
        print(f"MIDI file: {status['result']['midi_file']}")
        break
    elif status['status'] == 'failed':
        print(f"Error: {status['error']}")
        break

    time.sleep(1)  # Poll every second
```

### JavaScript (Fetch API)
```javascript
// Generate pattern
const response = await fetch('http://localhost:8000/api/v1/generate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    producer_name: 'Timbaland',
    bars: 4,
    tempo: 100,
    humanize: true
  })
});

const {task_id} = await response.json();

// Poll status
const pollStatus = async () => {
  const statusResp = await fetch(`http://localhost:8000/api/v1/status/${task_id}`);
  const status = await statusResp.json();

  if (status.status === 'completed') {
    console.log('MIDI file:', status.result.midi_file);
  } else if (status.status === 'failed') {
    console.error('Error:', status.error);
  } else {
    setTimeout(pollStatus, 1000);  // Poll every second
  }
};

pollStatus();
```

---

**Last Updated:** 2025-11-17
**Verified By:** Brownfield Project Documentation Workflow
