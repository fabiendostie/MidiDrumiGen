# Phase 4 Handover - Complete API Routes

## Project Context

**Project:** MidiDrumiGen - AI-powered MIDI drum pattern generator with producer style emulation
**Tech Stack:** PyTorch 2.4+, FastAPI, Celery, Redis, mido (for MIDI I/O)
**Current Status:** Phase 3 Complete (Model Loading and Inference)
**Next Phase:** Phase 4 - Complete API Routes

---

## What Has Been Completed

### Phase 1: System Architecture (FastAPI + Celery + Redis) ✅
- Project structure with all directories and `__init__.py` files
- FastAPI application skeleton (`src/api/main.py`)
- Celery worker setup (`src/tasks/worker.py`)
- Redis configuration and health checks
- CORS middleware and request logging
- Global error handling

### Phase 2: MIDI Export Pipeline ✅
- **`src/midi/export.py`** - Complete pattern-to-MIDI conversion
- **`src/midi/humanize.py`** - All humanization algorithms
- **`src/midi/validate.py`** - Pattern validation
- Producer styles: J Dilla, Metro Boomin, Questlove
- Test suite passing with 3 MIDI files generated

### Phase 3: Model Loading and Inference ✅
- **`src/inference/model_loader.py`** - LRU-cached model loading with GPU/CPU auto-detection
- **`src/models/styles.py`** - Producer style registry with 3 styles
- **`src/inference/generate.py`** - Pattern generation with temperature/top-k/top-p sampling
- **`src/inference/mock.py`** - Mock model for testing full pipeline
- **All 7 tests passing** - Device detection, style registry, generation, etc.

**Available Infrastructure:**
```python
# From Phase 3
from src.inference import load_model, generate_pattern
from src.models.styles import (
    get_style_id,
    get_style_params,
    list_available_styles,
    get_all_styles_info,
    validate_tempo_for_style
)

# From Phase 2
from src.midi.export import export_pattern_to_midi
from src.midi.humanize import apply_style_humanization
```

---

## Phase 4 Objectives

Implement complete REST API routes and Celery tasks that:
1. Accept pattern generation requests via FastAPI endpoints
2. Queue generation tasks asynchronously with Celery
3. Return task status and results
4. Integrate Phase 3 (inference) with Phase 2 (MIDI export)
5. Provide style catalog endpoint
6. Handle file downloads and cleanup
7. Implement comprehensive error handling and validation

---

## Tasks for Phase 4

### Task 4.1: Create Pydantic Models for API

**File to create:** `src/api/models.py`

**Requirements:**
1. Define request/response models using Pydantic
2. Include validation rules for all parameters
3. Match the expected API schema from project docs

**Models to Implement:**

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Tuple
from enum import Enum

class ProducerStyle(str, Enum):
    """Available producer styles."""
    J_DILLA = "J Dilla"
    METRO_BOOMIN = "Metro Boomin"
    QUESTLOVE = "Questlove"


class PatternGenerationRequest(BaseModel):
    """Request model for pattern generation."""

    producer_style: ProducerStyle = Field(
        ...,
        description="Producer style to emulate"
    )
    bars: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of bars to generate (1-32)"
    )
    tempo: int = Field(
        default=120,
        ge=60,
        le=200,
        description="Tempo in BPM (60-200)"
    )
    time_signature: Tuple[int, int] = Field(
        default=(4, 4),
        description="Time signature as (numerator, denominator)"
    )
    humanize: bool = Field(
        default=True,
        description="Apply humanization (timing, velocity, ghost notes)"
    )
    pattern_type: Optional[str] = Field(
        default="verse",
        description="Pattern type (intro, verse, chorus, bridge, outro)"
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (0.1-2.0, lower = more deterministic)"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )

    @validator('time_signature')
    def validate_time_signature(cls, v):
        """Validate time signature."""
        numerator, denominator = v
        if denominator not in [2, 4, 8, 16]:
            raise ValueError("Denominator must be 2, 4, 8, or 16")
        if numerator < 1 or numerator > 16:
            raise ValueError("Numerator must be between 1 and 16")
        return v

    class Config:
        schema_extra = {
            "example": {
                "producer_style": "J Dilla",
                "bars": 4,
                "tempo": 95,
                "time_signature": [4, 4],
                "humanize": True,
                "pattern_type": "verse",
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.9
            }
        }


class TaskResponse(BaseModel):
    """Response model for task creation."""

    task_id: str = Field(..., description="Celery task ID")
    status: str = Field(..., description="Task status (queued, processing, completed, failed)")
    message: str = Field(..., description="Human-readable status message")

    class Config:
        schema_extra = {
            "example": {
                "task_id": "abc123-def456-ghi789",
                "status": "queued",
                "message": "Pattern generation queued successfully"
            }
        }


class TaskStatusResponse(BaseModel):
    """Response model for task status."""

    task_id: str
    status: str
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    result: Optional[dict] = Field(None, description="Task result (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    created_at: Optional[str] = Field(None, description="Task creation timestamp")
    completed_at: Optional[str] = Field(None, description="Task completion timestamp")

    class Config:
        schema_extra = {
            "example": {
                "task_id": "abc123-def456-ghi789",
                "status": "completed",
                "progress": 100,
                "result": {
                    "midi_file": "output/patterns/abc123_j_dilla_4bars.mid",
                    "duration_seconds": 1.234,
                    "tokens_generated": 256,
                    "style": "J Dilla",
                    "bars": 4,
                    "tempo": 95
                }
            }
        }


class StyleInfo(BaseModel):
    """Information about a producer style."""

    name: str
    model_id: str
    description: str
    preferred_tempo_range: Tuple[int, int]
    humanization: dict

    class Config:
        schema_extra = {
            "example": {
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
            }
        }


class StylesListResponse(BaseModel):
    """Response model for styles list."""

    styles: List[StyleInfo]
    count: int

    class Config:
        schema_extra = {
            "example": {
                "count": 3,
                "styles": [
                    {
                        "name": "J Dilla",
                        "model_id": "j_dilla_v1",
                        "description": "Signature swing and soulful groove",
                        "preferred_tempo_range": [85, 95],
                        "humanization": {"swing": 62.0}
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    path: Optional[str] = Field(None, description="Request path")
```

**Verification:**
```powershell
./venv/Scripts/python -c "from src.api.models import PatternGenerationRequest, TaskResponse; print('Pydantic models OK')"
```

---

### Task 4.2: Create Celery Generation Task

**File to create:** `src/tasks/tasks.py`

**Requirements:**
1. Implement `generate_pattern_task()` as a Celery task
2. Integrate model loading (Phase 3) with MIDI export (Phase 2)
3. Handle errors gracefully with proper status updates
4. Save MIDI files to `output/patterns/` directory
5. Return task result with file path and metadata

**Function Signature:**

```python
from celery import Task
from src.tasks.worker import celery_app
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='src.tasks.tasks.generate_pattern')
def generate_pattern_task(
    self: Task,
    producer_style: str,
    bars: int,
    tempo: int,
    time_signature: tuple,
    humanize: bool = True,
    pattern_type: str = "verse",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict:
    """
    Generate drum pattern using ML model and export to MIDI.

    This is a Celery task that:
    1. Loads the appropriate model (with caching)
    2. Generates pattern tokens using the model
    3. Converts tokens to MIDI notes (placeholder until tokenizer ready)
    4. Applies humanization based on producer style
    5. Exports to MIDI file
    6. Returns file path and metadata

    Args:
        producer_style: Producer style name (e.g., "J Dilla")
        bars: Number of bars to generate
        tempo: Tempo in BPM
        time_signature: Time signature as (numerator, denominator)
        humanize: Apply humanization
        pattern_type: Pattern type (intro, verse, etc.)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter

    Returns:
        Dict with:
            - midi_file: Path to generated MIDI file
            - duration_seconds: Generation time
            - tokens_generated: Number of tokens
            - style: Producer style
            - bars: Number of bars
            - tempo: BPM
    """
    pass
```

**Key Implementation Points:**

1. **Update task state:**
```python
self.update_state(state='PROGRESS', meta={'progress': 25, 'status': 'Loading model...'})
```

2. **Load model with error handling:**
```python
from src.inference.model_loader import load_model
from src.models.styles import get_model_path

try:
    model_path = get_model_path(producer_style)
    model, metadata = load_model(model_path)
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    # Use mock model as fallback for testing
    from src.inference.mock import MockDrumModel
    model = MockDrumModel()
```

3. **Generate pattern:**
```python
from src.inference.generate import generate_pattern

tokens = generate_pattern(
    model=model,
    tokenizer=None,  # Placeholder
    num_bars=bars,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    device="cuda",
    style_id=0  # Map from producer_style
)
```

4. **Convert to MIDI notes (placeholder):**
```python
# TODO: Replace with actual tokenizer detokenization in Phase 5
# For now, create simple test pattern
notes = [
    {'pitch': 36, 'velocity': 100, 'time': 0},      # Kick
    {'pitch': 42, 'velocity': 70, 'time': 240},     # Hi-hat
    {'pitch': 38, 'velocity': 90, 'time': 480},     # Snare
    {'pitch': 42, 'velocity': 70, 'time': 720},     # Hi-hat
]
```

5. **Apply humanization:**
```python
from src.midi.humanize import apply_style_humanization

if humanize:
    notes = apply_style_humanization(notes, producer_style, tempo)
```

6. **Export to MIDI:**
```python
from src.midi.export import export_pattern_to_midi

# Create unique filename
task_id = self.request.id
filename = f"{task_id}_{producer_style.lower().replace(' ', '_')}_{bars}bars.mid"
output_path = Path("output/patterns") / filename
output_path.parent.mkdir(parents=True, exist_ok=True)

success = export_pattern_to_midi(
    notes=notes,
    output_path=output_path,
    tempo=tempo,
    time_signature=time_signature
)
```

7. **Return result:**
```python
return {
    'midi_file': str(output_path),
    'duration_seconds': duration,
    'tokens_generated': len(tokens),
    'style': producer_style,
    'bars': bars,
    'tempo': tempo,
}
```

**Reference Documentation:**
- `@docs .cursorcontext/02_architecture.md` - System architecture
- `@file src/inference/generate.py` - Pattern generation
- `@file src/midi/export.py` - MIDI export
- `@file src/tasks/worker.py` - Celery configuration

**Verification:**
```powershell
# Test import
./venv/Scripts/python -c "from src.tasks.tasks import generate_pattern_task; print('Celery task OK')"
```

---

### Task 4.3: Create Generation API Route

**File to create:** `src/api/routes/generate.py`

**Requirements:**
1. Implement `POST /api/v1/generate` endpoint
2. Validate request using Pydantic models
3. Queue Celery task
4. Return task ID immediately (async response)
5. Handle validation errors with clear messages

**Implementation:**

```python
"""Pattern generation API routes."""

import logging
from fastapi import APIRouter, HTTPException, status
from src.api.models import PatternGenerationRequest, TaskResponse
from src.tasks.tasks import generate_pattern_task
from src.models.styles import validate_tempo_for_style, StyleNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate drum pattern",
    description="Queue a drum pattern generation task. Returns immediately with task ID."
)
async def generate_pattern(request: PatternGenerationRequest) -> TaskResponse:
    """
    Generate drum pattern with specified producer style.

    This endpoint queues a generation task and returns immediately.
    Use the task_id to check status via GET /status/{task_id}.

    **Workflow:**
    1. Validate request parameters
    2. Queue Celery task for async processing
    3. Return task ID
    4. Client polls /status/{task_id} for completion

    **Example:**
    ```json
    POST /api/v1/generate
    {
      "producer_style": "J Dilla",
      "bars": 4,
      "tempo": 95,
      "humanize": true
    }
    ```

    **Response:**
    ```json
    {
      "task_id": "abc-123",
      "status": "queued",
      "message": "Pattern generation queued successfully"
    }
    ```
    """
    try:
        logger.info(f"Pattern generation request: {request.producer_style}, {request.bars} bars @ {request.tempo} BPM")

        # Validate tempo for style (warning only)
        validate_tempo_for_style(request.producer_style.value, request.tempo, warn_only=True)

        # Queue Celery task
        task = generate_pattern_task.delay(
            producer_style=request.producer_style.value,
            bars=request.bars,
            tempo=request.tempo,
            time_signature=request.time_signature,
            humanize=request.humanize,
            pattern_type=request.pattern_type,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        logger.info(f"Task queued: {task.id}")

        return TaskResponse(
            task_id=task.id,
            status="queued",
            message=f"Pattern generation queued successfully for {request.producer_style.value}"
        )

    except StyleNotFoundError as e:
        logger.error(f"Invalid style: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid producer style: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Failed to queue generation task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue generation task: {str(e)}"
        )
```

**Verification:**
```powershell
# Test import
./venv/Scripts/python -c "from src.api.routes.generate import router; print('Generate route OK')"
```

---

### Task 4.4: Create Status API Route

**File to create:** `src/api/routes/status.py`

**Requirements:**
1. Implement `GET /api/v1/status/{task_id}` endpoint
2. Query Celery task status
3. Return progress information
4. Handle different task states (PENDING, PROGRESS, SUCCESS, FAILURE)

**Implementation:**

```python
"""Task status API routes."""

import logging
from fastapi import APIRouter, HTTPException, status
from celery.result import AsyncResult
from src.api.models import TaskStatusResponse
from src.tasks.worker import celery_app

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get task status",
    description="Get status and result of a generation task"
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get status of a pattern generation task.

    **Task States:**
    - `PENDING`: Task queued, waiting to start
    - `PROGRESS`: Task running, check progress field
    - `SUCCESS`: Task completed, result available
    - `FAILURE`: Task failed, error message available

    **Example:**
    ```
    GET /api/v1/status/abc-123
    ```

    **Response (completed):**
    ```json
    {
      "task_id": "abc-123",
      "status": "completed",
      "progress": 100,
      "result": {
        "midi_file": "output/patterns/abc-123_j_dilla_4bars.mid",
        "duration_seconds": 1.234,
        "style": "J Dilla"
      }
    }
    ```
    """
    try:
        # Get task result
        task = AsyncResult(task_id, app=celery_app)

        logger.debug(f"Checking status for task {task_id}: {task.state}")

        # Map Celery state to response
        if task.state == 'PENDING':
            return TaskStatusResponse(
                task_id=task_id,
                status="pending",
                progress=0
            )

        elif task.state == 'PROGRESS':
            # Get custom progress metadata
            progress_data = task.info or {}
            return TaskStatusResponse(
                task_id=task_id,
                status="processing",
                progress=progress_data.get('progress', 0)
            )

        elif task.state == 'SUCCESS':
            return TaskStatusResponse(
                task_id=task_id,
                status="completed",
                progress=100,
                result=task.result
            )

        elif task.state == 'FAILURE':
            error_msg = str(task.info) if task.info else "Unknown error"
            return TaskStatusResponse(
                task_id=task_id,
                status="failed",
                error=error_msg
            )

        else:
            # Unknown state
            return TaskStatusResponse(
                task_id=task_id,
                status=task.state.lower(),
                progress=None
            )

    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task status: {str(e)}"
        )
```

---

### Task 4.5: Create Styles API Route

**File to create:** `src/api/routes/styles.py`

**Requirements:**
1. Implement `GET /api/v1/styles` endpoint
2. Return list of available producer styles
3. Include descriptions, tempo ranges, and humanization params

**Implementation:**

```python
"""Producer styles API routes."""

import logging
from fastapi import APIRouter, HTTPException, status
from src.api.models import StylesListResponse, StyleInfo
from src.models.styles import get_all_styles_info

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/styles",
    response_model=StylesListResponse,
    summary="List available styles",
    description="Get catalog of all available producer styles"
)
async def list_styles() -> StylesListResponse:
    """
    List all available producer styles with parameters.

    Returns complete information about each style including:
    - Model ID and description
    - Preferred tempo range
    - Humanization parameters (swing, timing, ghost notes, etc.)

    **Example:**
    ```
    GET /api/v1/styles
    ```

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
        }
      ]
    }
    ```
    """
    try:
        styles_dict = get_all_styles_info()

        styles_list = [
            StyleInfo(
                name=name,
                model_id=info['model_id'],
                description=info['description'],
                preferred_tempo_range=info['preferred_tempo_range'],
                humanization=info['humanization']
            )
            for name, info in styles_dict.items()
        ]

        logger.info(f"Returning {len(styles_list)} available styles")

        return StylesListResponse(
            styles=styles_list,
            count=len(styles_list)
        )

    except Exception as e:
        logger.error(f"Failed to retrieve styles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve styles: {str(e)}"
        )
```

---

### Task 4.6: Integrate Routes with FastAPI App

**File to update:** `src/api/main.py`

**Requirements:**
1. Import all route modules
2. Register routers with API prefix
3. Add appropriate tags for documentation

**Code to add (after line 133):**

```python
# Import routes
from src.api.routes import generate, status, styles

# Register routers
app.include_router(generate.router, prefix="/api/v1", tags=["generation"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])
app.include_router(styles.router, prefix="/api/v1", tags=["styles"])

logger.info("✓ API routes registered")
```

---

### Task 4.7: Create API Test Suite

**File to create:** `scripts/test_api.py`

**Requirements:**
Test script that verifies:
1. API server starts correctly
2. All endpoints are registered
3. Request validation works
4. Task queueing works (with mock model)
5. Status endpoint returns correct states
6. Styles endpoint returns all styles

**Test Structure:**

```python
"""Test suite for Phase 4: API Routes."""

import sys
from pathlib import Path
import requests
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

API_BASE_URL = "http://localhost:8000"


def test_root_endpoint():
    """Test root endpoint."""
    print("\n[TEST] Root endpoint")
    response = requests.get(f"{API_BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    print(f"[PASS] Root endpoint: {data['name']}")


def test_health_endpoint():
    """Test health check."""
    print("\n[TEST] Health endpoint")
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    print(f"[PASS] Health status: {data['status']}")


def test_styles_endpoint():
    """Test styles list endpoint."""
    print("\n[TEST] Styles endpoint")
    response = requests.get(f"{API_BASE_URL}/api/v1/styles")
    assert response.status_code == 200
    data = response.json()
    assert "styles" in data
    assert data["count"] >= 3
    print(f"[PASS] Styles count: {data['count']}")
    for style in data["styles"]:
        print(f"  - {style['name']}: {style['description']}")


def test_generate_endpoint():
    """Test pattern generation endpoint."""
    print("\n[TEST] Pattern generation")

    request_data = {
        "producer_style": "J Dilla",
        "bars": 4,
        "tempo": 95,
        "time_signature": [4, 4],
        "humanize": True,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9
    }

    response = requests.post(
        f"{API_BASE_URL}/api/v1/generate",
        json=request_data
    )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    task_id = data["task_id"]
    print(f"[PASS] Task queued: {task_id}")

    # Poll for completion
    print("Waiting for task completion...")
    max_attempts = 30
    for i in range(max_attempts):
        time.sleep(1)
        status_response = requests.get(f"{API_BASE_URL}/api/v1/status/{task_id}")
        status_data = status_response.json()

        print(f"  Attempt {i+1}: Status = {status_data['status']}")

        if status_data["status"] == "completed":
            print(f"[PASS] Task completed")
            print(f"  Result: {json.dumps(status_data['result'], indent=2)}")
            return status_data

        elif status_data["status"] == "failed":
            print(f"[FAIL] Task failed: {status_data.get('error')}")
            return None

    print(f"[WARN] Task did not complete within {max_attempts} seconds")
    return None


def run_all_tests():
    """Run all API tests."""
    print("="*70)
    print("PHASE 4 API TEST SUITE")
    print("="*70)

    print("\nNOTE: Ensure API server is running:")
    print("  uvicorn src.api.main:app --reload")
    print("  celery -A src.tasks.worker worker -Q gpu_generation --loglevel=info")

    try:
        test_root_endpoint()
        test_health_endpoint()
        test_styles_endpoint()
        test_generate_endpoint()

        print("\n" + "="*70)
        print("[SUCCESS] All API tests passed!")
        print("="*70)

    except AssertionError as e:
        print(f"\n[FAIL] Test assertion failed: {e}")
    except requests.ConnectionError:
        print("\n[FAIL] Could not connect to API server")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\n[FAIL] Test error: {e}")


if __name__ == "__main__":
    run_all_tests()
```

**Verification:**
```powershell
# Start API server (terminal 1)
uvicorn src.api.main:app --reload

# Start Celery worker (terminal 2)
celery -A src.tasks.worker worker -Q gpu_generation --loglevel=info

# Run tests (terminal 3)
./venv/Scripts/python scripts/test_api.py
```

---

## Important Context Files

Load these files when starting work:

1. **Project Overview:**
   - `@docs .cursorcontext/01_project_overview.md`

2. **Architecture:**
   - `@docs .cursorcontext/02_architecture.md`

3. **Existing Code:**
   - `@file src/api/main.py` - FastAPI app skeleton
   - `@file src/tasks/worker.py` - Celery configuration
   - `@file src/inference/generate.py` - Pattern generation (Phase 3)
   - `@file src/midi/export.py` - MIDI export (Phase 2)
   - `@file src/models/styles.py` - Style registry (Phase 3)

4. **Phase Completion Reports:**
   - `@file docs/PHASE_2_COMPLETION.md` - MIDI pipeline details
   - `@file docs/PHASE_3_COMPLETION.md` - Inference details

---

## Expected Deliverables

By the end of Phase 4, you should have:

1. ✅ `src/api/models.py` - Pydantic models for all requests/responses
2. ✅ `src/tasks/tasks.py` - Celery task for pattern generation
3. ✅ `src/api/routes/generate.py` - POST /generate endpoint
4. ✅ `src/api/routes/status.py` - GET /status/{task_id} endpoint
5. ✅ `src/api/routes/styles.py` - GET /styles endpoint
6. ✅ `src/api/main.py` - Updated with route registration
7. ✅ `scripts/test_api.py` - API test suite
8. ✅ All endpoints functional and tested
9. ✅ End-to-end generation working (request → task → MIDI file)

---

## Development Environment

**Python Version:** 3.11
**Virtual Environment:** `./venv` (already activated)
**Working Directory:** `C:\\Users\\lefab\\Documents\\Dev\\MidiDrumiGen`

**Key Dependencies (already installed):**
- FastAPI 0.121.0+
- Celery 5.5.3+
- Redis-py 5.0.0+
- PyTorch 2.4.1 with CUDA 12.4
- mido 1.3.3

**Quick Commands:**
```powershell
# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker
celery -A src.tasks.worker worker -Q gpu_generation -c 2 --loglevel=info

# Start Redis (if not running)
docker run -d -p 6379:6379 redis:7-alpine

# Test API
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/styles
```

---

## Known Constraints

1. **Tokenizer not yet implemented** - Phase 5 will add MidiTok tokenizer
   - For now, create simple test patterns in the Celery task
   - Use mock model for generation
   - Focus on API/task infrastructure

2. **No trained models yet** - Phase 6 will handle training
   - Use MockDrumModel as fallback
   - Handle missing model files gracefully
   - Still test full pipeline with mock data

3. **Redis must be running** - Required for Celery
   - Check Redis connection in health endpoint
   - Provide clear error messages if Redis unavailable

4. **Windows environment**
   - Use PowerShell for commands
   - Test file paths with Windows separators
   - Avoid Unicode characters in console output

---

## Success Criteria

Phase 4 is complete when:

- ✅ All Pydantic models defined with validation
- ✅ Celery task integrates Phase 3 (inference) + Phase 2 (MIDI export)
- ✅ POST /api/v1/generate queues tasks successfully
- ✅ GET /api/v1/status/{task_id} returns correct task states
- ✅ GET /api/v1/styles returns all producer styles
- ✅ End-to-end test: API request → Celery task → MIDI file generation
- ✅ Error handling comprehensive and informative
- ✅ API documentation (FastAPI /docs) displays correctly
- ✅ Test suite runs without errors
- ✅ Ready for Phase 5 (Tokenization)

---

## API Documentation

Once the API is running, documentation will be available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Next Steps After Phase 4

Once Phase 4 is complete, proceed to:

**Phase 5: Tokenization Pipeline**
- Implement MidiTok tokenizer
- Create tokenize/detokenize functions
- Replace placeholder pattern in Celery task with actual tokenizer
- Add tokenization to training pipeline

---

## Getting Help

If you encounter issues:

1. **Reference context documents:**
   ```
   @docs .cursorcontext/02_architecture.md
   How does the API integrate with Celery?
   ```

2. **Check existing code patterns:**
   ```
   @file src/api/main.py
   How is the FastAPI app structured?
   ```

3. **Review Phase 3 completion:**
   ```
   @file docs/PHASE_3_COMPLETION.md
   How do I use the model loader?
   ```

---

## Start Working on Phase 4

**Recommended approach:**

1. Create todo list for tracking progress
2. Start with Task 4.1 (Pydantic Models)
3. Implement Task 4.2 (Celery Task)
4. Create Task 4.3 (Generate Route)
5. Create Task 4.4 (Status Route)
6. Create Task 4.5 (Styles Route)
7. Update Task 4.6 (Integrate Routes)
8. Write Task 4.7 (Test Suite)
9. Start API server and Celery worker
10. Run tests and verify end-to-end flow

**First command to run:**
```
I'm starting Phase 4: Complete API Routes. I've read the handover document. Let me create a todo list and begin with Task 4.1 (Pydantic Models).
```

Good luck! 🚀
