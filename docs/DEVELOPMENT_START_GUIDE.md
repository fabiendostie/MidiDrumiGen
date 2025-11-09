# 🚀 Development Start Guide - Step by Step

**Complete guide to start developing the Drum Pattern Generator using Claude Code in Cursor IDE**

This guide walks you through implementing the core components of the project, step by step, using Claude Code for assistance.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [x] ✅ Project structure created (`src/` directory with all subdirectories)
- [x] ✅ All `__init__.py` files in place
- [x] ✅ Basic model files created (`src/models/transformer.py`)
- [x] ✅ FastAPI app skeleton (`src/api/main.py`)
- [x] ✅ Celery worker setup (`src/tasks/worker.py`)
- [x] ✅ MIDI constants (`src/midi/constants.py`, `io.py`)
- [ ] ⏳ Python 3.11 virtual environment activated
- [ ] ⏳ Dependencies installed
- [ ] ⏳ Cursor IDE open with project loaded

---

## Phase 1: Design System Architecture (FastAPI + Celery + Redis)

**Goal:** Set up the complete backend infrastructure following the documented architecture.

### Step 1.1: Setup Redis and Verify Connection

**Goal:** Ensure Redis is running and accessible.

**In Cursor Chat:**
```
@docs 02_architecture.md @docs 03_dependencies.md
I need to set up Redis for the Celery message broker. Show me:
1. How to start Redis (Docker command for Windows)
2. How to verify Redis connection in Python
3. Update configs/redis.py with connection testing
```

**Setup Redis:**
```powershell
# Start Redis using Docker
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Verify Redis is running
docker ps | findstr redis
```

**Verify Connection:**
```powershell
python -c "import redis; r = redis.Redis(host='localhost', port=6379); print('Redis connected:', r.ping())"
```

### Step 1.2: Complete FastAPI Application Setup

**Goal:** Ensure FastAPI app is properly configured with all middleware and settings.

**In Cursor Chat:**
```
@docs 02_architecture.md @file src/api/main.py
Review and complete the FastAPI application setup in src/api/main.py:
1. Add proper CORS configuration
2. Add request logging middleware
3. Add error handling middleware
4. Configure OpenAPI documentation
5. Add startup/shutdown events
6. Follow the architecture patterns from context docs
```

**What Claude Code will create:**
- Complete FastAPI app configuration
- Middleware setup
- Error handlers
- API documentation setup

**Verify:**
```powershell
# Start API server
uvicorn src.api.main:app --reload

# Test health endpoint
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Should show Swagger UI
```

### Step 1.3: Complete Celery Worker Configuration

**Goal:** Ensure Celery is properly configured with Redis and task routing.

**In Cursor Chat:**
```
@docs 02_architecture.md @file src/tasks/worker.py @file configs/redis.py
Complete the Celery worker configuration:
1. Verify Redis connection settings
2. Configure task routes for different queues
3. Add task result backend configuration
4. Set up task serialization
5. Add worker signal handlers for logging
6. Follow the exact configuration from architecture docs
```

**What Claude Code will create:**
- Complete Celery configuration
- Task routing setup
- Result backend configuration
- Logging setup

**Verify:**
```powershell
# Start Celery worker
celery -A src.tasks.worker worker --loglevel=info

# In another terminal, check registered tasks
celery -A src.tasks.worker inspect registered
```

### Step 1.4: Verify Data Flow Architecture

**Goal:** Test that the basic request → queue → worker flow works.

**In Cursor Chat:**
```
@docs 02_architecture.md
Create a simple test to verify the architecture data flow:
1. API receives request
2. Task is queued in Redis
3. Celery worker picks up task
4. Task executes and stores result
5. API can retrieve result
```

**Create Test Script:**
```powershell
# Create scripts/test_architecture.py
python scripts/test_architecture.py
```

---

## Phase 2: Complete MIDI Export Pipeline

### Step 2.1: Implement MIDI Export Function

**Goal:** Create the complete MIDI export pipeline that converts tokens to MIDI files.

**In Cursor Chat:**
```
@docs 04_midi_operations.md @file src/midi/io.py @file src/midi/constants.py
Implement the complete export_pattern function in src/midi/export.py that:
1. Takes tokenized pattern and converts to MIDI
2. Applies humanization if enabled
3. Validates the pattern
4. Saves as MIDI file with proper metadata
5. Follows the architecture from the context documents
```

**What Claude Code will create:**
- `src/midi/export.py` with `export_pattern()` function
- Integration with tokenizer (placeholder for now)
- Humanization application
- Pattern validation
- MIDI file creation with metadata

**Verify:**
```powershell
# In Cursor terminal
python -c "from src.midi.export import export_pattern; print('Import successful')"
```

### Step 2.2: Implement Humanization Functions

**Goal:** Add humanization algorithms for timing and velocity variation.

**In Cursor Chat:**
```
@docs 04_midi_operations.md
Create src/midi/humanize.py with functions for:
1. apply_swing() - Apply swing timing
2. apply_micro_timing() - Add random timing offsets
3. apply_velocity_variation() - Vary note velocities
4. add_ghost_notes() - Add subtle ghost notes
5. Use the exact algorithms from the context document
```

**What Claude Code will create:**
- `src/midi/humanize.py` with all humanization functions
- Producer-specific style parameters
- Integration with export pipeline

**Verify:**
```powershell
python -c "from src.midi.humanize import apply_swing; print('Humanization module OK')"
```

### Step 2.3: Implement Pattern Validation

**Goal:** Add validation to ensure generated patterns are musically valid.

**In Cursor Chat:**
```
@docs 04_midi_operations.md @file src/midi/constants.py
Create src/midi/validate.py with validate_drum_pattern() function that checks:
1. Note range (35-81 for GM drums)
2. Velocity range (1-127)
3. Pattern density (not too sparse/dense)
4. No impossible simultaneous hits (e.g., closed/open hi-hat)
5. Maximum simultaneous hits limit
```

**What Claude Code will create:**
- `src/midi/validate.py` with validation logic
- Error messages for invalid patterns
- Integration with export pipeline

**Verify:**
```powershell
python -c "from src.midi.validate import validate_drum_pattern; print('Validation module OK')"
```

---

## Phase 3: Model Loading and Inference

### Step 3.1: Create Model Loader with Caching

**Goal:** Implement model loading utilities with LRU caching.

**In Cursor Chat:**
```
@docs 02_architecture.md @docs 05_ml_pipeline.md @file src/models/transformer.py
Create src/inference/model_loader.py with:
1. load_model() function with @lru_cache decorator
2. Support for loading different style models
3. Device management (CUDA/CPU)
4. Error handling for missing models
5. Model metadata loading
```

**What Claude Code will create:**
- `src/inference/__init__.py`
- `src/inference/model_loader.py` with caching
- Model path resolution
- Device detection and management

**Verify:**
```powershell
python -c "from src.inference.model_loader import load_model; print('Model loader OK')"
```

### Step 3.2: Create Style Registry

**Goal:** Map producer style names to model IDs and parameters.

**In Cursor Chat:**
```
@docs 04_midi_operations.md
Create src/models/styles.py with:
1. PRODUCER_STYLES dictionary mapping style names to IDs
2. Style parameters (swing, micro_timing, etc.)
3. get_style_id() function
4. get_style_params() function
5. Include at least: J Dilla, Metro Boomin, Questlove
```

**What Claude Code will create:**
- `src/models/styles.py` with style registry
- Style parameter definitions
- Helper functions for style lookup

**Verify:**
```powershell
python -c "from src.models.styles import get_style_id; print(f'J Dilla ID: {get_style_id(\"J Dilla\")}')"
```

---

## Phase 4: Complete API Routes

### Step 4.1: Implement Pattern Generation Endpoint

**Goal:** Create the main API endpoint for pattern generation.

**In Cursor Chat:**
```
@docs 02_architecture.md @file src/api/main.py @file src/api/models/requests.py @file src/tasks/tasks.py
Create src/api/routes/generate.py with:
1. POST /api/v1/generate endpoint
2. Validates PatternRequest
3. Queues Celery task using tasks.generate_pattern
4. Returns TaskResponse with task_id
5. Follows FastAPI async patterns
6. Includes error handling
```

**What Claude Code will create:**
- `src/api/routes/generate.py` with endpoint
- Integration with Celery tasks
- Request validation
- Error responses

**Update main.py:**
```
@docs 02_architecture.md @file src/api/main.py
Add the generate router to the FastAPI app in src/api/main.py
```

**Verify:**
```powershell
# Start API server
uvicorn src.api.main:app --reload

# In another terminal, test endpoint
curl -X POST http://localhost:8000/api/v1/generate -H "Content-Type: application/json" -d '{\"producer_style\":\"J Dilla\",\"bars\":4,\"tempo\":95}'
```

### Step 4.2: Implement Task Status Endpoint

**Goal:** Allow clients to check generation task status.

**In Cursor Chat:**
```
@docs 02_architecture.md @file src/api/models/responses.py
Create src/api/routes/status.py with:
1. GET /api/v1/status/{task_id} endpoint
2. Queries Celery task status
3. Returns StatusResponse with progress
4. Handles completed tasks (returns MIDI path)
5. Handles failed tasks (returns error)
```

**What Claude Code will create:**
- `src/api/routes/status.py` with status endpoint
- Celery task status querying
- Progress tracking
- Error handling

**Update main.py:**
```
@file src/api/main.py
Add the status router to the FastAPI app
```

**Verify:**
```powershell
# Test status endpoint (use task_id from generate response)
curl http://localhost:8000/api/v1/status/{task_id}
```

### Step 4.3: Implement Styles Endpoint

**Goal:** Return list of available producer styles.

**In Cursor Chat:**
```
@docs 02_architecture.md @file src/models/styles.py
Create src/api/routes/admin.py with:
1. GET /api/v1/styles endpoint
2. Returns list of available styles from styles registry
3. Returns StylesResponse model
4. GET /health endpoint (already exists, verify it works)
```

**What Claude Code will create:**
- `src/api/routes/admin.py` with styles endpoint
- Integration with style registry
- Health check endpoint

**Update main.py:**
```
@file src/api/main.py
Add the admin router to the FastAPI app
```

**Verify:**
```powershell
curl http://localhost:8000/api/v1/styles
curl http://localhost:8000/health
```

---

## Phase 5: Complete Celery Tasks

### Step 5.1: Implement Pattern Generation Task

**Goal:** Complete the Celery task that generates patterns.

**In Cursor Chat:**
```
@docs 02_architecture.md @docs 05_ml_pipeline.md @file src/tasks/tasks.py @file src/inference/model_loader.py @file src/midi/export.py
Complete the generate_pattern Celery task in src/tasks/tasks.py:
1. Load model using model_loader
2. Generate tokens using model.generate()
3. Export to MIDI using export_pattern()
4. Handle GPU OOM errors (fallback to CPU)
5. Return result with MIDI path
6. Add proper logging
```

**What Claude Code will create:**
- Complete `generate_pattern` task implementation
- Error handling and retries
- Logging
- Result storage

**Verify:**
```powershell
# Start Celery worker
celery -A src.tasks.worker worker -Q gpu_generation -c 1 --loglevel=info

# Test task (in Python shell)
python -c "from src.tasks.tasks import generate_pattern; result = generate_pattern.delay({'producer_style':'J Dilla','bars':4,'tempo':95}); print(result.id)"
```

### Step 5.2: Implement Tokenization Task

**Goal:** Add task for MIDI tokenization (for training data preparation).

**In Cursor Chat:**
```
@docs 05_ml_pipeline.md @docs 03_dependencies.md
Complete the tokenize_midi Celery task in src/tasks/tasks.py:
1. Load MIDI file using mido
2. Tokenize using MidiTok REMI tokenizer
3. Save tokens to disk
4. Return token file path
5. Handle errors gracefully
```

**What Claude Code will create:**
- Complete `tokenize_midi` task
- MidiTok integration
- File I/O handling

---

## Phase 6: Training Pipeline

### Step 6.1: Create Dataset Class

**Goal:** Implement PyTorch Dataset for training.

**In Cursor Chat:**
```
@docs 05_ml_pipeline.md
Create src/training/dataset.py with DrumPatternDataset class:
1. Inherits from torch.utils.data.Dataset
2. Loads tokenized patterns from disk
3. Returns input_ids, labels, style_ids, attention_mask
4. Handles padding and truncation
5. Supports train/val split
```

**What Claude Code will create:**
- `src/training/dataset.py` with Dataset class
- Data loading logic
- Padding/truncation
- Style ID mapping

**Verify:**
```powershell
python -c "from src.training.dataset import DrumPatternDataset; print('Dataset class OK')"
```

### Step 6.2: Create Training Script

**Goal:** Implement the complete training loop.

**In Cursor Chat:**
```
@docs 05_ml_pipeline.md @file src/models/transformer.py @file src/training/dataset.py
Create src/training/train.py with:
1. train_model() function
2. Training loop with mixed precision
3. Validation loop
4. Checkpoint saving
5. Learning rate scheduling
6. Gradient clipping
7. Wandb logging (optional)
8. Follow the exact architecture from context docs
```

**What Claude Code will create:**
- `src/training/train.py` with training script
- Complete training loop
- Validation and checkpointing
- Logging integration

**Verify:**
```powershell
# Test training script (dry run)
python src/training/train.py --config configs/base.yaml --dry-run
```

---

## Phase 7: Testing and Validation

### Step 7.1: Write Unit Tests

**Goal:** Create tests for core functionality.

**In Cursor Chat:**
```
@docs 06_common_tasks.md
Create unit tests in tests/unit/:
1. tests/unit/test_midi_export.py - Test MIDI export functions
2. tests/unit/test_humanize.py - Test humanization algorithms
3. tests/unit/test_validate.py - Test pattern validation
4. tests/unit/test_model.py - Test model forward pass
5. Use pytest fixtures and follow testing best practices
```

**What Claude Code will create:**
- Test files with pytest
- Fixtures for test data
- Assertions and edge cases

**Run Tests:**
```powershell
pytest tests/unit/ -v
```

### Step 7.2: Integration Tests

**Goal:** Test API endpoints and task queue integration.

**In Cursor Chat:**
```
@docs 02_architecture.md
Create integration tests in tests/integration/:
1. tests/integration/test_api.py - Test API endpoints
2. tests/integration/test_celery.py - Test Celery tasks
3. Use test client for FastAPI
4. Mock external dependencies
```

**What Claude Code will create:**
- Integration test files
- Test client setup
- Mock configurations

**Run Tests:**
```powershell
pytest tests/integration/ -v
```

---

## Phase 8: End-to-End Testing

### Step 8.1: Test Complete Generation Flow

**Goal:** Verify the entire pipeline works end-to-end.

**Setup:**
```powershell
# Terminal 1: Start Redis (if not running)
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Terminal 2: Start API server
.\venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload

# Terminal 3: Start Celery worker
.\venv\Scripts\Activate.ps1
celery -A src.tasks.worker worker -Q gpu_generation -c 1 --loglevel=info
```

**Test Flow:**
```powershell
# 1. Generate pattern
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/generate" -Method POST -ContentType "application/json" -Body '{"producer_style":"J Dilla","bars":4,"tempo":95,"humanize":true}'
$taskId = $response.task_id
Write-Host "Task ID: $taskId"

# 2. Check status
Start-Sleep -Seconds 2
$status = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/status/$taskId"
Write-Host "Status: $($status.status)"

# 3. Wait for completion and check result
while ($status.status -eq "processing" -or $status.status -eq "queued") {
    Start-Sleep -Seconds 1
    $status = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/status/$taskId"
    Write-Host "Status: $($status.status)"
}

if ($status.status -eq "complete") {
    Write-Host "MIDI file: $($status.midi_path)"
}
```

---

## Development Workflow Summary

### Daily Development Routine

1. **Start Development Session:**
   ```powershell
   # Activate virtual environment
   .\venv\Scripts\Activate.ps1
   
   # Open Cursor IDE (if not already open)
   cursor .
   ```

2. **Before Implementing a Feature:**
   - Load relevant context documents in Cursor chat
   - Ask Claude Code to explain the architecture
   - Review existing code patterns

3. **Implementing:**
   - Ask Claude Code to implement using `@docs` references
   - Review generated code
   - Test immediately
   - Iterate based on results

4. **After Implementation:**
   - Run tests: `pytest tests/ -v`
   - Check linting: `ruff check src/`
   - Format code: `black src/`
   - Commit changes

### Quick Reference Commands

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run verification
python scripts/verify_installation.py

# Start API
uvicorn src.api.main:app --reload

# Start Celery worker
celery -A src.tasks.worker worker -Q gpu_generation -c 1 --loglevel=info

# Run tests
pytest tests/ -v --cov

# Lint code
ruff check src/
black src/

# Type check
mypy src/
```

---

## Troubleshooting

### Issue: Claude Code doesn't understand the architecture

**Solution:**
```
@docs 02_architecture.md
Explain the complete data flow from API request to MIDI file output.
```

### Issue: Generated code doesn't match patterns

**Solution:**
```
@docs 02_architecture.md @file src/api/main.py
Review this code and fix it to match our architecture patterns.
```

### Issue: Import errors

**Solution:**
```powershell
# Make sure you're in project root
cd C:\Users\lefab\Documents\Dev\MidiDrumiGen

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install missing dependencies
pip install -r requirements.txt
```

### Issue: Celery tasks not running

**Solution:**
1. Check Redis is running: `docker ps | findstr redis`
2. Check worker logs for errors
3. Verify task is registered: `celery -A src.tasks.worker inspect registered`

---

## Next Steps After Core Implementation

Once the core components are implemented:

1. **Add More Producer Styles**
   - Collect MIDI data for new producers
   - Add to style registry
   - Fine-tune models

2. **Improve Humanization**
   - Add more sophisticated algorithms
   - Style-specific humanization
   - Learn from real drum patterns

3. **Add Web UI**
   - Create React frontend
   - Connect to API
   - Real-time pattern preview

4. **Ableton Integration**
   - Implement MIDI port communication
   - Add OSC support
   - Real-time pattern insertion

5. **Performance Optimization**
   - Model quantization
   - Batch generation
   - Caching strategies

---

## Success Criteria

You'll know the setup is working when:

- ✅ All unit tests pass
- ✅ API endpoints respond correctly
- ✅ Celery tasks complete successfully
- ✅ MIDI files are generated and valid
- ✅ Patterns can be loaded in a DAW
- ✅ Claude Code generates code matching architecture
- ✅ No legacy dependencies are suggested

---

## Getting Help

If you get stuck:

1. **Reference Context Documents:**
   ```
   @docs 06_common_tasks.md
   I'm having this issue: [describe problem]
   ```

2. **Check Architecture:**
   ```
   @docs 02_architecture.md
   How should [component] work?
   ```

3. **Review Dependencies:**
   ```
   @docs 03_dependencies.md
   Is [library] compatible with our stack?
   ```

4. **Common Tasks:**
   ```
   @docs 06_common_tasks.md
   Show me how to [task]
   ```

---

**Ready to start? Begin with Phase 1, Step 1.1 (System Architecture) and work through each step systematically!** 🚀

**Note:** The order follows SETUP_GUIDE.md:
1. **Phase 1:** Design System Architecture (FastAPI + Celery + Redis) ← START HERE
2. **Phase 2:** Implement MIDI Export
3. **Phase 3:** Model Loading and Inference
4. **Phase 4:** Complete API Routes
5. **Phase 5:** Complete Celery Tasks
6. **Phase 6:** Training Pipeline
7. **Phase 7:** Testing and Validation
8. **Phase 8:** End-to-End Testing

