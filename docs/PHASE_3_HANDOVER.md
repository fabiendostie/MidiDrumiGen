# Phase 3 Handover - Model Loading and Inference

## Project Context

**Project:** MidiDrumiGen - AI-powered MIDI drum pattern generator with producer style emulation
**Tech Stack:** PyTorch 2.4+, FastAPI, Celery, Redis, mido (for MIDI I/O)
**Current Status:** Phase 2 Complete (MIDI Export Pipeline)
**Next Phase:** Phase 3 - Model Loading and Inference

---

## What Has Been Completed

### Phase 1: System Architecture (FastAPI + Celery + Redis) ✅
- Project structure created with all directories and `__init__.py` files
- FastAPI application skeleton (`src/api/main.py`)
- Celery worker setup (`src/tasks/worker.py`)
- Redis configuration ready
- Core MIDI constants defined (`src/midi/constants.py`, `src/midi/io.py`)

### Phase 2: MIDI Export Pipeline ✅
- **`src/midi/export.py`** - Complete pattern-to-MIDI conversion pipeline
- **`src/midi/humanize.py`** - All humanization algorithms (swing, micro-timing, velocity, ghost notes)
- **`src/midi/validate.py`** - Pattern validation with comprehensive checks
- **Producer styles defined:** J Dilla, Metro Boomin, Questlove with parameters
- **Test suite passing:** All 4 test suites (48 notes, 3 MIDI files generated)
- **Test files:** `scripts/test_midi_export.py` validates entire pipeline

**Generated Test Files:**
```
output/test_patterns/
├── test_pattern_no_humanize.mid (443 bytes)
├── test_pattern_j_dilla.mid (493 bytes)
└── test_pattern_metro_boomin.mid (458 bytes)
```

---

## Phase 3 Objectives

Implement the model loading and inference system that will:
1. Load PyTorch Transformer models with caching (LRU cache)
2. Manage GPU/CPU device placement
3. Define producer style registry with model mappings
4. Handle model generation inference
5. Integrate with MIDI export pipeline

---

## Tasks for Phase 3

### Task 3.1: Create Model Loader with Caching

**File to create:** `src/inference/model_loader.py`

**Requirements:**
1. Create `src/inference/__init__.py` (if doesn't exist)
2. Implement `load_model()` function with:
   - `@lru_cache` decorator for model caching
   - Support for loading models by style name or path
   - Automatic device detection (CUDA/CPU)
   - Graceful GPU OOM handling (fallback to CPU)
   - Error handling for missing model files
   - Model metadata loading (vocab_size, max_length, etc.)

**Function Signature:**
```python
from functools import lru_cache
from pathlib import Path
import torch
from typing import Optional, Tuple
from src.models.transformer import DrumPatternTransformer

@lru_cache(maxsize=4)
def load_model(
    model_path: Path,
    device: Optional[str] = None
) -> Tuple[DrumPatternTransformer, dict]:
    """
    Load PyTorch model with caching.

    Args:
        model_path: Path to model checkpoint (.pt or .pth)
        device: Target device ("cuda", "cpu", or None for auto-detect)

    Returns:
        (model, metadata) tuple
    """
    pass
```

**Key Features:**
- Check if CUDA is available: `torch.cuda.is_available()`
- Load checkpoint: `torch.load(model_path, map_location=device)`
- Initialize model from checkpoint state
- Set model to eval mode: `model.eval()`
- Log device placement and model info

**Reference Documentation:**
- `@docs .cursorcontext/05_ml_pipeline.md` - ML model architecture
- `@docs .cursorcontext/02_architecture.md` - System architecture
- `@file src/models/transformer.py` - Model definition (already exists)

**Verification:**
```powershell
# Test import
./venv/Scripts/python -c "from src.inference.model_loader import load_model; print('Model loader OK')"
```

---

### Task 3.2: Create Style Registry

**File to create:** `src/models/styles.py`

**Requirements:**
1. Define `PRODUCER_STYLES` dictionary mapping style names to model IDs
2. Include style parameters (already defined in `src/midi/humanize.py`, reference those)
3. Add helper functions:
   - `get_style_id(style_name: str) -> str` - Get model ID from style name
   - `get_style_params(style_name: str) -> dict` - Get humanization params
   - `list_available_styles() -> List[str]` - Return all style names
   - `get_model_path(style_id: str) -> Path` - Resolve model checkpoint path

**Style Registry Structure:**
```python
PRODUCER_STYLES = {
    'J Dilla': {
        'model_id': 'j_dilla_v1',
        'model_path': 'models/checkpoints/j_dilla_v1.pt',
        'description': 'Signature swing and soulful groove',
        'preferred_tempo_range': (85, 95),
        'humanization': {
            'swing': 62.0,
            'micro_timing_ms': 20.0,
            'ghost_note_prob': 0.4,
            'velocity_variation': 0.15,
        }
    },
    'Metro Boomin': {
        'model_id': 'metro_boomin_v1',
        'model_path': 'models/checkpoints/metro_boomin_v1.pt',
        'description': 'Tight trap drums with rolls',
        'preferred_tempo_range': (130, 150),
        'humanization': {
            'swing': 52.0,
            'micro_timing_ms': 5.0,
            'ghost_note_prob': 0.1,
            'velocity_variation': 0.08,
        }
    },
    'Questlove': {
        'model_id': 'questlove_v1',
        'model_path': 'models/checkpoints/questlove_v1.pt',
        'description': 'Dynamic funk drumming with ghost notes',
        'preferred_tempo_range': (90, 110),
        'humanization': {
            'swing': 58.0,
            'micro_timing_ms': 12.0,
            'ghost_note_prob': 0.5,
            'velocity_variation': 0.20,
        }
    },
}
```

**Important Notes:**
- Model checkpoint files don't exist yet (will be created in Phase 6: Training)
- For now, handle missing model files gracefully with clear error messages
- The humanization parameters should match those in `src/midi/humanize.py`

**Reference Documentation:**
- `@docs .cursorcontext/04_midi_operations.md` - Producer styles and humanization
- `@file src/midi/humanize.py` - Existing style parameters

**Verification:**
```powershell
# Test style registry
./venv/Scripts/python -c "from src.models.styles import get_style_id, list_available_styles; print(get_style_id('J Dilla')); print(list_available_styles())"
```

---

### Task 3.3: Create Inference Module

**File to create:** `src/inference/generate.py`

**Requirements:**
1. Implement `generate_pattern()` function that:
   - Takes model, tokenizer, style parameters
   - Generates tokens using model inference
   - Handles temperature, top_k, top_p sampling
   - Returns list of generated token IDs
   - Includes proper error handling

**Function Signature:**
```python
def generate_pattern(
    model,
    tokenizer,
    prompt_tokens: Optional[List[int]] = None,
    num_bars: int = 4,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    max_length: int = 512,
    device: str = "cpu"
) -> List[int]:
    """
    Generate drum pattern tokens using the model.

    Args:
        model: Loaded PyTorch model
        tokenizer: MidiTok tokenizer instance
        prompt_tokens: Optional starting tokens
        num_bars: Number of bars to generate (default: 4)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling parameter (default: 50)
        top_p: Nucleus sampling parameter (default: 0.9)
        max_length: Maximum sequence length (default: 512)
        device: Device to run on ("cuda" or "cpu")

    Returns:
        List of generated token IDs
    """
    pass
```

**Key Implementation Points:**
- Use `torch.no_grad()` context for inference
- Implement autoregressive generation loop
- Apply temperature scaling: `logits = logits / temperature`
- Implement top-k filtering: Keep only top k logits
- Implement nucleus (top-p) sampling: Cumulative probability filtering
- Sample from distribution: `torch.multinomial(probs, num_samples=1)`
- Stop at max_length or EOS token
- Handle batch dimension properly

**Reference Documentation:**
- `@docs .cursorcontext/05_ml_pipeline.md` - Generation algorithms
- `@file src/models/transformer.py` - Model forward pass

**Note:** This is a placeholder implementation since we don't have trained models yet. For now, focus on the structure and error handling. The actual generation logic can be tested once models are trained in Phase 6.

**Verification:**
```powershell
# Test import
./venv/Scripts/python -c "from src.inference.generate import generate_pattern; print('Generate module OK')"
```

---

### Task 3.4: Create Mock Model for Testing (Optional but Recommended)

**File to create:** `src/inference/mock.py`

**Purpose:** Create a mock model that generates simple deterministic patterns for testing the full pipeline without trained models.

**Requirements:**
1. Implement `MockDrumModel` class that:
   - Inherits from `torch.nn.Module`
   - Returns simple, valid drum patterns (e.g., kick on 1 and 3, snare on 2 and 4)
   - Works with the inference pipeline
   - Can be used for end-to-end testing

**Example Structure:**
```python
class MockDrumModel(torch.nn.Module):
    """Mock model for testing pipeline without trained weights."""

    def __init__(self, vocab_size: int = 500):
        super().__init__()
        self.vocab_size = vocab_size

    def generate(self, input_ids, max_length=512, **kwargs):
        """Generate simple deterministic pattern."""
        # Return token IDs for: kick, snare, hi-hat pattern
        pass
```

**Verification:**
```powershell
# Test mock model
./venv/Scripts/python -c "from src.inference.mock import MockDrumModel; m = MockDrumModel(); print('Mock model OK')"
```

---

### Task 3.5: Create Test Suite for Phase 3

**File to create:** `scripts/test_inference.py`

**Requirements:**
Test script that verifies:
1. Model loader correctly detects CUDA/CPU
2. Style registry returns correct parameters
3. Model path resolution works
4. Error handling for missing models works
5. Mock model (if implemented) generates valid patterns

**Test Structure:**
```python
def test_device_detection():
    """Test CUDA/CPU device detection."""
    pass

def test_style_registry():
    """Test style parameter retrieval."""
    pass

def test_model_path_resolution():
    """Test model path resolution for each style."""
    pass

def test_missing_model_handling():
    """Test graceful handling of missing model files."""
    pass

def test_mock_generation():
    """Test mock model generates valid patterns."""
    pass
```

**Verification:**
```powershell
./venv/Scripts/python scripts/test_inference.py
```

---

## Important Context Files

Load these files when starting work:

1. **Project Overview:**
   - `@docs .cursorcontext/01_project_overview.md`

2. **Architecture:**
   - `@docs .cursorcontext/02_architecture.md`

3. **ML Pipeline:**
   - `@docs .cursorcontext/05_ml_pipeline.md`

4. **Existing Code:**
   - `@file src/models/transformer.py` - Model definition
   - `@file src/midi/humanize.py` - Producer style parameters
   - `@file src/midi/export.py` - MIDI export (will integrate with this)

5. **Development Guide:**
   - `@file docs/DEVELOPMENT_START_GUIDE.md` - Full phase-by-phase guide

---

## Expected Deliverables

By the end of Phase 3, you should have:

1. ✅ `src/inference/__init__.py` - Module initialization
2. ✅ `src/inference/model_loader.py` - Model loading with LRU cache
3. ✅ `src/models/styles.py` - Producer style registry
4. ✅ `src/inference/generate.py` - Pattern generation inference
5. ✅ `src/inference/mock.py` - Mock model for testing (optional)
6. ✅ `scripts/test_inference.py` - Test suite for inference
7. ✅ All imports working without errors
8. ✅ Test suite passing (or gracefully handling missing models)

---

## Development Environment

**Python Version:** 3.11
**Virtual Environment:** `./venv` (already activated)
**Working Directory:** `C:\Users\lefab\Documents\Dev\MidiDrumiGen`

**Key Dependencies (already installed):**
- PyTorch 2.4.1 with CUDA 12.4
- mido 1.3.3
- FastAPI 0.121.0+
- Celery 5.5.3+

**Quick Commands:**
```powershell
# Activate environment
.\\venv\\Scripts\\Activate.ps1

# Run tests
./venv/Scripts/python scripts/test_inference.py

# Check imports
./venv/Scripts/python -c "from src.inference import model_loader"
```

---

## Known Constraints

1. **No trained models yet** - Phase 6 will handle training
   - Handle missing model files gracefully
   - Consider implementing mock model for testing
   - Focus on infrastructure and error handling

2. **Tokenizer not implemented yet** - Will be addressed later
   - Create placeholder tokenizer interface if needed
   - Focus on model loading and device management

3. **Windows environment** - Avoid Unicode characters in print statements
   - Use `[PASS]` and `[FAIL]` instead of emojis
   - Test console output encoding

---

## Success Criteria

Phase 3 is complete when:

- ✅ All files created and imports work
- ✅ Model loader has LRU caching implemented
- ✅ Style registry returns correct parameters
- ✅ Device detection (CUDA/CPU) works correctly
- ✅ Error handling is comprehensive and informative
- ✅ Test suite runs without crashes
- ✅ Code follows project conventions (type hints, docstrings, logging)
- ✅ Ready to integrate with API routes in Phase 4

---

## Next Steps After Phase 3

Once Phase 3 is complete, proceed to:

**Phase 4: Complete API Routes**
- POST `/api/v1/generate` - Queue pattern generation
- GET `/api/v1/status/{task_id}` - Check task status
- GET `/api/v1/styles` - List available styles

---

## Getting Help

If you encounter issues:

1. **Reference context documents:**
   ```
   @docs .cursorcontext/05_ml_pipeline.md
   How should model loading work?
   ```

2. **Check existing code patterns:**
   ```
   @file src/midi/export.py
   How is error handling structured?
   ```

3. **Review architecture:**
   ```
   @docs .cursorcontext/02_architecture.md
   How does the ML pipeline fit into the system?
   ```

---

## Start Working on Phase 3

**Recommended approach:**

1. Create todo list for tracking progress
2. Start with Task 3.1 (Model Loader)
3. Implement Task 3.2 (Style Registry)
4. Create Task 3.3 (Inference Module)
5. Build Task 3.4 (Mock Model - optional but helpful)
6. Write Task 3.5 (Test Suite)
7. Run tests and verify everything works

**First command to run:**
```
I'm starting Phase 3: Model Loading and Inference. I've read the handover document. Let me create a todo list and begin with Task 3.1 (Model Loader).
```

Good luck! 🚀
