# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Modern PyTorch-based MIDI drum pattern generator for Ableton Live 12 with producer style emulation capabilities. Built on a strictly modern tech stack, explicitly avoiding legacy components like Magenta/TensorFlow, pretty-midi, and pylive.

**Core Technologies:**
- PyTorch 2.4+ with CUDA 12.1+ for ML models
- mido 1.3.3 for MIDI I/O operations
- MidiTok 2.1+ for MIDI tokenization
- FastAPI 0.121.0+ for REST API
- Celery 5.5.3+ for async task processing
- Redis 7.0+ for message broker

**Python Version:** Python 3.11 (required)

## Common Commands

### Environment Setup

```bash
# Create and activate virtual environment (Windows with py launcher)
py -3.11 -m venv venv
venv\Scripts\activate

# On Linux/macOS:
# python3.11 -m venv venv
# source venv/bin/activate

# Install PyTorch with CUDA 12.4 (recommended for modern GPUs like RTX 4070)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Or use CUDA 12.1 if preferred:
# pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_installation.py
```

### Development Workflow

```bash
# Run API server (development)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery workers
celery -A src.tasks.worker worker -Q gpu_generation -c 2 --loglevel=info

# Monitor Celery with Flower
celery -A src.tasks.worker flower --port=5555
```

### Training and Generation

```bash
# Tokenize MIDI dataset
python scripts/tokenize_dataset.py \
  --input-dir data/groove_midi \
  --output-dir data/groove_tokenized \
  --tokenizer-config configs/remi_tokenizer.json

# Train model
python src/training/train_transformer.py \
  --config configs/base.yaml \
  --data-dir data/groove_tokenized \
  --checkpoint-dir models/checkpoints \
  --use-wandb

# Generate pattern from CLI
python scripts/generate_pattern.py \
  --style "J Dilla" \
  --bars 4 \
  --tempo 95 \
  --output "patterns/output.mid"
```

### Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Lint and format
ruff check src/
black --check src/
mypy src/

# Format code
black src/
ruff check --fix src/
```

## Architecture Overview

### Component Structure

The system follows a multi-tier architecture:

1. **API Layer** (`src/api/`): FastAPI endpoints for pattern generation requests
2. **Task Queue** (`src/tasks/`): Celery workers for asynchronous ML inference
3. **ML Models** (`src/models/`): PyTorch Transformer models for pattern generation
4. **MIDI Processing** (`src/midi/`): mido-based MIDI export with humanization
5. **Training Pipeline** (`src/training/`): Model training and fine-tuning workflows
6. **Ableton Integration** (`src/ableton/`): MIDI/OSC communication with Ableton Live

### Data Flow

```
User Request (FastAPI)
  → Celery Task Queue (Redis)
    → Model Inference (PyTorch on GPU)
      → Token Generation
        → MIDI Export (mido)
          → Humanization (timing/velocity)
            → Output File
```

### Critical Architecture Rules

**MIDI Operations:**
- Use `mido` for all MIDI I/O (NOT pretty-midi - abandoned since 2020)
- Use `miditoolkit` only for enhanced operations
- Use `music21` sparingly (heavy dependency, optional only)

**ML Framework:**
- PyTorch 2.4+ exclusively (NO TensorFlow, NO Magenta)
- Custom Transformer models (NOT MusicVAE, NOT GrooVAE)
- MidiTok for tokenization (framework-agnostic)

**Async Processing:**
- FastAPI for REST endpoints
- Celery for background tasks (GPU-intensive operations)
- Redis as message broker and result backend
- Separate queues for GPU vs CPU tasks

## Key Conventions

### File Organization

```
src/
├── api/           # FastAPI routes and models (Pydantic)
├── models/        # PyTorch model definitions
├── tasks/         # Celery task definitions
├── midi/          # MIDI processing (mido-based)
├── training/      # Training scripts and data loaders
├── ableton/       # Ableton integration (MIDI/OSC)
└── utils/         # Shared utilities
```

### Code Style

- Python 3.11 features (match/case, type hints, dataclasses)
- PEP 8 with Black formatter (line length: 100)
- Comprehensive type hints using `typing` module
- Async/await for I/O operations in FastAPI
- Pydantic models for all API inputs/outputs

### Import Order

```python
# 1. Standard library
import os
from pathlib import Path
from typing import List, Dict, Optional

# 2. Third-party packages
import torch
import torch.nn as nn
from fastapi import FastAPI
import mido

# 3. Local imports
from src.models.transformer import DrumPatternTransformer
from src.midi.utils import export_midi
```

### Error Handling Patterns

```python
# Use specific exceptions
class PatternGenerationError(Exception):
    """Raised when pattern generation fails."""
    pass

# Handle GPU OOM gracefully
try:
    pattern = model.generate(tokens)
except torch.cuda.OutOfMemoryError:
    logger.error("GPU OOM, falling back to CPU")
    pattern = model.to("cpu").generate(tokens)
```

## MIDI-Specific Knowledge

### General MIDI Drum Mapping

Standard drum note numbers:
- 35: Acoustic Bass Drum
- 36: Bass Drum 1
- 38: Acoustic Snare
- 42: Closed Hi-Hat
- 46: Open Hi-Hat
- 49: Crash Cymbal 1
- 51: Ride Cymbal 1

### Humanization Parameters

When implementing humanization:
- **Timing variance**: ±15ms (0.015 seconds) default
- **Velocity variation**: ±10% (0.1) default
- **Swing**: 50-67% range (16th note swing)
- **Ghost notes**: Low velocity (20-40) on offbeats

### Exporting MIDI with mido

```python
import mido
from mido import MidiFile, MidiTrack, Message

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# Add tempo
track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

# Add notes
track.append(Message('note_on', note=36, velocity=80, time=0))
track.append(Message('note_off', note=36, time=480))

mid.save('output.mid')
```

## PyTorch Model Patterns

### Model Definition

```python
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class DrumPatternTransformer(nn.Module):
    """Custom Transformer for drum pattern generation."""

    def __init__(self, vocab_size: int, n_positions: int = 2048):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
        self.transformer = GPT2LMHeadModel(config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.transformer(input_ids=input_ids, **kwargs)
```

### Training Loop

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = AdamW(model.parameters(), lr=5e-5)

for batch in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda'):
        outputs = model(batch['input_ids'])
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### GPU Memory Management

```python
# Clear cache between generations
torch.cuda.empty_cache()

# Enable gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Monitor memory usage
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
```

## Important Context Documents

This repository includes comprehensive context engineering documentation in `.cursorcontext/`:

- `01_project_overview.md` - High-level project goals and features
- `02_architecture.md` - Detailed system design and component interactions
- `03_dependencies.md` - Verified modern libraries and why legacy ones are avoided
- `04_midi_operations.md` - MIDI protocol knowledge and drum mapping
- `05_ml_pipeline.md` - Model architecture, training, and inference patterns
- `06_common_tasks.md` - Quick reference for frequent development tasks

Load these documents using `@docs` references in Cursor when working on specific features.

## Critical Constraints

### What NOT to Use (Legacy Components)

❌ **Magenta/TensorFlow** - Repository inactive since 2025, TensorFlow dependency hell
❌ **GrooVAE/MusicVAE** - Legacy TensorFlow 1.x models, unmaintained
❌ **pretty-midi** - Abandoned since 2020, no Python 3.11+ support
❌ **pylive** - Uncertain Ableton Live 12 compatibility

### What to Use Instead

✅ **PyTorch 2.4+** - Modern, actively maintained ML framework
✅ **mido 1.3.3** - Active MIDI library with Python 3.11+ support
✅ **MidiTok 2.1+** - Framework-agnostic MIDI tokenization
✅ **FastAPI + Celery + Redis** - Production-grade async processing

## Troubleshooting

### GPU Out of Memory
- Reduce batch size in training config
- Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
- Clear CUDA cache: `torch.cuda.empty_cache()`

### MIDI Port Not Found
- macOS: Enable IAC Driver in Audio MIDI Setup
- Windows: Install loopMIDI
- Linux: Use ALSA virtual MIDI ports

### Invalid Token IDs During Generation
- Verify tokenizer vocab size matches model: `tokenizer.vocab_size == model.vocab_size`
- Check for out-of-vocabulary tokens before detokenization
- Ensure BOS/EOS tokens are properly configured

### Celery Tasks Hanging
- Check Redis connection: `redis-cli ping`
- Verify worker queues match task routing
- Monitor with Flower: `celery -A src.tasks.worker flower`

## Development Notes

- All dependencies are verified for Python 3.11 compatibility
- CUDA 12.1+ required for GPU acceleration
- Minimum 8GB VRAM recommended for training
- Use `.cursorrules` file for Cursor IDE AI behavior configuration
- Run `scripts/verify_installation.py` after environment setup
- always refer to 'c:/Users/lefab/Documents/Dev/MidiDrumiGen/.cursorcontext' for information in this project