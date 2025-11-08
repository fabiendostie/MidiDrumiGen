# Project Overview - Context Document

**Document Type:** Instructional Context  
**Purpose:** High-level project understanding, goals, and constraints  
**Use Case:** Load this when starting new features or explaining the project

---

## What This Project Does

This is an **AI-powered MIDI drum pattern generator** that:

1. **Learns producer-specific drumming styles** from MIDI datasets
2. **Generates new drum patterns** matching requested styles
3. **Exports patterns as MIDI files** compatible with any DAW
4. **Integrates with Ableton Live 12** for real-time pattern insertion

### Key Differentiation

Unlike generic music generation models, this focuses on:
- **Symbolic MIDI generation** (not audio)
- **Drum-specific patterns** with proper GM mapping
- **Style emulation** of specific producers (J Dilla, Metro Boomin, etc.)
- **Musically coherent loops** (1-32 bars)
- **Humanization** (timing offsets, velocity variation, ghost notes)

## User Workflow

```
User Request → API Endpoint → Celery Task Queue → ML Model → MIDI File → Ableton
```

**Example User Interaction:**
```json
POST /api/v1/generate
{
  "producer_style": "J Dilla",
  "bars": 4,
  "time_signature": [4, 4],
  "tempo": 95,
  "humanize": true,
  "pattern_type": "intro"
}

Response:
{
  "task_id": "abc-123",
  "status": "queued"
}
```

## Core Features

### Phase 1 (MVP - Current)
✅ Custom PyTorch Transformer model  
✅ MidiTok tokenization pipeline  
✅ Groove MIDI Dataset training  
✅ Producer style classification  
✅ Basic pattern generation  
✅ MIDI file export  
✅ REST API with FastAPI  
✅ Async task processing with Celery

### Phase 2 (In Development)
🚧 Latent space interpolation  
🚧 Advanced humanization algorithms  
🚧 Real-time Ableton integration  
🚧 Web UI for pattern preview  
🚧 Style transfer between producers

### Phase 3 (Planned)
📋 Multi-track generation (drums + bass)  
📋 VST/AU plugin wrapper  
📋 Real-time parameter adjustment  
📋 Custom style training from user MIDI

## Technical Constraints

### Hard Requirements
- Python 3.11 (strict version)
- NVIDIA GPU with 8GB+ VRAM minimum
- CUDA 12.1+ compatible drivers
- Redis server for task queue
- 16GB+ system RAM

### Performance Targets
- Pattern generation: < 2 seconds on RTX 3060
- API response time: < 100ms
- Concurrent requests: 10+ simultaneous generations
- Model size: < 500MB for production deployment

## Non-Goals (What We Don't Do)

❌ **Audio synthesis** - We generate MIDI, not audio waveforms  
❌ **Melody generation** - Drums only (may expand to bass later)  
❌ **Real-time audio streaming** - Focus on MIDI file export  
❌ **Mobile deployment** - Desktop/server deployment only  
❌ **Legacy DAW support** - Ableton Live 12 is minimum version

## Technology Stack Justification

### Why PyTorch Instead of TensorFlow?
- Modern ecosystem (Transformers, Lightning, etc.)
- No dependency hell with CUDA versions
- Better Python integration and debugging
- Active community and development

### Why Custom Model Instead of Pretrained?
- No pretrained models exist for symbolic drum generation
- Full control over drum-specific features
- Smaller model size (500MB vs 5GB+)
- Faster inference for real-time use cases

### Why FastAPI + Celery?
- FastAPI: Modern async Python, auto-generated docs
- Celery: Industry-standard for background tasks
- Redis: Fast, reliable message broker
- Proven stack used by Uber, Netflix, Instagram

### Why mido Instead of pretty-midi?
- pretty-midi abandoned since 2020
- mido actively maintained with Python 3.12 support
- Simpler API for drum patterns
- Better error handling and edge cases

## Data Sources

### Primary Training Data
**Groove MIDI Dataset** (Google Magenta, 2019)
- 1,150+ MIDI drum patterns
- Professional studio recordings
- Tempo, time signature, style labels
- Multiple takes per groove
- License: Apache 2.0

### Secondary Data (Web-Scraped)
- Producer interview transcripts
- Production technique articles
- MIDI files from artist packs
- Forum discussions about specific styles

### Style Metadata
- BPM range per producer
- Preferred time signatures
- Characteristic drum sounds
- Common pattern structures
- Swing percentage

## Model Architecture Overview

```
Input: Producer Style + Parameters
  ↓
Tokenizer (MidiTok REMI)
  ↓
Transformer Encoder (12 layers)
  ↓
Style Conditioning (learned embeddings)
  ↓
Transformer Decoder (12 layers)
  ↓
Token Sampling (top-k, temperature)
  ↓
Detokenizer (MIDI reconstruction)
  ↓
Post-processing (humanization, validation)
  ↓
Output: MIDI File
```

## Key Concepts

### Tokenization
Converting MIDI to discrete tokens that a Transformer can process:
- Note events → Note tokens
- Timing → Bar/Position tokens
- Velocity → Velocity tokens
- Tempo → Tempo tokens

### Style Conditioning
Adding producer identity to model input:
- Learned style embeddings (128-dim)
- Concatenated with positional encoding
- Allows model to generate in specific style

### Humanization
Making patterns feel less "robotic":
- **Timing offsets**: ±15ms random variation
- **Velocity variation**: ±10% per note
- **Ghost notes**: Soft hits (20-40 velocity)
- **Swing**: Delayed offbeat notes

### Pattern Validation
Ensuring musical quality:
- Valid MIDI note ranges (35-81 for GM drums)
- Reasonable note density (not too sparse/dense)
- Proper timing quantization
- No simultaneous impossible hits

## Development Principles

1. **No Legacy Dependencies**: Zero TensorFlow, Zero Magenta
2. **Type Everything**: Full type hints for all functions
3. **Test Everything**: Unit tests for all core functions
4. **Document Everything**: Docstrings for all public APIs
5. **Modern Python**: Use Python 3.11 features liberally
6. **GPU First**: Optimize for GPU execution
7. **Async Where Possible**: Non-blocking I/O operations
8. **Fail Fast**: Validate inputs early, raise clear errors

## Project Status (November 2025)

**Current Phase:** MVP Development  
**Milestone:** Phase 1 completion by Q1 2026  
**Team Size:** Solo developer + AI assistants  
**Code Quality:** Targeting 80%+ test coverage  
**Documentation:** Comprehensive context engineering docs

## Success Metrics

### Technical Metrics
- Model validation loss < 0.5
- Generation time < 2 seconds
- MIDI file validity rate > 99%
- API uptime > 99.5%

### User Experience Metrics
- Pattern quality subjective rating > 4/5
- Style recognition accuracy > 80%
- User retention week-over-week > 60%
- Time to first generation < 30 seconds

## Related Documents

- **Architecture Details**: `.cursorcontext/02_architecture.md`
- **Dependencies**: `.cursorcontext/03_dependencies.md`
- **MIDI Operations**: `.cursorcontext/04_midi_operations.md`
- **ML Pipeline**: `.cursorcontext/05_ml_pipeline.md`
- **Common Tasks**: `.cursorcontext/06_common_tasks.md`

## Quick Reference Commands

```bash
# Start development environment
docker-compose up -d

# Run tests
pytest tests/ -v --cov

# Start API server
uvicorn src.api.main:app --reload

# Start Celery worker
celery -A src.tasks.worker worker --loglevel=info

# Train model
python src/training/train_transformer.py --config configs/base.yaml

# Generate test pattern
python scripts/generate_pattern.py --style "J Dilla" --bars 4
```
