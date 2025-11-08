# AI Drum Pattern Generator for Ableton Live 12

**Modern PyTorch-based MIDI drum pattern generation with producer style emulation**

## Architecture Overview

This project uses a **modern, non-legacy stack** built on PyTorch 2.x for generating intelligent drum patterns with specific producer style emulation capabilities.

### Tech Stack (All Modern Components)

**Core ML Framework:**
- PyTorch 2.4+ with CUDA 12.1+
- HuggingFace Transformers 4.40+
- MidiTok 2.1+ for tokenization

**MIDI Processing:**
- mido 1.3.3 (primary MIDI library)
- miditoolkit 1.0.1 (enhanced MIDI operations)
- music21 9.9.1 (only for advanced analysis, optional)

**Backend Infrastructure:**
- FastAPI 0.121.0+ (REST API)
- Celery 5.5.3+ (async task processing)
- Redis 7.0+ (message broker & result backend)

**Ableton Integration:**
- Direct MIDI via mido + virtual MIDI ports
- Or custom OSC implementation (python-osc 1.9.0)
- ⚠️ Avoiding pylive until Ableton Live 12 compatibility confirmed

**Data Sources:**
- Groove MIDI Dataset (primary training data)
- Producer-specific MIDI collections
- Web-scraped producer style metadata

### What We're NOT Using (Legacy Components Removed)

❌ **Magenta/TensorFlow** - Repository inactive since 2025, TensorFlow dependency hell  
❌ **GrooVAE/MusicVAE** - Legacy TensorFlow 1.x models, unmaintained  
❌ **pretty-midi** - Abandoned since 2020, no Python 3.11+ support  
❌ **pylive** - Uncertain Live 12 compatibility  
❌ **Magenta RealTime** - Audio-only, no MIDI/symbolic music support

## Project Structure

```
MidiDrumiGen/
├── .cursorrules                 # Cursor IDE AI behavior rules
├── .cursorcontext/             # Context engineering documents
│   ├── 01_project_overview.md
│   ├── 02_architecture.md
│   ├── 03_dependencies.md
│   ├── 04_midi_operations.md
│   ├── 05_ml_pipeline.md
│   └── 06_common_tasks.md
├── docs/                        # Detailed documentation
│   ├── setup.md
│   ├── training.md
│   ├── deployment.md
│   └── ableton_integration.md
├── src/
│   ├── api/                    # FastAPI endpoints
│   ├── models/                 # PyTorch model definitions
│   ├── tasks/                  # Celery task definitions
│   ├── midi/                   # MIDI processing utilities
│   ├── training/               # Training pipeline
│   └── ableton/                # Ableton Live integration
├── tests/
├── requirements.txt
├── pyproject.toml
└── docker-compose.yml
```

## Features

### Phase 1 (Current Focus)
- Custom PyTorch Transformer trained on Groove MIDI Dataset
- MidiTok tokenization for symbolic music representation
- Producer style analysis from web-scraped data
- Basic pattern generation with tempo, time signature control
- MIDI file export via mido

### Phase 2 (Planned)
- Latent space manipulation for style interpolation
- Humanization (timing offsets, velocity variation, ghost notes)
- Real-time Ableton Live integration via MIDI/OSC
- Web UI for pattern preview and control

### Phase 3 (Future)
- Multi-track generation (drums + bass + melody)
- Style transfer between producers
- Real-time adjustment during playback
- VST/AU plugin wrapper

## System Requirements

**Minimum:**
- Python 3.11+
- NVIDIA GPU with 8GB VRAM (RTX 3060 or better)
- 16GB System RAM
- Ubuntu 22.04+ / macOS 13+ / Windows 11

**Recommended:**
- Python 3.11
- NVIDIA GPU with 16GB VRAM (RTX 4090 / A4000)
- 32GB System RAM
- SSD for dataset storage

## Quick Start

**Windows (PowerShell):**
```powershell
# Navigate to project directory
cd C:\Users\lefab\Documents\Dev\MidiDrumiGen

# Create virtual environment
py -3.11 -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download Groove MIDI Dataset (when available)
python scripts/download_groove_midi.py

# Train model (or download pretrained)
python src/training/train_transformer.py

# Start API server
uvicorn src.api.main:app --reload

# Start Celery worker (in separate terminal)
celery -A src.tasks.worker worker --loglevel=info
```

**Linux/macOS:**
```bash
# Navigate to project directory
cd ~/Documents/Dev/MidiDrumiGen

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Groove MIDI Dataset
python scripts/download_groove_midi.py

# Train model (or download pretrained)
python src/training/train_transformer.py

# Start API server
uvicorn src.api.main:app --reload

# Start Celery worker
celery -A src.tasks.worker worker --loglevel=info
```

## Development with Cursor IDE & Claude Code

This project is optimized for development with **Cursor IDE** and **Claude Code** using comprehensive context engineering:

1. **`.cursorrules`** defines AI behavior and coding standards
2. **`.cursorcontext/`** contains modular context documents for Claude Code
3. **Use `@docs` and `@folder` references** in Cursor chat to query specific contexts
4. **Claude Code will automatically reference** context documents when implementing features
5. See `docs/cursor_guide.md` for detailed usage

### Typical Workflow

1. Open project in Cursor IDE
2. Use Cursor's chat (Claude Code) to implement features:
   ```
   @docs 02_architecture.md @docs 04_midi_operations.md
   Implement the MIDI export function in src/midi/export.py
   ```
3. Claude Code will:
   - Reference context documents automatically
   - Follow `.cursorrules` for coding standards
   - Use modern libraries (mido, PyTorch, etc.)
   - Write type-hinted, tested code

## Verification Status

All dependencies verified as of November 2025:

✅ PyTorch 2.4.1 - Python 3.11+ support, CUDA 12.1+ compatible  
✅ mido 1.3.3 - Active maintenance, Python 3.7-3.12 support  
✅ MidiTok 2.1+ - Active development, framework-agnostic  
✅ FastAPI 0.121.0 - Latest release, Python 3.14 support  
✅ Celery 5.5.3 - Enterprise-grade, Python 3.8-3.13 support  
✅ redis-py 7.0.1 - Official Redis Inc. maintenance  

## License

MIT License - See LICENSE file

## Contributing

See CONTRIBUTING.md for development guidelines and code standards.
