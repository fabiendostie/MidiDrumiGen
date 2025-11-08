# Dependencies - Context Document

**Document Type:** Knowledge Context (Dependencies & Compatibility)  
**Purpose:** Verified dependency information, version requirements, and compatibility matrices  
**Use Case:** Load when debugging dependency issues or adding new libraries

---

## Dependency Philosophy

**Core Principle:** Use only modern, actively maintained libraries with Python 3.11+ support.

**Verification Criteria:**
✅ Last update within 12 months  
✅ Official Python 3.11/3.12 support  
✅ Active GitHub repository (issues, PRs, commits)  
✅ Clear migration path from legacy alternatives  
✅ Production-grade (used by major companies or projects)

---

## Core Dependencies (Verified November 2025)

### PyTorch Ecosystem

#### pytorch 2.4.1
- **Status**: ✅ Production-ready, actively maintained
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **CUDA Support**: 11.8, 12.1, 12.4
- **Install Command**:
  ```bash
  # CUDA 12.1 (recommended)
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
  
  # CPU only
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
  ```
- **Why This Version**: Latest stable, optimal CUDA 12.1 support
- **GitHub**: https://github.com/pytorch/pytorch (79K+ stars)
- **Last Release**: October 2025

#### transformers 4.46.3
- **Status**: ✅ Production-ready, HuggingFace official
- **Python Support**: 3.8+
- **Key Features**: GPT-2, BERT, T5 architectures
- **Install**: `pip install transformers==4.46.3`
- **Why We Use It**: Pre-built Transformer blocks, configuration management
- **Usage**:
  ```python
  from transformers import GPT2Config, GPT2LMHeadModel
  
  config = GPT2Config(vocab_size=1000, n_layer=12)
  model = GPT2LMHeadModel(config)
  ```
- **GitHub**: https://github.com/huggingface/transformers (144K+ stars)
- **Last Release**: November 2025

### MIDI Processing

#### mido 1.3.3 ⭐ PRIMARY MIDI LIBRARY
- **Status**: ✅ Production-ready, active maintenance
- **Python Support**: 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
- **Last Update**: October 2024
- **Install**: `pip install mido==1.3.3`
- **Why We Use It**:
  - Actively maintained (recent updates)
  - Simple, Pythonic API
  - Excellent documentation
  - Cross-platform MIDI ports
- **Key Features**:
  ```python
  import mido
  
  # Read MIDI
  mid = mido.MidiFile('pattern.mid')
  
  # Write MIDI
  mid = mido.MidiFile()
  track = mido.MidiTrack()
  track.append(mido.Message('note_on', note=36, velocity=80, time=0))
  mid.tracks.append(track)
  mid.save('output.mid')
  
  # MIDI ports
  with mido.open_output('IAC Driver Bus 1') as port:
      port.send(mido.Message('note_on', note=60))
  ```
- **GitHub**: https://github.com/mido/mido (1.4K+ stars)

#### miditoolkit 1.0.1
- **Status**: ✅ Actively maintained
- **Python Support**: 3.6+
- **Install**: `pip install miditoolkit==1.0.1`
- **Why We Use It**: Enhanced MIDI manipulation (optional, complements mido)
- **Usage**:
  ```python
  import miditoolkit
  
  midi_obj = miditoolkit.MidiFile('input.mid')
  notes = midi_obj.instruments[0].notes
  
  # Modify timing
  for note in notes:
      note.start += 10  # Add 10 ticks offset
  ```
- **GitHub**: https://github.com/YatingMusic/miditoolkit (600+ stars)

#### MidiTok 2.1.8 ⭐ TOKENIZATION
- **Status**: ✅ Actively developed
- **Python Support**: 3.7+
- **Last Update**: October 2024
- **Install**: `pip install miditok==2.1.8`
- **Why We Use It**:
  - Framework-agnostic tokenization
  - Multiple tokenization schemes (REMI, Compound Word, etc.)
  - BPE/Unigram/WordPiece support
  - Drum-specific configurations
- **Configuration**:
  ```python
  from miditok import REMI, TokenizerConfig
  
  config = TokenizerConfig(
      use_chords=False,  # Not needed for drums
      use_programs=False,  # Drums only
      beat_res={(0, 4): 8, (4, 12): 4},  # Variable resolution
      num_velocities=32,
      special_tokens=["PAD", "BOS", "EOS"],
  )
  tokenizer = REMI(config)
  
  # Tokenize
  tokens = tokenizer("path/to/midi.mid")
  
  # Detokenize
  midi = tokenizer.decode(tokens)
  ```
- **GitHub**: https://github.com/Natooz/MidiTok (800+ stars)

### Backend Infrastructure

#### fastapi 0.121.0 ⭐ REST API
- **Status**: ✅ Production-ready, extremely active
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14
- **Last Update**: November 2025 (monthly releases)
- **Install**: `pip install fastapi==0.121.0 uvicorn[standard]==0.34.0`
- **Why We Use It**:
  - Modern async Python framework
  - Auto-generated OpenAPI docs
  - Pydantic v2 integration
  - Industry standard (Uber, Netflix)
- **Example**:
  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel
  
  app = FastAPI()
  
  class PatternRequest(BaseModel):
      style: str
      bars: int
  
  @app.post("/generate")
  async def generate(request: PatternRequest):
      return {"task_id": "abc-123"}
  ```
- **GitHub**: https://github.com/fastapi/fastapi (78K+ stars)

#### celery 5.5.3 ⭐ TASK QUEUE
- **Status**: ✅ Enterprise-grade
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
- **Last Update**: June 2025
- **Install**: `pip install celery==5.5.3 redis==7.0.1`
- **Why We Use It**:
  - Industry standard (Instagram, Uber, Mozilla)
  - Reliable task distribution
  - Multiple broker support
  - Extensive monitoring tools
- **Configuration**:
  ```python
  from celery import Celery
  
  celery_app = Celery(
      'drum_generator',
      broker='redis://localhost:6379/0',
      backend='redis://localhost:6379/1',
  )
  
  @celery_app.task
  def generate_pattern(params):
      # Heavy computation here
      return result
  ```
- **GitHub**: https://github.com/celery/celery (24K+ stars)

#### redis-py 7.0.1 ⭐ REDIS CLIENT
- **Status**: ✅ Official Redis Inc. client
- **Python Support**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Redis Server Support**: 7.2, 8.0, 8.2
- **Last Update**: October 2025
- **Install**: `pip install redis==7.0.1 hiredis==3.0.0`  # hiredis for performance
- **Why We Use It**:
  - Official Redis client
  - Regular updates
  - Full feature support
- **Usage**:
  ```python
  import redis
  
  r = redis.Redis(host='localhost', port=6379, db=0)
  r.set('key', 'value')
  r.get('key')
  ```
- **GitHub**: https://github.com/redis/redis-py (12K+ stars)

### Optional Dependencies

#### music21 9.9.1
- **Status**: ✅ MIT maintained, Python 3.11+ support as of v9.3
- **Python Support**: 3.10, 3.11, 3.12
- **Install**: `pip install music21==9.9.1`
- **Use Case**: Advanced music analysis (optional, heavy dependency)
- **When to Use**: Complex music theory operations (rarely needed)
- **When NOT to Use**: Simple MIDI I/O (use mido instead)
- **GitHub**: https://github.com/cuthbertLab/music21 (2.1K+ stars)
- **Documentation**: https://web.mit.edu/music21/doc/

#### python-osc 1.9.0
- **Status**: ✅ Actively maintained
- **Python Support**: 3.7+
- **Install**: `pip install python-osc==1.9.0`
- **Use Case**: Ableton Live OSC communication (alternative to MIDI)
- **Example**:
  ```python
  from pythonosc import udp_client
  
  client = udp_client.SimpleUDPClient("127.0.0.1", 11000)
  client.send_message("/live/track/0/clip/0/play", [])
  ```
- **GitHub**: https://github.com/attwad/python-osc (300+ stars)

---

## Dependencies We DON'T Use (Legacy/Abandoned)

### ❌ Magenta (TensorFlow-based)
- **Status**: Inactive since July 2025
- **Why Avoided**: 
  - TensorFlow dependency hell
  - CUDA version conflicts with PyTorch
  - No Python 3.11+ support
  - No modern CUDA 12.x support
- **Replacement**: Custom PyTorch models + MidiTok

### ❌ pretty-midi 0.2.10
- **Status**: Abandoned (last update September 2020)
- **Why Avoided**:
  - 5+ years no updates
  - No Python 3.11+ confirmation
  - Unmaintained dependency tree
- **Replacement**: mido 1.3.3

### ❌ pylive 0.4.0
- **Status**: Uncertain Ableton Live 12 compatibility
- **Why Avoided**:
  - No Live 12 confirmation
  - Small community
  - 14+ months since update
- **Replacement**: Direct MIDI via mido or custom OSC

### ❌ tensorflow 2.x
- **Status**: Conflicts with PyTorch in GPU environments
- **Why Avoided**:
  - CUDA version conflicts
  - Cannot coexist with PyTorch reliably
  - Magenta models are legacy TensorFlow
- **Replacement**: PyTorch 2.4+

---

## Compatibility Matrix

### Python Version Support

| Package        | 3.8 | 3.9 | 3.10 | 3.11 | 3.12 | 3.13 |
|----------------|-----|-----|------|------|------|------|
| torch          | ✅   | ✅   | ✅    | ✅    | ✅    | ❌    |
| transformers   | ✅   | ✅   | ✅    | ✅    | ✅    | ✅    |
| mido           | ✅   | ✅   | ✅    | ✅    | ✅    | ✅    |
| miditok        | ✅   | ✅   | ✅    | ✅    | ✅    | ✅    |
| fastapi        | ✅   | ✅   | ✅    | ✅    | ✅    | ✅    |
| celery         | ✅   | ✅   | ✅    | ✅    | ✅    | ✅    |
| redis-py       | ❌   | ✅   | ✅    | ✅    | ✅    | ✅    |
| music21        | ❌   | ❌   | ✅    | ✅    | ✅    | ❌    |

**Recommended:** Python 3.11 (best balance of compatibility and features)

### CUDA Compatibility

| PyTorch Version | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 |
|-----------------|-----------|-----------|-----------|
| 2.3.x           | ✅         | ✅         | ❌         |
| 2.4.x           | ✅         | ✅         | ✅         |
| 2.5.x (nightly) | ❌         | ✅         | ✅         |

**Recommended:** CUDA 12.1 with PyTorch 2.4.1

### Operating System Support

**Linux (Recommended for Production):**
- Ubuntu 22.04 LTS ✅
- Ubuntu 24.04 LTS ✅
- Debian 12 ✅
- All dependencies fully supported

**macOS (Development):**
- macOS 13+ (Ventura) ✅
- Apple Silicon (M1/M2/M3) ✅
- PyTorch MPS acceleration supported
- Note: CUDA not available (use CPU or MPS)

**Windows (Development):**
- Windows 11 ✅
- Windows 10 (with WSL2) ✅
- CUDA support available
- Some path handling differences (use `pathlib`)

---

## Complete requirements.txt

```txt
# Core ML
torch==2.4.1
torchvision==0.19.1
transformers==4.46.3

# MIDI Processing
mido==1.3.3
miditoolkit==1.0.1
miditok==2.1.8

# Backend
fastapi==0.121.0
uvicorn[standard]==0.34.0
celery==5.5.3
redis==7.0.1
hiredis==3.0.0

# Data Processing
numpy==2.0.2
pandas==2.2.3

# Utilities
python-dotenv==1.0.1
pydantic==2.10.2
pydantic-settings==2.6.1

# Optional
music21==9.9.1  # Advanced analysis only
python-osc==1.9.0  # Ableton OSC

# Development
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-cov==6.0.0
black==24.10.0
ruff==0.8.4
mypy==1.13.0

# Monitoring
structlog==24.4.0
sentry-sdk==2.19.2
```

---

## Installation Guide

### 1. System Requirements

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo apt install -y build-essential portaudio19-dev

# CUDA (if using GPU)
# Install from https://developer.nvidia.com/cuda-downloads
```

**macOS:**
```bash
brew install python@3.11
brew install portaudio
```

**Windows:**
```powershell
# Install Python 3.11 from python.org
# Install Visual Studio Build Tools
# Install CUDA Toolkit (if using GPU)
```

### 2. Virtual Environment Setup

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch (Choose One)

**CUDA 12.1 (Recommended for NVIDIA GPUs):**
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only:**
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

**macOS (MPS Acceleration):**
```bash
pip install torch==2.4.1 torchvision==0.19.1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python scripts/verify_installation.py
```

**Expected Output:**
```
✓ Python 3.11.10
✓ PyTorch 2.4.1 (CUDA 12.1)
✓ GPU: NVIDIA RTX 4090 (24GB)
✓ mido 1.3.3
✓ miditok 2.1.8
✓ FastAPI 0.121.0
✓ Celery 5.5.3
✓ Redis connection OK

All dependencies verified!
```

---

## Troubleshooting Common Issues

### Issue: PyTorch CUDA not available

**Symptoms:**
```python
import torch
torch.cuda.is_available()  # Returns False
```

**Solutions:**
1. Verify NVIDIA driver: `nvidia-smi`
2. Check CUDA version: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version
4. Verify no TensorFlow interference

### Issue: mido MIDI ports not found

**Symptoms:**
```python
mido.get_output_names()  # Returns []
```

**Solutions:**
```bash
# macOS: Enable IAC Driver
# Open Audio MIDI Setup → Window → Show MIDI Studio → IAC Driver → Device is online

# Linux: Install ALSA
sudo apt install alsa-utils

# Windows: Use loopMIDI
# Download from https://www.tobias-erichsen.de/software/loopmidi.html
```

### Issue: Redis connection refused

**Symptoms:**
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.
```

**Solutions:**
```bash
# Check Redis status
sudo systemctl status redis

# Start Redis
sudo systemctl start redis

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

---

## Dependency Update Policy

**Security Updates:** Apply immediately  
**Minor Updates:** Monthly review  
**Major Updates:** Quarterly evaluation  

**Before Updating:**
1. Check changelog for breaking changes
2. Test in development environment
3. Run full test suite
4. Update documentation if needed

---

## Related Documents

- **Project Overview**: `.cursorcontext/01_project_overview.md`
- **Architecture**: `.cursorcontext/02_architecture.md`
- **MIDI Operations**: `.cursorcontext/04_midi_operations.md`
- **ML Pipeline**: `.cursorcontext/05_ml_pipeline.md`
