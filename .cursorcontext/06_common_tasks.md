# Common Tasks - Context Document

**Document Type:** Instructional Context (Task-Specific Guidance)  
**Purpose:** Quick reference for frequent development tasks  
**Use Case:** Load when performing specific operations

---

## Quick Task Reference

### Generate Pattern from CLI

```bash
# Basic generation
python scripts/generate_pattern.py \
  --style "J Dilla" \
  --bars 4 \
  --tempo 95 \
  --output "patterns/jdilla_001.mid"

# With humanization disabled
python scripts/generate_pattern.py \
  --style "Metro Boomin" \
  --bars 8 \
  --tempo 140 \
  --no-humanize \
  --output "patterns/metro_001.mid"

# Batch generation
python scripts/generate_batch.py \
  --styles "J Dilla,Metro Boomin,Questlove" \
  --count 10 \
  --output-dir "patterns/batch/"
```

### Train New Model

```bash
# Full training pipeline
python src/training/train_transformer.py \
  --config configs/base.yaml \
  --data-dir data/groove_tokenized \
  --checkpoint-dir models/checkpoints \
  --use-wandb

# Resume from checkpoint
python src/training/train_transformer.py \
  --config configs/base.yaml \
  --resume-from models/checkpoints/epoch_50.pt

# Fine-tune on specific style
python src/training/finetune.py \
  --base-model models/checkpoints/base_v1.pt \
  --style "J Dilla" \
  --data-dir data/jdilla_patterns \
  --epochs 20
```

### Tokenize MIDI Dataset

```bash
# Tokenize Groove MIDI Dataset
python scripts/tokenize_dataset.py \
  --input-dir data/groove_midi \
  --output-dir data/groove_tokenized \
  --tokenizer-config configs/remi_tokenizer.json

# Verify tokenization
python scripts/verify_tokens.py \
  --tokens-dir data/groove_tokenized \
  --sample-size 100
```

### Run API Server

**Windows/Linux/macOS:**
```bash
# Development mode (auto-reload)
# Make sure virtual environment is activated first
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With HTTPS (Linux/macOS)
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
```

**Note:** In Cursor IDE, you can also use the integrated terminal which will use your activated virtual environment.

### Start Celery Workers

**Windows/Linux/macOS:**
```bash
# Make sure virtual environment is activated and Redis is running

# GPU worker (pattern generation)
celery -A src.tasks.worker worker -Q gpu_generation -c 2 --loglevel=info --max-tasks-per-child=100

# CPU worker (MIDI processing)
celery -A src.tasks.worker worker -Q midi_processing -c 4 --loglevel=info

# Monitor with Flower
celery -A src.tasks.worker flower --port=5555
```

**Note:** Run these in separate terminal windows/tabs. In Cursor IDE, you can open multiple integrated terminals.

### Test Ableton Integration

```bash
# Test MIDI port connectivity
python scripts/test_midi_ports.py

# Send test pattern to Ableton
python scripts/send_to_ableton.py \
  --midi-file patterns/test.mid \
  --port "IAC Driver Bus 1"

# Test OSC communication
python scripts/test_osc.py \
  --host 127.0.0.1 \
  --port 11000
```

---

## Development Workflows

### Adding New Producer Style

**1. Collect MIDI Data**
```bash
# Create directory
mkdir -p data/producer_midi/newproducer

# Add MIDI files to directory
# Files should be named: newproducer_001.mid, newproducer_002.mid, etc.
```

**2. Tokenize Data**
```bash
python scripts/tokenize_dataset.py \
  --input-dir data/producer_midi/newproducer \
  --output-dir data/producer_tokenized/newproducer \
  --style-name "New Producer"
```

**3. Update Style Registry**
```python
# Edit src/models/styles.py
PRODUCER_STYLES['new_producer'] = {
    'swing': 0.55,
    'micro_timing': 0.010,
    'ghost_note_prob': 0.2,
    'velocity_variation': 0.12,
    'preferred_tempo': (100, 120),
    'characteristic_notes': [36, 38, 42, 49],
}
```

**4. Fine-tune Model**
```bash
python src/training/finetune.py \
  --base-model models/checkpoints/base_v1.pt \
  --style "New Producer" \
  --data-dir data/producer_tokenized/newproducer \
  --output models/checkpoints/newproducer_v1.pt
```

### Debugging Generation Issues

**Problem: Model generates invalid MIDI**

```python
# 1. Check tokenizer configuration
from miditok import REMI
tokenizer = REMI.load("models/tokenizers/remi_config.json")
print(tokenizer.config)

# 2. Validate tokens before detokenization
tokens = model.generate(style_id=1, max_length=256)
print(f"Generated {len(tokens)} tokens")
print(f"Vocabulary range: {tokens.min()} - {tokens.max()}")
print(f"Vocab size: {tokenizer.vocab_size}")

# 3. Check for out-of-vocabulary tokens
oov_tokens = tokens[tokens >= tokenizer.vocab_size]
if len(oov_tokens) > 0:
    print(f"WARNING: {len(oov_tokens)} out-of-vocabulary tokens")

# 4. Test detokenization
try:
    midi = tokenizer.decode(tokens)
    print("Detokenization successful")
except Exception as e:
    print(f"Detokenization failed: {e}")
```

**Problem: GPU Out of Memory**

```python
# 1. Clear cache
import torch
torch.cuda.empty_cache()

# 2. Reduce batch size
# Edit configs/base.yaml
# training:
#   batch_size: 16  # Reduce from 32

# 3. Enable gradient checkpointing
model.transformer.gradient_checkpointing_enable()

# 4. Use mixed precision
from torch.cuda.amp import autocast
with autocast(device_type='cuda'):
    outputs = model(input_ids, style_ids)
```

**Problem: Pattern quality is poor**

```bash
# 1. Check training loss
python scripts/plot_training_curves.py \
  --checkpoint-dir models/checkpoints

# 2. Evaluate on validation set
python src/training/evaluate.py \
  --model models/checkpoints/base_v1.pt \
  --data-dir data/groove_tokenized/val

# 3. Generate test patterns
python scripts/generate_test_suite.py \
  --model models/checkpoints/base_v1.pt \
  --output-dir patterns/test_suite

# 4. Adjust generation parameters
# Try different temperature, top_k, top_p values
python scripts/parameter_sweep.py \
  --model models/checkpoints/base_v1.pt \
  --style "J Dilla"
```

---

## Testing Checklist

### Before Committing Code

```bash
# 1. Run unit tests
pytest tests/ -v

# 2. Check test coverage
pytest tests/ --cov=src --cov-report=html

# 3. Lint code
ruff check src/
black --check src/

# 4. Type checking
mypy src/

# 5. Format code
black src/
ruff check --fix src/
```

### Before Deploying

```bash
# 1. Run integration tests
pytest tests/integration/ -v

# 2. Test API endpoints
python scripts/test_api.py --base-url http://localhost:8000

# 3. Load test
locust -f tests/load/locustfile.py --host http://localhost:8000

# 4. Test Celery workers
python scripts/test_celery_tasks.py

# 5. Check GPU availability
python scripts/check_gpu.py
```

---

## Useful Code Snippets

### Load and Inspect Model

```python
import torch
from src.models.transformer import DrumPatternTransformer

# Load checkpoint
checkpoint = torch.load("models/checkpoints/base_v1.pt")

# Print model info
print(f"Epoch: {checkpoint['epoch']}")
print(f"Train Loss: {checkpoint['train_loss']:.4f}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")

# Load model
model = DrumPatternTransformer(vocab_size=1000, n_styles=50)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
```

### Generate Pattern Programmatically

```python
import torch
from src.inference.model_loader import load_model
from src.midi.export import export_pattern
from miditok import REMI

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("jdilla", device=device)
tokenizer = REMI.load("models/tokenizers/remi_config.json")

# Generate tokens
style_id = 5  # J Dilla
tokens = model.generate(
    style_id=style_id,
    max_length=512,
    temperature=0.9,
    top_k=50,
    device=device
)

# Export to MIDI
output_path = export_pattern(
    tokens=tokens.cpu().numpy(),
    tokenizer=tokenizer,
    output_path=Path("output.mid"),
    tempo=95,
    time_signature=(4, 4),
    humanize=True,
    style_name="J Dilla"
)

print(f"Pattern saved to: {output_path}")
```

### Analyze MIDI File

```python
import mido
from collections import Counter

mid = mido.MidiFile("patterns/jdilla_001.mid")

# Get tempo
for track in mid.tracks:
    for msg in track:
        if msg.type == 'set_tempo':
            tempo = mido.tempo2bpm(msg.tempo)
            print(f"Tempo: {tempo} BPM")

# Count notes
notes = []
for track in mid.tracks:
    for msg in track:
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append(msg.note)

# Most common drums
note_counts = Counter(notes)
print("Most common drums:")
for note, count in note_counts.most_common(5):
    print(f"  Note {note}: {count} hits")

# Pattern length
total_ticks = sum(msg.time for track in mid.tracks for msg in track)
beats = total_ticks / mid.ticks_per_beat
bars = beats / 4  # Assuming 4/4
print(f"Pattern length: {bars:.1f} bars")
```

### Quick API Test

```python
import requests

# Generate pattern
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "producer_style": "J Dilla",
        "bars": 4,
        "tempo": 95,
        "time_signature": [4, 4],
        "humanize": True
    }
)

task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")

# Poll status
import time
while True:
    status_response = requests.get(
        f"http://localhost:8000/api/v1/status/{task_id}"
    )
    status = status_response.json()["status"]
    print(f"Status: {status}")
    
    if status == "complete":
        midi_path = status_response.json()["midi_path"]
        print(f"MIDI saved to: {midi_path}")
        break
    elif status == "failed":
        print("Generation failed")
        break
    
    time.sleep(0.5)
```

---

## Environment Setup

### Create Development Environment

**Windows (PowerShell):**
```powershell
# 1. Navigate to project directory
cd C:\Users\lefab\Documents\Dev\MidiDrumiGen

# 2. Create virtual environment (using Python 3.11)
py -3.11 -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1
# If execution policy error: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Install dependencies
pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 5. Download datasets (when script is available)
python scripts/download_groove_midi.py

# 6. Setup Redis (if Docker is available)
docker run -d -p 6379:6379 --name redis redis:7-alpine

# 7. Verify installation
python scripts/verify_installation.py
```

**Linux/macOS:**
```bash
# 1. Navigate to project directory
cd ~/Documents/Dev/MidiDrumiGen

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Download datasets
python scripts/download_groove_midi.py

# 5. Setup Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine

# 6. Verify installation
python scripts/verify_installation.py
```

### Docker Setup

```bash
# Build image
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Run tests in Docker
docker-compose run --rm api pytest tests/
```

---

## Monitoring and Debugging

### Check System Status

```bash
# GPU status
nvidia-smi

# GPU memory
python -c "import torch; print(f'GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB')"

# Redis status
redis-cli ping

# Celery workers
celery -A src.tasks.worker inspect active

# API health
curl http://localhost:8000/health
```

### View Logs

```bash
# FastAPI logs
tail -f logs/api.log

# Celery logs
tail -f logs/celery.log

# Application logs
tail -f logs/app.log

# Filter errors only
tail -f logs/app.log | grep ERROR
```

### Profile Model Performance

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = load_model("jdilla", device="cuda")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():
        tokens = model.generate(style_id=1, max_length=256)

print(prof.key_averages().table(sort_by="cuda_time_total"))

# Save profile
prof.export_chrome_trace("profile.json")
# View at chrome://tracing
```

---

## Troubleshooting

### Common Errors and Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Activate virtual environment: `source venv/bin/activate` |
| `CUDA out of memory` | Reduce batch size or clear cache: `torch.cuda.empty_cache()` |
| `Connection refused (Redis)` | Start Redis: `docker run -d -p 6379:6379 redis:7-alpine` |
| `MIDI port not found` | Enable IAC Driver (macOS) or install loopMIDI (Windows) |
| `Permission denied` (file access) | Check file permissions: `chmod 644 file.mid` |
| `Model checkpoint not found` | Download or train model first |
| `Invalid token IDs` | Check tokenizer vocab size matches model |
| `Slow generation` | Use GPU, reduce max_length, or enable caching |

---

## Related Documents

- **Project Overview**: `.cursorcontext/01_project_overview.md`
- **Architecture**: `.cursorcontext/02_architecture.md`
- **Dependencies**: `.cursorcontext/03_dependencies.md`
- **MIDI Operations**: `.cursorcontext/04_midi_operations.md`
- **ML Pipeline**: `.cursorcontext/05_ml_pipeline.md`
