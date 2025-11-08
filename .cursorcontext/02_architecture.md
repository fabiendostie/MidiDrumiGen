# Architecture - Context Document

**Document Type:** Knowledge Context (System Design)  
**Purpose:** Detailed system architecture and component interactions  
**Use Case:** Load when designing new features or debugging integration issues

---

## System Architecture Overview

**Architecture Pattern:** Unified Application with Async Task Processing  
**NOT using:** Microservices (unnecessary complexity for this project)

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  Web UI     │  │  Ableton Live│  │  CLI Tool       │   │
│  │  (React)    │  │  (MIDI/OSC)  │  │  (Python)       │   │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘   │
└─────────┼─────────────────┼───────────────────┼────────────┘
          │                 │                   │
          └─────────────────┴───────────────────┘
                            │
                     [REST API / MIDI]
                            │
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Layer (async)                                    │  │
│  │  • /api/v1/generate                                   │  │
│  │  • /api/v1/status/{task_id}                           │  │
│  │  • /api/v1/styles                                     │  │
│  │  • /api/v1/download/{file_id}                         │  │
│  └─────────────────────┬────────────────────────────────┘  │
│                        │                                    │
│  ┌─────────────────────▼────────────────────────────────┐  │
│  │  Task Queue Interface                                 │  │
│  │  • Submit tasks to Celery                             │  │
│  │  • Query task status                                  │  │
│  │  • Retrieve results                                   │  │
│  └─────────────────────┬────────────────────────────────┘  │
└────────────────────────┼─────────────────────────────────────┘
                         │
                    [Redis Queue]
                         │
┌────────────────────────┼─────────────────────────────────────┐
│                        ▼                                      │
│                  Redis (v7.0+)                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Message Broker (Celery tasks)                     │   │
│  │  • Result Backend (task outputs)                     │   │
│  │  • Session Storage (API state)                       │   │
│  │  • Cache Layer (model metadata)                      │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                    [Task Messages]
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                   Celery Workers (v5.5+)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Worker Pool (3 queues)                              │   │
│  │                                                       │   │
│  │  Queue 1: MIDI Processing (CPU)                      │   │
│  │  • MIDI file validation                              │   │
│  │  • Tokenization                                      │   │
│  │  • Post-processing                                   │   │
│  │                                                       │   │
│  │  Queue 2: ML Generation (GPU)                        │   │
│  │  • Model inference                                   │   │
│  │  • Style conditioning                                │   │
│  │  • Pattern generation                                │   │
│  │                                                       │   │
│  │  Queue 3: Heavy Tasks (CPU/GPU)                      │   │
│  │  • Model training                                    │   │
│  │  • Dataset preprocessing                             │   │
│  │  • Batch generation                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                    [Accesses]
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                   Core Components                             │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐   │
│  │  ML Models     │  │  MIDI Engine   │  │  Storage     │   │
│  │  (PyTorch)     │  │  (mido)        │  │  (Local FS)  │   │
│  │                │  │                │  │              │   │
│  │  • Transformer │  │  • Read/Write  │  │  • Models    │   │
│  │  • Tokenizer   │  │  • Validation  │  │  • MIDI files│   │
│  │  • Embeddings  │  │  • Humanize    │  │  • Cache     │   │
│  └────────────────┘  └────────────────┘  └──────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. FastAPI Application Layer

**Responsibility:** Handle HTTP requests, validate inputs, queue tasks

**Technology:**
- FastAPI 0.121.0+ (ASGI framework)
- Uvicorn (ASGI server)
- Pydantic v2 (data validation)

**Key Files:**
```
src/api/
├── main.py              # FastAPI app initialization
├── routes/
│   ├── generate.py      # Pattern generation endpoints
│   ├── status.py        # Task status endpoints
│   └── admin.py         # Admin/health endpoints
├── models/
│   ├── requests.py      # Pydantic request models
│   └── responses.py     # Pydantic response models
└── middleware/
    ├── auth.py          # API key validation
    └── rate_limit.py    # Request rate limiting
```

**Example Endpoint Implementation:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Tuple

app = FastAPI(title="Drum Pattern Generator API")

class PatternRequest(BaseModel):
    producer_style: str = Field(..., example="J Dilla")
    bars: int = Field(4, ge=1, le=32)
    time_signature: Tuple[int, int] = Field((4, 4))
    tempo: int = Field(120, ge=40, le=300)
    humanize: bool = True

@app.post("/api/v1/generate")
async def generate_pattern(request: PatternRequest):
    # Validate style exists
    if request.producer_style not in AVAILABLE_STYLES:
        raise HTTPException(404, "Style not found")
    
    # Queue Celery task
    task = tasks.generate_pattern.delay(request.dict())
    
    return {
        "task_id": task.id,
        "status": "queued",
        "estimated_time": 2.0  # seconds
    }
```

### 2. Redis Message Broker

**Responsibility:** Queue management, result storage, caching

**Configuration:**
```python
# config/redis.py
REDIS_CONFIG = {
    'broker_url': 'redis://localhost:6379/0',
    'result_backend': 'redis://localhost:6379/1',
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_time_limit': 300,  # 5 minutes
    'worker_prefetch_multiplier': 1,  # Fair task distribution
}
```

**Redis Data Structures:**
```
Keys:
├── celery:task:{task_id}           # Task metadata
├── celery:result:{task_id}         # Task results
├── model:metadata:{style_name}     # Model metadata cache
├── session:{session_id}            # API session data
└── ratelimit:{api_key}:{endpoint}  # Rate limiting counters
```

### 3. Celery Worker Pool

**Responsibility:** Execute background tasks (generation, training, processing)

**Queue Configuration:**
```python
# src/tasks/worker.py
from celery import Celery

celery_app = Celery('drum_generator')
celery_app.config_from_object('config.redis:REDIS_CONFIG')

# Define queues with priorities
celery_app.conf.task_routes = {
    'tasks.generate_pattern': {'queue': 'gpu_generation'},
    'tasks.tokenize_midi': {'queue': 'midi_processing'},
    'tasks.train_model': {'queue': 'heavy_tasks'},
}

# Worker command examples:
# celery -A src.tasks.worker worker -Q gpu_generation -c 2 --max-tasks-per-child 100
# celery -A src.tasks.worker worker -Q midi_processing -c 4
# celery -A src.tasks.worker worker -Q heavy_tasks -c 1
```

**Task Definition Pattern:**
```python
@celery_app.task(bind=True, max_retries=3)
def generate_pattern(self, params: dict) -> dict:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model (cached)
        model = load_model(params['producer_style']).to(device)
        
        # Generate pattern
        with torch.no_grad():
            tokens = model.generate(
                style=params['producer_style'],
                bars=params['bars'],
                tempo=params['tempo'],
            )
        
        # Export MIDI
        midi_path = export_midi(tokens, params)
        
        return {
            "status": "complete",
            "midi_path": str(midi_path),
            "duration_ms": self.request.time_elapsed * 1000
        }
        
    except torch.cuda.OutOfMemoryError:
        # Fallback to CPU
        logger.warning("GPU OOM, retrying on CPU")
        model.to("cpu")
        self.retry(countdown=5)
        
    except Exception as exc:
        self.retry(exc=exc, countdown=60)
```

### 4. ML Model Layer

**Responsibility:** Pattern generation via Transformer model

**Model Architecture:**
```python
# src/models/transformer.py
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class DrumPatternTransformer(nn.Module):
    """
    Custom Transformer for drum pattern generation.
    
    Architecture:
    - Input: Tokenized MIDI + Style embedding
    - Encoder: 12-layer Transformer (768-dim)
    - Style Conditioning: Learned embeddings (128-dim)
    - Decoder: Causal self-attention
    - Output: Token logits → Sampling → Detokenization
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_styles: int,
        n_positions: int = 2048,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
    ):
        super().__init__()
        
        # Transformer config
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.transformer = GPT2LMHeadModel(config)
        
        # Style conditioning
        self.style_embeddings = nn.Embedding(n_styles, 128)
        self.style_projection = nn.Linear(128, n_embd)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        style_ids: torch.Tensor,
        **kwargs
    ):
        # Get style embeddings
        style_emb = self.style_embeddings(style_ids)  # (batch, 128)
        style_emb = self.style_projection(style_emb)  # (batch, 768)
        
        # Add style to inputs
        inputs_embeds = self.transformer.transformer.wte(input_ids)
        inputs_embeds = inputs_embeds + style_emb.unsqueeze(1)
        
        # Forward through transformer
        return self.transformer(inputs_embeds=inputs_embeds, **kwargs)
    
    @torch.no_grad()
    def generate(
        self,
        style_id: int,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate drum pattern tokens."""
        # Implementation of sampling strategy
        pass
```

**Model Files Structure:**
```
models/
├── checkpoints/
│   ├── jdilla_v1.pt          # Trained model weights
│   ├── metroboomin_v1.pt
│   └── base_v1.pt
├── tokenizers/
│   ├── remi_config.json      # MidiTok configuration
│   └── vocab.json            # Token vocabulary
└── metadata/
    ├── styles.json           # Style name → ID mapping
    └── training_stats.json   # Training metrics
```

### 5. MIDI Processing Engine

**Responsibility:** MIDI I/O, validation, humanization

**Technology Stack:**
- **mido 1.3.3**: Core MIDI operations
- **miditoolkit 1.0.1**: Enhanced MIDI manipulation
- **MidiTok 2.1+**: Tokenization/detokenization

**Key Modules:**
```
src/midi/
├── io.py           # Read/write MIDI files
├── validate.py     # MIDI validation rules
├── humanize.py     # Humanization algorithms
├── export.py       # Export with metadata
└── constants.py    # GM drum mappings
```

**MIDI Processing Pipeline:**
```python
# src/midi/export.py
import mido
from pathlib import Path

def export_midi(
    tokens: torch.Tensor,
    params: dict,
    output_path: Path
) -> Path:
    """
    Export tokens to MIDI file.
    
    Pipeline:
    1. Detokenize (tokens → MIDI events)
    2. Validate (check GM drum mapping)
    3. Humanize (timing + velocity variation)
    4. Add metadata (tempo, time signature, track name)
    5. Write file
    """
    # Detokenize
    tokenizer = load_tokenizer()
    midi_events = tokenizer.decode(tokens)
    
    # Create MIDI file
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    
    # Add tempo
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(params['tempo'])))
    
    # Add time signature
    numerator, denominator = params['time_signature']
    track.append(mido.MetaMessage('time_signature', numerator=numerator, denominator=denominator))
    
    # Convert events to messages
    for event in midi_events:
        if params['humanize']:
            event = apply_humanization(event)
        
        track.append(mido.Message(
            'note_on' if event.type == 'on' else 'note_off',
            note=event.pitch,
            velocity=event.velocity,
            time=event.time,
            channel=9  # GM drums channel
        ))
    
    mid.tracks.append(track)
    mid.save(output_path)
    
    return output_path
```

### 6. Ableton Live Integration

**Two Integration Approaches:**

#### Approach A: MIDI Ports (Recommended)
```python
# src/ableton/midi_sender.py
import mido
import time

def send_to_ableton(midi_path: Path):
    """Send MIDI file to Ableton via virtual MIDI port."""
    # Open virtual MIDI port
    with mido.open_output('IAC Driver Bus 1') as port:
        mid = mido.MidiFile(midi_path)
        
        # Send all messages
        for msg in mid.play():
            port.send(msg)
            time.sleep(0.001)  # Small delay for stability
```

#### Approach B: OSC Protocol (Alternative)
```python
# src/ableton/osc_sender.py
from pythonosc import udp_client

def create_clip(track_idx: int, clip_idx: int, midi_path: Path):
    """Create clip in Ableton via OSC."""
    client = udp_client.SimpleUDPClient("127.0.0.1", 11000)
    
    # Create empty clip
    client.send_message(f"/live/track/{track_idx}/clip/{clip_idx}/create", [4])
    
    # Load MIDI data
    with open(midi_path, 'rb') as f:
        midi_data = f.read()
    
    client.send_message(f"/live/track/{track_idx}/clip/{clip_idx}/set_notes", [midi_data])
```

## Data Flow Examples

### Pattern Generation Flow

```
1. User → API Request
   POST /api/v1/generate
   {
     "producer_style": "J Dilla",
     "bars": 4,
     "tempo": 95
   }

2. FastAPI → Input Validation
   ✓ Style exists in database
   ✓ Parameters in valid ranges
   ✓ API key valid (if required)

3. FastAPI → Queue Task
   task = celery_app.send_task(
     'tasks.generate_pattern',
     args=[request_dict],
     queue='gpu_generation'
   )

4. Celery → Pick Up Task
   Worker receives from Redis queue

5. Worker → Load Model
   model = load_cached_model("jdilla")
   model.to("cuda")

6. Worker → Generate Tokens
   tokens = model.generate(
     style_id=STYLES["J Dilla"],
     max_length=512,
     temperature=0.9
   )

7. Worker → Export MIDI
   midi_path = export_midi(tokens, params)

8. Worker → Store Result
   return {"midi_path": str(midi_path)}

9. User → Poll Status
   GET /api/v1/status/{task_id}
   Response: {"status": "complete", "midi_path": "..."}

10. User → Download MIDI
    GET /api/v1/download/{file_id}
    Response: MIDI file binary
```

### Training Pipeline Flow

```
1. Script → Load Dataset
   dataset = GrooveMIDIDataset("data/groove/")

2. Script → Tokenize
   tokenizer = REMI(config)
   for midi_file in dataset:
       tokens = tokenizer(midi_file)
       save_tokens(tokens, midi_file.id)

3. Script → Create DataLoader
   dataloader = DataLoader(
       TokenizedDataset("data/tokens/"),
       batch_size=32,
       shuffle=True
   )

4. Script → Training Loop
   for epoch in range(num_epochs):
       for batch in dataloader:
           outputs = model(batch['input_ids'], batch['style_ids'])
           loss = outputs.loss
           loss.backward()
           optimizer.step()

5. Script → Save Checkpoint
   torch.save(model.state_dict(), f"models/checkpoints/epoch_{epoch}.pt")

6. Script → Evaluate
   validation_loss = evaluate(model, val_dataloader)
   log_metrics(epoch, training_loss, validation_loss)
```

## Deployment Architecture

### Development Environment
```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [redis]
    volumes: [".:/app"]
    
  celery_gpu:
    build: .
    command: celery -A src.tasks.worker worker -Q gpu_generation -c 2
    depends_on: [redis]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
  celery_cpu:
    build: .
    command: celery -A src.tasks.worker worker -Q midi_processing -c 4
    depends_on: [redis]
    
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### Production Deployment
```
┌─────────────────────────────────────────┐
│  Load Balancer (Nginx)                  │
│  • SSL termination                      │
│  • Rate limiting                        │
│  • Static file serving                  │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────┐              ┌────▼────┐
│ API-1  │              │  API-2  │
│ Docker │              │ Docker  │
└───┬────┘              └────┬────┘
    │                        │
    └────────────┬───────────┘
                 │
         ┌───────▼────────┐
         │  Redis Cluster │
         │  (Sentinel)    │
         └───────┬────────┘
                 │
    ┌────────────┴─────────────┐
    │                          │
┌───▼────┐               ┌────▼────┐
│Worker-1│               │ Worker-2│
│GPU Tier│               │GPU Tier │
└────────┘               └─────────┘
```

## Performance Optimization Strategies

### 1. Model Caching
```python
from functools import lru_cache

@lru_cache(maxsize=10)
def load_model(style_name: str) -> DrumPatternTransformer:
    """Cache loaded models in memory."""
    path = f"models/checkpoints/{style_name}_v1.pt"
    model = DrumPatternTransformer(vocab_size=1000, n_styles=50)
    model.load_state_dict(torch.load(path))
    return model
```

### 2. Batch Processing
```python
# Process multiple requests in single forward pass
def generate_batch(requests: List[PatternRequest]) -> List[torch.Tensor]:
    style_ids = torch.tensor([STYLES[r.style] for r in requests])
    patterns = model.generate_batch(style_ids)
    return patterns
```

### 3. GPU Memory Management
```python
# Clear cache between generations
torch.cuda.empty_cache()

# Use mixed precision training
from torch.cuda.amp import autocast
with autocast():
    outputs = model(input_ids)
```

## Monitoring and Observability

### Metrics to Track
- API response times (p50, p95, p99)
- Celery task queue length
- GPU utilization and memory
- Pattern generation success rate
- Model inference latency
- Redis connection pool usage

### Logging Strategy
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "pattern_generated",
    task_id=task.id,
    style=params['style'],
    duration_ms=elapsed_time,
    gpu_memory_mb=torch.cuda.memory_allocated() / 1e6
)
```

## Security Considerations

1. **API Key Authentication**: Required for production endpoints
2. **Rate Limiting**: 100 requests/hour per API key
3. **Input Sanitization**: Validate all file paths
4. **CORS Configuration**: Whitelist allowed origins
5. **Redis AUTH**: Enable password authentication
6. **Model Isolation**: Run workers in separate containers

## Related Documents

- **Project Overview**: `.cursorcontext/01_project_overview.md`
- **Dependencies**: `.cursorcontext/03_dependencies.md`
- **MIDI Operations**: `.cursorcontext/04_midi_operations.md`
- **ML Pipeline**: `.cursorcontext/05_ml_pipeline.md`
