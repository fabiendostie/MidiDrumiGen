# System Architecture - MidiDrumiGen v2.0

**Version:** 2.0.0  
**Last Updated:** 2025-11-17  
**Architecture Type:** Microservices with Event-Driven Research

---

## Architecture Overview

MidiDrumiGen v2.0 uses an **agent-based architecture** where specialized sub-agents coordinate to research artists and generate MIDI patterns.

### Core Paradigm Shift

**v1.x (Training):**  
Pre-train PyTorch models → Load model → Generate

**v2.0 (On-Demand):**  
Research artist → Cache profile → LLM generates → MIDI export

---

## System Layers

### 1. Presentation Layer
**Max for Live Device** (Ableton Live 11+)
- User interface (artist input, parameters, progress)
- JavaScript bridge to FastAPI backend
- MIDI clip import to Ableton

### 2. API Gateway Layer
**FastAPI REST API** (`src/api/main.py`)
- Request validation (Pydantic models)
- Authentication & rate limiting
- Response caching (Redis)
- Error handling & logging

**Key Endpoints:**
```python
POST /api/v1/research          # Trigger artist research
GET  /api/v1/research/{artist} # Check cache status
POST /api/v1/generate          # Generate patterns
POST /api/v1/augment/{artist}  # Add more sources
GET  /api/v1/task/{task_id}    # Poll task status
GET  /api/v1/similar/{artist}  # Find similar artists
```

### 3. Orchestration Layer
**Main Orchestrator** (`src/orchestrator/main.py`)

```python
class Orchestrator:
    async def process_request(artist_name, params):
        # 1. Check cache
        profile = await db.get_style_profile(artist_name)
        
        if not profile:
            # 2. Delegate to Research Orchestrator
            profile = await research_orchestrator.research_artist(artist_name)
        
        # 3. Delegate to Generation Coordinator
        midi_files = await generation_coordinator.generate(profile, params)
        
        return midi_files
```

**Responsibilities:**
- Entry point for all requests
- Cache hit/miss decisions
- Route to Research or Generation
- Monitor progress
- Handle global errors

### 4. Research Layer
**Research Orchestrator** (`src/research/orchestrator.py`)

```python
class ResearchOrchestrator:
    def __init__(self):
        self.collectors = {
            'papers': ScholarPaperCollector(),
            'articles': WebArticleCollector(),
            'audio': AudioAnalysisCollector(),
            'midi': MidiDatabaseCollector()
        }
        self.profile_builder = StyleProfileBuilder()
    
    async def research_artist(artist_name):
        # Run all collectors in parallel
        sources = await gather_all_sources(artist_name)
        
        # Build StyleProfile
        profile = await profile_builder.build(artist_name, sources)
        
        # Store in database
        await db.save_profile(profile)
        
        return profile
```

**Sub-Components:**

#### 4a. Scholar Paper Collector
- **APIs:** Semantic Scholar, arXiv, CrossRef
- **Output:** Text descriptions, tempo data, style features
- **Timeout:** 5 minutes

#### 4b. Web Article Collector
- **Sources:** Pitchfork, Rolling Stone, Drummerworld, Wikipedia
- **Technologies:** BeautifulSoup4, Scrapy, spaCy (NLP)
- **Output:** Style descriptions, equipment mentions
- **Timeout:** 5 minutes

#### 4c. Audio Analysis Collector
- **Sources:** YouTube (via yt-dlp)
- **Technologies:** Librosa (tempo/beat), Essentia (rhythm features)
- **Output:** Tempo, swing ratio, syncopation, velocity distribution
- **Timeout:** 8 minutes

#### 4d. MIDI Database Collector
- **Sources:** BitMIDI, FreeMIDI, Musescore
- **Technologies:** mido (MIDI parsing)
- **Output:** Pattern templates (kick/snare/hihat rhythms)
- **Timeout:** 2 minutes

#### 4e. Style Profile Builder
```python
@dataclass
class StyleProfile:
    artist_name: str
    text_description: str  # For LLM prompt
    quantitative_params: dict  # tempo, swing, velocities, etc.
    midi_templates: List[Path]
    embedding: np.ndarray  # For similarity search
    confidence_score: float  # 0.0-1.0
    sources_count: dict  # Count per source type
```

**Aggregation Logic:**
- Resolve conflicting data (e.g., multiple tempo values)
- Calculate weighted averages (by source confidence)
- Generate embedding for similarity search
- Calculate overall confidence score

### 5. Generation Layer

#### 5a. LLM Provider Manager
```python
class LLMProviderManager:
    providers = {
        'openai': OpenAIProvider(GPT-4-turbo),
        'anthropic': AnthropicProvider(Claude-3-opus),
        'google': GoogleProvider(Gemini-1.5-pro)
    }
    
    async def generate(prompt, system_prompt, provider='auto'):
        # Try primary provider
        try:
            return await providers[primary].generate(prompt, system_prompt)
        except:
            # Fallback to next provider
            ...
```

**Provider Interface:**
```python
class BaseLLMProvider:
    async def generate_midi_tokens(prompt, system_prompt) -> dict:
        # Returns JSON with MIDI notes array
        pass
```

#### 5b. Prompt Engineering
```python
SYSTEM_PROMPT = """
You are an expert drummer AI. Generate drum MIDI as JSON.

Output format:
{
  "notes": [
    {"pitch": 36, "velocity": 90, "time": 0, "duration": 480},
    ...
  ],
  "tempo": 120,
  "time_signature": [4, 4]
}

Drum mapping:
- 36: Kick
- 38: Snare
- 42: Closed Hi-Hat
- 46: Open Hi-Hat
...
"""

def build_user_prompt(style_profile, params):
    return f"""
Generate {params.bars} bars for {style_profile.artist_name}.

Style characteristics:
{style_profile.text_description}

Quantitative parameters:
- Tempo: {params.tempo} BPM (typical: {profile.tempo_range})
- Swing: {profile.swing_percent}%
- Ghost notes: {profile.ghost_note_prob}

Output JSON only.
"""
```

#### 5c. Template-Based Generator (Fallback)
```python
class TemplateGenerator:
    def generate(style_profile, params) -> dict:
        # Load MIDI templates from profile
        templates = load_templates(style_profile.midi_templates)
        
        # Extract patterns
        patterns = extract_patterns(templates)
        
        # Apply variations based on params
        patterns = apply_swing(patterns, profile.swing_percent)
        patterns = add_ghost_notes(patterns, profile.ghost_note_prob)
        patterns = vary_velocities(patterns, profile.velocity_mean)
        
        # Convert to JSON
        return patterns_to_json(patterns)
```

#### 5d. Hybrid Coordinator
```python
class HybridCoordinator:
    async def generate(profile, params):
        # Try LLM first
        try:
            result = await llm_manager.generate(profile, params)
            if validate_midi(result):
                return result
        except:
            pass
        
        # Fallback to templates
        return template_generator.generate(profile, params)
```

### 6. Data Layer

#### 6a. PostgreSQL Database
```sql
-- Artists table
CREATE TABLE artists (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    research_status VARCHAR(50),  -- pending/researching/cached/failed
    last_updated TIMESTAMP,
    sources_count INTEGER,
    confidence_score REAL
);

-- Research sources table
CREATE TABLE research_sources (
    id SERIAL PRIMARY KEY,
    artist_id INTEGER REFERENCES artists(id),
    source_type VARCHAR(50),  -- paper/article/audio/midi
    url TEXT,
    raw_content TEXT,
    extracted_data JSONB,
    confidence REAL
);

-- Style profiles table
CREATE TABLE style_profiles (
    id SERIAL PRIMARY KEY,
    artist_id INTEGER UNIQUE REFERENCES artists(id),
    text_description TEXT,
    quantitative_params JSONB,
    midi_templates_json JSONB,
    embedding vector(384),  -- pgvector for similarity search
    confidence_score REAL
);

-- Generation history table
CREATE TABLE generation_history (
    id SERIAL PRIMARY KEY,
    artist_id INTEGER REFERENCES artists(id),
    provider_used VARCHAR(50),
    generation_time_ms INTEGER,
    user_params JSONB,
    output_files JSONB
);
```

#### 6b. Redis Cache
- **Task Queue:** Celery tasks
- **Rate Limiting:** API request limits
- **Response Caching:** StyleProfile caching (30 days TTL)

### 7. MIDI Export Layer
```python
class MIDIExporter:
    def json_to_midi(midi_data: dict, humanize=True) -> Path:
        mid = MidiFile()
        track = MidiTrack()
        
        # Convert JSON notes to MIDI messages
        for note in midi_data['notes']:
            # Add micro-timing if humanize
            time = note['time']
            if humanize:
                time += random.randint(-15, 15)  # ±15 ticks
            
            track.append(Message('note_on', note=note['pitch'], 
                                velocity=note['velocity'], time=time))
        
        mid.save(output_path)
        return output_path
```

---

## Data Flow Diagrams

### Flow 1: First-Time Artist Research

```
[User Input: "John Bonham"]
        ↓
[Max for Live Device]
        ↓ HTTP POST /api/v1/generate
[FastAPI API Gateway]
        ↓
[Main Orchestrator]
        ↓ Check cache
[Database] → Cache MISS
        ↓
[Research Orchestrator]
        ↓
    ┌───┴───┬───┬───┐
    │       │   │   │
[Papers][Articles][Audio][MIDI]  ← Parallel collection
    │       │   │   │
    └───┬───┴───┴───┘
        ↓
[Style Profile Builder]
  - Aggregate data
  - Extract parameters
  - Generate embedding
  - Calculate confidence
        ↓
[Database] ← Save StyleProfile
        ↓
[LLM Provider Manager]
  - OpenAI → Success!
        ↓
[MIDI Exporter]
  - JSON → MIDI
  - Humanization
  - 4 variations
        ↓
[File Storage]
        ↓
[API Response] → MIDI file paths
        ↓
[Max for Live] → Import clips to Ableton
```

**Time:** ~15-20 minutes

### Flow 2: Cached Artist Generation

```
[User Input: "John Bonham"]
        ↓
[Max for Live Device]
        ↓ HTTP POST /api/v1/generate
[FastAPI API Gateway]
        ↓
[Main Orchestrator]
        ↓ Check cache
[Database] → Cache HIT → Load StyleProfile
        ↓
[LLM Provider Manager]
  - Build prompt with profile
  - OpenAI → Generate
        ↓
[MIDI Exporter]
  - JSON → MIDI
  - Humanization
        ↓
[File Storage]
        ↓
[API Response] → MIDI file paths
        ↓
[Max for Live] → Import clips to Ableton
```

**Time:** < 2 minutes

---

## Component Communication

### Message Formats

**Research Request:**
```json
{
  "command": "research_artist",
  "artist_name": "John Bonham",
  "depth": "full",
  "timeout_minutes": 20
}
```

**StyleProfile:**
```json
{
  "artist_name": "John Bonham",
  "text_description": "Known for powerful, syncopated beats...",
  "quantitative_params": {
    "tempo_min": 85,
    "tempo_max": 120,
    "swing_percent": 62.0,
    "ghost_note_prob": 0.4
  },
  "confidence_score": 0.89
}
```

**Generation Request:**
```json
{
  "artist_name": "John Bonham",
  "bars": 4,
  "tempo": 95,
  "time_signature": [4, 4],
  "variations": 4,
  "provider": "openai"
}
```

---

## Technology Stack Details

### Python Packages
```
# Core Framework
fastapi==0.109.2
uvicorn[standard]==0.27.1
pydantic==2.6.1

# Task Queue
celery==5.3.6
redis==5.0.1

# Database
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
pgvector==0.2.4

# LLM Providers
openai==1.12.0
anthropic==0.18.1
google-generativeai==0.3.2

# Research
beautifulsoup4==4.12.3
scrapy==2.11.1
librosa==0.10.1
essentia==2.1b6.dev1110
yt-dlp==2024.3.10
spacy==3.7.4
sentence-transformers==2.3.1

# MIDI
mido==1.3.2
```

---

## Deployment Architecture

### Development
```
[Local Machine]
├── FastAPI (localhost:8000)
├── PostgreSQL (Docker, port 5432)
├── Redis (Docker, port 6379)
├── Celery Worker (local process)
└── Ableton Live + Max for Live
```

### Production (Future)
```
[Load Balancer (Nginx)]
        ↓
[FastAPI Instances (ECS)]
        ↓
┌───────┴────────┐
│                │
[PostgreSQL (RDS)]  [Redis (ElastiCache)]
        │                │
        └────────┬───────┘
                 ↓
        [Celery Workers (ECS)]
                 ↓
        [S3 for MIDI files]
```

---

## Security & Performance

### Security
- API key authentication
- Rate limiting (100 req/hour)
- Input validation (Pydantic)
- HTTPS in production
- Environment variables for secrets

### Performance Optimization
- Database indexing (artist name, embeddings)
- Redis caching (30-day TTL)
- Async operations (asyncio)
- Parallel research collection
- Connection pooling

### Monitoring
- Prometheus metrics
- Structured logging (loguru)
- Error tracking (future: Sentry)
- Performance profiling

---

## Migration from v1.x

### Removed Components
- ❌ `src/training/` - PyTorch training
- ❌ `src/inference/mock.py` - MockDrumModel
- ❌ `src/inference/model_loader.py` - Model loading
- ❌ `src/models/transformer.py` - PyTorch model
- ❌ Training scripts, Groove MIDI data, tokenization

### New Components
- ✅ `src/orchestrator/` - Main coordinator
- ✅ `src/research/collectors/` - 4 data collectors
- ✅ `src/generation/providers/` - 3 LLM providers
- ✅ `src/database/` - Database layer
- ✅ `src/ableton/` - Max for Live device

### Modified Components
- 🔄 `src/research/` - Expanded with collectors
- 🔄 `src/api/` - New endpoints
- 🔄 `src/tasks/` - New Celery tasks
- 🔄 `src/midi/` - Enhanced for LLM output

---

## Quick Reference

- **Full Architecture Doc:** `docs/ARCHITECTURE.md` (75KB, comprehensive)
- **PRD:** `docs/PRD.md` (product requirements)
- **UI Spec:** `docs/UI.md` (Max for Live design)
- **Implementation Guide:** `docs/ORCHESTRATOR_META_PROMPT.md`

---

**This architecture enables unlimited artist support through on-demand research and caching, replacing the limited pre-training approach.**
