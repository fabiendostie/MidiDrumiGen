# Project Overview - MidiDrumiGen v2.0

**Version:** 2.0.0  
**Last Updated:** 2025-11-17  
**Architecture:** On-Demand Research + LLM Generation

---

## What is MidiDrumiGen?

MidiDrumiGen is an intelligent MIDI drum pattern generation plugin for Ableton Live that allows users to input **ANY** artist or band name and receive authentic drum patterns in that artist's style.

**Key Innovation:** On-demand multi-source research eliminates the need for pre-trained models, enabling unlimited artist support.

---

## Core Capabilities

### 1. On-Demand Artist Research
- **Automated Data Collection:**
  - Academic papers (Semantic Scholar, arXiv)
  - Music journalism (Pitchfork, Rolling Stone, Drummerworld)
  - Audio analysis (YouTube, tempo/rhythm extraction)
  - MIDI databases (BitMIDI, FreeMIDI)

### 2. Multi-Provider LLM Generation
- **Primary Providers:**
  - OpenAI GPT-4-turbo
  - Anthropic Claude 3 Opus/Sonnet
  - Google Gemini 1.5 Pro
- **Automatic Failover:** If one provider fails, system tries next
- **Cost Tracking:** Monitor API usage per provider

### 3. Style Profile Database
- **PostgreSQL with pgvector:** Vector similarity search for "artists like X"
- **Caching System:** First-time research takes 5-20 min, cached artists < 2 min
- **Augmentation:** Add more sources to improve quality

### 4. Ableton Live Integration
- **Max for Live Device:** Direct integration into Ableton workflow
- **Automatic Clip Creation:** Generated patterns appear in clip slots
- **User-Friendly UI:** Artist input, parameters, progress tracking

---

## System Architecture (High-Level)

```
User (Ableton Live)
       ↓
Max for Live Device
       ↓
FastAPI REST API
       ↓
┌──────────────┼──────────────┐
│              │              │
Orchestrator   │              │
│              │              │
├──────────────┼──────────────┤
│              │              │
Research ←────┘   Generation  │
(if not cached)   (always)    │
│                             │
- Papers Collector            - LLM Provider Manager
- Articles Collector            (OpenAI/Claude/Gemini)
- Audio Analyzer              - Template Generator
- MIDI Collector              - Hybrid Coordinator
│                             │
└─→ StyleProfile ─────────────┘
            ↓
    MIDI Export & Humanization
            ↓
    Ableton Clip Import
```

---

## Technology Stack

### Backend (Python 3.11+)
- **Web Framework:** FastAPI 0.109.2
- **Task Queue:** Celery 5.3.6 + Redis 7.0+
- **Database:** PostgreSQL 15+ with pgvector 0.2.4
- **LLM APIs:** OpenAI 1.12.0, Anthropic 0.18.1, Google GenAI 0.3.2

### Research & Analysis
- **Web Scraping:** BeautifulSoup4 4.12.3, Scrapy 2.11.1
- **Audio Analysis:** Librosa 0.10.1, Essentia 2.1b6
- **NLP:** spaCy 3.7.4, sentence-transformers 2.3.1

### MIDI Processing
- **MIDI I/O:** mido 1.3.2
- **Validation:** Custom validation pipeline

### Frontend
- **Max for Live:** Max 8.5+ (Ableton Live 11+)
- **JavaScript Bridge:** HTTP client for API communication

---

## Project Structure

```
MidiDrumiGen/
├── src/
│   ├── orchestrator/           # NEW: Main coordinator
│   ├── research/
│   │   ├── orchestrator.py     # NEW: Research coordinator
│   │   ├── collectors/         # NEW: 4 data collectors
│   │   │   ├── papers.py
│   │   │   ├── articles.py
│   │   │   ├── audio.py
│   │   │   └── midi_db.py
│   │   ├── profile_builder.py  # NEW: StyleProfile builder
│   │   ├── producer_agent.py   # KEEP: Original research
│   │   └── llm_synthesizer.py  # KEEP: LLM synthesis
│   ├── generation/             # NEW: LLM-based generation
│   │   ├── providers/
│   │   │   ├── openai.py
│   │   │   ├── anthropic.py
│   │   │   ├── google.py
│   │   │   └── manager.py
│   │   ├── prompt_builder.py
│   │   ├── template_generator.py
│   │   └── hybrid_coordinator.py
│   ├── database/               # NEW: Database layer
│   │   ├── models.py
│   │   └── manager.py
│   ├── midi/                   # KEEP: MIDI operations
│   │   ├── export.py
│   │   ├── humanize.py
│   │   ├── validate.py
│   │   └── style_transfer.py
│   ├── api/                    # UPDATE: New endpoints
│   │   ├── main.py
│   │   └── routes/
│   │       ├── research.py     # NEW
│   │       ├── generate.py     # UPDATE
│   │       └── utils.py        # NEW
│   ├── tasks/                  # UPDATE: Celery tasks
│   │   └── tasks.py
│   └── ableton/                # NEW: Max for Live
│       ├── MidiDrumGen.amxd
│       └── js/bridge.js
├── data/
│   └── producer_cache/         # KEEP: Cached research (will migrate to DB)
├── configs/
│   └── generation.yaml         # NEW: LLM & research config
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── UI.md
│   ├── ORCHESTRATOR_META_PROMPT.md
│   ├── DOCUMENTATION_INDEX.md
│   └── agents/                 # Sub-agent specs
└── docs/_old/                  # Archived training code
    └── archived/
        ├── src_training/       # Old PyTorch training
        ├── models/             # Old checkpoints
        ├── groove_midi/        # Old training data
        └── tokenized/          # Old tokenized data
```

---

## Key Differences from v1.x

### ❌ REMOVED (Training Approach)
- PyTorch model training (`src/training/`)
- Pre-trained model checkpoints (`models/`)
- MidiTok tokenization (`scripts/tokenize_*.py`)
- Groove MIDI Dataset (`data/groove_midi/`)
- MockDrumModel (`src/inference/mock.py`)
- Model loader (`src/inference/model_loader.py`)

### ✅ NEW (On-Demand Approach)
- Multi-source research pipeline (papers, articles, audio, MIDI)
- LLM-based generation (OpenAI, Claude, Gemini)
- PostgreSQL database with vector search
- Style profile caching and augmentation
- Max for Live device
- Research orchestration
- Hybrid generation (LLM + templates)

### 🔄 UPDATED (Enhanced)
- `src/research/` - Expanded with 4 collectors + orchestrator
- `src/api/` - New endpoints for research and generation
- `src/tasks/` - Async tasks for research and generation
- `src/midi/` - Enhanced MIDI export with LLM output support

---

## User Workflow

### First-Time Artist (15-20 minutes)
1. User types artist name in Max for Live device
2. System checks database (cache miss)
3. Research pipeline activates:
   - Papers collector searches academic databases
   - Articles collector scrapes music journalism
   - Audio analyzer extracts rhythm from YouTube
   - MIDI collector finds existing patterns
4. Style Profile Builder aggregates data
5. Profile stored in database
6. LLM generates patterns using profile
7. MIDI clips appear in Ableton

### Cached Artist (< 2 minutes)
1. User types artist name
2. System loads StyleProfile from database
3. LLM generates patterns immediately
4. MIDI clips appear in Ableton

### Augmentation (5 minutes)
1. User clicks "Augment Research"
2. System collects additional sources
3. Profile updated with new data
4. Confidence score improves
5. Better generation quality

---

## Performance Targets

- **Research Time:** < 20 min (first-time)
- **Generation Time:** < 2 min (cached)
- **Database Queries:** < 100ms
- **LLM API Calls:** < 30s
- **Confidence Score:** > 0.6 (usable), > 0.8 (excellent)

---

## Development Status

### Phase 1: Documentation ✅ COMPLETE
- PRD created
- Architecture designed
- UI specified
- Agent specs documented

### Phase 2: Infrastructure (IN PROGRESS)
- PostgreSQL + pgvector setup
- Database schema creation
- Environment configuration
- Archive old training code

### Phase 3-7: Implementation (PLANNED)
- Research pipeline (Week 2-3)
- Generation engine (Week 3-4)
- MIDI export (Week 4)
- Ableton integration (Week 5)
- Testing (Week 6)

---

## Quick Links

- **Complete Architecture:** `docs/ARCHITECTURE.md`
- **Product Requirements:** `docs/PRD.md`
- **UI Specification:** `docs/UI.md`
- **Implementation Guide:** `docs/ORCHESTRATOR_META_PROMPT.md`
- **Sub-Agent Specs:** `docs/agents/`

---

## Getting Started (Development)

### Setup
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies (will be updated)
pip install -r requirements.txt

# Setup PostgreSQL
# (Instructions in ARCHITECTURE.md)

# Configure environment
cp .env.example .env
# Edit .env with API keys
```

### Run API Server
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Run Celery Worker
```bash
celery -A src.tasks.worker worker --loglevel=info
```

### Run Tests
```bash
pytest tests/
```

---

**This is the definitive overview for v2.0 architecture. Refer to specific docs for implementation details.**
