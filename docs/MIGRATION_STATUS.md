# Migration Status: v1.0 → v2.0

**Project:** MidiDrumiGen
**Migration Start:** 2025-11-15
**Current Phase:** Phase 2 - Infrastructure Setup
**Status:** In Progress (30% complete)

---

## Executive Summary

MidiDrumiGen is undergoing a **fundamental architecture shift** from a training-based ML system (v1.0) to an on-demand LLM-based research and generation system (v2.0).

**Key Shift:**
- ❌ **v1.0:** PyTorch model training → Pre-trained models → Limited artists
- ✅ **v2.0:** Multi-source research → LLM generation → Unlimited artists

---

## Migration Overview

### Phase Completion Status

| Phase | Status | Progress | Completion |
|-------|--------|----------|------------|
| **Phase 1: Documentation** | ✅ Complete | 100% | 2025-11-17 |
| **Phase 2: Infrastructure** | 🔄 In Progress | 30% | ETA: Week 1 |
| **Phase 3: Research Pipeline** | ⏳ Not Started | 0% | ETA: Week 2-3 |
| **Phase 4: Generation Engine** | ⏳ Not Started | 0% | ETA: Week 3-4 |
| **Phase 5: MIDI Export** | ⏳ Not Started | 0% | ETA: Week 4 |
| **Phase 6: Ableton Integration** | ⏳ Not Started | 0% | ETA: Week 5 |
| **Phase 7: Testing** | ⏳ Not Started | 0% | ETA: Week 6 |

---

## Architecture Comparison

### v1.0 Architecture (Legacy - Being Replaced)

```
User Request
    ↓
FastAPI Endpoint
    ↓
Celery Task Queue
    ↓
Load Pre-Trained PyTorch Model
    ↓
Generate Tokens (MidiTok)
    ↓
Detokenize → MIDI
    ↓
Humanize & Export
```

**Components:**
- ✅ FastAPI REST API (routes exist)
- ✅ Celery task queue (basic setup)
- ❌ PyTorch training pipeline (TO BE ARCHIVED)
- ❌ MockDrumModel (TO BE REMOVED)
- ❌ MidiTok tokenization (TO BE REMOVED)
- ✅ MIDI export with mido (KEEP & ENHANCE)
- ✅ Humanization (KEEP & ENHANCE)

### v2.0 Architecture (Target - To Be Implemented)

```
User Request (Max for Live)
    ↓
FastAPI Backend
    ↓
Main Orchestrator
    ↓
┌─────────────┼─────────────┐
│             │             │
Research ←────┘  Generation │
(if not cached)  (always)   │
│                          │
- Paper Collector         - LLM Provider Manager
- Article Collector         (Claude/Gemini/OpenAI)
- Audio Analyzer          - Template Generator
- MIDI Collector          - Hybrid Coordinator
│                          │
└→ StyleProfile ──────────┘
        ↓
PostgreSQL + pgvector
        ↓
MIDI Export & Humanization
        ↓
Ableton Live (Max for Live)
```

**New Components (To Be Built):**
- ⏳ Main Orchestrator (`src/orchestrator/`)
- ⏳ Research Orchestrator (`src/research/orchestrator.py`)
- ⏳ 4 Research Collectors (`src/research/collectors/`)
- ⏳ Style Profile Builder (`src/research/profile_builder.py`)
- ⏳ LLM Provider Manager (`src/generation/providers/manager.py`)
- ⏳ 3 LLM Providers (`src/generation/providers/`)
- ⏳ Prompt Engineering (`src/generation/prompt_builder.py`)
- ⏳ Template Generator (`src/generation/template_generator.py`)
- ⏳ Hybrid Coordinator (`src/generation/hybrid_coordinator.py`)
- ⏳ Database Layer (`src/database/models.py`, `manager.py`)
- ⏳ Max for Live Device (`src/ableton/MidiDrumGen.amxd`)

---

## What's Been Completed ✅

### Phase 1: Documentation (100% Complete)

**Core Documents:**
- ✅ `docs/PRD.md` - Complete product requirements
- ✅ `docs/ARCHITECTURE.md` - Full system architecture
- ✅ `docs/UI.md` - Max for Live UI specification
- ✅ `docs/ORCHESTRATOR_META_PROMPT.md` - AI implementation guide
- ✅ `docs/DOCUMENTATION_INDEX.md` - Master documentation index
- ✅ `docs/GIT_WORKFLOW.md` - Git conventions

**Agent Specifications:**
- ✅ `docs/agents/RESEARCH_ORCHESTRATOR_AGENT.md` - Research coordinator spec
- ✅ `docs/agents/LLM_PROVIDER_MANAGER_AGENT.md` - LLM provider spec

**Context Engineering:**
- ✅ `.cursorcontext/01_project_overview.md` - v2.0 overview
- ✅ `.cursorcontext/02_architecture.md` - Orchestrator-agent design
- ✅ `.cursorcontext/03_dependencies.md` - Verified dependencies (2025-11-17)
- ✅ `.cursorcontext/04_midi_operations.md` - MIDI processing
- ✅ `.cursorcontext/05_generation_pipeline.md` - LLM generation flow
- ✅ `.cursorcontext/06_common_tasks.md` - Quick reference

**Development Guides:**
- ✅ `CLAUDE.md` - Claude Code CLI guidance (v2.0)
- ✅ `GEMINI.md` - Gemini AI guidance
- ✅ `README.md` - Updated for v2.0 architecture

**Status:** All planning and architecture documentation is complete and comprehensive.

---

## What's In Progress 🔄

### Phase 2: Infrastructure Setup (30% Complete)

**Completed:**
- ✅ `requirements.txt` - Updated with v2.0 dependencies (verified 2025-11-17)
  - Added: anthropic, google-generativeai, openai (LLM providers)
  - Added: pgvector, sqlalchemy, alembic (database)
  - Added: librosa, madmom (cross-platform audio analysis)
  - Added: beautifulsoup4, scrapy (research)
  - Removed: torch, torchvision, miditok (no training)

- ✅ `pyproject.toml` - Updated build config, linting, testing
- ✅ Git cleanup - Removed old training docs, archived legacy code

**In Progress:**
- 🔄 Database schema creation (PostgreSQL + pgvector)
- 🔄 Alembic migrations setup
- 🔄 Environment configuration (`.env` template)

**Not Started:**
- ⏳ PostgreSQL installation guide
- ⏳ Redis setup for Celery
- ⏳ Initial database migrations
- ⏳ Directory restructuring (`src/orchestrator/`, `src/generation/`, etc.)

---

## What Needs to Be Built ⏳

### Phase 3: Research Pipeline (Week 2-3)

**Priority: High**

**Components:**
1. **Research Orchestrator** (`src/research/orchestrator.py`)
   - Coordinates 4 collectors in parallel
   - Aggregates results into StyleProfile
   - Handles caching and error recovery

2. **Scholar Paper Collector** (`src/research/collectors/papers.py`)
   - Semantic Scholar API integration
   - arXiv API integration
   - Extract tempo, style descriptors

3. **Web Article Collector** (`src/research/collectors/articles.py`)
   - BeautifulSoup4 scraping
   - Scrapy crawling
   - spaCy NLP for entity extraction

4. **Audio Analysis Collector** (`src/research/collectors/audio.py`)
   - yt-dlp for YouTube download
   - librosa for tempo/beat detection
   - madmom for advanced rhythm analysis

5. **MIDI Database Collector** (`src/research/collectors/midi_db.py`)
   - BitMIDI, FreeMIDI search
   - mido MIDI parsing
   - Pattern template extraction

6. **Style Profile Builder** (`src/research/profile_builder.py`)
   - Aggregate multi-source data
   - Generate sentence-transformers embeddings
   - Calculate confidence scores
   - Store in PostgreSQL

**Testing:**
- Test with 10 diverse artists
- Validate confidence scores > 0.7
- Ensure research completes < 20 min

---

### Phase 4: Generation Engine (Week 3-4)

**Priority: High**

**Components:**
1. **LLM Provider Manager** (`src/generation/providers/manager.py`)
   - Provider abstraction layer
   - Automatic failover: Claude → Gemini → OpenAI
   - Cost tracking
   - Rate limit handling

2. **Anthropic Provider** (`src/generation/providers/anthropic.py`)
   - Claude 3.5 Sonnet integration
   - Streaming support
   - Error handling

3. **Google Provider** (`src/generation/providers/google.py`)
   - Gemini 2.5/3 integration
   - JSON mode
   - Fallback logic

4. **OpenAI Provider** (`src/generation/providers/openai.py`)
   - ChatGPT 5.1 integration
   - Structured output mode
   - Tertiary fallback

5. **Prompt Engineering** (`src/generation/prompt_builder.py`)
   - System prompt template
   - User prompt with StyleProfile
   - Few-shot examples from MIDI templates

6. **Template-Based Generator** (`src/generation/template_generator.py`)
   - Rule-based MIDI generation
   - Fallback when LLM fails
   - Pattern variation algorithms

7. **Hybrid Coordinator** (`src/generation/hybrid_coordinator.py`)
   - Try LLM first
   - Validate output
   - Fallback to templates if needed
   - Generate 4-8 variations

**Testing:**
- Validate all 3 LLM providers work
- Test failover logic
- Ensure generation < 2 min
- Verify output is valid MIDI

---

### Phase 5: MIDI Export (Week 4)

**Priority: Medium**

**Components:**
1. **Enhanced MIDI Exporter** (`src/midi/export.py`)
   - JSON → MIDI conversion
   - Support LLM output format
   - Humanization integration

2. **Validation Pipeline** (`src/midi/validate.py`)
   - Check note ranges (35-81)
   - Validate velocities (1-127)
   - Ensure timing consistency

**Existing (Keep & Enhance):**
- ✅ `src/midi/humanize.py` - Already exists, enhance for LLM output
- ✅ `src/midi/io.py` - mido wrapper, update for new formats

---

### Phase 6: Ableton Integration (Week 5)

**Priority: Medium**

**Components:**
1. **Max for Live Device** (`src/ableton/MidiDrumGen.amxd`)
   - UI layout (375×600px)
   - Text input for artist name
   - Progress bar
   - Parameter controls

2. **JavaScript Bridge** (`src/ableton/js/bridge.js`)
   - HTTP client to FastAPI
   - Task polling logic
   - Clip import via Live API

---

### Phase 7: Testing (Week 6)

**Priority: High**

**Test Coverage:**
- Unit tests for all new components
- Integration tests (database, Redis, LLM APIs)
- End-to-end workflow tests
- Performance benchmarks

---

## What to Archive/Remove ❌

### Legacy v1.x Components (Archive to `docs/_old/`)

**Training Infrastructure:**
- ❌ `src/training/` → `docs/_old/archived/src_training/`
  - `train_transformer.py`
  - `dataset.py`
  - `data_loader.py`

**Model Files:**
- ❌ `src/models/transformer.py` → Archive (PyTorch model definition)
- ❌ `src/inference/model_loader.py` → Archive (model loading)
- ❌ `src/inference/mock.py` → Archive (MockDrumModel)

**Tokenization:**
- ❌ `scripts/tokenize_dataset.py` → Archive
- ❌ MidiTok usage → Remove

**Old Configs:**
- ❌ `configs/base.yaml` → Archive (training config)

**Status:** Not yet archived, still present in codebase.

---

## Database Schema (To Be Created)

### StyleProfile Table (PostgreSQL + pgvector)

```sql
CREATE TABLE style_profiles (
    id UUID PRIMARY KEY,
    artist_name VARCHAR(255) UNIQUE NOT NULL,
    text_description TEXT,
    quantitative_params JSONB,
    midi_templates_json JSONB,
    embedding VECTOR(384),  -- sentence-transformers
    confidence_score FLOAT,
    sources_count JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX ON style_profiles USING ivfflat (embedding vector_cosine_ops);
```

### Research Sources Table

```sql
CREATE TABLE research_sources (
    id UUID PRIMARY KEY,
    artist_id UUID REFERENCES style_profiles(id),
    source_type VARCHAR(50),  -- paper/article/audio/midi
    url TEXT,
    raw_content TEXT,
    extracted_data JSONB,
    confidence FLOAT,
    collected_at TIMESTAMP
);
```

**Status:** Not yet created, needs Alembic migration.

---

## Directory Restructuring Required

### Current Structure (v1.x)
```
src/
├── api/              # Exists (update needed)
├── tasks/            # Exists (update needed)
├── models/           # Exists (legacy, needs cleanup)
├── inference/        # Exists (legacy, TO ARCHIVE)
├── training/         # Exists (legacy, TO ARCHIVE)
├── midi/             # Exists (keep & enhance)
├── research/         # Exists (partial, needs expansion)
└── ableton/          # Exists (placeholder)
```

### Target Structure (v2.0)
```
src/
├── orchestrator/     # NEW - Main coordinator
├── research/         # EXPAND - Add collectors, orchestrator
│   ├── orchestrator.py
│   ├── profile_builder.py
│   └── collectors/   # NEW
│       ├── papers.py
│       ├── articles.py
│       ├── audio.py
│       └── midi_db.py
├── generation/       # NEW - LLM providers
│   ├── providers/    # NEW
│   │   ├── base.py
│   │   ├── anthropic.py
│   │   ├── google.py
│   │   ├── openai.py
│   │   └── manager.py
│   ├── prompt_builder.py
│   ├── template_generator.py
│   └── hybrid_coordinator.py
├── database/         # NEW - Database layer
│   ├── models.py
│   └── manager.py
├── api/              # UPDATE - New endpoints
├── tasks/            # UPDATE - New Celery tasks
├── midi/             # ENHANCE - Keep & update
└── ableton/          # IMPLEMENT - Max for Live device
```

---

## Critical Dependencies (v2.0)

### Newly Added (2025-11-17)
- ✅ `anthropic==0.39.0` - Claude 3.5 Sonnet
- ✅ `google-generativeai==0.8.3` - Gemini 2.5/3
- ✅ `openai==1.54.5` - ChatGPT 5.1
- ✅ `pgvector==0.3.6` - Vector similarity search
- ✅ `sqlalchemy==2.0.36` - ORM
- ✅ `alembic==1.14.0` - Migrations
- ✅ `librosa==0.10.2.post1` - Audio analysis
- ✅ `madmom==0.16.1` - Beat tracking (cross-platform)
- ✅ `beautifulsoup4==4.12.3` - Web scraping
- ✅ `scrapy==2.12.0` - Web crawling
- ✅ `sentence-transformers==3.3.1` - Embeddings

### Removed (v1.x → v2.0)
- ❌ `torch` - No ML training
- ❌ `torchvision` - Not needed
- ❌ `miditok` - No tokenization

### Kept (Enhanced)
- ✅ `fastapi==0.115.4` - REST API
- ✅ `celery==5.4.0` - Task queue
- ✅ `redis==5.2.1` - Message broker
- ✅ `mido==1.3.3` - MIDI I/O

---

## Next Steps (Immediate Priority)

### Week 1: Complete Phase 2 Infrastructure

1. **Setup PostgreSQL + pgvector**
   - Install PostgreSQL 16+
   - Enable pgvector extension
   - Create database: `mididrumigen_db`

2. **Create Alembic Migrations**
   - Initialize Alembic
   - Create initial migration for StyleProfile + ResearchSource tables
   - Run migrations

3. **Archive Legacy Code**
   - Move `src/training/` → `docs/_old/archived/src_training/`
   - Move `src/inference/mock.py` → `docs/_old/archived/`
   - Move `src/inference/model_loader.py` → `docs/_old/archived/`
   - Move `src/models/transformer.py` → `docs/_old/archived/`

4. **Create New Directory Structure**
   - `mkdir src/orchestrator/`
   - `mkdir src/research/collectors/`
   - `mkdir src/generation/providers/`
   - `mkdir src/database/`

5. **Environment Configuration**
   - Update `.env.example` with LLM API keys
   - Add PostgreSQL connection string
   - Document setup instructions

---

## Risk Assessment

### High Risk
- **LLM API Costs:** Multi-provider usage could be expensive
  - Mitigation: Implement cost tracking, set budget alerts

- **Research Quality:** Some artists have limited data
  - Mitigation: Confidence scoring, manual curation option

### Medium Risk
- **Migration Time:** Underestimating implementation complexity
  - Mitigation: Incremental implementation, regular checkpoints

### Low Risk
- **MIDI Compatibility:** Export format issues
  - Mitigation: Use standard MIDI format, test with multiple DAWs

---

## Success Criteria

**Phase 2 Complete When:**
- ✅ PostgreSQL + pgvector installed and running
- ✅ Database schema created via Alembic
- ✅ Legacy code archived
- ✅ New directory structure in place
- ✅ Environment configuration documented

**MVP Complete When:**
- ✅ Can research any artist on-demand (5-20 min)
- ✅ Can generate patterns for cached artists (< 2 min)
- ✅ Generated MIDI matches artist style
- ✅ Max for Live device functional in Ableton

---

## References

**Complete Documentation:**
- `docs/PRD.md` - Product requirements
- `docs/ARCHITECTURE.md` - System architecture
- `docs/ORCHESTRATOR_META_PROMPT.md` - Implementation guide
- `docs/DOCUMENTATION_INDEX.md` - Master index

**Agent Specifications:**
- `docs/agents/RESEARCH_ORCHESTRATOR_AGENT.md`
- `docs/agents/LLM_PROVIDER_MANAGER_AGENT.md`

**Context Files:**
- `.cursorcontext/*.md` - Complete v2.0 context

---

**Last Updated:** 2025-11-17
**Next Review:** 2025-11-24 (Week 1 checkpoint)
**Migration Lead:** Development Team
