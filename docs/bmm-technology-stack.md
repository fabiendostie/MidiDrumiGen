# Technology Stack - MidiDrumiGen v2.0

**Generated:** 2025-11-17
**Project Type:** Backend (Python/FastAPI)
**Architecture Pattern:** Orchestrator-Agent with Service-Oriented Backend

---

## Technology Overview

| Category | Technology | Version | Justification |
|----------|-----------|---------|---------------|
| **Language** | Python | 3.11+ (3.12 recommended) | Modern async support, type hints, rich ML/AI ecosystem |
| **Web Framework** | FastAPI | 0.115.4 | High-performance async REST API, automatic OpenAPI docs, Pydantic validation |
| **ASGI Server** | Uvicorn | 0.32.1 | Production-ready async server with HTTP/2 support |
| **Validation** | Pydantic | 2.10.3 | Runtime type checking, data validation, settings management |

---

## Async Task Processing

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Celery** | 5.4.0 | Distributed async task queue for research & generation |
| **Redis** | 5.2.1 | Message broker, result backend, caching layer |
| **Flower** | 2.0.1 | Real-time Celery monitoring UI |
| **Kombu** | 5.4.2 | Messaging abstraction layer |

**Architecture Rationale:** Research tasks (5-20 min) and LLM calls (10-30s) run async to avoid blocking API.

---

## Database & Persistence

| Technology | Version | Purpose |
|-----------|---------|---------|
| **PostgreSQL** | 16+ | Primary relational database for style profiles & research |
| **SQLAlchemy** | 2.0.36 | ORM with async support (asyncpg) |
| **Alembic** | 1.14.0 | Database migration management |
| **psycopg2-binary** | 2.9.10 | PostgreSQL adapter (sync operations) |
| **asyncpg** | 0.30.0 | High-performance async PostgreSQL driver |
| **pgvector** | 0.3.6 | Vector similarity search for artist embeddings |

**Schema:** StyleProfiles with 384-dim embeddings (sentence-transformers) for "similar artists" feature.

---

## LLM Providers (v2.0 Core Feature)

| Provider | Library | Version | Role | Model |
|----------|---------|---------|------|-------|
| **Anthropic** | anthropic | 0.39.0 | **Primary** | Claude 3.5 Sonnet (best quality) |
| **Google** | google-generativeai | 0.8.3 | **Secondary** | Gemini 2.5/3 (fast & cheap) |
| **OpenAI** | openai | 1.54.5 | **Tertiary/Fallback** | ChatGPT 5.1 |

**Failover Strategy:** Claude → Gemini → OpenAI (automatic cascade on failure)

---

## Research & Data Collection

### Web Scraping
| Technology | Version | Purpose |
|-----------|---------|---------|
| **BeautifulSoup4** | 4.12.3 | HTML parsing for articles & web content |
| **lxml** | 5.3.0 | Fast XML/HTML parser backend |
| **Scrapy** | 2.12.0 | Web crawling framework for structured data |
| **newspaper3k** | 0.2.8 | Article extraction & NLP |
| **requests** | 2.32.3 | HTTP library (sync) |
| **aiohttp** | 3.11.10 | Async HTTP client |
| **httpx** | 0.28.1 | Modern HTTP client (sync/async) |

### Audio Analysis
| Technology | Version | Purpose |
|-----------|---------|---------|
| **librosa** | 0.10.2.post1 | **PRIMARY** - Tempo detection, rhythm analysis, beat tracking |
| **madmom** | 0.16.1 | Advanced beat/downbeat detection, 100% cross-platform |
| **yt-dlp** | 2024.12.13 | YouTube audio download (research source) |
| **soundfile** | 0.12.1 | Audio file I/O |
| **audioread** | 3.0.1 | Audio file decoding |

**Note:** Essentia removed due to poor Windows compatibility. Librosa + madmom provide complete coverage.

### NLP & Embeddings
| Technology | Version | Purpose |
|-----------|---------|---------|
| **spaCy** | 3.8.2 | NLP, entity extraction from text sources |
| **sentence-transformers** | 3.3.1 | Text embeddings for style profile similarity (384-dim) |
| **transformers** | 4.47.1 | HuggingFace models (embeddings only, NO training) |
| **tokenizers** | 0.21.0 | Fast tokenization |

---

## MIDI Processing

| Technology | Version | Purpose |
|-----------|---------|---------|
| **mido** | 1.3.3 | MIDI I/O, file parsing, message creation |
| **python-rtmidi** | 1.5.8 | Real-time MIDI communication |

**Justification:** mido is actively maintained (2025), cross-platform, pure Python. Replaces abandoned pretty-midi.

---

## Scientific Computing

| Technology | Version | Purpose |
|-----------|---------|---------|
| **NumPy** | 1.26.4 | Numerical arrays, audio processing |
| **SciPy** | 1.14.1 | Scientific algorithms, signal processing |
| **pandas** | 2.2.3 | Data manipulation, CSV handling |

---

## Utilities

| Technology | Version | Purpose |
|-----------|---------|---------|
| **python-dotenv** | 1.0.1 | Environment variable management (.env files) |
| **PyYAML** | 6.0.2 | Configuration file parsing |
| **loguru** | 0.7.3 | Structured logging with rotation |
| **click** | 8.1.8 | CLI tool framework |
| **tqdm** | 4.67.1 | Progress bars for research/generation |
| **python-dateutil** | 2.9.0 | Date/time utilities |

---

## Testing & Development

| Technology | Version | Purpose |
|-----------|---------|---------|
| **pytest** | 8.3.4 | Testing framework |
| **pytest-asyncio** | 0.24.0 | Async test support |
| **pytest-cov** | 6.0.0 | Code coverage reporting |
| **pytest-mock** | 3.14.0 | Mocking utilities |
| **black** | 24.10.0 | Code formatter (line length: 100) |
| **ruff** | 0.8.4 | Fast linter (replaces flake8, isort, etc.) |
| **mypy** | 1.13.0 | Static type checker |
| **pre-commit** | 4.0.1 | Git hooks for automated checks |

**Test Markers:**
- `unit` - Fast tests, no external deps
- `integration` - Database/Redis required
- `slow` - LLM calls, audio processing
- `llm` - Requires API keys

---

## Performance Optimizations

| Technology | Version | Platform | Purpose |
|-----------|---------|----------|---------|
| **orjson** | 3.10.12 | All | Faster JSON serialization (C extension) |
| **uvloop** | 0.21.0 | Linux/macOS | Faster asyncio event loop |
| **httptools** | 0.6.4 | All | Faster HTTP parsing for Uvicorn |

---

## Monitoring & Observability (Production)

| Technology | Version | Purpose |
|-----------|---------|---------|
| **sentry-sdk** | 2.19.2 | Error tracking & performance monitoring |
| **prometheus-client** | 0.21.1 | Metrics export for Prometheus |

---

## Removed Dependencies (v1.x → v2.0 Migration)

| Library | Reason for Removal |
|---------|-------------------|
| ❌ **PyTorch** | No ML training in v2.0 (LLM-based generation) |
| ❌ **torchvision** | No training needed |
| ❌ **MidiTok** | No tokenization (MIDI generated from LLM JSON) |
| ❌ **Magenta** | TensorFlow-based, inactive since 2025 |
| ❌ **pretty-midi** | Abandoned since 2020 |
| ❌ **TensorFlow** | Never used, conflicts with PyTorch |

---

## Architecture Pattern Summary

### Orchestrator-Agent Pattern
```
Main Orchestrator
├── Research Orchestrator (Celery task)
│   ├── Scholar Paper Collector
│   ├── Web Article Collector
│   ├── Audio Analysis Collector
│   └── MIDI Database Collector
├── LLM Provider Manager
│   ├── Anthropic Provider (Claude)
│   ├── Google Provider (Gemini)
│   └── OpenAI Provider (ChatGPT)
└── MIDI Processor
    ├── JSON → MIDI Converter
    ├── Humanizer (timing/velocity variance)
    └── Validator
```

### Service Layers
1. **API Layer** (FastAPI) - HTTP endpoints, request validation
2. **Task Queue Layer** (Celery + Redis) - Async research & generation
3. **Data Layer** (PostgreSQL + pgvector) - Persistence & vector search
4. **Integration Layer** (Max for Live) - Ableton communication

---

## Deployment Requirements

### Minimum System Requirements
- **Python:** 3.11+ (3.12 recommended)
- **PostgreSQL:** 16+ with pgvector extension
- **Redis:** 7.2+
- **Memory:** 8GB RAM minimum, 16GB recommended
- **Storage:** 10GB for database, 100GB for MIDI cache (scalable)

### External Services
- **Anthropic API:** API key required (primary LLM)
- **Google Cloud:** API key for Gemini (secondary)
- **OpenAI API:** API key (fallback)
- **YouTube:** yt-dlp for research (no API key needed)

### Platform Support
- **Primary:** Windows 10/11, macOS 11+
- **Production:** Linux (Ubuntu 22.04+, Debian 12+)
- **Note:** All dependencies are 100% cross-platform verified (2025-11-17)

---

## Development Workflow

### Setup
```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Setup PostgreSQL with pgvector
createdb mididrumigen_db
psql -d mididrumigen_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations
alembic upgrade head
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit        # Fast unit tests only
pytest -m integration # Integration tests (needs DB/Redis)
pytest -m slow        # Slow tests (LLM, audio)

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

---

**Last Updated:** 2025-11-17
**Verified By:** Brownfield Project Documentation Workflow
