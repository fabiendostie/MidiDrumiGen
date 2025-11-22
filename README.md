# MidiDrumiGen v2.0
## AI-Powered Artist-Style Drum Pattern Generator for Ableton Live

**On-demand research + LLM-based MIDI generation for any producer or band style**

---

## What is MidiDrumiGen v2.0?

MidiDrumiGen is an intelligent Max for Live device that generates authentic drum patterns in the style of *any* artist or producer you specify. Using advanced multi-source research and state-of-the-art language models, it creates MIDI clips directly in Ableton Live that capture the essence of your chosen drumming style.

### Key Features

✨ **Universal Style Emulation** - Input any producer or band name and generate patterns matching their style
🔍 **Comprehensive Research** - Analyzes scholarly papers, MIDI databases, audio features, and web articles
🤖 **Multi-Provider LLM** - Uses OpenAI GPT-4, Claude, or Gemini for intelligent generation
⚡ **Fast Generation** - Creates patterns in under 2 minutes
💾 **Smart Caching** - Builds a database of researched artists for instant re-use
🎛️ **Max for Live Integration** - Native Ableton Live device with intuitive UI
📊 **Style Augmentation** - Continuously improves style profiles with more data

---

## Architecture Overview

### v2.0 Design Philosophy

**v2.0 shifts from pre-trained models to on-demand intelligence:**

- ❌ No pre-training required
- ❌ No fixed artist dataset
- ✅ Research any artist on demand
- ✅ LLM-based generation
- ✅ Continuous learning via augmentation

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Max for Live Device (UI)                     │
│              User inputs artist name + parameters               │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP API
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Orchestrator (Meta-Agent)                       │  │
│  │  Coordinates: Research → Profile → LLM → MIDI → Export   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │ Celery Tasks
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Sub-Agent Pipeline                           │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────────┐  │
│  │   Research    │→ │ LLM Provider  │→ │ MIDI Processor    │  │
│  │ Orchestrator  │  │   Manager     │  │ & Humanizer       │  │
│  └───────────────┘  └───────────────┘  └────────────────────┘  │
│         │                   │                      │            │
│         ▼                   ▼                      ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            PostgreSQL + pgvector                        │   │
│  │  Stores: StyleProfiles, Research, Embeddings, MIDI      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack (v2.0)

**Backend:**
- Python 3.11+ (Recommended: 3.12)
- FastAPI 0.115+ for REST API
- Celery 5.4+ for async task processing
- Redis 7.2+ for message queue & caching
- PostgreSQL 16+ with pgvector for vector search

**LLM Providers:**
- Anthropic Claude 3.5 Sonnet (Primary)
- Google Gemini 2.5/3 (Secondary)
- OpenAI ChatGPT 5.1 (Tertiary/Fallback)
- Future: Local LLM support (Ollama, LM Studio)

**Research Tools:**
- BeautifulSoup4 + Scrapy for web scraping
- Librosa + madmom for audio analysis (100% cross-platform)
- yt-dlp for YouTube content
- spaCy + Sentence Transformers for NLP

**MIDI Processing:**
- mido 1.3.3 for MIDI I/O
- Custom humanization algorithms

**Ableton Integration:**
- Max for Live 8.5+
- Ableton Live 11+ (Suite/Standard)

---

## How It Works

### 1. User Input (Max for Live Device)
```
Artist: "J Dilla"
Bars: 4
Time Signature: 4/4
Tempo: 95 BPM
Humanize: On
```

### 2. Research Phase (If New Artist)
The Research Orchestrator coordinates 4 specialized collectors:

**📚 Scholar Collector**
- Searches Semantic Scholar, ArXiv, CrossRef
- Extracts: Tempo ranges, swing percentages, rhythmic patterns
- Example: "J Dilla's signature 'drunken' swing at 10-12% humanization"

**🎵 MIDI Collector**
- Queries MIDI databases and repositories
- Analyzes: Note patterns, velocity curves, ghost note frequency
- Extracts template patterns for reference

**📝 Text Collector**
- Scrapes music production forums, articles, interviews
- NLP extraction of: Techniques, influences, equipment mentions
- Example: "Known for MPC3000 swing, quantization at 62%"

**🎧 Audio Collector**
- Downloads representative tracks via yt-dlp
- Librosa analysis: Tempo stability, beat grid, frequency content, rhythm patterns
- madmom analysis: Advanced beat tracking, downbeat detection, tempo confidence

### 3. Style Profile Creation
Collected data is synthesized into a `StyleProfile`:

```json
{
  "artist_name": "J Dilla",
  "tempo_range": [85, 105],
  "swing_percentage": 11.5,
  "velocity_variation": 0.15,
  "ghost_note_probability": 0.25,
  "quantization_grid": "1/16",
  "rhythmic_complexity": "high",
  "kick_pattern_templates": [...],
  "snare_pattern_templates": [...],
  "hihat_pattern_templates": [...],
  "textual_description": "Neo-soul hip-hop with heavy swing...",
  "source_count": {
    "scholarly": 4,
    "midi": 12,
    "text": 23,
    "audio": 8
  }
}
```

### 4. LLM Generation
The LLM Provider Manager:
1. Constructs a detailed prompt with StyleProfile
2. Sends to primary LLM (Claude 3.5 Sonnet)
3. Falls back to Gemini → OpenAI if needed
4. Parses JSON response with MIDI note events

**Example Prompt:**
```
You are a drum programmer specializing in J Dilla's style.

Style Profile:
- Tempo: 85-105 BPM (target: 95)
- Swing: 11.5% (signature 'drunken' feel)
- Velocity: High variation (0.15), emphasize weak beats
- Ghost notes: 25% probability on hi-hats
...

Generate a 4-bar drum pattern as JSON:
[
  {"time": 0.0, "note": 36, "velocity": 100, "duration": 0.1},
  {"time": 0.48, "note": 42, "velocity": 45, "duration": 0.05},
  ...
]
```

### 5. MIDI Processing & Humanization
- Parse LLM JSON output
- Apply humanization (timing jitter, velocity variation)
- Validate MIDI (no overlaps, valid ranges)
- Export to `.mid` file

### 6. Delivery to Ableton
- Max for Live device receives MIDI file path
- Creates MIDI clip in session view
- Populates multiple clip slots (variations)
- User can drag to arrangement or trigger immediately

---

## Installation

### Prerequisites
- **Ableton Live 11+** (Suite or Standard with Max for Live)
- **Python 3.11+** (3.12 recommended)
- **PostgreSQL 16+** with pgvector extension
- **Redis 7.2+**
- **API Keys** for at least one LLM provider (OpenAI, Anthropic, or Google)

### 1. Clone Repository
```bash
git clone https://github.com/your-org/MidiDrumiGen.git
cd MidiDrumiGen
```

### 2. Setup Python Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 4. Setup PostgreSQL
```bash
# Option A: Using Docker (recommended for Windows/development)
docker run -d --name postgres-mididrum \
  -e POSTGRES_DB=mididrumigen_db \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_USER=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Option B: Using local PostgreSQL installation
createdb mididrumigen_db
psql -d mididrumigen_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations (both options)
alembic upgrade head
```

### 5. Setup Redis
```bash
# Using Docker (recommended for Windows)
docker run -d --name redis-mididrum -p 6379:6379 redis:7.4-alpine

# Or use system service (macOS/Linux)
brew services start redis  # macOS
sudo systemctl start redis # Linux
```

### 6. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys and database URLs
```

Required environment variables:
```bash
# At least one LLM provider (all three recommended for fallback)
ANTHROPIC_API_KEY=sk-ant-api03-...  # Primary (Claude 3.5 Sonnet)
GOOGLE_API_KEY=AIza...              # Secondary (Gemini 2.5/3)
OPENAI_API_KEY=sk-proj-...          # Tertiary/Fallback (ChatGPT 5.1)

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mididrumigen_db

# Redis
REDIS_URL=redis://localhost:6379/0
```

### 7. Start Services
```bash
# Terminal 1: FastAPI server
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Celery worker
celery -A src.tasks.worker worker --loglevel=info

# Terminal 3 (Optional): Celery monitoring
celery -A src.tasks.worker flower --port=5555
```

### 8. Install Max for Live Device
1. Open Ableton Live
2. Drag `max_for_live/MidiDrumiGen.amxd` to a MIDI track
3. Configure API endpoint in device settings (default: `http://localhost:8000`)

---

## Usage

### Basic Generation

1. **Open Max for Live Device** in Ableton Live
2. **Enter Artist Name**: e.g., "J Dilla", "Questlove", "Stewart Copeland"
3. **Set Parameters**:
   - Bars: 1-32
   - Time Signature: 3/4, 4/4, 5/4, 6/8, 7/8
   - Tempo: 40-300 BPM
   - Humanize: On/Off
4. **Click Generate**
5. **Wait** (first-time research: 30-120 seconds, cached: 5-15 seconds)
6. **MIDI clips appear** in session view clip slots

### Advanced Features

**Style Augmentation:**
```python
# Via API or CLI
python scripts/augment_style.py --artist "J Dilla" --add-sources 5
```

**Batch Generation:**
```python
# Generate variations
POST /api/v1/generate-batch
{
  "artist": "J Dilla",
  "variations": 8,
  "bars": 4,
  "tempo": 95
}
```

**Custom Style Profile:**
```python
# Create from scratch
POST /api/v1/profiles
{
  "artist_name": "Custom Style",
  "tempo_range": [120, 130],
  "swing_percentage": 5.0,
  ...
}
```

---

## API Reference

### Generate Pattern
```http
POST /api/v1/generate
Content-Type: application/json

{
  "artist": "J Dilla",
  "bars": 4,
  "time_signature": [4, 4],
  "tempo": 95,
  "humanize": true,
  "llm_provider": "openai"  # optional, auto-selects by default
}

Response:
{
  "task_id": "abc123...",
  "status": "queued",
  "estimated_time": 45
}
```

### Check Task Status
```http
GET /api/v1/tasks/{task_id}

Response:
{
  "status": "complete",
  "result": {
    "midi_path": "/output/patterns/jdilla_4bars_95bpm.mid",
    "style_profile_id": "uuid..."
  }
}
```

### Research Artist
```http
POST /api/v1/research/{artist}

Response:
{
  "task_id": "def456...",
  "status": "researching",
  "progress": 25
}
```

See `docs/API.md` for complete API documentation.

---

## Project Structure

```
MidiDrumiGen/
├── .cursor/rules/          # BMAD-METHOD framework
├── .cursorcontext/         # Updated for v2.0
├── docs/                   # Comprehensive documentation
│   ├── ORCHESTRATOR_META_PROMPT.md
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── UI.md
│   ├── GIT_WORKFLOW.md
│   ├── agents/             # Sub-agent specifications
│   └── _old/               # Archived v1.x files
├── src/
│   ├── api/                # FastAPI routes
│   ├── research/           # Research pipeline
│   │   ├── orchestrator.py
│   │   └── collectors/     # Scholar, MIDI, Text, Audio
│   ├── generation/         # LLM generation
│   │   ├── providers/      # OpenAI, Claude, Gemini
│   │   ├── prompts.py
│   │   └── parser.py
│   ├── midi/               # MIDI processing
│   │   ├── humanizer.py
│   │   ├── validator.py
│   │   └── export.py
│   ├── database/           # SQLAlchemy models
│   │   ├── models.py
│   │   └── vector_ops.py
│   ├── tasks/              # Celery tasks
│   └── utils/              # Shared utilities
├── max_for_live/
│   └── MidiDrumiGen.amxd   # Max for Live device
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/                # CLI tools & Git helpers
│   ├── commit.sh
│   ├── commit.ps1
│   └── augment_style.py
├── requirements.txt        # v2.0 dependencies
├── .env.example
├── docker-compose.yml
└── README.md
```

---

## Development

### BMAD-METHOD Framework

This project uses the **BMAD-METHOD** development framework with:
- Orchestrator-Agent architecture
- Conventional commits
- main → dev → feature branching

See `docs/GIT_WORKFLOW.md` for detailed workflow.

### Quick Commit
```bash
# Using helper script
./scripts/commit.sh

# Select type, enter scope and message
# Auto-pushes to current branch
```

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/unit/test_llm_providers.py -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

---

## Documentation

- **[Product Requirements](docs/PRD.md)** - User stories, features, success metrics
- **[Architecture](docs/ARCHITECTURE.md)** - Technical design, database schema, API spec
- **[UI Design](docs/UI.md)** - Max for Live device specification
- **[Git Workflow](docs/GIT_WORKFLOW.md)** - Branching, commits, PR process
- **[Agent Specifications](docs/agents/)** - Sub-agent detailed designs
- **[Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Central hub

---

## Performance

**Generation Times (Benchmark: AMD Ryzen 9 5950X, 64GB RAM):**
- Cached artist: 5-15 seconds
- New artist research: 30-120 seconds
- Audio analysis (optional): +30-60 seconds

**Costs (per generation):**
- Claude 3.5 Sonnet (Primary): ~$0.01-0.03
- Gemini 2.5/3 (Secondary): ~$0.005-0.015
- ChatGPT 5.1 (Fallback): ~$0.02-0.05

---

## Roadmap

### v2.1 (Q1 2026)
- [ ] Local LLM support (Ollama, LM Studio)
- [ ] Real-time style morphing (interpolate between artists)
- [ ] MIDI export to DAW via MIDI ports (alternative to clip creation)

### v2.2 (Q2 2026)
- [ ] Multi-track generation (drums + bass + chords)
- [ ] Live performance mode (generate on-the-fly)
- [ ] Style transfer between artists

### v3.0 (Future)
- [ ] VST3/AU plugin version
- [ ] Standalone desktop application
- [ ] Community style profile sharing

---

## Breaking Changes from v1.x

**v2.0 is a complete rewrite. Key differences:**

| Feature | v1.x (Training) | v2.0 (LLM) |
|---------|----------------|------------|
| Approach | Pre-trained PyTorch model | On-demand research + LLM |
| Artists | Fixed dataset (4-10) | Unlimited (any artist) |
| Dependencies | PyTorch, TensorFlow | LLM APIs, PostgreSQL |
| Generation Time | 1-2 seconds | 5-120 seconds (cached vs new) |
| Flexibility | Limited to trained styles | Any style, continuously improving |
| Setup Complexity | Requires GPU training | API keys + database |

**No migration path from v1.x.** v2.0 is a fresh start.

---

## Troubleshooting

### "OpenAI API Key Invalid" (or Anthropic/Google)
```bash
# Verify key format (should start with sk-proj- or sk-)
# Test key
python -c "import openai; client = openai.OpenAI(); print(client.models.list())"
```

### Max for Live Device Not Connecting
1. Check API server is running: `curl http://localhost:8000/health`
2. Verify port in Max device settings matches server
3. Check firewall/antivirus isn't blocking connection

See `docs/TROUBLESHOOTING.md` for more issues.

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch from `dev`
3. Follow conventional commit format
4. Write tests for new features
5. Submit PR to `dev` branch

See `docs/GIT_WORKFLOW.md` for detailed process.

---

## License

MIT License - See LICENSE file

---

## Credits

**Built with:**
- Anthropic Claude, Google Gemini, OpenAI ChatGPT
- FastAPI, Celery, PostgreSQL, Redis
- Librosa, madmom, spaCy
- mido (MIDI processing)

**Inspired by:**
- Producer communities on Reddit, Gearspace, VI-Control
- Music information retrieval research
- Ableton's Max for Live ecosystem

---

## Contact & Support

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A
- **Documentation**: `docs/` folder
- **Email**: support@mididrumigen.dev

---

**MidiDrumiGen v2.0** - Making every beat authentic. 🥁
