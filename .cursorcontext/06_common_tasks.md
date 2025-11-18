# Common Tasks - MidiDrumiGen v2.0

**Document Type:** Instructional Context (Task-Specific Guidance)  
**Purpose:** Quick reference for frequent development tasks in v2.0  
**Use Case:** Load when performing specific operations  
**Last Updated:** 2025-11-17

---

## Quick Task Reference

### Generate Pattern from CLI

```bash
# Basic generation (artist name)
python scripts/generate_pattern.py \
  --artist "J Dilla" \
  --bars 4 \
  --tempo 95 \
  --output "output/patterns/jdilla_001.mid"

# With specific LLM provider
python scripts/generate_pattern.py \
  --artist "Metro Boomin" \
  --bars 8 \
  --tempo 140 \
  --provider "claude" \
  --output "output/patterns/metro_001.mid"

# Without humanization
python scripts/generate_pattern.py \
  --artist "Questlove" \
  --bars 4 \
  --tempo 120 \
  --no-humanize \
  --output "output/patterns/questlove_001.mid"

# Batch generation with variations
python scripts/generate_batch.py \
  --artist "J Dilla" \
  --variations 10 \
  --bars 4 \
  --tempo 95 \
  --output-dir "output/patterns/jdilla_batch/"
```

### Research Artist Style

```bash
# Initial research (full pipeline)
python scripts/research_artist.py \
  --artist "J Dilla" \
  --sources all \
  --min-total 10

# Research with specific source types
python scripts/research_artist.py \
  --artist "Questlove" \
  --sources scholarly,audio,midi \
  --timeout 30

# Augment existing style profile
python scripts/augment_style.py \
  --artist "J Dilla" \
  --add-sources 5 \
  --source-type audio

# View style profile
python scripts/view_style_profile.py \
  --artist "J Dilla" \
  --format json
```

### Run API Server

**Windows/Linux/macOS:**
```bash
# Development mode (auto-reload)
# Make sure virtual environment is activated and .env is configured
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode with multiple workers
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With HTTPS (Linux/macOS)
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
```

**Note:** Ensure PostgreSQL and Redis are running before starting the API server.

### Start Celery Workers

**Windows/Linux/macOS:**
```bash
# Make sure virtual environment is activated and Redis is running

# Research worker (handles collector tasks)
celery -A src.tasks.worker worker -Q research -c 4 --loglevel=info

# Generation worker (handles LLM generation)
celery -A src.tasks.worker worker -Q generation -c 2 --loglevel=info

# MIDI processing worker
celery -A src.tasks.worker worker -Q midi -c 4 --loglevel=info

# Monitor with Flower
celery -A src.tasks.worker flower --port=5555
```

**Note:** Run these in separate terminal windows/tabs. In Cursor IDE, you can open multiple integrated terminals.

### Database Management

```bash
# Create new migration
alembic revision --autogenerate -m "Add StyleProfile table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Check current version
alembic current

# View migration history
alembic history
```

### Test LLM Providers

```bash
# Test OpenAI
python scripts/test_llm_provider.py --provider openai --prompt "test"

# Test Anthropic Claude
python scripts/test_llm_provider.py --provider anthropic --prompt "test"

# Test Google Gemini
python scripts/test_llm_provider.py --provider google --prompt "test"

# Test fallback logic
python scripts/test_llm_fallback.py
```

---

## Development Workflows

### Adding New Artist Style Profile

**1. Research Artist (Automated)**
```bash
# Run research pipeline
python scripts/research_artist.py \
  --artist "New Artist" \
  --sources all \
  --min-total 12

# This will:
# - Query scholarly databases
# - Search MIDI repositories
# - Scrape web articles
# - Download and analyze audio samples
# - Create StyleProfile in database
```

**2. Verify Style Profile**
```python
from src.database.models import StyleProfile
from src.database.session import get_db

async with get_db() as db:
    profile = await db.get_profile_by_artist("New Artist")
    print(f"Tempo Range: {profile.tempo_range}")
    print(f"Swing: {profile.swing_percentage}%")
    print(f"Sources: {profile.source_count}")
```

**3. Generate Test Patterns**
```bash
# Generate several test patterns
python scripts/generate_batch.py \
  --artist "New Artist" \
  --variations 5 \
  --bars 4 \
  --output-dir "output/test_patterns/new_artist/"
```

**4. Manually Review & Augment**
```bash
# If quality is poor, augment with more sources
python scripts/augment_style.py \
  --artist "New Artist" \
  --add-sources 10 \
  --source-type "audio,midi"
```

### Debugging Generation Issues

**Problem: Research taking too long**

```bash
# Check collector status
python scripts/check_collectors.py --artist "Artist Name"

# View research logs
tail -f logs/research.log

# Skip slow collectors (audio downloading/processing can be slow)
python scripts/research_artist.py \
  --artist "Artist Name" \
  --sources scholarly,text \
  --skip-audio  # Audio analysis takes 30-60s per track

# Or limit audio samples
python scripts/research_artist.py \
  --artist "Artist Name" \
  --max-audio-samples 3  # Default is 5-10
```

**Problem: LLM generation fails**

```python
# 1. Check API keys
import os
from dotenv import load_dotenv
load_dotenv()

print(f"OpenAI: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
print(f"Anthropic: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")
print(f"Google: {'✓' if os.getenv('GOOGLE_API_KEY') else '✗'}")

# 2. Test API connectivity
from openai import OpenAI
client = OpenAI()
models = client.models.list()
print(f"OpenAI connection: ✓")

# 3. Check prompt length
from src.generation.prompts import build_generation_prompt
profile = await db.get_profile_by_artist("J Dilla")
prompt = build_generation_prompt(profile, params)
print(f"Prompt length: {len(prompt)} chars")
print(f"Est. tokens: {len(prompt) // 4}")

# 4. Test with simpler prompt
from src.generation.providers import LLMProviderManager
manager = LLMProviderManager(config)
result = await manager.generate("Generate a simple 2-bar drum pattern", provider="openai")
```

**Problem: Database connection issues**

```bash
# Check PostgreSQL status
pg_isready -h localhost -p 5432

# Test connection
psql -d mididrumigen_db -c "SELECT version();"

# Check pgvector extension
psql -d mididrumigen_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Reset database (CAUTION: deletes all data)
alembic downgrade base
alembic upgrade head
```

**Problem: MIDI quality is poor**

```bash
# 1. Check style profile quality
python scripts/analyze_style_profile.py --artist "Artist Name"

# 2. Augment profile with more sources
python scripts/augment_style.py \
  --artist "Artist Name" \
  --add-sources 10

# 3. Adjust LLM parameters
python scripts/generate_pattern.py \
  --artist "Artist Name" \
  --temperature 0.7 \
  --max-tokens 3000

# 4. Try different LLM provider
python scripts/generate_pattern.py \
  --artist "Artist Name" \
  --provider claude  # vs openai
```

**Problem: Redis connection errors**

```bash
# Check Redis status
redis-cli ping

# If not running, start Redis
# Docker:
docker run -d --name redis-mididrum -p 6379:6379 redis:7.4-alpine

# macOS:
brew services start redis

# Linux:
sudo systemctl start redis

# Test connection
python -c "import redis; r = redis.from_url('redis://localhost:6379'); print(r.ping())"
```

---

## Testing Checklist

### Before Committing Code

```bash
# 1. Run unit tests
pytest tests/unit/ -v

# 2. Check test coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report

# 3. Lint code
ruff check src/ tests/

# 4. Format code
black src/ tests/

# 5. Type checking
mypy src/

# 6. Fix linting issues
ruff check --fix src/ tests/
```

### Before Deploying

```bash
# 1. Run integration tests
pytest tests/integration/ -v

# 2. Test API endpoints
python scripts/test_api_endpoints.py --base-url http://localhost:8000

# 3. Test research pipeline
python scripts/test_research_pipeline.py --artist "Test Artist"

# 4. Test LLM providers
python scripts/test_llm_providers.py

# 5. Test database operations
pytest tests/integration/test_database.py -v

# 6. Test Celery workers
python scripts/test_celery_tasks.py

# 7. Load test (optional)
locust -f tests/load/locustfile.py --host http://localhost:8000

# 8. Check environment variables
python scripts/validate_env.py
```

---

## Useful Code Snippets

### Research Artist Programmatically

```python
from src.research.orchestrator import ResearchOrchestrator
from src.database.session import get_db

async def research_artist(artist_name: str):
    """Research an artist and create StyleProfile."""
    
    orchestrator = ResearchOrchestrator(config)
    
    # Execute research
    profile = await orchestrator.research_artist(artist_name)
    
    print(f"Artist: {profile.artist_name}")
    print(f"Tempo Range: {profile.tempo_range}")
    print(f"Swing: {profile.swing_percentage}%")
    print(f"Sources: {profile.source_count}")
    
    return profile

# Usage
import asyncio
profile = asyncio.run(research_artist("J Dilla"))
```

### Generate Pattern Programmatically

```python
from src.generation.providers import LLMProviderManager
from src.generation.prompts import build_generation_prompt
from src.midi.export import export_midi
from src.database.session import get_db

async def generate_pattern(artist_name: str, bars: int = 4, tempo: int = 120):
    """Generate drum pattern using LLM."""
    
    # Get style profile
    async with get_db() as db:
        profile = await db.get_profile_by_artist(artist_name)
    
    # Build prompt
    params = GenerationParams(bars=bars, tempo=tempo, time_signature=(4, 4))
    prompt = build_generation_prompt(profile, params)
    
    # Generate with LLM
    manager = LLMProviderManager(config)
    result = await manager.generate(prompt, provider="openai", fallback=True)
    
    # Parse MIDI events
    midi_events = parse_llm_response(result)
    
    # Export to MIDI
    midi_path = export_midi(
        events=midi_events,
        tempo=tempo,
        time_signature=(4, 4),
        output_path=f"output/{artist_name}_{bars}bars.mid",
        humanize=True
    )
    
    print(f"Pattern saved to: {midi_path}")
    return midi_path

# Usage
import asyncio
midi_path = asyncio.run(generate_pattern("J Dilla", bars=4, tempo=95))
```

### Query Similar Artists (Vector Search)

```python
from sqlalchemy import select
from src.database.models import StyleProfile
from src.database.session import get_db
from sentence_transformers import SentenceTransformer

async def find_similar_artists(artist_name: str, limit: int = 5):
    """Find similar artists using vector similarity."""
    
    async with get_db() as db:
        # Get target artist profile
        target = await db.get_profile_by_artist(artist_name)
        
        # Find similar using pgvector
        query = select(StyleProfile).order_by(
            StyleProfile.embedding.cosine_distance(target.embedding)
        ).limit(limit + 1)  # +1 to exclude self
        
        results = await db.execute(query)
        similar = results.scalars().all()
        
        # Exclude the artist itself
        similar = [s for s in similar if s.artist_name != artist_name][:limit]
        
        print(f"Artists similar to {artist_name}:")
        for i, artist in enumerate(similar, 1):
            print(f"{i}. {artist.artist_name}")
            print(f"   Tempo: {artist.tempo_range}")
            print(f"   Swing: {artist.swing_percentage}%\n")
        
        return similar

# Usage
import asyncio
similar = asyncio.run(find_similar_artists("J Dilla", limit=5))
```

### Analyze MIDI File

```python
import mido
from collections import Counter

def analyze_midi(midi_path: str):
    """Analyze MIDI file statistics."""
    
    mid = mido.MidiFile(midi_path)
    
    # Get tempo
    tempo = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = mido.tempo2bpm(msg.tempo)
                break
    
    # Count notes
    notes = []
    velocities = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
                velocities.append(msg.velocity)
    
    # Calculate statistics
    note_counts = Counter(notes)
    avg_velocity = sum(velocities) / len(velocities) if velocities else 0
    
    # Pattern length
    total_ticks = sum(msg.time for track in mid.tracks for msg in track)
    beats = total_ticks / mid.ticks_per_beat
    bars = beats / 4  # Assuming 4/4
    
    print(f"MIDI Analysis: {midi_path}")
    print(f"Tempo: {tempo} BPM")
    print(f"Length: {bars:.1f} bars")
    print(f"Total notes: {len(notes)}")
    print(f"Avg velocity: {avg_velocity:.1f}")
    print(f"\nMost common drums:")
    for note, count in note_counts.most_common(5):
        drum_name = {36: 'Kick', 38: 'Snare', 42: 'Hi-Hat', 46: 'Open Hi-Hat'}.get(note, f'Note {note}')
        print(f"  {drum_name}: {count} hits")

# Usage
analyze_midi("output/patterns/jdilla_001.mid")
```

### Quick API Test

```python
import requests
import time

def test_generation_api(artist: str, bars: int = 4, tempo: int = 120):
    """Test pattern generation via API."""
    
    base_url = "http://localhost:8000"
    
    # 1. Check health
    health = requests.get(f"{base_url}/health")
    print(f"API Health: {health.json()}")
    
    # 2. Request generation
    response = requests.post(
        f"{base_url}/api/v1/generate",
        json={
            "artist": artist,
            "bars": bars,
            "tempo": tempo,
            "time_signature": [4, 4],
            "humanize": True
        }
    )
    
    task_id = response.json()["task_id"]
    print(f"Task ID: {task_id}")
    
    # 3. Poll status
    max_attempts = 120  # 2 minutes max
    for attempt in range(max_attempts):
        status_response = requests.get(f"{base_url}/api/v1/tasks/{task_id}")
        status_data = status_response.json()
        status = status_data["status"]
        
        print(f"[{attempt}s] Status: {status}")
        
        if status == "complete":
            midi_path = status_data["result"]["midi_path"]
            print(f"✓ MIDI saved to: {midi_path}")
            return midi_path
        elif status == "failed":
            error = status_data.get("error", "Unknown error")
            print(f"✗ Generation failed: {error}")
            return None
        
        time.sleep(1)
    
    print("✗ Timeout waiting for generation")
    return None

# Usage
test_generation_api("J Dilla", bars=4, tempo=95)
```

### Batch Export Style Profiles

```python
from src.database.session import get_db
from src.database.models import StyleProfile
import json

async def export_all_profiles(output_path: str = "style_profiles_export.json"):
    """Export all style profiles to JSON."""
    
    async with get_db() as db:
        profiles = await db.query(StyleProfile).all()
        
        export_data = []
        for profile in profiles:
            export_data.append({
                "artist_name": profile.artist_name,
                "tempo_range": profile.tempo_range,
                "swing_percentage": profile.swing_percentage,
                "velocity_variation": profile.velocity_variation,
                "ghost_note_probability": profile.ghost_note_probability,
                "rhythmic_complexity": profile.rhythmic_complexity,
                "source_count": profile.source_count,
                "textual_description": profile.textual_description[:200],  # Truncate
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Exported {len(export_data)} profiles to {output_path}")

# Usage
import asyncio
asyncio.run(export_all_profiles())
```

---

## Environment Setup

### Create Development Environment

**Windows (PowerShell):**
```powershell
# 1. Navigate to project directory
cd C:\Users\lefab\Documents\Dev\MidiDrumiGen

# 2. Create virtual environment (using Python 3.11 or 3.12)
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1
# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download spaCy model
python -m spacy download en_core_web_sm

# 6. Setup PostgreSQL (Docker recommended)
docker run -d --name postgres-mididrum `
  -e POSTGRES_USER=mididrumigen `
  -e POSTGRES_PASSWORD=changeme `
  -e POSTGRES_DB=mididrumigen_db `
  -p 5432:5432 `
  pgvector/pgvector:pg16

# 7. Setup Redis
docker run -d --name redis-mididrum -p 6379:6379 redis:7.4-alpine

# 8. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 9. Run migrations
alembic upgrade head

# 10. Verify installation
python scripts/verify_installation.py
```

**Linux/macOS:**
```bash
# 1. Navigate to project directory
cd ~/Documents/Dev/MidiDrumiGen

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Setup PostgreSQL (Docker)
docker run -d --name postgres-mididrum \
  -e POSTGRES_USER=mididrumigen \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=mididrumigen_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 6. Setup Redis
docker run -d --name redis-mididrum -p 6379:6379 redis:7.4-alpine

# 7. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 8. Run migrations
alembic upgrade head

# 9. Verify installation
python scripts/verify_installation.py
```

### Docker Compose Setup (Recommended)

```bash
# Build images
docker-compose build

# Start all services (API, Celery, PostgreSQL, Redis)
docker-compose up -d

# View logs
docker-compose logs -f api

# Run migrations
docker-compose exec api alembic upgrade head

# Run tests
docker-compose exec api pytest tests/

# Stop services
docker-compose down

# Stop and remove volumes (CAUTION: deletes data)
docker-compose down -v
```

---

## Monitoring and Debugging

### Check System Status

```bash
# API health
curl http://localhost:8000/health

# PostgreSQL status
docker exec postgres-mididrum pg_isready

# Redis status
docker exec redis-mididrum redis-cli ping

# Celery workers
celery -A src.tasks.worker inspect active
celery -A src.tasks.worker inspect stats

# Database connection
psql postgresql://mididrumigen:changeme@localhost:5432/mididrumigen_db -c "SELECT COUNT(*) FROM style_profiles;"
```

### View Logs

```bash
# FastAPI logs
tail -f logs/api.log

# Celery logs
tail -f logs/celery.log

# Research pipeline logs
tail -f logs/research.log

# LLM provider logs
tail -f logs/llm.log

# Filter errors only
tail -f logs/app.log | grep ERROR

# View last 100 lines
tail -n 100 logs/app.log
```

### Monitor Resource Usage

```bash
# Docker container stats
docker stats

# PostgreSQL activity
psql -d mididrumigen_db -c "SELECT * FROM pg_stat_activity;"

# Redis memory
redis-cli info memory

# Celery task queue length
redis-cli llen celery
```

---

## Troubleshooting

### Common Errors and Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'openai'` | Activate virtual environment: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Linux/macOS) |
| `Connection refused (PostgreSQL)` | Start PostgreSQL: `docker start postgres-mididrum` |
| `Connection refused (Redis)` | Start Redis: `docker start redis-mididrum` |
| `OpenAI API Key Invalid` | Check `.env` file, verify key at https://platform.openai.com/api-keys |
| `pgvector extension not found` | Use `pgvector/pgvector:pg16` Docker image, or install extension manually |
| `spaCy model not found` | Download model: `python -m spacy download en_core_web_sm` |
| `Research timeout` | Increase `RESEARCH_TIMEOUT_MINUTES` in `.env` or skip slow collectors |
| `LLM rate limit exceeded` | Wait or switch to fallback provider in `.env` |
| `MIDI file empty` | Check LLM response parsing, try different provider or temperature |
| `Permission denied` (file access) | Check file permissions: `chmod 644 file.mid` (Linux/macOS) |
| `Slow generation` | Check cache, use primary LLM provider (OpenAI), optimize prompt |

---

## Git Workflow (BMAD-METHOD)

### Quick Commit

```bash
# Windows
.\scripts\commit.ps1

# Linux/macOS
./scripts/commit.sh
```

### Conventional Commit Examples

```bash
# Feature
git commit -m "feat(research): add audio analysis collector"

# Fix
git commit -m "fix(llm): resolve timeout in Claude provider"

# Documentation
git commit -m "docs: update API endpoint documentation"

# Test
git commit -m "test(database): add vector search integration tests"
```

See `docs/GIT_WORKFLOW.md` for complete workflow guide.

---

## Related Documents

- **Project Overview**: `.cursorcontext/01_project_overview.md` - v2.0 overview
- **Architecture**: `.cursorcontext/02_architecture.md` - Orchestrator-Agent design
- **Dependencies**: `.cursorcontext/03_dependencies.md` - All dependencies (verified 2025-11-17)
- **MIDI Operations**: `.cursorcontext/04_midi_operations.md` - MIDI processing
- **Generation Pipeline**: `.cursorcontext/05_generation_pipeline.md` - LLM generation flow
- **Orchestrator Meta Prompt**: `docs/ORCHESTRATOR_META_PROMPT.md` - System blueprint
- **PRD**: `docs/PRD.md` - Product requirements
- **Architecture**: `docs/ARCHITECTURE.md` - Technical architecture
- **Git Workflow**: `docs/GIT_WORKFLOW.md` - Branching, commits, PR process
