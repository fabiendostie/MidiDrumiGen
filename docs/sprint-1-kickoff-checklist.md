# Sprint 1 Kickoff Checklist - MidiDrumiGen v2.0

**Sprint:** Sprint 1
**Duration:** 2025-11-18 to 2025-12-01 (2 weeks)
**Stories:** 5 stories (14 story points)
**Team:** Solo Developer (Full-Stack)
**Project:** MidiDrumiGen v2.0 Brownfield Migration

---

## 🎯 Sprint Goals

1. Implement first two research collectors (Scholar Papers + Web Articles)
2. Complete database analytics infrastructure (Vector Search, Redis Caching, Analytics)
3. Establish robust testing foundation with comprehensive test coverage
4. Set up development workflow and CI/CD pipelines

---

## ✅ Pre-Sprint Setup Verification

### Infrastructure Setup

- [x] **PostgreSQL 16 + pgvector** installed and running
  - [x] Container: `postgres-mididrum` running on port 5432
  - [x] Database: `mididrumigen_db` created
  - [x] pgvector extension v0.8.1 enabled
  - [x] 5 tables migrated (artists, style_profiles, research_sources, generation_history, alembic_version)

- [x] **Redis 7.4** installed and running
  - [x] Container: `redis-mididrum` running on port 6379
  - [x] Connection verified with `redis-cli ping`

- [x] **Environment Variables** configured
  - [x] `.env` file exists with API keys
  - [x] `DATABASE_URL` configured
  - [x] `REDIS_URL` configured
  - [x] `ANTHROPIC_API_KEY` set
  - [x] `GOOGLE_API_KEY` set
  - [x] `OPENAI_API_KEY` set

### Python Environment

- [x] **Python 3.11+** installed
  - [x] Virtual environment created: `venv/`
  - [x] Virtual environment activated

- [x] **Dependencies Installed**
  - [x] Core dependencies (FastAPI, SQLAlchemy, Redis, etc.)
  - [x] Sprint 1 dependencies (spaCy, BeautifulSoup4, aiohttp, etc.)
  - [x] spaCy model `en_core_web_sm` downloaded
  - [x] Known exclusions: `madmom` (Windows incompatible), `uvloop` (Linux/macOS only)

### Development Tools

- [ ] **IDE/Editor** configured
  - [ ] Cursor/VS Code with Python extension
  - [ ] `.cursorrules` loaded for AI assistance
  - [ ] `.cursorcontext/` available for context references

- [ ] **Git Workflow** ready
  - [ ] Branch `dev` checked out
  - [ ] Conventional commits helper scripts available
  - [ ] Pre-commit hooks configured (optional)

- [ ] **Testing Framework** ready
  - [ ] pytest installed and configured
  - [ ] `tests/unit/` directory structure verified
  - [ ] Test fixtures in `tests/conftest.py` available

---

## 📋 Sprint 1 Stories Breakdown

### Epic 3: Database & Caching

#### ✅ E3.S1: Database Models (DONE - Phase 2)
**Status:** COMPLETED
**Story Points:** 3

#### ✅ E3.S2: Database Manager (DONE - Phase 2)
**Status:** COMPLETED
**Story Points:** 3

#### ✅ E3.S3: Database Migrations (DONE - Phase 2)
**Status:** COMPLETED
**Story Points:** 2

#### ⏳ E3.S4: Vector Similarity Search
**Status:** TODO
**Story Points:** 3
**Priority:** HIGH

**Files:**
- `src/database/manager.py` - `find_similar_artists()` method
- `src/api/routes/utils.py` - `/api/v1/similar/{artist}` endpoint
- `tests/unit/test_database_similarity.py` - Unit tests

**Acceptance Criteria:**
- [ ] `find_similar_artists()` uses pgvector cosine distance
- [ ] Returns top N similar artists with similarity scores
- [ ] Query completes in < 200ms using IVFFlat index
- [ ] API endpoint returns formatted JSON response
- [ ] Unit tests cover edge cases (no embedding, artist not found)
- [ ] Integration test with real pgvector queries

**Definition of Done:**
- [ ] Code implemented and passes all unit tests
- [ ] API endpoint tested with Postman/curl
- [ ] Documentation updated (docstrings, README)
- [ ] Code reviewed (self-review checklist)
- [ ] Committed with conventional commit format

---

#### ⏳ E3.S5: Redis Caching Layer
**Status:** TODO
**Story Points:** 3
**Priority:** HIGH

**Files:**
- `src/database/manager.py` - `get_style_profile()`, `_cache_profile()`, `_invalidate_cache()`
- `tests/unit/test_database_caching.py` - Unit tests

**Acceptance Criteria:**
- [ ] `get_style_profile()` checks Redis cache first
- [ ] Cache hit returns profile in < 10ms
- [ ] Cache miss queries PostgreSQL and caches result
- [ ] TTL set to 7 days (configurable)
- [ ] Cache invalidation on profile updates
- [ ] Handles Redis connection failures gracefully
- [ ] Unit tests cover cache hit/miss scenarios

**Definition of Done:**
- [ ] Code implemented and passes all unit tests
- [ ] Cache hit/miss logging verified
- [ ] Performance benchmarks documented
- [ ] Redis connection pooling tested
- [ ] Code reviewed and committed

---

#### ⏳ E3.S6: Generation History Analytics
**Status:** TODO
**Story Points:** 2
**Priority:** MEDIUM

**Files:**
- `src/database/manager.py` - `get_generation_stats()`, `get_artist_generation_history()`
- `src/api/routes/utils.py` - `/api/v1/stats`, `/api/v1/artists/{artist}/history`
- `tests/unit/test_database_analytics.py` - Unit tests

**Acceptance Criteria:**
- [ ] `get_generation_stats()` aggregates total generations, avg time, total cost, provider usage
- [ ] `get_artist_generation_history()` returns paginated history
- [ ] API endpoints return formatted JSON
- [ ] Statistics query completes in < 500ms
- [ ] Unit tests cover empty database and data aggregation

**Definition of Done:**
- [ ] Code implemented and passes all unit tests
- [ ] API endpoints tested with sample data
- [ ] Documentation includes example responses
- [ ] Code reviewed and committed

---

### Epic 1: Research Pipeline

#### ⏳ E1.S1: Scholar Paper Collection
**Status:** TODO
**Story Points:** 2
**Priority:** MEDIUM

**Files:**
- `src/research/collectors/base.py` - Abstract base classes
- `src/research/collectors/papers.py` - `ScholarPaperCollector`
- `tests/unit/test_collector_papers.py` - Unit tests

**Acceptance Criteria:**
- [ ] Searches Semantic Scholar, arXiv, CrossRef APIs
- [ ] Extracts tempo mentions from abstracts
- [ ] Calculates confidence based on citations
- [ ] Parallel async API calls
- [ ] Handles API errors gracefully (404, timeouts, rate limits)
- [ ] Unit tests with mocked API responses

**Definition of Done:**
- [ ] Code implemented and passes all unit tests
- [ ] Integration test with real Semantic Scholar API
- [ ] API rate limiting tested
- [ ] Documentation includes API usage examples
- [ ] Code reviewed and committed

---

#### ⏳ E1.S2: Web Article Collection
**Status:** TODO
**Story Points:** 3
**Priority:** HIGH

**Files:**
- `src/research/collectors/articles.py` - `WebArticleCollector`
- `tests/unit/test_collector_articles.py` - Unit tests

**Acceptance Criteria:**
- [ ] Scrapes 4+ sites (Drummerworld, Wikipedia, Pitchfork, Rolling Stone)
- [ ] Uses spaCy NLP to filter drumming-related content
- [ ] Extracts equipment and technique mentions
- [ ] Site-specific HTML parsing strategies
- [ ] Respects robots.txt
- [ ] Unit tests with mocked HTML responses

**Definition of Done:**
- [ ] Code implemented and passes all unit tests
- [ ] Integration test with real website scraping
- [ ] robots.txt compliance verified
- [ ] Documentation includes site configurations
- [ ] Code reviewed and committed

---

## 🛠️ Development Guidelines

### Code Standards

**Python Style:**
- PEP 8 with Black formatter (line length: 100)
- Type hints for all function signatures
- Comprehensive docstrings (Google style)
- Async/await for ALL I/O operations

**Import Order:**
```python
# 1. Standard library
import os
from typing import List, Dict

# 2. Third-party packages
from fastapi import FastAPI
import aiohttp

# 3. Local imports
from src.database.manager import DatabaseManager
```

**Error Handling:**
```python
# Use specific exceptions
class ResearchError(Exception):
    """Raised when research pipeline fails."""
    pass

# Log errors with context
logger.error(f"Failed to collect papers: {e}", exc_info=True)
```

### Testing Requirements

**Unit Test Structure:**
```python
# Use fixtures for reusable test data
@pytest.fixture
def collector():
    return ScholarPaperCollector(timeout=60)

# Organize tests by feature
class TestTempoExtraction:
    def test_extract_single_bpm(self, collector):
        # Test implementation
        pass

# Mock external dependencies
@patch('aiohttp.ClientSession')
async def test_api_call(mock_session):
    # Test implementation
    pass
```

**Test Coverage Goals:**
- Unit tests: > 90% coverage
- Integration tests for critical paths
- Mark slow tests: `@pytest.mark.slow`

### Git Workflow

**Conventional Commits:**
```bash
# Format: <type>(<scope>): <subject>
git commit -m "feat(research): implement scholar paper collector"
git commit -m "test(database): add vector similarity tests"
git commit -m "fix(caching): handle Redis connection errors"
git commit -m "docs(api): update endpoint documentation"
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `test`: Add/update tests
- `docs`: Documentation changes
- `refactor`: Code restructuring
- `chore`: Build/config changes

**Branch Strategy:**
```
main (production)
  └── dev (integration) ← [YOU ARE HERE]
       ├── feature/E3.S4-vector-similarity
       ├── feature/E3.S5-redis-caching
       ├── feature/E3.S6-analytics
       ├── feature/E1.S1-scholar-papers
       └── feature/E1.S2-web-articles
```

**Commit Frequency:**
- Commit every 1-2 hours of development
- Push after each commit
- Small, atomic commits preferred

---

## 📅 Daily Workflow

### Development Cycle

**Morning (30 min):**
1. Review sprint status: `docs/sprint-status.yaml`
2. Select next story from TODO
3. Review acceptance criteria
4. Update todo list with story tasks

**Development (4-6 hours):**
1. Create feature branch: `git checkout -b feature/E3.S4-vector-similarity`
2. Implement functionality (Red → Green → Refactor)
3. Write unit tests alongside code
4. Run tests frequently: `pytest tests/unit/test_*.py -v`
5. Commit regularly with conventional commits

**Testing (1-2 hours):**
1. Run full test suite: `pytest tests/unit/ -v`
2. Check coverage: `pytest --cov=src --cov-report=html`
3. Fix failing tests
4. Run integration tests (if applicable)

**Code Review (30 min):**
1. Self-review checklist (see below)
2. Check code against Definition of Done
3. Update documentation (docstrings, README)
4. Merge feature branch to dev

**End of Day (15 min):**
1. Update `docs/sprint-status.yaml` with progress
2. Push all commits
3. Note any blockers or questions

---

## ✅ Definition of Done (Per Story)

### Code Quality
- [ ] All acceptance criteria met
- [ ] Code follows PEP 8 and project conventions
- [ ] Type hints on all public functions
- [ ] Comprehensive docstrings (Google style)
- [ ] No `# TODO` or `# FIXME` comments left
- [ ] No hardcoded values (use config/env vars)

### Testing
- [ ] Unit tests written and passing
- [ ] Test coverage > 90% for new code
- [ ] Edge cases covered (empty input, errors, timeouts)
- [ ] Integration tests for external dependencies
- [ ] Slow tests marked with `@pytest.mark.slow`

### Documentation
- [ ] Docstrings updated for all public APIs
- [ ] README updated with new features/endpoints
- [ ] API endpoints documented with example responses
- [ ] Architecture diagrams updated (if applicable)

### Code Review
- [ ] Self-review completed (checklist below)
- [ ] No linting errors: `ruff check src/`
- [ ] Code formatted: `black src/`
- [ ] Type checking passes: `mypy src/` (if enabled)

### Git
- [ ] Committed with conventional commit format
- [ ] Pushed to remote repository
- [ ] Feature branch merged to dev

---

## 🔍 Self-Review Checklist

**Before Committing:**
- [ ] Did I test this code manually?
- [ ] Are there any edge cases I missed?
- [ ] Is error handling comprehensive?
- [ ] Are all magic numbers/strings extracted to constants?
- [ ] Could this code be simplified?
- [ ] Did I add logging for debugging?
- [ ] Is async/await used correctly?

**Code Smells to Avoid:**
- [ ] No duplicate code (DRY principle)
- [ ] No functions > 50 lines
- [ ] No deeply nested conditionals (> 3 levels)
- [ ] No unused imports or variables
- [ ] No commented-out code

---

## 🎪 Sprint Ceremonies

### Sprint Planning (Completed)
**Date:** 2025-11-18
**Outcome:** 5 stories committed (14 story points)

### Daily Standup (Solo - Self-Check)
**Time:** Daily at 9:00 AM
**Questions:**
1. What did I accomplish yesterday?
2. What am I working on today?
3. Are there any blockers?

**Format:** Quick note in sprint status YAML

### Sprint Review (End of Sprint)
**Date:** 2025-12-01
**Goals:**
- Demo all completed stories
- Review acceptance criteria fulfillment
- Update documentation
- Prepare for Sprint 2 planning

### Sprint Retrospective (End of Sprint)
**Date:** 2025-12-01
**Questions:**
- What went well?
- What could be improved?
- What actions will I take in Sprint 2?

---

## 🚨 Risk Assessment & Mitigation

### Known Risks

**Risk 1: External API Rate Limits**
- **Impact:** Research collectors may fail
- **Probability:** MEDIUM
- **Mitigation:**
  - Implement exponential backoff retry
  - Cache API responses
  - Use multiple API endpoints (fallback)
  - Test with real APIs early

**Risk 2: Windows Compatibility Issues**
- **Impact:** Some Python packages may fail
- **Probability:** LOW (already addressed)
- **Mitigation:**
  - Exclude incompatible packages (`madmom`, `uvloop`)
  - Use cross-platform alternatives (`librosa`)
  - Test on Windows environment regularly

**Risk 3: PostgreSQL Vector Search Performance**
- **Impact:** Similarity queries may be slow
- **Probability:** LOW
- **Mitigation:**
  - Use IVFFlat index (already configured)
  - Benchmark query performance
  - Optimize embedding dimensions if needed
  - Consider HNSW index for production

**Risk 4: Redis Connection Failures**
- **Impact:** Caching layer may degrade
- **Probability:** LOW
- **Mitigation:**
  - Graceful degradation (fall back to PostgreSQL)
  - Connection pooling and retry logic
  - Monitor Redis health in production

### Blockers

**Current Blockers:**
- None

**Potential Blockers:**
- API keys not working (verify early)
- Semantic Scholar API changes (check documentation)
- Website structure changes (test scraping regularly)

---

## 📊 Sprint Tracking

### Progress Tracking
- **File:** `docs/sprint-status.yaml`
- **Update Frequency:** End of each day
- **Metrics Tracked:**
  - Stories completed
  - Story points burned down
  - Velocity (actual vs. planned)
  - Blockers and risks

### Velocity Calculation
- **Planned Velocity:** 14 story points / 2 weeks = 7 SP/week
- **Daily Target:** 1-2 story points/day
- **Adjustment:** Monitor and adjust in Sprint 2

---

## 🎉 Sprint 1 Success Criteria

Sprint 1 is successful if:
- [ ] All 5 stories meet Definition of Done
- [ ] Vector similarity search working with < 200ms queries
- [ ] Redis caching provides < 10ms cache hits
- [ ] Analytics endpoints return accurate statistics
- [ ] Scholar paper collector successfully queries 3 APIs
- [ ] Web article collector scrapes 4+ websites
- [ ] Test coverage > 90% for all new code
- [ ] All code committed and pushed to `dev` branch
- [ ] Documentation updated and comprehensive
- [ ] Sprint 2 planning ready with lessons learned

---

## 📚 Resources & References

### Documentation
- `docs/epics.md` - All 40 user stories
- `docs/sprint-status.yaml` - Real-time sprint tracking
- `.cursorcontext/` - Comprehensive project context
- `CLAUDE.md` - Project overview and conventions

### API Documentation
- Semantic Scholar API: https://api.semanticscholar.org/
- arXiv API: https://arxiv.org/help/api/
- CrossRef API: https://api.crossref.org/

### Libraries
- pgvector: https://github.com/pgvector/pgvector
- spaCy: https://spacy.io/usage
- BeautifulSoup4: https://www.crummy.com/software/BeautifulSoup/bs4/doc/

### Testing
- pytest async: https://pytest-asyncio.readthedocs.io/
- pytest-mock: https://pytest-mock.readthedocs.io/

---

## ✨ Next Steps After Kickoff

1. **Select First Story:** E3.S4 (Vector Similarity Search) or E1.S2 (Web Articles)
2. **Create Feature Branch:** `git checkout -b feature/E3.S4-vector-similarity`
3. **Review Stub Files:** Check `src/database/manager.py` and test templates
4. **Start Implementation:** Follow TDD (test → implement → refactor)
5. **Commit Regularly:** Every 1-2 hours with conventional commits
6. **Update Sprint Status:** End of each day in `docs/sprint-status.yaml`

---

**Sprint 1 Start Date:** 2025-11-18
**Sprint 1 End Date:** 2025-12-01
**Let's build something amazing! 🚀**
