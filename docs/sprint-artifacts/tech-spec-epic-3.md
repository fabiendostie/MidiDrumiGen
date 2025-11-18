# Epic Technical Specification: Database & Caching

Date: 2025-11-18
Author: Fabz
Epic ID: epic-3
Status: Draft

---

## Overview

Epic 3 delivers the foundational database and caching infrastructure that enables MidiDrumiGen v2.0 to achieve sub-2-minute generation times for cached artists. This epic implements a PostgreSQL + pgvector database layer for persistent storage of artist research data (StyleProfiles), combined with a Redis caching layer for ultra-fast retrieval of frequently accessed profiles.

The epic builds upon the Research Pipeline (Epic 1) and LLM Generation Engine (Epic 2) by providing the critical data persistence layer that transforms first-time research (5-20 minutes) into near-instant retrieval (< 100ms) for subsequent requests. Additionally, it enables vector similarity search to power the "similar artists" discovery feature, allowing users to explore related drumming styles based on embedding similarity.

## Objectives and Scope

**In Scope:**
- SQLAlchemy ORM models for Artists, StyleProfiles, ResearchSources, GenerationHistory
- DatabaseManager class with async CRUD operations
- Alembic migrations for schema versioning (already completed in Phase 2)
- PostgreSQL + pgvector integration for vector similarity search
- Redis caching layer with 7-day TTL for StyleProfiles
- Vector similarity search API endpoint (`GET /api/v1/similar/{artist}`)
- Generation history tracking with provider performance analytics
- Analytics API endpoint (`GET /api/v1/stats`)

**Out of Scope:**
- Database sharding (deferred to Phase 3 scaling)
- Read replicas for high-traffic scenarios (future)
- Advanced analytics dashboard UI (future Phase 2)
- Multi-region database replication (future)
- Cost optimization automation (manual analysis only in v2.0)

## System Architecture Alignment

This epic implements the **Data Layer** components shown in the system architecture (ARCHITECTURE.md, lines 99-109):

**Architectural Components:**
- **PostgreSQL + pgvector:** Primary persistence store for Artists, StyleProfiles, ResearchSources, GenerationHistory
- **Redis:** High-speed cache for frequently accessed StyleProfiles, task queue for Celery (already configured)
- **DatabaseManager:** Abstraction layer providing async CRUD operations to upstream services

**Integration Points:**
- **Research Orchestrator** (Epic 1) → DatabaseManager: Stores StyleProfiles after research completion
- **LLM Provider Manager** (Epic 2) → DatabaseManager: Retrieves cached StyleProfiles, logs generation history
- **Main Orchestrator** → DatabaseManager: Cache hit/miss logic, similarity search queries
- **FastAPI Routes** → DatabaseManager: Exposes analytics and similarity search endpoints

**Constraints from Architecture:**
- Query performance: < 100ms for profile retrieval (NFR-1, line 380)
- Vector index: IVFFlat with cosine distance (Schema, line 1098)
- Connection pooling: 10 base + 20 overflow (Performance, line 1549-1554)

## Detailed Design

### Services and Modules

| Module/Service | Responsibility | Inputs | Outputs | Owner/File |
|----------------|----------------|--------|---------|------------|
| **DatabaseManager** | Primary database abstraction layer providing async CRUD operations | Artist names, StyleProfile objects, query parameters | Persisted entities, query results | `src/database/manager.py` |
| **SQLAlchemy Models** | ORM models defining database schema | N/A (declarative) | Mapped Python classes | `src/database/models.py` |
| **Alembic Migrations** | Version-controlled schema migrations | Schema changes | Database DDL operations | `alembic/versions/a86af6571aac_*.py` |
| **Redis Cache Layer** | High-speed cache for StyleProfiles | Artist names, StyleProfile objects | Cached profiles, cache stats | Embedded in DatabaseManager |
| **Analytics Service** | Generation history aggregation and reporting | GenerationHistory records | Usage statistics, cost metrics | `src/database/analytics.py` (new) |
| **Vector Similarity** | pgvector-based similarity search | Artist name, limit | List of similar artists with scores | DatabaseManager method |

### Data Models and Contracts

**Artist Model** (`src/database/models.py`)
```python
class Artist(Base):
    __tablename__ = 'artists'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False, index=True)
    research_status = Column(String(50), default='pending')  # pending, researching, cached, failed
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    sources_count = Column(Integer, default=0)
    confidence_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    style_profile = relationship("StyleProfile", back_populates="artist", uselist=False)
    research_sources = relationship("ResearchSource", back_populates="artist", cascade="all, delete-orphan")
    generation_history = relationship("GenerationHistory", back_populates="artist")
```

**StyleProfile Model** (`src/database/models.py`)
```python
from pgvector.sqlalchemy import Vector

class StyleProfile(Base):
    __tablename__ = 'style_profiles'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artist_id = Column(UUID(as_uuid=True), ForeignKey('artists.id'), unique=True, nullable=False)

    text_description = Column(Text, nullable=False)
    quantitative_params = Column(JSON, nullable=False)
    # JSON structure: {tempo_min, tempo_max, tempo_avg, swing_percent, ghost_note_prob, syncopation_level, velocity_mean, velocity_std}

    midi_templates_json = Column(JSON)  # Array of file paths
    embedding = Column(Vector(384))  # sentence-transformers embedding
    confidence_score = Column(Float, nullable=False)
    sources_count = Column(JSON)  # {papers: int, articles: int, audio: int, midi: int}

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    artist = relationship("Artist", back_populates="style_profile")

    # Vector index (created in migration)
    __table_args__ = (
        Index('idx_embedding_ivfflat', 'embedding', postgresql_using='ivfflat',
              postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )
```

**ResearchSource Model** (`src/database/models.py`)
```python
class ResearchSource(Base):
    __tablename__ = 'research_sources'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artist_id = Column(UUID(as_uuid=True), ForeignKey('artists.id'), nullable=False)

    source_type = Column(String(50), nullable=False)  # paper, article, audio, midi
    url = Column(Text)
    file_path = Column(Text)
    raw_content = Column(Text)
    extracted_data = Column(JSON)
    confidence = Column(Float, default=0.5)
    collected_at = Column(DateTime, default=func.now())

    # Relationships
    artist = relationship("Artist", back_populates="research_sources")

    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_artist_source_type', 'artist_id', 'source_type'),
    )
```

**GenerationHistory Model** (`src/database/models.py`)
```python
class GenerationHistory(Base):
    __tablename__ = 'generation_history'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artist_id = Column(UUID(as_uuid=True), ForeignKey('artists.id', ondelete='SET NULL'))

    provider_used = Column(String(50), nullable=False)  # anthropic, google, openai, template
    generation_time_ms = Column(Integer, nullable=False)
    tokens_used = Column(Integer)
    cost_usd = Column(Float)
    user_params = Column(JSON, nullable=False)  # {bars, tempo, time_signature, variations}
    output_files = Column(JSON)  # Array of file paths
    created_at = Column(DateTime, default=func.now(), index=True)

    # Relationships
    artist = relationship("Artist", back_populates="generation_history")
```

### APIs and Interfaces

**DatabaseManager Interface** (`src/database/manager.py`)

```python
class DatabaseManager:
    """Async database operations manager with Redis caching."""

    async def get_or_create_artist(self, artist_name: str) -> Artist:
        """Get artist by name, create if doesn't exist."""

    async def save_style_profile(self, profile: StyleProfile) -> StyleProfile:
        """Save or update StyleProfile, invalidate cache."""

    async def get_style_profile(self, artist_name: str) -> Optional[StyleProfile]:
        """Get StyleProfile with Redis caching (7-day TTL)."""

    async def find_similar_artists(self, artist_name: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Vector similarity search using pgvector.
        Returns: [(artist_name, similarity_score), ...]
        """

    async def save_generation_history(self, history: GenerationHistory) -> GenerationHistory:
        """Log generation event for analytics."""

    async def get_generation_stats(self) -> dict:
        """Aggregate statistics across all generations."""

    async def get_artist_generation_history(self, artist_name: str, limit: int = 10) -> List[GenerationHistory]:
        """Get recent generations for specific artist."""
```

**REST API Endpoints** (New in Epic 3)

**GET /api/v1/similar/{artist}**
```
Request:
  Query params: ?limit=5 (default: 5, max: 20)

Response 200 OK:
{
  "artist": "J Dilla",
  "similar_artists": [
    {"name": "Questlove", "similarity": 0.91},
    {"name": "John Bonham", "similarity": 0.73},
    {"name": "Travis Barker", "similarity": 0.42}
  ]
}

Response 404 Not Found:
{
  "error": "artist_not_found",
  "message": "Artist 'Unknown Artist' not found in database"
}
```

**GET /api/v1/stats**
```
Response 200 OK:
{
  "total_generations": 1523,
  "avg_generation_time_ms": 1847,
  "total_cost_usd": 12.34,
  "provider_usage": {
    "anthropic": 912,
    "google": 456,
    "openai": 123,
    "template": 32
  },
  "avg_cost_per_generation": 0.0081,
  "cached_artists": 342
}
```

**GET /api/v1/artists/{artist}/history**
```
Request:
  Query params: ?limit=10 (default: 10, max: 100)

Response 200 OK:
{
  "artist": "J Dilla",
  "total_generations": 47,
  "history": [
    {
      "id": "uuid",
      "provider_used": "anthropic",
      "generation_time_ms": 1847,
      "tokens_used": 2340,
      "cost_usd": 0.0123,
      "user_params": {"bars": 4, "tempo": 95, "variations": 4},
      "created_at": "2025-11-18T10:30:00Z"
    },
    ...
  ]
}
```

### Workflows and Sequencing

**Workflow 1: Save StyleProfile after Research**
```
Research Orchestrator completes research
  → StyleProfile object created with embedding
  → db.save_style_profile(profile)
    → Begin transaction
    → db.get_or_create_artist(artist_name)
      → Query Artist by name
      → If not exists: INSERT new Artist
    → Upsert StyleProfile (artist_id FK)
    → Update Artist.confidence_score
    → Update Artist.sources_count
    → Update Artist.research_status = 'cached'
    → Commit transaction
    → Invalidate Redis cache: DEL profile:{artist_name}
  → Return saved profile
```

**Workflow 2: Get StyleProfile (with caching)**
```
Main Orchestrator requests profile
  → db.get_style_profile("J Dilla")
    → Redis GET profile:J Dilla
      → Cache HIT?
        → YES: Deserialize JSON → Return profile (< 10ms)
        → NO: Continue to PostgreSQL
    → PostgreSQL query:
        SELECT style_profiles.* FROM style_profiles
        JOIN artists ON artists.id = style_profiles.artist_id
        WHERE artists.name = 'J Dilla'
    → If found:
        → Serialize to JSON
        → Redis SETEX profile:J Dilla 604800 {json}  (7 days TTL)
        → Return profile (< 100ms)
    → If not found:
        → Return None
```

**Workflow 3: Vector Similarity Search**
```
User requests similar artists to "J Dilla"
  → GET /api/v1/similar/J%20Dilla?limit=5
  → db.find_similar_artists("J Dilla", limit=5)
    → Get embedding for "J Dilla"
      → SELECT embedding FROM style_profiles WHERE artist_id = (SELECT id FROM artists WHERE name = 'J Dilla')
    → Vector similarity query:
        SELECT artists.name,
               (embedding <=> {query_embedding}) AS distance
        FROM style_profiles
        JOIN artists ON artists.id = style_profiles.artist_id
        WHERE artists.name != 'J Dilla'
        ORDER BY embedding <=> {query_embedding}
        LIMIT 5
      → Uses IVFFlat index for fast approximate search
    → Convert distance to similarity: 1 - distance
    → Return [(name, similarity), ...]
  → Return JSON response
```

**Workflow 4: Log Generation History**
```
LLM Provider Manager completes generation
  → history = GenerationHistory(
      artist_id=artist.id,
      provider_used="anthropic",
      generation_time_ms=1847,
      tokens_used=2340,
      cost_usd=0.0123,
      user_params={...},
      output_files=[...]
    )
  → db.save_generation_history(history)
    → INSERT INTO generation_history
    → Commit
  → (Async) Analytics aggregation triggered
```

## Non-Functional Requirements

### Performance

**Database Query Performance:**
- **StyleProfile retrieval (Redis cache hit):** < 10ms (target: 5ms avg)
- **StyleProfile retrieval (PostgreSQL):** < 100ms (target: 50ms avg, PRD NFR-1)
- **Vector similarity search:** < 200ms for 10,000 artists using IVFFlat index
- **Generation history query:** < 150ms for last 100 records per artist

**Caching Performance:**
- **Redis cache hit rate:** > 80% after 30 days of operation
- **Cache invalidation:** Immediate on profile update (< 5ms)
- **Memory usage:** < 500MB Redis memory for 10,000 cached profiles

**Connection Pooling:**
- **PostgreSQL pool size:** 10 base connections, 20 overflow (ARCHITECTURE.md line 1549)
- **Pool checkout time:** < 50ms under normal load
- **Connection reuse:** > 95% of queries use pooled connections

**Scalability Targets:**
- **Concurrent database operations:** Support 100 simultaneous queries (PRD NFR-1)
- **Database size:** Support 100,000+ artists without performance degradation
- **Query optimization:** All frequent queries use appropriate indexes

### Security

**Database Access:**
- **Connection security:** TLS/SSL encryption for all PostgreSQL connections
- **Credential management:** Database credentials stored in environment variables (`.env`), never in code
- **Least privilege:** Application uses dedicated database user with restricted permissions (no DROP, ALTER on production)
- **Read-only endpoints:** Analytics endpoints use read-only database role

**Input Validation:**
- **Artist name sanitization:** Prevent SQL injection via parameterized queries (SQLAlchemy ORM)
- **Limit validation:** API endpoints enforce max limits (e.g., similarity search limit ≤ 20)
- **UUID validation:** All ID-based queries validate UUID format before querying

**Redis Security:**
- **Network isolation:** Redis accessible only from application servers (firewall rules)
- **Authentication:** Redis password protection enabled (Redis ACL)
- **Data encryption:** Redis over TLS in production (optional in dev)

**Data Protection:**
- **Sensitive data:** No user PII stored in database (artist names are public data)
- **API rate limiting:** Implemented at API layer (Epic 4), database queries inherit limits
- **Audit logging:** All write operations logged with timestamps

### Reliability/Availability

**Database Reliability:**
- **Uptime target:** 99.5% availability (PRD NFR-2)
- **Connection pooling:** Automatic reconnection on transient failures
- **Transaction management:** ACID guarantees for all write operations
- **Backup strategy:** Daily automated backups with 30-day retention
- **Recovery:** Point-in-time recovery (PITR) capability for disaster scenarios

**Redis Reliability:**
- **Persistence:** RDB snapshots every 15 minutes + AOF (append-only file) for durability
- **Failover:** Graceful degradation on Redis failure (fallback to PostgreSQL only)
- **Cache warming:** Automatic re-population on Redis restart from PostgreSQL

**Error Handling:**
- **Database connection failures:** Retry logic with exponential backoff (max 3 retries)
- **Query timeouts:** 30-second timeout for all queries, return error to caller
- **Cache failures:** Transparent fallback to database on Redis errors
- **Transaction rollback:** Automatic rollback on any error during multi-statement transactions

**Data Integrity:**
- **Foreign key constraints:** Enforced at database level (ON DELETE CASCADE/SET NULL)
- **Unique constraints:** Prevent duplicate artists, duplicate profiles per artist
- **Validation:** Pydantic models validate data before database writes

### Observability

**Logging:**
- **Structured logging:** JSON format using Loguru (ARCHITECTURE.md line 1616-1629)
- **Log levels:** DEBUG (dev), INFO (production), ERROR (always)
- **Key events logged:**
  - Database connection established/closed
  - Cache hit/miss events
  - Query execution time > 500ms (slow query warning)
  - All write operations (save_style_profile, save_generation_history)
  - Vector similarity searches
- **Log rotation:** 500MB per file, 10-day retention

**Metrics (Prometheus):**
- **Database metrics:**
  - `db_query_duration_seconds{operation, table}` - Histogram of query times
  - `db_cache_hit_rate` - Gauge of Redis cache hit percentage
  - `db_pool_active_connections` - Gauge of active PostgreSQL connections
  - `db_pool_idle_connections` - Gauge of idle connections
- **Operation counters:**
  - `db_style_profile_saves_total` - Counter of profile saves
  - `db_generation_history_saves_total` - Counter of history logs
  - `db_similarity_searches_total` - Counter of similarity searches
- **Error metrics:**
  - `db_query_errors_total{operation, error_type}` - Counter of database errors
  - `db_cache_errors_total{error_type}` - Counter of Redis errors

**Health Checks:**
- **Database health:** `SELECT 1` query on /health endpoint
- **Redis health:** `PING` command on /health endpoint
- **Connection pool health:** Check active vs. idle connections ratio

**Tracing:**
- **Distributed tracing:** Correlate database operations with API requests (request_id propagation)
- **Query tracing:** Log slow queries with execution plan (EXPLAIN) for optimization

## Dependencies and Integrations

**Python Dependencies** (from requirements.txt):
- **sqlalchemy==2.0.36** - ORM for database abstraction
- **alembic==1.14.0** - Database schema migrations
- **psycopg2-binary==2.9.10** - PostgreSQL driver (synchronous)
- **asyncpg==0.29.0** - PostgreSQL driver (asynchronous operations)
- **pgvector==0.3.6** - pgvector extension bindings for vector operations
- **redis==5.2.1** - Redis client for caching and Celery message broker
- **pydantic==2.6.1** - Data validation for model serialization/deserialization

**External Services:**
- **PostgreSQL 16+** - Primary relational database with pgvector extension
- **Redis 7.2+** - Cache layer and Celery message broker (already configured in Phase 2)

**Integration Dependencies:**
| Upstream Component | Depends On | Interface | Notes |
|--------------------|------------|-----------|-------|
| Research Orchestrator (Epic 1) | DatabaseManager | `save_style_profile()` | Stores completed research |
| LLM Provider Manager (Epic 2) | DatabaseManager | `get_style_profile()`, `save_generation_history()` | Retrieves profiles, logs usage |
| Main Orchestrator | DatabaseManager | `get_style_profile()`, `find_similar_artists()` | Cache checks, discovery feature |
| FastAPI Routes (Epic 4) | DatabaseManager | All methods | Exposes database operations via REST API |

**Downstream Components (Epic 3 provides to):**
- **Epic 1 (Research):** Persistent storage for StyleProfiles and ResearchSources
- **Epic 2 (Generation):** Fast StyleProfile retrieval via Redis caching
- **Epic 4 (API):** Analytics and similarity search endpoints
- **Epic 6 (Ableton):** Indirect - faster generation via caching improves UX

## Acceptance Criteria (Authoritative)

**AC-3.1: Database Models Defined**
- All 4 SQLAlchemy models (Artist, StyleProfile, ResearchSource, GenerationHistory) are implemented
- All columns, constraints, indexes, and relationships match schema specification
- Models can be imported and instantiated without errors
- pgvector Vector(384) type is properly configured for embedding column

**AC-3.2: DatabaseManager CRUD Operations**
- `get_or_create_artist(name)` creates new artist or returns existing
- `save_style_profile(profile)` persists StyleProfile and updates Artist record
- `get_style_profile(name)` retrieves profile from cache (if available) or database
- `find_similar_artists(name, limit)` returns top N similar artists using vector search
- `save_generation_history(history)` logs generation event
- All operations are async and use SQLAlchemy async session

**AC-3.3: Alembic Migrations Working**
- Migration file `a86af6571aac_initial_v2_0_schema.py` exists and is valid
- `alembic upgrade head` creates all tables and indexes without errors
- `alembic downgrade -1` rolls back schema changes successfully
- pgvector extension is created automatically

**AC-3.4: Redis Caching Layer**
- `get_style_profile()` checks Redis before querying PostgreSQL
- Cache miss: profile loaded from DB, cached in Redis with 7-day TTL
- Cache hit: profile returned from Redis in < 10ms
- `save_style_profile()` invalidates Redis cache for updated artist
- Redis failure does not crash application (graceful fallback to PostgreSQL)

**AC-3.5: Vector Similarity Search**
- `find_similar_artists(name, limit)` returns artists ordered by embedding similarity
- Query uses IVFFlat index for performance (< 200ms for 10,000 artists)
- Similarity score calculated as `1 - cosine_distance`
- Query artist is excluded from results
- GET /api/v1/similar/{artist} endpoint returns JSON response with similar artists

**AC-3.6: Generation History Analytics**
- `save_generation_history()` logs provider, tokens, cost, params, timing
- GET /api/v1/stats endpoint returns aggregate statistics (total generations, avg time, cost, provider usage)
- GET /api/v1/artists/{artist}/history endpoint returns recent generations for artist
- Analytics queries complete in < 500ms

**AC-3.7: Performance Targets Met**
- StyleProfile retrieval (cache hit): < 10ms
- StyleProfile retrieval (database): < 100ms
- Vector similarity search: < 200ms (10,000 artists)
- Connection pool maintains 10 base + 20 overflow connections

## Traceability Mapping

| Acceptance Criteria | PRD Requirement | Architecture Component | Implementation | Test Coverage |
|---------------------|-----------------|------------------------|----------------|---------------|
| AC-3.1: Database Models | FR-3.1, FR-3.2, FR-3.3 (PRD lines 277-297) | Data Layer (ARCH lines 99-109), Schema (lines 1033-1134) | `src/database/models.py` | `tests/unit/test_database_models.py` |
| AC-3.2: DatabaseManager CRUD | FR-3.1, FR-3.2, FR-3.3 (PRD lines 277-297) | DatabaseManager (ARCH lines 1247-1262) | `src/database/manager.py` | `tests/unit/test_database_manager.py`, `tests/integration/test_database_integration.py` |
| AC-3.3: Alembic Migrations | FR-3.1 (PRD lines 277-280) | Migrations (ARCH lines 1136-1162) | `alembic/versions/a86af6571aac_*.py` | `tests/integration/test_migrations.py` |
| AC-3.4: Redis Caching | NFR-1 Performance (PRD lines 376-383) | Caching Strategy (ARCH lines 1557-1591) | DatabaseManager with Redis client | `tests/unit/test_redis_caching.py`, `tests/integration/test_cache_integration.py` |
| AC-3.5: Vector Similarity | US-002 (PRD lines 91-101) | Vector Search (ARCH lines 1318-1377) | DatabaseManager.find_similar_artists(), API route | `tests/unit/test_similarity_search.py`, `tests/integration/test_vector_search.py` |
| AC-3.6: Analytics | FR-3.4 (PRD lines 294-297) | Generation History (ARCH lines 1102-1121) | `src/database/analytics.py`, API routes | `tests/unit/test_analytics.py` |
| AC-3.7: Performance | NFR-1 (PRD lines 376-383) | Performance Optimization (ARCH lines 1523-1609) | All database components | `tests/integration/test_performance_benchmarks.py` |

**Story-to-Component Mapping:**
- **E3.S1 (Database Models)** → `src/database/models.py` → AC-3.1
- **E3.S2 (DatabaseManager)** → `src/database/manager.py` → AC-3.2
- **E3.S3 (Alembic Migrations)** → `alembic/versions/*.py` → AC-3.3
- **E3.S4 (Vector Similarity)** → DatabaseManager + API route → AC-3.5
- **E3.S5 (Redis Caching)** → DatabaseManager caching logic → AC-3.4
- **E3.S6 (Analytics)** → `src/database/analytics.py` + API routes → AC-3.6

## Risks, Assumptions, Open Questions

**Risks:**

**RISK-3.1: Vector Index Performance Degradation** (Medium)
- **Description:** IVFFlat index may degrade with > 100,000 artists
- **Impact:** Similarity search exceeds 200ms target
- **Mitigation:**
  - Monitor query performance with Prometheus metrics
  - Tune IVFFlat `lists` parameter (currently 100)
  - Consider HNSW index upgrade in future (pgvector 0.5.0+)
- **Contingency:** Implement result caching for popular similarity queries

**RISK-3.2: Redis Memory Exhaustion** (Low)
- **Description:** 10,000 cached profiles × ~50KB each = 500MB, may grow beyond capacity
- **Impact:** Redis evictions increase cache miss rate
- **Mitigation:**
  - Set Redis `maxmemory-policy` to `allkeys-lru` (evict least recently used)
  - Monitor cache hit rate; alert if < 70%
  - Compress JSON before caching (gzip) to reduce memory footprint
- **Contingency:** Reduce TTL from 7 days to 3 days, or increase Redis memory allocation

**RISK-3.3: Database Connection Pool Exhaustion** (Medium)
- **Description:** 100 concurrent users may exceed 30 total connections (10 base + 20 overflow)
- **Impact:** New requests timeout waiting for connections
- **Mitigation:**
  - Implement request queuing at API layer (Epic 4)
  - Monitor pool utilization (`db_pool_active_connections` metric)
  - Tune pool size based on load testing results
- **Contingency:** Scale horizontally (add more API workers) or increase pool size to 20 base + 40 overflow

**Assumptions:**

**ASSUMPTION-3.1:** PostgreSQL 16+ with pgvector extension is available in production environment
- **Validation:** Verify infrastructure requirements with DevOps team before Epic 3 start

**ASSUMPTION-3.2:** Redis is already configured and running (Phase 2 infrastructure)
- **Validation:** Test Redis connectivity in development environment

**ASSUMPTION-3.3:** StyleProfile embeddings are always 384-dimensional (sentence-transformers default)
- **Validation:** Coordinate with Epic 1 (Research Pipeline) to confirm embedding model

**ASSUMPTION-3.4:** 7-day cache TTL is sufficient for most use cases
- **Validation:** Monitor cache hit rates in production; adjust TTL if hit rate < 80%

**Open Questions:**

**QUESTION-3.1:** Should we implement read replicas for analytics queries to reduce load on primary database?
- **Status:** DEFERRED - Not needed for MVP, revisit in Phase 2 if query latency becomes issue

**QUESTION-3.2:** What is the expected cache hit rate after 30 days of operation?
- **Status:** OPEN - Need to model user behavior patterns; target > 80% based on Zipf distribution assumption

**QUESTION-3.3:** Should generation history be partitioned by date for better query performance?
- **Status:** DEFERRED - Implement if GenerationHistory table exceeds 1M rows

**QUESTION-3.4:** Do we need to support database backups in development environment?
- **Status:** RESOLVED - YES, implement automated daily backups to prevent data loss during testing

## Test Strategy Summary

**Test Levels:**

**L1: Unit Tests** (`tests/unit/`)
- **Target:** 80%+ code coverage for database module
- **Scope:**
  - `test_database_models.py`: Model instantiation, relationships, constraints
  - `test_database_manager.py`: CRUD operations with mock database (sqlite in-memory)
  - `test_redis_caching.py`: Cache logic with mock Redis (fakeredis library)
  - `test_similarity_search.py`: Vector distance calculations, result ordering
  - `test_analytics.py`: Aggregation logic, statistics calculations
- **Tools:** pytest, pytest-asyncio, fakeredis, SQLAlchemy in-memory sqlite
- **Execution:** `pytest tests/unit/test_database_*.py -v --cov=src.database`

**L2: Integration Tests** (`tests/integration/`)
- **Target:** All acceptance criteria validated
- **Scope:**
  - `test_database_integration.py`: Real PostgreSQL + pgvector operations
  - `test_cache_integration.py`: Real Redis caching with TTL verification
  - `test_vector_search.py`: Vector similarity with 100+ test profiles
  - `test_migrations.py`: Alembic upgrade/downgrade cycle
  - `test_performance_benchmarks.py`: Query timing validation (< 100ms, < 200ms targets)
- **Environment:** Docker Compose with PostgreSQL 16 + pgvector + Redis
- **Execution:** `pytest tests/integration/test_database_*.py -v`

**L3: End-to-End Tests** (`tests/e2e/`)
- **Scope:**
  - Research → Database → Cache → Generation flow
  - Similarity search via API endpoint
  - Analytics endpoint responses
- **Environment:** Full stack (API + Database + Redis)
- **Execution:** `pytest tests/e2e/ -v`

**Test Data:**
- **Fixtures:** Create 10 sample StyleProfiles with realistic embeddings
- **Factories:** Use factory_boy for generating test artists, profiles, sources
- **Cleanup:** Teardown all test data after each test (pytest fixtures with `autouse=True`)

**Performance Testing:**
- **Load test:** Simulate 100 concurrent `get_style_profile()` calls
- **Stress test:** Insert 1,000 StyleProfiles and measure similarity search time
- **Benchmark:** Measure P50, P95, P99 latencies for all CRUD operations
- **Tool:** pytest-benchmark for micro-benchmarks

**Edge Cases to Test:**
1. Concurrent updates to same StyleProfile (transaction isolation)
2. Redis failure during `get_style_profile()` (fallback to PostgreSQL)
3. Vector search with no similar artists (empty result set)
4. Profile update with stale cache (cache invalidation)
5. Connection pool exhaustion (max connections reached)
6. Invalid embeddings (wrong dimensions, null values)
7. Very large generation history (pagination, query performance)

**Acceptance Test Mapping:**
- **AC-3.1:** Unit test all models, verify schema matches specification
- **AC-3.2:** Integration test all DatabaseManager methods with real PostgreSQL
- **AC-3.3:** Integration test Alembic migrations (upgrade/downgrade)
- **AC-3.4:** Integration test Redis caching (hit/miss, TTL, invalidation)
- **AC-3.5:** Integration test vector similarity with performance benchmark
- **AC-3.6:** Integration test analytics queries, verify aggregation correctness
- **AC-3.7:** Performance benchmarks for all operations against targets
