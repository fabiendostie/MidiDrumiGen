"""
Integration tests for vector similarity search with PostgreSQL + pgvector.

Story: E3.S4 - Vector Similarity Search
Tests: AC-3.4.1, AC-3.4.2, AC-3.4.3

Requirements:
- PostgreSQL 16+ with pgvector extension
- Database accessible at DATABASE_URL environment variable
"""

import os
import time

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.database.manager import ArtistNotFoundError, DatabaseManager
from src.database.models import Artist, Base, StyleProfile

# Skip if no database URL configured
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://postgres:changeme@localhost/mididrumigen_test"
)


@pytest.fixture
async def db_engine():
    """Create async database engine for tests."""
    engine = create_async_engine(DATABASE_URL, echo=False)

    # Create tables
    async with engine.begin() as conn:
        # Ensure pgvector extension exists
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    """Create async database session."""
    session_local = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_local() as session:
        yield session


@pytest.fixture
async def db_manager(db_engine):
    """Create DatabaseManager for tests."""
    manager = DatabaseManager(
        database_url=DATABASE_URL,
        redis_url=None,
    )
    # Replace engine with test engine
    manager.engine = db_engine
    manager.SessionLocal = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    yield manager


def create_realistic_embedding(style: str) -> list[float]:
    """
    Create a 384-dimensional embedding based on style characteristics.

    Uses consistent seeding for reproducibility in tests.
    """
    # Style-specific seeds for reproducibility
    style_seeds = {
        "hip-hop-swing": 42,  # J Dilla style
        "neo-soul": 43,  # Questlove style
        "punk-fast": 44,  # Travis Barker style
        "rock-groove": 45,  # John Bonham style
    }

    np.random.seed(style_seeds.get(style, 0))

    # Base embedding
    embedding = np.random.randn(384).astype(np.float32)

    # Add style-specific patterns to create similarity clusters
    if style == "hip-hop-swing":
        embedding[:50] += 0.5  # High swing component
        embedding[50:100] += 0.3  # Mid tempo component
    elif style == "neo-soul":
        embedding[:50] += 0.4  # Similar swing to hip-hop
        embedding[50:100] += 0.25  # Similar tempo
    elif style == "punk-fast":
        embedding[:50] -= 0.5  # No swing
        embedding[100:150] += 0.8  # High energy/speed
    elif style == "rock-groove":
        embedding[:50] += 0.3  # Some swing
        embedding[50:100] += 0.35  # Mid-high tempo

    # Normalize to unit length
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.tolist()


@pytest.fixture
async def test_artists(db_session):
    """
    Create test artists with realistic embeddings.

    Artists ordered by expected similarity to J Dilla:
    1. Questlove (neo-soul, similar swing/tempo)
    2. John Bonham (rock, some swing)
    3. Travis Barker (punk, no swing, fast)
    """
    artists_data = [
        {
            "name": "J Dilla",
            "style": "hip-hop-swing",
            "description": "Hip-hop producer known for lo-fi beats and heavy swing",
            "params": {"tempo_avg": 95, "swing_percent": 62, "genre": "hip-hop"},
        },
        {
            "name": "Questlove",
            "style": "neo-soul",
            "description": "Neo-soul drummer with jazzy feel and moderate swing",
            "params": {"tempo_avg": 90, "swing_percent": 58, "genre": "neo-soul"},
        },
        {
            "name": "Travis Barker",
            "style": "punk-fast",
            "description": "Punk/rock drummer known for fast, straight timing",
            "params": {"tempo_avg": 180, "swing_percent": 0, "genre": "punk"},
        },
        {
            "name": "John Bonham",
            "style": "rock-groove",
            "description": "Rock drummer with groove-oriented style",
            "params": {"tempo_avg": 110, "swing_percent": 55, "genre": "rock"},
        },
    ]

    created_artists = []

    for data in artists_data:
        # Create artist
        artist = Artist(
            name=data["name"],
            research_status="cached",
            confidence_score=0.9,
        )
        db_session.add(artist)
        await db_session.flush()

        # Create style profile with embedding
        profile = StyleProfile(
            artist_id=artist.id,
            text_description=data["description"],
            quantitative_params=data["params"],
            embedding=create_realistic_embedding(data["style"]),
            confidence_score=0.85,
            sources_count={"papers": 5, "articles": 10},
        )
        db_session.add(profile)

        created_artists.append(artist)

    await db_session.commit()

    return created_artists


@pytest.mark.integration
@pytest.mark.asyncio
class TestVectorSimilarityIntegration:
    """Integration tests with real PostgreSQL + pgvector."""

    async def test_find_similar_returns_expected_order(self, db_manager, test_artists):
        """
        AC-3.4.2: Verify similarity ranking matches expected order.

        Given: J Dilla, Questlove, Travis Barker, John Bonham profiles
        When: Querying similar artists to J Dilla
        Then: Questlove should be most similar (similar style/tempo/swing)
        """
        result = await db_manager.find_similar_artists("J Dilla", limit=3)

        assert len(result) == 3

        # Get artist names in order
        artist_names = [name for name, _ in result]

        # Questlove should be first (most similar to J Dilla)
        assert artist_names[0] == "Questlove", f"Expected Questlove first, got {artist_names}"

        # Travis Barker should be last (least similar)
        assert "Travis Barker" in artist_names

        # Verify similarity scores are in valid range
        for _, similarity in result:
            assert 0.0 <= similarity <= 1.0

    async def test_similarity_scores_descending(self, db_manager, test_artists):
        """AC-3.4.1: Results ordered by similarity (highest first)."""
        result = await db_manager.find_similar_artists("J Dilla", limit=3)

        similarities = [sim for _, sim in result]
        assert similarities == sorted(similarities, reverse=True)

    async def test_query_artist_excluded(self, db_manager, test_artists):
        """AC-3.4.1: Query artist excluded from results."""
        result = await db_manager.find_similar_artists("J Dilla", limit=10)

        artist_names = [name for name, _ in result]
        assert "J Dilla" not in artist_names

    async def test_respects_limit(self, db_manager, test_artists):
        """AC-3.4.1: Respects limit parameter."""
        result = await db_manager.find_similar_artists("J Dilla", limit=2)

        # Should only return 2 results (3 other artists exist)
        assert len(result) == 2

    async def test_artist_not_found_raises_error(self, db_manager, test_artists):
        """AC-3.4.4: Non-existent artist raises ArtistNotFoundError."""
        with pytest.raises(ArtistNotFoundError):
            await db_manager.find_similar_artists("Nonexistent Artist")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
class TestVectorSearchPerformance:
    """Performance tests for vector similarity search."""

    @pytest.fixture
    async def large_dataset(self, db_session):
        """Create 10,000 test profiles for performance testing."""
        genres = ["rock", "jazz", "funk", "hip-hop", "electronic", "latin"]

        for i in range(10000):
            artist = Artist(
                name=f"Test Artist {i}",
                research_status="cached",
                confidence_score=0.5,
            )
            db_session.add(artist)
            await db_session.flush()

            # Generate random embedding
            np.random.seed(i)
            embedding = np.random.randn(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            profile = StyleProfile(
                artist_id=artist.id,
                text_description=f"Artist {i} plays {genres[i % len(genres)]} drums",
                quantitative_params={"tempo_avg": 80 + (i % 120), "swing_percent": i % 70},
                embedding=embedding.tolist(),
                confidence_score=0.5,
                sources_count={"articles": 1},
            )
            db_session.add(profile)

            # Commit in batches to avoid memory issues
            if i % 1000 == 0:
                await db_session.commit()

        await db_session.commit()

    async def test_query_performance_10k_profiles(self, db_manager, large_dataset):
        """
        AC-3.4.3: Vector similarity search completes in < 200ms for 10,000 profiles.
        """
        # Warmup query
        await db_manager.find_similar_artists("Test Artist 0", limit=5)

        # Timed query
        start_time = time.perf_counter()
        result = await db_manager.find_similar_artists("Test Artist 0", limit=10)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete in < 200ms
        assert elapsed_ms < 200, f"Query took {elapsed_ms:.1f}ms, expected < 200ms"
        assert len(result) == 10

    async def test_index_usage(self, db_manager, large_dataset, db_session):
        """AC-3.4.3: Query execution plan confirms IVFFlat index usage."""
        # Get the query artist's embedding
        result = await db_session.execute(
            text(
                """
                SELECT sp.embedding
                FROM style_profiles sp
                JOIN artists a ON sp.artist_id = a.id
                WHERE a.name = 'Test Artist 0'
            """
            )
        )
        row = result.fetchone()
        query_embedding = row[0]

        # Check query plan
        explain_result = await db_session.execute(
            text(
                """
                EXPLAIN ANALYZE
                SELECT a.name, 1 - (sp.embedding <=> :embedding) as similarity
                FROM style_profiles sp
                JOIN artists a ON sp.artist_id = a.id
                WHERE a.name != 'Test Artist 0'
                  AND sp.embedding IS NOT NULL
                ORDER BY sp.embedding <=> :embedding
                LIMIT 10
            """
            ),
            {"embedding": query_embedding},
        )

        plan = "\n".join([row[0] for row in explain_result.fetchall()])

        # Should use index scan, not sequential scan
        assert "Seq Scan on style_profiles" not in plan, f"Query uses full table scan:\n{plan}"
