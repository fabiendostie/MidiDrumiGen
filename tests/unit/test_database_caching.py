"""
Unit Tests for Redis Caching Layer

Story: E3.S5 - Redis Caching Layer
Test Coverage: Cache hit/miss, TTL, invalidation, serialization
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.manager import DatabaseManager
from src.database.models import Artist, StyleProfile

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db_manager_with_redis():
    """Create DatabaseManager with Redis enabled."""
    return DatabaseManager(
        database_url="postgresql+asyncpg://test:test@localhost/test_db",
        redis_url="redis://localhost:6379/0",
        cache_ttl=604800,  # 7 days
    )


@pytest.fixture
def db_manager_no_redis():
    """Create DatabaseManager without Redis."""
    return DatabaseManager(
        database_url="postgresql+asyncpg://test:test@localhost/test_db", redis_url=None
    )


@pytest.fixture
def sample_style_profile():
    """Sample StyleProfile for caching tests."""
    artist = Artist(id=1, name="J Dilla", research_status="completed", confidence_score=0.89)

    return StyleProfile(
        id=1,
        artist_id=1,
        artist=artist,
        text_description="J Dilla is known for his loose, behind-the-beat swing",
        quantitative_params={
            "tempo_range": [85, 100],
            "swing_percentage": 58,
            "velocity_variation": 0.25,
        },
        midi_templates_json=[{"bar": 1, "notes": [{"time": 0, "note": 36, "velocity": 90}]}],
        confidence_score=0.89,
        sources_count=15,
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def cached_profile_json(sample_style_profile):
    """JSON-serialized StyleProfile for Redis cache."""
    return json.dumps(
        {
            "artist_name": sample_style_profile.artist.name,
            "text_description": sample_style_profile.text_description,
            "quantitative_params": sample_style_profile.quantitative_params,
            "midi_templates_json": sample_style_profile.midi_templates_json,
            "confidence_score": sample_style_profile.confidence_score,
            "sources_count": sample_style_profile.sources_count,
            "updated_at": sample_style_profile.updated_at.isoformat(),
        }
    )


# =============================================================================
# Test Cache Hit/Miss
# =============================================================================


@pytest.mark.asyncio
class TestCacheHitMiss:
    """Test cache hit and miss scenarios."""

    async def test_cache_hit(self, db_manager_with_redis, cached_profile_json):
        """Should return cached profile on cache hit."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = cached_profile_json

        # Patch get_redis to return mock Redis (async)
        async def mock_get_redis():
            return mock_redis

        with patch.object(db_manager_with_redis, "get_redis", side_effect=mock_get_redis):
            # Should deserialize and return profile from cache
            profile = await db_manager_with_redis.get_style_profile("J Dilla")

            # Verify profile was returned from cache
            assert profile is not None
            assert profile.artist.name == "J Dilla"
            assert (
                profile.text_description == "J Dilla is known for his loose, behind-the-beat swing"
            )

            # Verify Redis was queried
            mock_redis.get.assert_called_once_with("profile:J Dilla")

    async def test_cache_miss_queries_database(self, db_manager_with_redis, sample_style_profile):
        """Should query database on cache miss."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Cache miss
        mock_redis.setex = AsyncMock()  # Mock cache write

        # Patch get_redis to return mock Redis (async)
        async def mock_get_redis():
            return mock_redis

        with patch.object(db_manager_with_redis, "get_redis", side_effect=mock_get_redis):
            # Mock the database session
            mock_result = MagicMock()  # NOT AsyncMock - scalar_one_or_none is sync
            mock_result.scalar_one_or_none.return_value = sample_style_profile

            mock_session_ctx = AsyncMock()
            mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

            with patch.object(db_manager_with_redis, "SessionLocal", return_value=mock_session_ctx):
                # Execute
                profile = await db_manager_with_redis.get_style_profile("J Dilla")

                # Assertions
                assert profile is not None
                assert profile.artist.name == "J Dilla"
                mock_redis.get.assert_called_once()

    async def test_no_redis_skips_cache(self, db_manager_no_redis, sample_style_profile):
        """Should skip caching when Redis not configured."""
        # Mock the database session
        mock_result = MagicMock()  # NOT AsyncMock - scalar_one_or_none is sync
        mock_result.scalar_one_or_none.return_value = sample_style_profile

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

        with patch.object(db_manager_no_redis, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            profile = await db_manager_no_redis.get_style_profile("J Dilla")

            # Should query database directly
            assert profile is not None
            assert profile.artist.name == "J Dilla"


# =============================================================================
# Test Cache Write Operations
# =============================================================================


@pytest.mark.asyncio
class TestCacheWrite:
    """Test cache writing and serialization."""

    async def test_cache_profile(self, db_manager_with_redis, sample_style_profile):
        """Should cache profile with correct TTL."""
        mock_redis = AsyncMock()

        with patch.object(db_manager_with_redis, "get_redis", return_value=mock_redis):
            await db_manager_with_redis._cache_profile(sample_style_profile)

            # Verify Redis setex called with correct parameters
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args

            cache_key = call_args[0][0]
            ttl = call_args[0][1]
            cached_data = call_args[0][2]

            assert cache_key == "profile:J Dilla"
            assert ttl == 604800  # 7 days
            assert "J Dilla" in cached_data

    async def test_cache_serialization(self, db_manager_with_redis, sample_style_profile):
        """Should correctly serialize profile to JSON."""
        mock_redis = AsyncMock()

        with patch.object(db_manager_with_redis, "get_redis", return_value=mock_redis):
            await db_manager_with_redis._cache_profile(sample_style_profile)

            # Get serialized data
            cached_data = mock_redis.setex.call_args[0][2]
            profile_data = json.loads(cached_data)

            # Verify serialization
            assert profile_data["artist_name"] == "J Dilla"
            assert profile_data["confidence_score"] == 0.89
            assert "tempo_range" in profile_data["quantitative_params"]
            assert isinstance(profile_data["updated_at"], str)


# =============================================================================
# Test Cache Invalidation
# =============================================================================


@pytest.mark.asyncio
class TestCacheInvalidation:
    """Test cache invalidation logic."""

    async def test_invalidate_cache(self, db_manager_with_redis):
        """Should delete cache key on invalidation."""
        mock_redis = AsyncMock()

        with patch.object(db_manager_with_redis, "get_redis", return_value=mock_redis):
            await db_manager_with_redis._invalidate_cache("J Dilla")

            mock_redis.delete.assert_called_once_with("profile:J Dilla")

    async def test_save_style_profile_invalidates_cache(
        self, db_manager_with_redis, sample_style_profile
    ):
        """Should invalidate cache when saving style profile."""
        mock_redis = AsyncMock()

        with (
            patch.object(db_manager_with_redis, "get_redis", return_value=mock_redis),
            patch.object(db_manager_with_redis, "SessionLocal") as mock_session,
        ):
            mock_session_instance = AsyncMock()
            mock_session_instance.commit = AsyncMock()
            mock_session_instance.refresh = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance

            await db_manager_with_redis.save_style_profile(
                mock_session_instance, sample_style_profile
            )

            # Verify cache was invalidated
            mock_redis.delete.assert_called_once()


# =============================================================================
# Test Cache TTL
# =============================================================================


class TestCacheTTL:
    """Test cache TTL configuration."""

    def test_default_ttl(self):
        """Should use default 7-day TTL."""
        db = DatabaseManager(
            database_url="postgresql+asyncpg://test:test@localhost/test_db",
            redis_url="redis://localhost:6379/0",
        )
        assert db.cache_ttl == 604800  # 7 days in seconds

    def test_custom_ttl(self):
        """Should respect custom TTL."""
        custom_ttl = 3600  # 1 hour
        db = DatabaseManager(
            database_url="postgresql+asyncpg://test:test@localhost/test_db",
            redis_url="redis://localhost:6379/0",
            cache_ttl=custom_ttl,
        )
        assert db.cache_ttl == custom_ttl


# =============================================================================
# Test Performance
# =============================================================================


@pytest.mark.asyncio
class TestCachePerformance:
    """Test cache performance characteristics."""

    async def test_cache_hit_faster_than_database(self):
        """
        Cache hit should be significantly faster than database query.

        Acceptance Criteria:
        - Cache hit: < 10ms
        - Database query: > 50ms
        """
        # TODO: Implement performance benchmark
        # This would measure:
        # - Redis GET latency
        # - PostgreSQL query latency
        # - Deserialization overhead
        pass


# =============================================================================
# Integration Tests (Slow - Marked)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
class TestRedisCaching:
    """Integration tests with real Redis instance."""

    async def test_real_redis_cache(self):
        """
        Test real Redis caching operations.

        Note: This test requires:
        - Redis server running
        - Network access to Redis
        - May be slow
        """
        # TODO: Implement integration test with real Redis
        # This would test:
        # - Real cache writes
        # - Real cache reads
        # - TTL expiration
        # - Connection pooling
        pass


# TODO: Add more tests
# - Test cache key collisions
# - Test Redis connection failures (graceful degradation)
# - Test concurrent cache access
# - Test cache warming strategies
# - Test cache statistics (hit rate, miss rate)
# - Test Redis connection pooling
# - Test cache eviction policies
