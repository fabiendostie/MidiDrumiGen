"""
Unit Tests for Generation History Analytics

Story: E3.S6 - Generation History Analytics
Test Coverage: Stats aggregation, history queries, API endpoints
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.manager import DatabaseManager
from src.database.models import Artist, GenerationHistory

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db_manager():
    """Create DatabaseManager instance for testing."""
    return DatabaseManager(
        database_url="postgresql+asyncpg://test:test@localhost/test_db", redis_url=None
    )


@pytest.fixture
def sample_generation_history():
    """Sample generation history records."""
    artist = Artist(id=1, name="J Dilla", research_status="completed", confidence_score=0.89)

    return [
        GenerationHistory(
            id=1,
            artist_id=1,
            artist=artist,
            provider_used="anthropic",
            generation_time_ms=1847,
            tokens_used=2340,
            cost_usd=0.0123,
            user_params={"bars": 4, "tempo": 95},
            output_files=["output/patterns/jdilla_001.mid"],
            created_at=datetime.now(),
        ),
        GenerationHistory(
            id=2,
            artist_id=1,
            artist=artist,
            provider_used="google",
            generation_time_ms=1200,
            tokens_used=1800,
            cost_usd=0.0045,
            user_params={"bars": 8, "tempo": 90},
            output_files=["output/patterns/jdilla_002.mid"],
            created_at=datetime.now() - timedelta(hours=1),
        ),
        GenerationHistory(
            id=3,
            artist_id=1,
            artist=artist,
            provider_used="anthropic",
            generation_time_ms=2100,
            tokens_used=2500,
            cost_usd=0.0150,
            user_params={"bars": 4, "tempo": 100},
            output_files=["output/patterns/jdilla_003.mid"],
            created_at=datetime.now() - timedelta(hours=2),
        ),
    ]


# =============================================================================
# Test Generation Statistics
# =============================================================================


@pytest.mark.asyncio
class TestGenerationStats:
    """Test get_generation_stats() method."""

    async def test_stats_with_data(self, db_manager):
        """Should calculate correct statistics with data."""
        # Mock results for multiple execute() calls
        # get_generation_stats calls execute() 4 times:
        # 1. Total count  2. Avg time  3. Total cost  4. Provider breakdown

        mock_total = MagicMock()
        mock_total.scalar.return_value = 3

        mock_avg_time = MagicMock()
        mock_avg_time.scalar.return_value = 1715.67

        mock_total_cost = MagicMock()
        mock_total_cost.scalar.return_value = 0.0318

        mock_providers = MagicMock()
        mock_providers.fetchall.return_value = [("anthropic", 2), ("google", 1)]

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(
            side_effect=[mock_total, mock_avg_time, mock_total_cost, mock_providers]
        )

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            stats = await db_manager.get_generation_stats()

            # Assertions
            assert stats["total_generations"] == 3
            assert stats["avg_generation_time_ms"] == 1715.67
            assert stats["total_cost_usd"] == 0.0318
            assert stats["provider_usage"] == {"anthropic": 2, "google": 1}
            assert stats["avg_cost_per_generation"] == pytest.approx(0.0106, rel=1e-3)

    async def test_stats_empty_database(self, db_manager):
        """Should return zero stats when database is empty."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0  # No generations

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            stats = await db_manager.get_generation_stats()

            # Assertions
            assert stats["total_generations"] == 0
            assert stats["avg_generation_time_ms"] == 0
            assert stats["total_cost_usd"] == 0.0
            assert stats["provider_usage"] == {}
            assert stats["avg_cost_per_generation"] == 0.0

    async def test_stats_provider_breakdown(self, db_manager):
        """Should correctly aggregate provider usage."""
        # Mock 4 execute() calls for get_generation_stats
        mock_total = MagicMock()
        mock_total.scalar.return_value = 10

        mock_avg_time = MagicMock()
        mock_avg_time.scalar.return_value = 1500.0

        mock_total_cost = MagicMock()
        mock_total_cost.scalar.return_value = 0.10

        mock_providers = MagicMock()
        mock_providers.fetchall.return_value = [("anthropic", 6), ("google", 3), ("openai", 1)]

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(
            side_effect=[mock_total, mock_avg_time, mock_total_cost, mock_providers]
        )

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            stats = await db_manager.get_generation_stats()

            # Assertions
            assert stats["provider_usage"]["anthropic"] == 6
            assert stats["provider_usage"]["google"] == 3
            assert stats["provider_usage"]["openai"] == 1
            assert sum(stats["provider_usage"].values()) == 10


# =============================================================================
# Test Artist Generation History
# =============================================================================


@pytest.mark.asyncio
class TestArtistHistory:
    """Test get_artist_generation_history() method."""

    async def test_get_artist_history(self, db_manager, sample_generation_history):
        """Should retrieve generation history for artist."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_generation_history

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            history = await db_manager.get_artist_generation_history("J Dilla", limit=100)

            # Assertions
            assert len(history) == 3
            assert history[0]["provider_used"] == "anthropic"
            assert history[0]["generation_time_ms"] == 1847
            assert history[0]["cost_usd"] == 0.0123

    async def test_history_respects_limit(self, db_manager, sample_generation_history):
        """Should respect limit parameter."""
        # Return only 2 records
        limited_history = sample_generation_history[:2]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = limited_history

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute with limit=2
            history = await db_manager.get_artist_generation_history("J Dilla", limit=2)

            # Assertions
            assert len(history) == 2

    async def test_history_ordered_by_date(self, db_manager, sample_generation_history):
        """Should return history ordered by created_at DESC."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_generation_history

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            history = await db_manager.get_artist_generation_history("J Dilla")

            # Verify order (most recent first)
            timestamps = [datetime.fromisoformat(record["created_at"]) for record in history]
            assert timestamps == sorted(timestamps, reverse=True)

    async def test_history_empty_for_unknown_artist(self, db_manager):
        """Should return empty list for unknown artist."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            history = await db_manager.get_artist_generation_history("Unknown Artist")

            # Assertions
            assert history == []


# =============================================================================
# Test API Endpoint Integration
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(reason="API endpoints are Epic 4 - requires full API setup")
class TestAnalyticsAPIEndpoints:
    """Test analytics API endpoints."""

    async def test_stats_endpoint(self, db_manager):
        """Should return statistics through API endpoint."""
        from src.api.routes.utils import get_generation_statistics

        mock_stats = {
            "total_generations": 1523,
            "avg_generation_time_ms": 1847.5,
            "total_cost_usd": 12.34,
            "provider_usage": {"anthropic": 980, "google": 450, "openai": 93},
            "avg_cost_per_generation": 0.0081,
        }

        with patch.object(db_manager, "get_generation_stats", return_value=mock_stats):
            response = await get_generation_statistics(db=db_manager)

            assert response.total_generations == 1523
            assert response.avg_generation_time_ms == 1847.5
            assert response.provider_usage["anthropic"] == 980

    async def test_history_endpoint(self, db_manager):
        """Should return artist history through API endpoint."""
        from src.api.routes.utils import get_artist_generation_history

        mock_history = [
            {
                "provider_used": "anthropic",
                "generation_time_ms": 1847,
                "tokens_used": 2340,
                "cost_usd": 0.0123,
                "user_params": {"bars": 4, "tempo": 95},
                "output_files": ["output/patterns/jdilla_001.mid"],
                "created_at": datetime.now().isoformat(),
            }
        ]

        with patch.object(db_manager, "get_artist_generation_history", return_value=mock_history):
            response = await get_artist_generation_history(
                artist="J Dilla", limit=100, db=db_manager
            )

            assert response.artist == "J Dilla"
            assert response.total_generations == 1
            assert response.history[0].provider_used == "anthropic"

    async def test_history_endpoint_not_found(self, db_manager):
        """Should raise 404 when no history found."""
        from fastapi import HTTPException

        from src.api.routes.utils import get_artist_generation_history

        with patch.object(db_manager, "get_artist_generation_history", return_value=[]):
            with pytest.raises(HTTPException) as exc_info:
                await get_artist_generation_history(
                    artist="Unknown Artist", limit=100, db=db_manager
                )

            assert exc_info.value.status_code == 404


# =============================================================================
# Test Save Generation History
# =============================================================================


@pytest.mark.asyncio
class TestSaveHistory:
    """Test save_generation_history() method."""

    async def test_save_history_record(self, db_manager, sample_generation_history):
        """Should save generation history record."""
        record = sample_generation_history[0]

        with patch.object(db_manager, "SessionLocal") as mock_session:
            mock_session_instance = AsyncMock()
            mock_session_instance.commit = AsyncMock()
            mock_session_instance.refresh = AsyncMock()

            mock_session.return_value.__aenter__.return_value = mock_session_instance

            # Execute
            saved_record = await db_manager.save_generation_history(mock_session_instance, record)

            # Verify commit was called
            mock_session_instance.commit.assert_called_once()
            assert saved_record == record


# =============================================================================
# Integration Tests (Slow - Marked)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
class TestRealAnalytics:
    """Integration tests with real PostgreSQL database."""

    async def test_real_analytics_queries(self):
        """
        Test real analytics queries against PostgreSQL.

        Note: This test requires:
        - PostgreSQL database with sample data
        - generation_history table populated
        - May be slow
        """
        # TODO: Implement integration test with real database
        # This would test:
        # - Real aggregation queries
        # - Query performance (< 500ms for stats)
        # - Complex GROUP BY operations
        # - JOIN performance
        pass


# TODO: Add more tests
# - Test time-based filtering (last 7 days, last 30 days)
# - Test cost analysis by provider
# - Test token usage analytics
# - Test generation time percentiles (p50, p95, p99)
# - Test export to CSV/JSON for analytics
# - Test dashboard data aggregation
# - Test concurrent analytics queries
