"""
Unit tests for vector similarity search functionality.

Story: E3.S4 - Vector Similarity Search
Tests: AC-3.4.1, AC-3.4.2, AC-3.4.4
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.manager import ArtistNotFoundError, DatabaseError, DatabaseManager
from src.database.models import Artist, StyleProfile


@pytest.fixture
def db_manager():
    """Create DatabaseManager with test configuration."""
    return DatabaseManager(
        database_url="postgresql+asyncpg://test:test@localhost/test_db",
        redis_url=None,  # Disable Redis for unit tests
    )


@pytest.fixture
def mock_artist():
    """Create a mock artist."""
    artist = MagicMock(spec=Artist)
    artist.id = uuid.uuid4()
    artist.name = "J Dilla"
    return artist


@pytest.fixture
def mock_profile():
    """Create a mock StyleProfile with embedding."""
    profile = MagicMock(spec=StyleProfile)
    profile.id = uuid.uuid4()
    profile.artist_id = uuid.uuid4()
    profile.embedding = [0.1] * 384  # 384-dimensional embedding
    profile.confidence_score = 0.85
    return profile


class TestFindSimilarArtistsValidation:
    """Test limit parameter validation."""

    @pytest.mark.asyncio
    async def test_limit_zero_returns_empty(self, db_manager):
        """AC-3.4.4: Query with limit=0 returns empty list."""
        result = await db_manager.find_similar_artists("J Dilla", limit=0)
        assert result == []

    @pytest.mark.asyncio
    async def test_negative_limit_returns_empty(self, db_manager):
        """AC-3.4.4: Query with negative limit returns empty list."""
        result = await db_manager.find_similar_artists("J Dilla", limit=-5)
        assert result == []

    @pytest.mark.asyncio
    async def test_limit_capped_at_20(self, db_manager, mock_artist, mock_profile):
        """AC-3.4.1: Query with limit > 20 is capped at 20."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            # Setup mock session
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            # Mock artist lookup
            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            # Mock profile lookup
            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = mock_profile

            # Mock similarity search results
            mock_search_result = MagicMock()
            mock_search_result.fetchall.return_value = [
                ("Artist1", 0.95),
                ("Artist2", 0.90),
            ]

            # Setup execute to return different results for each call
            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
                mock_search_result,
            ]

            await db_manager.find_similar_artists("J Dilla", limit=100)

            # Verify limit was passed as 20 (capped)
            call_args = mock_session.execute.call_args_list[2]
            query_params = call_args[0][1]
            assert query_params["limit"] == 20


class TestFindSimilarArtistsErrors:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_artist_not_found_raises_error(self, db_manager):
        """AC-3.4.4: Query for non-existent artist raises ArtistNotFoundError."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            # Mock artist lookup returning None
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = mock_result

            with pytest.raises(ArtistNotFoundError) as exc_info:
                await db_manager.find_similar_artists("NonExistent Artist")

            assert "NonExistent Artist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_profile_returns_empty(self, db_manager, mock_artist):
        """AC-3.4.4: Query for artist without StyleProfile returns empty list."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            # Mock artist found
            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            # Mock profile not found
            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = None

            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
            ]

            result = await db_manager.find_similar_artists("J Dilla")
            assert result == []

    @pytest.mark.asyncio
    async def test_null_embedding_returns_empty(self, db_manager, mock_artist, mock_profile):
        """AC-3.4.4: Query for artist with NULL embedding returns empty list."""
        mock_profile.embedding = None

        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = mock_profile

            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
            ]

            result = await db_manager.find_similar_artists("J Dilla")
            assert result == []

    @pytest.mark.asyncio
    async def test_database_error_wrapped(self, db_manager):
        """AC-3.4.4: Database connection errors wrapped in DatabaseError."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            # Simulate database error
            mock_session.execute.side_effect = Exception("Connection refused")

            with pytest.raises(DatabaseError) as exc_info:
                await db_manager.find_similar_artists("J Dilla")

            assert "Connection refused" in str(exc_info.value)


class TestFindSimilarArtistsSuccess:
    """Test successful similarity search scenarios."""

    @pytest.mark.asyncio
    async def test_returns_correct_format(self, db_manager, mock_artist, mock_profile):
        """AC-3.4.1: Returns list of tuples (artist_name, similarity)."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = mock_profile

            # Mock similarity results
            mock_search_result = MagicMock()
            mock_search_result.fetchall.return_value = [
                ("Questlove", 0.92),
                ("John Bonham", 0.78),
                ("Travis Barker", 0.45),
            ]

            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
                mock_search_result,
            ]

            result = await db_manager.find_similar_artists("J Dilla", limit=3)

            # Verify format
            assert isinstance(result, list)
            assert len(result) == 3
            for item in result:
                assert isinstance(item, tuple)
                assert len(item) == 2
                assert isinstance(item[0], str)
                assert isinstance(item[1], float)

    @pytest.mark.asyncio
    async def test_similarity_scores_in_range(self, db_manager, mock_artist, mock_profile):
        """AC-3.4.2: Similarity scores are in range 0.0-1.0."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = mock_profile

            mock_search_result = MagicMock()
            mock_search_result.fetchall.return_value = [
                ("Artist1", 0.95),
                ("Artist2", 0.50),
                ("Artist3", 0.10),
            ]

            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
                mock_search_result,
            ]

            result = await db_manager.find_similar_artists("J Dilla")

            for _, similarity in result:
                assert 0.0 <= similarity <= 1.0

    @pytest.mark.asyncio
    async def test_query_artist_excluded(self, db_manager, mock_artist, mock_profile):
        """AC-3.4.1: Query artist is excluded from results."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = mock_profile

            mock_search_result = MagicMock()
            mock_search_result.fetchall.return_value = [
                ("Questlove", 0.92),
                ("Travis Barker", 0.45),
            ]

            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
                mock_search_result,
            ]

            result = await db_manager.find_similar_artists("J Dilla")

            # Query artist should not be in results
            artist_names = [name for name, _ in result]
            assert "J Dilla" not in artist_names

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self, db_manager, mock_artist, mock_profile):
        """AC-3.4.1: Respects limit parameter."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = mock_profile

            mock_search_result = MagicMock()
            mock_search_result.fetchall.return_value = [
                ("Questlove", 0.92),
                ("John Bonham", 0.78),
            ]

            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
                mock_search_result,
            ]

            result = await db_manager.find_similar_artists("J Dilla", limit=2)

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_database_returns_empty(self, db_manager, mock_artist, mock_profile):
        """AC-3.4.4: Database with 0 other artists returns empty list."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = mock_profile

            # No similar artists found
            mock_search_result = MagicMock()
            mock_search_result.fetchall.return_value = []

            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
                mock_search_result,
            ]

            result = await db_manager.find_similar_artists("J Dilla")
            assert result == []


class TestSimilarityScoring:
    """Test similarity score calculations."""

    @pytest.mark.asyncio
    async def test_results_ordered_by_similarity(self, db_manager, mock_artist, mock_profile):
        """AC-3.4.1: Results ordered by similarity (highest first)."""
        with patch.object(db_manager, "SessionLocal") as mock_session_local:
            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_artist_result = MagicMock()
            mock_artist_result.scalar_one_or_none.return_value = mock_artist

            mock_profile_result = MagicMock()
            mock_profile_result.scalar_one_or_none.return_value = mock_profile

            # Results should be ordered by similarity (database does ordering)
            mock_search_result = MagicMock()
            mock_search_result.fetchall.return_value = [
                ("Questlove", 0.92),
                ("John Bonham", 0.78),
                ("Travis Barker", 0.45),
            ]

            mock_session.execute.side_effect = [
                mock_artist_result,
                mock_profile_result,
                mock_search_result,
            ]

            result = await db_manager.find_similar_artists("J Dilla")

            # Verify ordering
            similarities = [sim for _, sim in result]
            assert similarities == sorted(similarities, reverse=True)
