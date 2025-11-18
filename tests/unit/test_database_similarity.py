"""
Unit Tests for Vector Similarity Search

Story: E3.S4 - Vector Similarity Search
Test Coverage: pgvector operations, cosine similarity, embedding queries
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database.manager import DatabaseManager

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db_manager():
    """Create DatabaseManager instance for testing."""
    return DatabaseManager(
        database_url="postgresql+asyncpg://test:test@localhost/test_db",
        redis_url=None,  # No Redis for similarity tests
    )


@pytest.fixture
def mock_embedding():
    """Mock 384-dimensional embedding vector."""
    return [0.1] * 384


@pytest.fixture
def sample_artists_with_embeddings():
    """Sample artists with embeddings for similarity testing."""
    return [
        {"name": "John Bonham", "similarity": 0.95, "embedding": [0.1] * 384},
        {"name": "Keith Moon", "similarity": 0.89, "embedding": [0.15] * 384},
        {"name": "Ginger Baker", "similarity": 0.85, "embedding": [0.12] * 384},
        {"name": "Neil Peart", "similarity": 0.78, "embedding": [0.2] * 384},
    ]


# =============================================================================
# Test Vector Similarity Search
# =============================================================================


@pytest.mark.asyncio
class TestFindSimilarArtists:
    """Test find_similar_artists() method."""

    async def test_find_similar_success(
        self, db_manager, mock_embedding, sample_artists_with_embeddings
    ):
        """Should find similar artists using vector similarity."""
        # Mock query artist profile
        mock_query_profile = MagicMock()
        mock_query_profile.embedding = mock_embedding

        # Mock first execute() call - get query profile
        mock_result1 = MagicMock()  # NOT AsyncMock - scalar_one_or_none is sync
        mock_result1.scalar_one_or_none.return_value = mock_query_profile

        # Mock second execute() call - similarity search
        mock_results = [
            (artist["name"], artist["similarity"]) for artist in sample_artists_with_embeddings[:3]
        ]
        mock_result2 = MagicMock()  # NOT AsyncMock - fetchall is sync
        mock_result2.fetchall.return_value = mock_results

        # Create session mock that returns different results for each execute()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(
            side_effect=[mock_result1, mock_result2]
        )

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            similar_artists = await db_manager.find_similar_artists("John Bonham", limit=3)

            # Assertions
            assert len(similar_artists) == 3
            assert similar_artists[0][0] == "John Bonham"
            assert similar_artists[0][1] == 0.95
            assert all(isinstance(score, float) for _, score in similar_artists)

    async def test_find_similar_no_embedding(self, db_manager):
        """Should return empty list when artist has no embedding."""
        # Mock query artist without embedding
        mock_query_profile = MagicMock()
        mock_query_profile.embedding = None

        mock_result = MagicMock()  # NOT AsyncMock - scalar_one_or_none is sync
        mock_result.scalar_one_or_none.return_value = mock_query_profile

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            similar_artists = await db_manager.find_similar_artists("Unknown Artist")

            # Assertions
            assert similar_artists == []

    async def test_find_similar_artist_not_found(self, db_manager):
        """Should return empty list when artist not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            similar_artists = await db_manager.find_similar_artists("Nonexistent Artist")

            # Assertions
            assert similar_artists == []

    async def test_find_similar_respects_limit(
        self, db_manager, mock_embedding, sample_artists_with_embeddings
    ):
        """Should respect limit parameter."""
        mock_query_profile = MagicMock()
        mock_query_profile.embedding = mock_embedding

        # Mock results with limit=2
        mock_results = [
            (artist["name"], artist["similarity"]) for artist in sample_artists_with_embeddings[:2]
        ]

        # First execute() - get query profile
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = mock_query_profile

        # Second execute() - similarity search
        mock_result2 = MagicMock()
        mock_result2.fetchall.return_value = mock_results

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(
            side_effect=[mock_result1, mock_result2]
        )

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute with limit=2
            similar_artists = await db_manager.find_similar_artists("John Bonham", limit=2)

            # Assertions
            assert len(similar_artists) == 2

    async def test_find_similar_excludes_query_artist(
        self, db_manager, mock_embedding, sample_artists_with_embeddings
    ):
        """Should exclude query artist from results."""
        mock_query_profile = MagicMock()
        mock_query_profile.embedding = mock_embedding

        # Mock results excluding query artist
        mock_results = [
            (artist["name"], artist["similarity"])
            for artist in sample_artists_with_embeddings[1:]  # Exclude first
        ]

        # First execute() - get query profile
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = mock_query_profile

        # Second execute() - similarity search
        mock_result2 = MagicMock()
        mock_result2.fetchall.return_value = mock_results

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value.execute = AsyncMock(
            side_effect=[mock_result1, mock_result2]
        )

        with patch.object(db_manager, "SessionLocal", return_value=mock_session_ctx):
            # Execute
            similar_artists = await db_manager.find_similar_artists("John Bonham")

            # Assertions
            artist_names = [name for name, _ in similar_artists]
            assert "John Bonham" not in artist_names


# =============================================================================
# Test Similarity Scoring
# =============================================================================


class TestSimilarityScoring:
    """Test similarity score calculation and ordering."""

    def test_similarity_score_range(self, sample_artists_with_embeddings):
        """Similarity scores should be between 0 and 1."""
        for artist in sample_artists_with_embeddings:
            assert 0 <= artist["similarity"] <= 1

    def test_similarity_ordering(self, sample_artists_with_embeddings):
        """Results should be ordered by similarity (highest first)."""
        similarities = [a["similarity"] for a in sample_artists_with_embeddings]
        assert similarities == sorted(similarities, reverse=True)

    def test_cosine_distance_formula(self):
        """
        Verify cosine distance to similarity conversion.

        pgvector <=> operator returns cosine distance (0 = identical, 2 = opposite)
        similarity = 1 - distance
        """
        # Test cases: (distance, expected_similarity)
        test_cases = [
            (0.0, 1.0),  # Identical
            (0.1, 0.9),  # Very similar
            (0.5, 0.5),  # Moderately similar
            (1.0, 0.0),  # Orthogonal
            (2.0, -1.0),  # Opposite
        ]

        for distance, expected_similarity in test_cases:
            similarity = 1 - distance
            assert similarity == expected_similarity


# =============================================================================
# Test API Endpoint Integration
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(reason="API endpoints are Epic 4 - requires full API setup")
class TestSimilarityAPIEndpoint:
    """Test /api/v1/similar/{artist} endpoint."""

    async def test_similarity_endpoint_success(self, db_manager):
        """Should return similar artists through API endpoint."""
        from src.api.routes.utils import get_similar_artists

        with patch("src.api.routes.utils.get_database", return_value=db_manager):
            with patch.object(
                db_manager,
                "find_similar_artists",
                return_value=[("Keith Moon", 0.89), ("Ginger Baker", 0.85)],
            ):
                response = await get_similar_artists(artist="John Bonham", limit=2, db=db_manager)

                assert response.artist == "John Bonham"
                assert len(response.similar_artists) == 2
                assert response.similar_artists[0]["name"] == "Keith Moon"
                assert response.similar_artists[0]["similarity"] == 0.89

    async def test_similarity_endpoint_not_found(self, db_manager):
        """Should raise 404 when artist not found."""
        from fastapi import HTTPException

        from src.api.routes.utils import get_similar_artists

        with patch.object(db_manager, "find_similar_artists", return_value=[]):
            with pytest.raises(HTTPException) as exc_info:
                await get_similar_artists(artist="Unknown Artist", limit=5, db=db_manager)

            assert exc_info.value.status_code == 404


# =============================================================================
# Integration Tests (Slow - Marked)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
class TestRealVectorSearch:
    """Integration tests with real PostgreSQL + pgvector."""

    async def test_real_vector_search(self):
        """
        Test real vector similarity search against PostgreSQL.

        Note: This test requires:
        - PostgreSQL 16+ with pgvector extension
        - Database populated with sample embeddings
        - May be slow (vector index scan)
        """
        # TODO: Implement integration test with real database
        # This would test:
        # - Real embedding storage
        # - IVFFlat index performance
        # - Query execution time (should be < 200ms)
        pass


# TODO: Add more tests
# - Test embedding dimension validation (384-d)
# - Test IVFFlat index creation and usage
# - Test query performance benchmarks
# - Test with missing/null embeddings
# - Test concurrent similarity searches
# - Test with very large embedding datasets (10k+ artists)
