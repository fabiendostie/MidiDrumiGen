"""
Utility API Endpoints for MidiDrumiGen v2.0

Provides statistics, artist listing, and task status endpoints.

Story: E3.S6 - Generation History Analytics
Epic: E3 - Database & Caching
Priority: MEDIUM
Story Points: 2
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...database.manager import DatabaseManager

router = APIRouter(prefix="/api/v1", tags=["utilities"])


async def get_database() -> DatabaseManager:
    """
    FastAPI dependency to get database manager instance.

    Returns:
        DatabaseManager instance

    Note: In production, this should be a singleton
    managed by the application lifespan context.
    """
    # TODO: Get from app state or dependency injection container
    # For now, this is a placeholder
    import os

    db = DatabaseManager(database_url=os.getenv("DATABASE_URL"), redis_url=os.getenv("REDIS_URL"))
    try:
        yield db
    finally:
        await db.close()


# =============================================================================
# Response Models
# =============================================================================


class GenerationStatsResponse(BaseModel):
    """
    Response model for GET /api/v1/stats

    Story: E3.S6 - Generation History Analytics
    """

    total_generations: int = Field(..., description="Total number of generations completed")
    avg_generation_time_ms: float = Field(
        ..., description="Average generation time in milliseconds"
    )
    total_cost_usd: float = Field(..., description="Total cost across all generations")
    provider_usage: dict[str, int] = Field(..., description="Generation count per LLM provider")
    avg_cost_per_generation: float = Field(..., description="Average cost per generation")


class GenerationHistoryRecord(BaseModel):
    """Individual generation history record."""

    provider_used: str
    generation_time_ms: int
    tokens_used: int | None
    cost_usd: float | None
    user_params: dict[str, Any]
    output_files: list[str] | None
    created_at: str


class ArtistHistoryResponse(BaseModel):
    """
    Response model for GET /api/v1/artists/{artist}/history

    Story: E3.S6 - Generation History Analytics
    """

    artist: str
    total_generations: int
    history: list[GenerationHistoryRecord]


class ArtistListResponse(BaseModel):
    """Response model for GET /api/v1/artists"""

    total: int
    cached: int
    researching: int
    failed: int
    recent: list[dict[str, Any]]


class SimilarArtistsResponse(BaseModel):
    """
    Response model for GET /api/v1/similar/{artist}

    Story: E3.S4 - Vector Similarity Search
    """

    artist: str
    similar_artists: list[dict[str, Any]]


# =============================================================================
# E3.S6: Generation History Analytics Endpoints
# =============================================================================


@router.get("/stats", response_model=GenerationStatsResponse)
async def get_generation_statistics(db: DatabaseManager = Depends(get_database)):
    """
    Get overall generation statistics.

    Story: E3.S6 - Generation History Analytics

    Returns:
        Statistics including:
        - Total generations
        - Average generation time
        - Total cost
        - Provider usage breakdown
        - Average cost per generation

    Example Response:
        {
            "total_generations": 1523,
            "avg_generation_time_ms": 1847.5,
            "total_cost_usd": 12.34,
            "provider_usage": {
                "anthropic": 980,
                "google": 450,
                "openai": 93
            },
            "avg_cost_per_generation": 0.0081
        }
    """
    try:
        stats = await db.get_generation_stats()
        return GenerationStatsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve generation statistics: {str(e)}"
        ) from e


@router.get("/artists/{artist}/history", response_model=ArtistHistoryResponse)
async def get_artist_generation_history(
    artist: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    db: DatabaseManager = Depends(get_database),
):
    """
    Get generation history for specific artist.

    Story: E3.S6 - Generation History Analytics

    Args:
        artist: Name of artist
        limit: Maximum number of records to return (1-1000, default: 100)

    Returns:
        List of generation history records for the artist

    Example Response:
        {
            "artist": "John Bonham",
            "total_generations": 45,
            "history": [
                {
                    "provider_used": "anthropic",
                    "generation_time_ms": 1847,
                    "tokens_used": 2340,
                    "cost_usd": 0.0123,
                    "user_params": {"bars": 4, "tempo": 120},
                    "output_files": ["path/to/file1.mid", "path/to/file2.mid"],
                    "created_at": "2025-11-18T10:30:00Z"
                },
                ...
            ]
        }
    """
    try:
        history = await db.get_artist_generation_history(artist, limit=limit)

        if not history:
            raise HTTPException(
                status_code=404, detail=f"No generation history found for artist: {artist}"
            )

        return ArtistHistoryResponse(
            artist=artist,
            total_generations=len(history),
            history=[GenerationHistoryRecord(**record) for record in history],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve artist history: {str(e)}"
        ) from e


# =============================================================================
# E3.S4: Vector Similarity Search Endpoint
# =============================================================================


@router.get("/similar/{artist}", response_model=SimilarArtistsResponse)
async def get_similar_artists(
    artist: str,
    limit: int = Query(5, ge=1, le=20, description="Maximum similar artists to return"),
    db: DatabaseManager = Depends(get_database),
):
    """
    Find artists similar to given artist using vector similarity search.

    Story: E3.S4 - Vector Similarity Search

    Args:
        artist: Name of artist to find similar artists for
        limit: Maximum similar artists to return (1-20, default: 5)

    Returns:
        List of similar artists with similarity scores

    Example Response:
        {
            "artist": "John Bonham",
            "similar_artists": [
                {"name": "Keith Moon", "similarity": 0.91},
                {"name": "Ginger Baker", "similarity": 0.85},
                {"name": "Neil Peart", "similarity": 0.78}
            ]
        }
    """
    try:
        similar_artists = await db.find_similar_artists(artist, limit=limit)

        if not similar_artists:
            raise HTTPException(
                status_code=404, detail=f"Artist not found or has no embedding: {artist}"
            )

        return SimilarArtistsResponse(
            artist=artist,
            similar_artists=[
                {"name": name, "similarity": round(score, 3)} for name, score in similar_artists
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to find similar artists: {str(e)}"
        ) from e


# =============================================================================
# E4.S4: Artist Listing Endpoint
# =============================================================================


@router.get("/artists", response_model=ArtistListResponse)
async def list_artists(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=200, description="Artists per page"),
    search: str | None = Query(None, description="Search by artist name"),
    db: DatabaseManager = Depends(get_database),
):
    """
    List all cached artists with pagination and search.

    Story: E4.S4 - Utility Endpoints

    Args:
        page: Page number (1-indexed)
        limit: Artists per page (1-200, default: 50)
        search: Optional name filter

    Returns:
        Paginated list of artists with status counts

    Example Response:
        {
            "total": 1523,
            "cached": 1500,
            "researching": 15,
            "failed": 8,
            "recent": [
                {
                    "name": "John Bonham",
                    "confidence": 0.89,
                    "last_updated": "2025-11-17T10:30:00Z"
                },
                ...
            ]
        }
    """
    # TODO: Implement artist listing with pagination
    # This will be completed in Epic 4
    raise HTTPException(status_code=501, detail="Artist listing not yet implemented (Epic 4)")


# TODO: Implement unit tests
# - Test stats aggregation with mock data
# - Test artist history retrieval
# - Test vector similarity search
# - Test pagination
# - Test error handling (404, 500)
# - Test with FastAPI TestClient
