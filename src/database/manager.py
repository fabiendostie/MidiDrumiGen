"""
Database Manager for MidiDrumiGen v2.0

Handles all database operations including CRUD, vector similarity search,
Redis caching, and analytics queries.

Stories:
- E3.S1: Database Models (DONE - Phase 2)
- E3.S2: Database Manager (DONE - Phase 2)
- E3.S4: Vector Similarity Search (Sprint 1 - TODO)
- E3.S5: Redis Caching Layer (Sprint 1 - TODO)
- E3.S6: Generation History Analytics (Sprint 1 - TODO)
"""

import json
import logging
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from .models import Artist, GenerationHistory, StyleProfile

logger = logging.getLogger(__name__)


class ArtistNotFoundError(Exception):
    """Raised when an artist is not found in the database."""

    pass


class DatabaseError(Exception):
    """Raised when a database operation fails."""

    pass


class DatabaseManager:
    """
    Manages all database operations for MidiDrumiGen v2.0.

    Features:
        - CRUD operations for all models
        - Vector similarity search (E3.S4)
        - Redis caching layer (E3.S5)
        - Generation history analytics (E3.S6)
    """

    def __init__(
        self,
        database_url: str,
        redis_url: str | None = None,
        cache_ttl: int = 604800,  # 7 days default
    ):
        """
        Initialize Database Manager.

        Args:
            database_url: PostgreSQL connection string
            redis_url: Redis connection string (optional, for caching)
            cache_ttl: Cache time-to-live in seconds (default: 7 days)
        """
        # PostgreSQL setup
        self.engine = create_async_engine(database_url, echo=False)
        self.SessionLocal = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Redis setup (E3.S5)
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self._redis_client = None

    async def get_redis(self) -> aioredis.Redis | None:
        """
        Get or create Redis client.

        Returns:
            Redis client or None if Redis not configured
        """
        if not self.redis_url:
            return None

        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )

        return self._redis_client

    async def close(self):
        """Close database and Redis connections."""
        await self.engine.dispose()

        if self._redis_client:
            await self._redis_client.close()

    # =========================================================================
    # E3.S1 & E3.S2: Basic CRUD Operations (DONE - Phase 2)
    # =========================================================================

    async def get_or_create_artist(self, session: AsyncSession, artist_name: str) -> Artist:
        """
        Get existing artist or create new one.

        Args:
            session: Database session
            artist_name: Name of artist

        Returns:
            Artist object
        """
        # Try to find existing
        result = await session.execute(select(Artist).where(Artist.name == artist_name))
        artist = result.scalar_one_or_none()

        if not artist:
            # Create new
            artist = Artist(name=artist_name, research_status="pending", confidence_score=0.0)
            session.add(artist)
            await session.commit()
            await session.refresh(artist)

        return artist

    async def save_style_profile(
        self, session: AsyncSession, profile: StyleProfile
    ) -> StyleProfile:
        """
        Save or update StyleProfile.

        Args:
            session: Database session
            profile: StyleProfile object to save

        Returns:
            Saved StyleProfile object
        """
        session.add(profile)
        await session.commit()
        await session.refresh(profile)

        # Invalidate Redis cache (E3.S5)
        await self._invalidate_cache(profile.artist.name)

        return profile

    async def get_style_profile(self, artist_name: str) -> StyleProfile | None:
        """
        Get StyleProfile for artist with Redis caching.

        Story: E3.S5 - Redis Caching Layer

        Args:
            artist_name: Name of artist

        Returns:
            StyleProfile or None if not found

        Implementation Notes:
            1. Check Redis cache first (< 10ms if cached)
            2. If cache miss, query PostgreSQL
            3. Cache result in Redis with 7-day TTL
        """
        # Try Redis cache first (E3.S5)
        redis = await self.get_redis()
        if redis:
            cached = await redis.get(f"profile:{artist_name}")
            if cached:
                try:
                    logger.debug(f"Cache HIT for artist: {artist_name}")
                    # Deserialize from JSON
                    profile_data = json.loads(cached)

                    # Reconstruct StyleProfile object from cached data
                    profile = StyleProfile(
                        text_description=profile_data["text_description"],
                        quantitative_params=profile_data["quantitative_params"],
                        midi_templates_json=profile_data.get("midi_templates_json", []),
                        confidence_score=profile_data["confidence_score"],
                        sources_count=profile_data.get("sources_count", {}),
                        updated_at=datetime.fromisoformat(profile_data["updated_at"]),
                    )

                    # Create temporary Artist object for relationship
                    # (Avoids full DB query for artist data)
                    artist = Artist(name=artist_name)
                    profile.artist = artist

                    logger.info(
                        f"Retrieved profile for '{artist_name}' from cache "
                        f"(confidence: {profile.confidence_score:.2f})"
                    )
                    return profile

                except (KeyError, ValueError, json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Cache corruption detected for '{artist_name}': {e}. "
                        f"Invalidating cache and reloading from database."
                    )
                    # Invalidate corrupted cache
                    await self._invalidate_cache(artist_name)
                    # Fall through to database query

        # Cache miss - query database
        logger.debug(f"Cache MISS for artist: {artist_name}")

        async with self.SessionLocal() as session:
            result = await session.execute(
                select(StyleProfile)
                .join(Artist)
                .where(Artist.name == artist_name)
                .options(selectinload(StyleProfile.artist))
            )
            profile = result.scalar_one_or_none()

            if profile and redis:
                # Cache in Redis (E3.S5)
                await self._cache_profile(profile)

            return profile

    async def _cache_profile(self, profile: StyleProfile):
        """
        Cache StyleProfile in Redis.

        Story: E3.S5 - Redis Caching Layer

        Args:
            profile: StyleProfile to cache
        """
        redis = await self.get_redis()
        if not redis:
            return

        # Serialize to JSON
        profile_data = {
            "artist_name": profile.artist.name,
            "text_description": profile.text_description,
            "quantitative_params": profile.quantitative_params,
            "midi_templates_json": profile.midi_templates_json,
            "confidence_score": profile.confidence_score,
            "sources_count": profile.sources_count,
            "updated_at": profile.updated_at.isoformat(),
        }

        cache_key = f"profile:{profile.artist.name}"
        await redis.setex(cache_key, self.cache_ttl, json.dumps(profile_data))

        logger.debug(f"Cached profile for {profile.artist.name} " f"(TTL: {self.cache_ttl}s)")

    async def _invalidate_cache(self, artist_name: str):
        """
        Invalidate Redis cache for artist.

        Story: E3.S5 - Redis Caching Layer

        Args:
            artist_name: Name of artist to invalidate
        """
        redis = await self.get_redis()
        if not redis:
            return

        cache_key = f"profile:{artist_name}"
        await redis.delete(cache_key)
        logger.debug(f"Invalidated cache for: {artist_name}")

    # =========================================================================
    # E3.S4: Vector Similarity Search (Sprint 1 - TODO)
    # =========================================================================

    async def find_similar_artists(
        self, artist_name: str, limit: int = 5
    ) -> list[tuple[str, float]]:
        """
        Find artists similar to given artist using vector similarity search.

        Story: E3.S4 - Vector Similarity Search

        Args:
            artist_name: Name of artist to find similar artists for
            limit: Maximum number of similar artists to return (default 5, max 20)

        Returns:
            List of tuples (artist_name, similarity_score)
            Similarity score is between 0 and 1 (higher = more similar)

        Raises:
            ArtistNotFoundError: If the query artist doesn't exist in the database

        Implementation Notes:
            - Uses pgvector cosine distance: <=>
            - Query should complete in < 200ms using IVFFlat index
            - Excludes the query artist from results
            - Filters out profiles with NULL embeddings
        """
        # Validate and cap limit
        if limit <= 0:
            return []
        limit = min(limit, 20)  # Cap at 20 to prevent resource exhaustion

        try:
            async with self.SessionLocal() as session:
                # First check if artist exists
                artist_result = await session.execute(
                    select(Artist).where(Artist.name == artist_name)
                )
                artist = artist_result.scalar_one_or_none()

                if not artist:
                    raise ArtistNotFoundError(f"Artist '{artist_name}' not found")

                # Get query artist's StyleProfile with embedding
                result = await session.execute(
                    select(StyleProfile).where(StyleProfile.artist_id == artist.id)
                )
                query_profile = result.scalar_one_or_none()

                if not query_profile:
                    logger.info(f"No StyleProfile found for artist: {artist_name}")
                    return []

                if query_profile.embedding is None or len(query_profile.embedding) == 0:
                    logger.info(f"No embedding found for artist: {artist_name}")
                    return []

                # Vector similarity search using pgvector
                # Note: <=> operator calculates cosine distance
                # Smaller distance = more similar
                # similarity = 1 - distance
                query = text(
                    """
                SELECT
                    a.name,
                    1 - (sp.embedding <=> :query_embedding) as similarity
                FROM style_profiles sp
                JOIN artists a ON sp.artist_id = a.id
                WHERE a.name != :artist_name
                  AND sp.embedding IS NOT NULL
                ORDER BY sp.embedding <=> :query_embedding
                LIMIT :limit
                """
                )

                # Convert embedding to string format for pgvector
                embedding = query_profile.embedding
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                # pgvector expects string format like '[1.0, 2.0, 3.0]'
                embedding_str = str(embedding)

                result = await session.execute(
                    query,
                    {
                        "query_embedding": embedding_str,
                        "artist_name": artist_name,
                        "limit": limit,
                    },
                )

                similar_artists = [(row[0], float(row[1])) for row in result.fetchall()]

                logger.info(f"Found {len(similar_artists)} similar artists to {artist_name}")

                return similar_artists

        except ArtistNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Database error during similarity search: {e}")
            raise DatabaseError(f"Failed to search for similar artists: {e}") from e

    # =========================================================================
    # E3.S6: Generation History Analytics (Sprint 1 - TODO)
    # =========================================================================

    async def save_generation_history(
        self, session: AsyncSession, history: GenerationHistory
    ) -> GenerationHistory:
        """
        Save generation history record.

        Args:
            session: Database session
            history: GenerationHistory object

        Returns:
            Saved GenerationHistory object
        """
        session.add(history)
        await session.commit()
        await session.refresh(history)
        return history

    async def get_generation_stats(self) -> dict[str, Any]:
        """
        Get overall generation statistics.

        Story: E3.S6 - Generation History Analytics
        Endpoint: GET /api/v1/stats

        Returns:
            Dictionary with statistics:
            {
                'total_generations': int,
                'avg_generation_time_ms': float,
                'total_cost_usd': float,
                'provider_usage': {'anthropic': int, 'google': int, ...},
                'avg_cost_per_generation': float
            }
        """
        async with self.SessionLocal() as session:
            # Total generations
            total_result = await session.execute(select(func.count(GenerationHistory.id)))
            total_generations = total_result.scalar()

            if total_generations == 0:
                return {
                    "total_generations": 0,
                    "avg_generation_time_ms": 0,
                    "total_cost_usd": 0.0,
                    "provider_usage": {},
                    "avg_cost_per_generation": 0.0,
                }

            # Average generation time
            avg_time_result = await session.execute(
                select(func.avg(GenerationHistory.generation_time_ms))
            )
            avg_time = avg_time_result.scalar() or 0

            # Total cost
            total_cost_result = await session.execute(select(func.sum(GenerationHistory.cost_usd)))
            total_cost = total_cost_result.scalar() or 0.0

            # Provider usage count
            provider_result = await session.execute(
                select(GenerationHistory.provider_used, func.count(GenerationHistory.id)).group_by(
                    GenerationHistory.provider_used
                )
            )
            provider_usage = {row[0]: row[1] for row in provider_result.fetchall()}

            return {
                "total_generations": total_generations,
                "avg_generation_time_ms": float(avg_time),
                "total_cost_usd": float(total_cost),
                "provider_usage": provider_usage,
                "avg_cost_per_generation": float(total_cost / total_generations),
            }

    async def get_artist_generation_history(
        self, artist_name: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get generation history for specific artist.

        Story: E3.S6 - Generation History Analytics
        Endpoint: GET /api/v1/artists/{artist}/history

        Args:
            artist_name: Name of artist
            limit: Maximum number of records to return

        Returns:
            List of generation history records
        """
        async with self.SessionLocal() as session:
            result = await session.execute(
                select(GenerationHistory)
                .join(Artist)
                .where(Artist.name == artist_name)
                .order_by(desc(GenerationHistory.created_at))
                .limit(limit)
                .options(selectinload(GenerationHistory.artist))
            )

            history_records = result.scalars().all()

            return [
                {
                    "provider_used": record.provider_used,
                    "generation_time_ms": record.generation_time_ms,
                    "tokens_used": record.tokens_used,
                    "cost_usd": record.cost_usd,
                    "user_params": record.user_params,
                    "output_files": record.output_files,
                    "created_at": record.created_at.isoformat(),
                }
                for record in history_records
            ]


# TODO: Implement unit tests
# - Test CRUD operations
# - Test Redis caching (cache hit/miss)
# - Test cache invalidation
# - Test vector similarity search
# - Test generation statistics aggregation
# - Test with mock database and Redis
