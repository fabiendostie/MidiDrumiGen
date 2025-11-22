"""Producer style profile caching with Redis and file fallback."""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import redis, but make it optional
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using file-based caching only")


class ProducerStyleCache:
    """
    Cache for producer style profiles with Redis primary + file fallback.

    Caches producer research results to avoid repeated web scraping and LLM calls.
    30-day TTL for cached profiles.

    Example:
        >>> cache = ProducerStyleCache()
        >>> profile = cache.get("timbaland")
        >>> if profile is None:
        ...     profile = research_producer("timbaland")
        ...     cache.set("timbaland", profile)
    """

    def __init__(
        self,
        redis_url: str | None = None,
        cache_dir: Path | str = "data/producer_cache",
        ttl_days: int = 30,
    ):
        """
        Initialize cache.

        Args:
            redis_url: Redis connection URL (optional, e.g., "redis://localhost:6379/0")
            cache_dir: Directory for file-based cache fallback
            ttl_days: Time-to-live in days for cached profiles
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)

        # Try to connect to Redis
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {redis_url}")
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using file cache only.")
                self.redis_client = None
        elif not REDIS_AVAILABLE:
            logger.info("Redis not installed, using file-based cache only")
        else:
            logger.info("No Redis URL provided, using file-based cache only")

    def _normalize_name(self, producer_name: str) -> str:
        """
        Normalize producer name for consistent caching.

        Args:
            producer_name: Producer name (any format)

        Returns:
            Normalized name (lowercase, underscores, no special chars)

        Example:
            >>> cache._normalize_name("J. Dilla")
            'j_dilla'
            >>> cache._normalize_name("Metro Boomin'")
            'metro_boomin'
        """
        # Convert to lowercase
        normalized = producer_name.lower()

        # Replace spaces and special chars with underscores
        import re

        normalized = re.sub(r"[^a-z0-9_]", "_", normalized)

        # Remove consecutive underscores
        normalized = re.sub(r"_+", "_", normalized)

        # Remove leading/trailing underscores
        normalized = normalized.strip("_")

        return normalized

    def _get_cache_key(self, producer_name: str) -> str:
        """Get Redis cache key for producer."""
        normalized = self._normalize_name(producer_name)
        return f"producer:style:{normalized}"

    def _get_file_path(self, producer_name: str) -> Path:
        """Get file path for cached profile."""
        normalized = self._normalize_name(producer_name)
        return self.cache_dir / f"{normalized}.json"

    def get(self, producer_name: str) -> dict[str, Any] | None:
        """
        Get cached producer profile.

        Checks Redis first, then file cache.

        Args:
            producer_name: Producer name

        Returns:
            Cached profile dict or None if not found/expired

        Example:
            >>> cache = ProducerStyleCache()
            >>> profile = cache.get("timbaland")
            >>> if profile:
            ...     print(f"Found cached profile: {profile['style_params']}")
        """
        normalized = self._normalize_name(producer_name)

        # Try Redis first
        if self.redis_client:
            try:
                key = self._get_cache_key(producer_name)
                data = self.redis_client.get(key)
                if data:
                    logger.debug(f"Cache HIT (Redis): {normalized}")
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        # Fallback to file cache
        file_path = self._get_file_path(producer_name)
        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                # Check if expired
                cached_at_str = data.get("cached_at", "1970-01-01T00:00:00Z").replace("Z", "+00:00")
                cached_at = datetime.fromisoformat(cached_at_str)
                now = datetime.now(cached_at.tzinfo) if cached_at.tzinfo else datetime.now(UTC)
                if now - cached_at < self.ttl:
                    logger.debug(f"Cache HIT (file): {normalized}")
                    return data
                else:
                    logger.debug(f"Cache EXPIRED (file): {normalized}")
                    file_path.unlink()  # Delete expired cache
                    return None

            except Exception as e:
                logger.warning(f"File cache read error: {e}")

        logger.debug(f"Cache MISS: {normalized}")
        return None

    def set(self, producer_name: str, profile: dict[str, Any]) -> None:
        """
        Cache producer profile.

        Writes to both Redis and file cache.

        Args:
            producer_name: Producer name
            profile: Producer profile dict

        Example:
            >>> cache = ProducerStyleCache()
            >>> profile = {
            ...     'producer_name': 'Timbaland',
            ...     'style_params': {...},
            ...     'cached_at': datetime.now(timezone.utc).isoformat() + 'Z'
            ... }
            >>> cache.set("timbaland", profile)
        """
        normalized = self._normalize_name(producer_name)

        # Ensure cached_at timestamp
        if "cached_at" not in profile:
            profile["cached_at"] = datetime.now(UTC).isoformat() + "Z"

        # Ensure normalized name in profile
        if "normalized_name" not in profile:
            profile["normalized_name"] = normalized

        # Write to Redis
        if self.redis_client:
            try:
                key = self._get_cache_key(producer_name)
                self.redis_client.setex(key, int(self.ttl.total_seconds()), json.dumps(profile))
                logger.info(f"Cached to Redis: {normalized}")
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

        # Write to file cache (always, as fallback)
        file_path = self._get_file_path(producer_name)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
            logger.info(f"Cached to file: {file_path}")
        except Exception as e:
            logger.error(f"File cache write error: {e}")

    def delete(self, producer_name: str) -> None:
        """
        Delete cached profile.

        Args:
            producer_name: Producer name
        """
        normalized = self._normalize_name(producer_name)

        # Delete from Redis
        if self.redis_client:
            try:
                key = self._get_cache_key(producer_name)
                self.redis_client.delete(key)
                logger.info(f"Deleted from Redis: {normalized}")
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")

        # Delete from file cache
        file_path = self._get_file_path(producer_name)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted from file cache: {normalized}")

    def clear_all(self) -> None:
        """Clear all cached profiles."""
        # Clear Redis
        if self.redis_client:
            try:
                pattern = "producer:style:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
                logger.info("Cleared all Redis cache")
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")

        # Clear file cache
        for file_path in self.cache_dir.glob("*.json"):
            file_path.unlink()
        logger.info("Cleared all file cache")

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (file count, Redis keys, etc.)
        """
        stats = {
            "backend": "redis+file" if self.redis_client else "file_only",
            "file_cache_count": len(list(self.cache_dir.glob("*.json"))),
            "cache_dir": str(self.cache_dir),
            "ttl_days": self.ttl.days,
        }

        if self.redis_client:
            try:
                pattern = "producer:style:*"
                redis_keys = list(self.redis_client.scan_iter(match=pattern))
                stats["redis_cache_count"] = len(redis_keys)
                stats["redis_connected"] = True
            except Exception:
                stats["redis_connected"] = False

        return stats
