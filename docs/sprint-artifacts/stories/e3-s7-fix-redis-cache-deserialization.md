# Story E3.S7: Fix Redis Cache Deserialization

**Epic:** Epic 3 - Database & Caching
**Story ID:** e3-s7-fix-redis-cache-deserialization
**Status:** Backlog
**Priority:** High
**Estimate:** 2 hours

---

## User Story

**As a** system
**I want** to properly deserialize StyleProfiles from Redis cache
**So that** cached profile retrieval is truly < 10ms (currently falls back to PostgreSQL)

---

## Context

The Redis caching layer (E3.S5) was partially implemented in Phase 2. The code structure exists but has a critical TODO at `src/database/manager.py:184`:

```python
# Current implementation (BROKEN)
if cached:
    logger.debug(f"Cache HIT for artist: {artist_name}")
    # Deserialize from JSON
    profile_data = json.loads(cached)
    # TODO: Convert back to StyleProfile object
    # For now, return None and fall through to DB query
    pass  # <-- This causes cache hits to fall through to DB!
```

This means even when profiles are cached, we're still querying PostgreSQL, defeating the purpose of caching.

---

## Acceptance Criteria

**AC-1: Deserialize Cached StyleProfile**
- WHEN a cache hit occurs in `get_style_profile()`
- THEN the cached JSON is deserialized into a StyleProfile object
- AND the StyleProfile object is returned without querying PostgreSQL
- AND retrieval time is < 10ms

**AC-2: Reconstruct Artist Relationship**
- WHEN deserializing from cache
- THEN create a temporary Artist object with name (no full DB query)
- AND attach it to StyleProfile.artist relationship
- AND downstream code can access profile.artist.name without errors

**AC-3: Handle Cache Corruption Gracefully**
- WHEN cached JSON is malformed or missing fields
- THEN log warning and fall back to PostgreSQL query
- AND cache is invalidated (deleted)
- AND fresh profile is re-cached

---

## Implementation Details

**File:** `src/database/manager.py`

**Approach:**

```python
async def get_style_profile(self, artist_name: str) -> Optional[StyleProfile]:
    """Get StyleProfile with working Redis cache."""

    # Try Redis cache
    redis = await self.get_redis()
    if redis:
        cached = await redis.get(f"profile:{artist_name}")
        if cached:
            try:
                logger.debug(f"Cache HIT for artist: {artist_name}")
                profile_data = json.loads(cached)

                # Reconstruct StyleProfile from cached data
                profile = StyleProfile(
                    text_description=profile_data['text_description'],
                    quantitative_params=profile_data['quantitative_params'],
                    midi_templates_json=profile_data.get('midi_templates_json', []),
                    confidence_score=profile_data['confidence_score'],
                    sources_count=profile_data.get('sources_count', {}),
                    updated_at=datetime.fromisoformat(profile_data['updated_at'])
                )

                # Create temporary Artist object for relationship
                # (We don't need full artist data from DB for most use cases)
                artist = Artist(name=artist_name)
                profile.artist = artist

                return profile

            except (KeyError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Cache corruption for {artist_name}: {e}. "
                    f"Invalidating and reloading from DB."
                )
                await self._invalidate_cache(artist_name)
                # Fall through to DB query

    # Cache miss or corruption - query database
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
            await self._cache_profile(profile)

        return profile
```

**Note:** We might also need to cache the embedding vector if it's used downstream. Check if embedding needs to be serialized.

---

## Testing Strategy

**Unit Tests** (`tests/unit/test_redis_caching.py`):

```python
@pytest.mark.asyncio
async def test_cache_hit_deserializes_profile():
    """Test that cache hit returns StyleProfile without DB query."""
    # Setup: Mock Redis with cached profile JSON
    cached_data = {
        'artist_name': 'J Dilla',
        'text_description': 'Known for swing and ghost notes',
        'quantitative_params': {'tempo_avg': 95, 'swing_percent': 62},
        'confidence_score': 0.85,
        'sources_count': {'papers': 5, 'articles': 12},
        'updated_at': '2025-11-18T10:00:00'
    }

    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps(cached_data)

    db_manager = DatabaseManager(DATABASE_URL, redis_url=REDIS_URL)
    db_manager._redis_client = mock_redis

    # Execute
    profile = await db_manager.get_style_profile('J Dilla')

    # Assert
    assert profile is not None
    assert profile.text_description == 'Known for swing and ghost notes'
    assert profile.artist.name == 'J Dilla'
    assert profile.confidence_score == 0.85

    # Verify no DB query was made (only Redis.get called)
    mock_redis.get.assert_called_once_with('profile:J Dilla')


@pytest.mark.asyncio
async def test_cache_corruption_falls_back_to_db():
    """Test that corrupted cache invalidates and queries DB."""
    # Setup: Mock Redis with invalid JSON
    mock_redis = AsyncMock()
    mock_redis.get.return_value = "INVALID JSON{"

    # ... test implementation

    # Assert cache was invalidated
    mock_redis.delete.assert_called_once_with('profile:J Dilla')
```

**Integration Test** (`tests/integration/test_cache_integration.py`):

```python
@pytest.mark.asyncio
async def test_cache_hit_performance():
    """Test that cache hit retrieval is < 10ms."""
    # Setup: Insert profile into DB and cache it
    # ...

    # Measure cache hit time
    start = time.time()
    profile = await db_manager.get_style_profile('J Dilla')
    elapsed_ms = (time.time() - start) * 1000

    assert profile is not None
    assert elapsed_ms < 10, f"Cache hit took {elapsed_ms}ms (expected < 10ms)"
```

---

## Definition of Done

- [ ] `get_style_profile()` deserializes cached profiles correctly
- [ ] Cache hit does NOT query PostgreSQL
- [ ] Cache hit retrieval time < 10ms (integration test)
- [ ] Artist relationship properly reconstructed from cache
- [ ] Cache corruption handled gracefully (invalidate + fallback)
- [ ] Unit tests pass (cache hit, cache miss, corruption)
- [ ] Integration test with real Redis passes
- [ ] Performance benchmark confirms < 10ms retrieval
- [ ] Code reviewed and merged
- [ ] TODO comment removed from manager.py:184

---

## Related Stories

- **E3.S2:** Database Manager (parent story, currently in-progress)
- **E3.S5:** Redis Caching Layer (this fixes the incomplete implementation)

---

## Notes

- This is a small fix but critical for achieving the < 2 min cached generation target
- Without this fix, all generations query PostgreSQL even when cached
- May need to serialize embedding vector if used in cached retrieval path
