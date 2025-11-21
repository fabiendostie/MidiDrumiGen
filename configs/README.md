# Configuration Files

This directory contains configuration files for MidiDrumiGen v2.0.

## Files

### `v2_config.yaml`
**Main configuration file for v2.0 architecture.**

Contains settings for:
- LLM providers (Claude, Gemini, OpenAI)
- Research pipeline (collectors, timeouts)
- Database (PostgreSQL + pgvector)
- Redis & Celery (queues, task routing)
- Generation (humanization, validation)
- Logging (JSON/text formatters)
- Performance (workers, caching)
- Monitoring (metrics, Sentry)

Usage:
```python
from src.utils.config import load_config

config = load_config("configs/v2_config.yaml")
```

### `redis.py`
**Legacy Redis configuration** (kept for backward compatibility).

For v2.0, use `v2_config.yaml` instead, or reference this for Celery configuration:
```python
from configs.redis import REDIS_CONFIG

app = Celery('mididrumigen')
app.config_from_object(REDIS_CONFIG)
```

## Environment Variables

Configuration files support environment variable interpolation using `${VAR_NAME}` syntax:

```yaml
database:
  url: ${DATABASE_URL}
redis:
  broker_url: ${REDIS_URL}
llm:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

Ensure these are set in your `.env` file.

## v2.0 vs v1.x

**v1.x (Removed):**
- ❌ `base.yaml` - Training configuration
- ❌ `remi_tokenizer.json` - Tokenizer config

**v2.0 (New):**
- ✅ `v2_config.yaml` - Comprehensive v2.0 settings
- ✅ Environment-based configuration
- ✅ Multi-provider LLM settings
- ✅ Research pipeline configuration

## Configuration Priority

1. **Environment variables** (highest priority)
2. **`v2_config.yaml`** (default settings)
3. **Code defaults** (fallback)

Example:
```bash
# Override in environment
export PRIMARY_LLM_PROVIDER=google  # Use Gemini instead of Claude

# Or in .env file
PRIMARY_LLM_PROVIDER=google
```

## Development vs Production

**Development:**
```yaml
logging:
  level: DEBUG
database:
  echo: true  # Log SQL queries
monitoring:
  enable_prometheus: false
```

**Production:**
```yaml
logging:
  level: INFO
database:
  echo: false
monitoring:
  enable_prometheus: true
  enable_sentry: true
```

Create separate config files:
- `configs/v2_config.yaml` (development)
- `configs/v2_config.prod.yaml` (production)

Load with:
```python
import os
config_file = os.getenv('CONFIG_FILE', 'configs/v2_config.yaml')
config = load_config(config_file)
```
