# Epic Technical Specification: LLM Generation Engine

Date: 2025-11-19
Author: Fabz
Epic ID: 2
Status: Draft

---

## Overview

Epic 2 implements the core LLM Generation Engine for MidiDrumiGen v2.0, enabling authentic drum pattern generation through state-of-the-art language models. The engine uses a multi-provider architecture with automatic fallback (Claude 3.5 Sonnet → Gemini 2.5/3 → ChatGPT 5.1) to ensure 98%+ success rate. The system generates MIDI patterns by constructing style-aware prompts from StyleProfile data and validating structured JSON outputs that conform to General MIDI drum specifications.

This epic transforms artist research data (Epic 1 output) into playable MIDI patterns, serving as the creative core of the v2.0 system. Unlike v1.x which required trained models, this approach leverages LLM reasoning to understand and reproduce drumming styles from natural language descriptions and quantitative parameters.

## Objectives and Scope

**In Scope:**
- Unified LLM provider interface (BaseLLMProvider abstract class)
- Three provider implementations: Anthropic (Claude), Google (Gemini), OpenAI (ChatGPT)
- LLM Provider Manager with automatic failover logic
- Prompt engineering system (system + user prompt templates)
- Hybrid generation coordinator (LLM → Template fallback)
- Template-based generation (fallback for LLM failures)
- MIDI JSON output validation and parsing
- Cost tracking per provider
- Provider performance metrics (success rate, latency, cost)

**Out of Scope:**
- MIDI humanization (Epic 5)
- MIDI file export (Epic 5)
- API endpoints (Epic 4)
- Real-time generation during playback (v3.0)
- Local LLM support (v2.3.0)
- Style blending (v2.3.0)

**Success Criteria:**
- 98%+ generation success rate across all providers
- < 30 seconds average LLM API call latency
- < 2 minutes total generation time (4-8 variations)
- Valid MIDI JSON output from LLMs (strict schema compliance)
- Automatic fallback triggers within 5 seconds of primary failure
- Cost tracking accurate to $0.01 USD

## System Architecture Alignment

This epic aligns with the **Generation Layer** defined in ARCHITECTURE.md Section 2.5:

**Referenced Components:**
- LLM Provider Manager (`src/generation/providers/manager.py`)
- Individual providers (`src/generation/providers/anthropic.py`, `google.py`, `openai.py`)
- Prompt Builder (`src/generation/prompt_builder.py`)
- Template Generator (`src/generation/template_generator.py`)
- Hybrid Coordinator (`src/generation/hybrid_coordinator.py`)

**Architectural Constraints:**
- All LLM calls must be async (asyncio/aiohttp)
- Providers implement `BaseLLMProvider` interface
- System MUST NOT fail if one provider is down (fallback required)
- Output format MUST match MIDI JSON schema (strict validation)
- Cost tracking MUST log every API call for billing analysis

**Integration Points:**
- **Input:** StyleProfile from Epic 3 (Database & Caching layer)
- **Output:** MIDI JSON (dict) to Epic 5 (MIDI Export & Humanization)
- **Dependencies:** `anthropic`, `google-generativeai`, `openai` Python SDKs
- **Error Handling:** All provider errors trigger fallback, never crash

## Detailed Design

### Services and Modules

| Module | Responsibility | Input | Output | Owner |
|--------|---------------|-------|--------|-------|
| `BaseLLMProvider` | Abstract base class defining provider interface | N/A | N/A | Core |
| `AnthropicProvider` | Claude 3.5 Sonnet implementation (primary) | `prompt: str`, `system_prompt: str` | `dict` (MIDI JSON) | Generation |
| `GoogleProvider` | Gemini 2.5/3 implementation (secondary) | `prompt: str`, `system_prompt: str` | `dict` (MIDI JSON) | Generation |
| `OpenAIProvider` | ChatGPT 5.1 implementation (tertiary) | `prompt: str`, `system_prompt: str` | `dict` (MIDI JSON) | Generation |
| `LLMProviderManager` | Multi-provider coordination with fallback | `StyleProfile`, `GenerationParams`, `provider: Optional[str]` | `dict` (MIDI JSON + metadata) | Generation |
| `PromptBuilder` | Constructs system + user prompts | `StyleProfile`, `GenerationParams` | `system_prompt: str`, `user_prompt: str` | Generation |
| `TemplateGenerator` | MIDI pattern generation from templates (fallback) | `StyleProfile`, `GenerationParams` | `dict` (MIDI JSON) | Generation |
| `HybridCoordinator` | Orchestrates LLM → Template fallback chain | `StyleProfile`, `GenerationParams` | `List[Path]` (MIDI files) | Generation |

### Data Models and Contracts

**StyleProfile (Input from Epic 3):**
```python
@dataclass
class StyleProfile:
    artist_name: str
    text_description: str  # For LLM prompts
    quantitative_params: dict  # tempo, swing, velocities, etc.
    midi_templates: List[str]  # File paths for template fallback
    embedding: np.ndarray  # Vector (not used in Epic 2)
    confidence_score: float  # 0.0-1.0
    sources_count: dict  # {'papers': 5, 'articles': 12, ...}
```

**GenerationParams (Input):**
```python
@dataclass
class GenerationParams:
    bars: int  # 1-16
    tempo: int  # 40-300 BPM
    time_signature: Tuple[int, int]  # (4, 4), (3, 4), etc.
    variations: int  # 1-8
    provider: Optional[str] = "auto"  # "anthropic", "google", "openai", "auto"
    humanize: bool = True  # Apply humanization (Epic 5)
```

**MIDI JSON Output Schema:**
```python
{
  "notes": [
    {
      "pitch": 36,       # MIDI note number (35-81 for drums)
      "velocity": 90,    # 1-127
      "time": 0,         # Position in ticks (480 per quarter note)
      "duration": 120    # Note length in ticks
    },
    # ... more notes
  ],
  "tempo": 120,
  "time_signature": [4, 4],
  "total_bars": 4
}
```

**GenerationResult (Output):**
```python
@dataclass
class GenerationResult:
    status: str  # "success", "failed"
    midi_json: Optional[dict]  # MIDI JSON if success
    provider_used: str  # "anthropic", "google", "openai", "template"
    generation_time_ms: int
    cost_usd: float
    error: Optional[str] = None
```

### APIs and Interfaces

**BaseLLMProvider Interface:**
```python
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (e.g., 'anthropic', 'openai')."""
        pass

    @property
    @abstractmethod
    def cost_per_1k_tokens(self) -> dict:
        """Return cost structure: {'input': float, 'output': float}."""
        pass

    @abstractmethod
    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.8
    ) -> dict:
        """
        Generate MIDI pattern as JSON dict.

        Raises:
            LLMProviderError: If API call fails or output is invalid.
        """
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Check if API key is valid."""
        pass

    def record_success(self, tokens_used: int, cost_usd: float):
        """Track successful generation."""
        self.total_requests += 1
        self.tokens_used += tokens_used
        self.cost_usd += cost_usd

    def record_failure(self):
        """Track failed generation."""
        self.total_requests += 1
        self.total_failures += 1

    @property
    def success_rate(self) -> float:
        """Calculate provider success rate."""
        if self.total_requests == 0:
            return 0.0
        return 1.0 - (self.total_failures / self.total_requests)
```

**LLMProviderManager Public API:**
```python
class LLMProviderManager:
    async def generate(
        self,
        style_profile: StyleProfile,
        params: GenerationParams,
        provider_name: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate MIDI pattern with automatic fallback.

        Args:
            style_profile: Artist style data
            params: Generation parameters
            provider_name: Specific provider or "auto"

        Returns:
            GenerationResult with MIDI JSON and metadata

        Raises:
            GenerationError: If all providers fail
        """
```

**PromptBuilder Public API:**
```python
class PromptBuilder:
    SYSTEM_PROMPT: str = """You are an expert AI drummer..."""

    def build_user_prompt(
        self,
        style_profile: StyleProfile,
        bars: int,
        tempo: int,
        time_signature: Tuple[int, int]
    ) -> str:
        """Construct user prompt with style characteristics."""

    def format_midi_templates(self, midi_paths: List[str]) -> str:
        """Format MIDI templates as reference examples."""
```

### Workflows and Sequencing

**Primary Generation Flow (LLM Success):**
```
1. HybridCoordinator.generate() called
   Input: StyleProfile, GenerationParams

2. Build prompts
   → PromptBuilder.build_user_prompt()
   → SYSTEM_PROMPT (constant)

3. LLMProviderManager.generate()
   a. Try Primary Provider (Anthropic - Claude 3.5 Sonnet)
      → AnthropicProvider.generate_midi_tokens()
      → Async HTTP call to api.anthropic.com
      → Response: JSON string

   b. Parse JSON response
      → Extract MIDI JSON from response
      → Validate against schema

   c. SUCCESS → Record metrics
      → provider.record_success(tokens, cost)
      → Return GenerationResult

4. Repeat for N variations (default: 4)
   → Store all MIDI JSON dicts

5. Pass to Epic 5 (MIDI Export)
   → Convert JSON → .mid files
```

**Fallback Flow (Primary Fails → Secondary → Tertiary):**
```
3a. Try Primary (Anthropic)
    → FAIL (rate limit / timeout / invalid JSON)
    → Log error
    → provider.record_failure()

3b. Try Secondary (Google Gemini)
    → GoogleProvider.generate_midi_tokens()
    → SUCCESS
    → provider.record_success(tokens, cost)
    → Return GenerationResult with provider_used="google"

If 3b fails → Try Tertiary (OpenAI ChatGPT)
If all fail → Template fallback
```

**Template Fallback Flow (All LLMs Fail):**
```
4. TemplateGenerator.generate()
   Input: StyleProfile.midi_templates (file paths)

   a. Load MIDI templates
      → mido.MidiFile(path) for each template

   b. Extract drum patterns
      → Parse channel 10 notes
      → Extract velocities, timing

   c. Apply style variations
      → Adjust swing (quantitative_params['swing_percent'])
      → Add ghost notes (quantitative_params['ghost_note_prob'])
      → Vary velocities (mean ± std)

   d. Convert to MIDI JSON
      → Build notes array
      → Return dict

5. Return GenerationResult with provider_used="template"
```

## Non-Functional Requirements

### Performance

- **LLM API Latency:** < 30 seconds per generation (95th percentile)
  - Claude 3.5 Sonnet: ~15-20 seconds (typical)
  - Gemini 2.5/3: ~10-15 seconds (typical)
  - ChatGPT 5.1: ~20-25 seconds (typical)

- **Total Generation Time:** < 2 minutes for 4 variations
  - 4 LLM calls × 20s avg = 80 seconds
  - Validation + processing: < 10 seconds
  - Total: ~90 seconds (well under 2 min target)

- **Fallback Trigger Time:** < 5 seconds
  - Detect failure immediately (HTTP timeout or JSON parse error)
  - Switch to next provider without delay

- **Template Fallback:** < 5 seconds for 4 variations
  - Load MIDI: < 1 second
  - Pattern extraction + variation: < 4 seconds

**Measurement:** Track generation_time_ms in GenerationResult, log to database (generation_history table).

### Security

- **API Key Storage:** Environment variables only (never commit to code)
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`
  - `OPENAI_API_KEY`

- **Input Validation:**
  - Sanitize all user inputs before building prompts
  - Escape special characters in artist names
  - Validate tempo/bars/time_signature ranges

- **Output Validation:**
  - Strict JSON schema validation (reject invalid structures)
  - MIDI note range validation (35-81 for drums)
  - Velocity range validation (1-127)
  - Time values must be positive integers

- **Rate Limiting Compliance:**
  - Respect provider rate limits (exponential backoff)
  - Anthropic: No published limits (monitor usage)
  - Google Gemini: 60 requests/minute (free tier)
  - OpenAI: Tier-based (monitor usage)

### Reliability/Availability

- **Success Rate Target:** 98%+ across all providers
  - Primary (Claude): 95%+ success
  - Secondary (Gemini): 90%+ success
  - Tertiary (OpenAI): 85%+ success
  - Template Fallback: 99%+ success (local processing)
  - Combined: ~99.8% success rate

- **Retry Logic:**
  - HTTP timeout: 30 seconds
  - Retry on transient errors (503, 429): max 3 retries with exponential backoff
  - Permanent errors (401, 403): skip to next provider

- **Graceful Degradation:**
  - If all LLMs fail → Template fallback (always succeeds)
  - If templates missing → Generic drum patterns (last resort)

- **Error Recovery:**
  - Provider-level: Automatic failover to next provider
  - System-level: Template fallback ensures zero downtime
  - Monitoring: Alert if provider success rate < 80%

### Observability

- **Logging:**
  - Log every LLM API call (provider, latency, cost, success/failure)
  - Log prompt construction (style profile used, parameters)
  - Log JSON validation results (pass/fail, error details)
  - Structured logs (JSON format) for easy parsing

- **Metrics:**
  - Provider success rate (per provider, per hour)
  - Average latency (per provider)
  - Cost per generation (per provider)
  - Fallback trigger frequency
  - Template usage frequency

- **Tracing:**
  - Trace generation requests end-to-end
  - Track which provider was used for each request
  - Measure time spent in each stage (prompt build, LLM call, validation, fallback)

- **Dashboards:**
  - Real-time provider health (success rate, latency)
  - Cost analysis (total spend per provider per day)
  - Fallback frequency trends

## Dependencies and Integrations

**External Dependencies (requirements.txt):**
```python
# LLM Providers
anthropic==0.39.0             # Claude 3.5 Sonnet API (Primary)
google-generativeai==0.8.3    # Gemini 2.5/3 API (Secondary)
openai==1.54.5                # ChatGPT 5.1 API (Tertiary/Fallback)

# Async HTTP
aiohttp==3.11.10
httpx==0.28.1

# Data Validation
pydantic==2.10.3

# MIDI Processing (for template fallback)
mido==1.3.3

# Utilities
python-dotenv==1.0.1
loguru==0.7.3
```

**Version Constraints:**
- anthropic: >= 0.39.0 (uses AsyncAnthropic client)
- google-generativeai: >= 0.8.0 (Gemini 2.5/3 support)
- openai: >= 1.50.0 (ChatGPT 5.1 support)

**Integration Points:**

1. **Epic 1 (Research Pipeline) → Epic 2:**
   - Consumes: `StyleProfile` object
   - Fields used:
     - `text_description` → LLM prompt
     - `quantitative_params` → Template fallback
     - `midi_templates` → Template fallback
     - `confidence_score` → Quality indicator

2. **Epic 2 → Epic 3 (Database):**
   - Stores: `generation_history` records
   - Fields: `artist_id`, `provider_used`, `generation_time_ms`, `user_params`, `output_files`

3. **Epic 2 → Epic 5 (MIDI Export):**
   - Passes: MIDI JSON dict (list of dicts for variations)
   - Epic 5 converts to .mid files and applies humanization

4. **Epic 2 ← Epic 4 (API Layer):**
   - Called via: `POST /api/v1/generate` endpoint
   - Returns: File paths after Epic 5 processing

**External APIs:**
- Anthropic API: `https://api.anthropic.com/v1/messages`
- Google Gemini API: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent`
- OpenAI API: `https://api.openai.com/v1/chat/completions`

## Acceptance Criteria (Authoritative)

**AC-1:** BaseLLMProvider interface defines all required methods
- `generate_midi_tokens()` (abstract)
- `validate_credentials()` (abstract)
- `provider_name` (property)
- `cost_per_1k_tokens` (property)
- `record_success()`, `record_failure()`
- `success_rate` (property)

**AC-2:** Anthropic provider implementation complete
- Uses `AsyncAnthropic` client
- Model: `claude-3-5-sonnet-20241022`
- Parses JSON from response content
- Validates MIDI JSON schema
- Tracks cost: $3 input / $15 output per 1M tokens

**AC-3:** Google provider implementation complete
- Uses `google.generativeai` SDK
- Model: `gemini-2.5-flash` or `gemini-3.0-pro`
- Parses JSON from response
- Validates MIDI JSON schema
- Tracks cost: $0.125 input / $0.375 output per 1M tokens

**AC-4:** OpenAI provider implementation complete
- Uses `AsyncOpenAI` client
- Model: `chatgpt-5.1-latest` or `gpt-4`
- Parses JSON from response
- Validates MIDI JSON schema
- Tracks cost: $2.50 input / $10 output per 1M tokens

**AC-5:** LLMProviderManager implements fallback logic
- Primary → Secondary → Tertiary → Template
- Fallback triggers on: HTTP errors, timeouts, invalid JSON
- Logs every fallback event
- Returns provider_used in result

**AC-6:** Prompt engineering produces valid prompts
- System prompt defines MIDI JSON format
- User prompt includes style characteristics
- Tempo, swing, syncopation parameters included
- MIDI templates formatted as examples (if available)

**AC-7:** Template-based fallback generates valid MIDI
- Loads MIDI templates from file paths
- Extracts drum patterns (channel 10)
- Applies swing from quantitative_params
- Adds ghost notes based on probability
- Varies velocities (mean ± std)
- Returns valid MIDI JSON

**AC-8:** Hybrid coordinator orchestrates full flow
- Tries LLM first (via ProviderManager)
- Falls back to Template if LLM fails
- Generates N variations (1-8)
- Returns GenerationResult for each variation

**AC-9:** Cost tracking accurate to $0.01
- Calculates input + output token costs
- Logs to generation_history table
- Accessible via GET /api/v1/stats

**AC-10:** Success rate >= 98% in integration tests
- Test with 100 requests
- Max 2 failures allowed
- Primary provider should handle 90%+
- Fallback handles remainder

## Traceability Mapping

| AC | Spec Section | Component | Test Idea |
|----|-------------|-----------|-----------|
| AC-1 | APIs & Interfaces | `BaseLLMProvider` | Unit: Verify all abstract methods defined |
| AC-2 | Services & Modules | `AnthropicProvider` | Unit: Mock Anthropic API, verify JSON parsing |
| AC-3 | Services & Modules | `GoogleProvider` | Unit: Mock Gemini API, verify JSON parsing |
| AC-4 | Services & Modules | `OpenAIProvider` | Unit: Mock OpenAI API, verify JSON parsing |
| AC-5 | Workflows & Sequencing | `LLMProviderManager` | Integration: Simulate primary failure, verify fallback |
| AC-6 | Detailed Design | `PromptBuilder` | Unit: Generate prompt, verify all params included |
| AC-7 | Workflows & Sequencing | `TemplateGenerator` | Unit: Load test MIDI, verify valid JSON output |
| AC-8 | Workflows & Sequencing | `HybridCoordinator` | Integration: End-to-end generation with fallback |
| AC-9 | Non-Functional Req | `LLMProviderManager` | Integration: Generate pattern, verify cost logged |
| AC-10 | Non-Functional Req | Full System | Integration: 100 requests, assert success >= 98% |

## Risks, Assumptions, Open Questions

**Risks:**

1. **LLM API Reliability (HIGH):**
   - **Risk:** Provider outages or rate limits cause failures
   - **Mitigation:** Multi-provider fallback + template fallback ensures 99%+ uptime
   - **Monitoring:** Alert if any provider success rate < 80%

2. **LLM Output Quality (MEDIUM):**
   - **Risk:** Generated patterns may not match artist style
   - **Mitigation:** Prompt engineering with quantitative params + examples
   - **Validation:** User feedback loop (future: adjust prompts based on ratings)

3. **Cost Overruns (MEDIUM):**
   - **Risk:** High token usage drives up API costs
   - **Mitigation:** Monitor cost per generation, set monthly budget alerts
   - **Optimization:** Use cheaper models (Gemini) as primary if quality acceptable

4. **JSON Schema Violations (LOW):**
   - **Risk:** LLMs occasionally return invalid JSON
   - **Mitigation:** Strict validation, fallback to template on parse errors
   - **Monitoring:** Log all validation failures for prompt refinement

**Assumptions:**

1. All three LLM providers (Anthropic, Google, OpenAI) maintain stable APIs through 2025
2. StyleProfile from Epic 1 contains sufficient data for quality prompts (confidence >= 0.6)
3. MIDI templates exist for popular artists (template fallback works for 80%+ of artists)
4. Users accept 15-30 second latency for LLM generation (vs instant for templates)

**Open Questions:**

1. **Q:** Which provider should be primary? Claude (best quality) vs Gemini (cheapest)?
   - **A:** Start with Claude (PRD specifies quality over cost), monitor costs

2. **Q:** Should we cache LLM outputs to avoid redundant API calls for same params?
   - **A:** Yes, implement in Epic 3 (Redis cache keyed by artist + params hash)

3. **Q:** How to handle provider model updates (e.g., Claude 4.0 released)?
   - **A:** Provider classes should support model version as config param

4. **Q:** Should we support local LLMs (Llama, Mistral) for privacy-focused users?
   - **A:** Out of scope for v2.0 MVP, planned for v2.3.0

## Test Strategy Summary

**Unit Tests (pytest):**
- `test_base_provider_interface()`: Verify abstract methods raise NotImplementedError
- `test_anthropic_provider_generate()`: Mock API call, verify JSON parsing
- `test_google_provider_generate()`: Mock API call, verify JSON parsing
- `test_openai_provider_generate()`: Mock API call, verify JSON parsing
- `test_provider_manager_fallback()`: Mock failures, verify fallback order
- `test_prompt_builder()`: Verify system + user prompts contain all params
- `test_template_generator()`: Load test MIDI, verify output valid
- `test_cost_calculation()`: Verify token cost math (input + output)

**Integration Tests (pytest-asyncio):**
- `test_end_to_end_generation()`: Real LLM calls (use test API keys), verify MIDI JSON
- `test_fallback_chain()`: Disable primary, verify secondary used
- `test_template_fallback()`: Disable all LLMs, verify template used
- `test_multi_variation_generation()`: Generate 4 variations, verify all valid
- `test_cost_logging()`: Generate pattern, verify cost logged to DB

**Performance Tests:**
- `test_generation_latency()`: Assert < 30 seconds for single generation
- `test_multi_variation_latency()`: Assert < 2 minutes for 4 variations
- `test_fallback_latency()`: Assert fallback triggers < 5 seconds

**Load Tests (pytest + locust):**
- Simulate 100 concurrent requests
- Verify success rate >= 98%
- Verify no provider gets overwhelmed (rate limit errors)

**Coverage Target:** 85%+ for all provider classes, 80%+ for manager/coordinator
