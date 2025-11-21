# LLM Provider Manager Agent
# Sub-Agent Specification Document

**Agent Name:** LLM Provider Manager
**Version:** 2.0.0
**Date:** 2025-11-17
**Parent:** Generation Coordinator

---

## Agent Overview

### Purpose
Manage multiple LLM providers (OpenAI, Anthropic, Google) for MIDI generation with automatic failover and load balancing.

### Responsibilities
1. Initialize and maintain connections to all configured LLM providers
2. Route generation requests to appropriate provider
3. Implement fallback logic when primary provider fails
4. Track provider performance and costs
5. Handle rate limiting and retries
6. Format prompts for each provider's API
7. Validate and parse LLM responses

---

## Architecture

### File Location
`src/generation/providers/manager.py`

### Dependencies
```python
from typing import Dict, List, Optional, Any
import os
import logging
from datetime import datetime
import asyncio

import openai
from anthropic import AsyncAnthropic
import google.generativeai as genai

from src.generation.providers.base import BaseLLMProvider
from src.generation.providers.openai_provider import OpenAIProvider
from src.generation.providers.anthropic_provider import AnthropicProvider
from src.generation.providers.google_provider import GoogleProvider
```

---

## Provider Interface

### Base Provider Class

```python
# src/generation/providers/base.py

from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    All providers must implement this interface for consistent
    interaction with the Provider Manager.
    """

    def __init__(self, api_key: str, model: str, config: dict):
        self.api_key = api_key
        self.model = model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance tracking
        self.total_requests = 0
        self.total_failures = 0
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0

    @abstractmethod
    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate MIDI tokens from prompts.

        Args:
            prompt: User prompt with artist style and parameters
            system_prompt: System instructions and format spec
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with 'notes' array and metadata

        Raises:
            LLMProviderError: If generation fails
        """
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """
        Check if API credentials are valid.

        Returns:
            True if credentials work, False otherwise
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier for this provider."""
        pass

    @property
    @abstractmethod
    def cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Cost structure for this provider.

        Returns:
            {'input': cost_usd, 'output': cost_usd}
        """
        pass

    def record_success(self, tokens_used: int, cost_usd: float):
        """Track successful generation."""
        self.total_requests += 1
        self.total_tokens_used += tokens_used
        self.total_cost_usd += cost_usd

    def record_failure(self):
        """Track failed generation."""
        self.total_requests += 1
        self.total_failures += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)."""
        if self.total_requests == 0:
            return 0.0
        return 1.0 - (self.total_failures / self.total_requests)
```

---

## LLM Provider Manager

### Class Implementation

```python
# src/generation/providers/manager.py

class LLMProviderManager:
    """
    Manages multiple LLM providers with failover and load balancing.
    """

    def __init__(self, config: dict):
        """
        Initialize provider manager.

        Args:
            config: Configuration dict from generation.yaml
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Provider registry
        self.providers: Dict[str, BaseLLMProvider] = {}

        # Configuration
        self.primary = config['llm']['primary_provider']
        self.fallbacks = config['llm']['fallback_providers']

        # Initialize all configured providers
        self._initialize_providers()

        # Performance tracking
        self.generation_history = []

    def _initialize_providers(self):
        """Initialize all configured LLM providers."""
        llm_config = self.config['llm']

        # OpenAI
        if 'openai' in llm_config and llm_config['openai'].get('enabled', True):
            try:
                self.providers['openai'] = OpenAIProvider(
                    api_key=os.getenv(llm_config['openai']['api_key_env']),
                    model=llm_config['openai']['model'],
                    config=llm_config['openai']
                )
                self.logger.info("OpenAI provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI: {e}")

        # Anthropic
        if 'anthropic' in llm_config and llm_config['anthropic'].get('enabled', True):
            try:
                self.providers['anthropic'] = AnthropicProvider(
                    api_key=os.getenv(llm_config['anthropic']['api_key_env']),
                    model=llm_config['anthropic']['model'],
                    config=llm_config['anthropic']
                )
                self.logger.info("Anthropic provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Anthropic: {e}")

        # Google
        if 'google' in llm_config and llm_config['google'].get('enabled', True):
            try:
                self.providers['google'] = GoogleProvider(
                    api_key=os.getenv(llm_config['google']['api_key_env']),
                    model=llm_config['google']['model'],
                    config=llm_config['google']
                )
                self.logger.info("Google provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google: {e}")

        if not self.providers:
            raise Exception("No LLM providers could be initialized")

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        provider_name: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate MIDI tokens with automatic failover.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            provider_name: Specific provider to use, or None for auto
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Dict with generation result and metadata:
            {
                'result': {...},  # MIDI JSON
                'provider_used': 'openai',
                'tokens_used': 1234,
                'cost_usd': 0.0123,
                'generation_time_ms': 1847
            }

        Raises:
            LLMProviderError: If all providers fail
        """
        start_time = datetime.now()

        # Determine provider order
        if provider_name and provider_name in self.providers:
            providers_to_try = [provider_name]
        else:
            providers_to_try = [self.primary] + [
                p for p in self.fallbacks if p != self.primary
            ]

        # Filter to only available providers
        providers_to_try = [p for p in providers_to_try if p in self.providers]

        if not providers_to_try:
            raise LLMProviderError("No available providers")

        # Try each provider in order
        last_error = None
        for provider_name in providers_to_try:
            provider = self.providers[provider_name]

            try:
                self.logger.info(f"Attempting generation with {provider_name}")

                result = await provider.generate_midi_tokens(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Calculate metrics
                end_time = datetime.now()
                generation_time_ms = int((end_time - start_time).total_seconds() * 1000)

                # Estimate tokens and cost (provider-specific)
                tokens_used = self._estimate_tokens(prompt, system_prompt, result)
                cost_usd = self._calculate_cost(provider, tokens_used)

                # Record success
                provider.record_success(tokens_used, cost_usd)

                # Track history
                self.generation_history.append({
                    'timestamp': end_time,
                    'provider': provider_name,
                    'success': True,
                    'tokens_used': tokens_used,
                    'cost_usd': cost_usd,
                    'generation_time_ms': generation_time_ms
                })

                self.logger.info(
                    f"Generation successful with {provider_name} "
                    f"({generation_time_ms}ms, {tokens_used} tokens, ${cost_usd:.4f})"
                )

                return {
                    'result': result,
                    'provider_used': provider_name,
                    'tokens_used': tokens_used,
                    'cost_usd': cost_usd,
                    'generation_time_ms': generation_time_ms
                }

            except Exception as e:
                self.logger.warning(f"{provider_name} failed: {e}")
                provider.record_failure()
                last_error = e

                # Track failure
                self.generation_history.append({
                    'timestamp': datetime.now(),
                    'provider': provider_name,
                    'success': False,
                    'error': str(e)
                })

                continue  # Try next provider

        # All providers failed
        raise LLMProviderError(
            f"All providers failed. Last error: {last_error}"
        )

    def _estimate_tokens(
        self,
        prompt: str,
        system_prompt: str,
        result: dict
    ) -> int:
        """
        Estimate total tokens used (input + output).

        Rough estimate: ~4 characters per token
        """
        input_chars = len(prompt) + len(system_prompt)
        output_chars = len(str(result))
        return (input_chars + output_chars) // 4

    def _calculate_cost(
        self,
        provider: BaseLLMProvider,
        tokens_used: int
    ) -> float:
        """
        Calculate cost in USD for this generation.
        """
        costs = provider.cost_per_1k_tokens
        input_cost = costs['input'] * (tokens_used // 2) / 1000  # Assume 50/50 split
        output_cost = costs['output'] * (tokens_used // 2) / 1000
        return input_cost + output_cost

    def get_provider_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all providers.

        Returns:
            Dict mapping provider name to stats
        """
        stats = {}
        for name, provider in self.providers.items():
            stats[name] = {
                'total_requests': provider.total_requests,
                'total_failures': provider.total_failures,
                'success_rate': provider.success_rate,
                'total_tokens_used': provider.total_tokens_used,
                'total_cost_usd': provider.total_cost_usd
            }
        return stats

    def get_recommended_provider(self) -> str:
        """
        Get recommended provider based on performance.

        Returns:
            Provider name with best success rate and cost
        """
        if not self.providers:
            raise Exception("No providers available")

        # Score providers (success_rate * 100 - cost_per_1k_tokens)
        scores = {}
        for name, provider in self.providers.items():
            if provider.total_requests == 0:
                score = 50.0  # Neutral score for untested providers
            else:
                avg_cost = provider.cost_per_1k_tokens['output']
                score = (provider.success_rate * 100) - (avg_cost * 10)
            scores[name] = score

        return max(scores.items(), key=lambda x: x[1])[0]
```

---

## Provider Implementations

### OpenAI Provider

```python
# src/generation/providers/openai_provider.py

from openai import AsyncOpenAI
import json

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT-4 provider."""

    def __init__(self, api_key: str, model: str, config: dict):
        super().__init__(api_key, model, config)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}  # Force JSON output
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        return result

    def validate_credentials(self) -> bool:
        """Test API key validity."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Costs as of Nov 2025 (subject to change).
        """
        if 'gpt-4-turbo' in self.model:
            return {'input': 0.01, 'output': 0.03}  # $10/$30 per 1M
        elif 'gpt-4' in self.model:
            return {'input': 0.03, 'output': 0.06}  # $30/$60 per 1M
        else:  # gpt-3.5-turbo
            return {'input': 0.0005, 'output': 0.0015}  # $0.50/$1.50 per 1M
```

### Anthropic Provider

```python
# src/generation/providers/anthropic_provider.py

from anthropic import AsyncAnthropic
import json
import re

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str, config: dict):
        super().__init__(api_key, model, config)
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate using Anthropic API."""
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = message.content[0].text

        # Claude sometimes wraps JSON in markdown code blocks
        if "```json" in content:
            content = re.search(r'```json\n(.*?)\n```', content, re.DOTALL).group(1)

        result = json.loads(content)

        return result

    def validate_credentials(self) -> bool:
        """Test API key validity."""
        try:
            # Simple test message
            asyncio.run(self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            ))
            return True
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Costs as of Nov 2025.
        """
        if 'opus' in self.model:
            return {'input': 0.015, 'output': 0.075}  # $15/$75 per 1M
        elif 'sonnet' in self.model:
            return {'input': 0.003, 'output': 0.015}  # $3/$15 per 1M
        else:  # haiku
            return {'input': 0.00025, 'output': 0.00125}  # $0.25/$1.25 per 1M
```

### Google Provider

```python
# src/generation/providers/google_provider.py

import google.generativeai as genai
import json
import re

class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model: str, config: dict):
        super().__init__(api_key, model, config)
        genai.configure(api_key=api_key)
        self.model_obj = genai.GenerativeModel(model)

    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate using Google Gemini API."""
        # Gemini combines system + user prompts
        full_prompt = f"{system_prompt}\n\n{prompt}"

        response = await self.model_obj.generate_content_async(
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )

        content = response.text

        # Extract JSON if wrapped in code blocks
        if "```json" in content:
            content = re.search(r'```json\n(.*?)\n```', content, re.DOTALL).group(1)

        result = json.loads(content)

        return result

    def validate_credentials(self) -> bool:
        """Test API key validity."""
        try:
            self.model_obj.generate_content("test")
            return True
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def cost_per_1k_tokens(self) -> Dict[str, float]:
        """
        Costs as of Nov 2025.
        """
        if '1.5-pro' in self.model:
            return {'input': 0.00125, 'output': 0.00375}  # $1.25/$3.75 per 1M
        else:  # flash
            return {'input': 0.000125, 'output': 0.000375}  # $0.125/$0.375 per 1M
```

---

## Configuration

### Config File: `configs/generation.yaml`

```yaml
llm:
  primary_provider: anthropic  # Claude 3.5 Sonnet (Primary)
  fallback_providers: [anthropic, google]  # Try in this order if primary fails

  openai:
    enabled: true
    model: gpt-4-turbo-preview
    api_key_env: OPENAI_API_KEY
    temperature: 0.8
    max_tokens: 2000

  anthropic:
    enabled: true
    model: claude-3-opus-20240229
    api_key_env: ANTHROPIC_API_KEY
    temperature: 0.8
    max_tokens: 2000

  google:
    enabled: true
    model: gemini-1.5-pro
    api_key_env: GOOGLE_API_KEY
    temperature: 0.8
    max_tokens: 2000

  # Future: local LLM support
  local:
    enabled: false
    model: llama3-70b
    endpoint: http://localhost:11434
    temperature: 0.8
    max_tokens: 2000
```

---

## Error Handling

### Error Types

```python
class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass

class APIKeyError(LLMProviderError):
    """Invalid or missing API key."""
    pass

class RateLimitError(LLMProviderError):
    """Rate limit exceeded."""
    pass

class InvalidResponseError(LLMProviderError):
    """LLM returned invalid response."""
    pass
```

### Retry Logic

```python
async def generate_with_retry(
    self,
    provider: BaseLLMProvider,
    prompt: str,
    system_prompt: str,
    max_retries: int = 3
) -> Dict:
    """Generate with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await provider.generate_midi_tokens(prompt, system_prompt)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            self.logger.warning(
                f"Rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)

    raise LLMProviderError(f"Failed after {max_retries} retries")
```

---

## Testing

### Unit Tests

```python
# tests/unit/test_llm_provider_manager.py

@pytest.mark.asyncio
async def test_generate_success(mock_openai):
    """Test successful generation."""
    manager = LLMProviderManager(TEST_CONFIG)

    result = await manager.generate(
        prompt="Generate 4 bars for John Bonham",
        system_prompt=SYSTEM_PROMPT
    )

    assert result['provider_used'] == 'openai'
    assert 'result' in result
    assert result['tokens_used'] > 0

@pytest.mark.asyncio
async def test_fallback_on_primary_failure(mock_providers):
    """Test fallback to secondary provider."""
    # Mock OpenAI to fail
    mock_providers['openai'].generate_midi_tokens = AsyncMock(
        side_effect=Exception("OpenAI failed")
    )

    manager = LLMProviderManager(TEST_CONFIG)

    result = await manager.generate(
        prompt="Generate 4 bars",
        system_prompt=SYSTEM_PROMPT
    )

    # Should have fallen back to Anthropic
    assert result['provider_used'] == 'anthropic'

@pytest.mark.asyncio
async def test_all_providers_fail(mock_providers):
    """Test error when all providers fail."""
    # Mock all providers to fail
    for provider in mock_providers.values():
        provider.generate_midi_tokens = AsyncMock(
            side_effect=Exception("Failed")
        )

    manager = LLMProviderManager(TEST_CONFIG)

    with pytest.raises(LLMProviderError):
        await manager.generate(
            prompt="Generate 4 bars",
            system_prompt=SYSTEM_PROMPT
        )
```

---

## Performance Monitoring

### Metrics Tracked

```python
{
    'provider': 'openai',
    'timestamp': '2025-11-17T10:30:00Z',
    'success': True,
    'generation_time_ms': 1847,
    'tokens_used': 1234,
    'cost_usd': 0.0123,
    'model': 'gpt-4-turbo-preview',
    'artist': 'John Bonham'
}
```

### Provider Comparison Report

```python
def generate_provider_report(self) -> str:
    """Generate performance comparison report."""
    stats = self.get_provider_stats()

    report = "LLM Provider Performance Report\n"
    report += "=" * 50 + "\n\n"

    for name, data in stats.items():
        report += f"{name.upper()}:\n"
        report += f"  Requests: {data['total_requests']}\n"
        report += f"  Success Rate: {data['success_rate']:.1%}\n"
        report += f"  Total Tokens: {data['total_tokens_used']:,}\n"
        report += f"  Total Cost: ${data['total_cost_usd']:.2f}\n"
        report += f"  Avg Cost/Request: ${data['total_cost_usd'] / max(1, data['total_requests']):.4f}\n"
        report += "\n"

    return report
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Agent Status:** Specification Complete
**Next Step:** Implementation
