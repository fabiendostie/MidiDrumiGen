"""Anthropic Claude provider implementation."""

from __future__ import annotations

import asyncio

from anthropic import APIError, AsyncAnthropic  # type: ignore[import]

from .base import (
    BaseLLMProvider,
    LLMGenerationResult,
    LLMProviderCredentialsError,
    LLMUsage,
    extract_json_payload,
)

CLAUDE_4_5_SONNET = "claude-4-5-sonnet-20250514"


class AnthropicProvider(BaseLLMProvider):
    """Primary LLM provider powered by Anthropic Claude 4.5 Sonnet."""

    provider_name = "anthropic"
    default_model = CLAUDE_4_5_SONNET
    input_cost_per_1k = 0.003  # $3 per 1M tokens
    output_cost_per_1k = 0.015  # $15 per 1M tokens

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncAnthropic | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or self.read_env("ANTHROPIC_API_KEY")
        self._client = client

    @property
    def client(self) -> AsyncAnthropic:
        if not self._client:
            if not self.api_key:
                raise LLMProviderCredentialsError(
                    "ANTHROPIC_API_KEY is not configured for AnthropicProvider."
                )
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    def validate_credentials(self) -> bool:
        return bool(self.api_key)

    async def _generate_impl(self, *, user_prompt: str, system_prompt: str) -> LLMGenerationResult:
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            temperature=0.8,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(block.text for block in response.content if getattr(block, "text", None))
        payload = extract_json_payload(text)
        usage = LLMUsage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
        )

        return LLMGenerationResult(
            payload=payload,
            raw_response=response,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
            cost_usd=self.calculate_cost(usage),
        )


async def warm_up_anthropic(provider: AnthropicProvider) -> None:
    """
    Perform a lightweight request to validate connectivity.

    Useful in deployment/CI scenarios where we want to fail fast if the API key is invalid.
    """

    try:
        await asyncio.wait_for(
            provider.client.messages.create(
                model=provider.model_name,
                max_tokens=1,
                temperature=0.0,
                system="You are a health check endpoint.",
                messages=[{"role": "user", "content": "Reply with {}"}],
            ),
            timeout=5,
        )
    except (TimeoutError, APIError):  # pragma: no cover - best-effort warmup
        provider.logger.warning("Anthropic warm-up failed. Verify connectivity and quotas.")
