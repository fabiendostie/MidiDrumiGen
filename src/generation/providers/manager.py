"""LLM Provider Manager with automatic failover."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .anthropic import AnthropicProvider
from .base import BaseLLMProvider, LLMGenerationResult, LLMProviderError
from .google import GoogleProvider
from .openai import OpenAIProvider

PRIMARY_PROVIDER = "anthropic"
SECONDARY_PROVIDER = "google"
TERTIARY_PROVIDER = "openai"
DEFAULT_ORDER = [PRIMARY_PROVIDER, SECONDARY_PROVIDER, TERTIARY_PROVIDER]


class LLMProviderManager:
    """Coordinates multiple providers and handles failover."""

    def __init__(
        self,
        providers: Mapping[str, BaseLLMProvider] | None = None,
        *,
        primary: str = PRIMARY_PROVIDER,
        fallback_order: Sequence[str] | None = None,
    ) -> None:
        self.providers: dict[str, BaseLLMProvider] = dict(providers or {})
        self.primary = primary
        self.fallback_order: list[str] = list(fallback_order or DEFAULT_ORDER)

    # ------------------------------------------------------------------ #
    # Provider registration
    # ------------------------------------------------------------------ #

    def register(self, provider: BaseLLMProvider) -> None:
        self.providers[provider.provider_name] = provider

    def ensure_defaults(self) -> None:
        if PRIMARY_PROVIDER not in self.providers:
            self.register(AnthropicProvider())
        if SECONDARY_PROVIDER not in self.providers:
            self.register(GoogleProvider())
        if TERTIARY_PROVIDER not in self.providers:
            self.register(OpenAIProvider())

    # ------------------------------------------------------------------ #
    # Generation logic
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        provider: str | None = None,
        allow_fallback: bool = True,
    ) -> LLMGenerationResult:
        """Generate MIDI JSON using the requested provider with automatic fallback."""

        self.ensure_defaults()
        order = self._resolve_order(provider, allow_fallback)
        last_error: Exception | None = None

        for provider_name in order:
            llm = self.providers.get(provider_name)
            if not llm:
                continue
            try:
                return await llm.generate_midi_tokens(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
            except Exception as exc:  # pragma: no cover - dependent on API behavior
                last_error = exc
        raise LLMProviderError(
            f"All providers failed. Last error: {last_error!r}"
            if last_error
            else "No providers available."
        )

    def _resolve_order(self, provider: str | None, allow_fallback: bool) -> list[str]:
        if provider and provider != "auto":
            if allow_fallback:
                remainder = [p for p in self.fallback_order if p != provider]
                return [provider, *remainder]
            return [provider]
        return list(self.fallback_order)

    # ------------------------------------------------------------------ #
    # Stats & helpers
    # ------------------------------------------------------------------ #

    def get_provider_stats(self) -> dict[str, dict[str, float]]:
        self.ensure_defaults()
        return {name: provider.stats() for name, provider in self.providers.items()}


def build_default_manager() -> LLMProviderManager:
    """Convenience helper to create a manager with default providers."""

    manager = LLMProviderManager()
    manager.ensure_defaults()
    return manager
