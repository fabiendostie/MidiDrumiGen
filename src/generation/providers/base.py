"""Base abstractions shared by all LLM providers."""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]


class LLMProviderError(RuntimeError):
    """Raised when a provider request fails."""


class LLMProviderCredentialsError(LLMProviderError):
    """Raised when provider credentials are missing or invalid."""


class LLMResponseParsingError(LLMProviderError):
    """Raised when the provider returns content that cannot be parsed as JSON."""


@dataclass(slots=True)
class LLMUsage:
    """Tracks token usage for a single request."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class LLMGenerationResult:
    """Normalized result returned by every provider."""

    payload: JsonDict
    raw_response: Any
    provider: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    cost_usd: float = 0.0


JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def extract_json_payload(content: str) -> JsonDict:
    """
    Parse JSON from provider output.

    Providers occasionally wrap JSON inside Markdown code fences. This helper removes
    those fences (if present) before attempting to load the JSON payload.
    """

    candidate = content.strip()
    match = JSON_FENCE_PATTERN.search(candidate)
    if match:
        candidate = match.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        raise LLMResponseParsingError(f"Unable to parse provider JSON: {exc}") from exc


class BaseLLMProvider(ABC):
    """Abstract base class that every LLM provider must implement."""

    provider_name: str = "base"
    default_model: str = "unknown"
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    def __init__(
        self, model: str | None = None, logger: logging.Logger | None = None
    ) -> None:
        self.model_name = model or self.default_model
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.total_requests: int = 0
        self.total_failures: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

    # --------------------------------------------------------------------- #
    # Metrics & bookkeeping
    # --------------------------------------------------------------------- #

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        successes = self.total_requests - self.total_failures
        return max(successes, 0) / self.total_requests

    def record_success(self, usage: LLMUsage) -> None:
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cost_usd += self.calculate_cost(usage)

    def record_failure(self) -> None:
        self.total_failures += 1

    def calculate_cost(self, usage: LLMUsage) -> float:
        input_cost = (usage.input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * self.output_cost_per_1k
        return round(input_cost + output_cost, 6)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    async def generate_midi_tokens(
        self, *, user_prompt: str, system_prompt: str
    ) -> LLMGenerationResult:
        """
        Execute a provider request and return normalized results.

        Subclasses only need to implement `_generate_impl`, which must return
        `LLMGenerationResult`. This wrapper manages bookkeeping to keep metrics
        consistent across providers.
        """

        self.total_requests += 1
        if not self.validate_credentials():
            raise LLMProviderCredentialsError(
                f"{self.provider_name} credentials are missing. "
                "Set the appropriate API key before attempting generation."
            )

        try:
            result = await self._generate_impl(user_prompt=user_prompt, system_prompt=system_prompt)
            self.record_success(result.usage)
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            self.record_failure()
            self.logger.exception("%s provider call failed", self.provider_name)
            raise LLMProviderError(str(exc)) from exc

    @abstractmethod
    async def _generate_impl(self, *, user_prompt: str, system_prompt: str) -> LLMGenerationResult:
        """Provider-specific implementation."""

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Return True if the provider can be invoked with the current configuration."""

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #

    @staticmethod
    def read_env(key: str) -> str | None:
        """Convenience helper to fetch environment variables with consistent mocking."""

        value = os.getenv(key)
        if value:
            return value.strip()
        return None

    def stats(self) -> JsonDict:
        """Return provider metrics in a serializable structure."""

        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "requests": self.total_requests,
            "failures": self.total_failures,
            "success_rate": round(self.success_rate, 4),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost_usd": round(self.total_cost_usd, 4),
        }
