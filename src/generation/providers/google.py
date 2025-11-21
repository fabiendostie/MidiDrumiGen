"""Google Gemini provider implementation."""

from __future__ import annotations

import asyncio

import google.generativeai as genai  # type: ignore[import]

from .base import (
    BaseLLMProvider,
    LLMGenerationResult,
    LLMProviderCredentialsError,
    LLMUsage,
    extract_json_payload,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"


class GoogleProvider(BaseLLMProvider):
    """Secondary provider leveraging Google Gemini 2.5 Pro."""

    provider_name = "google"
    default_model = DEFAULT_GEMINI_MODEL
    input_cost_per_1k = 0.000125  # $0.125 per 1M tokens
    output_cost_per_1k = 0.000375  # $0.375 per 1M tokens

    def __init__(self, api_key: str | None = None, *, model: str | None = None) -> None:
        super().__init__(model=model or DEFAULT_GEMINI_MODEL)
        self.api_key = api_key or self.read_env("GOOGLE_API_KEY")
        self._model_instance: genai.GenerativeModel | None = None

    def _ensure_client(self) -> genai.GenerativeModel:
        if not self.api_key:
            raise LLMProviderCredentialsError(
                "GOOGLE_API_KEY is not configured for GoogleProvider."
            )
        if not self._model_instance:
            genai.configure(api_key=self.api_key)
            self._model_instance = genai.GenerativeModel(self.model_name)
        return self._model_instance

    def validate_credentials(self) -> bool:
        return bool(self.api_key)

    async def _generate_impl(self, *, user_prompt: str, system_prompt: str) -> LLMGenerationResult:
        model = self._ensure_client()
        combined_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        response = await model.generate_content_async(
            combined_prompt,
            generation_config={
                "temperature": 0.8,
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
        )

        text = response.text or ""
        payload = extract_json_payload(text)
        metadata = getattr(response, "usage_metadata", None)
        usage = LLMUsage(
            input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
            output_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
        )

        return LLMGenerationResult(
            payload=payload,
            raw_response=response,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
            cost_usd=self.calculate_cost(usage),
        )


async def warm_up_gemini(provider: GoogleProvider) -> None:
    """Perform a minimal request to ensure Gemini credentials are valid."""

    try:
        model = provider._ensure_client()
        await asyncio.wait_for(
            model.generate_content_async(
                "Return {} if you can read this. This is a health check.",
                generation_config={"max_output_tokens": 5},
            ),
            timeout=5,
        )
    except TimeoutError:  # pragma: no cover - best effort
        provider.logger.warning("Gemini warm-up timed out. Ensure API access is available.")
