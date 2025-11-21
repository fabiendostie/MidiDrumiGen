"""OpenAI ChatGPT 5.1 fallback provider."""

from __future__ import annotations

from openai import AsyncOpenAI  # type: ignore[import]

from .base import (
    BaseLLMProvider,
    LLMGenerationResult,
    LLMProviderCredentialsError,
    LLMUsage,
    extract_json_payload,
)

CHATGPT_5_1_MODEL = "chatgpt-5.1-latest"


class OpenAIProvider(BaseLLMProvider):
    """Tertiary provider leveraging OpenAI ChatGPT 5.1."""

    provider_name = "openai"
    default_model = CHATGPT_5_1_MODEL
    input_cost_per_1k = 0.0025  # $2.50 per 1M tokens
    output_cost_per_1k = 0.01  # $10 per 1M tokens

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or self.read_env("OPENAI_API_KEY")
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if not self._client:
            if not self.api_key:
                raise LLMProviderCredentialsError(
                    "OPENAI_API_KEY is not configured for OpenAIProvider."
                )
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    def validate_credentials(self) -> bool:
        return bool(self.api_key)

    async def _generate_impl(self, *, user_prompt: str, system_prompt: str) -> LLMGenerationResult:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.8,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        choice = response.choices[0]
        content = choice.message.content or "{}"
        payload = extract_json_payload(content)
        usage = LLMUsage(
            input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        )

        return LLMGenerationResult(
            payload=payload,
            raw_response=response,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
            cost_usd=self.calculate_cost(usage),
        )
