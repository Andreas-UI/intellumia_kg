from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"  # change as needed
    temperature: float = 0.0
    max_tokens: int = 1500
    timeout: int = 60  # seconds
    json_mode: bool = True  # ask model for pure JSON


class LLMClient:
    """Reusable LLM base client. Subclass to plug any provider (OpenAI, Azure, local vLLM)."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw string output from provider."""
        raise NotImplementedError("Implement in a provider-specific subclass")


class OpenAIClient(LLMClient):
    """
    Example OpenAI client (pseudo). Uncomment and wire up with openai SDK.
    """

    def __init__(self, config: LLMConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key
        import openai
        openai.api_key = api_key

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            response_format={"type": "json_object"} if self.config.json_mode else None,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content
