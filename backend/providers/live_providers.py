import json
import logging
from typing import Optional
from backend.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger("forge.providers")

class GeminiProvider(BaseProvider):
    def __init__(self, api_key: str = ""):
        super().__init__(name="gemini", is_paid=True)
        self.api_key = api_key

    async def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        capability: str = "general_reasoning",
        model: Optional[str] = "gemini-1.5-pro",
        **kwargs
    ) -> ProviderResponse:
        # Structured Gemini provider response stub
        content = json.dumps({
            "agent": "web",
            "action": "request_capability",
            "capability": capability,
            "reason": f"Gemini reasoning model processing capability '{capability}'."
        })
        return ProviderResponse(
            provider_name="gemini",
            model_name=model or "gemini-1.5-pro",
            content=content,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(content) // 4,
            estimated_cost_usd=0.001,
            latency_ms=120.0,
            is_refusal=False
        )

class NvidiaProvider(BaseProvider):
    def __init__(self, api_key: str = ""):
        super().__init__(name="nvidia", is_paid=True)
        self.api_key = api_key

    async def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        capability: str = "code_analysis",
        model: Optional[str] = "nvidia/llama-3.1-405b-instruct",
        **kwargs
    ) -> ProviderResponse:
        content = json.dumps({
            "agent": "rev",
            "action": "request_capability",
            "capability": capability,
            "reason": f"NVIDIA Nim engine analyzing code for '{capability}'."
        })
        return ProviderResponse(
            provider_name="nvidia",
            model_name=model or "llama-3.1-405b",
            content=content,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(content) // 4,
            estimated_cost_usd=0.002,
            latency_ms=90.0,
            is_refusal=False
        )

class CerebrasProvider(BaseProvider):
    def __init__(self, api_key: str = ""):
        super().__init__(name="cerebras", is_paid=False)
        self.api_key = api_key

    async def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        capability: str = "fast_reasoning",
        model: Optional[str] = "llama3.1-8b",
        **kwargs
    ) -> ProviderResponse:
        content = json.dumps({
            "agent": "recon",
            "action": "request_capability",
            "capability": capability,
            "reason": f"Cerebras ultra-fast inference engine processing '{capability}'."
        })
        return ProviderResponse(
            provider_name="cerebras",
            model_name=model or "llama3.1-8b",
            content=content,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(content) // 4,
            estimated_cost_usd=0.0,
            latency_ms=15.0,
            is_refusal=False
        )
