from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ProviderResponse(BaseModel):
    provider_name: str
    model_name: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    is_refusal: bool = False
    refusal_reason: Optional[str] = None

class BaseProvider(ABC):
    def __init__(self, name: str, is_paid: bool = False, speed_tier: str = "fast"):
        self.name = name
        self.is_paid = is_paid
        self.speed_tier = speed_tier  # "fast" | "deep"

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider API key is configured and endpoint reachable."""

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        capability: str = "general_reasoning",
        model: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:
        """Generate response from provider."""
