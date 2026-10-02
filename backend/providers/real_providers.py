import base64
import os
import httpx
import logging
from typing import Optional, Dict, Any
from backend.providers.base import BaseProvider, ProviderResponse
from backend.providers.rate_limits import parse_ratelimit_headers
from backend.providers.quota_manager import quota_manager
from backend.config import settings

logger = logging.getLogger("forge.providers")


def _detect_mime_type(image_path: str, header_bytes: bytes = b"") -> str:
    """Detect image MIME type from magic bytes or file extension."""
    if header_bytes:
        if header_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header_bytes.startswith(b"GIF8"):
            return "image/gif"
        if header_bytes.startswith(b"RIFF") and b"WEBP" in header_bytes:
            return "image/webp"
        if header_bytes.startswith(b"BM"):
            return "image/bmp"

    ext = os.path.splitext(image_path or "")[1].lower()
    ext_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return ext_map.get(ext, "image/png")


def _max_tokens_floor(model: str) -> int:
    """Per-model minimum for max_tokens. GLM (z-ai) is a reasoning model: too low a ceiling
    is spent entirely on hidden reasoning tokens, returning HTTP 200 with empty content.
    Enforce a floor so a low caller value can't starve visible output."""
    m = (model or "").lower()
    if "glm" in m:
        return int(getattr(settings, "GLM_MIN_MAX_TOKENS", 2048))
    return 0

class HTTPBaseProvider(BaseProvider):
    def __init__(self, name: str, is_paid: bool, api_key: str, default_model: str, base_url: str, speed_tier: str = "fast"):
        super().__init__(name=name, is_paid=is_paid, speed_tier=speed_tier)
        self.api_key = api_key.strip()
        self.default_model = default_model
        self.base_url = base_url.rstrip("/")

    async def is_available(self) -> bool:
        return bool(self.api_key)

    async def _post_json(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float = 30.0):
        """POST and return (json_body, response_headers). Headers are returned so callers
        can passively read rate-limit metadata; raises on non-200 as before."""
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code in (401, 403):
                raise PermissionError(f"HTTP {res.status_code} Unauthorized/Forbidden for {self.name}: {res.text[:200]}")
            elif res.status_code == 402:
                raise RuntimeError(f"HTTP 402 Quota Exhausted for {self.name}: {res.text[:200]}")
            elif res.status_code == 429:
                raise RuntimeError(f"HTTP 429 Rate Limit for {self.name}: {res.text[:200]}")
            elif res.status_code >= 500:
                raise RuntimeError(f"HTTP {res.status_code} Server Error for {self.name}: {res.text[:200]}")
            elif res.status_code != 200:
                raise RuntimeError(f"HTTP {res.status_code} for {self.name} at {url}: {res.text[:200]}")
            return res.json(), res.headers

class GeminiProvider(HTTPBaseProvider):
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
    ]

    def __init__(self, api_key: str = "", api_keys: Optional[list] = None):
        super().__init__(
            name="gemini",
            is_paid=True,
            api_key=api_key,
            default_model="gemini-3.6-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/models"
        )
        # Initialize key pool (split comma-separated or list)
        keys_pool = []
        if api_keys:
            keys_pool.extend([k.strip() for k in api_keys if k and k.strip()])
        if api_key and api_key.strip() not in keys_pool:
            keys_pool.insert(0, api_key.strip())
        self.api_keys = list(dict.fromkeys(keys_pool))
        self.current_key_idx = 0
        if self.api_keys:
            self.api_key = self.api_keys[0]

    def _rotate_key(self) -> str:
        """Rotates to the next available API key in the pool."""
        if not self.api_keys:
            return self.api_key
        prev_idx = self.current_key_idx
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_idx]
        logger.warning(f"[GEMINI KEY ROTATION] Quota exhausted on key #{prev_idx + 1}. Rotated to key #{self.current_key_idx + 1} of {len(self.api_keys)}.")
        return self.api_key

    async def is_available(self) -> bool:
        return bool(self.api_keys or self.api_key)

    async def generate_response(
        self, prompt: str, system_instruction: Optional[str] = None, capability: str = "general_reasoning", model: Optional[str] = None, image_path: Optional[str] = None, **kwargs
    ) -> ProviderResponse:
        model_to_use = model or self.default_model
        if not await self.is_available():
            return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason="API Key unconfigured")

        img_path = image_path or kwargs.get("image_path")
        parts: list[Dict[str, Any]] = [{"text": prompt}]

        if img_path and os.path.isfile(img_path):
            try:
                with open(img_path, "rb") as fh:
                    img_bytes = fh.read()
                mime_type = _detect_mime_type(img_path, img_bytes[:32])
                b64_data = base64.b64encode(img_bytes).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": b64_data
                    }
                })
            except Exception as read_err:
                logger.warning(f"Could not read image file '{img_path}' for Gemini vision request: {read_err}")

        payload = {
            "contents": [{"parts": parts}],
            "safetySettings": self.SAFETY_SETTINGS
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # Try up to number of keys in pool
        max_attempts = max(1, len(self.api_keys))
        last_err = ""

        for attempt in range(max_attempts):
            active_key = self.api_keys[self.current_key_idx] if self.api_keys else self.api_key
            url = f"{self.base_url}/{model_to_use}:generateContent?key={active_key}"

            try:
                data, _hdrs = await self._post_json(url, {"Content-Type": "application/json"}, payload)
                candidates = data.get("candidates", [])
                if not candidates:
                    return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason="No candidates returned")
                
                text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return ProviderResponse(
                    provider_name=self.name,
                    model_name=model_to_use,
                    content=text_content,
                    prompt_tokens=len(prompt) // 4,
                    completion_tokens=len(text_content) // 4,
                    estimated_cost_usd=0.001
                )
            except Exception as e:
                err_str = str(e)
                last_err = err_str
                # If 429 Resource Exhausted or 402, rotate to next key and retry immediately
                if "429" in err_str or "402" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logger.warning(f"Gemini key #{self.current_key_idx + 1} quota exhausted: {err_str[:120]}. Rotating...")
                    self._rotate_key()
                    continue
                else:
                    logger.error(f"Gemini API error on model '{model_to_use}': {err_str}")
                    return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason=err_str)

        return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason=f"All {max_attempts} Gemini API keys exhausted: {last_err}")

class OpenAISpecProvider(HTTPBaseProvider):
    """Generic Provider for OpenAI-compatible REST APIs (OpenRouter, NVIDIA NIM, Cerebras, AgentRouter, Groq, Mistral)."""
    def __init__(self, name: str, is_paid: bool, api_key: str = "", default_model: str = "", base_url: str = "", extra_headers: Optional[Dict[str, str]] = None, speed_tier: str = "fast", api_keys: Optional[list] = None):
        super().__init__(name=name, is_paid=is_paid, api_key=api_key, default_model=default_model, base_url=base_url, speed_tier=speed_tier)
        self.extra_headers = extra_headers or {}
        # Multi-key pool setup
        keys_pool = []
        if api_keys:
            keys_pool.extend([k.strip() for k in api_keys if k and k.strip()])
        if api_key and api_key.strip() not in keys_pool:
            keys_pool.insert(0, api_key.strip())
        self.api_keys = list(dict.fromkeys(keys_pool))
        self.current_key_idx = 0
        if self.api_keys:
            self.api_key = self.api_keys[0]

    def _rotate_key(self) -> str:
        """Rotates to the next available API key in the pool."""
        if not self.api_keys:
            return self.api_key
        prev_idx = self.current_key_idx
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_idx]
        logger.warning(f"[{self.name.upper()} KEY ROTATION] Rotated from key #{prev_idx + 1} to key #{self.current_key_idx + 1} of {len(self.api_keys)}.")
        return self.api_key

    async def is_available(self) -> bool:
        return bool(self.api_keys or self.api_key)

    async def generate_response(
        self, prompt: str, system_instruction: Optional[str] = None, capability: str = "general_reasoning", model: Optional[str] = None, **kwargs
    ) -> ProviderResponse:
        model_to_use = model or self.default_model
        if not await self.is_available():
            return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason="API Key unconfigured")

        if self.base_url.endswith("/chat/completions"):
            url = self.base_url
        elif "deepseek-v31.p.rapidapi.com" in self.base_url:
            url = self.base_url if self.base_url.endswith("/") else f"{self.base_url}/"
        else:
            url = f"{self.base_url}/chat/completions"

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt if (prompt and prompt.strip()) else "Hello"})
        # Enforce a per-model floor so a low caller value can't starve a reasoning model's
        # visible output (GLM 200-empty). max() keeps any higher caller request intact.
        requested_max = kwargs.get("max_tokens", 4096)
        payload = {
            "model": model_to_use,
            "messages": messages,
            "max_tokens": max(int(requested_max), _max_tokens_floor(model_to_use))
        }

        max_attempts = max(1, len(self.api_keys))
        last_err = ""

        for attempt in range(max_attempts):
            active_key = self.api_keys[self.current_key_idx] if self.api_keys else self.api_key
            headers = {
                "Content-Type": "application/json",
                **self.extra_headers
            }
            if "x-rapidapi-key" not in [k.lower() for k in headers.keys()] and active_key:
                headers["Authorization"] = f"Bearer {active_key}"

            try:
                data, resp_headers = await self._post_json(url, headers, payload)
                # Passively record rate-limit headroom from this real response (no probes).
                try:
                    quota_manager.record_ratelimit_snapshot(
                        self.name, parse_ratelimit_headers(self.name, resp_headers))
                except Exception:
                    pass
                choices = data.get("choices", [])
                if not choices:
                    return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason="Empty choices")

                text_content = choices[0].get("message", {}).get("content", "")
                # HTTP 200 with empty content is an execution-level failure disguised as a
                # valid response (e.g. a reasoning route whose token budget was spent before
                # any visible output). Treat it as a refusal so the router fails over instead
                # of returning empty analysis that wastes an agent iteration.
                if text_content is None or not str(text_content).strip():
                    finish = (choices[0].get("finish_reason") or "").lower()
                    logger.warning(f"[{self.name}] Empty completion for model '{model_to_use}' "
                                   f"(finish_reason={finish or 'unknown'}). Treating as failover.")
                    return ProviderResponse(
                        provider_name=self.name, model_name=model_to_use, content="", is_refusal=True,
                        refusal_reason=f"empty_completion (finish_reason={finish or 'unknown'}; "
                                       f"possible reasoning-token starvation)",
                    )
                usage = data.get("usage", {})
                return ProviderResponse(
                    provider_name=self.name,
                    model_name=model_to_use,
                    content=text_content,
                    prompt_tokens=usage.get("prompt_tokens", len(prompt) // 4),
                    completion_tokens=usage.get("completion_tokens", len(text_content) // 4),
                    estimated_cost_usd=0.001 if self.is_paid else 0.0
                )
            except Exception as e:
                err_str = str(e)
                last_err = err_str
                # Rotate on 429 rate limit or 402/403/401 quota exhaustion if we have backup keys
                if len(self.api_keys) > 1 and any(code in err_str for code in ("429", "402", "401", "403", "rate_limit", "quota")):
                    logger.warning(f"[{self.name}] Rate limit / error on key #{self.current_key_idx + 1}: {err_str[:100]}. Rotating key...")
                    self._rotate_key()
                    continue
                else:
                    logger.error(f"{self.name} API error: {err_str}")
                    return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason=err_str)

        return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason=f"All {max_attempts} {self.name} API keys exhausted: {last_err}")

class CloudflareProvider(HTTPBaseProvider):
    def __init__(self, api_key: str = "", account_id: str = ""):
        super().__init__(
            name="cloudflare",
            is_paid=False,
            api_key=api_key,
            default_model="@cf/meta/llama-3.1-8b-instruct",
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run" if account_id else ""
        )
        self.account_id = account_id

    async def is_available(self) -> bool:
        return bool(self.api_key and self.account_id)

    async def generate_response(
        self, prompt: str, system_instruction: Optional[str] = None, capability: str = "general_reasoning", model: Optional[str] = None, **kwargs
    ) -> ProviderResponse:
        model_to_use = model or self.default_model
        if not await self.is_available():
            return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason="Cloudflare API Token or Account ID unconfigured")

        url = f"{self.base_url}/{model_to_use}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {"messages": messages}

        try:
            data, _hdrs = await self._post_json(url, headers, payload)
            result = data.get("result", {})
            text_content = result.get("response", "")
            return ProviderResponse(
                provider_name=self.name,
                model_name=model_to_use,
                content=text_content,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(text_content) // 4
            )
        except Exception as e:
            logger.error(f"Cloudflare Workers AI error: {str(e)}")
            return ProviderResponse(provider_name=self.name, model_name=model_to_use, content="", is_refusal=True, refusal_reason=str(e))
