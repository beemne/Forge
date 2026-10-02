import re
from typing import Dict, Any, Optional

class SnippetParser:
    """
    Intelligent snippet parser that automatically detects and extracts:
    - API Key (e.g. nvapi-..., sk-..., AIzaSy...)
    - Model ID (e.g. moonshotai/kimi-k3, meta/llama-3.3-70b-instruct)
    - Base URL / Invoke URL (e.g. https://integrate.api.nvidia.com/v1)
    - Generation Parameters (temperature, max_tokens, etc.)
    from Python code, cURL commands, OpenAI SDK blocks, or raw JSON.
    """

    @staticmethod
    def parse_snippet(snippet_text: str) -> Dict[str, Any]:
        text = snippet_text.strip()
        if not text:
            return {"success": False, "error": "Snippet is empty"}

        api_key: Optional[str] = None
        model: Optional[str] = None
        invoke_url: Optional[str] = None
        provider_name = "nvidia"
        parameters: Dict[str, Any] = {}

        # 1. Extract API Key
        key_patterns = [
            # Bearer header: "Authorization": "Bearer nvapi-..." or -H "Authorization: Bearer ..."
            r'Authorization["\']?\s*:\s*["\']Bearer\s+([a-zA-Z0-9_\-\.]{15,})["\']',
            r'-H\s+["\']Authorization:\s+Bearer\s+([a-zA-Z0-9_\-\.]{15,})["\']',
            r'api_key\s*=\s*["\']([a-zA-Z0-9_\-\.]{15,})["\']',
            r'"api_key"\s*:\s*["\']([a-zA-Z0-9_\-\.]{15,})["\']',
            r'x-rapidapi-key["\']?\s*:\s*["\']([a-zA-Z0-9_\-\.]{15,})["\']',
            r'-H\s+["\']x-rapidapi-key:\s+([a-zA-Z0-9_\-\.]{15,})["\']',
            r'--header\s+["\']x-rapidapi-key:\s+([a-zA-Z0-9_\-\.]{15,})["\']',
            r'\b(nvapi-[a-zA-Z0-9_\-]{30,})\b',
            r'\b([a-zA-Z0-9]{20,}msh[a-zA-Z0-9_\-]{20,})\b',
            r'\b(sk-or-v1-[a-zA-Z0-9_\-]{30,})\b',
            r'\b(sk-[a-zA-Z0-9_\-]{30,})\b',
            r'\b(gsk_[a-zA-Z0-9_\-]{30,})\b',
            r'\b(csk-[a-zA-Z0-9_\-]{30,})\b'
        ]

        for pattern in key_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                api_key = match.group(1).strip()
                break

        # 2. Extract Model Name
        model_patterns = [
            r'["\']model["\']\s*:\s*["\']([^"\']+)["\']',
            r'model\s*=\s*["\']([^"\']+)["\']',
            r'--model\s+["\']?([^"\s\']+)["\']?',
            r'["\']model_name["\']\s*:\s*["\']([^"\']+)["\']',
            r'model_name\s*=\s*["\']([^"\']+)["\']'
        ]

        for pattern in model_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                model = match.group(1).strip()
                break

        # 3. Extract URL / Endpoint
        url_patterns = [
            r'--url\s+["\']?([^"\s\']+)["\']?',
            r'invoke_url\s*=\s*["\']([^"\']+)["\']',
            r'base_url\s*=\s*["\']([^"\']+)["\']',
            r'url\s*=\s*["\']([^"\']+)["\']',
            r'https?://[a-zA-Z0-9\.\-_]*rapidapi\.com[^\s"\'\)]*',
            r'https?://integrate\.api\.nvidia\.com[^\s"\'\)]*',
            r'https?://[a-zA-Z0-9\.\-_]+(?::\d+)?/v1[^\s"\'\)]*',
            r'https?://api\.cerebras\.ai[^\s"\'\)]*',
            r'https?://openrouter\.ai/api/v1[^\s"\'\)]*',
            r'https?://api\.groq\.com[^\s"\'\)]*'
        ]

        for pattern in url_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                invoke_url = match.group(1 if "(" in pattern else 0).strip()
                break

        # 4. Extract Parameters
        temp_match = re.search(r'["\']?temperature["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if temp_match:
            try:
                parameters["temperature"] = float(temp_match.group(1))
            except ValueError:
                pass

        max_tokens_match = re.search(r'["\']?max_tokens["\']?\s*[:=]\s*(\d+)', text)
        if max_tokens_match:
            try:
                parameters["max_tokens"] = int(max_tokens_match.group(1))
            except ValueError:
                pass

        # 5. Extract Extra Headers (e.g. RapidAPI)
        extra_headers: Dict[str, str] = {}
        rapidapi_host_match = re.search(r'x-rapidapi-host["\']?\s*:\s*["\']?([a-zA-Z0-9\.\-_]+)', text, re.IGNORECASE)
        if rapidapi_host_match:
            extra_headers["x-rapidapi-host"] = rapidapi_host_match.group(1).strip()
            if api_key:
                extra_headers["x-rapidapi-key"] = api_key
        elif invoke_url and "rapidapi.com" in invoke_url:
            host_m = re.search(r'https?://([a-zA-Z0-9\.\-_]*rapidapi\.com)', invoke_url)
            if host_m:
                extra_headers["x-rapidapi-host"] = host_m.group(1).strip()
                if api_key:
                    extra_headers["x-rapidapi-key"] = api_key

        # 6. Detect Provider & Normalize Base URL
        if "rapidapi.com" in (invoke_url or "") or rapidapi_host_match:
            provider_name = "rapidapi"
            if not invoke_url and rapidapi_host_match:
                invoke_url = f"https://{rapidapi_host_match.group(1).strip()}"
        elif (api_key and api_key.startswith("nvapi-")) or (invoke_url and "nvidia.com" in invoke_url):
            provider_name = "nvidia"
            if not invoke_url:
                invoke_url = "https://integrate.api.nvidia.com/v1"
        elif (api_key and api_key.startswith("sk-or-v1-")) or (invoke_url and "openrouter.ai" in invoke_url):
            provider_name = "openrouter"
            if not invoke_url:
                invoke_url = "https://openrouter.ai/api/v1"
        elif (api_key and api_key.startswith("csk-")) or (invoke_url and "cerebras.ai" in invoke_url):
            provider_name = "cerebras"
            if not invoke_url:
                invoke_url = "https://api.cerebras.ai/v1"
        elif (api_key and api_key.startswith("gsk_")) or (invoke_url and "groq.com" in invoke_url):
            provider_name = "groq"
            if not invoke_url:
                invoke_url = "https://api.groq.com/openai/v1"
        elif (api_key and api_key.startswith("AIzaSy")) or (invoke_url and "googleapis.com" in invoke_url):
            provider_name = "gemini"
            if not invoke_url:
                invoke_url = "https://generativelanguage.googleapis.com/v1beta"

        # Clean Base URL (remove trailing /chat/completions if not custom direct URL)
        base_url = invoke_url or "https://integrate.api.nvidia.com/v1"
        if base_url.endswith("/chat/completions"):
            base_url = base_url[:-len("/chat/completions")]
        base_url = base_url.rstrip("/")

        masked_key = ""
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"

        return {
            "success": bool(api_key or model),
            "provider_name": provider_name,
            "api_key": api_key,
            "api_key_masked": masked_key,
            "model": model or "default",
            "base_url": base_url,
            "invoke_url": invoke_url,
            "extra_headers": extra_headers,
            "parameters": parameters,
            "raw_snippet_length": len(text)
        }
