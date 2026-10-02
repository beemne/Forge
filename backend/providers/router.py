import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from backend.config import settings
from backend.providers.base import BaseProvider, ProviderResponse

from backend.providers.real_providers import GeminiProvider, OpenAISpecProvider, CloudflareProvider
from backend.providers.quota_manager import quota_manager

logger = logging.getLogger("forge.router")

async def _notify_fallback(failed_provider: str, reason: str, next_candidate: Optional[str] = None):
    """Broadcast real-time WebSocket notification when a provider fails/exhausts quota and triggers cascade."""
    try:
        from backend.websocket.manager import ws_manager
        await ws_manager.broadcast({
            "type": "PROVIDER_FALLBACK_TRIGGERED",
            "data": {
                "failed_provider": failed_provider,
                "reason": str(reason)[:160],
                "next_provider": next_candidate or "Next available candidate",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.debug(f"[ModelRouter] WS notification skip: {e}")

class ModelRouter:
    """Model Router selecting appropriate provider/model based on capability, cost, budget, and CLI routing."""

    DEFAULT_ROUTING_MAP = {
        # Curated order: Groq (multi-key), xKiro free models incl. Mistral/Ministral, OpenRouter (GLM/DeepSeek),
        # Gemini (multi-key), RapidAPI, Cloudflare, NVIDIA. Direct Mistral API sits LAST among LLM options
        # because its free-tier mistral-small/medium are gated to limit=0; xkiro_mistral serves them free.
        "recon": ["groq", "xkiro", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "rapidapi_gpt54_mini", "cloudflare", "nvidia", "mistral"],
        "directory_enumeration": ["groq", "xkiro", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "rapidapi_gpt54_mini", "cloudflare", "nvidia", "mistral"],
        "web_analysis": ["groq", "xkiro", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "rapidapi_gpt54_mini", "cloudflare", "nvidia", "mistral"],
        "web_testing": ["groq", "xkiro", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "rapidapi_gpt54_mini", "cloudflare", "nvidia", "mistral"],
        "code_analysis": ["mistral_codestral", "xkiro_coder", "groq", "xkiro", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "cloudflare", "nvidia", "mistral"],
        "reverse_engineering": ["mistral_codestral", "xkiro_coder", "groq", "xkiro", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "cloudflare", "nvidia", "mistral"],
        "fast_reasoning": ["groq", "xkiro", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "cloudflare", "rapidapi_gpt54_mini", "nvidia", "mistral"],
        "general_reasoning": ["groq", "xkiro", "xkiro_planner", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "rapidapi_gpt54_mini", "cloudflare", "nvidia", "mistral"],
        "verification": ["groq", "xkiro", "xkiro_mistral", "openrouter", "gemini", "rapidapi_deepseek_v32", "cloudflare", "nvidia", "mistral"],
        # Writeup / report authoring — Gemini is deliberately FIRST for this task
        # (strong long-form technical prose); the rest of the free chain is the
        # fallback if Gemini is unavailable / quota-exhausted.
        "report_generation": ["gemini", "groq", "xkiro", "xkiro_planner", "xkiro_mistral", "openrouter", "rapidapi_deepseek_v32", "rapidapi_gpt54_mini", "cloudflare", "nvidia", "mistral"],
        # Multimodal image reading — mapped explicitly to Gemini (vision-capable provider)
        "vision_read": ["gemini"]
    }

    # Model to Provider/Transport Mapping
    MODEL_PROVIDER_MAP = {
        # Groq (Curated fast models)
        "qwen-3.8-27b": ("groq", "qwen/qwen3.8-27b"),
        "gpt-oss-120b": ("groq", "openai/gpt-oss-120b"),
        "groq-compound": ("groq", "groq/compound"),
        # RapidAPI (High reliability & speed)
        "gpt-5.4-mini": ("rapidapi_gpt54_mini", "gpt-5.4-mini"),
        "deepseek-v3.2": ("rapidapi_deepseek_v32", "DeepSeek-V3.2"),
        "gpt-5-nano": ("rapidapi_gpt5_nano", "GPT-5-nano"),
        # OpenRouter (Low cost / accurate GLM)
        "glm-5.3-flash": ("openrouter", "z-ai/glm-5.3-flash"),
        "z-ai/glm-5.3-flash": ("openrouter", "z-ai/glm-5.3-flash"),
        "glm-5.3": ("openrouter", "z-ai/glm-5.3"),
        "z-ai/glm-5.3": ("openrouter", "z-ai/glm-5.3"),
        "deepseek-v4-flash": ("openrouter", "deepseek/deepseek-chat"),
        "deepseek/deepseek-chat": ("openrouter", "deepseek/deepseek-chat"),
        "deepseek-chat": ("openrouter", "deepseek/deepseek-chat"),
        # Cloudflare Workers AI
        "cloudflare-llama": ("cloudflare", "@cf/meta/llama-3.1-8b-instruct"),
        # Gemini (Fallback with 18-key pool)
        "gemini-3.6-flash": ("gemini", "gemini-3.6-flash"),
        "gemini-1.5-pro": ("gemini", "gemini-1.5-pro"),
        # NVIDIA NIM
        "deepseek-v4-pro": ("nvidia", "deepseek-ai/deepseek-v4-pro-0813"),
        "nemotron-lightning": ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
        "nemotron-ultra": ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
        "kimi-k3": ("nvidia", "moonshotai/kimi-k3"),
        # Mistral AI (Codestral for code, Mistral Small/Medium for general)
        "codestral-latest": ("mistral_codestral", "codestral-latest"),
        "codestral-2508": ("mistral_codestral", "codestral-2508"),
        "mistral-small-latest": ("mistral", "mistral-small-latest"),
        "mistral-medium-latest": ("mistral", "mistral-medium-latest"),
        "mistral-medium-3.5": ("mistral", "mistral-medium-3.5"),
        "magistral-medium-latest": ("mistral", "magistral-medium-latest"),
        "magistral-small-latest": ("mistral", "magistral-small-latest"),
        # xKiro Free-Tier Models (Zero cost, high quota)
        "xkiro-sensenova": ("xkiro", "sensenova/sensenova-6.8-flash-lite"),
        "xkiro-sensenova-6.7": ("xkiro", "sensenova/sensenova-6.7-flash-lite"),
        "xkiro-codestral": ("xkiro_coder", "mistralai/codestral-2508"),
        "xkiro-devstral": ("xkiro_coder", "mistralai/devstral-medium"),
        "xkiro-mistral-large": ("xkiro_planner", "mistralai/mistral-large-2512"),
        "xkiro-mistral-medium": ("xkiro_mistral", "mistralai/mistral-medium-3.5"),
        "xkiro-mistral-small": ("xkiro_mistral", "mistralai/mistral-small-2603"),
        # xKiro-hosted Mistral family — free & unthrottled, unlike the direct Mistral API
        # where mistral-small/medium are gated to limit=0 on the free tier.
        "ministral-8b": ("xkiro_mistral", "mistralai/ministral-8b"),
        "ministral-3b": ("xkiro_mistral", "mistralai/ministral-3b"),
        "ministral-14b": ("xkiro_mistral", "mistralai/ministral-14b"),
        # Direct aliases
        # NOTE: "codestral-2508" is intentionally NOT redefined here. Its primary
        # mapping is the direct Mistral Codestral API above; the xKiro-hosted
        # variant is reachable via the distinct "xkiro-codestral" alias. A second
        # "codestral-2508" entry here silently shadowed the Mistral mapping.
        "devstral-medium": ("xkiro_coder", "mistralai/devstral-medium"),
        "sensenova-6.8": ("xkiro", "sensenova/sensenova-6.8-flash-lite"),
    }

    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.paid_allowed = settings.PAID_MODEL_ALLOWED
        self.daily_budget_usd = settings.DAILY_BUDGET_USD
        self.current_spent_usd = 0.0
        self._initialize_env_providers()

    def _initialize_env_providers(self):
        # 1. RapidAPI Verified Working Models
        rapidapi_key = (settings.RAPIDAPI_KEY or os.getenv("RAPIDAPI_KEY", "")).strip()
        if rapidapi_key:
            self.register_provider("rapidapi_gpt54_mini", OpenAISpecProvider(
                name="rapidapi_gpt54_mini",
                is_paid=True,
                api_key=rapidapi_key,
                default_model="gpt-5.4-mini",
                base_url="https://gpt-5-4-mini.p.rapidapi.com",
                extra_headers={"x-rapidapi-host": "gpt-5-4-mini.p.rapidapi.com", "x-rapidapi-key": rapidapi_key, "User-Agent": "Mozilla/5.0"},
                speed_tier="fast"
            ))
            self.register_provider("rapidapi_deepseek_v32", OpenAISpecProvider(
                name="rapidapi_deepseek_v32",
                is_paid=True,
                api_key=rapidapi_key,
                default_model="DeepSeek-V3.2",
                base_url="https://deepseek-v31.p.rapidapi.com/",
                extra_headers={"x-rapidapi-host": "deepseek-v31.p.rapidapi.com", "x-rapidapi-key": rapidapi_key, "User-Agent": "Mozilla/5.0"},
                speed_tier="fast"
            ))
            self.register_provider("rapidapi_gpt5_nano", OpenAISpecProvider(
                name="rapidapi_gpt5_nano",
                is_paid=True,
                api_key=rapidapi_key,
                default_model="GPT-5-nano",
                base_url="https://gpt-5-nano.p.rapidapi.com",
                extra_headers={"x-rapidapi-host": "gpt-5-nano.p.rapidapi.com", "x-rapidapi-key": rapidapi_key, "User-Agent": "Mozilla/5.0"},
                speed_tier="fast"
            ))
        # 2. Groq Provider with Multi-Key Rotation Pool
        groq_keys = [k.strip() for k in (getattr(settings, "GROQ_API_KEYS", "") or "").split(",") if k.strip()]
        groq_single_key = (settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")).strip()
        if groq_single_key or groq_keys:
            self.register_provider("groq", OpenAISpecProvider(
                name="groq",
                is_paid=False,
                api_key=groq_single_key,
                api_keys=groq_keys,
                default_model="qwen/qwen3.8-27b",
                base_url="https://api.groq.com/openai/v1",
                speed_tier="fast"
            ))
        # 3. Gemini Provider with 18-Key Rotation Pool
        if settings.GEMINI_API_KEY or settings.GEMINI_API_KEYS:
            gem_keys = [k.strip() for k in (settings.GEMINI_API_KEYS or "").split(",") if k.strip()]
            self.register_provider("gemini", GeminiProvider(
                api_key=settings.GEMINI_API_KEY,
                api_keys=gem_keys
            ))
        # 4. OpenRouter Provider
        if settings.OPENROUTER_API_KEY:
            self.register_provider("openrouter", OpenAISpecProvider(
                name="openrouter",
                is_paid=True,
                api_key=settings.OPENROUTER_API_KEY,
                default_model="z-ai/glm-5.3-flash",
                base_url="https://openrouter.ai/api/v1",
                extra_headers={"HTTP-Referer": "https://forge.local", "X-Title": "FORGE CTF"}
            ))
        # 5. Cloudflare Workers AI Provider
        if settings.CLOUDFLARE_API_TOKEN and settings.CLOUDFLARE_ACCOUNT_ID:
            self.register_provider("cloudflare", CloudflareProvider(
                api_key=settings.CLOUDFLARE_API_TOKEN,
                account_id=settings.CLOUDFLARE_ACCOUNT_ID
            ))
        # 6. NVIDIA NIM Provider
        if settings.NVIDIA_API_KEY:
            self.register_provider("nvidia", OpenAISpecProvider(
                name="nvidia",
                is_paid=True,
                api_key=settings.NVIDIA_API_KEY,
                default_model="deepseek-ai/deepseek-v4-pro-0813",
                base_url="https://integrate.api.nvidia.com/v1"
            ))
        # 7. Mistral AI Provider (46 models, no content filtering, verified 2026-09-06)
        mistral_key = (settings.MISTRAL_API_KEY or os.getenv("MISTRAL_API_KEY", "")).strip()
        if mistral_key:
            # Codestral - specialized for code analysis & reverse engineering
            self.register_provider("mistral_codestral", OpenAISpecProvider(
                name="mistral_codestral",
                is_paid=True,
                api_key=mistral_key,
                default_model="codestral-latest",
                base_url="https://api.mistral.ai/v1",
                speed_tier="fast"
            ))
            # Mistral general - ministral-8b for reasoning & CTF tasks.
            # NOTE: mistral-small/medium-latest are gated to limit=0 on the free tier
            # (persistent 429). ministral-8b-latest is served on the same key, so it's
            # the default here; the free ministral/mistral models on xKiro are preferred
            # ahead of this provider in the routing chain anyway.
            self.register_provider("mistral", OpenAISpecProvider(
                name="mistral",
                is_paid=True,
                api_key=mistral_key,
                default_model="ministral-8b-latest",
                base_url="https://api.mistral.ai/v1",
                speed_tier="fast"
            ))
        # 8. xKiro AI Gateway Provider (Free models, verified CTF unrestricted)
        xkiro_key = (getattr(settings, "XKIRO_API_KEY", "") or os.getenv("XKIRO_API_KEY", "")).strip()
        if xkiro_key:
            # Default fast reasoning / recon (Free SenseNova 6.8 Flash-Lite, 262k context)
            self.register_provider("xkiro", OpenAISpecProvider(
                name="xkiro",
                is_paid=False,
                api_key=xkiro_key,
                default_model="sensenova/sensenova-6.8-flash-lite",
                base_url="https://api.xkiro.com/v1",
                speed_tier="fast"
            ))
            # Binary decompilation and code reversing (Free Mistral Codestral, 256k context)
            self.register_provider("xkiro_coder", OpenAISpecProvider(
                name="xkiro_coder",
                is_paid=False,
                api_key=xkiro_key,
                default_model="mistralai/codestral-2508",
                base_url="https://api.xkiro.com/v1",
                speed_tier="fast"
            ))
            # Large multi-step planner
            self.register_provider("xkiro_planner", OpenAISpecProvider(
                name="xkiro_planner",
                is_paid=False,
                api_key=xkiro_key,
                default_model="mistralai/mistral-large-2512",
                base_url="https://api.xkiro.com/v1",
                speed_tier="fast"
            ))
            # Free Mistral/Ministral family via xKiro — replaces the direct Mistral API's
            # free-tier models that are gated to limit=0 (mistral-small/medium). Confirmed
            # 200 on: ministral-3b/8b/14b, mistral-small-2603, mistral-medium-3.5.
            self.register_provider("xkiro_mistral", OpenAISpecProvider(
                name="xkiro_mistral",
                is_paid=False,
                api_key=xkiro_key,
                default_model="mistralai/ministral-8b",
                base_url="https://api.xkiro.com/v1",
                speed_tier="fast"
            ))
        # Note: Direct HTTP REST calls to agentrouter.org/v1 return 401 Unauthorized Client.
        # AgentRouter access is strictly mediated via terminal CLI tools (agentrouter_claude_code & agentrouter_codex).

    def register_provider(self, name: str, provider: BaseProvider):
        self.providers[name.lower()] = provider

    def register_custom_model(self, provider_name: str, api_key: str, model_id: str, base_url: str = "https://integrate.api.nvidia.com/v1", is_paid: bool = True) -> OpenAISpecProvider:
        """Dynamically register or update a model provider from snippet or UI configuration."""
        key = provider_name.lower()
        provider = OpenAISpecProvider(
            name=key,
            is_paid=is_paid,
            api_key=api_key,
            default_model=model_id,
            base_url=base_url
        )
        self.register_provider(key, provider)
        # Register in model to provider map
        self.MODEL_PROVIDER_MAP[model_id] = (key, model_id)
        
        # Prepend to routing maps for high priority
        for capability in self.DEFAULT_ROUTING_MAP:
            if key not in self.DEFAULT_ROUTING_MAP[capability]:
                self.DEFAULT_ROUTING_MAP[capability].insert(0, key)
            else:
                # Move to front
                self.DEFAULT_ROUTING_MAP[capability].remove(key)
                self.DEFAULT_ROUTING_MAP[capability].insert(0, key)

        logger.info(f"[ModelRouter] Successfully registered custom model '{model_id}' under provider '{key}' (base_url: {base_url})")
        return provider

    def set_paid_allowed(self, allowed: bool):
        self.paid_allowed = allowed

    def get_quota_status(self) -> Dict:
        """Get current AgentRouter quota status summary."""
        return quota_manager.get_quota_status_summary()

    async def route_request(
        self,
        prompt: str,
        capability: str = "general_reasoning",
        system_instruction: Optional[str] = None,
        target_model: Optional[str] = None,
        speed_tier: Optional[str] = None,
        **kwargs
    ) -> ProviderResponse:

        # 1. Direct Model Request (e.g. claude-opus-5, deepseek-v4-flash, glm-5.3)
        if target_model and target_model in self.MODEL_PROVIDER_MAP:
            # Check if this model is quota-limited and should be skipped (outside 3h window or exhausted)
            if quota_manager.is_quota_limited_model(target_model) and quota_manager.should_skip_quota_limited_models():
                fallback_model = quota_manager.get_fallback_model(target_model) or "deepseek/deepseek-chat"
                logger.info(
                    f"[ModelRouter] Model '{target_model}' skipped (outside 3h post-reset window or quota exhausted). "
                    f"Directing to always-available model '{fallback_model}' via OpenRouter."
                )
                target_model = fallback_model

            provider_name, target_model_id = self.MODEL_PROVIDER_MAP[target_model]
            if not quota_manager.is_blacklisted_for_session(provider_name):
                provider = self.providers.get(provider_name)
                if provider and await provider.is_available():
                    try:
                        logger.info(f"[ModelRouter] Direct routing model '{target_model}' to provider '{provider_name}' (target: {target_model_id})")
                        res = await provider.generate_response(
                            prompt=prompt,
                            system_instruction=system_instruction,
                            capability=capability,
                            model=target_model_id,
                            **kwargs
                        )
                        if not res.is_refusal:
                            quota_manager.record_successful_request(target_model)
                            return res

                        # Separate a transient 429 rate-limit from real 402/budget quota exhaustion.
                        # A 429 recovers on its own, so only trip a cooldown after repeated hits;
                        # a 402 means the pool is drained → blacklist + fail over immediately.
                        refusal = res.refusal_reason or ""
                        is_rate_limit = ("429" in refusal or "rate limit" in refusal.lower() or "rate_limited" in refusal.lower()) and "402" not in refusal
                        is_quota = ("402" in refusal or "budget" in refusal.lower() or "insufficient_quota" in refusal.lower()
                                    or "quota has been exhausted" in refusal.lower() or "budget pool" in refusal.lower())

                        if is_rate_limit and not is_quota:
                            # Persistent 429 (e.g. free-tier model gated to limit=0) trips a cooldown after N hits.
                            quota_manager.record_rate_limit(provider_name, refusal)
                            logger.warning(f"[ModelRouter] Rate limit (429) on '{target_model}' via '{provider_name}'. Falling back to capability chain...")
                            asyncio.create_task(_notify_fallback(provider_name, refusal))
                        elif is_quota:
                            quota_manager.record_quota_exhaustion(target_model, refusal)
                            quota_manager.blacklist_for_session(provider_name, refusal)
                            # Instant auto-fallback to OpenRouter always-available model
                            fallback_model = quota_manager.get_fallback_model(target_model) or "deepseek/deepseek-chat"
                            logger.info(
                                f"[ModelRouter] Quota 402 detected for '{target_model}'. "
                                f"Instantly failing over to OpenRouter model '{fallback_model}'..."
                            )
                            fb_provider_name, fb_model_id = self.MODEL_PROVIDER_MAP.get(fallback_model, ("openrouter", "deepseek/deepseek-chat"))
                            fb_provider = self.providers.get(fb_provider_name)
                            if fb_provider and await fb_provider.is_available():
                                try:
                                    fb_res = await fb_provider.generate_response(
                                        prompt=prompt,
                                        system_instruction=system_instruction,
                                        capability=capability,
                                        model=fb_model_id,
                                        **kwargs
                                    )
                                    if not fb_res.is_refusal:
                                        quota_manager.record_successful_request(fallback_model)
                                        return fb_res
                                except Exception as fb_err:
                                    logger.warning(f"[ModelRouter] Fallback model '{fallback_model}' also failed: {fb_err}")
                        else:
                            # Generic refusal (not rate-limit, not quota) — just cascade to the capability chain.
                            logger.warning(f"Provider '{provider_name}' failed for model '{target_model}': {res.refusal_reason}. Falling back to capability chain...")
                            asyncio.create_task(_notify_fallback(provider_name, res.refusal_reason or "Direct model refusal"))

                    except Exception as direct_err:
                        # Direct model path crashed (network error, timeout, etc.) — fall through to capability chain
                        error_str = str(direct_err)
                        logger.warning(f"[ModelRouter] Direct model '{target_model}' via '{provider_name}' raised exception: {error_str}. Falling back to capability chain...")
                        # Blacklist only on confirmed quota/auth failure; a bare 429 is transient (count it instead).
                        is_rl = ("429" in error_str or "rate limit" in error_str.lower() or "rate_limited" in error_str.lower()) and "402" not in error_str
                        if "402" in error_str or "401" in error_str or (quota_manager.detect_quota_error(error_str) and not is_rl):
                            quota_manager.blacklist_for_session(provider_name, error_str)
                        elif is_rl:
                            quota_manager.record_rate_limit(provider_name, error_str)
                        asyncio.create_task(_notify_fallback(provider_name, error_str))

        # 2. Capability Candidates Fallback Chain
        skip_quota_limited = quota_manager.should_skip_quota_limited_models()
        candidates = list(self.DEFAULT_ROUTING_MAP.get(capability, ["openrouter", "gemini", "cloudflare"]))

        if speed_tier:
            # Sort providers matching requested speed_tier first
            candidates.sort(key=lambda p_name: 0 if getattr(self.providers.get(p_name), "speed_tier", "fast") == speed_tier else 1)

        # Proactively push providers already near their (correctly-scoped) rate-limit
        # ceiling to the back — using headroom passively observed from real response headers
        # (no probe calls). Stable sort preserves the curated order among un-throttled peers.
        candidates.sort(key=lambda p_name: 1 if quota_manager.is_near_ceiling(p_name) else 0)

        paid_rejected = False
        budget_rejected = False

        for provider_name in candidates:
            # Time-limited circuit breaker check
            if quota_manager.is_blacklisted_for_session(provider_name):
                continue

            # Apply the quota-window decision computed above. The direct
            # target-model path already honours it; without this the fallback
            # cascade kept calling AgentRouter batch (Claude/GPT) models known to
            # be exhausted or outside their replenishment window — wasted attempts
            # that surfaced as "all providers exhausted" noise.
            if skip_quota_limited:
                models_under = [m for m, (pn, _) in self.MODEL_PROVIDER_MAP.items() if pn == provider_name]
                if models_under and any(quota_manager.is_quota_limited_model(m) for m in models_under):
                    continue

            provider = self.providers.get(provider_name)
            if not provider:
                continue

            if provider.is_paid:
                if not self.paid_allowed:
                    logger.warning(f"Provider {provider_name} rejected: Paid models disabled.")
                    paid_rejected = True
                    continue
                # Note: Budget restrictions removed - cost is tracked for telemetry without halting runs.

            if await provider.is_available():
                try:
                    response = await provider.generate_response(
                        prompt=prompt,
                        system_instruction=system_instruction,
                        capability=capability,
                        **kwargs
                    )
                    
                    if response.is_refusal:
                        refusal = response.refusal_reason or ""
                        # Confirmed quota exhaustion (402) or auth failure (401) → immediate blacklist.
                        if ("402" in refusal or "budget" in refusal.lower()) and quota_manager.detect_quota_error(refusal):
                            quota_manager.blacklist_for_session(provider_name, refusal)
                        elif "402" in refusal:
                            quota_manager.blacklist_for_session(provider_name, refusal)
                        elif "401" in refusal or "unauthorized" in refusal.lower():
                            quota_manager.blacklist_for_session(provider_name, refusal)
                        elif "429" in refusal or "rate limit" in refusal.lower() or "rate_limited" in refusal.lower():
                            # Transient on the first hit, but a free-tier model gated to zero
                            # never recovers — trip a cooldown after repeated 429s so we skip it fast.
                            quota_manager.record_rate_limit(provider_name, refusal)
                        # generic refusals: skip this provider for now but don't blacklist
                        logger.warning(f"Model refusal from {provider_name}: {response.refusal_reason}. Fallback...")
                        asyncio.create_task(_notify_fallback(provider_name, refusal or "Model Refusal / Quota Limit"))
                        continue

                    if response.estimated_cost_usd > 0:
                        self.current_spent_usd += response.estimated_cost_usd

                    quota_manager.record_successful_request(response.model_name)
                    quota_manager.record_successful_request(provider_name)
                    return response
                except Exception as e:
                    error_str = str(e)
                    # Only blacklist for confirmed quota/auth errors, not transient network issues
                    if quota_manager.detect_quota_error(error_str) or "402" in error_str:
                        quota_manager.blacklist_for_session(provider_name, error_str)
                    elif "401" in error_str:
                        quota_manager.blacklist_for_session(provider_name, error_str)
                    elif "429" in error_str or "rate limit" in error_str.lower() or "rate_limited" in error_str.lower():
                        quota_manager.record_rate_limit(provider_name, error_str)
                    logger.error(f"Error calling provider {provider_name}: {error_str}. Fallback...")
                    asyncio.create_task(_notify_fallback(provider_name, error_str))
                    continue

        logger.error("[ModelRouter] ALL providers exhausted. No valid response obtained.")
        refusal_msg = "All providers exhausted — no live provider could fulfill this request."
        if paid_rejected:
            refusal_msg = "Capability unavailable because paid models are disabled (PAID_MODEL_ALLOWED=False)."
            if capability == "vision_read":
                refusal_msg = "Vision read capability (vision_read) is unavailable because paid models are disabled (PAID_MODEL_ALLOWED=False)."
        elif budget_rejected:
            refusal_msg = f"Capability unavailable because daily budget ${self.daily_budget_usd} exceeded."

        return ProviderResponse(
            provider_name="none",
            model_name="none",
            content=f"[FORGE ERROR] {refusal_msg}",
            prompt_tokens=0,
            completion_tokens=0,
            estimated_cost_usd=0.0,
            latency_ms=0.0,
            is_refusal=True,
            refusal_reason=refusal_msg
        )

model_router = ModelRouter()
