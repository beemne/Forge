"""Provider & model availability discovery (Workstream B).

Providers silently change their catalogs: a configured default model can become
"410 Gone", advertised models can return "404 not found" (listed but not provisioned
on our account), and whole catalogs shift. Previously FORGE discovered this only by
failing mid-run. This module probes each configured provider's LIVE catalog at startup
(and on demand), optionally verifies callability with ONE minimal completion, persists
provider health, and feeds dead models into the circuit breaker so routing drops them
automatically (dynamic routing — B3).

Design:
- Bounded, background, best-effort: a probe never blocks startup and never raises.
- Catalog-only by default; verification does at most one minimal completion per provider
  so we never spend real quota probing (the brief's cost constraint).
- Results cached in-memory with a timestamp and persisted to the ``providers`` table.
- A model that is missing/unprovisioned/gone is tripped in quota_manager, so the existing
  router fallback already skips it — no parallel routing path.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("forge.providers.discovery")

PROBE_TIMEOUT_SECONDS = 8.0


@dataclass
class ProviderHealth:
    name: str
    reachable: bool = False
    auth_ok: bool = False
    latency_ms: float = 0.0
    catalog_models: List[str] = field(default_factory=list)
    default_model_present: Optional[bool] = None
    health_status: str = "unknown"      # healthy | degraded | unavailable | unknown
    detail: str = ""
    probed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name, "reachable": self.reachable, "auth_ok": self.auth_ok,
            "latency_ms": round(self.latency_ms, 1), "catalog_model_count": len(self.catalog_models),
            "catalog_models": self.catalog_models[:50], "default_model_present": self.default_model_present,
            "health_status": self.health_status, "detail": self.detail[:200],
            "probed_at": self.probed_at,
        }


class DiscoveryService:
    """Owns the live provider/model catalog probes and the cached health report."""

    def __init__(self):
        self._health: Dict[str, ProviderHealth] = {}
        self._last_run_ts: float = 0.0
        self._running: bool = False

    # -- public API ---------------------------------------------------------- #

    def get_report(self) -> dict:
        return {
            "last_probe_ts": self._last_run_ts,
            "last_probe_age_seconds": (time.time() - self._last_run_ts) if self._last_run_ts else None,
            "providers": {name: h.to_dict() for name, h in self._health.items()},
        }

    def health_for(self, provider_name: str) -> Optional[ProviderHealth]:
        return self._health.get(provider_name)

    async def discover_all(self, *, verify: bool = True) -> dict:
        """Probe every registered provider. Never raises. Returns the report dict."""
        if self._running:
            return self.get_report()
        self._running = True
        try:
            from backend.providers.router import model_router
            providers = dict(model_router.providers)
            for name, provider in providers.items():
                try:
                    health = await self._probe_one(name, provider, verify=verify)
                except Exception as e:  # a single provider must never break discovery
                    health = ProviderHealth(name=name, health_status="unavailable",
                                            detail=f"probe error: {e}", probed_at=time.time())
                self._health[name] = health
                self._persist(health)
                self._persist_models(name, provider, health)
                self._apply_to_breaker(name, provider, health)
            self._last_run_ts = time.time()
            healthy = sum(1 for h in self._health.values() if h.health_status == "healthy")
            logger.info(f"[Discovery] Probed {len(self._health)} provider(s): {healthy} healthy.")
            return self.get_report()
        finally:
            self._running = False

    # -- probing ------------------------------------------------------------- #

    async def _probe_one(self, name: str, provider, *, verify: bool) -> ProviderHealth:
        import httpx

        health = ProviderHealth(name=name, probed_at=time.time())
        url, headers = self._catalog_request(name, provider)
        if not url:
            # No catalog endpoint known for this provider: fall back to the provider's own
            # availability check so we still report something truthful.
            try:
                health.reachable = bool(await provider.is_available())
                health.auth_ok = health.reachable
                health.health_status = "healthy" if health.reachable else "unavailable"
                health.detail = "no catalog endpoint; used provider.is_available()"
            except Exception as e:
                health.health_status = "unavailable"
                health.detail = str(e)
            return health

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
            health.latency_ms = (time.time() - t0) * 1000.0
            health.reachable = True
            if resp.status_code in (401, 403):
                health.auth_ok = False
                health.health_status = "unavailable"
                health.detail = f"auth failed (HTTP {resp.status_code})"
                return health
            if resp.status_code >= 400:
                health.health_status = "degraded"
                health.detail = f"catalog HTTP {resp.status_code}"
                return health
            health.auth_ok = True
            health.catalog_models = self._parse_catalog(name, resp)
            default_model = getattr(provider, "default_model", "") or ""
            if default_model and health.catalog_models:
                health.default_model_present = any(default_model in m or m in default_model
                                                   for m in health.catalog_models)
            health.health_status = "healthy"
            health.detail = f"{len(health.catalog_models)} models in catalog"
        except Exception as e:
            health.health_status = "unavailable"
            health.detail = f"unreachable: {e}"
            return health

        if verify and health.auth_ok:
            await self._verify_once(provider, health)
        return health

    def _catalog_request(self, name: str, provider):
        """Return (url, headers) for the provider's catalog endpoint, or (None, {})."""
        base = (getattr(provider, "base_url", "") or "").rstrip("/")
        api_key = getattr(provider, "api_key", "") or ""
        extra = dict(getattr(provider, "extra_headers", {}) or {})
        if name == "gemini":
            # base_url already ends in /models; listing is GET {base}?key=
            key = getattr(provider, "api_key", "") or ""
            if getattr(provider, "api_keys", None):
                key = provider.api_keys[0]
            return (f"{base}?key={key}", {})
        if getattr(provider, "account_id", None):  # Cloudflare Workers AI
            acct = provider.account_id
            return (f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/models/search",
                    {"Authorization": f"Bearer {api_key}"})
        if base:  # OpenAI-compatible
            headers = {"Authorization": f"Bearer {api_key}"}
            headers.update(extra)
            return (f"{base}/models", headers)
        return (None, {})

    def _parse_catalog(self, name: str, resp) -> List[str]:
        try:
            data = resp.json()
        except Exception:
            return []
        models: List[str] = []
        if name == "gemini":
            for m in data.get("models", []) or []:
                mid = (m.get("name") or "").split("/")[-1]
                if mid:
                    models.append(mid)
        elif isinstance(data, dict) and "result" in data:  # Cloudflare
            for m in data.get("result", []) or []:
                if m.get("name"):
                    models.append(m["name"])
        elif isinstance(data, dict) and "data" in data:  # OpenAI-compatible
            for m in data.get("data", []) or []:
                if m.get("id"):
                    models.append(m["id"])
        return models

    async def _verify_once(self, provider, health: ProviderHealth) -> None:
        """At most ONE minimal completion to prove callability (not just catalog presence)."""
        try:
            resp = await provider.generate_response(prompt="ping", capability="general_reasoning")
            if resp and not resp.is_refusal:
                health.detail += " | verified callable"
            else:
                reason = (getattr(resp, "refusal_reason", "") or "").lower()
                # A refusal that is a hard quota/model error downgrades health.
                if any(code in reason for code in ("404", "410", "not found", "gone")):
                    health.health_status = "degraded"
                    health.detail += " | default model not callable"
        except Exception as e:
            health.detail += f" | verify error: {e}"

    # -- side effects -------------------------------------------------------- #

    def _apply_to_breaker(self, name: str, provider, health: ProviderHealth) -> None:
        """Feed dead providers/models into the circuit breaker so routing drops them (B3)."""
        from backend.providers.quota_manager import quota_manager
        if health.health_status == "unavailable":
            quota_manager.blacklist_for_session(name, f"discovery: {health.detail}")
        elif health.default_model_present is False:
            default_model = getattr(provider, "default_model", "") or name
            quota_manager.trip_breaker(name, default_model, reason=f"404 default model '{default_model}' not in live catalog")

    def _persist_models(self, provider_name: str, provider, health: ProviderHealth) -> None:
        """Persist per-model discovery state to the models table (B2): present_in_catalog,
        callable, context_length, last_probe_ts for every model mapped to this provider.
        Best-effort; never raises."""
        try:
            import time as _t
            from backend.database.session import SessionLocal
            from backend.database.models import ModelConfigModel
            from backend.providers.router import model_router
            from backend.agent_runtime.context_budget import static_model_window

            mapped = [(alias, wire) for alias, (pn, wire) in model_router.MODEL_PROVIDER_MAP.items()
                      if pn == provider_name]
            if not mapped:
                return
            catalog = list(health.catalog_models or [])
            default_model = getattr(provider, "default_model", "") or ""
            now = _t.time()
            db = SessionLocal()
            try:
                seen = set()
                for _alias, wire in mapped:
                    if wire in seen:
                        continue
                    seen.add(wire)
                    # present_in_catalog: if we have a catalog, require a match; if the probe
                    # returned no catalog at all, don't claim the model is gone (leave True).
                    present = True if not catalog else any(wire in c or c in wire for c in catalog)
                    row = (db.query(ModelConfigModel)
                           .filter(ModelConfigModel.provider_name == provider_name,
                                   ModelConfigModel.model_name == wire).first())
                    if row is None:
                        row = ModelConfigModel(provider_name=provider_name, model_name=wire,
                                               capability="general_reasoning")
                        db.add(row)
                    row.present_in_catalog = present
                    # callable is only *proven* for the default model (the one verify() called)
                    # when the provider is healthy; otherwise fall back to catalog presence.
                    if wire == default_model:
                        row.callable = bool(present and health.health_status == "healthy")
                    else:
                        row.callable = present
                    row.context_length = static_model_window(wire)
                    row.last_probe_ts = now
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"[Discovery] per-model persist skip for {provider_name}: {e}")

    def _persist(self, health: ProviderHealth) -> None:
        """Persist provider health to the providers table (best-effort, never raises)."""
        try:
            from backend.database.session import SessionLocal
            from backend.database.models import ProviderConfigModel
            db = SessionLocal()
            try:
                row = db.query(ProviderConfigModel).filter(ProviderConfigModel.name == health.name).first()
                if row is None:
                    row = ProviderConfigModel(name=health.name)
                    db.add(row)
                row.latency_ms = health.latency_ms
                row.health_status = health.health_status
                row.api_key_configured = health.auth_ok
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"[Discovery] persist skip for {health.name}: {e}")


# Module-level singleton.
discovery_service = DiscoveryService()
