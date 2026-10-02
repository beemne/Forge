"""Provider routing, the Playbook Vault, knowledge coverage observability, and the
FORGE experience-memory layer (learned from real runs only — no demo data)."""

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.models import KnowledgeEntryModel, ProviderUsageModel
from backend.database.session import get_db
from backend.providers.router import model_router
from backend.providers.snippet_parser import SnippetParser

router = APIRouter()
logger = logging.getLogger("forge.routes")


# ----------------------------------------------------
# PROVIDERS & MODEL ROUTER
# ----------------------------------------------------

class ParseSnippetRequest(BaseModel):
    snippet: str

class RegisterSnippetRequest(BaseModel):
    snippet: Optional[str] = None
    provider_name: Optional[str] = "nvidia"
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    base_url: Optional[str] = None
    test_connection: bool = True

class UpdateProviderKeyRequest(BaseModel):
    provider_name: str
    api_key: str
    model_id: Optional[str] = None
    base_url: Optional[str] = None


@router.get("/providers")
def get_providers(db: Session = Depends(get_db)):
    from backend.providers.quota_manager import quota_manager
    in_claude_window = quota_manager.is_in_claude_allowed_window()
    next_batch_str = quota_manager.get_next_batch_time_str()

    # Pre-aggregate usage counts and last errors per provider
    usage_counts = {}
    last_errors = {}
    try:
        from sqlalchemy import func
        counts = db.query(ProviderUsageModel.provider_name, func.count(ProviderUsageModel.id)).group_by(ProviderUsageModel.provider_name).all()
        usage_counts = {pname: cnt for pname, cnt in counts}

        err_rows = (db.query(ProviderUsageModel)
                    .filter(ProviderUsageModel.success == False)
                    .order_by(ProviderUsageModel.timestamp.desc())
                    .all())
        for er in err_rows:
            if er.provider_name not in last_errors:
                last_errors[er.provider_name] = f"Last failure at {er.timestamp.strftime('%H:%M:%S')}" if er.timestamp else "Request error"
    except Exception as e:
        logger.debug(f"[get_providers] DB aggregate skip: {e}")

    # Derive fallback priorities from router routing map order
    priority_order = model_router.DEFAULT_ROUTING_MAP.get("general_reasoning", [])

    provider_list = []
    for idx, p in enumerate(model_router.providers.values()):
        models_under_provider = [m for m, (pname, _) in model_router.MODEL_PROVIDER_MAP.items() if pname == p.name]
        is_quota_limited = any(quota_manager.is_quota_limited_model(m) for m in models_under_provider)

        quota_label = "100% Available"
        if is_quota_limited:
            if in_claude_window:
                quota_label = "3h Window Active (Trial Batch)"
            else:
                quota_label = f"Standby (Next Reset: {next_batch_str})"
        elif p.name == "agentrouter_codex":
            quota_label = "∞ Always Available (No Limit)"

        # Fallback priority index (1-based index in router priority list, or default)
        priority = (priority_order.index(p.name) + 1) if p.name in priority_order else (idx + 10)
        req_count = usage_counts.get(p.name, getattr(p, "request_count", 0))
        err_msg = last_errors.get(p.name, getattr(p, "last_error", "None"))

        provider_list.append({
            "name": p.name,
            "is_paid": p.is_paid,
            "status": "HEALTHY",
            "models": models_under_provider,
            "default_model": getattr(p, "default_model", getattr(p, "cli_binary_default", p.name)),
            "quota": quota_label,
            "is_quota_limited": is_quota_limited,
            "in_batch_window": in_claude_window if is_quota_limited else True,
            "transport": "CLI" if "agentrouter" in p.name else "API",
            "requests": req_count,
            "last_error": err_msg,
            "fallback_priority": priority
        })

    # Surface providers that have recorded usage/errors but are not currently
    # registered in the router (e.g. no key loaded this session, or a key was
    # rotated out). Without this, the dashboard's provider panel goes blank
    # whenever no live key is configured even though real call history exists.
    registered = {entry["name"] for entry in provider_list}
    seen_usage = set(usage_counts) | set(last_errors)
    for idx, pname in enumerate(sorted(seen_usage - registered)):
        provider_list.append({
            "name": pname,
            "is_paid": False,
            "status": "INACTIVE",
            "models": [m for m, (pn, _) in model_router.MODEL_PROVIDER_MAP.items() if pn == pname],
            "default_model": pname,
            "quota": "Not Configured",
            "is_quota_limited": False,
            "in_batch_window": True,
            "transport": "API",
            "requests": usage_counts.get(pname, 0),
            "last_error": last_errors.get(pname, "None"),
            "fallback_priority": len(provider_list) + idx + 1,
        })
    return provider_list


@router.get("/providers/health")
def get_providers_health():
    from backend.providers.quota_manager import quota_manager
    in_claude_window = quota_manager.is_in_claude_allowed_window()

    return {
        "paid_allowed": model_router.paid_allowed,
        "budget_usd": model_router.daily_budget_usd,
        "spent_usd": model_router.current_spent_usd,
        "in_claude_allowed_window": in_claude_window,
        "next_batch_replenishment": quota_manager.get_next_batch_time_str(),
        "registered_models": [
            {
                "model_id": model_id,
                "provider": p_info[0],
                "transport": p_info[1],
                "is_quota_limited": quota_manager.is_quota_limited_model(model_id),
                "is_always_available": quota_manager.is_always_available_model(model_id) or not quota_manager.is_quota_limited_model(model_id),
                "status": "ACTIVE" if (not quota_manager.is_quota_limited_model(model_id) or in_claude_window) else "STANDBY (Outside 3h Window)"
            }
            for model_id, p_info in model_router.MODEL_PROVIDER_MAP.items()
        ],
        "providers": [
            {
                "name": p.name,
                "is_paid": p.is_paid,
                "status": "HEALTHY",
                "default_model": getattr(p, "default_model", ""),
                "latency_ms": 120 if "cerebras" in p.name else (680 if "codex" in p.name else 420)
            }
            for p in model_router.providers.values()
        ]
    }

@router.get("/providers/discovery")
def get_provider_discovery():
    """Live provider/model availability report from the last catalog probe (Workstream B5):
    which providers are reachable/auth-ok, catalog model counts, and whether each provider's
    default model is actually present. Wired into the Providers UI page."""
    from backend.providers.discovery import discovery_service
    return discovery_service.get_report()


@router.post("/providers/discover")
async def trigger_provider_discovery(verify: bool = True):
    """Manually re-probe every provider's live catalog now (B1 'on demand'). Returns the
    fresh report. verify=false skips the one-completion callability check to save quota."""
    from backend.providers.discovery import discovery_service
    return await discovery_service.discover_all(verify=verify)


@router.post("/providers/parse-snippet")
def parse_provider_snippet(req: ParseSnippetRequest):
    result = SnippetParser.parse_snippet(req.snippet)
    return result

@router.post("/providers/register-snippet")
async def register_provider_snippet(req: RegisterSnippetRequest):
    if req.snippet:
        parsed = SnippetParser.parse_snippet(req.snippet)
        if not parsed.get("success"):
            raise HTTPException(status_code=400, detail="Could not parse API key or model from snippet.")
        api_key = parsed.get("api_key")
        model_id = parsed.get("model") or req.model_id or "default"
        base_url = parsed.get("base_url") or req.base_url or "https://integrate.api.nvidia.com/v1"
        provider_name = parsed.get("provider_name") or req.provider_name or "nvidia"
    else:
        if not req.api_key:
            raise HTTPException(status_code=400, detail="API key is required.")
        api_key = req.api_key
        model_id = req.model_id or "default"
        base_url = req.base_url or "https://integrate.api.nvidia.com/v1"
        provider_name = req.provider_name or "nvidia"

    # Register in router
    provider = model_router.register_custom_model(
        provider_name=provider_name,
        api_key=api_key,
        model_id=model_id,
        base_url=base_url
    )

    test_status = "untested"
    latency_ms = 0
    test_response = ""

    if req.test_connection:
        start_time = asyncio.get_event_loop().time()
        try:
            res = await provider.generate_response(
                prompt="Say 'pong'",
                model=model_id,
                capability="general_reasoning"
            )
            latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            if res.is_refusal:
                test_status = "failed"
                test_response = res.refusal_reason or "Refusal"
            else:
                test_status = "healthy"
                test_response = res.content[:200]
        except Exception as e:
            test_status = "failed"
            test_response = str(e)

    return {
        "success": True,
        "provider_name": provider_name,
        "model_id": model_id,
        "base_url": base_url,
        "api_key_masked": api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***",
        "test_status": test_status,
        "latency_ms": latency_ms,
        "test_response": test_response
    }

@router.post("/providers/update-key")
async def update_provider_key(req: UpdateProviderKeyRequest):
    p_name = req.provider_name.lower()
    if p_name in model_router.providers:
        prov = model_router.providers[p_name]
        if hasattr(prov, 'api_key'):
            prov.api_key = req.api_key
        if req.model_id and hasattr(prov, 'default_model'):
            prov.default_model = req.model_id
        if req.base_url and hasattr(prov, 'base_url'):
            prov.base_url = req.base_url.rstrip("/")
    else:
        model_router.register_custom_model(
            provider_name=p_name,
            api_key=req.api_key,
            model_id=req.model_id or "default",
            base_url=req.base_url or "https://integrate.api.nvidia.com/v1"
        )
    return {"success": True, "provider_name": req.provider_name, "status": "updated"}


# ----------------------------------------------------
# KNOWLEDGE ENTRIES
# ----------------------------------------------------

@router.get("/knowledge")
def list_knowledge(db: Session = Depends(get_db)):
    return db.query(KnowledgeEntryModel).all()


# ----------------------------------------------------
# AGENTROUTER QUOTA STATUS
# ----------------------------------------------------

@router.get("/providers/quota-status")
def get_agentrouter_quota_status():
    """
    Returns the current AgentRouter quota status.

    AgentRouter provides Claude & GPT models on a limited daily quota,
    released in 2 batches per day (Beijing 07:00/19:00, UTC 23:00/11:00).
    When a batch is exhausted, HTTP 402 is returned.
    DeepSeek & GLM models are always available (no quota limit).
    """
    return model_router.get_quota_status()

# ----------------------------------------------------
# PLAYBOOK VAULT (Phase 2)
# ----------------------------------------------------

class IngestWriteupRequest(BaseModel):
    text: str
    category: str = "web"
    source_type: str = "raw_text"  # "url", "raw_text", or "file"
    title: Optional[str] = None
    auto_approve: bool = True

class SearchPlaybooksRequest(BaseModel):
    query: str
    category: Optional[str] = None
    top_k: int = 5
    include_unpromoted: bool = False


@router.post("/playbooks/ingest")
async def ingest_writeup(req: IngestWriteupRequest):
    """Ingest a CTF writeup (from URL, raw text, or markdown file) into the Playbook Vault."""
    from backend.knowledge.ingest_writeup import ingest_url, ingest_raw_text

    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Source text or URL is required.")

    source_str = req.text.strip()
    if req.source_type == "url" or source_str.startswith("http://") or source_str.startswith("https://"):
        try:
            pb = ingest_url(source_str, category=req.category, title=req.title)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch and parse URL: {str(e)}")
    else:
        pb = ingest_raw_text(source_str, category=req.category, title=req.title)

    return {
        "status": "INGESTED",
        "playbook_id": pb.id,
        "category": pb.category,
        "tags": pb.tags,
        "is_promoted": pb.is_promoted,
        "playbook": pb.model_dump()
    }

@router.post("/playbooks/search")
def search_playbooks(req: SearchPlaybooksRequest):
    """Search the Playbook Vault using FTS5 index."""
    from backend.knowledge.playbook_vault import playbook_vault
    results = playbook_vault.search_playbooks(req.query, req.category, req.top_k, req.include_unpromoted)
    return {
        "query": req.query,
        "count": len(results),
        "playbooks": [pb.model_dump() for pb in results]
    }

@router.get("/playbooks")
def list_playbooks():
    """List all indexed playbooks from the vault."""
    from backend.knowledge.playbook_vault import playbook_vault
    playbook_vault.reload_index()
    all_pbs = []
    import yaml
    for root, _, files in os.walk(playbook_vault.base_dir):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict) and "id" in data:
                            all_pbs.append(data)
                except Exception:
                    pass
    return {"count": len(all_pbs), "playbooks": all_pbs}


# -------------------------------------------------------------------
# KNOWLEDGE COVERAGE OBSERVABILITY
# -------------------------------------------------------------------

COVERAGE_CATEGORIES = [
    "web", "pwn", "reverse", "crypto", "forensics",
    "osint", "network", "mobile", "cloud", "hardware", "ai_llm", "misc"
]

CATEGORY_ALIASES = {
    "web": "web", "pwn": "pwn", "binary": "pwn", "bof": "pwn",
    "reverse": "reverse", "rev": "reverse", "reversing": "reverse",
    "crypto": "crypto", "cryptography": "crypto",
    "forensics": "forensics", "stego": "forensics", "steganography": "forensics", "memory": "forensics",
    "osint": "osint", "recon": "osint",
    "network": "network", "pcap": "network", "wireshark": "network",
    "mobile": "mobile", "android": "mobile", "ios": "mobile",
    "cloud": "cloud", "aws": "cloud", "azure": "cloud", "gcp": "cloud",
    "hardware": "hardware", "iot": "hardware", "embedded": "hardware",
    "ai_llm": "ai_llm", "ai": "ai_llm", "llm": "ai_llm", "ml": "ai_llm",
    "misc": "misc", "auto_generated": "misc", "pending_review": "misc"
}

def _normalize_category(raw_cat: str) -> str:
    return CATEGORY_ALIASES.get(raw_cat.lower().strip(), "misc")


@router.get("/knowledge/coverage")
def get_knowledge_coverage():
    """Returns per-category coverage stats for the Knowledge Coverage observability view."""
    import yaml as _yaml
    from backend.knowledge.playbook_vault import playbook_vault

    # Initialize empty stats for all 12 categories
    stats = {}
    for cat in COVERAGE_CATEGORIES:
        stats[cat] = {
            "category": cat,
            "total": 0,
            "by_source": {},
            "by_confidence_tier": {"pending": 0, "low": 0, "trusted": 0},
            "tags": {},  # tag -> count
        }

    # Scan all playbook YAML files
    for root, _, files in os.walk(playbook_vault.base_dir):
        for file in files:
            if not (file.endswith(".yaml") or file.endswith(".yml")):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = _yaml.safe_load(f)
                if not isinstance(data, dict) or "id" not in data:
                    continue

                raw_cat = data.get("category", "misc")
                cat = _normalize_category(raw_cat)
                source = data.get("source", "ingested")
                conf = float(data.get("confidence_score", 0.5))
                tags = data.get("tags", [])

                bucket = stats[cat]
                bucket["total"] += 1

                # Source breakdown
                bucket["by_source"][source] = bucket["by_source"].get(source, 0) + 1

                # Confidence tier
                if conf < 0.5:
                    bucket["by_confidence_tier"]["pending"] += 1
                elif conf < 0.8:
                    bucket["by_confidence_tier"]["low"] += 1
                else:
                    bucket["by_confidence_tier"]["trusted"] += 1

                # Tag frequency
                if isinstance(tags, list):
                    for tag in tags:
                        tag_str = str(tag).lower().strip()
                        if tag_str:
                            bucket["tags"][tag_str] = bucket["tags"].get(tag_str, 0) + 1
            except Exception:
                continue

    # Build sorted response (lowest coverage first)
    categories_list = sorted(stats.values(), key=lambda c: c["total"])

    # Convert tag dicts to sorted lists for frontend
    for cat_data in categories_list:
        tag_dict = cat_data.pop("tags")
        cat_data["distinct_tags"] = sorted(
            [{"tag": t, "count": c} for t, c in tag_dict.items()],
            key=lambda x: x["count"],
            reverse=True
        )

    grand_total = sum(c["total"] for c in categories_list)

    return {
        "grand_total": grand_total,
        "categories": categories_list
    }


# =============================================================================
# EXPERIENCE MEMORY — FORGE-learned experience layer (§14)
# Read/search/feedback over experiences distilled from real runs. No demo data:
# every row originates from a verified/failed FORGE execution.
# =============================================================================

class MemorySearchRequest(BaseModel):
    query: str = ""
    category: Optional[str] = None
    top_k: int = 6
    include_failures: bool = True


class MemoryFeedbackRequest(BaseModel):
    success: bool
    note: str = ""
    run_id: Optional[str] = None
    challenge_id: Optional[str] = None


@router.get("/memory")
def list_memory(limit: int = Query(100), category: Optional[str] = None, outcome: Optional[str] = None):
    """Memory dashboard payload: aggregate stats + the experience list (§15)."""
    from backend.knowledge.experience_memory import experience_memory
    return {
        "stats": experience_memory.get_stats(),
        "experiences": experience_memory.list_experiences(limit=limit, category=category, outcome=outcome),
    }


@router.get("/memory/stats")
def memory_stats():
    """Aggregate memory counters + leaderboards for the Memory UI header (§15)."""
    from backend.knowledge.experience_memory import experience_memory
    return experience_memory.get_stats()


@router.post("/memory/search")
def search_memory(req: MemorySearchRequest):
    """Unified memory search (FORGE experience + reference playbooks), ranked (§6, §7)."""
    from backend.knowledge.memory_retriever import memory_retriever
    memories = memory_retriever.retrieve(
        query=req.query, category=req.category, top_k=req.top_k, include_failures=req.include_failures
    )
    return {"query": req.query, "count": len(memories), "memories": [m.model_dump() for m in memories]}


@router.get("/memory/technique-stats")
def technique_statistics(
    technique: str = Query(..., description="Technique/strategy label to look up"),
    category: Optional[str] = Query(None),
    target_type: Optional[str] = Query(None),
    technologies: Optional[str] = Query(None, description="Comma-separated technology list"),
):
    """Phase 6 §9/§13 — global + contextual success statistics for a learned technique.

    Answers "how well has this technique worked, overall vs. against targets like the
    current one?" — the observability behind why a memory-sourced candidate scored the
    way it did. Deterministic aggregation over stored experiences; no demo data.
    """
    from backend.knowledge.technique_stats import technique_stats
    techs = [t.strip() for t in (technologies or "").split(",") if t.strip()]
    return technique_stats.lookup(technique, category=category, target_type=target_type,
                                  technologies=techs)


@router.get("/memory/{experience_id}")
def get_memory(experience_id: str):
    """Full experience detail incl. attempts + usage log + provenance (§13)."""
    from backend.knowledge.experience_memory import experience_memory
    exp = experience_memory.get(experience_id, with_children=True)
    if not exp:
        raise HTTPException(status_code=404, detail="Experience not found")
    return exp


@router.post("/memory/{experience_id}/feedback")
def memory_feedback(experience_id: str, req: MemoryFeedbackRequest):
    """Record whether a retrieved memory actually helped — updates its stats (§12)."""
    from backend.knowledge.experience_memory import experience_memory
    ok = experience_memory.record_feedback(
        experience_id, success=req.success, note=req.note, run_id=req.run_id, challenge_id=req.challenge_id
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Experience not found")
    return {"status": "RECORDED", "experience_id": experience_id, "success": req.success}


@router.get("/experiences")
def list_experiences_alias(limit: int = Query(100), category: Optional[str] = None, outcome: Optional[str] = None):
    """Alias returning the raw experience list (§14)."""
    from backend.knowledge.experience_memory import experience_memory
    return experience_memory.list_experiences(limit=limit, category=category, outcome=outcome)
