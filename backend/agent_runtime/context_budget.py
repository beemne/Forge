"""Per-model context budgeting + auto-compaction (Workstream D).

Context windows differ per model and were never tracked, so a long run could silently
overflow the window or get truncated by the provider. This module estimates the token
cost of the assembled prompt for the CURRENT model and decides when to compact.

Token estimation uses the same ~4-chars/token heuristic the providers already use for
usage accounting — deliberately cheap and dependency-free (no tokenizer download). It is
an estimate; the 80% threshold leaves headroom for its error.

Compaction itself is EXTRACTIVE and performed by the ContextBuilder: it trims the oldest
trajectory turns first and never touches the protected blocks (authoritative state, flag
candidates, evidence, failed-approach ledger, recovery directive) — the same
"never invent" discipline as checkpoint_pipeline.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("forge.agent_runtime.context_budget")

# Fraction of the model window at which we auto-compact.
COMPACT_THRESHOLD = 0.8

# Conservative fallback when a model's window is unknown.
FALLBACK_WINDOW = 32768

# Known windows for configured models (exact ids from MODEL_PROVIDER_MAP).
KNOWN_WINDOWS = {
    "qwen/qwen3.8-27b": 131072,
    "openai/gpt-oss-120b": 131072,
    "groq/compound": 131072,
    "gpt-5.4-mini": 131072,
    "DeepSeek-V3.2": 65536,
    "GPT-5-nano": 131072,
    "z-ai/glm-5.3-flash": 131072,
    "z-ai/glm-5.3": 131072,
    "deepseek/deepseek-chat": 65536,
    "@cf/meta/llama-3.1-8b-instruct": 131072,
    "gemini-3.6-flash": 1000000,
    "codestral-2508": 256000,
    "mistralai/codestral-2508": 256000,
}

# Family fallbacks matched as substrings (lowercased), most-specific first.
FAMILY_WINDOWS = [
    ("gemini", 1000000),
    ("codestral", 256000),
    ("glm", 131072),
    ("qwen", 131072),
    ("llama", 131072),
    ("gpt-oss", 131072),
    ("deepseek", 65536),
    ("mistral", 131072),
    ("ministral", 131072),
    ("nemotron", 131072),
]


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token)."""
    if not text:
        return 0
    return max(0, len(text) // 4)


def model_window(model_name: str) -> int:
    """Return the context window (in tokens) for *model_name*.

    Order: live/persisted catalog (ModelConfigModel.context_length) → known map →
    family fallback → FALLBACK_WINDOW. Never raises.
    """
    if not model_name:
        return FALLBACK_WINDOW
    # 1. Persisted/live catalog value (Workstream B stores context_length here).
    try:
        from backend.database.session import SessionLocal
        from backend.database.models import ModelConfigModel
        db = SessionLocal()
        try:
            row = (db.query(ModelConfigModel)
                   .filter(ModelConfigModel.model_name == model_name).first())
            if row and row.context_length and row.context_length > 1024:
                return int(row.context_length)
        finally:
            db.close()
    except Exception:
        pass
    # 2. Exact known window.
    if model_name in KNOWN_WINDOWS:
        return KNOWN_WINDOWS[model_name]
    # 3. Family fallback.
    low = model_name.lower()
    for frag, win in FAMILY_WINDOWS:
        if frag in low:
            return win
    return FALLBACK_WINDOW


def budget_tokens(model_name: str, threshold: float = COMPACT_THRESHOLD) -> int:
    """The token ceiling (threshold × window) at which compaction kicks in."""
    return int(model_window(model_name) * threshold)


def over_budget(system_instruction: str, user_prompt: str, model_name: str,
                threshold: float = COMPACT_THRESHOLD) -> bool:
    """True when the assembled prompt exceeds the model's compaction threshold."""
    total = estimate_tokens(system_instruction) + estimate_tokens(user_prompt)
    return total > budget_tokens(model_name, threshold)
