"""Normalization for chat-driven challenge creation (Workstream F4/F5).

Keeps the conversational flow tolerant of free-text: the operator can type
"binary exploitation" / "pwn" / "reverse engineering" and get a canonical category,
"insane"/"hard"/"easy" in any case for difficulty, and paste several targets separated
by commas / newlines / '+' and have them joined with the FORGE multi-target convention
(' + ', never commas).
"""
from __future__ import annotations

import re
from typing import List, Optional

# Canonical categories and the free-text synonyms that map onto them.
_CATEGORY_SYNONYMS = {
    "web": ["web", "web exploitation", "web exp", "http", "webapp", "web app"],
    "pwn": ["pwn", "binary exploitation", "binary exp", "binexp", "exploitation", "binary"],
    "rev": ["rev", "reverse", "reverse engineering", "reversing", "re"],
    "crypto": ["crypto", "cryptography", "crypto challenge"],
    "forensics": ["forensics", "forensic", "dfir"],
    "recon": ["recon", "reconnaissance", "osint", "enumeration"],
    "stego": ["stego", "steganography"],
    "misc": ["misc", "miscellaneous", "other", "general"],
    "mobile": ["mobile", "android", "ios"],
    "hardware": ["hardware", "hw", "embedded"],
    "ai": ["ai", "ml", "ai reverse", "machine learning", "llm"],
}

_DIFFICULTIES = ["EASY", "MEDIUM", "HARD", "INSANE"]
_DIFFICULTY_SYNONYMS = {
    "EASY": ["easy", "beginner", "trivial", "baby", "warmup", "low"],
    "MEDIUM": ["medium", "med", "normal", "moderate", "intermediate"],
    "HARD": ["hard", "difficult", "advanced", "high"],
    "INSANE": ["insane", "expert", "extreme", "elite", "very hard"],
}

CATEGORIES = tuple(_CATEGORY_SYNONYMS.keys())
DIFFICULTIES = tuple(_DIFFICULTIES)


def normalize_category(raw: Optional[str]) -> Optional[str]:
    """Map free text to a canonical category, or None if nothing recognizable."""
    if not raw:
        return None
    s = raw.strip().lower()
    if not s:
        return None
    for canon, syns in _CATEGORY_SYNONYMS.items():
        if s == canon or s in syns:
            return canon
    # Substring fallback: "a web login challenge" -> web.
    for canon, syns in _CATEGORY_SYNONYMS.items():
        for syn in syns:
            if syn in s:
                return canon
    return None


def normalize_difficulty(raw: Optional[str], default: Optional[str] = "MEDIUM") -> Optional[str]:
    """Map free text to EASY/MEDIUM/HARD/INSANE; returns *default* when unrecognizable."""
    if not raw:
        return default
    s = raw.strip().lower()
    if not s:
        return default
    up = s.upper()
    if up in _DIFFICULTIES:
        return up
    for canon, syns in _DIFFICULTY_SYNONYMS.items():
        if s in syns or any(syn in s for syn in syns):
            return canon
    return default


def normalize_targets(raw: Optional[str]) -> str:
    """Join one or more targets with the FORGE multi-target separator ' + '.

    Accepts comma-, newline-, or '+'-separated input and preserves order while
    dropping empties/duplicates. 'nc host port' style targets are kept intact.
    """
    if not raw:
        return ""
    parts = re.split(r"\s*(?:\+|,|\n)\s*", raw.strip())
    seen: List[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
    return " + ".join(seen)
