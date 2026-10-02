"""Challenge log path resolution — mirrored challenge file structure.

Single source of truth for WHERE a challenge's dedicated log file lives. The log
base stays the canonical FORGE logs directory (``backend/logs`` — the same base
``backend/main.py`` and every existing call site compute), but the log is now
stored inside a subtree that MIRRORS the challenge's own file structure:

    backend/logs/<Platform>/<Category>/<Difficulty>/<Name>/challenge_<id>.log

so the logs directory reflects the same hierarchy a challenge occupies under the
CTF workspace (e.g. ``<Platform>/<Category>/<Difficulty>/<Name>``) instead of a
single flat pile of ``challenge_<id>.log`` files.

Design
------
- ONE base definition, reused by every call site (routes, swarm orchestrator,
  CLI runner, orchestrator loop) so the location never drifts apart.
- ``register_challenge_log_path`` is called when the challenge metadata is known
  (challenge creation, swarm start) — it computes, creates, and CACHES the path.
- ``resolve_challenge_log_path`` is the hot-path lookup used by every log append:
  cache → DB metadata → existing-file search → legacy flat fallback. Cheap after
  the first resolution because the result is memoized per process.
- Purely a path helper: no demo/mock data, deterministic, non-fatal.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger("forge.challenge_paths")

# Canonical logs base: …/backend/logs  (this file lives at …/backend/utils/challenge_paths.py)
_LOGS_BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"))

# Memoized challenge_id -> absolute log path (warm for the whole process once resolved).
_PATH_CACHE: Dict[str, str] = {}

# Set of challenge_ids whose cached path is provisional (resolved with empty/unset metadata).
_PROVISIONAL_PATHS: set[str] = set()

# Characters allowed verbatim in a mirrored path segment; everything else (path
# separators, ``:`` drive markers, ``..`` traversal, control chars) is replaced.
_SAFE_SEGMENT_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.")


def logs_base() -> str:
    """Absolute canonical logs directory (``backend/logs``)."""
    return _LOGS_BASE


def sanitize_segment(value: str) -> str:
    """Sanitize a single path segment so it can never escape the logs base.

    Keeps alphanumerics, spaces, dash, underscore and dot; replaces anything else
    with ``_``. Rejects traversal (``.``/``..``) and collapses to empty when there
    is nothing usable, so the caller can skip it.
    """
    if not value:
        return ""
    cleaned = "".join(c if c in _SAFE_SEGMENT_CHARS else "_" for c in str(value)).strip()
    # Collapse runs of underscores/spaces introduced by substitution.
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip(" _")
    if cleaned in (".", ".."):
        return ""
    return cleaned[:120]


def _mirror_segments(platform: str, category: str, difficulty: str, name: str) -> list:
    """Ordered, sanitized, non-empty path segments mirroring the challenge structure."""
    segs = [sanitize_segment(platform), sanitize_segment(category),
            sanitize_segment(difficulty), sanitize_segment(name)]
    return [s for s in segs if s]


def register_challenge_log_path(challenge_id: str, platform: str = "", category: str = "",
                                difficulty: str = "", name: str = "") -> str:
    """Compute + create + cache the mirrored log path for a challenge.

    Called when the challenge metadata is known (creation / swarm start). The
    parent directory is created so the first append just opens the file.
    If early writes occurred to a provisional flat file, those lines are automatically
    migrated into the final mirrored file.
    """
    segments = _mirror_segments(platform, category, difficulty, name)
    parent = os.path.join(_LOGS_BASE, *segments) if segments else _LOGS_BASE
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as exc:
        logger.debug("[challenge_paths] mkdir failed for %s: %s", parent, exc)
        parent = _LOGS_BASE
        os.makedirs(parent, exist_ok=True)
    path = os.path.join(parent, f"challenge_{challenge_id}.log")

    # If metadata was non-empty and a provisional flat file exists on disk, migrate lines.
    flat_file = os.path.join(_LOGS_BASE, f"challenge_{challenge_id}.log")
    if segments and os.path.abspath(flat_file) != os.path.abspath(path):
        if os.path.isfile(flat_file):
            try:
                with open(flat_file, "r", encoding="utf-8", errors="replace") as f_flat:
                    flat_content = f_flat.read()
                if flat_content:
                    existing_content = ""
                    if os.path.isfile(path):
                        with open(path, "r", encoding="utf-8", errors="replace") as f_exist:
                            existing_content = f_exist.read()
                    with open(path, "w", encoding="utf-8") as f_target:
                        f_target.write(flat_content)
                        if existing_content and not flat_content.endswith("\n"):
                            f_target.write("\n")
                        if existing_content:
                            f_target.write(existing_content)
                # Remove provisional flat file now that lines are migrated
                os.remove(flat_file)
                logger.info(f"[challenge_paths] Migrated provisional log lines from '{flat_file}' to '{path}'")
            except OSError as exc:
                logger.warning(f"[challenge_paths] Failed migrating provisional log for {challenge_id}: {exc}")

    if not segments:
        _PROVISIONAL_PATHS.add(challenge_id)
    else:
        _PROVISIONAL_PATHS.discard(challenge_id)

    _PATH_CACHE[challenge_id] = path
    return path


def _resolve_from_db(challenge_id: str) -> Optional[str]:
    """Compute the mirrored path from the challenge row's metadata, if available."""
    try:
        from backend.database.session import SessionLocal
        from backend.database.models import ChallengeModel
        db = SessionLocal()
        try:
            ch = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
            if ch is not None:
                segments = _mirror_segments(
                    ch.platform_name or "", ch.category or "",
                    ch.difficulty or "", ch.name or "")
                if segments:
                    return register_challenge_log_path(
                        challenge_id, ch.platform_name or "", ch.category or "",
                        ch.difficulty or "", ch.name or "")
        finally:
            db.close()
    except Exception as exc:  # DB not ready / import cycle — non-fatal
        logger.debug("[challenge_paths] DB resolve skip for %s: %s", challenge_id, exc)
    return None


def _find_existing_log(challenge_id: str) -> Optional[str]:
    """Search the logs base for an already-written ``challenge_<id>.log`` (any depth).

    Handles a resume in a fresh process where the challenge metadata is no longer
    available but the mirrored log file already exists on disk.
    """
    fname = f"challenge_{challenge_id}.log"
    try:
        for root, _dirs, files in os.walk(_LOGS_BASE):
            if fname in files:
                return os.path.join(root, fname)
    except OSError:
        pass
    return None


def resolve_challenge_log_path(challenge_id: str) -> str:
    """Resolve the challenge log path (mirrored structure), creating the parent dir.

    Resolution order: process cache → challenge-row metadata → existing-file search
    → legacy flat ``backend/logs/challenge_<id>.log``. The result is memoized so the
    hot append path pays the lookup cost at most once per process.
    """
    cached = _PATH_CACHE.get(challenge_id)
    if cached and challenge_id not in _PROVISIONAL_PATHS:
        try:
            os.makedirs(os.path.dirname(cached), exist_ok=True)
        except OSError:
            pass
        return cached

    # If cached was provisional or missing, attempt DB resolution first
    resolved = _resolve_from_db(challenge_id)
    if resolved:
        try:
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
        except OSError:
            pass
        return resolved

    if cached:
        try:
            os.makedirs(os.path.dirname(cached), exist_ok=True)
        except OSError:
            pass
        return cached

    existing = _find_existing_log(challenge_id)
    if existing:
        _PATH_CACHE[challenge_id] = existing
        try:
            os.makedirs(os.path.dirname(existing), exist_ok=True)
        except OSError:
            pass
        return existing

    # Legacy flat fallback — keeps logging working even with zero metadata.
    os.makedirs(_LOGS_BASE, exist_ok=True)
    flat = os.path.join(_LOGS_BASE, f"challenge_{challenge_id}.log")
    _PATH_CACHE[challenge_id] = flat
    _PROVISIONAL_PATHS.add(challenge_id)
    return flat


def forget_challenge_log_path(challenge_id: str) -> None:
    """Drop the cache entry for a challenge (called on delete)."""
    _PATH_CACHE.pop(challenge_id, None)
    _PROVISIONAL_PATHS.discard(challenge_id)


def iter_candidate_log_paths(challenge_id: str) -> list:
    """All on-disk log paths that could belong to a challenge (mirrored + legacy flat).

    Used by deletion so both the mirrored file and any legacy flat file are removed.
    """
    candidates = []
    resolved = resolve_challenge_log_path(challenge_id)
    if resolved:
        candidates.append(resolved)
    found = _find_existing_log(challenge_id)
    if found and found not in candidates:
        candidates.append(found)
    flat = os.path.join(_LOGS_BASE, f"challenge_{challenge_id}.log")
    if flat not in candidates:
        candidates.append(flat)
    return candidates


