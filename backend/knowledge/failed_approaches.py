"""Shared cross-mission failed-approach record (Workstream A4).

Both swarm engines call these helpers so a technique that repeatedly dead-ended in
earlier missions is surfaced (advisorily) in later ones. This is deliberately a
light, string-keyed signal — NOT a hard global block: a per-mission hard block already
lives in each engine (blocked_failure_sigs / mission.failed_approaches). Keeping the
cross-mission layer advisory avoids the trap where a generic command shape
("curl -s <URL>") gets globally banned and breaks every future run.

All functions are best-effort and never raise — recording/recalling cross-mission
memory must never break a live run.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger("forge.knowledge.failed_approaches")


def record_failed_approach(category: str, signature: str, reason: str = "") -> None:
    """Upsert a (category, signature) failure, incrementing its count. Never raises."""
    if not signature:
        return
    try:
        from backend.database.session import SessionLocal
        from backend.database.models import FailedApproachModel
        cat = (category or "").strip().lower()
        sig = signature.strip()[:300]
        db = SessionLocal()
        try:
            row = (db.query(FailedApproachModel)
                   .filter(FailedApproachModel.category == cat,
                           FailedApproachModel.signature == sig).first())
            if row is None:
                db.add(FailedApproachModel(category=cat, signature=sig,
                                           fail_count=1, last_reason=(reason or "")[:300]))
            else:
                row.fail_count = (row.fail_count or 0) + 1
                row.last_reason = (reason or "")[:300]
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"[FailedApproaches] record skip: {e}")


def recall_failed_approaches(category: str, limit: int = 6) -> List[Tuple[str, int, str]]:
    """Return up to *limit* (signature, fail_count, last_reason) for a category,
    most-failed first. Empty on any error."""
    try:
        from backend.database.session import SessionLocal
        from backend.database.models import FailedApproachModel
        cat = (category or "").strip().lower()
        db = SessionLocal()
        try:
            rows = (db.query(FailedApproachModel)
                    .filter(FailedApproachModel.category == cat)
                    .order_by(FailedApproachModel.fail_count.desc())
                    .limit(limit).all())
            return [(r.signature, int(r.fail_count or 1), r.last_reason or "") for r in rows]
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"[FailedApproaches] recall skip: {e}")
        return []


def recall_block(category: str, limit: int = 6) -> str:
    """Formatted advisory block for prompt injection, or '' when there is nothing."""
    rows = recall_failed_approaches(category, limit=limit)
    if not rows:
        return ""
    lines = ["CROSS-MISSION DEAD-ENDS (failed in prior missions of this category — avoid repeating):"]
    for sig, count, reason in rows:
        entry = f"  ✗ {sig[:120]} (failed x{count})"
        if reason:
            entry += f" — {reason[:100]}"
        lines.append(entry)
    return "\n".join(lines)
