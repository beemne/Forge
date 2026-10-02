"""Durable circuit-breaker storage (Workstream A2).

The in-memory breaker in ``quota_manager`` used to reset to "all healthy" on every
launch, so a provider/model that was quota-exhausted or permission-denied got hammered
again immediately after a restart. This module persists breaker entries to the
``provider_breakers`` table and restores the still-active ones at startup.

All functions are best-effort and isolated here so ``quota_manager`` (a provider-layer
singleton created at import time) never imports the database layer directly — avoiding an
import cycle and keeping DB access lazy.
"""
from __future__ import annotations

import logging
import time
from typing import List, Tuple

logger = logging.getLogger("forge.providers.breaker_store")


def save_breaker(key: str, expiry_ts: float, reason: str) -> None:
    """Upsert one breaker entry. Raises nothing the caller must handle."""
    from backend.database.session import SessionLocal
    from backend.database.models import ProviderBreakerModel

    db = SessionLocal()
    try:
        row = db.get(ProviderBreakerModel, key)
        if row is None:
            db.add(ProviderBreakerModel(key=key, expiry_ts=float(expiry_ts), reason=(reason or "")[:500]))
        else:
            row.expiry_ts = float(expiry_ts)
            row.reason = (reason or "")[:500]
        db.commit()
    finally:
        db.close()


def load_active_breakers() -> List[Tuple[str, float, str]]:
    """Return [(key, expiry_ts, reason)] for entries that have not yet expired."""
    from backend.database.session import SessionLocal
    from backend.database.models import ProviderBreakerModel

    now = time.time()
    db = SessionLocal()
    try:
        rows = db.query(ProviderBreakerModel).filter(ProviderBreakerModel.expiry_ts > now).all()
        return [(r.key, float(r.expiry_ts), r.reason or "") for r in rows]
    finally:
        db.close()


def purge_expired() -> int:
    """Delete expired rows so the table does not grow unbounded. Returns rows removed."""
    from backend.database.session import SessionLocal
    from backend.database.models import ProviderBreakerModel

    now = time.time()
    db = SessionLocal()
    try:
        n = db.query(ProviderBreakerModel).filter(ProviderBreakerModel.expiry_ts <= now).delete()
        db.commit()
        return int(n or 0)
    finally:
        db.close()
