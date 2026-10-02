"""Timezone-safe UTC helpers.

``datetime.utcnow()`` is deprecated from Python 3.12 and returns a *naive*
datetime. FORGE stores naive UTC timestamps in SQLite (its ``DateTime`` columns
carry no tzinfo), and a large amount of code compares those stored values
against "now". Switching to an *aware* UTC datetime would silently introduce
aware/naive comparison errors across the codebase (the exact hazard flagged as
bug C2 in the project analysis).

``utcnow()`` therefore returns the SAME wall-clock value ``datetime.utcnow()``
produced — naive UTC — built the non-deprecated way. It is a drop-in
replacement: behaviour is identical, only the deprecation warning is gone.
"""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return naive UTC ``now`` — identical in value to the deprecated ``datetime.utcnow()``."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
