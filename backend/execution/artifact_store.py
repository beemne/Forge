"""
Artifact store for the FORGE execution layer.

Tracks files generated or downloaded during a mission with provenance metadata
and SHA-256 hashes.  Storage is local — no external cloud.

Provenance fields retained per artifact:
    path, sha256, size, created_at, session_id, agent_id,
    command, backend, workspace
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger("forge.execution.artifacts")

_HASH_CHUNK = 65_536  # 64 KiB


def sha256_file(path: str) -> Optional[str]:
    """Return the hex SHA-256 of *path*, or None if hashing fails."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        logger.debug(f"[ArtifactStore] sha256 failed for {path}: {exc}")
        return None


@dataclass
class ArtifactRecord:
    path: str
    session_id: str = ""
    agent_id: str = ""
    command: str = ""
    backend: str = ""
    workspace: str = ""
    sha256: Optional[str] = None
    size: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


class ArtifactStore:
    """
    Records artifacts produced during execution.

    Usage:
        store.record(path, session_id=..., agent_id=..., command=...)
        store.for_session(session_id) -> list of ArtifactRecord
    """

    def __init__(self) -> None:
        self._records: List[ArtifactRecord] = []

    def record(
        self,
        path: str,
        *,
        session_id: str = "",
        agent_id: str = "",
        command: str = "",
        backend: str = "",
        workspace: str = "",
        compute_hash: bool = True,
    ) -> ArtifactRecord:
        """
        Register *path* as an artifact.  Hash and size are computed only if the
        file exists; failure to hash does not prevent registration.
        """
        sha = None
        size = 0
        if os.path.isfile(path):
            size = os.path.getsize(path)
            if compute_hash:
                sha = sha256_file(path)

        rec = ArtifactRecord(
            path=os.path.abspath(path),
            session_id=session_id,
            agent_id=agent_id,
            command=command,
            backend=backend,
            workspace=workspace,
            sha256=sha,
            size=size,
        )
        self._records.append(rec)
        logger.debug(
            f"[ArtifactStore] recorded {os.path.basename(path)}"
            f" sha256={sha} size={size}"
        )
        return rec

    def for_session(self, session_id: str) -> List[ArtifactRecord]:
        return [r for r in self._records if r.session_id == session_id]

    def all_records(self) -> List[ArtifactRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()


# Module-level singleton.
artifact_store = ArtifactStore()
