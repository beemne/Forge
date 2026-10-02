"""The structured Task model (Phase 4 §3).

A Task is a unit of coordinated work the supervisor assigns to a specialist role.
Tasks are persisted (durable ``swarm_tasks`` table) so a mission can resume with
completed work still completed and in-flight work re-dispatched. Task state is an
explicit lifecycle:

    PENDING → READY → RUNNING → COMPLETED
                        │  └────→ FAILED → (retry → READY) | (reassign → REASSIGNED)
                        └───────→ CANCELLED
    BLOCKED (dependencies not yet COMPLETED)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from backend.utils.time import utcnow
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.swarm.dedup import task_signature


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REASSIGNED = "REASSIGNED"


TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.REASSIGNED}


@dataclass
class Task:
    mission_id: str
    role: str
    objective: str
    run_id: Optional[str] = None
    challenge_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    assigned_agent: str = ""
    priority: int = 50                      # higher = more urgent
    status: str = TaskStatus.PENDING.value
    dependencies: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    retry_count: int = 0
    timeout_seconds: int = 0
    signature: str = ""
    # Phase 4.x coordination hints. Persisted to swarm_tasks (Phase 4.x hardening
    # §5/§6) so the pre-dispatch capability gate and target-mismatch gate still fire
    # for a task reloaded after a checkpoint/resume — not silently dropped.
    required_capabilities: List[str] = field(default_factory=list)
    target_type: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    agent_session_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.signature:
            self.signature = task_signature(self.role, self.objective)

    # ── State helpers ────────────────────────────────────────────────── #

    @property
    def is_terminal(self) -> bool:
        return self.status in {s.value for s in TERMINAL_STATUSES} or self.status == TaskStatus.FAILED.value

    @property
    def is_active(self) -> bool:
        return self.status in (TaskStatus.READY.value, TaskStatus.RUNNING.value,
                               TaskStatus.PENDING.value, TaskStatus.BLOCKED.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "mission_id": self.mission_id, "run_id": self.run_id,
            "challenge_id": self.challenge_id, "parent_task_id": self.parent_task_id,
            "role": self.role, "assigned_agent": self.assigned_agent, "objective": self.objective,
            "priority": self.priority, "status": self.status, "dependencies": list(self.dependencies),
            "evidence_ids": list(self.evidence_ids), "retry_count": self.retry_count,
            "timeout_seconds": self.timeout_seconds, "signature": self.signature,
            "required_capabilities": list(self.required_capabilities or []),
            "target_type": self.target_type or "",
            "result": dict(self.result or {}), "failure_reason": self.failure_reason,
            "agent_session_id": self.agent_session_id, "created_at": self.created_at,
            "started_at": self.started_at, "completed_at": self.completed_at,
        }

    # ── Persistence (durable swarm_tasks table) ──────────────────────── #

    def save(self) -> None:
        try:
            from backend.database.session import SessionLocal
            from backend.database.models import SwarmTaskModel
            db = SessionLocal()
            try:
                row = db.query(SwarmTaskModel).filter(SwarmTaskModel.id == self.id).first()
                if row is None:
                    row = SwarmTaskModel(id=self.id)
                    db.add(row)
                row.mission_id = self.mission_id
                row.run_id = self.run_id
                row.challenge_id = self.challenge_id
                row.parent_task_id = self.parent_task_id
                row.role = self.role
                row.assigned_agent = self.assigned_agent or ""
                row.objective = self.objective or ""
                row.priority = int(self.priority)
                row.status = self.status
                row.dependencies = list(self.dependencies or [])
                row.evidence_ids = list(self.evidence_ids or [])
                row.retry_count = int(self.retry_count)
                row.timeout_seconds = int(self.timeout_seconds or 0)
                row.signature = self.signature
                row.result = dict(self.result or {})
                row.failure_reason = self.failure_reason or ""
                row.agent_session_id = self.agent_session_id
                row.required_capabilities = list(self.required_capabilities or [])
                row.target_type = self.target_type or ""
                row.started_at = _parse_dt(self.started_at)
                row.completed_at = _parse_dt(self.completed_at)
                db.commit()
            finally:
                db.close()
        except Exception:
            # Persistence is best-effort; in-memory scheduling still works without it.
            pass

    @classmethod
    def from_row(cls, row: Any) -> "Task":
        return cls(
            mission_id=row.mission_id, role=row.role or "recon", objective=row.objective or "",
            run_id=row.run_id, challenge_id=row.challenge_id, parent_task_id=row.parent_task_id,
            assigned_agent=row.assigned_agent or "", priority=row.priority or 50,
            status=row.status or TaskStatus.PENDING.value, dependencies=list(row.dependencies or []),
            evidence_ids=list(row.evidence_ids or []), retry_count=row.retry_count or 0,
            timeout_seconds=row.timeout_seconds or 0, signature=row.signature or "",
            result=dict(row.result or {}), failure_reason=row.failure_reason or "",
            required_capabilities=list(getattr(row, "required_capabilities", None) or []),
            target_type=(getattr(row, "target_type", "") or ""),
            agent_session_id=row.agent_session_id, id=row.id,
            created_at=(row.created_at.isoformat() if row.created_at else utcnow().isoformat()),
            started_at=(row.started_at.isoformat() if row.started_at else None),
            completed_at=(row.completed_at.isoformat() if row.completed_at else None),
        )


def _parse_dt(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None
