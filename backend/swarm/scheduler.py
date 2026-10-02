"""Dependency-aware task scheduler / queue (Phase 4 §4).

Deterministic and asyncio-friendly: the coordinator asks for the highest-priority
READY tasks (dependencies satisfied) up to the free concurrency slots, runs them,
and reports outcomes back. The scheduler owns task state transitions and their
persistence; it does NOT spawn agents itself (that is the coordinator's job) — this
keeps the scheduling policy small, pure, and unit-testable without any runtime.

State model:
    PENDING ─┬─(deps COMPLETED)────────────→ READY ─→ RUNNING ─→ COMPLETED/FAILED
             ├─(deps still active)──────────→ BLOCKED
             └─(a dependency is unsatisfiable)→ CANCELLED
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from backend.swarm.tasks import Task, TaskStatus
from backend.swarm.dedup import is_duplicate
from backend.utils.time import utcnow

logger = logging.getLogger("forge.swarm.scheduler")

_UNSATISFIABLE = {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value, TaskStatus.REASSIGNED.value}


class TaskScheduler:
    def __init__(self, *, persist: bool = True):
        self._tasks: Dict[str, Task] = {}
        self._order: List[str] = []              # insertion order (stable tiebreak)
        self.persist = persist

    # ── Registration ─────────────────────────────────────────────────── #

    def add(self, task: Task) -> bool:
        """Add a task unless a task with the same signature already exists (§8 dedup)."""
        if self.has_signature(task.signature):
            return False
        self._tasks[task.id] = task
        self._order.append(task.id)
        self._refresh_one(task)
        if self.persist:
            task.save()
        return True

    def has_signature(self, signature: str) -> bool:
        return is_duplicate(signature, (t.signature for t in self._tasks.values()))

    # ── Queries ───────────────────────────────────────────────────────── #

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def all(self) -> List[Task]:
        return [self._tasks[i] for i in self._order if i in self._tasks]

    def by_status(self, status: str) -> List[Task]:
        return [t for t in self.all() if t.status == status]

    def running_count(self) -> int:
        return len(self.by_status(TaskStatus.RUNNING.value))

    def active_count(self) -> int:
        return sum(1 for t in self.all() if t.is_active)

    def total_count(self) -> int:
        return len(self._tasks)

    def has_open_work(self) -> bool:
        """True while any task is PENDING/READY/BLOCKED/RUNNING."""
        return any(t.is_active for t in self.all())

    def ready_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """Recompute states, then return the highest-priority READY tasks."""
        self.refresh_states()
        ready = [t for t in self.all() if t.status == TaskStatus.READY.value]
        # priority desc, then insertion order (stable, deterministic).
        idx = {tid: n for n, tid in enumerate(self._order)}
        ready.sort(key=lambda t: (-int(t.priority), idx.get(t.id, 0)))
        return ready[:limit] if limit is not None else ready

    # ── State transitions ─────────────────────────────────────────────── #

    def refresh_states(self) -> None:
        for t in self.all():
            self._refresh_one(t)

    def _refresh_one(self, task: Task) -> None:
        if task.status in (TaskStatus.RUNNING.value, TaskStatus.COMPLETED.value,
                           TaskStatus.FAILED.value, TaskStatus.CANCELLED.value,
                           TaskStatus.REASSIGNED.value):
            return
        deps = [self._tasks.get(d) for d in (task.dependencies or [])]
        deps = [d for d in deps if d is not None]
        if any(d.status in _UNSATISFIABLE for d in deps):
            self._set(task, TaskStatus.CANCELLED.value,
                      failure_reason="A prerequisite task did not complete.")
            return
        if all(d.status == TaskStatus.COMPLETED.value for d in deps):
            if task.status != TaskStatus.READY.value:
                self._set(task, TaskStatus.READY.value)
        else:
            if task.status != TaskStatus.BLOCKED.value:
                self._set(task, TaskStatus.BLOCKED.value)

    def mark_running(self, task: Task, *, assigned_agent: str = "", session_id: Optional[str] = None) -> None:
        task.assigned_agent = assigned_agent or task.assigned_agent
        task.agent_session_id = session_id or task.agent_session_id
        task.started_at = task.started_at or utcnow().isoformat()
        self._set(task, TaskStatus.RUNNING.value)

    def mark_completed(self, task: Task, *, result: Optional[dict] = None,
                       evidence_ids: Optional[List[str]] = None) -> None:
        if result is not None:
            task.result = result
        if evidence_ids:
            task.evidence_ids = list(dict.fromkeys((task.evidence_ids or []) + evidence_ids))
        task.completed_at = utcnow().isoformat()
        self._set(task, TaskStatus.COMPLETED.value)

    def mark_failed(self, task: Task, *, reason: str = "") -> None:
        task.failure_reason = reason or task.failure_reason
        task.completed_at = utcnow().isoformat()
        self._set(task, TaskStatus.FAILED.value)

    def mark_cancelled(self, task: Task, *, reason: str = "") -> None:
        task.failure_reason = reason or task.failure_reason
        self._set(task, TaskStatus.CANCELLED.value)

    def mark_reassigned(self, task: Task, *, reason: str = "") -> None:
        task.failure_reason = reason or task.failure_reason
        self._set(task, TaskStatus.REASSIGNED.value)

    def requeue_for_retry(self, task: Task) -> None:
        """Reset a failed task to be re-attempted (retry_count already incremented)."""
        task.failure_reason = ""
        task.agent_session_id = None
        task.started_at = None
        task.completed_at = None
        # Back to PENDING so dependency refresh puts it in READY/BLOCKED correctly.
        self._set(task, TaskStatus.PENDING.value)
        self._refresh_one(task)

    def cancel_all_open(self, *, reason: str = "Mission stopped.") -> int:
        n = 0
        for t in self.all():
            if t.is_active:
                self.mark_cancelled(t, reason=reason)
                n += 1
        return n

    # ── Persistence / resume ──────────────────────────────────────────── #

    def _set(self, task: Task, status: str, **fields) -> None:
        task.status = status
        for k, v in fields.items():
            setattr(task, k, v)
        if self.persist:
            task.save()

    def load(self, mission_id: str) -> int:
        """Rehydrate tasks for a mission; stale RUNNING tasks are reset for re-dispatch.

        Running processes are never serialized as live (§15) — a task left RUNNING by
        an interrupted mission is treated as stale and reset to PENDING so it re-enters
        the queue. COMPLETED work stays completed.
        """
        try:
            from backend.database.session import SessionLocal
            from backend.database.models import SwarmTaskModel
            db = SessionLocal()
            try:
                rows = (db.query(SwarmTaskModel)
                        .filter(SwarmTaskModel.mission_id == mission_id)
                        .order_by(SwarmTaskModel.created_at.asc()).all())
                tasks = [Task.from_row(r) for r in rows]
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"[TaskScheduler] load skipped: {e}")
            return 0
        loaded = 0
        for t in tasks:
            if t.status == TaskStatus.RUNNING.value:
                t.status = TaskStatus.PENDING.value        # stale in-flight → re-dispatch
                t.agent_session_id = None
                t.started_at = None
            self._tasks[t.id] = t
            self._order.append(t.id)
            loaded += 1
        self.refresh_states()
        return loaded
