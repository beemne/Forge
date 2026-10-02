"""
FORGE Agent Runtime — session lifecycle (Step 2).

A session owns one mission end-to-end and is the unit that SURVIVES model/provider
changes, pauses, and process crashes. The durable row is ``AgentSessionModel``; the
live handle is :class:`AgentSession`, which wraps the resumable :class:`MissionState`.

Lifecycle: create · resume · pause · cancel · complete · checkpoint · restore.

State is persisted on the session row every turn (via :meth:`SessionManager.save`),
so a crash after turn N is recoverable by :meth:`resume`, which reloads the state and
lets the runtime continue from turn N+1 with zero information loss. :meth:`checkpoint`
additionally writes a ``CheckpointModel`` through the existing ``checkpoint_manager``
so the runtime integrates with FORGE's established checkpoint/restore system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from backend.utils.time import utcnow
from typing import Optional

from backend.database.session import SessionLocal
from backend.database.models import AgentSessionModel
from backend.agent_runtime.state import MissionState
from backend.agent_runtime.trajectory import trajectory_store

logger = logging.getLogger("forge.agent_runtime.session")


@dataclass
class AgentSession:
    """Live, in-memory handle over an AgentSessionModel row."""
    id: str
    run_id: Optional[str]
    challenge_id: Optional[str]
    agent_id: str
    engine: str
    status: str
    state: MissionState
    last_sequence: int = 0
    provider_name: str = ""
    model_name: str = ""
    verified_flag: Optional[str] = None
    outcome: Optional[str] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def objective(self) -> str:
        return self.state.current_objective

    @property
    def target_scope(self) -> str:
        return self.state.target

    def add_tokens(self, prompt: int, completion: int) -> None:
        self.total_prompt_tokens += int(prompt or 0)
        self.total_completion_tokens += int(completion or 0)


class SessionManager:
    """Create/resume/persist/finalize durable agent sessions."""

    TERMINAL = {"COMPLETED", "CANCELLED", "FAILED"}

    # ------------------------------------------------------------------ #
    # Create / resume
    # ------------------------------------------------------------------ #

    def create(
        self,
        *,
        challenge_id: Optional[str] = None,
        run_id: Optional[str] = None,
        target_scope: str = "",
        objective: str = "",
        agent_id: str = "orchestrator",
        engine: str = "runtime",
        challenge_name: str = "",
        category: str = "",
        difficulty: str = "",
        platform: str = "",
        description: str = "",
        flag_format: str = "",
    ) -> AgentSession:
        scope = [t.strip() for t in (target_scope or "").split("+") if t.strip()]
        state = MissionState(
            target=target_scope or "", scope=scope, challenge_name=challenge_name,
            category=category, difficulty=difficulty, platform=platform,
            description=description, flag_format=flag_format,
            current_objective=objective or "Find and extract the challenge flag.",
        )
        db = SessionLocal()
        try:
            row = AgentSessionModel(
                run_id=run_id, challenge_id=challenge_id, agent_id=agent_id, engine=engine,
                status="CREATED", phase=state.phase, objective=state.current_objective,
                target_scope=target_scope or "", state=state.to_dict(), last_sequence=0,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            sess = self._to_session(row, state)
        finally:
            db.close()

        trajectory_store.record(
            session_id=sess.id, event_type="SESSION_START", run_id=run_id,
            challenge_id=challenge_id, agent_id=agent_id, sequence=0,
            decision_summary=f"Session created for '{challenge_name or challenge_id}' "
                             f"(target: {target_scope or 'n/a'})",
        )
        logger.info(f"[SessionManager] Created session {sess.id} (challenge={challenge_id}, run={run_id})")
        return sess

    def resume(self, session_id: str) -> Optional[AgentSession]:
        """Reload a session + its state so the runtime can continue from the next turn."""
        db = SessionLocal()
        try:
            row = db.query(AgentSessionModel).filter(AgentSessionModel.id == session_id).first()
            if not row:
                return None
            state = MissionState.from_dict(row.state or {})
            row.status = "RUNNING"
            db.commit()
            sess = self._to_session(row, state)
            sess.status = "RUNNING"
        finally:
            db.close()
        logger.info(f"[SessionManager] Resumed session {session_id} at sequence {sess.last_sequence}")
        return sess

    def get(self, session_id: str) -> Optional[AgentSession]:
        db = SessionLocal()
        try:
            row = db.query(AgentSessionModel).filter(AgentSessionModel.id == session_id).first()
            if not row:
                return None
            return self._to_session(row, MissionState.from_dict(row.state or {}))
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Persist / transition
    # ------------------------------------------------------------------ #

    def save(self, sess: AgentSession, *, status: Optional[str] = None) -> None:
        """Persist the live state + counters back to the row (called each turn)."""
        db = SessionLocal()
        try:
            row = db.query(AgentSessionModel).filter(AgentSessionModel.id == sess.id).first()
            if not row:
                return
            if status:
                sess.status = status
                row.status = status
            row.phase = sess.state.phase
            row.objective = sess.state.current_objective
            row.state = sess.state.to_dict()
            row.last_sequence = sess.last_sequence
            row.provider_name = sess.provider_name or row.provider_name
            row.model_name = sess.model_name or row.model_name
            row.verified_flag = sess.verified_flag
            row.total_prompt_tokens = sess.total_prompt_tokens
            row.total_completion_tokens = sess.total_completion_tokens
            db.commit()
        except Exception as e:
            logger.warning(f"[SessionManager] save failed for {sess.id}: {e}")
            db.rollback()
        finally:
            db.close()

    def pause(self, sess: AgentSession) -> None:
        self.save(sess, status="PAUSED")
        logger.info(f"[SessionManager] Paused session {sess.id}")

    def cancel(self, sess: AgentSession) -> None:
        sess.outcome = sess.outcome or "cancelled"
        self._finalize(sess, "CANCELLED")

    def complete(self, sess: AgentSession, *, outcome: str = "success",
                 verified_flag: Optional[str] = None) -> None:
        if verified_flag:
            sess.verified_flag = verified_flag
        sess.outcome = outcome
        self._finalize(sess, "COMPLETED")
        trajectory_store.record(
            session_id=sess.id, event_type="SESSION_COMPLETE", run_id=sess.run_id,
            challenge_id=sess.challenge_id, agent_id=sess.agent_id,
            result=outcome.upper(), decision_summary=(
                f"Mission complete — flag verified: {sess.verified_flag}" if sess.verified_flag
                else f"Mission ended: {outcome}"),
        )

    def fail(self, sess: AgentSession, reason: str = "") -> None:
        sess.outcome = "failure"
        self._finalize(sess, "FAILED")
        trajectory_store.record(
            session_id=sess.id, event_type="SESSION_COMPLETE", run_id=sess.run_id,
            challenge_id=sess.challenge_id, agent_id=sess.agent_id, result="FAILED",
            decision_summary=f"Mission failed: {reason}"[:400],
        )

    def _finalize(self, sess: AgentSession, status: str) -> None:
        db = SessionLocal()
        try:
            row = db.query(AgentSessionModel).filter(AgentSessionModel.id == sess.id).first()
            if row:
                row.status = status
                row.outcome = sess.outcome
                row.verified_flag = sess.verified_flag
                row.state = sess.state.to_dict()
                row.last_sequence = sess.last_sequence
                # Persist the last provider/model actually used (may have changed mid-mission).
                row.provider_name = sess.provider_name or row.provider_name
                row.model_name = sess.model_name or row.model_name
                row.total_prompt_tokens = sess.total_prompt_tokens
                row.total_completion_tokens = sess.total_completion_tokens
                row.completed_at = utcnow()
                db.commit()
            sess.status = status
        except Exception as e:
            logger.warning(f"[SessionManager] finalize failed for {sess.id}: {e}")
            db.rollback()
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Checkpoint / restore (integrates with the existing checkpoint_manager)
    # ------------------------------------------------------------------ #

    def checkpoint(self, sess: AgentSession, last_action: str = "") -> None:
        """Persist state to the row AND (if a run exists) a CheckpointModel snapshot."""
        self.save(sess)
        if not sess.run_id:
            return
        try:
            from backend.checkpoints.manager import checkpoint_manager
            db = SessionLocal()
            try:
                checkpoint_manager.create_checkpoint(
                    db=db, run_id=sess.run_id, current_phase=sess.state.phase,
                    current_agent=sess.agent_id, last_action=last_action or "runtime turn",
                    state_snapshot={"agent_session_id": sess.id, "mission_state": sess.state.to_dict(),
                                    "last_sequence": sess.last_sequence},
                )
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"[SessionManager] checkpoint (CheckpointModel) skipped: {e}")

    def restore(self, session_id: str) -> Optional[AgentSession]:
        """Restore a session, preferring the richest available snapshot.

        The session row already holds current state; if a later CheckpointModel
        snapshot exists for the run it is merged in. Then behaves like resume.
        """
        sess = self.resume(session_id)
        if sess is None:
            return None
        if sess.run_id:
            try:
                from backend.checkpoints.manager import checkpoint_manager
                db = SessionLocal()
                try:
                    cp = checkpoint_manager.get_latest_checkpoint(db, sess.run_id)
                finally:
                    db.close()
                data = (cp.state_snapshot or {}).get("data", {}) if cp else {}
                snap = data.get("mission_state") if isinstance(data, dict) else None
                if snap and (data.get("last_sequence", 0) >= sess.last_sequence):
                    sess.state = MissionState.from_dict(snap)
                    sess.last_sequence = max(sess.last_sequence, data.get("last_sequence", 0))
            except Exception as e:
                logger.debug(f"[SessionManager] restore checkpoint merge skipped: {e}")
        return sess

    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_session(row: AgentSessionModel, state: MissionState) -> AgentSession:
        return AgentSession(
            id=row.id, run_id=row.run_id, challenge_id=row.challenge_id, agent_id=row.agent_id,
            engine=row.engine, status=row.status, state=state, last_sequence=row.last_sequence or 0,
            provider_name=row.provider_name or "", model_name=row.model_name or "",
            verified_flag=row.verified_flag, outcome=row.outcome,
            total_prompt_tokens=row.total_prompt_tokens or 0,
            total_completion_tokens=row.total_completion_tokens or 0,
        )


session_manager = SessionManager()
