"""Audit-log consistency tests for the per-command privilege approval workflow.

Regression coverage for the split between classification-time logging and the
async operator decision:

  * ``PrivilegeManager.evaluate_privilege_ex`` writes the AuditLogModel row when a
    PRIVILEGED/DANGEROUS command is FIRST classified — necessarily as
    ``approved=False``, because no operator decision exists yet.
  * The swarm worker later learns the REAL decision (approve / deny / timeout) and
    calls ``record_privilege_decision`` to reconcile that same row.

Before the fix, the classification-time row was committed as ``approved=False`` and
never updated, so a command that actually ran with operator approval left a
contradictory audit trail (AuditLogModel said denied; ToolExecutionModel said
approved). These tests drive the real ``_agent_worker`` loop end-to-end so the
actual wiring — not just the manager methods — is exercised.

Isolation (project rule #5): every write targets the isolated ``test_forge.db``.
"""

import asyncio
import os
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.swarm_orchestrator import SwarmOrchestrator
from backend.agents.swarm_state import SwarmBlackboard
from backend.database.models import AuditLogModel, ToolExecutionModel
from backend.database.session import SessionLocal, init_db
from backend.privilege.classify import classify_command_privilege
from backend.privilege.manager import PrivilegeManager

def _make_llm_response(command: str):
    """A model response that emits exactly one bash command with the required
    STRATEGY: prefix, and is not a refusal."""
    resp = MagicMock()
    resp.is_refusal = False
    resp.refusal_reason = ""
    resp.content = f"STRATEGY: exploit\n```bash\n{command}\n```"
    resp.model_name = "test-model"
    return resp

def _make_exec_result():
    """A successful tool-execution result the worker can record without error."""
    res = MagicMock()
    res.stdout = "audit-recon-output"
    res.stderr = ""
    res.exit_code = 0
    res.status = "SUCCESS"
    res.execution_failure = False
    res.failure_category = None
    res.tool_name = "bash"
    res.duration_ms = 1.0
    return res

class TestAuditLogReconciliation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Idempotent: guarantees audit_logs / tool_executions exist in test_forge.db.
        init_db()

    async def _resolve_when_pending(self, orchestrator, board, decision, timeout_s=20.0):
        """Poll the board until the worker registers a pending approval, then deliver
        the operator's decision via the real submit_approval_response path."""
        deadline_iters = int(timeout_s / 0.05)
        for _ in range(deadline_iters):
            if board.pending_approvals:
                req_id = next(iter(board.pending_approvals))
                return await orchestrator.submit_approval_response(req_id, decision)
            await asyncio.sleep(0.05)
        return None

    # ── Sanity: manager methods write then reconcile the SAME row ────────────────

    def test_manager_writes_then_reconciles_same_row(self):
        """evaluate_privilege_ex returns the row id; record_privilege_decision flips
        that exact row from False to True. Direct unit check of the mechanism."""
        manager = PrivilegeManager()
        agent = f"unit_recon_{uuid.uuid4().hex[:8]}"
        db = SessionLocal()
        try:
            approved, audit_id = manager.evaluate_privilege_ex(
                agent=agent, tool_name="sqlmap", privilege_level="PRIVILEGED", db=db
            )
            self.assertFalse(approved)
            self.assertTrue(audit_id, "evaluate_privilege_ex must return the audit row id")

            # Row starts False (no decision yet).
            row = db.query(AuditLogModel).filter(AuditLogModel.id == audit_id).first()
            self.assertIsNotNone(row)
            self.assertFalse(bool(row.approved))

            # Reconcile to approved=True and confirm it updated in place (no new row).
            ok = manager.record_privilege_decision(audit_id, True, db)
            self.assertTrue(ok)
            db.expire_all()
            row = db.query(AuditLogModel).filter(AuditLogModel.id == audit_id).first()
            self.assertTrue(bool(row.approved))
            total = db.query(AuditLogModel).filter(AuditLogModel.agent == agent).count()
            self.assertEqual(total, 1, "Reconciliation must UPDATE the row, not insert a new one")
        finally:
            db.close()

    def test_manager_missing_id_is_safe_noop(self):
        """record_privilege_decision tolerates a missing/None id without raising."""
        manager = PrivilegeManager()
        db = SessionLocal()
        try:
            self.assertFalse(manager.record_privilege_decision(None, True, db))
            self.assertFalse(manager.record_privilege_decision("does-not-exist", True, db))
        finally:
            db.close()

    # ── Backward-compat: bare-bool API unchanged for existing callers ────────────

    def test_bare_bool_api_preserved(self):
        """evaluate_privilege still returns a plain bool (acquisition.py, harness and
        classification tests depend on this) and SAFE remains immediately approved."""
        manager = PrivilegeManager()
        db = SessionLocal()
        try:
            self.assertIs(
                manager.evaluate_privilege(agent="a", tool_name="nmap", privilege_level="SAFE", db=db),
                True,
            )
            self.assertIs(
                manager.evaluate_privilege(agent="a", tool_name="sqlmap", privilege_level="PRIVILEGED", db=db),
                False,
            )
        finally:
            db.close()

    # ── Acceptance criterion #1: approve → AuditLogModel row shows approved=True ──

    async def test_audit_row_reconciled_to_true_on_operator_approve(self):
        agent_id = f"worker_audit_approve_{uuid.uuid4().hex[:8]}"
        cmd = "sqlmap -u http://127.0.0.1:8000 --dump"
        self.assertEqual(classify_command_privilege(cmd, "sqlmap"), "PRIVILEGED")

        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_audit_ok", f"run_{uuid.uuid4().hex[:8]}", "http://127.0.0.1:8000")
        orchestrator.active_swarms[board.challenge_id] = board

        llm_calls = 0

        async def _llm_side_effect(*args, **kwargs):
            nonlocal llm_calls
            llm_calls += 1
            # _agent_worker has no turn cap by design -- the iteration restrictions were
            # removed, so max_iterations does not bound it and a fixed command would spin
            # on the dedup path until the wait_for below times out. Stop the board after
            # the first turn, which is the loop's real termination signal.
            if llm_calls > 1:
                board.is_stopped = True
            return _make_llm_response(cmd)

        with patch("backend.agents.swarm_orchestrator.model_router.route_request", new_callable=AsyncMock) as mock_route, \
             patch("backend.agents.swarm_orchestrator.tool_manager.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_route.side_effect = _llm_side_effect
            mock_exec.return_value = _make_exec_result()

            worker = asyncio.create_task(orchestrator._agent_worker(agent_id, board, ".", "code_execution"))
            resp = await self._resolve_when_pending(orchestrator, board, "approve")
            await asyncio.wait_for(worker, timeout=25)

        self.assertIsNotNone(resp, "worker never registered a pending approval")
        self.assertTrue(resp.get("accepted"))
        mock_exec.assert_awaited()  # approval → the command actually ran

        # Query the AuditLogModel table directly: the row for this event must reflect
        # the REAL decision (approved=True), NOT the stale classification-time False.
        db = SessionLocal()
        try:
            rows = db.query(AuditLogModel).filter(
                AuditLogModel.agent == agent_id,
                AuditLogModel.action == "execute_tool:sqlmap",
            ).all()
            self.assertTrue(rows, "expected an AuditLogModel row for the approved command")
            self.assertTrue(
                all(bool(r.approved) for r in rows),
                "AuditLogModel row(s) must be reconciled to approved=True after operator approval",
            )
        finally:
            db.close()

    # ── Acceptance criterion #2: deny → row stays False, no ToolExecutionModel row ─

    async def test_audit_row_false_and_no_execution_on_operator_deny(self):
        agent_id = f"worker_audit_deny_{uuid.uuid4().hex[:8]}"
        cmd = "hydra -l admin -P /tmp/wl.txt http://127.0.0.1:8000"
        self.assertEqual(classify_command_privilege(cmd, "hydra"), "PRIVILEGED")

        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_audit_deny", f"run_{uuid.uuid4().hex[:8]}", "http://127.0.0.1:8000")
        orchestrator.active_swarms[board.challenge_id] = board

        llm_calls = 0

        async def _llm_side_effect(*args, **kwargs):
            nonlocal llm_calls
            llm_calls += 1
            # _agent_worker has no turn cap by design -- the iteration restrictions were
            # removed, so max_iterations does not bound it and a fixed command would spin
            # on the dedup path until the wait_for below times out. Stop the board after
            # the first turn, which is the loop's real termination signal.
            if llm_calls > 1:
                board.is_stopped = True
            return _make_llm_response(cmd)

        with patch("backend.agents.swarm_orchestrator.model_router.route_request", new_callable=AsyncMock) as mock_route, \
             patch("backend.agents.swarm_orchestrator.tool_manager.execute_tool", new_callable=AsyncMock) as mock_exec:
            mock_route.side_effect = _llm_side_effect
            mock_exec.return_value = _make_exec_result()

            worker = asyncio.create_task(orchestrator._agent_worker(agent_id, board, ".", "code_execution"))
            resp = await self._resolve_when_pending(orchestrator, board, "deny")
            # The denied command never executes, so the worker does not consume another
            # LLM turn and the side_effect above never fires. Stop the board directly --
            # _agent_worker has no turn cap, and otherwise it re-requests approval for the
            # same command and blocks in require_approval (which has no timeout).
            board.is_stopped = True
            await asyncio.wait_for(worker, timeout=25)

        self.assertIsNotNone(resp, "worker never registered a pending approval")
        self.assertTrue(resp.get("accepted"))
        # Denied → the command must never reach tool execution.
        mock_exec.assert_not_awaited()

        db = SessionLocal()
        try:
            rows = db.query(AuditLogModel).filter(
                AuditLogModel.agent == agent_id,
                AuditLogModel.action == "execute_tool:hydra",
            ).all()
            self.assertTrue(rows, "expected an AuditLogModel row for the denied command")
            self.assertTrue(
                all(not bool(r.approved) for r in rows),
                "AuditLogModel row(s) must remain approved=False on operator deny",
            )

            # A denied command never ran, so there must be NO ToolExecutionModel row.
            exec_rows = db.query(ToolExecutionModel).filter(
                ToolExecutionModel.agent == agent_id,
            ).all()
            self.assertEqual(
                exec_rows, [],
                "no ToolExecutionModel row may exist for a denied (never-executed) command",
            )
        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
