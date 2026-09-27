"""Operator-approval gate coverage for EVERY agent command path.

The per-command privilege gate used to exist only inside
``SwarmOrchestrator._agent_worker``. Every other agent-reachable execution path
(the legacy ReAct loop in ``agents/orchestrator_loop.py``, the
``agent_runtime.RealToolExecutor``, and ``POST /tools/execute``) could run a command
— including ``rm -rf`` — with no approval, no pending-approval entry, and no operator
prompt. These tests pin the shared gate (``backend/privilege/gate.py``) and each of
those newly-gated call sites.

The security property under test is FAIL-CLOSED: only an explicit operator "approve"
allows execution. No decision, a broken WebSocket, or any exception during the wait
must resolve to DENIED — never to a default-allow.

Mode coverage: the gate is mode-aware (``settings.FORGE_APPROVAL_MODE``).
  * "manual" (the default) asks on every PRIVILEGED/DANGEROUS command and waits with
    NO timeout at all — silence means "still waiting", never "yes".
  * "auto" runs PRIVILEGED/DANGEROUS commands unattended; only a command that
    literally needs ``sudo`` still stops, and it stops for the CREDENTIAL alone.

Isolation (project rule #5): the DATABASE_URL pin is set above this module's first backend
import. ``backend/config.py`` no longer writes to ``os.environ`` (real environment variables
win over ``.env``), and ``backend/database/guard.py`` refuses to build an engine for the
production ``forge.db`` without an authorization only the server's startup hook makes.
"""

import asyncio
import os
import tempfile
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

from backend.config import settings  # noqa: E402
from backend.database.models import (  # noqa: E402
    AgentStateModel, AuditLogModel, ChallengeModel, CheckpointModel, ReportModel,
    RunModel, ToolExecutionModel,
)
from backend.database.session import SessionLocal, get_engine, init_db  # noqa: E402
from backend.privilege.classify import classify_command_privilege  # noqa: E402
from backend.privilege.gate import SHARED_PENDING_APPROVALS, require_approval  # noqa: E402
from backend.agents.swarm_orchestrator import swarm_orchestrator  # noqa: E402

DANGEROUS_CMD = "rm -rf /tmp/x"
PRIVILEGED_CMD = "sqlmap -u http://127.0.0.1:8000 --dump"

# Window used ONLY by the tests to prove the gate is still parked and has NOT
# auto-resolved itself. The gate imposes no timeout of its own in either mode.
PENDING_PROBE_SECONDS = 1.0

def _make_llm_response(content: str):
    resp = MagicMock()
    resp.is_refusal = False
    resp.refusal_reason = ""
    resp.content = content
    resp.model = "test-model"
    return resp

def _manual_mode():
    """Pin the gate to its default mode so a stray FORGE_APPROVAL_MODE=auto in the
    environment can never turn a fail-closed assertion into a false pass."""
    return patch.object(settings, "FORGE_APPROVAL_MODE", "manual")

def _auto_mode():
    return patch.object(settings, "FORGE_APPROVAL_MODE", "auto")

def _broadcast_events(broadcast) -> list:
    """The events actually pushed to the operator console, in order.

    The gate emits two distinct kinds: APPROVAL_REQUIRED, which REGISTERS a pending
    request the operator has to answer, and the trailing APPROVAL_RESOLVED
    announcement (see ``_broadcast_resolved``) that merely logs the outcome — the
    auto-approve path fires the latter without ever firing the former. Assertions
    about "was the operator asked?" must therefore read the event stream rather than
    treating any broadcast at all as a question.
    """
    return [call.args[0]["event"] for call in broadcast.await_args_list]

async def _resolve_pending_via_api(decision: str, sudo_password=None,
                                   timeout_s: float = 20.0) -> bool:
    """Poll the shared registry until a gate registers a request, then deliver a REAL
    operator response through the production respond path (the same helper
    ``POST /approvals/{id}/respond`` calls). False if no request ever appeared."""
    for _ in range(int(timeout_s / 0.05)):
        pending = list(SHARED_PENDING_APPROVALS)
        if pending:
            resp = await swarm_orchestrator.submit_approval_response(
                pending[0], decision, sudo_password=sudo_password
            )
            return bool(resp.get("accepted"))
        await asyncio.sleep(0.05)
    return False

async def _deny_pending_via_api(timeout_s: float = 20.0) -> bool:
    return await _resolve_pending_via_api("deny", timeout_s=timeout_s)

class _GateTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        init_db()
        self.assertTrue(
            str(get_engine().url).endswith("test_forge.db"),
            "gate tests must run against the isolated test database",
        )
        SHARED_PENDING_APPROVALS.clear()

    def tearDown(self):
        SHARED_PENDING_APPROVALS.clear()

# =========================================================================== #
# 1. The shared gate itself — fail-closed semantics, MANUAL mode (the default)
# =========================================================================== #

class TestRequireApprovalFailClosed(_GateTestBase):

    async def test_dangerous_command_without_a_decision_is_never_approved(self):
        """With no operator response the gate stays PARKED — it must neither time out
        into an approval nor resolve itself into one. It only completes once an
        explicit deny arrives, and that deny is honoured."""
        self.assertEqual(classify_command_privilege(DANGEROUS_CMD, "rm"), "DANGEROUS")

        with _manual_mode():
            task = asyncio.create_task(require_approval(
                cmd=DANGEROUS_CMD,
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=AsyncMock(),
                challenge_id="ch_gate",
                run_id="run_gate",
            ))

            # Prove the gate imposed no window of its own: after a real delay (well
            # past any fast test timeout) with nobody answering, it is STILL pending.
            done, _pending = await asyncio.wait({task}, timeout=PENDING_PROBE_SECONDS)
            self.assertEqual(
                done, set(),
                "the gate resolved itself with no operator decision — it must wait "
                "indefinitely rather than timing out or defaulting to allow",
            )

            self.assertTrue(await _deny_pending_via_api())
            approved, decision, sudo_pw = await asyncio.wait_for(task, timeout=10)

        self.assertFalse(approved, "a DANGEROUS command denied by the operator must NEVER be approved")
        self.assertEqual(decision, "deny")
        self.assertIsNone(sudo_pw)

    async def test_dangerous_broadcast_failure_denies_without_raising(self):
        """A broken WebSocket must DENY — not propagate a crash that some outer
        `except: pass` could swallow into a default-allow path. This must hold even
        though the indefinite wait no longer has a timeout to fall back on."""
        async def _broken_broadcast(_payload):
            raise RuntimeError("websocket disconnected")

        with _manual_mode():
            approved, decision, sudo_pw = await asyncio.wait_for(
                require_approval(
                    cmd=DANGEROUS_CMD,
                    agent_id="test_agent",
                    pending_approvals=SHARED_PENDING_APPROVALS,
                    broadcast_fn=_broken_broadcast,
                    challenge_id="ch_gate",
                    run_id="run_gate",
                ),
                timeout=10,
            )

        self.assertFalse(approved, "a broadcast failure must resolve to DENY, never allow")
        self.assertIsNone(decision)
        self.assertIsNone(sudo_pw)

    async def test_privileged_command_waits_for_an_explicit_decision(self):
        cmd = "some-unregistered-binary --flag"
        self.assertEqual(classify_command_privilege(cmd, "some-unregistered-binary"), "PRIVILEGED")

        with _manual_mode():
            task = asyncio.create_task(require_approval(
                cmd=cmd,
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=AsyncMock(),
                challenge_id="ch_gate",
            ))
            done, _ = await asyncio.wait({task}, timeout=PENDING_PROBE_SECONDS)
            self.assertEqual(done, set(), "a PRIVILEGED command must wait for the operator")

            self.assertTrue(await _resolve_pending_via_api("approve"))
            approved, decision, _ = await asyncio.wait_for(task, timeout=10)

        self.assertTrue(approved)
        self.assertEqual(decision, "approve")

    async def test_safe_command_auto_approves_without_an_operator(self):
        """SAFE tools (registry-marked) need no round-trip and register no request."""
        broadcast = AsyncMock()
        approved, decision, sudo_pw = await require_approval(
            cmd="nmap -sV 127.0.0.1",
            agent_id="test_agent",
            pending_approvals=SHARED_PENDING_APPROVALS,
            broadcast_fn=broadcast,
            challenge_id="ch_gate",
        )

        self.assertTrue(approved)
        self.assertIsNone(decision)
        self.assertIsNone(sudo_pw)
        broadcast.assert_not_awaited()
        self.assertEqual(SHARED_PENDING_APPROVALS, {})

    async def test_pending_entry_registered_then_cleaned_up(self):
        """The request must be visible to the operator UI while pending and removed
        (with its single-use password wiped) once resolved."""
        seen = {}

        async def _capture(payload):
            if payload["event"] != "APPROVAL_REQUIRED":
                return  # the trailing APPROVAL_RESOLVED announcement is not the request
            seen["request_id"] = payload["request_id"]
            seen["event"] = payload["event"]
            # Respond as the operator would, through the same dict the gate polls.
            SHARED_PENDING_APPROVALS[payload["request_id"]]["decision"] = "approve"
            SHARED_PENDING_APPROVALS[payload["request_id"]]["event"].set()

        with _manual_mode():
            approved, decision, _ = await require_approval(
                cmd=PRIVILEGED_CMD,
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=_capture,
                challenge_id="ch_gate",
                run_id="run_gate",
            )

        self.assertEqual(seen["event"], "APPROVAL_REQUIRED")
        self.assertTrue(approved, "an explicit operator approve must allow execution")
        self.assertEqual(decision, "approve")
        self.assertEqual(SHARED_PENDING_APPROVALS, {}, "the pending entry must be cleaned up")

    async def test_sudo_password_returned_on_approve_and_wiped(self):
        async def _approve_with_password(payload):
            entry = SHARED_PENDING_APPROVALS[payload["request_id"]]
            self.assertTrue(entry["requires_sudo"])
            entry["decision"] = "approve"
            entry["sudo_password"] = "s3cr3t"
            entry["event"].set()

        with _manual_mode():
            approved, _, sudo_pw = await require_approval(
                cmd="sudo cat /etc/shadow",
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=_approve_with_password,
                challenge_id="ch_gate",
            )
        self.assertTrue(approved)
        self.assertEqual(sudo_pw, "s3cr3t")
        self.assertEqual(SHARED_PENDING_APPROVALS, {})

    async def test_deny_discards_sudo_password(self):
        async def _deny_with_password(payload):
            entry = SHARED_PENDING_APPROVALS[payload["request_id"]]
            entry["decision"] = "deny"
            entry["sudo_password"] = "s3cr3t"
            entry["event"].set()

        with _manual_mode():
            approved, decision, sudo_pw = await require_approval(
                cmd="sudo cat /etc/shadow",
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=_deny_with_password,
                challenge_id="ch_gate",
            )
        self.assertFalse(approved)
        self.assertEqual(decision, "deny")
        self.assertIsNone(sudo_pw, "a denied command's password must never be handed back")

    async def test_audit_row_reconciled_on_approve(self):
        """The classification-time audit row must end up reflecting the REAL decision."""
        agent_id = f"gate_audit_{uuid.uuid4().hex[:8]}"

        async def _approve(payload):
            entry = SHARED_PENDING_APPROVALS[payload["request_id"]]
            entry["decision"] = "approve"
            entry["event"].set()

        with _manual_mode():
            approved, _, _ = await require_approval(
                cmd=PRIVILEGED_CMD,
                agent_id=agent_id,
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=_approve,
                challenge_id="ch_gate",
            )
        self.assertTrue(approved)

        db = SessionLocal()
        try:
            rows = db.query(AuditLogModel).filter(AuditLogModel.agent == agent_id).all()
            self.assertTrue(rows, "the gate must write an audit row")
            self.assertTrue(all(bool(r.approved) for r in rows),
                            "audit row must be reconciled to the operator's real approval")
        finally:
            db.close()

    async def test_audit_row_stays_false_on_deny(self):
        agent_id = f"gate_audit_to_{uuid.uuid4().hex[:8]}"

        async def _deny(payload):
            entry = SHARED_PENDING_APPROVALS[payload["request_id"]]
            entry["decision"] = "deny"
            entry["event"].set()

        with _manual_mode():
            approved, _, _ = await require_approval(
                cmd=DANGEROUS_CMD,
                agent_id=agent_id,
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=_deny,
                challenge_id="ch_gate",
            )
        self.assertFalse(approved)

        db = SessionLocal()
        try:
            rows = db.query(AuditLogModel).filter(AuditLogModel.agent == agent_id).all()
            self.assertTrue(rows)
            self.assertTrue(all(not bool(r.approved) for r in rows),
                            "a denied command must stay approved=False in the audit trail")
        finally:
            db.close()

# =========================================================================== #
# 2. AUTO mode — run unattended; the ONLY stop is a missing sudo credential
# =========================================================================== #

class TestRequireApprovalAutoMode(_GateTestBase):

    async def test_auto_mode_privileged_without_sudo_runs_unattended(self):
        """Auto mode: a PRIVILEGED command with no sudo is approved immediately — no
        pending entry, no APPROVAL_REQUIRED, no wait at all.

        The gate still emits one APPROVAL_RESOLVED announcement so the operator console
        can account afterwards for a privileged command that ran with nobody watching —
        that is a logged entry, never a question. The invariant pinned here is that the
        operator is never ASKED: nothing is registered and no APPROVAL_REQUIRED goes out.
        """
        broadcast = AsyncMock()
        with _auto_mode():
            approved, decision, sudo_pw = await require_approval(
                cmd=PRIVILEGED_CMD,
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=broadcast,
                challenge_id="ch_gate",
                run_id="run_gate",
            )

        self.assertTrue(approved)
        self.assertEqual(decision, "auto-approved")
        self.assertIsNone(sudo_pw)
        self.assertEqual(SHARED_PENDING_APPROVALS, {}, "auto-approve must register nothing")
        self.assertEqual(
            _broadcast_events(broadcast), ["APPROVAL_RESOLVED"],
            "auto mode must announce its decision, never ASK for one (no APPROVAL_REQUIRED)",
        )
        self.assertTrue(broadcast.await_args_list[0].args[0]["approved"])

    async def test_auto_mode_dangerous_without_sudo_also_runs_unattended(self):
        """DELIBERATE, operator-requested tradeoff: in auto mode a DANGEROUS command
        that does not literally contain "sudo" runs with no operator interaction. This
        test exists so the behaviour is pinned and obvious, not accidental."""
        broadcast = AsyncMock()
        with _auto_mode():
            approved, decision, _ = await require_approval(
                cmd=DANGEROUS_CMD,  # "rm -rf /tmp/x" — classified DANGEROUS
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=broadcast,
                challenge_id="ch_gate",
            )

        self.assertTrue(approved)
        self.assertEqual(decision, "auto-approved")
        self.assertEqual(
            _broadcast_events(broadcast), ["APPROVAL_RESOLVED"],
            "even a DANGEROUS command auto-approved unattended must never be put to the operator",
        )

    async def test_auto_mode_sudo_command_still_waits_for_the_credential(self):
        """A sudo command in auto mode still halts: the operator is asked for the
        PASSWORD only, never for an approve/deny decision."""
        seen = {}

        async def _capture(payload):
            if payload["event"] != "APPROVAL_REQUIRED":
                return  # the trailing APPROVAL_RESOLVED announcement is not the request
            seen.update(payload)
            entry = SHARED_PENDING_APPROVALS[payload["request_id"]]
            seen["auto_mode_sudo_only"] = entry.get("auto_mode_sudo_only")
            entry["decision"] = "approve"
            entry["sudo_password"] = "s3cr3t"
            entry["event"].set()

        with _auto_mode():
            approved, decision, sudo_pw = await require_approval(
                cmd="sudo nmap -sS 10.0.0.1",
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=_capture,
                challenge_id="ch_gate",
                run_id="run_gate",
            )

        self.assertEqual(seen["event"], "APPROVAL_REQUIRED")
        self.assertFalse(
            seen["decision_required"],
            "auto+sudo must tell the UI that no yes/no decision is being requested",
        )
        self.assertTrue(seen["requires_sudo"])
        self.assertTrue(seen["auto_mode_sudo_only"])
        self.assertTrue(approved)
        self.assertEqual(decision, "approve")
        self.assertEqual(sudo_pw, "s3cr3t")
        self.assertEqual(SHARED_PENDING_APPROVALS, {})

    async def test_auto_mode_sudo_command_without_a_password_is_denied(self):
        """An empty/missing password must NEVER be read as approval to run sudo."""
        async def _approve_without_password(payload):
            entry = SHARED_PENDING_APPROVALS[payload["request_id"]]
            entry["decision"] = "approve"
            entry["sudo_password"] = ""
            entry["event"].set()

        with _auto_mode():
            approved, decision, sudo_pw = await require_approval(
                cmd="sudo cat /etc/shadow",
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=_approve_without_password,
                challenge_id="ch_gate",
            )

        self.assertFalse(approved, "a blank password submission is not an approval")
        self.assertEqual(decision, "deny")
        self.assertIsNone(sudo_pw)

    async def test_auto_mode_sudo_cancel_is_a_deny(self):
        """The UI's Cancel path submits decision="deny" with no password — the gate
        must treat that as a hard deny, not as a missing password to retry."""
        seen = {}

        async def _capture(payload):
            seen["request_id"] = payload["request_id"]
            entry = SHARED_PENDING_APPROVALS[payload["request_id"]]
            entry["decision"] = "deny"
            entry["event"].set()

        with _auto_mode():
            approved, decision, sudo_pw = await require_approval(
                cmd="sudo cat /etc/shadow",
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=_capture,
                challenge_id="ch_gate",
            )

        self.assertFalse(approved)
        self.assertEqual(decision, "deny")
        self.assertIsNone(sudo_pw)
        self.assertEqual(SHARED_PENDING_APPROVALS, {})

    async def test_auto_mode_waits_indefinitely_for_the_sudo_password(self):
        """Same no-timeout guarantee on the credential wait as in manual mode."""
        with _auto_mode():
            task = asyncio.create_task(require_approval(
                cmd="sudo id",
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=AsyncMock(),
                challenge_id="ch_gate",
            ))
            done, _ = await asyncio.wait({task}, timeout=PENDING_PROBE_SECONDS)
            self.assertEqual(done, set(), "the credential wait must not time out")

            self.assertTrue(await _resolve_pending_via_api("approve", sudo_password="pw"))
            approved, decision, sudo_pw = await asyncio.wait_for(task, timeout=10)

        self.assertTrue(approved)
        self.assertEqual(decision, "approve")
        self.assertEqual(sudo_pw, "pw")

    async def test_auto_mode_reconciles_the_audit_row_to_approved(self):
        """An auto-approved command really ran, so the audit trail must say so."""
        agent_id = f"gate_auto_audit_{uuid.uuid4().hex[:8]}"
        with _auto_mode():
            approved, _, _ = await require_approval(
                cmd=PRIVILEGED_CMD,
                agent_id=agent_id,
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=AsyncMock(),
                challenge_id="ch_gate",
            )
        self.assertTrue(approved)

        db = SessionLocal()
        try:
            rows = db.query(AuditLogModel).filter(AuditLogModel.agent == agent_id).all()
            self.assertTrue(rows, "the gate must write an audit row")
            self.assertTrue(all(bool(r.approved) for r in rows),
                            "an auto-approved command must be reconciled to approved=True")
        finally:
            db.close()

    async def test_invalid_mode_falls_back_to_manual(self):
        """A misconfigured mode must behave exactly like manual — ask, and wait."""
        with patch.object(settings, "FORGE_APPROVAL_MODE", "AUTO-LOL"):
            task = asyncio.create_task(require_approval(
                cmd=PRIVILEGED_CMD,
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=AsyncMock(),
                challenge_id="ch_gate",
            ))
            done, _ = await asyncio.wait({task}, timeout=PENDING_PROBE_SECONDS)
            self.assertEqual(done, set(), "an unknown mode must fail closed into manual")

            self.assertTrue(await _deny_pending_via_api())
            approved, _, _ = await asyncio.wait_for(task, timeout=10)
        self.assertFalse(approved)

# =========================================================================== #
# 3. Legacy ReAct loop (agents/orchestrator_loop.py) — all four call sites
# =========================================================================== #

class TestLegacyLoopIsGated(_GateTestBase):

    async def test_denied_dangerous_command_never_executes(self):
        """The legacy loop's main execution call must not run a DENIED DANGEROUS
        command. Drives the REAL loop with the REAL classifier and gate; the operator
        answers with an explicit deny.
        """
        from backend.agents.orchestrator_loop import orchestrator_loop
        from backend.agents import orchestrator_loop as loop_module

        db = SessionLocal()
        try:
            ch = ChallengeModel(
                name="Privilege Gate Legacy Loop",
                category="web",
                difficulty="EASY",
                description="Gate coverage fixture (test DB only).",
                working_directory=".",
                platform_name="TEST",
            )
            db.add(ch)
            db.commit()
            run = RunModel(challenge_id=ch.id, status="RUNNING",
                           current_phase="recon", current_agent="orchestrator")
            db.add(run)
            db.commit()
            challenge_id, run_id = ch.id, run.id
        finally:
            db.close()

        # Turn 1 runs the DANGEROUS command and is denied; turn 2 sees the kill switch
        # so the loop terminates instead of pivoting to a different (approved) command.
        cancelled = MagicMock(side_effect=[False] + [True] * 20)

        try:
            with patch.object(loop_module, "model_router") as mock_router, \
                 patch.object(loop_module, "tool_manager") as mock_tm, \
                 patch.object(loop_module.workflow_runner, "is_cancelled", cancelled), \
                 patch("backend.agents.orchestrator_loop.ws_manager.broadcast", AsyncMock()), \
                 _manual_mode():
                mock_router.route_request = AsyncMock(
                    return_value=_make_llm_response(DANGEROUS_CMD)
                )
                mock_tm.execute_raw_command = AsyncMock()

                loop_task = asyncio.create_task(orchestrator_loop.run_autonomous_loop(
                    run_id, challenge_id, "http://127.0.0.1:8000"
                ))
                denied = await _deny_pending_via_api()
                await asyncio.wait_for(loop_task, timeout=60)

                self.assertTrue(denied, "the gate never registered a pending approval")
                mock_router.route_request.assert_awaited()
                mock_tm.execute_raw_command.assert_not_awaited()
        finally:
            db = SessionLocal()
            try:
                # Child rows hold FKs onto runs.id / challenges.id (foreign_keys=ON), so
                # clear them before the parents. The challenge itself is removed through
                # the ORM so its cascade (targets/runs/evidence/findings) also fires.
                for model in (AgentStateModel, CheckpointModel, ToolExecutionModel):
                    db.query(model).filter(model.run_id == run_id).delete()
                db.query(ReportModel).filter(ReportModel.challenge_id == challenge_id).delete()
                db.flush()
                ch_obj = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
                if ch_obj is not None:
                    db.delete(ch_obj)
                db.commit()
            finally:
                db.close()

    async def test_automation_script_is_not_gated_in_manual_mode(self):
        """`python3 solve.py` is automation, not a privileged action: the loop must run
        it in manual mode WITHOUT any operator round-trip."""
        from backend.agents.orchestrator_loop import orchestrator_loop
        from backend.agents import orchestrator_loop as loop_module

        self.assertEqual(classify_command_privilege("python3 solve.py", "python3"), "SAFE")

        db = SessionLocal()
        try:
            ch = ChallengeModel(
                name="Privilege Gate Automation Script",
                category="web",
                difficulty="EASY",
                description="Gate coverage fixture (test DB only).",
                working_directory=".",
                platform_name="TEST",
            )
            db.add(ch)
            db.commit()
            run = RunModel(challenge_id=ch.id, status="RUNNING",
                           current_phase="recon", current_agent="orchestrator")
            db.add(run)
            db.commit()
            challenge_id, run_id = ch.id, run.id
        finally:
            db.close()

        cancelled = MagicMock(side_effect=[False] + [True] * 20)
        try:
            with patch.object(loop_module, "model_router") as mock_router, \
                 patch.object(loop_module, "tool_manager") as mock_tm, \
                 patch.object(loop_module.workflow_runner, "is_cancelled", cancelled), \
                 patch("backend.agents.orchestrator_loop.ws_manager.broadcast", AsyncMock()) as mock_bcast, \
                 _manual_mode():
                mock_router.route_request = AsyncMock(
                    return_value=_make_llm_response("STRATEGY: run the solver\n```bash\npython3 solve.py\n```")
                )
                mock_tm.execute_raw_command = AsyncMock(
                    return_value=_tool_result(stdout="ok")
                )

                await asyncio.wait_for(
                    orchestrator_loop.run_autonomous_loop(run_id, challenge_id, "http://127.0.0.1:8000"),
                    timeout=60,
                )

                mock_tm.execute_raw_command.assert_awaited()
                # No operator round-trip was needed for a plain interpreter invocation.
                self.assertEqual(SHARED_PENDING_APPROVALS, {})
        finally:
            db = SessionLocal()
            try:
                for model in (AgentStateModel, CheckpointModel, ToolExecutionModel):
                    db.query(model).filter(model.run_id == run_id).delete()
                db.query(ReportModel).filter(ReportModel.challenge_id == challenge_id).delete()
                db.flush()
                ch_obj = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
                if ch_obj is not None:
                    db.delete(ch_obj)
                db.commit()
            finally:
                db.close()

def _tool_result(stdout="", stderr="", exit_code=0):
    res = MagicMock()
    res.stdout = stdout
    res.stderr = stderr
    res.exit_code = exit_code
    res.status = "SUCCESS" if exit_code == 0 else "FAILED"
    res.execution_failure = False
    res.failure_category = None
    res.tool_name = "raw_cmd"
    res.duration_ms = 1.0
    return res

# =========================================================================== #
# 4. agent_runtime.RealToolExecutor — COMMAND / PYTHON_SCRIPT / shell TOOL_CALL
# =========================================================================== #

class TestRealToolExecutorIsGated(_GateTestBase):

    def _denying_executor(self):
        """A RealToolExecutor using its PRODUCTION default gate (no stub injected)."""
        from backend.agent_runtime.runtime import RealToolExecutor
        executor = RealToolExecutor(tool_manager=MagicMock())
        return executor

    async def test_denied_command_never_executes(self):
        from backend.agent_runtime.action import Action, ActionType

        executor = self._denying_executor()
        action = Action(type=ActionType.COMMAND, command=DANGEROUS_CMD)

        with patch("backend.agent_runtime.runtime.ws_manager.broadcast", AsyncMock()), \
             _manual_mode():
            task = asyncio.create_task(executor.execute(action))
            denied = await _deny_pending_via_api()
            res = await asyncio.wait_for(task, timeout=30)

        self.assertTrue(denied, "the executor gate never registered a pending approval")
        executor.tool_manager.execute_raw_command.assert_not_called()
        self.assertEqual(res.status, "FAILED")
        self.assertTrue(res.execution_failure)
        self.assertIn(res.failure_category, ("PRIVILEGE_DENIED", "CAPABILITY_GAP"))
        self.assertIn("PRIVILEGE DENIED", res.stderr)

    async def test_denied_python_script_never_executes(self):
        """The gate classifies the command that ACTUALLY runs — the quoted interpreter
        plus the script path — so a PYTHON_SCRIPT action still needs approval: the
        quoted path is not a bare interpreter name, so the automation allowlist does
        not apply and the fail-closed PRIVILEGED default does."""
        from backend.agent_runtime.action import Action, ActionType

        executor = self._denying_executor()
        action = Action(type=ActionType.PYTHON_SCRIPT,
                        script="print('this must never run')\n")

        # Throwaway workspace: RealToolExecutor materialises the script on disk before
        # the gate runs, so a cwd of "." would overwrite the repo-root solve.py.
        with tempfile.TemporaryDirectory() as tmp:
            with patch("backend.agent_runtime.runtime.ws_manager.broadcast", AsyncMock()), \
                 _manual_mode():
                task = asyncio.create_task(executor.execute(action, cwd=tmp))
                denied = await _deny_pending_via_api()
                res = await asyncio.wait_for(task, timeout=30)

        self.assertTrue(denied, "the script gate never registered a pending approval")
        executor.tool_manager.execute_raw_command.assert_not_called()
        self.assertEqual(res.status, "FAILED")
        self.assertIn(res.failure_category, ("PRIVILEGE_DENIED", "CAPABILITY_GAP"))

    async def test_automation_python_command_is_not_gated(self):
        """Regression guard for the new classification: `python3 solve.py` runs with no
        operator round-trip even in manual mode."""
        from backend.agent_runtime.action import Action, ActionType

        from backend.agent_runtime.runtime import RealToolExecutor
        tm = MagicMock()
        tm.execute_raw_command = AsyncMock(return_value=_tool_result(stdout="solved"))
        executor = RealToolExecutor(tool_manager=tm)

        with patch("backend.agent_runtime.runtime.ws_manager.broadcast", AsyncMock()) as mock_bcast, \
             _manual_mode():
            res = await asyncio.wait_for(
                executor.execute(Action(type=ActionType.COMMAND, command="python3 solve.py")),
                timeout=30,
            )

        tm.execute_raw_command.assert_awaited()
        mock_bcast.assert_not_awaited()
        self.assertEqual(res.status, "SUCCESS")

    async def test_denied_shell_capability_never_executes(self):
        """`interactive_open` runs its target as a command line, so a DANGEROUS target
        must be gated even though the capability name itself is registry-marked SAFE."""
        from backend.agent_runtime.action import Action, ActionType

        executor = self._denying_executor()
        action = Action(type=ActionType.TOOL_CALL, capability="interactive_open",
                        tool_args={"target": DANGEROUS_CMD})

        with patch("backend.agent_runtime.runtime.ws_manager.broadcast", AsyncMock()), \
             _manual_mode():
            task = asyncio.create_task(executor.execute(action))
            denied = await _deny_pending_via_api()
            res = await asyncio.wait_for(task, timeout=30)

        self.assertTrue(denied, "the capability gate never registered a pending approval")
        executor.tool_manager.execute_capability.assert_not_called()
        self.assertEqual(res.status, "FAILED")
        self.assertIn(res.failure_category, ("PRIVILEGE_DENIED", "CAPABILITY_GAP"))

    async def test_approved_command_still_executes(self):
        """Regression guard: the gate must not block an approved command."""
        from backend.agent_runtime.action import Action, ActionType

        async def _approve(cmd, **kwargs):
            return True, "approve", None

        from backend.agent_runtime.runtime import RealToolExecutor
        tm = MagicMock()
        tm.execute_raw_command = AsyncMock(return_value=_tool_result(stdout="ok"))
        executor = RealToolExecutor(tool_manager=tm, approval_gate=_approve)

        res = await executor.execute(Action(type=ActionType.COMMAND, command="python solve.py"))
        tm.execute_raw_command.assert_awaited()
        self.assertEqual(res.status, "SUCCESS")

# =========================================================================== #
# 5. POST /tools/execute — an HTTP execution entry point reachable by an agent
# =========================================================================== #

class TestToolsExecuteRouteIsGated(_GateTestBase):

    async def test_denied_shell_capability_is_403_and_never_executes(self):
        """`execute_capability('interactive_open', target=<cmd>)` reaches the shell, so
        the route must gate it. A denied request must not touch the tool manager."""
        from fastapi import HTTPException
        from backend.api.routes.execution import ExecuteToolRequest, execute_tool

        req = ExecuteToolRequest(capability="interactive_open", target=DANGEROUS_CMD)

        with patch("backend.api.routes.execution.tool_manager") as mock_tm, \
             patch("backend.api.routes.execution.ws_manager.broadcast", AsyncMock()), \
             _manual_mode():
            mock_tm.execute_capability = AsyncMock()
            task = asyncio.create_task(execute_tool(req))
            denied = await _deny_pending_via_api()
            with self.assertRaises(HTTPException) as ctx:
                await asyncio.wait_for(task, timeout=30)
            mock_tm.execute_capability.assert_not_called()

        self.assertTrue(denied, "the route gate never registered a pending approval")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("PRIVILEGE", ctx.exception.detail)

    async def test_approved_capability_still_runs(self):
        """Regression guard: a SAFE/approved capability is not blocked by the route gate."""
        from backend.api.routes.execution import ExecuteToolRequest, execute_tool

        req = ExecuteToolRequest(capability="nmap", target="127.0.0.1")

        with patch("backend.api.routes.execution.tool_manager") as mock_tm:
            mock_tm.execute_capability = AsyncMock(return_value={"status": "SUCCESS"})
            result = await execute_tool(req)

        mock_tm.execute_capability.assert_awaited_once()
        self.assertEqual(result, {"status": "SUCCESS"})

if __name__ == "__main__":
    unittest.main()
