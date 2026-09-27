"""Operator-approval SURFACE coverage: what the decision UI is told, and when.

``test_privilege_gate.py`` pins the gate's fail-closed semantics — whether a command may
run. This module pins the other half: what the operator console is TOLD about a request,
because a correct decision the operator cannot see is not an approval workflow.

Three properties, all of which the /tools "Operator Approvals" panel depends on:

  1. CONTEXT — a tool-install or capability-gap retry carries enough detail (which
     capability, which install command, which prior denial) to be decidable. Without it
     both render as a bare command line, indistinguishable from any other privileged
     command.
  2. RESOLUTION — every decided request is announced via APPROVAL_RESOLVED so the UI can
     log the outcome. Critically this includes the AUTO-mode approval, which asks nobody
     and must appear as an already-resolved entry rather than a pending card.
  3. NO SECRETS — the read-only pending listing must never expose the transient sudo
     password or the non-serializable asyncio.Event held on a registry entry.

Isolation (project rule #5): the DATABASE_URL pin is set above this module's first backend
import. ``backend/config.py`` no longer writes to ``os.environ`` (real environment variables
win over ``.env``), and ``backend/database/guard.py`` refuses to build an engine for the
production ``forge.db`` without an authorization only the server's startup hook makes.
"""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.config import settings  # noqa: E402
from backend.database.session import SessionLocal, get_engine, init_db  # noqa: E402
from backend.privilege.gate import (  # noqa: E402
    SHARED_PENDING_APPROVALS,
    require_approval,
)
from backend.agents.swarm_orchestrator import (  # noqa: E402
    _capability_gap_retry_context,
    swarm_orchestrator,
)
from backend.agents.swarm_state import SwarmBlackboard  # noqa: E402
from backend.api.routes.privilege_and_approvals import list_pending_approvals  # noqa: E402

# PRIVILEGED (not DANGEROUS, not SAFE): the classification that makes the gate ask.
PRIVILEGED_CMD = "sqlmap -u http://127.0.0.1:8000 --dump"

TOOL_INSTALL_CONTEXT = {
    "request_kind": "tool_install",
    "capability": "web_fuzzing",
    "provider": "ffuf",
    "method": "system_level",
    "install_command": "apt-get install -y ffuf",
    "reason": "ffuf is not installed",
}

def _manual_mode():
    """Pin the gate to its default mode so a stray FORGE_APPROVAL_MODE=auto in the
    environment cannot silently turn this module's manual-mode assertions into no-ops."""
    return patch.object(settings, "FORGE_APPROVAL_MODE", "manual")

def _auto_mode():
    return patch.object(settings, "FORGE_APPROVAL_MODE", "auto")

def _recording_broadcast():
    """Collect broadcast payloads; returns (async_fn, payloads)."""
    payloads = []

    async def _broadcast(payload):
        payloads.append(payload)

    return _broadcast, payloads

def _events(payloads, event_name):
    return [p for p in payloads if p.get("event") == event_name]

async def _answer_first_pending(decision: str, timeout_s: float = 20.0) -> bool:
    """Poll the shared registry until the gate registers a request, then answer it
    through the production respond path (the same helper the real endpoint calls)."""
    for _ in range(int(timeout_s / 0.05)):
        pending = list(SHARED_PENDING_APPROVALS)
        if pending:
            resp = await swarm_orchestrator.submit_approval_response(pending[0], decision)
            return bool(resp.get("accepted"))
        await asyncio.sleep(0.05)
    return False

class _SurfaceTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        init_db()
        self.assertTrue(
            str(get_engine().url).endswith("test_forge.db"),
            "approval-surface tests must run against the isolated test database",
        )
        SHARED_PENDING_APPROVALS.clear()
        swarm_orchestrator.active_swarms.clear()

    def tearDown(self):
        SHARED_PENDING_APPROVALS.clear()
        swarm_orchestrator.active_swarms.clear()

# =========================================================================== #
# 1. Context — the operator is told WHY a request exists
# =========================================================================== #

class TestApprovalContext(_SurfaceTestBase):

    async def test_context_reaches_both_the_broadcast_and_the_pending_entry(self):
        """A tool install must be decidable from the card alone: the UI reads `context`
        off the broadcast, and the pending listing reads it off the registry entry."""
        broadcast, payloads = _recording_broadcast()

        with _manual_mode():
            task = asyncio.create_task(require_approval(
                cmd="apt-get install -y ffuf",
                agent_id="capability_manager",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=broadcast,
                challenge_id="ch_ctx",
                run_id="run_ctx",
                context=TOOL_INSTALL_CONTEXT,
            ))
            self.assertTrue(await _answer_first_pending("deny"))
            await asyncio.wait_for(task, timeout=10)

        required = _events(payloads, "APPROVAL_REQUIRED")
        self.assertEqual(len(required), 1)
        self.assertEqual(required[0]["context"], TOOL_INSTALL_CONTEXT)

    async def test_registry_entry_carries_context_and_ownership(self):
        """The pending listing rebuilds the card after a page reload, so the entry itself
        must hold the context plus the challenge/run it belongs to."""
        broadcast, _payloads = _recording_broadcast()
        captured = {}

        with _manual_mode():
            task = asyncio.create_task(require_approval(
                cmd="apt-get install -y ffuf",
                agent_id="capability_manager",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=broadcast,
                challenge_id="ch_ctx",
                run_id="run_ctx",
                context=TOOL_INSTALL_CONTEXT,
            ))
            for _ in range(400):
                if SHARED_PENDING_APPROVALS:
                    captured = dict(next(iter(SHARED_PENDING_APPROVALS.values())))
                    break
                await asyncio.sleep(0.05)
            self.assertTrue(await _answer_first_pending("deny"))
            await asyncio.wait_for(task, timeout=10)

        self.assertEqual(captured.get("context"), TOOL_INSTALL_CONTEXT)
        self.assertEqual(captured.get("challenge_id"), "ch_ctx")
        self.assertEqual(captured.get("run_id"), "run_ctx")

    async def test_context_is_copied_so_a_later_mutation_cannot_rewrite_history(self):
        """The caller's dict must not be able to change what the operator was shown."""
        broadcast, _payloads = _recording_broadcast()
        caller_context = dict(TOOL_INSTALL_CONTEXT)
        captured = {}

        with _manual_mode():
            task = asyncio.create_task(require_approval(
                cmd="apt-get install -y ffuf",
                agent_id="capability_manager",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=broadcast,
                challenge_id="ch_ctx",
                context=caller_context,
            ))
            for _ in range(400):
                if SHARED_PENDING_APPROVALS:
                    captured = next(iter(SHARED_PENDING_APPROVALS.values()))
                    break
                await asyncio.sleep(0.05)
            caller_context["capability"] = "TAMPERED"
            self.assertTrue(await _answer_first_pending("deny"))
            await asyncio.wait_for(task, timeout=10)

        self.assertEqual(captured["context"]["capability"], "web_fuzzing")

    async def test_command_without_context_still_works_and_reports_an_empty_context(self):
        """Every pre-existing caller passes no context. They must be unaffected: the same
        payload shape, with an empty context rather than a missing key."""
        broadcast, payloads = _recording_broadcast()

        with _manual_mode():
            task = asyncio.create_task(require_approval(
                cmd=PRIVILEGED_CMD,
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=broadcast,
                challenge_id="ch_ctx",
            ))
            self.assertTrue(await _answer_first_pending("deny"))
            await asyncio.wait_for(task, timeout=10)

        self.assertEqual(_events(payloads, "APPROVAL_REQUIRED")[0]["context"], {})

# =========================================================================== #
# 2. Resolution — every decision is announced, including the unattended one
# =========================================================================== #

class TestResolutionEvents(_SurfaceTestBase):

    async def test_operator_deny_is_announced_as_resolved(self):
        broadcast, payloads = _recording_broadcast()

        with _manual_mode():
            task = asyncio.create_task(require_approval(
                cmd=PRIVILEGED_CMD,
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=broadcast,
                challenge_id="ch_res",
                context=TOOL_INSTALL_CONTEXT,
            ))
            self.assertTrue(await _answer_first_pending("deny"))
            approved, _decision, _pw = await asyncio.wait_for(task, timeout=10)

        self.assertFalse(approved)
        resolved = _events(payloads, "APPROVAL_RESOLVED")
        self.assertEqual(len(resolved), 1)
        self.assertFalse(resolved[0]["approved"])
        self.assertEqual(resolved[0]["decision"], "deny")
        # The resolved event must repeat the context, or the logged row loses its meaning.
        self.assertEqual(resolved[0]["context"], TOOL_INSTALL_CONTEXT)

    async def test_auto_mode_announces_a_resolved_entry_without_ever_being_pending(self):
        """Auto mode runs the command unattended. The operator must still be able to
        account for it afterwards — so it is announced as ALREADY RESOLVED and is never
        registered as pending (a pending entry would render as a clickable card for a
        decision that no longer exists)."""
        broadcast, payloads = _recording_broadcast()

        with _auto_mode():
            approved, decision, sudo_pw = await asyncio.wait_for(
                require_approval(
                    cmd=PRIVILEGED_CMD,  # PRIVILEGED, and contains no 'sudo'
                    agent_id="test_agent",
                    pending_approvals=SHARED_PENDING_APPROVALS,
                    broadcast_fn=broadcast,
                    challenge_id="ch_res",
                    context=TOOL_INSTALL_CONTEXT,
                ),
                timeout=10,
            )

        self.assertTrue(approved)
        self.assertEqual(decision, "auto-approved")
        self.assertIsNone(sudo_pw)

        # Never pending: nothing for the operator to click, and nothing left registered.
        self.assertEqual(_events(payloads, "APPROVAL_REQUIRED"), [])
        self.assertEqual(SHARED_PENDING_APPROVALS, {})

        resolved = _events(payloads, "APPROVAL_RESOLVED")
        self.assertEqual(len(resolved), 1)
        self.assertTrue(resolved[0]["approved"])
        self.assertEqual(resolved[0]["decision"], "auto-approved")
        self.assertEqual(resolved[0]["context"], TOOL_INSTALL_CONTEXT)
        self.assertEqual(resolved[0]["challenge_id"], "ch_res")

    async def test_broken_broadcast_still_denies_and_announces_no_decision(self):
        """A dead socket must deny (fail-closed) and report the absence of a decision
        honestly as `decision: None` rather than inventing a deny the operator never gave."""
        calls = []

        async def _flaky_broadcast(payload):
            calls.append(payload)
            # Fail only the request; the resolved announcement on the way out is allowed
            # to attempt-succeed so we can assert on it.
            if payload.get("event") == "APPROVAL_REQUIRED":
                raise RuntimeError("websocket disconnected")

        with _manual_mode():
            approved, decision, sudo_pw = await asyncio.wait_for(
                require_approval(
                    cmd=PRIVILEGED_CMD,
                    agent_id="test_agent",
                    pending_approvals=SHARED_PENDING_APPROVALS,
                    broadcast_fn=_flaky_broadcast,
                    challenge_id="ch_res",
                ),
                timeout=10,
            )

        self.assertFalse(approved, "a failed broadcast must never approve")
        self.assertIsNone(decision)
        self.assertIsNone(sudo_pw)
        resolved = _events(calls, "APPROVAL_RESOLVED")
        self.assertEqual(len(resolved), 1)
        self.assertFalse(resolved[0]["approved"])
        self.assertIsNone(resolved[0]["decision"])

    async def test_a_broken_socket_on_the_resolved_announcement_cannot_change_the_decision(self):
        """The announcement is best-effort: it happens after the decision is settled, so
        its failure must not turn an approval into a denial (or vice versa)."""
        async def _always_broken(_payload):
            raise RuntimeError("websocket disconnected")

        with _auto_mode():
            approved, decision, _pw = await asyncio.wait_for(
                require_approval(
                    cmd=PRIVILEGED_CMD,
                    agent_id="test_agent",
                    pending_approvals=SHARED_PENDING_APPROVALS,
                    broadcast_fn=_always_broken,
                    challenge_id="ch_res",
                ),
                timeout=10,
            )

        self.assertTrue(approved, "a failed resolved-announcement must not undo the auto-approval")
        self.assertEqual(decision, "auto-approved")

# =========================================================================== #
# 3. The pending listing must not leak secrets or crash on non-JSON values
# =========================================================================== #

class TestPendingListing(_SurfaceTestBase):

    async def test_listing_omits_sudo_password_and_the_asyncio_event(self):
        """A pending entry holds a live asyncio.Event and, for sudo commands, a transient
        single-use password. The listing is reached from the operator console, so it must
        expose neither — by allowlist, not by remembering to delete them."""
        event = asyncio.Event()
        SHARED_PENDING_APPROVALS["req_secret"] = {
            "event": event,
            "decision": None,
            "command": "sudo -S id",
            "privilege_level": "PRIVILEGED",
            "agent_id": "test_agent",
            "requires_sudo": True,
            "sudo_password": "hunter2",
            "challenge_id": "ch_sec",
            "run_id": "run_sec",
            "context": TOOL_INSTALL_CONTEXT,
            "decision_required": True,
        }

        listing = list_pending_approvals()

        self.assertEqual(len(listing), 1)
        row = listing[0]
        self.assertEqual(row["request_id"], "req_secret")
        self.assertEqual(row["command"], "sudo -S id")
        self.assertEqual(row["context"], TOOL_INSTALL_CONTEXT)
        self.assertEqual(row["challenge_id"], "ch_sec")
        # The two things that must never leave the process.
        self.assertNotIn("sudo_password", row)
        self.assertNotIn("event", row)
        # And the payload is actually serializable (an Event in the dict would not be).
        json.dumps(listing)

    async def test_listing_includes_approvals_owned_by_a_live_swarm_board(self):
        """Swarm runs keep their own per-board registry, so a listing that only read the
        shared one would show the operator an empty panel mid-run."""
        board = SwarmBlackboard(
            challenge_id="ch_board",
            run_id="run_board",
            target_scope="10.10.14.23",
        )
        board.pending_approvals["req_board"] = {
            "event": asyncio.Event(),
            "decision": None,
            "command": "nmap -sV 10.10.14.23",
            "privilege_level": "PRIVILEGED",
            "agent_id": "worker_1",
            "requires_sudo": False,
            "sudo_password": None,
            "challenge_id": "ch_board",
            "run_id": "run_board",
            "context": {},
            "decision_required": True,
        }
        swarm_orchestrator.active_swarms["run_board"] = board

        listing = list_pending_approvals()
        ids = {row["request_id"] for row in listing}
        self.assertIn("req_board", ids)

    async def test_listing_is_json_serializable_with_a_real_pending_gate_request(self):
        """End-to-end: an actual parked gate request must survive the listing round-trip."""
        broadcast, _payloads = _recording_broadcast()

        with _manual_mode():
            task = asyncio.create_task(require_approval(
                cmd=PRIVILEGED_CMD,
                agent_id="test_agent",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=broadcast,
                challenge_id="ch_live",
                context=TOOL_INSTALL_CONTEXT,
            ))
            for _ in range(400):
                if SHARED_PENDING_APPROVALS:
                    break
                await asyncio.sleep(0.05)

            listing = list_pending_approvals()
            json.dumps(listing)  # must not raise
            self.assertEqual(len(listing), 1)
            self.assertEqual(listing[0]["context"], TOOL_INSTALL_CONTEXT)

            self.assertTrue(await _answer_first_pending("deny"))
            await asyncio.wait_for(task, timeout=10)

# =========================================================================== #
# 4. Capability-gap retries are labelled, not re-asked anonymously
# =========================================================================== #

class TestCapabilityGapRetryTagging(_SurfaceTestBase):

    def _board(self) -> SwarmBlackboard:
        return SwarmBlackboard(
            challenge_id="ch_gap",
            run_id="run_gap",
            target_scope="10.10.14.23",
        )

    def test_a_command_matching_a_recorded_gap_is_tagged_as_a_retry(self):
        board = self._board()
        board.record_capability_gap({
            "agent_id": "worker_1",
            "capability": "sqlmap",
            "target": "10.10.14.23",
            "command": "sqlmap -u http://10.10.14.23 --dump",
            "privilege_level": "PRIVILEGED",
            "strategy": "sqli",
            "decision": "deny",
            "reason": "Privilege check denied (PRIVILEGED)",
        })

        context = _capability_gap_retry_context(board, "sqlmap", "PRIVILEGED")

        self.assertIsNotNone(context)
        self.assertEqual(context["request_kind"], "capability_gap_retry")
        self.assertEqual(context["capability"], "sqlmap")
        self.assertEqual(context["target"], "10.10.14.23")
        self.assertEqual(context["previous_decision"], "deny")
        self.assertEqual(context["denied_reason"], "Privilege check denied (PRIVILEGED)")

    def test_a_command_with_no_recorded_gap_is_not_tagged(self):
        """The common case: a first-time command must not be mislabelled a retry."""
        board = self._board()
        self.assertIsNone(_capability_gap_retry_context(board, "nmap", "PRIVILEGED"))

    def test_an_empty_binary_name_is_never_tagged(self):
        """`bin_name` is "" for a blank command; it must not match a gap recorded with an
        empty capability and dress a malformed command up as a deliberate retry."""
        board = self._board()
        board.record_capability_gap({"capability": "", "reason": "blank", "decision": "deny"})
        self.assertIsNone(_capability_gap_retry_context(board, "", "PRIVILEGED"))

    def test_the_newest_gap_wins_when_a_capability_was_denied_more_than_once(self):
        board = self._board()
        board.record_capability_gap({
            "capability": "sqlmap", "target": "old", "decision": "deny", "reason": "first",
        })
        board.record_capability_gap({
            "capability": "sqlmap", "target": "new", "decision": "deny", "reason": "second",
        })

        context = _capability_gap_retry_context(board, "sqlmap", "PRIVILEGED")
        self.assertEqual(context["target"], "new")
        self.assertEqual(context["denied_reason"], "second")

if __name__ == "__main__":
    unittest.main()
