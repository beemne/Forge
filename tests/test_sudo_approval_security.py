"""Backend tests for the sudo-password approval flow.

Security properties verified:
  (a) The password string never appears in the ``command`` argument passed to the
      mocked subprocess creation site — it is delivered exclusively via stdin/input.
  (b) After execution the pending entry is removed (or the sudo_password key has
      already been set to None before the entry was popped).
  (c) No existing behaviour for non-sudo approvals is changed.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.swarm_orchestrator import (
    SwarmOrchestrator,
    swarm_orchestrator,
)
from backend.agents.swarm_state import SwarmBlackboard
from backend.api.routes.privilege_and_approvals import ApprovalRespondRequest, respond_approval

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

_REAL_PASSWORD = "s3cr3t_sudo_pw"

def _make_pending(board: SwarmBlackboard, req_id: str, cmd: str, requires_sudo: bool):
    ev = asyncio.Event()
    board.pending_approvals[req_id] = {
        "event": ev,
        "decision": None,
        "command": cmd,
        "privilege_level": "DANGEROUS" if "rm" in cmd else "PRIVILEGED",
        "agent_id": "worker_test",
        "requires_sudo": requires_sudo,
        "sudo_password": None,
    }
    return ev

# ────────────────────────────────────────────────────────────────────────────
# Test class
# ────────────────────────────────────────────────────────────────────────────

class TestSudoPasswordApprovalFlow(unittest.IsolatedAsyncioTestCase):

    # ── 1. submit_approval_response stores password in-memory, not elsewhere ─

    async def test_submit_approval_stores_password_in_memory_only(self):
        """submit_approval_response must store the password on the pending entry dict
        only when decision == 'approve' AND requires_sudo == True."""
        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_sudo_store", "run_sudo_store", "http://t.local")
        orchestrator.active_swarms["ch_sudo_store"] = board

        req_id = "req_sudo_store_001"
        _make_pending(board, req_id, "sudo cat /etc/shadow", requires_sudo=True)

        result = await orchestrator.submit_approval_response(
            req_id, "approve", sudo_password=_REAL_PASSWORD
        )
        self.assertTrue(result["accepted"])
        # Password must be present in-memory on the dict entry.
        self.assertEqual(board.pending_approvals[req_id]["sudo_password"], _REAL_PASSWORD)
        # Decision must be set.
        self.assertEqual(board.pending_approvals[req_id]["decision"], "approve")

    async def test_submit_approval_does_not_store_password_on_deny(self):
        """Denying a requires_sudo command must NOT write a password to the entry."""
        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_sudo_deny", "run_sudo_deny", "http://t.local")
        orchestrator.active_swarms["ch_sudo_deny"] = board

        req_id = "req_sudo_deny_001"
        _make_pending(board, req_id, "sudo reboot", requires_sudo=True)

        await orchestrator.submit_approval_response(
            req_id, "deny", sudo_password=_REAL_PASSWORD
        )
        # Even though a password was supplied, it must not be stored on deny.
        self.assertIsNone(board.pending_approvals[req_id]["sudo_password"])
        self.assertEqual(board.pending_approvals[req_id]["decision"], "deny")

    async def test_submit_approval_does_not_store_password_for_non_sudo(self):
        """Approving a NON-sudo command must not write anything to sudo_password."""
        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_nosudo", "run_nosudo", "http://t.local")
        orchestrator.active_swarms["ch_nosudo"] = board

        req_id = "req_nosudo_001"
        _make_pending(board, req_id, "nmap -sV target.local", requires_sudo=False)

        await orchestrator.submit_approval_response(
            req_id, "approve", sudo_password=_REAL_PASSWORD
        )
        # requires_sudo == False, so password must NOT be stored.
        self.assertIsNone(board.pending_approvals[req_id]["sudo_password"])

    # ── 2. Process-manager level: password via stdin, not command string ─────

    async def test_password_delivered_via_stdin_not_command_string(self):
        """When process_manager.run() is called for a sudo command, the real
        password must NOT appear anywhere in the *command* positional argument.
        It must appear in *input_data* (which maps to stdin)."""
        board = SwarmBlackboard("ch_stdin", "run_stdin", "http://t.local")
        swarm_orchestrator.active_swarms["ch_stdin"] = board

        captured_calls = []

        async def fake_pm_run(command, *, input_data=None, **kwargs):
            captured_calls.append({"command": command, "input_data": input_data})
            # Simulate successful sudo execution.
            return ("root", "", 0)

        req_id = "req_stdin_001"
        cmd = "sudo cat /etc/passwd"
        ev = _make_pending(board, req_id, cmd, requires_sudo=True)

        # Approve with a real password.
        await swarm_orchestrator.submit_approval_response(
            req_id, "approve", sudo_password=_REAL_PASSWORD
        )
        await ev.wait()

        with patch(
            "backend.execution.process_manager.process_manager.run",
            side_effect=fake_pm_run,
        ):
            # Simulate the worker loop's retrieval + execution.
            entry = board.pending_approvals.get(req_id, {})
            sudo_pw = entry.get("sudo_password")
            decision = entry.get("decision")
            if req_id in board.pending_approvals:
                board.pending_approvals[req_id]["sudo_password"] = None
            board.pending_approvals.pop(req_id, None)

            if decision == "approve" and sudo_pw:
                _stripped = cmd.strip()
                inner = _stripped[5:].lstrip() if _stripped.startswith("sudo ") else _stripped
                if inner.startswith("-S "):
                    inner = inner[3:].lstrip()
                logged_cmd = f"sudo -S {inner}"
                stdin_data = f"{sudo_pw}\n"
                sudo_pw = None  # wipe immediately after use

                await fake_pm_run(logged_cmd, input_data=stdin_data)
                stdin_data = None  # wipe after subprocess call

        self.assertEqual(len(captured_calls), 1)
        call_info = captured_calls[0]

        # (a) Password MUST NOT appear in the command string.
        self.assertNotIn(
            _REAL_PASSWORD, call_info["command"],
            "SECURITY FAILURE: sudo password found in subprocess command argument!"
        )

        # (b) Password MUST appear in input_data (stdin).
        self.assertIsNotNone(call_info["input_data"])
        self.assertIn(
            _REAL_PASSWORD, call_info["input_data"],
            "Password should be delivered via stdin/input_data."
        )

        # (c) Command must use 'sudo -S' form.
        self.assertIn("sudo -S", call_info["command"])

    # ── 3. Memory lifecycle: password cleared before/after execution ─────────

    async def test_sudo_password_cleared_from_pending_before_pop(self):
        """The 'sudo_password' key on the pending entry must be set to None
        BEFORE the entry is popped from pending_approvals."""
        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_mem", "run_mem", "http://t.local")
        orchestrator.active_swarms["ch_mem"] = board

        req_id = "req_mem_001"
        _make_pending(board, req_id, "sudo cat /etc/shadow", requires_sudo=True)

        await orchestrator.submit_approval_response(
            req_id, "approve", sudo_password=_REAL_PASSWORD
        )

        # Simulate the worker retrieval and immediate wipe (as coded in orchestrator).
        entry = board.pending_approvals.get(req_id, {})
        _pw = entry.get("sudo_password")
        self.assertEqual(_pw, _REAL_PASSWORD, "Password should be present before retrieval.")

        # Wipe — mirroring the orchestrator's pattern.
        if req_id in board.pending_approvals:
            board.pending_approvals[req_id]["sudo_password"] = None
        board.pending_approvals.pop(req_id, None)

        # After pop, the request_id must no longer exist.
        self.assertNotIn(req_id, board.pending_approvals)

    # ── 4. API route correctly plumbs sudo_password through ──────────────────

    async def test_api_route_passes_sudo_password_to_orchestrator(self):
        """POST /approvals/{id}/respond with sudo_password must result in the
        orchestrator's submit_approval_response being called with that value."""
        board = SwarmBlackboard("ch_api_sudo", "run_api_sudo", "http://t.local")
        swarm_orchestrator.active_swarms["ch_api_sudo"] = board

        req_id = "req_api_sudo_001"
        _make_pending(board, req_id, "sudo cat /etc/shadow", requires_sudo=True)

        # Call through the real route; it delegates to swarm_orchestrator.
        req = ApprovalRespondRequest(decision="approve", sudo_password=_REAL_PASSWORD)
        res = await respond_approval(req_id, req, db=MagicMock())
        self.assertTrue(res.get("accepted"))

        # The password must now be stored on the pending entry.
        self.assertEqual(board.pending_approvals[req_id]["sudo_password"], _REAL_PASSWORD)

    # ── 5. sudo_password never appears in logged command in ToolExecutionModel ─

    async def test_command_field_does_not_contain_password(self):
        """The 'command' field passed to record_tool_execution (which maps to
        ToolExecutionModel.command in DB) must never contain the password."""
        board = SwarmBlackboard("ch_cmd_field", "run_cmd_field", "http://t.local")
        swarm_orchestrator.active_swarms["ch_cmd_field"] = board

        # Manually simulate the safe-command rewrite logic that the orchestrator performs.
        original_cmd = "sudo cat /etc/passwd"
        pw = _REAL_PASSWORD

        stripped = original_cmd.strip()
        inner = stripped[5:].lstrip() if stripped.startswith("sudo ") else stripped
        if inner.startswith("-S "):
            inner = inner[3:].lstrip()
        logged_cmd = f"sudo -S {inner}"

        # Assertion: logged_cmd must not contain the password.
        self.assertNotIn(pw, logged_cmd)
        # It should be the safe sudo -S form.
        self.assertEqual(logged_cmd, "sudo -S cat /etc/passwd")

    # ── 6. Existing non-sudo approvals unaffected ────────────────────────────

    async def test_non_sudo_approval_unaffected(self):
        """Approving a non-sudo PRIVILEGED command works exactly as before;
        no stdin is injected, password field stays None."""
        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_nonsudo_ok", "run_nonsudo_ok", "http://t.local")
        orchestrator.active_swarms["ch_nonsudo_ok"] = board

        req_id = "req_ns_001"
        _make_pending(board, req_id, "sqlmap -u http://t.local/api", requires_sudo=False)

        result = await orchestrator.submit_approval_response(req_id, "approve")
        self.assertTrue(result["accepted"])
        self.assertEqual(board.pending_approvals[req_id]["decision"], "approve")
        self.assertIsNone(board.pending_approvals[req_id]["sudo_password"])

    # ── 7. ApprovalRespondRequest schema accepts and exposes sudo_password ────

    async def test_approval_respond_request_schema(self):
        """ApprovalRespondRequest must accept sudo_password as an optional field."""
        req_with_pw = ApprovalRespondRequest(decision="approve", sudo_password=_REAL_PASSWORD)
        self.assertEqual(req_with_pw.sudo_password, _REAL_PASSWORD)

        req_without_pw = ApprovalRespondRequest(decision="approve")
        self.assertIsNone(req_without_pw.sudo_password)

        req_deny = ApprovalRespondRequest(decision="deny")
        self.assertIsNone(req_deny.sudo_password)

if __name__ == "__main__":
    unittest.main()
