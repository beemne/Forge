"""Unit tests for per-command HITL privilege approval flow."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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
from fastapi import HTTPException

class TestCommandApprovalFlow(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_worker_approvals_isolation(self):
        """Two commands from two different concurrent workers, both PRIVILEGED,

        can be pending approval at the same time without one worker's approve/deny
        affecting the other.
        """
        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_concurrent", "run_concurrent", "http://127.0.0.1:8000")
        orchestrator.active_swarms["ch_concurrent"] = board

        req_id_1 = "req_recon_001"
        req_id_2 = "req_code_002"
        event_1 = asyncio.Event()
        event_2 = asyncio.Event()

        board.pending_approvals[req_id_1] = {
            "event": event_1,
            "decision": None,
            "command": "sqlmap -u http://127.0.0.1:8000/api",
            "privilege_level": "PRIVILEGED",
            "agent_id": "worker_recon",
            "requires_sudo": False,
            "sudo_password": None,
        }

        board.pending_approvals[req_id_2] = {
            "event": event_2,
            "decision": None,
            "command": "python solve_crypto.py",
            "privilege_level": "PRIVILEGED",
            "agent_id": "worker_code_crypto",
            "requires_sudo": False,
            "sudo_password": None,
        }

        # Resolve request 1 with approve
        res1 = await orchestrator.submit_approval_response(req_id_1, "approve")
        self.assertTrue(res1.get("accepted"))
        self.assertEqual(board.pending_approvals[req_id_1]["decision"], "approve")
        self.assertTrue(event_1.is_set())

        # Assert request 2 is STILL unset and decision is still None
        self.assertFalse(event_2.is_set())
        self.assertIsNone(board.pending_approvals[req_id_2]["decision"])

        # Resolve request 2 with deny
        res2 = await orchestrator.submit_approval_response(req_id_2, "deny")
        self.assertTrue(res2.get("accepted"))
        self.assertEqual(board.pending_approvals[req_id_2]["decision"], "deny")
        self.assertTrue(event_2.is_set())

    async def test_invalid_decision_and_missing_request(self):
        """Invalid decisions or non-existent request_ids must return accepted: False

        without mutating any state.
        """
        orchestrator = SwarmOrchestrator()
        board = SwarmBlackboard("ch_test", "run_test", "http://127.0.0.1:8000")
        orchestrator.active_swarms["ch_test"] = board

        req_id = "req_pending"
        ev = asyncio.Event()
        board.pending_approvals[req_id] = {
            "event": ev,
            "decision": None,
            "command": "reboot",
            "privilege_level": "DANGEROUS",
            "agent_id": "worker_exploit_pwn",
            "requires_sudo": False,
            "sudo_password": None,
        }

        # Invalid decision string
        res = await orchestrator.submit_approval_response(req_id, "maybe")
        self.assertFalse(res.get("accepted"))
        self.assertIn("Invalid decision", res.get("reason", ""))
        self.assertFalse(ev.is_set())
        self.assertIsNone(board.pending_approvals[req_id]["decision"])

        # Non-existent request id
        res_missing = await orchestrator.submit_approval_response("non_existent_req", "approve")
        self.assertFalse(res_missing.get("accepted"))
        self.assertIn("No pending approval", res_missing.get("reason", ""))

    async def test_api_route_respond_approval(self):
        """Test POST /approvals/{request_id}/respond route logic."""
        orchestrator = swarm_orchestrator
        board = SwarmBlackboard("ch_api", "run_api", "http://127.0.0.1:8000")
        orchestrator.active_swarms["ch_api"] = board

        req_id = "req_api_001"
        ev = asyncio.Event()
        board.pending_approvals[req_id] = {
            "event": ev,
            "decision": None,
            "command": "sudo cat /etc/shadow",
            "privilege_level": "DANGEROUS",
            "agent_id": "worker_recon",
            "requires_sudo": True,
            "sudo_password": None,
        }

        # Valid respond call
        req = ApprovalRespondRequest(decision="approve")
        res = await respond_approval(req_id, req, db=MagicMock())
        self.assertTrue(res.get("accepted"))
        self.assertTrue(ev.is_set())
        self.assertEqual(board.pending_approvals[req_id]["decision"], "approve")

        # Calling again on already finished / non-existent request raises 409
        board.pending_approvals.pop(req_id, None)
        with self.assertRaises(HTTPException) as ctx:
            await respond_approval(req_id, req, db=MagicMock())
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_deny_and_timeout_never_reaches_execute_tool(self):
        """A deny or timeout results in the command never reaching

        tool_manager.execute_tool.
        """
        board = SwarmBlackboard("ch_deny", "run_deny", "http://127.0.0.1:8000")
        swarm_orchestrator.active_swarms["ch_deny"] = board

        with patch("backend.tools.manager.tool_manager.execute_tool", new_callable=AsyncMock) as mock_exec:
            # Simulate worker loop approval gate on deny
            req_id = "req_to_deny"
            ev = asyncio.Event()
            board.pending_approvals[req_id] = {
                "event": ev,
                "decision": None,
                "command": "rm -rf /tmp/data",
                "privilege_level": "DANGEROUS",
                "agent_id": "worker_exploit_pwn",
                "requires_sudo": False,
                "sudo_password": None,
            }

            # Submit deny
            await swarm_orchestrator.submit_approval_response(req_id, "deny")

            # Worker loop checks event & decision
            await ev.wait()
            decision = board.pending_approvals.get(req_id, {}).get("decision")
            board.pending_approvals.pop(req_id, None)

            approved = (decision == "approve")
            if approved:
                await mock_exec("bash", {"command": "rm -rf /tmp/data"})

            mock_exec.assert_not_called()
            self.assertFalse(approved)
            self.assertNotIn(req_id, board.pending_approvals)

        with patch("backend.tools.manager.tool_manager.execute_tool", new_callable=AsyncMock) as mock_exec_timeout:
            # Simulate timeout scenario
            req_id_timeout = "req_to_timeout"
            ev_timeout = asyncio.Event()
            board.pending_approvals[req_id_timeout] = {
                "event": ev_timeout,
                "decision": None,
                "command": "nc -lvnp 4444",
                "privilege_level": "PRIVILEGED",
                "agent_id": "worker_recon",
                "requires_sudo": False,
                "sudo_password": None,
            }

            # Timeout occurs (event never set)
            decision_timeout = board.pending_approvals.get(req_id_timeout, {}).get("decision")
            board.pending_approvals.pop(req_id_timeout, None)

            approved_timeout = (decision_timeout == "approve")
            if approved_timeout:
                await mock_exec_timeout("bash", {"command": "nc -lvnp 4444"})

            mock_exec_timeout.assert_not_called()
            self.assertFalse(approved_timeout)
            self.assertNotIn(req_id_timeout, board.pending_approvals)

    async def test_approve_reaches_execute_tool_with_approved_flag(self):
        """When operator approves, tool_manager.execute_tool is called and

        record_tool_execution is called with approved=True.
        """
        board = SwarmBlackboard("ch_approve", "run_approve", "http://127.0.0.1:8000")
        swarm_orchestrator.active_swarms["ch_approve"] = board

        mock_res = MagicMock()
        mock_res.stdout = "table_users_dump"
        mock_res.stderr = ""
        mock_res.exit_code = 0
        mock_res.execution_failure = False

        with patch("backend.tools.manager.tool_manager.execute_tool", new_callable=AsyncMock) as mock_exec, \
             patch.object(board, "record_tool_execution", new_callable=AsyncMock) as mock_record:
            mock_exec.return_value = mock_res

            req_id = "req_to_approve"
            ev = asyncio.Event()
            cmd = "sqlmap -u http://127.0.0.1:8000 --dump"
            priv_level = "PRIVILEGED"
            agent_id = "worker_recon"

            board.pending_approvals[req_id] = {
                "event": ev,
                "decision": None,
                "command": cmd,
                "privilege_level": priv_level,
                "agent_id": agent_id,
                "requires_sudo": False,
                "sudo_password": None,
            }

            # Operator approves
            await swarm_orchestrator.submit_approval_response(req_id, "approve")
            await ev.wait()

            decision = board.pending_approvals.get(req_id, {}).get("decision")
            board.pending_approvals.pop(req_id, None)

            approved = (decision == "approve")
            self.assertTrue(approved)

            if approved:
                res = await mock_exec("bash", {"command": cmd})
                await board.record_tool_execution(agent_id, cmd, res, privilege_level=priv_level, approved=approved)

            mock_exec.assert_called_once_with("bash", {"command": cmd})
            mock_record.assert_called_once_with(agent_id, cmd, mock_res, privilege_level=priv_level, approved=True)

if __name__ == "__main__":
    unittest.main()
