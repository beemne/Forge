"""Tests for Capability Gap classification, intent preservation, and privilege approval re-entry."""

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

from backend.config import settings
from backend.database.session import SessionLocal, init_db
from backend.agent_runtime.recovery import RecoveryEngine, FailureCategory, RecoveryStrategy
from backend.swarm.reasoning import FailureClass, RecoveryHint, classify_failure, recovery_hint_for
from backend.swarm.supervisor import Supervisor
from backend.swarm.tasks import Task
from backend.agent_runtime.state import MissionState
from backend.agents.swarm_state import SwarmBlackboard
from backend.agent_runtime.action import Action, ActionType, ExecResult
from backend.agent_runtime.runtime import RealToolExecutor, AgentRuntime
from backend.agent_runtime.session import AgentSession
from backend.privilege.gate import require_approval, SHARED_PENDING_APPROVALS
from backend.agents.swarm_orchestrator import swarm_orchestrator

class TestCapabilityGapClassification(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_recovery_engine_classifies_capability_gap(self):
        engine = RecoveryEngine()
        exec_res = ExecResult(
            command="sqlmap -u http://example.com",
            status="FAILED",
            stderr="[PRIVILEGE DENIED] Operator did not approve this command",
            exit_code=-1,
            execution_failure=True,
            failure_category="CAPABILITY_GAP",
        )
        plan = engine.diagnose(exec_result=exec_res)
        self.assertEqual(plan.category, FailureCategory.CAPABILITY_GAP)
        self.assertEqual(plan.strategy, RecoveryStrategy.ESCALATE_PRIVILEGE)
        self.assertTrue(plan.escalate)
        self.assertTrue(plan.needs_action)

    def test_reasoning_classifies_capability_gap(self):
        # 1. From failure_category="CAPABILITY_GAP"
        res1 = MagicMock(status="FAILED", failure_category="CAPABILITY_GAP", reason="")
        self.assertEqual(classify_failure(res1), FailureClass.CAPABILITY_GAP)
        self.assertEqual(recovery_hint_for(FailureClass.CAPABILITY_GAP), RecoveryHint.RECOVER_CAPABILITY)

        # 2. From failure_category="PRIVILEGE_DENIED"
        res2 = MagicMock(status="FAILED", failure_category="PRIVILEGE_DENIED", reason="")
        self.assertEqual(classify_failure(res2), FailureClass.CAPABILITY_GAP)

        # 3. From reason text mentioning privilege denial
        res3 = MagicMock(status="FAILED", failure_category="", reason="Privilege check denied for tool")
        self.assertEqual(classify_failure(res3), FailureClass.CAPABILITY_GAP)

    def test_supervisor_classifies_and_decides_capability_gap(self):
        sup = Supervisor("mission_1")
        res = MagicMock(status="FAILED", failure_category="CAPABILITY_GAP", reason="")
        category = sup.classify_failure(res)
        self.assertEqual(category, "capability_gap")

        task = Task(mission_id="m1", role="web", objective="Extract flag using sqlmap")
        decision = sup.decide_recovery(task, res, max_retries=3)
        self.assertEqual(decision.action, "retry")
        self.assertIn("escalat", decision.reason.lower())

class TestIntentPreservationAndApproval(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        init_db()
        SHARED_PENDING_APPROVALS.clear()

    def test_mission_state_records_capability_gap_intent(self):
        state = MissionState(target="http://target.local")
        state.record_capability_gap(
            action="sqlmap -u http://target.local",
            capability="sqlmap",
            target="http://target.local",
            reason="Privilege denied by operator policy",
            privilege_level="PRIVILEGED",
        )
        self.assertEqual(len(state.capability_gaps), 1)
        gap = state.capability_gaps[0]
        self.assertEqual(gap["capability"], "sqlmap")
        self.assertEqual(gap["target"], "http://target.local")
        self.assertEqual(gap["privilege_level"], "PRIVILEGED")
        self.assertIn("timestamp", gap)

    def test_swarm_blackboard_preserves_intent_in_snapshot(self):
        board = SwarmBlackboard("ch_1", "run_1", target_scope="http://target.local")
        intent = {
            "agent_id": "agent_1",
            "capability": "nmap",
            "target": "http://target.local",
            "command": "nmap -sV http://target.local",
            "privilege_level": "PRIVILEGED",
            "strategy": "recon_deep",
            "decision": "deny",
            "reason": "Privilege check denied (PRIVILEGED)",
        }
        board.record_capability_gap(intent)
        self.assertEqual(len(board.capability_gaps), 1)
        self.assertEqual(board.capability_gaps[0]["capability"], "nmap")

        # Snapshot round-trip
        snap = board.save_snapshot()
        board2 = SwarmBlackboard("ch_1", "run_1", target_scope="http://target.local")
        board2.load_snapshot(snap["blackboard_state"])
        self.assertEqual(len(board2.capability_gaps), 1)
        self.assertEqual(board2.capability_gaps[0]["command"], "nmap -sV http://target.local")

    async def test_preserved_intent_reenters_approval_pipeline_auto_mode(self):
        """Preserved privileged action re-enters require_approval and auto-approves when AUTO_APPROVE_PRIVILEGED=True."""
        with patch.object(settings, "AUTO_APPROVE_PRIVILEGED", True):
            approved, decision, sudo_pw = await require_approval(
                cmd="sqlmap -u http://target.local",
                agent_id="agent_1",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=AsyncMock(),
                challenge_id=None,
                run_id=None,
            )
            self.assertTrue(approved)
            self.assertEqual(decision, "auto-approved")

    async def test_preserved_intent_reenters_approval_pipeline_manual_mode(self):
        """Preserved action enters require_approval in manual mode and can be approved/denied via existing respond endpoint."""
        with patch.object(settings, "AUTO_APPROVE_PRIVILEGED", False), \
             patch.object(settings, "FORGE_APPROVAL_MODE", "manual"):

            async def resolve_later():
                for _ in range(50):
                    await asyncio.sleep(0.05)
                    if SHARED_PENDING_APPROVALS:
                        req_id = list(SHARED_PENDING_APPROVALS)[0]
                        resp = await swarm_orchestrator.submit_approval_response(
                            req_id, "approve", sudo_password=None
                        )
                        self.assertTrue(resp.get("accepted"))
                        return

            resolver_task = asyncio.create_task(resolve_later())

            approved, decision, _ = await require_approval(
                cmd="sqlmap -u http://target.local",
                agent_id="agent_1",
                pending_approvals=SHARED_PENDING_APPROVALS,
                broadcast_fn=AsyncMock(),
                challenge_id=None,
                run_id=None,
            )
            await resolver_task
            self.assertTrue(approved)
            self.assertEqual(decision, "approve")

if __name__ == "__main__":
    unittest.main()
