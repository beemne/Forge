import os
import sys
import unittest
sys.path.insert(0, ".")

from fastapi.testclient import TestClient

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.main import app
from backend.database.session import get_db, get_engine, SessionLocal, init_db
from backend.database.models import Base, ChallengeModel, TargetProfileModel, ProviderUsageModel, TrajectoryEventModel, RunModel, FindingModel, EvidenceModel, CheckpointModel, AgentStateModel, ReportModel

class TestDataWiringEndpoints(unittest.TestCase):
    """Test suite verifying full data wiring endpoints for candidates, artifacts, decisions, targets, and providers."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()
        try:
            # Clean test records in FK order
            self.db.query(AgentStateModel).delete()
            self.db.query(CheckpointModel).delete()
            self.db.query(EvidenceModel).delete()
            self.db.query(FindingModel).delete()
            self.db.query(ReportModel).delete()
            self.db.query(RunModel).delete()
            self.db.query(TargetProfileModel).delete()
            self.db.query(ChallengeModel).delete()
            self.db.query(ProviderUsageModel).delete()
            self.db.query(TrajectoryEventModel).delete()
            self.db.commit()
        finally:
            self.db.close()
            self.db = SessionLocal()

    def tearDown(self):
        if hasattr(self, "db") and self.db:
            self.db.close()

    def test_challenge_candidates_artifact_decisions_endpoints(self):
        """Verify GET /challenges/{id}/candidates, /derived-artifacts, and /decisions return persisted/live state."""
        # 1. Create a challenge with persisted blackboard snapshot
        ch = ChallengeModel(
            id="ch-test-wiring-1",
            name="Test Wiring Challenge",
            category="WEB",
            status="RUNNING",
            mission_plan={
                "blackboard_state": {
                    "flag_candidates": [{"flag": "picoCTF{test_candidate_123}", "worker": "RECON", "source": "regex"}],
                    "derived_artifacts": [{"filename": "decoded_secret.png", "artifact_type": "image", "size_bytes": 1024, "status": "RECONSTRUCTED"}]
                }
            }
        )
        self.db.add(ch)
        self.db.commit()
        self.db.close()

        # 2. Simulate an actual _agent_worker turn to test real pipeline persistence
        import asyncio
        from unittest.mock import AsyncMock, patch
        from backend.agents.swarm_orchestrator import SwarmOrchestrator
        from backend.agents.swarm_state import SwarmBlackboard

        board = SwarmBlackboard("ch-test-wiring-1", "run-wiring-1", "http://target.local")
        board.env_info = {"os": "windows", "tools": {}, "cpu_cores": 2}
        board.flag_candidates = [{"flag": "picoCTF{test_candidate_123}", "worker": "RECON", "source": "regex"}]
        board.derived_artifacts = [{"filename": "decoded_secret.png", "artifact_type": "image", "size_bytes": 1024, "status": "RECONSTRUCTED"}]
        orchestrator = SwarmOrchestrator()
        from backend.agents.swarm_orchestrator import swarm_orchestrator
        swarm_orchestrator.active_swarms["ch-test-wiring-1"] = board

        mock_tool_res = AsyncMock()
        mock_tool_res.exit_code = 0
        mock_tool_res.stdout = "Found 200 OK on /admin"
        mock_tool_res.stderr = ""
        mock_tool_res.execution_failure = False

        turns = []

        async def fake_route_request(*args, **kwargs):
            # _agent_worker has no turn cap by design: it runs until the board stops
            # (flag captured or is_stopped), so max_iterations does not bound it. This
            # fake returns the same command every time, which means that after the first
            # execution every later turn is a duplicate and the worker spins on the
            # dedup path forever. Stop the board after the first turn so the loop ends
            # through its real termination signal.
            turns.append(1)
            if len(turns) > 1:
                board.is_stopped = True
            resp = AsyncMock()
            resp.is_refusal = False
            resp.content = "I will scan the target:\n```bash\nffuf -u http://target.local/FUZZ -w wordlist.txt\n```"
            resp.model_name = "test-model"
            return resp

        async def fake_execute_tool(*args, **kwargs):
            return mock_tool_res

        with patch("backend.agents.swarm_orchestrator.model_router.route_request", side_effect=fake_route_request), \
             patch("backend.agents.swarm_orchestrator.tool_manager.execute_tool", side_effect=fake_execute_tool):
            asyncio.run(orchestrator._agent_worker("agent_1", board, ".", "web_analysis"))

        # 3. Query GET /challenges/{id}/candidates
        resp_cand = self.client.get("/api/challenges/ch-test-wiring-1/candidates")
        self.assertEqual(resp_cand.status_code, 200)
        cands = resp_cand.json().get("candidates", [])
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["flag"], "picoCTF{test_candidate_123}")

        # 4. Query GET /challenges/{id}/derived-artifacts
        resp_art = self.client.get("/api/challenges/ch-test-wiring-1/derived-artifacts")
        self.assertEqual(resp_art.status_code, 200)
        arts = resp_art.json().get("artifacts", [])
        self.assertEqual(len(arts), 1)
        self.assertEqual(arts[0]["filename"], "decoded_secret.png")

        # 5. Query GET /challenges/{id}/decisions (verifying decision persisted by _agent_worker)
        resp_dec = self.client.get("/api/challenges/ch-test-wiring-1/decisions")
        self.assertEqual(resp_dec.status_code, 200)
        decs = resp_dec.json().get("decisions", [])
        self.assertTrue(len(decs) >= 1)
        self.assertEqual(decs[0]["agent"], "agent_1")
        self.assertIn("ffuf", decs[0]["goal"] + decs[0]["selectedTool"] + decs[0]["result"])

    def test_targets_endpoint_returns_real_fields(self):
        """Verify GET /targets returns expected_services, technologies, address_history, discovery_method."""
        ch = ChallengeModel(id="ch-target-test", name="Target Test", category="WEB", status="RUNNING")
        self.db.add(ch)

        target = TargetProfileModel(
            id="t-1",
            challenge_id="ch-target-test",
            current_address="10.10.10.10",
            hostname="target.ctf",
            expected_services=[{"port": 8080, "proto": "tcp", "service": "HTTP-ALT", "version": "Apache 2.4"}],
            technologies=["Linux", "Apache", "PHP"],
            address_history=["10.10.10.10", "192.168.1.50"],
            discovery_method="TURBO RECON",
            verification_status="verified_network"
        )
        self.db.add(target)
        self.db.commit()

        resp = self.client.get("/api/targets")
        self.assertEqual(resp.status_code, 200)
        targets_list = resp.json()
        self.assertTrue(len(targets_list) >= 1)
        t_data = targets_list[0]
        self.assertEqual(t_data["current_address"], "10.10.10.10")
        self.assertEqual(t_data["expected_services"][0]["port"], 8080)
        self.assertIn("PHP", t_data["technologies"])
        self.assertEqual(t_data["discovery_method"], "TURBO RECON")

    def test_providers_endpoint_returns_requests_errors_priority(self):
        """Verify GET /providers returns calculated requests, last_error, and fallback_priority."""
        usage = ProviderUsageModel(
            provider_name="groq",
            model_name="groq-compound",
            prompt_tokens=100,
            completion_tokens=50,
            success=True
        )
        self.db.add(usage)
        self.db.commit()

        resp = self.client.get("/api/providers")
        self.assertEqual(resp.status_code, 200)
        provs = resp.json()
        self.assertTrue(len(provs) > 0)
        groq_prov = next((p for p in provs if p["name"] == "groq"), None)
        self.assertIsNotNone(groq_prov)
        self.assertIn("requests", groq_prov)
        self.assertIn("last_error", groq_prov)
        self.assertIn("fallback_priority", groq_prov)
        self.assertEqual(groq_prov["requests"], 1)

if __name__ == "__main__":
    unittest.main()
