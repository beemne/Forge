import unittest
import asyncio
import os

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.environment.detector import environment_detector
from backend.providers.router import model_router
from backend.providers.live_providers import GeminiProvider, NvidiaProvider, CerebrasProvider
from backend.tools.registry import tool_registry
from backend.tools.manager import tool_manager
from backend.agents.manager import agent_manager
from backend.database.session import init_db, SessionLocal
from backend.database.models import ChallengeModel, RunModel
from backend.checkpoints.manager import checkpoint_manager
from backend.reporting.generator import report_generator

class TestForgeFullStack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_environment_detector(self):
        env = environment_detector.detect_environment()
        self.assertIn("os", env)
        self.assertIn("installed_tools", env)

    def test_live_provider_registration(self):
        gemini = GeminiProvider(api_key="test-key")
        model_router.register_provider("gemini", gemini)
        self.assertIn("gemini", model_router.providers)

    def test_agent_manager_resolution(self):
        agent = agent_manager.get_agent("recon")
        self.assertEqual(agent.name, "recon")

    def test_checkpoint_creation(self):
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Test Challenge", category="web")
            db.add(ch)
            db.commit()

            # Create the parent run the checkpoint references. checkpoints.run_id
            # is a FK to runs.id and SQLite enforces it (PRAGMA foreign_keys=ON),
            # so the run must exist first — the test must not rely on a stale row
            # left in the shared test_forge.db by an earlier run.
            run = RunModel(challenge_id=ch.id, status="RUNNING", current_phase="recon")
            db.add(run)
            db.commit()

            cp = checkpoint_manager.create_checkpoint(
                db=db,
                run_id=run.id,
                current_phase="recon",
                current_agent="orchestrator",
                last_action="nmap_scan",
                state_snapshot={"target": "10.10.10.5"}
            )
            self.assertIsNotNone(cp.id)
            self.assertEqual(cp.last_successful_action, "nmap_scan")
        finally:
            db.close()

    def test_report_generation(self):
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Report Test", category="web")
            db.add(ch)
            db.commit()
            
            path = report_generator.generate_readme(db, ch.id)
            self.assertTrue(path.endswith(".md"))
        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
