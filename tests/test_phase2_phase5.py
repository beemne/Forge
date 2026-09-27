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

from backend.database.session import init_db, SessionLocal
from backend.database.models import ChallengeModel, TargetProfileModel, RunModel, CheckpointModel, EvidenceModel
from backend.providers.router import model_router
from backend.providers.real_providers import GeminiProvider, OpenAISpecProvider
from backend.tools.manager import tool_manager
from backend.agents.orchestrator_loop import orchestrator_loop
from backend.api.runner import workflow_runner

class TestForgePhase2Phase5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_database_models_persistence(self):
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Postgres DB Persistence Test", category="web")
            db.add(ch)
            db.commit()
            db.refresh(ch)

            target = TargetProfileModel(challenge_id=ch.id, current_address="10.10.10.50")
            db.add(target)
            db.commit()

            fetched = db.query(TargetProfileModel).filter(TargetProfileModel.challenge_id == ch.id).first()
            self.assertEqual(fetched.current_address, "10.10.10.50")
        finally:
            db.close()

    def test_real_provider_initialization_unconfigured(self):
        gemini = GeminiProvider(api_key="")
        openai_spec = OpenAISpecProvider(name="openrouter", is_paid=True, api_key="", default_model="gpt-4o", base_url="https://openrouter.ai/api/v1")
        
        async def check_available():
            return await gemini.is_available(), await openai_spec.is_available()

        gem_avail, oai_avail = asyncio.run(check_available())
        self.assertFalse(gem_avail)
        self.assertFalse(oai_avail)

    def test_orchestrator_loop_execution(self):
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Orchestrator Loop Test", category="recon")
            db.add(ch)
            db.commit()

            target = TargetProfileModel(challenge_id=ch.id, current_address="127.0.0.1")
            db.add(target)
            db.commit()

            run = RunModel(challenge_id=ch.id, status="RUNNING", current_phase="recon", current_agent="recon")
            db.add(run)
            db.commit()

            workflow_runner.start_run(run.id, ch.id, "127.0.0.1")

            async def step():
                return await orchestrator_loop.execute_run_step(db, run.id)

            res = asyncio.run(step())
            self.assertIn("status", res)
            self.assertEqual(res["agent"], "recon")
            self.assertEqual(res["capability"], "network_scanning")

            # Verify evidence and checkpoint creation
            cp = db.query(CheckpointModel).filter(CheckpointModel.run_id == run.id).first()
            self.assertIsNotNone(cp)
        finally:
            db.close()

    def test_kill_switch_halting(self):
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Kill Switch Test", category="web")
            db.add(ch)
            db.commit()

            run = RunModel(challenge_id=ch.id, status="RUNNING", current_phase="web", current_agent="web")
            db.add(run)
            db.commit()

            workflow_runner.start_run(run.id, ch.id, "127.0.0.1")
            workflow_runner.activate_kill_switch(run.id)

            async def step():
                return await orchestrator_loop.execute_run_step(db, run.id)

            res = asyncio.run(step())
            self.assertEqual(res["status"], "CANCELLED")
        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
