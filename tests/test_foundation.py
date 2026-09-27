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
from backend.tools.registry import tool_registry
from backend.tools.manager import tool_manager

class TestFoundation(unittest.TestCase):

    def test_environment_detector(self):
        env = environment_detector.detect_environment()
        self.assertIn("os", env)
        self.assertIn("installed_tools", env)
        self.assertIsInstance(env["cpu_cores"], int)

    def test_tool_registry(self):
        tools = tool_registry.get_tools_for_capability("directory_enumeration")
        self.assertGreater(len(tools), 0)
        self.assertTrue(any(t.tool_name == "ffuf" for t in tools))

    def test_live_provider_routing(self):
        async def run_async():
            return await model_router.route_request(
                prompt="Analyze web target",
                capability="directory_enumeration"
            )
        response = asyncio.run(run_async())
        self.assertIn(response.provider_name, ["groq", "cloudflare", "openrouter", "gemini", "nvidia", "agentrouter_claude_code", "agentrouter_codex", "rapidapi_gpt54_mini", "rapidapi_deepseek_v32", "rapidapi_gpt5_nano", "mistral", "mistral_codestral", "xkiro", "xkiro_coder", "xkiro_planner", "xkiro_mistral", "none"])
        self.assertTrue(len(response.content) > 0)

    def test_findings_schema_and_challenge_delete(self):
        from backend.database.session import init_db, SessionLocal
        from backend.database.models import ChallengeModel, FindingModel
        import tempfile
        import shutil

        init_db()
        db = SessionLocal()
        try:
            # Create a test working directory
            test_dir = tempfile.mkdtemp(prefix="forge_test_ctf_")
            ch = ChallengeModel(
                name="Test Challenge To Delete",
                category="WEB",
                difficulty="EASY",
                working_directory=test_dir,
                status="RUNNING"
            )
            db.add(ch)
            db.commit()
            db.refresh(ch)

            # Add a finding with severity and endpoint
            f = FindingModel(
                challenge_id=ch.id,
                agent="web",
                title="Test SQLi",
                vulnerability_class="sqli",
                severity="HIGH",
                endpoint="/login"
            )
            db.add(f)
            db.commit()

            # Query findings
            findings = db.query(FindingModel).filter(FindingModel.challenge_id == ch.id).all()
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].severity, "HIGH")
            self.assertEqual(findings[0].endpoint, "/login")

            # Delete challenge via cascade
            db.delete(ch)
            db.commit()

            # Verify finding was cascade deleted
            findings_after = db.query(FindingModel).filter(FindingModel.challenge_id == ch.id).all()
            self.assertEqual(len(findings_after), 0)

            # Clean up test directory
            shutil.rmtree(test_dir, ignore_errors=True)
            self.assertFalse(os.path.exists(test_dir))
        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
