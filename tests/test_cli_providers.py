import unittest
import os
import sys
import asyncio
import tempfile

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import settings
from backend.providers.cli.claude_code import AgentRouterClaudeCodeProvider
from backend.providers.cli.codex import AgentRouterCodexProvider
from backend.providers.cli.base import redact_secrets
from backend.providers.router import model_router
from backend.api.runner import workflow_runner
from backend.database.session import init_db, SessionLocal
from backend.database.models import ChallengeModel, TargetProfileModel, RunModel, CheckpointModel

class TestCLIProviders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_1_claude_code_installed_detection(self):
        prov = AgentRouterClaudeCodeProvider()
        self.assertTrue(asyncio.run(prov.is_available()))

    def test_2_claude_code_missing_fallback(self):
        prov = AgentRouterClaudeCodeProvider()
        prov.custom_path = "/non/existent/path/claude_fake"
        prov.cli_binary_default = "non_existent_claude_binary_99"
        self.assertFalse(asyncio.run(prov.is_available()))

    def test_3_codex_installed_or_missing_discovery(self):
        prov = AgentRouterCodexProvider()
        exe = prov.find_executable()
        # Codex CLI may or may not be installed on host system
        self.assertTrue(exe is None or isinstance(exe, str))

    def test_4_missing_api_key_handling(self):
        prov = AgentRouterClaudeCodeProvider()
        orig_key = settings.AGENTROUTER_API_KEY
        orig_opus5 = settings.AGENTROUTER_CLAUDE_OPUS_5_KEY
        # Sentinels, not truthiness: now that backend/config.py no longer calls
        # load_dotenv(override=True), these variables are absent unless the operator
        # exported them -- so "restore only if truthy" would leave them deleted for the
        # rest of the process. Restore the exact prior state instead.
        missing = object()
        orig_env_key = os.environ.get("AGENTROUTER_API_KEY", missing)
        orig_env_opus5 = os.environ.get("AGENTROUTER_CLAUDE_OPUS_5_KEY", missing)
        try:
            settings.AGENTROUTER_API_KEY = ""
            settings.AGENTROUTER_CLAUDE_OPUS_5_KEY = ""
            if "AGENTROUTER_API_KEY" in os.environ:
                del os.environ["AGENTROUTER_API_KEY"]
            if "AGENTROUTER_CLAUDE_OPUS_5_KEY" in os.environ:
                del os.environ["AGENTROUTER_CLAUDE_OPUS_5_KEY"]
            res = asyncio.run(prov.generate_response("Test prompt", model="claude-opus-5"))
            self.assertTrue(res.is_refusal)
            self.assertIn("No API key", res.refusal_reason)
        finally:
            settings.AGENTROUTER_API_KEY = orig_key
            settings.AGENTROUTER_CLAUDE_OPUS_5_KEY = orig_opus5
            for name, value in (
                ("AGENTROUTER_API_KEY", orig_env_key),
                ("AGENTROUTER_CLAUDE_OPUS_5_KEY", orig_env_opus5),
            ):
                if value is missing:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_5_correct_model_key_mapping_resolution(self):
        prov = AgentRouterClaudeCodeProvider()
        try:
            settings.AGENTROUTER_CLAUDE_OPUS_5_KEY = "sk-test-opus5-key-12345"
            key = prov.resolve_api_key("claude-opus-5")
            self.assertEqual(key, "sk-test-opus5-key-12345")
        finally:
            settings.AGENTROUTER_CLAUDE_OPUS_5_KEY = ""

    def test_6_secret_redaction(self):
        secret = "sk-super-secret-agentrouter-key-99999"
        sample_log = f"Error calling endpoint with auth {secret} at https://agentrouter.org"
        cleaned = redact_secrets(sample_log, [secret])
        self.assertNotIn(secret, cleaned)
        self.assertIn("[REDACTED_API_KEY]", cleaned)

    def test_7_model_router_cli_direct_routing(self):
        async def run_routing():
            return await model_router.route_request(
                prompt="CLI test prompt",
                capability="code_analysis",
                target_model="claude-opus-5"
            )
        res = asyncio.run(run_routing())
        self.assertIsNotNone(res)
        # Integration path: a non-refusal requires a live provider for
        # claude-opus-5 (the AgentRouter Claude Code CLI plus its API key) or a
        # configured fallback. On an unprovisioned host the router *correctly*
        # reports that no live provider could serve the request — that exercises
        # the exhaustion path, not a routing bug — so skip instead of failing.
        reason = (res.refusal_reason or "").lower()
        if res.is_refusal and any(k in reason for k in ("exhausted", "paid models are disabled", "budget")):
            self.skipTest(
                "No provider configured for claude-opus-5 in this environment "
                "(no AgentRouter key / fallback); CLI routing logic is unexercised here."
            )
        self.assertFalse(res.is_refusal)

    def test_8_kill_switch_process_cancellation(self):
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="CLI Kill Switch Test", category="web")
            db.add(ch)
            db.commit()

            run = RunModel(challenge_id=ch.id, status="RUNNING", current_phase="recon", current_agent="orchestrator")
            db.add(run)
            db.commit()

            workflow_runner.start_run(run.id, ch.id, "127.0.0.1")
            workflow_runner.activate_kill_switch(run.id)

            self.assertTrue(workflow_runner.is_cancelled(run.id))
        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
