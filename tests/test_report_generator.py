"""
Tests for the technical CTF writeup generator (backend/reporting/generator.py).

Grounds the writeup in REAL run telemetry (ToolExecutionModel + blackboard snapshot +
distilled experience) and proves:
  * the deterministic fallback is real and technical (actual commands/endpoints/flag,
    no "No evidence collected yet", no placeholder flag),
  * save_writeup writes the .md into the target working folder,
  * the AI path routes with capability="report_generation" (Gemini first) and feeds the
    real telemetry into the prompt, falling back to the deterministic writeup on refusal.

Only the provider boundary (model_router.route_request) is mocked — no live LLM/network.
All state lives in the isolated unit-test database.
"""
import os
import asyncio
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db, SessionLocal
from backend.database.models import (
    ChallengeModel, TargetProfileModel, RunModel, ToolExecutionModel,
    ExperienceModel, ReportModel,
)
from backend.reporting.generator import report_generator
from backend.providers.base import ProviderResponse
import backend.providers.router as router_mod

CH_ID = "rpt-chal-1"
RUN_ID = "rpt-run-1"
FLAG = "picoCTF{s3t_s3ss10n_3xp1rat10n5_77b6684a}"
TARGET = "http://dolphin-cove.picoctf.net:49291"

class ReportGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self._clean()
        self.tmp = tempfile.mkdtemp(prefix="forge_rpt_")
        db = SessionLocal()
        try:
            db.add(ChallengeModel(
                id=CH_ID, name="Old Sessions", category="web", difficulty="EASY",
                description="Cookie/session handling challenge.",
                working_directory=self.tmp, platform_name="PicoCTF",
                status="SOLVED", flag=FLAG,
                mission_plan={"blackboard_state": {
                    "discovered_endpoints": ["/", "/login", "/profile"],
                    "extracted_headers": {"Server": "Werkzeug/2.0 Python/3.9", "X-Powered-By": "Flask"},
                    "observed_cookies": {"session": "eyJ1c2VyIjoiZ3Vlc3QifQ"},
                    "deobfuscated_secrets": [{"secret_key": "flask_default_key"}],
                }},
            ))
            db.add(TargetProfileModel(challenge_id=CH_ID, current_address=TARGET, hostname="dolphin-cove"))
            db.add(RunModel(id=RUN_ID, challenge_id=CH_ID, status="COMPLETED"))
            base = datetime(2026, 9, 9, 3, 0, 0)
            steps = [
                ("agent_1", f"curl -s -i {TARGET}/", "HTTP/1.1 200 OK\r\nServer: Werkzeug/2.0 Python/3.9", "SUCCESS", 0),
                ("agent_1", f"curl -s -i {TARGET}/admin", "HTTP/1.1 403 Forbidden", "FAILED", 0),
                ("agent_2", f"flask-unsign --decode --cookie 'eyJ1c2VyIjoiZ3Vlc3QifQ'", "{'user': 'guest'}", "SUCCESS", 0),
                ("agent_2", f"curl -s -i --cookie 'session=forged' {TARGET}/profile",
                 f"HTTP/1.1 200 OK\r\n\r\nWelcome admin! {FLAG}", "SUCCESS", 0),
            ]
            for i, (agent, cmd, out, status, code) in enumerate(steps):
                db.add(ToolExecutionModel(
                    run_id=RUN_ID, agent=agent, tool_name="bash", capability="swarm_" + agent,
                    command=cmd, status=status, stdout=out, stderr="", exit_code=code,
                    duration_ms=120.0 + i, created_at=base + timedelta(seconds=i),
                ))
            db.add(ExperienceModel(
                source="forge_run", source_run_id=RUN_ID, source_challenge_id=CH_ID,
                challenge_name="Old Sessions", category="web", technique="Session Forgery / Weak Secret",
                generalized_strategy="Decode the session cookie, recover/guess the signing key, forge an elevated session.",
                applicable_conditions="Signed client-side session cookie with a weak/default secret.",
                detection_indicators={"indicators": ["Session cookies signed with a default framework key"],
                                      "containment": ["Rotate the signing key; set strong SECRET_KEY"]},
                outcome="success", confidence=0.8, success_rate=1.0, times_successful=1,
            ))
            db.commit()
        finally:
            db.close()

    def tearDown(self):
        self._clean()
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _clean():
        db = SessionLocal()
        try:
            db.query(ToolExecutionModel).filter(ToolExecutionModel.run_id == RUN_ID).delete()
            db.query(ReportModel).filter(ReportModel.challenge_id == CH_ID).delete()
            db.query(ExperienceModel).filter(ExperienceModel.source_challenge_id == CH_ID).delete()
            db.query(RunModel).filter(RunModel.challenge_id == CH_ID).delete()
            db.query(TargetProfileModel).filter(TargetProfileModel.challenge_id == CH_ID).delete()
            db.query(ChallengeModel).filter(ChallengeModel.id == CH_ID).delete()
            db.commit()
        finally:
            db.close()

    # ------------------------------------------------------------------ #

    def test_gather_context_reads_real_telemetry(self):
        db = SessionLocal()
        try:
            ctx = report_generator.gather_context(db, CH_ID)
            self.assertIsNotNone(ctx)
            self.assertEqual(len(ctx["execs"]), 4, "all tool executions should be gathered")
            self.assertEqual(ctx["target_str"], TARGET)
            self.assertIsNotNone(ctx["flag_step"], "the flag-yielding command must be identified")
            self.assertIn("/profile", ctx["flag_step"].command)
            self.assertIn("/login", ctx["blackboard"]["discovered_endpoints"])
        finally:
            db.close()

    def test_deterministic_writeup_is_grounded_and_technical(self):
        db = SessionLocal()
        try:
            ctx = report_generator.gather_context(db, CH_ID)
            md = report_generator.render_deterministic(ctx)
        finally:
            db.close()
        # Real, technical content present:
        self.assertIn("Old Sessions", md)
        self.assertIn(FLAG, md)                                   # real flag included (operator's own writeup)
        self.assertIn("flask-unsign --decode", md)                # a real command from the trace
        self.assertIn("curl -s -i", md)
        self.assertIn("/profile", md)                             # the flag-yielding endpoint
        self.assertIn("Flask", md)                                # fingerprinted stack
        self.assertIn("Attack Chain", md)
        self.assertIn("Flag Extraction & Verification", md)
        self.assertIn("Session Forgery", md)                      # technique from experience
        # No hollow filler / placeholders:
        self.assertNotIn("No evidence collected yet", md)
        self.assertNotIn("FORGE{flag_captured}", md)
        self.assertNotIn("conducted targeted analysis", md)

    def test_save_writeup_writes_markdown_file(self):
        db = SessionLocal()
        try:
            ctx = report_generator.gather_context(db, CH_ID)
            md = report_generator.render_deterministic(ctx)
            path = report_generator.save_writeup(db, CH_ID, md, output_dir=self.tmp)
        finally:
            db.close()
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith(".md"))
        with open(path, encoding="utf-8") as f:
            saved = f.read()
        self.assertIn(FLAG, saved)
        self.assertIn("flask-unsign", saved)

    def test_ai_path_routes_gemini_first_with_real_telemetry(self):
        captured = {}

        async def fake_route_request(prompt, capability="general_reasoning",
                                     system_instruction=None, target_model=None, **kwargs):
            captured.update(prompt=prompt, capability=capability,
                            system_instruction=system_instruction, target_model=target_model, kwargs=kwargs)
            return ProviderResponse(provider_name="gemini", model_name="gemini-3.6-flash",
                                    content="# CTF Writeup: Old Sessions (WEB)\n\nAI body citing the exploit.")

        orig = router_mod.model_router.route_request
        router_mod.model_router.route_request = fake_route_request
        try:
            db = SessionLocal()
            try:
                content, generated_by = asyncio.run(report_generator.craft_writeup(db, CH_ID))
            finally:
                db.close()
        finally:
            router_mod.model_router.route_request = orig

        self.assertEqual(captured["capability"], "report_generation", "must use the Gemini-first task chain")
        self.assertEqual(captured["target_model"], "gemini-3.6-flash", "Gemini prioritised for this task")
        # Real telemetry reached the model, and the anti-fabrication instruction is present.
        self.assertIn("flask-unsign --decode", captured["prompt"])
        self.assertIn(FLAG, captured["prompt"])
        self.assertIn("Do NOT invent", captured["system_instruction"])
        self.assertIn("gemini", generated_by)
        self.assertIn("AI body", content)

    def test_ai_refusal_falls_back_to_deterministic(self):
        async def fake_refusal(prompt, **kwargs):
            return ProviderResponse(provider_name="none", model_name="none", content="",
                                    is_refusal=True, refusal_reason="All providers exhausted")

        orig = router_mod.model_router.route_request
        router_mod.model_router.route_request = fake_refusal
        try:
            db = SessionLocal()
            try:
                content, generated_by = asyncio.run(report_generator.craft_writeup(db, CH_ID))
            finally:
                db.close()
        finally:
            router_mod.model_router.route_request = orig

        self.assertEqual(generated_by, "deterministic-fallback")
        self.assertIn(FLAG, content)                       # grounded fallback, never empty
        self.assertIn("flask-unsign", content)
        self.assertNotIn("No evidence collected yet", content)

if __name__ == "__main__":
    unittest.main()
