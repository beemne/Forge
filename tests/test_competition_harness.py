import unittest
import asyncio
import os
import sys
import tempfile

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db, SessionLocal
from backend.database.models import ChallengeModel, TargetProfileModel, RunModel, CheckpointModel, EvidenceModel, FindingModel
from backend.tools.manager import tool_manager
from backend.privilege.manager import privilege_manager
from backend.providers.router import model_router
from backend.agents.orchestrator_loop import orchestrator_loop
from backend.api.runner import workflow_runner
from backend.agent_runtime import (
    AgentRuntime, RealToolExecutor, Action, ActionType, ExecResult, session_manager,
    AnswerResolver, AnswerCandidate, AnswerSource, AnswerStatus, AnswerType as VerifierAnswerType,
    VerifierAgent,
)
from backend.agent_runtime.decision import DecisionEngine
from backend.agent_runtime.observation import ObservationEngine
from backend.agent_runtime.state import MissionState
from backend.agent_runtime.context import ContextBuilder
from backend.execution.interactive import interactive_manager
from backend.execution.process_manager import process_manager
from backend.swarm.evidence import ProvenanceType, Evidence, EvidenceType
from backend.agents.stream_condenser import StreamCondenser
from tests.fixtures.web_target import LocalCTFServer
from tests.fixtures.forensics_fixture import create_forensics_fixture

PY = sys.executable or "python"

async def _auto_approve_gate(cmd, **kwargs):
    """Test double standing in for the operator: approve every gated command.

    RealToolExecutor's production default is the shared require_approval() gate. Tests
    here exercise generated-script validation / real subprocess plumbing, not approval
    policy, so they inject an operator who approves unconditionally — the deny path has
    its own dedicated coverage in tests/test_privilege_gate.py.
    """
    return True, "approve", None

FAKE_DIALOGUE_CHILD = """\
import sys

sys.stdout.write("PROMPT_STAGE_0\\nEnter key 1: ")
sys.stdout.flush()

line1 = sys.stdin.readline().strip()
if line1 == "KEY_ALPHA":
    sys.stdout.write("STAGE_1_ACCEPTED\\nEnter key 2: ")
    sys.stdout.flush()
    line2 = sys.stdin.readline().strip()
    if line2 == "KEY_BETA":
        sys.stdout.write("STAGE_2_ACCEPTED\\nFINAL_SECRET_TOKEN_5544\\n")
        sys.stdout.flush()
    else:
        sys.stdout.write("STAGE_2_REJECTED\\n")
        sys.stdout.flush()
else:
    sys.stdout.write("STAGE_1_REJECTED\\n")
    sys.stdout.flush()
"""

class TestCompetitionHarness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.server = LocalCTFServer(port=8888)
        cls.server.start()
        # Generate the forensics fixture in a throwaway dir so the committed
        # tests/fixtures/evidence_sample.zip is never rewritten (keeps the tree clean).
        cls._fixture_dir = tempfile.mkdtemp(prefix="forge_forensics_")
        cls.forensics_path = create_forensics_fixture(
            output_path=os.path.join(cls._fixture_dir, "evidence_sample.zip"))

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        import shutil
        shutil.rmtree(getattr(cls, "_fixture_dir", ""), ignore_errors=True)

    def test_real_tool_manager_execution(self):
        """Tool Manager Validation: Execute real CLI tools on host system."""
        async def run_curl():
            return await tool_manager.execute_capability("web_testing", "http://127.0.0.1:8888/")

        res = asyncio.run(run_curl())
        self.assertEqual(res.status, "SUCCESS")
        self.assertIn("FORGE Web Target", res.stdout)
        self.assertEqual(res.exit_code, 0)
        self.assertGreater(res.duration_ms, 0)

    def test_security_validation_denied(self):
        """Security Validation: Verify arbitrary command injection & invalid tool requests are blocked."""
        db = SessionLocal()
        try:
            approved = privilege_manager.evaluate_privilege(
                agent="attacker_llm",
                tool_name="rm -rf /",
                privilege_level="DANGEROUS",
                db=db
            )
            self.assertFalse(approved)
        finally:
            db.close()

    def test_forensics_strings_tool_execution(self):
        """Forensics Execution: Run strings CLI tool on forensic sample."""
        async def run_strings():
            return await tool_manager.execute_capability("file_analysis", self.forensics_path)

        res = asyncio.run(run_strings())
        self.assertIn(res.status, ["SUCCESS", "MISSING_TOOL"])
        if res.status == "SUCCESS":
            self.assertIn("FLAG{forge_forensics_strings_found_9999}", res.stdout)

    def test_checkpoint_chaos_and_resume(self):
        """Checkpoint Chaos Testing: Simulate app restart, DB reload, and run resumption."""
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Chaos Test Challenge", category="web")
            db.add(ch)
            db.commit()

            target = TargetProfileModel(challenge_id=ch.id, current_address="127.0.0.1:8888")
            db.add(target)
            db.commit()

            run = RunModel(challenge_id=ch.id, status="RUNNING", current_phase="recon", current_agent="orchestrator")
            db.add(run)
            db.commit()

            workflow_runner.start_run(run.id, ch.id, "127.0.0.1:8888")

            # Execute step via orchestrator_loop
            async def step1():
                return await orchestrator_loop.execute_run_step(db, run.id)

            res1 = asyncio.run(step1())
            self.assertEqual(res1["status"], "RUNNING")

            # Reload run & checkpoint from DB
            db_fresh = SessionLocal()
            run_reloaded = db_fresh.query(RunModel).filter(RunModel.id == run.id).first()
            cp = db_fresh.query(CheckpointModel).filter(CheckpointModel.run_id == run.id).first()

            self.assertIsNotNone(cp)
            self.assertEqual(run_reloaded.current_phase, "web")
            db_fresh.close()
        finally:
            db.close()

    def test_kill_switch_process_interruption(self):
        """Kill Switch Validation: Immediate halt of autonomous loop."""
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Kill Switch E2E Test", category="recon")
            db.add(ch)
            db.commit()

            run = RunModel(challenge_id=ch.id, status="RUNNING", current_phase="recon", current_agent="recon")
            db.add(run)
            db.commit()

            workflow_runner.start_run(run.id, ch.id, "127.0.0.1:8888")
            workflow_runner.activate_kill_switch(run.id)

            self.assertTrue(workflow_runner.is_kill_switch_active(run.id))
            self.assertTrue(workflow_runner.is_cancelled(run.id))

            async def step():
                return await orchestrator_loop.execute_run_step(db, run.id)

            res = asyncio.run(step())
            self.assertEqual(res["status"], "CANCELLED")
        finally:
            db.close()

    # ── Requirement 2: Persistent Interactive Execution Lifecycle ───────────────
    def test_persistent_interaction_lifecycle_e2e(self):
        """Verify complete interactive cycle: open -> read -> send -> read -> send -> read -> close."""
        async def scenario():
            temp_dir = tempfile.TemporaryDirectory()
            script_path = os.path.join(temp_dir.name, "fake_interactive.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(FAKE_DIALOGUE_CHILD)

            try:
                # 1. interactive_open creates a session and returns initial prompt
                open_res = await tool_manager.execute_raw_command(f'interactive_open "{PY}" "{script_path}"')
                self.assertEqual(open_res.status, "SUCCESS")
                self.assertIn("[SESSION:", open_res.stdout)
                self.assertIn("PROMPT_STAGE_0", open_res.stdout)
                self.assertIn("Enter key 1:", open_res.stdout)

                # 2. Session ID is preserved
                sess_key = open_res.stdout.split("[SESSION:")[1].split("]")[0].strip()
                sess = interactive_manager.get(sess_key)
                self.assertIsNotNone(sess)
                self.assertTrue(sess.is_alive())
                self.assertIn(sess.pid, process_manager.active_pids())

                # 3. interactive_send sends data to THE SAME session
                send1 = await tool_manager.execute_raw_command(f"interactive_send {sess_key} KEY_ALPHA")
                self.assertEqual(send1.status, "SUCCESS")

                # 4. interactive_read receives the response
                read1 = await tool_manager.execute_raw_command(f"interactive_read {sess_key}")
                self.assertEqual(read1.status, "SUCCESS")
                self.assertIn("STAGE_1_ACCEPTED", read1.stdout)
                self.assertIn("Enter key 2:", read1.stdout)

                # 5. Another send/read on the same session
                send2 = await tool_manager.execute_raw_command(f"interactive_send {sess_key} KEY_BETA")
                self.assertEqual(send2.status, "SUCCESS")
                read2 = await tool_manager.execute_raw_command(f"interactive_read {sess_key}")
                self.assertEqual(read2.status, "SUCCESS")
                self.assertIn("STAGE_2_ACCEPTED", read2.stdout)
                self.assertIn("FINAL_SECRET_TOKEN_5544", read2.stdout)

                # 6. interactive_close terminates/cleans up the session
                pid = sess.pid
                close_res = await tool_manager.execute_raw_command(f"interactive_close {sess_key}")
                self.assertEqual(close_res.status, "SUCCESS")
                self.assertIsNone(interactive_manager.get(sess_key))

                # 7. No orphan process remains
                self.assertNotIn(pid, process_manager.active_pids())
            finally:
                await interactive_manager.close_all(reason="test_teardown")
                temp_dir.cleanup()

        asyncio.run(scenario())

    # ── Requirement 3: Autonomous Interaction Pivot Decision ────────────────────
    def test_autonomous_interaction_pivot_decision(self):
        """Verify DecisionEngine parses interactive commands without treating them as prose."""
        parser = DecisionEngine(gateway=None)
        raw_output = (
            "The service presents an interactive dialogue prompt.\n"
            "interactive_open python challenge.py"
        )
        decision, action, malformed = parser.parse(raw_output)
        self.assertFalse(malformed)
        self.assertEqual(action.type, ActionType.COMMAND)
        self.assertEqual(action.command, "interactive_open python challenge.py")

        raw_send = "interactive_send_and_read sess-42 user_input"
        decision, action, malformed = parser.parse(raw_send)
        self.assertFalse(malformed)
        self.assertEqual(action.type, ActionType.COMMAND)
        self.assertEqual(action.command, "interactive_send_and_read sess-42 user_input")

    # ── Requirement 4: Remote vs Local Artifact Provenance ──────────────────────
    def test_artifact_provenance_distinction(self):
        """Verify FORGE distinguishes LOCAL_FILE vs REMOTE_FILE vs SOURCE_CODE_REFERENCE."""
        obs_engine = ObservationEngine()

        # Direct file saved to local filesystem
        obs_local = obs_engine.observe(ExecResult(status="SUCCESS", stdout="Saved to flag_downloaded.txt"))
        self.assertIn("flag_downloaded.txt", obs_local.new_files)
        self.assertEqual(obs_local.file_provenance.get("flag_downloaded.txt"), "LOCAL_FILE")

        # Remote source code referencing a file path
        remote_source = 'def get_flag():\n    return open("remote_secret.txt", "r").read()'
        obs_remote = obs_engine.observe(ExecResult(status="SUCCESS", stdout=remote_source))
        self.assertIn("remote_secret.txt", obs_remote.new_files)
        self.assertEqual(obs_remote.file_provenance.get("remote_secret.txt"), "SOURCE_CODE_REFERENCE")

        # State and context block rendering
        state = MissionState()
        state.apply_observation(obs_local)
        state.apply_observation(obs_remote)

        ctx_builder = ContextBuilder()
        state_text = ctx_builder._state_block(state)
        self.assertIn("Local files (workspace): flag_downloaded.txt", state_text)
        self.assertIn("Remote / source-referenced files (NOT in local workspace): remote_secret.txt", state_text)

        # Evidence enum validation
        self.assertEqual(ProvenanceType.LOCAL_FILE.value, "LOCAL_FILE")
        self.assertEqual(ProvenanceType.REMOTE_FILE.value, "REMOTE_FILE")
        self.assertEqual(ProvenanceType.SOURCE_CODE_REFERENCE.value, "SOURCE_CODE_REFERENCE")
        self.assertEqual(ProvenanceType.REMOTE_PROCESS_STATE.value, "REMOTE_PROCESS_STATE")
        self.assertEqual(ProvenanceType.OBSERVED_OUTPUT.value, "OBSERVED_OUTPUT")

    # ── Requirement 5: Generated-Script Syntax Validation ──────────────────────
    def test_generated_script_syntax_validation(self):
        """Verify Python solver scripts are validated for syntax before execution."""
        async def scenario():
            executor = RealToolExecutor(tool_manager=tool_manager,
                                        approval_gate=_auto_approve_gate)

            # Run in a throwaway workspace so the generated solve.py is not written
            # into the repo root (keeps the working tree clean / zip-ready).
            with tempfile.TemporaryDirectory() as tmp:
                # Valid script
                valid_act = Action(type=ActionType.PYTHON_SCRIPT, script="a = 10\nb = 20\nprint(f'RES={a+b}')\n")
                res_valid = await executor.execute(valid_act, cwd=tmp)
                self.assertEqual(res_valid.status, "SUCCESS")
                self.assertIn("RES=30", res_valid.stdout)

                # Invalid syntax script
                invalid_act = Action(type=ActionType.PYTHON_SCRIPT, script="def broken_func(\n")
                res_invalid = await executor.execute(invalid_act, cwd=tmp)
                self.assertEqual(res_invalid.status, "FAILED")
                self.assertTrue(res_invalid.execution_failure)
                self.assertEqual(res_invalid.failure_category, "SYNTAX_ERROR")
                self.assertIn("SyntaxError in generated Python script", res_invalid.stderr)

        asyncio.run(scenario())

    # ── Requirement 6: Non-flag Answer Verification ────────────────────────────
    def test_non_flag_answer_verification_distinction(self):
        """Verify non-flag format answers (plain tokens/strings) are verified from evidence, not hallucinated."""
        async def scenario():
            resolver = AnswerResolver()
            verifier = VerifierAgent(resolver=resolver)

            # Candidate from observed tool output
            observed_candidate = AnswerCandidate(
                value="PLAIN_TOKEN_SECRET_123",
                answer_type=VerifierAnswerType.STRING,
                source=AnswerSource.TOOL_OUTPUT,
                confidence=0.9,
                evidence={"tool_output": "The secret key is PLAIN_TOKEN_SECRET_123"},
                task_context={"description": "Find the secret key token in output", "category": "crypto"}
            )
            resolved = resolver.resolve(observed_candidate)
            self.assertTrue(resolved.is_resolved)
            self.assertEqual(resolved.status, AnswerStatus.RESOLVED)

            verdict = await verifier.verify(resolved, authoritative=True)
            self.assertEqual(verdict.status, AnswerStatus.VERIFIED)

            # Candidate from LLM prose with no tool evidence should NOT be automatically verified
            hallucinated_candidate = AnswerCandidate(
                value="INVENTED_TOKEN_999",
                answer_type=VerifierAnswerType.STRING,
                source=AnswerSource.LLM_PROSE,
                confidence=0.3,
                evidence={},
                task_context={"description": "Find the secret key", "category": "crypto"}
            )
            resolved_hallucinated = resolver.resolve(hallucinated_candidate)
            self.assertFalse(resolved_hallucinated.is_resolved)
            self.assertNotEqual(resolved_hallucinated.status, AnswerStatus.VERIFIED)

            verdict_hallucinated = await verifier.verify(resolved_hallucinated)
            self.assertNotEqual(verdict_hallucinated.status, AnswerStatus.VERIFIED)

        asyncio.run(scenario())

    # ── Requirement 7: Structured Terminal Output Handling ─────────────────────
    def test_structured_visual_terminal_output_preservation(self):
        """Verify structured ASCII art / grid layouts preserve indentation and spatial layout."""
        ascii_grid = (
            "  ###   #####  \n"
            " #   #  #    # \n"
            " #####  #####  \n"
            " #   #  #      \n"
        )
        condensed = StreamCondenser.condense_output("custom_tool", ascii_grid, max_lines=25)
        # Verify leading whitespace and column alignment are not flattened
        self.assertIn("  ###   #####", condensed)
        self.assertIn(" #   #  #", condensed)
        self.assertIn(" #####  #####", condensed)

if __name__ == "__main__":
    unittest.main()
