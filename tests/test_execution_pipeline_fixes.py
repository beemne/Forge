"""
Regression tests for the autonomous-execution-pipeline fixes.

Each test maps to a concrete failure observed in a live run and to a specific,
general-purpose fix. Everything runs against the isolated unit-test database and uses
LOCAL test doubles (no API key, no network); the direct RealToolExecutor cases run a
real Python interpreter in a throwaway temp workspace, exactly like the competition
harness. No challenge-specific value, URL, filename, or flag is hardcoded here — the
tests exercise the GENERIC mechanisms.

Coverage:
  * Task 2 — missing local artifact dependency (pre-execution consistency check)
  * Task 3 — canonical target / relative-URL resolution
  * Task 4 — execution failure becomes structured reasoning evidence
  * Task 5 — speculative-retry thrashing is recognised (no second repetition system)
  * Task 6 — success claims are evidence-based, not prose-based
  * Task 7 — generated source code is never promoted as an answer
  * Task 8 — generated-Python quality (syntax / py2-isms) without over-reaching
"""

import os
import tempfile
import unittest

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db, SessionLocal
from backend.database.models import AgentSessionModel, TrajectoryEventModel

from backend.agent_runtime import (
    AgentRuntime, ActionType, ExecResult, RepetitionDetector, RepetitionKind,
    RecoveryEngine, RecoveryStrategy, FailureCategory, session_manager, trajectory_store,
    AnswerResolver, AnswerSource, AnswerStatus, VerifierAgent,
)
from backend.agent_runtime.action import Action
from backend.agent_runtime.runtime import (
    RealToolExecutor, resolve_target_url, analyze_python_script,
)
from backend.agent_runtime.decision import ProviderCompletion
from backend.tools.manager import classify_tool_execution, LOCAL_EXEC_CATEGORIES

# --------------------------------------------------------------------------- #
# Local test doubles (scripted provider + scripted executor)
# --------------------------------------------------------------------------- #

class ScriptedProvider:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
        self.prompts = []

    async def complete(self, *, prompt, system_instruction="", capability="general_reasoning",
                       urgency="normal", reasoning_depth="fast"):
        self.prompts.append(prompt)
        idx = self.calls
        self.calls += 1
        content = self.responses[idx] if idx < len(self.responses) else (self.responses[-1] if self.responses else None)
        if content is None:
            return ProviderCompletion(is_refusal=True, refusal_reason="exhausted")
        return ProviderCompletion(content=content, provider_name="stub", model_name="stub-m",
                                  prompt_tokens=10, completion_tokens=5)

class ScriptedToolExecutor:
    def __init__(self, sequence=None, default=None):
        self.sequence = list(sequence or [])
        self.default = default or ExecResult(status="SUCCESS", stdout="", exit_code=0)
        self.executed = []

    async def execute(self, action, *, cwd=None, timeout_seconds=120, canonical_target=None):
        self.executed.append(action.display())
        return self.sequence.pop(0) if self.sequence else self.default

def _py(script: str) -> str:
    """Wrap a script body in the model output contract (```python fenced block)."""
    return f"Here is my solver:\n```python\n{script}\n```"

class PipelineFixBase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        db = SessionLocal()
        try:
            db.query(TrajectoryEventModel).delete()
            db.query(AgentSessionModel).delete()
            db.commit()
        finally:
            db.close()

    def _session(self, **kw):
        d = dict(challenge_id="chal-fix", run_id=None, target_scope="http://target.ctf:8080/app",
                 challenge_name="Fix Test", category="web", difficulty="EASY", description="Find the flag.")
        d.update(kw)
        return session_manager.create(**d)

# =========================================================================== #
# Task 2 — missing local artifact dependency (pre-execution consistency check)
# =========================================================================== #

async def _auto_approve_gate(cmd, **kwargs):
    """Test double standing in for the operator: approve every gated command.

    RealToolExecutor's production default is the shared require_approval() gate. These
    tests exercise real subprocess plumbing (a real interpreter in a temp workspace),
    so they inject an operator who approves unconditionally — the deny path has its own
    dedicated coverage in tests/test_privilege_gate.py.
    """
    return True, "approve", None

class TestLocalArtifactDependency(PipelineFixBase):

    async def test_missing_local_dependency_is_structured_failure(self):
        ex = RealToolExecutor(approval_gate=_auto_approve_gate)
        with tempfile.TemporaryDirectory() as d:
            act = Action(type=ActionType.PYTHON_SCRIPT, script="data = open('artifact.bin', 'rb').read()\nprint(len(data))\n")
            res = await ex.execute(act, cwd=d, timeout_seconds=30)
        self.assertEqual(res.status, "FAILED")
        self.assertTrue(res.execution_failure)
        self.assertEqual(res.failure_category, "FILE_NOT_FOUND")
        self.assertIn("artifact.bin", res.stderr)

    async def test_existing_local_dependency_is_allowed(self):
        ex = RealToolExecutor(approval_gate=_auto_approve_gate)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "artifact.bin"), "wb") as f:
                f.write(b"hello")
            act = Action(type=ActionType.PYTHON_SCRIPT, script="print(len(open('artifact.bin','rb').read()))\n")
            res = await ex.execute(act, cwd=d, timeout_seconds=30)
        self.assertEqual(res.status, "SUCCESS")
        self.assertIn("5", res.stdout)

    async def test_dependency_created_by_previous_action_in_same_script_is_allowed(self):
        # A file the script itself writes before reading must NOT be flagged as missing.
        ex = RealToolExecutor(approval_gate=_auto_approve_gate)
        with tempfile.TemporaryDirectory() as d:
            act = Action(type=ActionType.PYTHON_SCRIPT,
                         script="open('made.txt','w').write('x')\nprint(open('made.txt').read())\n")
            res = await ex.execute(act, cwd=d, timeout_seconds=30)
        self.assertEqual(res.status, "SUCCESS")
        self.assertIn("x", res.stdout)

    async def test_dependency_created_by_earlier_mission_action_is_allowed(self):
        # Simulate an earlier action having produced the file (it exists on disk).
        ex = RealToolExecutor(approval_gate=_auto_approve_gate)
        with tempfile.TemporaryDirectory() as d:
            # earlier action: write the artifact
            await ex.execute(Action(type=ActionType.PYTHON_SCRIPT,
                                    script="open('prev.bin','wb').write(b'abc')\n"), cwd=d, timeout_seconds=30)
            # later action: read it — must be allowed
            res = await ex.execute(Action(type=ActionType.PYTHON_SCRIPT,
                                          script="print(open('prev.bin','rb').read())\n"), cwd=d, timeout_seconds=30)
        self.assertEqual(res.status, "SUCCESS")

    async def test_normal_script_with_no_local_dependency_runs(self):
        ex = RealToolExecutor(approval_gate=_auto_approve_gate)
        with tempfile.TemporaryDirectory() as d:
            act = Action(type=ActionType.PYTHON_SCRIPT, script="x = sum(range(5))\nprint(f'RES={x}')\n")
            res = await ex.execute(act, cwd=d, timeout_seconds=30)
        self.assertEqual(res.status, "SUCCESS")
        self.assertIn("RES=10", res.stdout)

    def test_absolute_and_system_paths_are_not_flagged(self):
        # /etc/passwd, ~/x, C:\ are target-/host-side, never "missing workspace artifacts".
        for p in ("/etc/passwd", "~/secret", "/flag.txt"):
            issue = analyze_python_script(f"open({p!r},'rb').read()\n", cwd=None)
            self.assertIsNone(issue, p)

# =========================================================================== #
# Task 3 — canonical target / relative-URL consistency
# =========================================================================== #

class TestCanonicalTargetResolution(unittest.TestCase):

    def test_absolute_target_plus_relative_path(self):
        self.assertEqual(resolve_target_url("upload.php", "http://target.ctf:8080/"),
                         "http://target.ctf:8080/upload.php")

    def test_absolute_target_plus_absolute_url_passthrough(self):
        self.assertEqual(resolve_target_url("http://other.host/x", "http://target.ctf:8080/"),
                         "http://other.host/x")

    def test_non_default_port_preserved(self):
        self.assertEqual(resolve_target_url("/login", "http://target.ctf:31337/"),
                         "http://target.ctf:31337/login")

    def test_target_with_path_prefix_preserved(self):
        self.assertEqual(resolve_target_url("upload.php", "http://target.ctf:8080/app"),
                         "http://target.ctf:8080/app/upload.php")

    def test_no_target_returns_reference_unchanged(self):
        self.assertEqual(resolve_target_url("upload.php", ""), "upload.php")
        self.assertEqual(resolve_target_url("upload.php", None), "upload.php")

    def test_scheme_less_base_is_upgraded(self):
        self.assertEqual(resolve_target_url("/x", "127.0.0.1:8888"), "http://127.0.0.1:8888/x")

class TestSchemelessUrlInScript(PipelineFixBase):

    async def test_scheme_less_request_is_pre_exec_failure_with_resolved_url(self):
        ex = RealToolExecutor()
        with tempfile.TemporaryDirectory() as d:
            act = Action(type=ActionType.PYTHON_SCRIPT, script="import requests\nrequests.post('upload.php', data={'a': 1})\n")
            res = await ex.execute(act, cwd=d, timeout_seconds=30, canonical_target="http://target.ctf:8080/")
        self.assertEqual(res.status, "FAILED")
        self.assertEqual(res.failure_category, "INVALID_URL")
        # The RESOLVED absolute URL is surfaced so the model can self-correct.
        self.assertIn("http://target.ctf:8080/upload.php", res.stderr)

    async def test_absolute_url_request_is_not_flagged(self):
        ex = RealToolExecutor()
        with tempfile.TemporaryDirectory() as d:
            # Not executed for real (no server); we only assert it is not pre-blocked.
            issue = analyze_python_script("import requests\nrequests.get('http://target.ctf:8080/')\n",
                                          cwd=d, canonical_target="http://target.ctf:8080/")
        self.assertIsNone(issue)

    def test_dict_get_is_not_mistaken_for_http_call(self):
        self.assertIsNone(analyze_python_script("d={'a':1}\nprint(d.get('a'))\n", cwd=None))

# =========================================================================== #
# Task 4 — execution failure becomes structured reasoning evidence
# =========================================================================== #

class TestFailureBecomesEvidence(PipelineFixBase):

    def test_recovery_classifies_local_execution_failures(self):
        eng = RecoveryEngine()
        for cat, expect in (("FILE_NOT_FOUND", FailureCategory.FILE_NOT_FOUND),
                            ("INVALID_URL", FailureCategory.INVALID_URL),
                            ("INTERPRETER_ASSUMPTION", FailureCategory.INTERPRETER_ASSUMPTION)):
            r = ExecResult(status="FAILED", exit_code=-1, execution_failure=True, failure_category=cat, stderr="x")
            plan = RecoveryEngine().diagnose(exec_result=r, command="python solve.py")
            self.assertEqual(plan.category, expect)
            self.assertTrue(plan.directive)
            self.assertEqual(plan.strategy, RecoveryStrategy.RETRY_MODIFIED)

    def test_invalid_url_is_classified_from_runtime_stderr(self):
        # A scheme-less request that slips past the pre-check still gets a structured category.
        r = classify_tool_execution("python", 1, "", "requests.exceptions.MissingSchema: Invalid URL 'x': No scheme supplied")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "INVALID_URL")
        self.assertIn("INVALID_URL", LOCAL_EXEC_CATEGORIES)

    async def test_failed_action_changes_next_decision_context(self):
        # Turn 1 emits a script that references a missing file → structured FILE_NOT_FOUND.
        # Turn 2's PROMPT must carry the actionable recovery directive (not just raw logs).
        provider = ScriptedProvider([_py("open('needed.bin','rb').read()\n"), "BUDGET_EXHAUSTED: stop"])
        executor = ScriptedToolExecutor(sequence=[
            ExecResult(command="python solve.py", status="FAILED", exit_code=-1,
                       execution_failure=True, failure_category="FILE_NOT_FOUND",
                       stderr="FileNotFoundError: [Errno 2] No such file or directory: 'needed.bin'"),
        ])
        sess = self._session()
        await AgentRuntime(tool_executor=executor, provider_gateway=provider).run(sess, max_turns=4)

        # (a) The second prompt the model saw contains the structured recovery directive.
        self.assertGreaterEqual(len(provider.prompts), 2)
        self.assertIn("does not exist in the workspace", provider.prompts[1])

        # (b) The failure was recorded as a structured RECOVERY/REPLAN trajectory event.
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any(e.event_type in ("RECOVERY", "REPLAN") and "FILE_NOT_FOUND" in (e.result or "")
                            for e in events))

# =========================================================================== #
# Task 5 — speculative-retry thrashing recognised (one repetition system)
# =========================================================================== #

class TestRetryThrashing(PipelineFixBase):

    def test_exact_repeated_failure_is_detected(self):
        d = RepetitionDetector()
        d.observe("curl http://t/a", failed=True, failure_category="FILE_NOT_FOUND", novel=False)
        self.assertEqual(d.classify("curl http://t/a"), RepetitionKind.EXACT_REPEAT)

    def test_equivalent_failed_actions_force_replan(self):
        # Differently-worded but semantically identical HTTP actions that fail the same way.
        d = RepetitionDetector()
        d.observe("curl http://t/a", failed=True, failure_category="FILE_NOT_FOUND", novel=False)
        rep = d.observe("curl -s http://t/a", failed=True, failure_category="FILE_NOT_FOUND", novel=False)
        self.assertEqual(rep.kind, RepetitionKind.SAME_FAILURE)
        self.assertTrue(rep.force_replan)

    def test_materially_different_action_is_not_a_repeat(self):
        # A different tool is fully novel; a different path on the same host is a SOFT
        # signal (SAME_TARGET_SAME_METHOD) — crucially, neither is the hard-blocked
        # EXACT_REPEAT, so legitimate different actions are never pre-blocked.
        d = RepetitionDetector()
        d.observe("curl http://t/a", failed=True, failure_category="FILE_NOT_FOUND", novel=False)
        self.assertNotEqual(d.classify("curl http://t/b"), RepetitionKind.EXACT_REPEAT)  # different path
        self.assertEqual(d.classify("nmap -p- t"), RepetitionKind.NONE)                  # different tool

    def test_successful_action_after_state_change_is_not_penalised(self):
        # A novel/successful outcome resets the no-progress streak (legitimate progress).
        d = RepetitionDetector()
        d.observe("curl http://t/a", failed=True, failure_category="X", novel=False)
        d.observe("curl http://t/b", failed=True, failure_category="X", novel=False)
        rep = d.observe("curl http://t/login", failed=False, failure_category=None, novel=True)
        self.assertFalse(rep.force_replan)
        self.assertEqual(rep.no_progress_streak, 0)

    async def test_loop_replans_instead_of_re_running_identical_failing_action(self):
        provider = ScriptedProvider(["curl http://target.ctf/x"] * 6)   # model insists on the same command
        executor = ScriptedToolExecutor(default=ExecResult(command="curl http://target.ctf/x", status="FAILED",
                                                           exit_code=1, execution_failure=True,
                                                           failure_category="COMMAND_NOT_FOUND",
                                                           stderr="curl: command not found"))
        sess = self._session()
        await AgentRuntime(tool_executor=executor, provider_gateway=provider).run(sess, max_turns=6)
        # The identical failing command is executed at most once — the repeat guard stops re-execution.
        runs = [c for c in executor.executed if "curl http://target.ctf/x" in c]
        self.assertEqual(len(runs), 1)
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any(e.event_type == "REPLAN" for e in events))

# =========================================================================== #
# Task 6 — success claims are evidence-based, not prose-based
# =========================================================================== #

class TestEvidenceBasedSuccess(PipelineFixBase):

    def setUp(self):
        super().setUp()
        self.resolver = AnswerResolver()
        self.verifier = VerifierAgent(resolver=self.resolver)   # deterministic (no router)

    def test_prose_only_success_is_not_authoritative(self):
        v = self.resolver.assess("picoCTF{prose_claim_only}", source=AnswerSource.LLM_PROSE)
        self.assertEqual(v.status, AnswerStatus.CANDIDATE)
        self.assertFalse(v.is_verified)

    def test_execution_evidence_supports_success(self):
        v = self.resolver.assess("picoCTF{from_real_output}", source=AnswerSource.TOOL_OUTPUT, action_succeeded=True)
        self.assertEqual(v.status, AnswerStatus.RESOLVED)

    def test_contradictory_evidence_is_handled_conservatively(self):
        # Candidate present in output but the action itself FAILED → not promoted to RESOLVED.
        v = self.resolver.assess("picoCTF{despite_failure}", source=AnswerSource.TOOL_OUTPUT, action_succeeded=False)
        self.assertEqual(v.status, AnswerStatus.CANDIDATE)
        self.assertFalse(v.is_verified)

    async def test_prose_upload_success_with_server_error_does_not_complete(self):
        # Script prints "Upload successful!" but the server body says the opposite and there is
        # no flag anywhere — the mission must NOT report a verified/resolved answer.
        provider = ScriptedProvider([_py("print('Upload successful!')\n"), "BUDGET_EXHAUSTED: stop"])
        executor = ScriptedToolExecutor(sequence=[
            ExecResult(command="python solve.py", status="SUCCESS", exit_code=0,
                       stdout="Upload successful!\nSorry, there was an error uploading your file."),
        ])
        sess = self._session()
        result = await AgentRuntime(tool_executor=executor, provider_gateway=provider).run(sess, max_turns=4)
        self.assertNotEqual(result.status, "COMPLETED")
        self.assertIsNone(result.verified_flag)
        events = trajectory_store.get_events(sess.id)
        self.assertFalse(any(e.event_type in ("FLAG_VERIFIED", "ANSWER_RESOLVED") for e in events))

    def test_verified_requires_authoritative(self):
        resolved = self.resolver.assess("picoCTF{evidence_flag}", source=AnswerSource.TOOL_OUTPUT, action_succeeded=True)
        self.assertEqual(resolved.status, AnswerStatus.RESOLVED)
        verified = self.resolver.assess("picoCTF{evidence_flag}", source=AnswerSource.TOOL_OUTPUT,
                                        action_succeeded=True, authoritative=True)
        self.assertEqual(verified.status, AnswerStatus.VERIFIED)

# =========================================================================== #
# Task 7 — generated source code is never promoted as an answer
# =========================================================================== #

class TestNoSourceCodeAsAnswer(unittest.TestCase):

    def setUp(self):
        self.resolver = AnswerResolver()

    def test_source_code_extraction_expression_is_rejected(self):
        expr = "the flag is {resp.text[resp.text.find('picoCTF{'):resp.text.find('}')+1]}"
        cands = self.resolver.extract_candidates(expr, task_context={"description": "find the flag", "category": "web"},
                                                 source=AnswerSource.TOOL_OUTPUT)
        self.assertEqual(cands, [])
        v = self.resolver.assess("picoCTF{'):resp.text.find('}", source=AnswerSource.TOOL_OUTPUT, action_succeeded=True)
        self.assertEqual(v.status, AnswerStatus.REJECTED)

    def test_placeholder_is_rejected(self):
        v = self.resolver.assess("picoCTF{...}", source=AnswerSource.TOOL_OUTPUT, action_succeeded=True)
        self.assertEqual(v.status, AnswerStatus.REJECTED)

    def test_prose_claim_is_candidate_not_verified(self):
        v = self.resolver.assess("picoCTF{model_thinks_so}", source=AnswerSource.LLM_PROSE)
        self.assertEqual(v.status, AnswerStatus.CANDIDATE)

    def test_real_extracted_answer_resolves(self):
        v = self.resolver.assess("picoCTF{a_real_captured_flag_42}", source=AnswerSource.TOOL_OUTPUT, action_succeeded=True)
        self.assertEqual(v.status, AnswerStatus.RESOLVED)

    def test_generic_non_flag_answers_are_not_treated_as_source_code(self):
        # Hashes, usernames, filenames, numbers, and URLs must pass the source-code gate.
        for val in ("2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
                    "shadow_operator_42", "config.php", "31337", "http://target.ctf/robots.txt"):
            self.assertFalse(self.resolver.looks_like_source_code(val), val)

# =========================================================================== #
# Task 8 — generated-Python quality (conservative, reliable checks only)
# =========================================================================== #

class TestGeneratedPythonQuality(PipelineFixBase):

    async def test_syntax_error_is_caught(self):
        ex = RealToolExecutor()
        res = await ex.execute(Action(type=ActionType.PYTHON_SCRIPT, script="def broken(\n"))
        self.assertEqual(res.status, "FAILED")
        self.assertEqual(res.failure_category, "SYNTAX_ERROR")

    def test_python2_string_codec_is_flagged(self):
        for script in ("x = 'data'.encode('base64')\n", "y = b'abc'.decode('hex')\n", "z = s.encode('rot13')\n"):
            issue = analyze_python_script(script, cwd=None)
            self.assertIsNotNone(issue, script)
            self.assertEqual(issue.failure_category, "INTERPRETER_ASSUMPTION")

    def test_correct_python3_codecs_usage_is_not_flagged(self):
        # The valid py3 way (codecs.encode(data, 'base64')) must NOT be flagged.
        self.assertIsNone(analyze_python_script("import codecs\nprint(codecs.encode(b'x','base64'))\n", cwd=None))
        self.assertIsNone(analyze_python_script("import base64\nprint(base64.b64encode(b'x'))\n", cwd=None))

    def test_bytes_str_concat_is_left_to_execution_feedback(self):
        # Not reliably detectable statically → deliberately NOT pre-blocked (per Task 8).
        self.assertIsNone(analyze_python_script("v = b'a' + 'b'\n", cwd=None))

if __name__ == "__main__":
    unittest.main(verbosity=2)
