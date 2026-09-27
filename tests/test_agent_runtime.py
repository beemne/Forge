"""
Tests for the FORGE Agent Runtime (backend/agent_runtime).

Covers the Step-11 checklist plus the Step-15 acceptance criterion (crash after a
turn → resume → continue with zero information loss). Everything runs against the
isolated unit-test database and uses LOCAL test doubles for the model provider and
the tool executor, so NO API key and NO network/subprocess are required.

The doubles (`ScriptedProvider`, `ScriptedToolExecutor`) are the spec's "mock
provider" and "mock tool executor"; they are named `Scripted*` and confined to this
test module so they are unambiguously test-only infrastructure.
"""

import os
import asyncio
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
from backend.database.models import (
    AgentSessionModel, TrajectoryEventModel, ChallengeModel, RunModel,
)
from backend.agent_runtime import (
    AgentRuntime, RunResult, MissionState, ObservationEngine, ExecResult,
    FlagVerifier, FlagSource, FlagStatus, RepetitionDetector, RepetitionKind,
    RecoveryEngine, RecoveryStrategy, DecisionEngine, ActionType,
    SessionManager, session_manager, trajectory_store, trajectory_search,
)
from backend.agent_runtime.decision import ProviderCompletion

# --------------------------------------------------------------------------- #
# Local test doubles (the spec's mock provider + mock tool executor)
# --------------------------------------------------------------------------- #

class ScriptedProvider:
    """A local provider gateway driven by a list or a callable(turn_index)->str.

    A response value of ``None`` simulates a provider failure (refusal), letting us
    test provider-failure recovery and mid-mission provider switching.
    """

    def __init__(self, responses, provider_name="stub-A", model_name="stub-model"):
        self.responses = responses
        self.provider_name = provider_name
        self.model_name = model_name
        self.calls = 0
        self.prompts = []

    async def complete(self, *, prompt, system_instruction="", capability="general_reasoning",
                       urgency="normal", reasoning_depth="fast"):
        self.prompts.append(prompt)
        idx = self.calls
        self.calls += 1
        if callable(self.responses):
            content = self.responses(idx)
        elif idx < len(self.responses):
            content = self.responses[idx]
        else:
            content = self.responses[-1] if self.responses else None
        if content is None:
            return ProviderCompletion(is_refusal=True, refusal_reason="All providers exhausted (simulated).")
        return ProviderCompletion(content=content, provider_name=self.provider_name,
                                  model_name=self.model_name, prompt_tokens=12, completion_tokens=8)

class ScriptedToolExecutor:
    """Returns queued ExecResults; matches by command substring or by call order."""

    def __init__(self, sequence=None, by_substring=None, default=None):
        self.sequence = list(sequence or [])
        self.by_substring = by_substring or {}
        self.default = default or ExecResult(status="SUCCESS", stdout="", exit_code=0)
        self.executed = []

    async def execute(self, action, *, cwd=None, timeout_seconds=120, canonical_target=None):
        cmd = action.display()
        self.executed.append(cmd)
        for key, res in self.by_substring.items():
            if key in cmd:
                return res
        if self.sequence:
            return self.sequence.pop(0)
        return self.default

def _cancel_after(n):
    """A cancel_check that returns True starting on the (n+1)-th call (simulating a stop)."""
    state = {"i": 0}

    def check():
        state["i"] += 1
        return state["i"] > n
    return check

class RuntimeTestBase(unittest.IsolatedAsyncioTestCase):
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

    def _make_session(self, **kw):
        defaults = dict(challenge_id="chal-rt-1", run_id=None, target_scope="http://target.ctf:8080",
                        challenge_name="Runtime Test", category="web", difficulty="EASY",
                        description="Find the flag.")
        defaults.update(kw)
        return session_manager.create(**defaults)

    def _runtime(self, provider, executor):
        return AgentRuntime(tool_executor=executor, provider_gateway=provider)

# --------------------------------------------------------------------------- #
# 1–4: session + trajectory persistence
# --------------------------------------------------------------------------- #

class TestSessionAndTrajectory(RuntimeTestBase):

    def test_01_create_session(self):
        sess = self._make_session()
        self.assertTrue(sess.id)
        self.assertEqual(sess.status, "CREATED")
        self.assertEqual(sess.state.target, "http://target.ctf:8080")
        # SESSION_START event was persisted.
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any(e.event_type == "SESSION_START" for e in events))

    def test_02_resume_session(self):
        sess = self._make_session()
        sess.state.known_endpoints.append("http://target.ctf/admin")
        sess.state.set_phase("exploit")
        session_manager.save(sess)

        resumed = session_manager.resume(sess.id)
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.status, "RUNNING")
        self.assertIn("http://target.ctf/admin", resumed.state.known_endpoints)
        self.assertEqual(resumed.state.phase, "exploit")

    def test_03_persist_trajectory(self):
        sess = self._make_session()
        for i in range(3):
            trajectory_store.record(session_id=sess.id, event_type="COMMAND",
                                    command=f"curl http://target.ctf/p{i}", result="SUCCESS")
        events = trajectory_store.get_events(sess.id)
        cmds = [e for e in events if e.event_type == "COMMAND"]
        self.assertEqual(len(cmds), 3)
        # Sequences are monotonic increasing.
        seqs = [e.sequence for e in events]
        self.assertEqual(seqs, sorted(seqs))

    def test_04_restore_trajectory_after_fresh_process(self):
        sess = self._make_session()
        trajectory_store.record(session_id=sess.id, event_type="COMMAND", command="nmap target.ctf")
        # Simulate a brand-new process: a fresh SessionManager reading the same DB.
        fresh_mgr = SessionManager()
        restored = fresh_mgr.resume(sess.id)
        self.assertIsNotNone(restored)
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any("nmap" in (e.command or "") for e in events))

# --------------------------------------------------------------------------- #
# 5–7: command execution → observation → state update (through the real loop)
# --------------------------------------------------------------------------- #

class TestLoopCoreCycle(RuntimeTestBase):

    async def test_05_06_07_command_observation_state(self):
        provider = ScriptedProvider([
            "curl -s http://target.ctf/",     # turn 1: recon
            "BUDGET_EXHAUSTED: done",          # turn 2: stop cleanly
        ])
        executor = ScriptedToolExecutor(by_substring={
            "curl": ExecResult(command="curl -s http://target.ctf/", status="SUCCESS", exit_code=0,
                               stdout="Server: nginx\n<a href=\"http://target.ctf/login\">login</a>"),
        })
        sess = self._make_session()
        result = await self._runtime(provider, executor).run(sess, max_turns=5)

        events = trajectory_store.get_events(sess.id)
        types = [e.event_type for e in events]
        self.assertIn("COMMAND", types)       # (5) command execution event
        self.assertIn("OBSERVATION", types)   # (6) observation creation
        self.assertIn("STATE_UPDATE", types)  # (7) state update

        # The COMMAND event captured real stdout, and state absorbed the discovery.
        cmd_ev = next(e for e in events if e.event_type == "COMMAND")
        self.assertIn("nginx", cmd_ev.stdout)
        reloaded = session_manager.get(sess.id)
        self.assertTrue(any("login" in ep for ep in reloaded.state.known_endpoints))
        self.assertIn("nginx", reloaded.state.technologies)

# --------------------------------------------------------------------------- #
# 8: repetition detection (unit + in-loop)
# --------------------------------------------------------------------------- #

class TestRepetition(RuntimeTestBase):

    def test_08a_detector_distinguishes_actions(self):
        d = RepetitionDetector()
        self.assertEqual(d.classify("curl http://t/login"), RepetitionKind.NONE)
        d.observe("curl http://t/login", failed=False, failure_category=None, novel=False)
        self.assertEqual(d.classify("curl http://t/login"), RepetitionKind.EXACT_REPEAT)
        # Different method / different tool are NOT repeats.
        self.assertEqual(d.classify("curl -X POST http://t/login"), RepetitionKind.NONE)
        self.assertEqual(d.classify("python exploit_login.py"), RepetitionKind.NONE)

    async def test_08b_loop_forces_replan_on_repeat(self):
        # Model insists on the same command; runtime must NOT execute it twice.
        provider = ScriptedProvider(lambda n: "curl http://target.ctf/login")
        executor = ScriptedToolExecutor(by_substring={
            "curl": ExecResult(command="curl http://target.ctf/login", status="SUCCESS",
                               exit_code=0, stdout="200 OK login page"),
        })
        sess = self._make_session()
        await self._runtime(provider, executor).run(sess, max_turns=5)
        # Executed the identical command at most once (the repeat guard skips re-exec).
        curl_runs = [c for c in executor.executed if "curl http://target.ctf/login" in c]
        self.assertEqual(len(curl_runs), 1)
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any(e.event_type == "REPLAN" for e in events))

# --------------------------------------------------------------------------- #
# 9: failed command recovery
# --------------------------------------------------------------------------- #

class TestRecovery(RuntimeTestBase):

    def test_09a_recovery_engine_classifies(self):
        eng = RecoveryEngine()
        r = ExecResult(command="curl http://down.ctf", status="FAILED", exit_code=7,
                       stderr="curl: (7) Failed to connect: Connection refused",
                       execution_failure=True, failure_category="NETWORK")
        plan = eng.diagnose(exec_result=r, command="curl http://down.ctf")
        self.assertEqual(plan.strategy, RecoveryStrategy.WAIT_RETRY)
        self.assertTrue(plan.directive)

    async def test_09b_failed_command_records_recovery(self):
        provider = ScriptedProvider([
            "curl http://target.ctf/x",
            "nmap -p- target.ctf",
            "BUDGET_EXHAUSTED: stop",
        ])
        executor = ScriptedToolExecutor(by_substring={
            "curl": ExecResult(command="curl http://target.ctf/x", status="FAILED", exit_code=1,
                               stderr="curl: command not found", execution_failure=True,
                               failure_category="MISSING_TOOL"),
            "nmap": ExecResult(command="nmap -p- target.ctf", status="SUCCESS", exit_code=0,
                               stdout="22/tcp open ssh"),
        })
        sess = self._make_session()
        await self._runtime(provider, executor).run(sess, max_turns=6)
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any(e.event_type in ("RECOVERY", "REPLAN") for e in events))
        reloaded = session_manager.get(sess.id)
        self.assertTrue(reloaded.state.failed_techniques)  # failure was recorded, not silently dropped

# --------------------------------------------------------------------------- #
# 10 & 15: provider failure recovery + switching without session loss
# --------------------------------------------------------------------------- #

class TestProviderIndependence(RuntimeTestBase):

    async def test_10_provider_failure_recovery(self):
        # First call fails (provider refusal), second succeeds — session must survive.
        provider = ScriptedProvider([None, "cat flag.txt", "BUDGET_EXHAUSTED: x"])
        executor = ScriptedToolExecutor(by_substring={
            "cat": ExecResult(command="cat flag.txt", status="SUCCESS", exit_code=0,
                              stdout="here it is: picoCTF{prov1der_f4ilover_ok}"),
        })
        sess = self._make_session()
        result = await self._runtime(provider, executor).run(sess, max_turns=6)
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any(e.result == "SWITCH_PROVIDER" for e in events))
        # Despite the provider hiccup, the mission continued and resolved the flag.
        self.assertEqual(result.status, "COMPLETED")
        self.assertIn("picoCTF{prov1der_f4ilover_ok}", result.flag_candidates)

    async def test_15_switch_provider_without_session_loss(self):
        # Run turns with provider A (interrupted), then RESUME the SAME session with
        # provider B. State + trajectory must carry over; only the provider changes.
        provider_a = ScriptedProvider(lambda n: "curl -s http://target.ctf/", provider_name="prov-A")
        exec_a = ScriptedToolExecutor(by_substring={
            "curl": ExecResult(command="curl -s http://target.ctf/", status="SUCCESS", exit_code=0,
                               stdout="Server: Apache\n<a href=\"http://target.ctf/db\">db</a>"),
        })
        sess = self._make_session()
        await self._runtime(provider_a, exec_a).run(sess, max_turns=10, cancel_check=_cancel_after(2))
        paused = session_manager.get(sess.id)
        self.assertEqual(paused.status, "PAUSED")
        self.assertTrue(paused.state.known_endpoints)  # discoveries persisted
        seq_before = trajectory_store.count(sess.id)
        self.assertEqual(paused.provider_name, "prov-A")

        # Fresh process + a DIFFERENT provider continues the same mission.
        resumed = SessionManager().resume(sess.id)
        provider_b = ScriptedProvider(["cat /flag", "BUDGET_EXHAUSTED: x"], provider_name="prov-B")
        exec_b = ScriptedToolExecutor(by_substring={
            "cat": ExecResult(command="cat /flag", status="SUCCESS", exit_code=0,
                              stdout="FLAG{cross_provider_continuity}".replace("FLAG", "flag")),
        })
        result = await self._runtime(provider_b, exec_b).run(resumed, max_turns=6)

        self.assertEqual(result.status, "COMPLETED")
        self.assertIn("flag{cross_provider_continuity}", result.flag_candidates)
        # Trajectory continued (did not restart); provider switched to B.
        self.assertGreater(trajectory_store.count(sess.id), seq_before)
        final = session_manager.get(sess.id)
        self.assertEqual(final.provider_name, "prov-B")
        self.assertTrue(final.state.known_endpoints)  # earlier provider-A discoveries retained

# --------------------------------------------------------------------------- #
# 11 & 12: flag candidate vs verification
# --------------------------------------------------------------------------- #

class TestFlagLifecycle(RuntimeTestBase):

    def test_11a_verifier_prose_is_candidate_only(self):
        v = FlagVerifier().assess("picoCTF{model_said_so_val}", source=FlagSource.LLM_PROSE)
        self.assertEqual(v.status, FlagStatus.CANDIDATE)
        self.assertFalse(v.is_verified)

    async def test_11b_model_asserted_flag_not_auto_completed(self):
        # Model claims a flag in prose (FLAG: ...) without any supporting tool output.
        provider = ScriptedProvider([
            "FLAG: picoCTF{unproven_claim_1234}",
            "BUDGET_EXHAUSTED: giving up",
        ])
        executor = ScriptedToolExecutor(default=ExecResult(status="SUCCESS", stdout=""))
        sess = self._make_session()
        result = await self._runtime(provider, executor).run(sess, max_turns=4)
        self.assertIsNone(result.verified_flag)                    # NOT auto-verified
        self.assertIn("picoCTF{unproven_claim_1234}", result.flag_candidates)  # kept as candidate
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any(e.event_type == "FLAG_CANDIDATE" for e in events))
        self.assertFalse(any(e.event_type == "FLAG_VERIFIED" for e in events))

    async def test_12_flag_resolved_from_tool_output(self):
        provider = ScriptedProvider(["cat flag.txt"])
        executor = ScriptedToolExecutor(by_substring={
            "cat": ExecResult(command="cat flag.txt", status="SUCCESS", exit_code=0,
                              stdout="picoCTF{v3rified_from_0utput}"),
        })
        sess = self._make_session()
        result = await self._runtime(provider, executor).run(sess, max_turns=4)
        self.assertEqual(result.status, "COMPLETED")
        self.assertIn("picoCTF{v3rified_from_0utput}", result.flag_candidates)
        final = session_manager.get(sess.id)
        self.assertEqual(final.status, "COMPLETED")
        events = trajectory_store.get_events(sess.id)
        self.assertTrue(any(e.event_type == "ANSWER_RESOLVED" for e in events))

# --------------------------------------------------------------------------- #
# 13: checkpoint restore (integrates with the existing checkpoint_manager)
# --------------------------------------------------------------------------- #

class TestCheckpointRestore(RuntimeTestBase):

    def _make_challenge_and_run(self):
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Checkpoint Chal", category="web")
            db.add(ch)
            db.flush()
            run = RunModel(challenge_id=ch.id, status="RUNNING")
            db.add(run)
            db.commit()
            return ch.id, run.id
        finally:
            db.close()

    def test_13_checkpoint_and_restore(self):
        challenge_id, run_id = self._make_challenge_and_run()
        sess = self._make_session(challenge_id=challenge_id, run_id=run_id)
        sess.state.known_endpoints.append("http://target.ctf/secret")
        sess.state.set_phase("escalate")
        sess.last_sequence = 7
        session_manager.checkpoint(sess, last_action="curl /secret")

        # Restore via a fresh manager (new process) — richest snapshot wins.
        restored = SessionManager().restore(sess.id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.state.phase, "escalate")
        self.assertIn("http://target.ctf/secret", restored.state.known_endpoints)
        self.assertGreaterEqual(restored.last_sequence, 7)

        # Clean up the FK-bearing rows we created (children before parents).
        db = SessionLocal()
        try:
            from backend.database.models import CheckpointModel
            db.query(CheckpointModel).filter(CheckpointModel.run_id == run_id).delete()
            db.query(RunModel).filter(RunModel.id == run_id).delete()
            db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).delete()
            db.commit()
        finally:
            db.close()

# --------------------------------------------------------------------------- #
# 14: FTS5 local search over the trajectory
# --------------------------------------------------------------------------- #

class TestFtsSearch(RuntimeTestBase):

    def test_14_fts_recall(self):
        sess = self._make_session()
        trajectory_store.record(session_id=sess.id, event_type="COMMAND",
                                command="sqlmap -u http://target.ctf/item?id=1",
                                stdout="parameter 'id' is vulnerable to SQL injection", result="SUCCESS")
        trajectory_store.record(session_id=sess.id, event_type="COMMAND",
                                command="ffuf -w list.txt -u http://target.ctf/FUZZ",
                                stdout="admin  [Status: 200]", result="SUCCESS")
        trajectory_search.reload_index()

        hits = trajectory_search.search("SQL injection vulnerable", top_k=5)
        self.assertTrue(hits)
        self.assertTrue(any("sqlmap" in (h["command"] or "") for h in hits))

        # Filter by event_type + a term that only the ffuf row matches.
        hits2 = trajectory_search.search("admin fuzz", top_k=5)
        self.assertTrue(any("ffuf" in (h["command"] or "") for h in hits2))

# --------------------------------------------------------------------------- #
# Acceptance criterion: crash after Turn 7 → resume → continue from Turn 8
# --------------------------------------------------------------------------- #

class TestCrashResumeAcceptance(RuntimeTestBase):

    async def test_resume_after_seven_turns_no_information_lost(self):
        # 7 distinct recon turns, each discovering a NEW endpoint (novel → no replan).
        provider_1 = ScriptedProvider(lambda n: f"curl -s http://target.ctf/p{n}")

        class PerPathExecutor:
            executed = []

            async def execute(self, action, *, cwd=None, timeout_seconds=120, canonical_target=None):
                self.executed.append(action.display())
                # echo a unique endpoint per path so every turn is novel
                path = action.command.rsplit("/", 1)[-1]
                return ExecResult(command=action.command, status="SUCCESS", exit_code=0,
                                  stdout=f"<a href=\"http://target.ctf/{path}/next\">n</a>")

        sess = self._make_session()
        exec1 = PerPathExecutor()
        await self._runtime(provider_1, exec1).run(sess, max_turns=50, cancel_check=_cancel_after(7))

        crashed = session_manager.get(sess.id)
        self.assertEqual(crashed.status, "PAUSED")
        endpoints_at_crash = list(crashed.state.known_endpoints)
        seq_at_crash = crashed.last_sequence
        self.assertGreaterEqual(len(endpoints_at_crash), 5)  # real progress captured before the "crash"
        self.assertEqual(len([c for c in exec1.executed]), 7)  # exactly 7 turns ran

        # ── Fresh process resumes and finishes the mission ──
        resumed = SessionManager().resume(sess.id)
        # No information lost: prior discoveries are all present.
        for ep in endpoints_at_crash:
            self.assertIn(ep, resumed.state.known_endpoints)

        provider_2 = ScriptedProvider(["cat /flag.txt"])
        exec2 = ScriptedToolExecutor(by_substring={
            "cat": ExecResult(command="cat /flag.txt", status="SUCCESS", exit_code=0,
                              stdout="picoCTF{resumed_from_turn_8}"),
        })
        result = await self._runtime(provider_2, exec2).run(resumed, max_turns=6)

        self.assertEqual(result.status, "COMPLETED")
        self.assertIn("picoCTF{resumed_from_turn_8}", result.flag_candidates)
        # Trajectory sequences CONTINUED past the crash point (did not reset to 0).
        events = trajectory_store.get_events(sess.id)
        max_seq = max(e.sequence for e in events)
        self.assertGreater(max_seq, seq_at_crash)
        # The post-resume events are strictly after the crash sequence.
        post = [e for e in events if e.command and "cat /flag.txt" in e.command]
        self.assertTrue(post and all(e.sequence > seq_at_crash for e in post))

if __name__ == "__main__":
    unittest.main(verbosity=2)
