"""
Tests for FORGE **Phase 2 — MEMORY + EXPERIENCE + SKILLS**.

Phase 2 unifies the Phase-1 AgentRuntime (session + durable trajectory = Level-1
episodic memory) with the existing knowledge layer (generalized experiences =
Level-2, promoted playbooks = Level-3) and makes both environment-aware. This
suite exercises that union end-to-end.

It covers Step-17's twelve required scenarios and the Step-20 acceptance loop
(Mission A learns an experience -> a *different* Mission B in the same
vulnerability family retrieves and reuses it -> the experience's confidence
rises), plus the Step-5/9 cross-session failed-approach recall.

Everything drives the REAL production objects — the real ``AgentRuntime``, the
real ``RuntimeLearner`` bridge, the real ``experience_memory`` / ``memory_retriever``
singletons and the real ``ExperienceExtractor`` — never a parallel mock pipeline.
The only test doubles are the model *provider* and the *tool executor* (the two
things that would otherwise need an API key + a live target); they are named
``Scripted*`` and confined to this module so they are unambiguously test-only
(Step 17/18: no API key, no network/subprocess).

Isolation (project rules): every write targets the isolated ``test_forge.db``;
the Playbook Vault *sink* is swapped to a temp directory so a promotion never
writes YAML into the real, gitignored vault (rules 6/7/8); tables are cleaned and
the FTS indexes resynced around every test so assertions see only what the test
stored.
"""

import os
import json
import glob
import shutil
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
from backend.database.models import (
    ExperienceModel, ExperienceAttemptModel, MemoryUsageModel,
    AgentSessionModel, TrajectoryEventModel,
)
from backend.agent_runtime import (
    AgentRuntime, MissionState, ExecResult, ContextBuilder,
    CapabilityReport, LocalExecutionBackend,
    session_manager, trajectory_store, trajectory_search,
    runtime_learner, RuntimeBoardAdapter,
)
from backend.agent_runtime.decision import ProviderCompletion
from backend.knowledge.experience_memory import experience_memory
from backend.knowledge.memory_retriever import memory_retriever
from backend.knowledge.experience_extractor import experience_extractor
from backend.knowledge.memory_models import ExperienceRecord, infer_environment_requirements
from backend.knowledge.playbook_vault import PlaybookVault

# --------------------------------------------------------------------------- #
# Local test doubles — the spec's mock provider + mock tool executor.
# (Deliberately self-contained so this suite does not depend on another test
#  module's internals.)
# --------------------------------------------------------------------------- #

class ScriptedProvider:
    """A local provider gateway driven by a list or a callable(turn_index)->str.

    A response of ``None`` simulates a provider refusal, so provider failover and
    mid-mission provider switching can be tested without any real provider.
    """

    def __init__(self, responses, provider_name="stub-A", model_name="stub-model"):
        self.responses = responses
        self.provider_name = provider_name
        self.model_name = model_name
        self.calls = 0

    async def complete(self, *, prompt, system_instruction="", capability="general_reasoning",
                       urgency="normal", reasoning_depth="fast"):
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
    """A cancel_check that returns True starting on the (n+1)-th call (a simulated stop)."""
    state = {"i": 0}

    def check():
        state["i"] += 1
        return state["i"] > n
    return check

# --------------------------------------------------------------------------- #

class Phase2TestBase(unittest.IsolatedAsyncioTestCase):
    """Shared isolation: clean DB + resynced FTS indexes + a throwaway vault sink."""

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self._clean_tables()
        # Resync the production singletons' in-memory FTS indexes to the now-empty
        # DB so retrieval/search assertions see only what THIS test stores.
        experience_memory.reload_index()
        trajectory_search.reload_index()
        # Swap the Playbook Vault sink to a temp dir so a promotion never pollutes
        # the real, gitignored vault (rules 6/7/8). Restored in tearDown.
        self.tmp_dir = tempfile.mkdtemp(prefix="forge_phase2_")
        self._tmp_vault = PlaybookVault(base_dir=os.path.join(self.tmp_dir, "pb"))
        self._orig_mem_vault = experience_memory.vault
        self._orig_ret_vault = memory_retriever.vault
        experience_memory.vault = self._tmp_vault
        memory_retriever.vault = self._tmp_vault

    def tearDown(self):
        experience_memory.vault = self._orig_mem_vault
        memory_retriever.vault = self._orig_ret_vault
        try:
            self._tmp_vault.db_conn.close()
        except Exception:
            pass
        self._clean_tables()
        experience_memory.reload_index()
        trajectory_search.reload_index()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        for f in glob.glob(os.path.join("backend", "logs", "challenge_p2-*.log")):
            try:
                os.remove(f)
            except OSError:
                pass

    @staticmethod
    def _clean_tables():
        db = SessionLocal()
        try:
            db.query(MemoryUsageModel).delete()
            db.query(ExperienceAttemptModel).delete()
            db.query(ExperienceModel).delete()
            db.query(TrajectoryEventModel).delete()
            db.query(AgentSessionModel).delete()
            db.commit()
        finally:
            db.close()

    # -- helpers -------------------------------------------------------------- #

    def _runtime(self, provider, executor, **kw):
        return AgentRuntime(tool_executor=executor, provider_gateway=provider, **kw)

    def _session(self, cid, name, target, description, category="web", difficulty="MEDIUM"):
        return session_manager.create(
            challenge_id=cid, run_id=None, target_scope=target, agent_id="orchestrator",
            engine="runtime", challenge_name=name, category=category, difficulty=difficulty,
            description=description)

    @staticmethod
    def _exp(**kw):
        """Build an ExperienceRecord with sensible defaults for seeding tests."""
        base = dict(source="forge_trajectory", category="web", difficulty="MEDIUM",
                    technique="Generic", tags=["web"], outcome="success", confidence=0.7)
        base.update(kw)
        return ExperienceRecord(**base)

    # A winning upload ExecResult whose stdout carries a real, verifiable flag.
    @staticmethod
    def _upload_hit(command, flag):
        return ExecResult(command=command, status="SUCCESS", exit_code=0,
                          stdout=f"HTTP/1.1 200 OK\nUpload successful. uploads/ executes -> {flag}")

    @staticmethod
    def _recon_hit(command, upload_url):
        return ExecResult(command=command, status="SUCCESS", exit_code=0,
                          stdout=("Server: Apache/2.4\nX-Powered-By: PHP/7.4\n"
                                  f"<a href=\"{upload_url}\">upload avatar</a>"))

# =========================================================================== #
# Steps 1-3: session/trajectory -> experience extraction (success & failure)
# =========================================================================== #

class TestTrajectoryToExperience(Phase2TestBase):

    def test_01_session_to_experience_extraction(self):
        """Step 17.1: a completed session + its trajectory distils into a stored,
        generalized experience via the RuntimeBoardAdapter -> ExperienceExtractor
        bridge (the runtime path's learning loop, which Phase 1 lacked)."""
        sess = self._session("p2-e1", "Extractor", "http://t.ctf", "PHP avatar upload portal.")
        sess.state.known_endpoints.append("http://t.ctf/upload.php")
        sess.state.headers["X-Powered-By"] = "PHP/7.4"
        # A real failed->success trace, recorded as trajectory events (evidence, not prose).
        trajectory_store.record(session_id=sess.id, event_type="COMMAND",
                                command="curl -s http://t.ctf/", stdout="Server: Apache", result="SUCCESS")
        trajectory_store.record(session_id=sess.id, event_type="COMMAND",
                                command="curl -F 'file=@a.php' http://t.ctf/upload.php",
                                stdout="400 Bad Request: .php rejected", result="FAILED")
        trajectory_store.record(session_id=sess.id, event_type="COMMAND",
                                command="curl -F 'file=@a.php5' http://t.ctf/upload.php",
                                stdout="Upload successful -> picoCTF{extractor_ok}", result="SUCCESS")
        sess.state.set_verified_flag("picoCTF{extractor_ok}")
        sess.verified_flag = "picoCTF{extractor_ok}"

        # The adapter presents the finished session as the "board" the extractor knows.
        adapter = RuntimeBoardAdapter(sess)
        self.assertEqual(len(adapter.execution_history), 3)
        self.assertEqual(adapter.flag_captured, "picoCTF{extractor_ok}")
        self.assertIn("http://t.ctf/upload.php", adapter.discovered_endpoints)

        exp_id = runtime_learner.learn_from_session(sess, outcome="success")
        self.assertTrue(exp_id)
        exp = experience_memory.get(exp_id, with_children=True)
        self.assertEqual(exp["outcome"], "success")
        # Provenance: this experience came from the runtime path, not the swarm (Step 13).
        self.assertEqual(exp["source"], "forge_trajectory")
        self.assertIn("Upload", exp["technique"])
        # Step 5: the rejected .php attempt is remembered as a failed approach.
        approaches = " ".join(f.get("approach", "") for f in exp["failed_techniques"])
        self.assertIn(".php", approaches)
        # Step 4: knowledge is grounded in evidence — the concrete flag never survives.
        self.assertNotIn("picoCTF{extractor_ok}", json.dumps(exp))

    async def test_02_successful_trajectory_learns_via_real_loop(self):
        """Step 17.2: running the REAL AgentRuntime to a verified-flag solve stores a
        success experience automatically (the runtime closes its own learning loop)."""
        provider = ScriptedProvider([
            "curl -s http://uploads.ctf/",
            "curl -F 'file=@s.php5' http://uploads.ctf/upload.php",
            "BUDGET_EXHAUSTED: done",
        ])
        executor = ScriptedToolExecutor(by_substring={
            "upload.php": self._upload_hit("curl -F 'file=@s.php5' http://uploads.ctf/upload.php",
                                           "picoCTF{loop_upl0ad}"),
            "curl -s http://uploads.ctf/": self._recon_hit("curl -s http://uploads.ctf/",
                                                           "http://uploads.ctf/upload.php"),
        })
        sess = self._session("p2-e2", "Avatar Upload", "http://uploads.ctf",
                             "A PHP avatar upload portal that filters extensions.")
        result = await self._runtime(provider, executor).run(sess, max_turns=6)

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.verified_flag, "picoCTF{loop_upl0ad}")
        exps = experience_memory.list_experiences(limit=5)
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0]["outcome"], "success")
        self.assertEqual(exps[0]["source"], "forge_trajectory")
        self.assertIn("Upload", exps[0]["technique"])

    async def test_03_failed_trajectory_becomes_negative_experience(self):
        """Step 17.3 / Step 5: a mission that ends WITHOUT a verified flag is still
        learned from — stored as a low-confidence ``failure`` experience so its
        dead-ends are recalled, not silently forgotten."""
        provider = ScriptedProvider(["curl -s http://x.ctf/a", "curl -s http://x.ctf/b"])
        executor = ScriptedToolExecutor(
            default=ExecResult(status="SUCCESS", stdout="200 OK, nothing useful here", exit_code=0))
        sess = self._session("p2-e3", "Dead End", "http://x.ctf", "A web page with no obvious flaw.")
        result = await self._runtime(provider, executor).run(sess, max_turns=2)

        self.assertEqual(result.status, "MAX_TURNS")
        self.assertIsNone(result.verified_flag)
        exps = experience_memory.list_experiences(limit=5)
        self.assertEqual(len(exps), 1)
        self.assertEqual(exps[0]["outcome"], "failure")
        self.assertEqual(exps[0]["source"], "forge_trajectory")
        self.assertLessEqual(exps[0]["confidence"], 0.5)  # a single unproven run is low-confidence

# =========================================================================== #
# Steps 4-6: retrieval, FTS5 search, ranking
# =========================================================================== #

class TestRetrievalSearchRanking(Phase2TestBase):

    def test_04_experience_retrieval(self):
        """Step 17.4: a stored experience is retrieved for evidence in its family."""
        eid = experience_memory.store(self._exp(
            technique="File Upload Validation Bypass", tags=["upload", "file_upload_bypass", "web"],
            technologies=["php", "apache"],
            generalized_strategy="Bypass the upload extension filter with .php5/.phtml, then reach the file URL.",
            confidence=0.78))
        experience_memory.reload_index()

        text, mems = memory_retriever.retrieve_and_format(
            evidence="PHP file upload portal; multipart/form-data; filename=avatar",
            category="web", technologies=["php"], query="upload validation bypass", top_k=5)
        self.assertTrue(mems)
        self.assertIn(eid, [m.id for m in mems])
        self.assertIn("upload", text.lower())

    def test_05_fts5_search_experiences_and_trajectory(self):
        """Step 17.5: local SQLite FTS5 search over BOTH experiences and the
        trajectory returns only relevant records."""
        experience_memory.store(self._exp(
            category="crypto", technique="Cryptographic Weakness Exploitation", tags=["crypto"],
            generalized_strategy="Recover plaintext from an RSA weak-modulus factorization."))
        experience_memory.store(self._exp(
            category="web", technique="SQL Injection", tags=["sqli", "web"],
            generalized_strategy="UNION-based extraction after a boolean fingerprint."))
        experience_memory.reload_index()

        hits = experience_memory.search(query="RSA weak modulus", category="crypto", top_k=5)
        self.assertTrue(hits)
        self.assertTrue(all(h["category"] == "crypto" for h in hits))
        self.assertTrue(any("Crypto" in h["technique"] for h in hits))

        # Trajectory FTS (record() auto-indexes into the in-memory FTS5 store).
        sess = self._session("p2-fts", "FTS", "http://t.ctf", "search")
        trajectory_store.record(session_id=sess.id, event_type="COMMAND",
                                command="sqlmap -u http://t.ctf/item?id=1",
                                stdout="parameter 'id' is vulnerable to SQL injection", result="SUCCESS")
        trajectory_store.record(session_id=sess.id, event_type="COMMAND",
                                command="ffuf -w list.txt -u http://t.ctf/FUZZ",
                                stdout="admin  [Status: 200]", result="SUCCESS")
        thits = trajectory_search.search("SQL injection vulnerable", top_k=5)
        self.assertTrue(any("sqlmap" in (h["command"] or "") for h in thits))

    def test_06_ranking_prefers_same_class_verified_experience(self):
        """Step 17.6 / Step 8: a verified experience in the SAME vulnerability class
        outranks an unrelated one for class-specific evidence."""
        up_id = experience_memory.store(self._exp(
            technique="File Upload Validation Bypass", tags=["upload", "web"], technologies=["php"],
            generalized_strategy="Defeat the weakest upload filter layer (.phtml), reach the file URL.",
            confidence=0.8))
        experience_memory.store(self._exp(
            source="external", category="crypto", technique="Cryptographic Weakness Exploitation",
            tags=["crypto"], generalized_strategy="Factor a weak RSA modulus.", confidence=0.8))
        experience_memory.reload_index()

        _, mems = memory_retriever.retrieve_and_format(
            evidence="php upload portal; multipart/form-data; filename=; .phtml",
            category="web", technologies=["php"], query="file upload bypass", top_k=5)
        self.assertTrue(mems)
        self.assertEqual(mems[0].id, up_id)  # same-class verified experience ranks first

# =========================================================================== #
# Steps 7-9: promotion, confidence update, staleness
# =========================================================================== #

class TestPromotionConfidenceStaleness(Phase2TestBase):

    def test_07_skill_promotion_after_repeated_success(self):
        """Step 17.7 / Step 11: an experience proven across repeated verified solves
        is promoted into a reusable Playbook (the temp vault sink)."""
        eid = experience_memory.store(self._exp(
            technique="File Upload Validation Bypass", tags=["upload", "web"], technologies=["php"],
            generalized_strategy="Bypass the upload filter, reach the uploaded file.", confidence=0.7))
        self.assertIsNone(experience_memory.get(eid)["promoted_playbook_id"], "one solve must not auto-promote")

        # Two more verified reuses -> proven -> promotion.
        experience_memory.record_feedback(eid, success=True)
        experience_memory.record_feedback(eid, success=True)

        after = experience_memory.get(eid)
        self.assertGreaterEqual(after["times_successful"], 2)
        self.assertIsNotNone(after["promoted_playbook_id"], "repeated success should promote to a playbook")
        promoted = self._tmp_vault.load_playbook(after["promoted_playbook_id"])
        self.assertIsNotNone(promoted, "the promoted playbook should exist in the (temp) vault")

    def test_08_confidence_rises_on_success_falls_on_failure(self):
        """Step 17.8 / Step 6: verified reuse raises confidence; a contradicting
        outcome lowers it. Confidence tracks evidence, not a single claim."""
        eid = experience_memory.store(self._exp(technique="X", confidence=0.6))
        c0 = experience_memory.get(eid)["confidence"]
        experience_memory.record_feedback(eid, success=True)
        c1 = experience_memory.get(eid)["confidence"]
        self.assertGreater(c1, c0)
        experience_memory.record_feedback(eid, success=False)
        c2 = experience_memory.get(eid)["confidence"]
        self.assertLess(c2, c1)

    def test_09_repeated_failure_decays_confidence(self):
        """Step 17.9 / Step 15: a technique that repeatedly fails has its confidence
        and success-rate decayed (staleness handled by re-weighting, not deletion)."""
        eid = experience_memory.store(self._exp(source="external", technique="Flaky", confidence=0.8))
        for _ in range(4):
            experience_memory.record_feedback(eid, success=False)
        row = experience_memory.get(eid)
        self.assertLess(row["confidence"], 0.6)
        self.assertLess(row["success_rate"], 0.5)
        self.assertGreaterEqual(row["times_failed"], 4)
        # Still present — knowledge is decayed, never blindly deleted (Step 15).
        self.assertIsNotNone(experience_memory.get(eid))

# =========================================================================== #
# Step 10 + 14: memory-usage telemetry
# =========================================================================== #

class TestMemoryUsageTelemetry(Phase2TestBase):

    def test_10_usage_events_are_tracked(self):
        """Step 17.10 / Step 14: retrieval and fine-grained usage ('contributed',
        'contradicted') are logged so FORGE can later learn which memories help."""
        eid = experience_memory.store(self._exp(technique="Y"))
        experience_memory.record_retrieval([eid], run_id="p2-run", challenge_id="p2-chal")
        experience_memory.record_usage_event(eid, "contributed", note="helped reach the flag",
                                             run_id="p2-run", challenge_id="p2-chal")
        experience_memory.record_usage_event(eid, "contradicted", note="output disproved it",
                                             run_id="p2-run", challenge_id="p2-chal")

        db = SessionLocal()
        try:
            events = sorted(r.event for r in
                            db.query(MemoryUsageModel).filter(MemoryUsageModel.experience_id == eid).all())
        finally:
            db.close()
        self.assertIn("retrieved", events)
        self.assertIn("contributed", events)
        self.assertIn("contradicted", events)
        self.assertGreaterEqual(experience_memory.get(eid)["times_retrieved"], 1)

# =========================================================================== #
# Steps 12 + 19: environment-aware skills (Windows core / Linux execution)
# =========================================================================== #

class TestEnvironmentAwareSkills(Phase2TestBase):

    def test_11a_environment_inference_is_conservative(self):
        """Step 12/19: Linux-only tools imply a Linux execution backend; portable
        techniques stay OS-independent so they remain runnable from the Windows host."""
        linux = infer_environment_requirements(["nmap -sV -p- 10.10.14.5", "ffuf -w l -u http://t/FUZZ"], "web")
        self.assertEqual(linux["required_os"], "linux")
        self.assertIn("nmap", linux["tools"])

        portable = infer_environment_requirements(["curl http://t/", "python solve.py"], "web")
        self.assertEqual(portable["required_os"], "any")

        pwn = infer_environment_requirements(["python exploit.py"], "pwn")
        self.assertEqual(pwn["required_os"], "linux")  # pwn is effectively Linux-bound

    def test_11b_retrieval_reweights_by_execution_capability(self):
        """Step 12/8: a Linux-only skill fits worse when the execution environment is
        Windows-without-the-tool — de-prioritised (never dropped) and annotated so the
        agent can install/adapt. The intelligence layer stays OS-independent."""
        eid = experience_memory.store(self._exp(
            technique="Network Recon + Fuzzing", tags=["recon", "web"],
            generalized_strategy="Scan services with nmap, then brute directories with ffuf.",
            commands_used=["nmap -sV target", "ffuf -u http://t/FUZZ"],
            required_os="linux", required_tools=["nmap", "ffuf"]))
        experience_memory.reload_index()

        win_caps = CapabilityReport(os="windows", available_tools=["curl", "python", "git"])
        lin_caps = CapabilityReport(os="linux", available_tools=["nmap", "ffuf", "curl", "python"])

        _, mems_win = memory_retriever.retrieve_and_format(
            evidence="nmap ffuf recon scan directories", category="web",
            query="network recon fuzzing", top_k=5, capabilities=win_caps)
        _, mems_lin = memory_retriever.retrieve_and_format(
            evidence="nmap ffuf recon scan directories", category="web",
            query="network recon fuzzing", top_k=5, capabilities=lin_caps)

        mw = next(m for m in mems_win if m.id == eid)
        ml = next(m for m in mems_lin if m.id == eid)
        self.assertEqual(ml.environment_fit, 1.0)          # fully runnable on Linux+tools
        self.assertLess(mw.environment_fit, ml.environment_fit)  # worse fit on Windows-without-nmap

        # The rendered prompt tells the agent the tooling is not runnable here.
        text_win = memory_retriever.format_for_prompt(mems_win, capabilities=win_caps)
        self.assertTrue("nmap" in text_win.lower() and "linux" in text_win.lower())

    def test_11c_local_backend_reports_a_capability_report(self):
        """Step 19: the execution backend REPORTS capabilities of whatever host it runs
        on (Windows in dev, Linux in the VM) — the core never assumes an OS."""
        caps = LocalExecutionBackend().capabilities()
        self.assertIsInstance(caps, CapabilityReport)
        self.assertIn(caps.os, ("windows", "linux", "darwin", "any"))
        # satisfies()/fit_score() are pure functions over the report — no host assumption.
        self.assertTrue(caps.satisfies({"required_os": "any"}))
        self.assertEqual(caps.fit_score({"required_os": "any", "tools": []}), 1.0)

# =========================================================================== #
# Step 12 (Step 17.12): provider switching preserves FORGE-owned memory
# =========================================================================== #

class TestProviderSwitchPreservesMemory(Phase2TestBase):

    async def test_12_switch_provider_without_losing_memory(self):
        """Step 17.12 / 'THE MODEL IS NOT THE MEMORY': a mission driven partly by
        provider A and finished by a DIFFERENT provider B keeps its session,
        trajectory, pre-existing memory AND learns a new experience — all owned by
        FORGE, not by any provider."""
        # A memory 'E' from an earlier mission that must survive the provider swap.
        seed_id = experience_memory.store(self._exp(
            technique="File Upload Validation Bypass", tags=["upload", "web"], technologies=["php"],
            generalized_strategy="Bypass the upload filter with .phtml.", confidence=0.75))
        experience_memory.reload_index()

        # Provider A drives recon, then an operator stop pauses the session.
        provider_a = ScriptedProvider(lambda n: "curl -s http://sw.ctf/", provider_name="prov-A")
        exec_a = ScriptedToolExecutor(by_substring={
            "curl": self._recon_hit("curl -s http://sw.ctf/", "http://sw.ctf/upload.php")})
        sess = self._session("p2-sw", "Switch", "http://sw.ctf", "PHP upload portal.")
        await self._runtime(provider_a, exec_a).run(sess, max_turns=10, cancel_check=_cancel_after(2))

        paused = session_manager.get(sess.id)
        self.assertEqual(paused.status, "PAUSED")
        self.assertEqual(paused.provider_name, "prov-A")
        self.assertTrue(paused.state.known_endpoints)  # A-era discoveries persisted

        # A fresh process + a DIFFERENT provider resumes the SAME mission and solves it.
        resumed = session_manager.resume(sess.id)
        provider_b = ScriptedProvider(
            ["curl -F 'file=@x.phtml' http://sw.ctf/upload.php", "BUDGET_EXHAUSTED: x"],
            provider_name="prov-B")
        exec_b = ScriptedToolExecutor(by_substring={
            "upload.php": self._upload_hit("curl -F 'file=@x.phtml' http://sw.ctf/upload.php",
                                           "picoCTF{switch_ok}")})
        result = await self._runtime(provider_b, exec_b).run(resumed, max_turns=6)

        self.assertEqual(result.verified_flag, "picoCTF{switch_ok}")
        final = session_manager.get(sess.id)
        self.assertEqual(final.provider_name, "prov-B")
        self.assertTrue(final.state.known_endpoints)             # A's discoveries retained
        self.assertIsNotNone(experience_memory.get(seed_id))     # pre-existing memory intact
        # The mission's own new experience was learned despite the provider change.
        self.assertGreaterEqual(len(experience_memory.list_experiences(limit=50)), 2)

# =========================================================================== #
# Step 5/9: cross-session failed-approach recall
# =========================================================================== #

class TestCrossSessionFailureRecall(Phase2TestBase):

    def test_13_recall_prior_dead_end_from_another_session(self):
        """Step 5/9: a failed approach recorded in one mission's trajectory is
        surfaced (as advisory evidence) to a DIFFERENT later mission with similar
        conditions — so FORGE does not rediscover the same dead-end."""
        s1 = self._session("p2-x1", "Prior", "http://p1.ctf", "php login")
        trajectory_store.record(
            session_id=s1.id, challenge_id="p2-x1", event_type="REPLAN",
            command="sqlmap -u http://p1.ctf/login --batch",
            decision_summary="php sql injection attempt blocked by WAF; approach failed", result="BLOCKED")

        s2 = self._session("p2-x2", "Now", "http://p2.ctf", "php login bypass")
        s2.state.technologies.append("php")
        s2.state.vulnerabilities.append("sql injection")

        text = ContextBuilder().recall_cross_session_failures(s2.state, exclude_session=s2.id)
        self.assertTrue(text)
        self.assertIn("sqlmap", text)
        self.assertIn("CROSS-SESSION", text.upper())
        # The current session's own events are excluded from its cross-session recall.
        self.assertNotIn(s2.id, text)

# =========================================================================== #
# Step 20: ACCEPTANCE — Mission A teaches a different Mission B (self-improvement)
# =========================================================================== #

class TestAcceptanceMissionAtoB(Phase2TestBase):

    async def _solve_upload_mission(self, cid, name, target, description, commands, hits):
        provider = ScriptedProvider(commands + ["BUDGET_EXHAUSTED: done"])
        executor = ScriptedToolExecutor(by_substring=hits)
        sess = self._session(cid, name, target, description)
        return await self._runtime(provider, executor).run(sess, max_turns=6)

    async def test_14_missionA_experience_is_reused_and_reinforced_by_missionB(self):
        """Step 20 acceptance loop, end-to-end through the REAL runtime:

        Mission A discovers + exploits an upload vulnerability and solves it. The
        trajectory is saved, an 'upload validation bypass' experience is extracted
        and stored. Later, a DIFFERENT Mission B in the same family searches memory,
        retrieves A's experience, adapts it, and also solves. A's confidence rises
        and the reuse is logged — FORGE's core self-improvement loop."""
        # ── MISSION A ──────────────────────────────────────────────────────────
        res_a = await self._solve_upload_mission(
            "p2-A", "Avatar Upload A", "http://a-upload.ctf",
            "A PHP avatar upload portal that filters extensions.",
            commands=["curl -s http://a-upload.ctf/",
                      "curl -F 'file=@shell.php5' http://a-upload.ctf/upload.php"],
            hits={
                "upload.php": self._upload_hit(
                    "curl -F 'file=@shell.php5' http://a-upload.ctf/upload.php", "picoCTF{miss1on_A}"),
                "curl -s http://a-upload.ctf/": self._recon_hit(
                    "curl -s http://a-upload.ctf/", "http://a-upload.ctf/upload.php"),
            })
        self.assertEqual(res_a.status, "COMPLETED")

        exps = experience_memory.list_experiences(limit=5)
        self.assertEqual(len(exps), 1)
        exp_a_id = exps[0]["id"]
        conf_before = exps[0]["confidence"]
        self.assertEqual(exps[0]["source"], "forge_trajectory")
        self.assertIn("Upload", exps[0]["technique"])
        # Generalized: neither the flag nor the concrete target leaked into memory (Step 4).
        blob = json.dumps(experience_memory.get(exp_a_id, with_children=True))
        self.assertNotIn("picoCTF{miss1on_A}", blob)
        self.assertNotIn("a-upload.ctf", blob)

        experience_memory.reload_index()  # A's experience is now searchable for B

        # ── MISSION B (different challenge, same vulnerability family) ──────────
        res_b = await self._solve_upload_mission(
            "p2-B", "Profile Picture B", "http://b-files.ctf",
            "A PHP profile-picture file upload that validates the extension.",
            commands=["curl -F 'file=@avatar.phtml' http://b-files.ctf/profile/upload"],
            hits={"upload": self._upload_hit(
                "curl -F 'file=@avatar.phtml' http://b-files.ctf/profile/upload", "picoCTF{miss1on_B}")})
        self.assertEqual(res_b.status, "COMPLETED")
        self.assertEqual(res_b.verified_flag, "picoCTF{miss1on_B}")

        # ── THE CORE PROOF: A taught B, and B's success reinforced A ────────────
        a_after = experience_memory.get(exp_a_id)
        self.assertGreaterEqual(a_after["times_retrieved"], 1, "A should have been retrieved for B")
        self.assertGreaterEqual(a_after["times_used"], 1, "A should have been reused on B's solve")
        self.assertGreaterEqual(a_after["times_successful"], 2, "B's success should reinforce A")
        self.assertGreater(a_after["confidence"], conf_before, "reuse in a 2nd solve raises confidence")

        # Fine-grained telemetry (Step 14): A was 'retrieved' for and 'contributed' to B.
        db = SessionLocal()
        try:
            retrieved = db.query(MemoryUsageModel).filter(
                MemoryUsageModel.experience_id == exp_a_id, MemoryUsageModel.event == "retrieved").count()
            contributed = db.query(MemoryUsageModel).filter(
                MemoryUsageModel.experience_id == exp_a_id, MemoryUsageModel.event == "contributed").count()
        finally:
            db.close()
        self.assertGreaterEqual(retrieved, 1)
        self.assertGreaterEqual(contributed, 1)

        # Two runtime-sourced experiences now exist (A + B); knowledge accumulated.
        all_exps = experience_memory.list_experiences(limit=10)
        self.assertEqual(len(all_exps), 2)
        self.assertTrue(all(e["source"] == "forge_trajectory" for e in all_exps))

if __name__ == "__main__":
    unittest.main(verbosity=2)
