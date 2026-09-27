"""
Phase 6 — Experience Learning & Playbook Intelligence tests.

Covers the Part 12 checklist that is NEW in Phase 6, built strictly on top of the
existing (Phase 1–5) memory / playbook / reasoning layers — no second memory
database, planner, or execution system:

  • Experience         — successful vs failed extraction; facts (success→vuln) vs
                          hypotheses (failure→no vuln); confidence handling.
  • Contextual stats   — global vs per-context success rates; bounded by min-samples;
                          empty when no data (§9 / Part 9).
  • Failure learning   — the SWARM now learns from a stalled run and down-weights the
                          memories that did not help; an empty no-op run learns nothing;
                          a failed experience is never promoted (Part 8).
  • Retrieval→reasoning — learned knowledge creates candidates but NEVER auto-executes;
                          current evidence outranks a memory; historical/contextual
                          success influences (not dictates) scoring (Parts 6/7).
  • Observability      — a memory candidate carries structured historical_support that
                          serialises into the reasoning decision (Part 13).
  • Zero-regression    — with no stats wired, candidate generation is byte-for-byte the
                          Phase-5 formula.

Everything runs against the isolated unit-test database with the Playbook Vault sink
swapped to a temp directory, per the project's NO-DEMO-DATA / isolated-test-DB /
private-vault directives. No API key, network, or subprocess is required.
"""
import os
import glob
import shutil
import asyncio
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

# Import AFTER DATABASE_URL is set so module-level singletons bind to the test DB.
from backend.database.session import init_db, SessionLocal
from backend.database.models import ExperienceModel, ExperienceAttemptModel, MemoryUsageModel

from backend.knowledge.technique_stats import TechniqueStats, technique_stats
from backend.knowledge.memory_models import ExperienceRecord
from backend.knowledge.experience_memory import experience_memory
from backend.knowledge.memory_retriever import memory_retriever
from backend.knowledge.playbook_vault import PlaybookVault
from backend.agents.swarm_orchestrator import swarm_orchestrator
from backend.agents.swarm_state import SwarmBlackboard

from backend.swarm import (
    CandidateGenerator, CandidateAction, ActionScorer, SharedMissionState, Supervisor,
    Evidence,
)

# --------------------------------------------------------------------------- #
# Local test doubles (mirror tests/test_phase5_reasoning.py; add an id so the
# observability provenance can carry a memory id).
# --------------------------------------------------------------------------- #

class FakeMemory:
    """A memory_retriever double returning canned RetrievedMemory-like objects."""
    def __init__(self, memories):
        self._memories = memories
        self.calls = 0

    def retrieve(self, **kwargs):
        self.calls += 1
        return list(self._memories)

class FakeRetrievedMemory:
    def __init__(self, technique, *, kind="experience", strategy="", confidence=0.6,
                 success_rate=1.0, category="web", id="", outcome="success"):
        self.technique = technique
        self.kind = kind
        self.strategy = strategy
        self.confidence = confidence
        self.success_rate = success_rate
        self.category = category
        self.id = id
        self.outcome = outcome

def _exp(technique, *, category="web", technologies=None, outcome="success",
         confidence=0.8, difficulty="MEDIUM", tags=None, source_run_id=None):
    """A minimal generalized ExperienceRecord for statistics tests (no secrets)."""
    return ExperienceRecord(
        source="forge_run",
        source_run_id=source_run_id,
        category=category,
        difficulty=difficulty,
        technique=technique,
        tags=tags or [],
        technologies=list(technologies or []),
        target_characteristics={"category": category, "technologies": list(technologies or [])},
        outcome=outcome,
        confidence=confidence,
    )

def _failed_web_board(challenge_id="p6-fail-1", run_id="p6-failrun-1",
                      name="Unsolved Portal", with_history=True):
    """A REAL SwarmBlackboard for a run that did NOT capture a flag."""
    board = SwarmBlackboard(challenge_id=challenge_id, run_id=run_id,
                            target_scope="http://hard.range.ctf:8080")
    board.challenge_name = name
    board.category = "WEB"
    board.difficulty = "HARD"
    board.description = "A login portal that resisted every attempt."
    board.discovered_endpoints.update(["/", "/login", "/admin"])
    board.extracted_headers.update({"Server": "nginx"})
    board.retrieved_memory_ids = []
    if with_history:
        board.record_agent_step("agent_1", command="curl -s http://hard.range.ctf:8080/",
                                output="HTTP/1.1 200 OK\nServer: nginx", note="recon")
        board.record_agent_step("agent_1",
                                command="ffuf -w list -u http://hard.range.ctf:8080/FUZZ",
                                output="no interesting endpoints found", note="dir enum dead end")
        board.record_agent_step("agent_2",
                                command="sqlmap -u http://hard.range.ctf:8080/login --batch",
                                output="no injection point identified", note="sqli attempt failed")
    # No flag captured — board.flag_captured left falsy.
    return board

# --------------------------------------------------------------------------- #
# Shared base: isolated DB + temp vault sink (private-vault rule).
# --------------------------------------------------------------------------- #

class Phase6Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self._clean_tables()
        experience_memory.reload_index()
        self.tmp_dir = tempfile.mkdtemp(prefix="forge_p6_")
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
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        for f in glob.glob(os.path.join("backend", "logs", "challenge_p6-*.log")):
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
            db.commit()
        finally:
            db.close()

    def _store(self, record):
        exp_id = experience_memory.store(record)
        self.assertTrue(exp_id)
        return exp_id

# =========================================================================== #
# Part 9 — Contextual success statistics                                       #
# =========================================================================== #

class TestContextualStatistics(Phase6Base):

    def _seed_split(self):
        """3 python-web successes + 4 windows-web attempts (1 success) of the same
        technique family — the canonical 'X against python vs windows' split (§9)."""
        for i in range(3):
            self._store(_exp("SQL injection union select", category="web",
                             technologies=["python", "flask"], outcome="success"))
        self._store(_exp("SQL injection union select", category="web",
                         technologies=["windows", "mssql"], outcome="success"))
        for i in range(3):
            self._store(_exp("SQL injection union select", category="web",
                             technologies=["windows", "mssql"], outcome="failed"))

    def test_global_and_contextual_split(self):
        self._seed_split()
        stats = TechniqueStats()
        py = stats.lookup("sql injection", category="web", technologies=["python"])
        win = stats.lookup("sql injection", category="web", technologies=["windows"])

        # Contextual: python web is a strong technique; windows is weak — same technique.
        self.assertEqual(py["contextual"]["sample_size"], 3)
        self.assertEqual(py["contextual"]["success_rate"], 1.0)
        self.assertEqual(win["contextual"]["sample_size"], 4)
        self.assertEqual(win["contextual"]["success_rate"], 0.25)
        # Global spans everything and sits between the two contexts.
        self.assertEqual(py["global"]["sample_size"], 7)
        self.assertEqual(py["global"]["success_rate"], round(4 / 7, 3))
        self.assertGreater(py["contextual"]["success_rate"], py["global"]["success_rate"])
        self.assertLess(win["contextual"]["success_rate"], win["global"]["success_rate"])

    def test_contextual_rate_prefers_context_over_global(self):
        self._seed_split()
        stats = TechniqueStats()
        rate_win, s = stats.contextual_success_rate("sql injection", category="web",
                                                    technologies=["windows"])
        # The windows context (0.25) is returned, NOT the global 0.571.
        self.assertEqual(rate_win, 0.25)
        self.assertEqual(s["contextual"]["sample_size"], 4)

    def test_insufficient_samples_returns_none(self):
        self._store(_exp("timing side channel", category="crypto",
                         technologies=["python"], outcome="success"))
        stats = TechniqueStats()
        # Only one sample → below the min-sample floor → no trusted rate (fall back).
        rate, s = stats.contextual_success_rate("timing side channel", category="crypto",
                                               technologies=["python"], min_samples=2)
        self.assertIsNone(rate)
        self.assertEqual(s["global"]["sample_size"], 1)

    def test_empty_when_no_matching_technique(self):
        self._seed_split()
        stats = TechniqueStats()
        s = stats.lookup("heap grooming tcache poisoning", category="pwn")
        self.assertEqual(s["global"]["sample_size"], 0)
        self.assertEqual(s["global"]["success_rate"], 0.0)
        rate, _ = stats.contextual_success_rate("heap grooming tcache poisoning", category="pwn")
        self.assertIsNone(rate)

# =========================================================================== #
# Part 8 — Learning from FAILURE (the swarm now learns from stalled runs)       #
# =========================================================================== #

class TestFailureLearning(Phase6Base):

    def _learn(self, board, outcome):
        asyncio.run(swarm_orchestrator._learn_from_run(board, outcome=outcome))

    def test_failed_run_stores_a_failure_experience(self):
        self._learn(_failed_web_board(), "failed")
        db = SessionLocal()
        try:
            rows = db.query(ExperienceModel).all()
            self.assertEqual(len(rows), 1, "a stalled run should still teach ONE experience")
            exp = rows[0]
            self.assertEqual(exp.outcome, "failed")
            self.assertEqual(exp.source_challenge_id, "p6-fail-1")
            # A failure is remembered with LOWER confidence than a solve, and records
            # what did NOT work rather than a confirmed vulnerability.
            self.assertLessEqual(exp.confidence, 0.5)
            self.assertEqual(exp.successful_techniques or [], [])
            self.assertEqual(exp.vulnerabilities or [], [])
        finally:
            db.close()

    def test_empty_no_op_failed_run_learns_nothing(self):
        # A run that never executed anything has no lesson — storing it would be noise
        # (and violates the no-demo-data spirit).
        self._learn(_failed_web_board(with_history=False), "failed")
        db = SessionLocal()
        try:
            self.assertEqual(db.query(ExperienceModel).count(), 0)
        finally:
            db.close()

    def test_failed_run_negatively_reinforces_retrieved_memories(self):
        # A memory that WAS retrieved for this run but the run still failed should lose
        # confidence — contextual decay, never a blacklist (Part 8).
        exp_id = self._store(_exp("directory brute force", category="web",
                                  technologies=["nginx"], outcome="success", confidence=0.7))
        before = experience_memory.get(exp_id)
        board = _failed_web_board()
        board.retrieved_memory_ids = [exp_id]
        self._learn(board, "failed")
        after = experience_memory.get(exp_id)
        self.assertEqual(after["times_failed"], (before["times_failed"] or 0) + 1)
        self.assertLess(after["confidence"], before["confidence"])
        # Still present — never deleted or blocked.
        self.assertIsNotNone(experience_memory.get(exp_id))

    def test_failed_experience_is_not_promoted_to_playbook(self):
        self._learn(_failed_web_board(), "failed")
        exp = experience_memory.list_experiences(limit=1)[0]
        self.assertIsNone(exp.get("promoted_playbook_id"))
        # No YAML written into the (temp) vault by a failed solve.
        yamls = glob.glob(os.path.join(self._tmp_vault.base_dir, "**", "*.yaml"), recursive=True)
        self.assertEqual(yamls, [])

    def test_solved_run_still_learns_positively(self):
        # Regression: the pre-existing success path is unchanged.
        exp_id = self._store(_exp("ssti jinja", category="web", technologies=["python"],
                                 outcome="success", confidence=0.6))
        before = experience_memory.get(exp_id)
        board = _failed_web_board(challenge_id="p6-solve-1", run_id="p6-solverun-1")
        board.flag_captured = "FLAG{redacted_in_test}"
        board.retrieved_memory_ids = [exp_id]
        self._learn(board, "success")
        after = experience_memory.get(exp_id)
        self.assertEqual(after["times_successful"], (before["times_successful"] or 0) + 1)
        self.assertGreater(after["confidence"], before["confidence"])

# =========================================================================== #
# Parts 6/7 — Retrieval → candidates; memory informs but never controls         #
# =========================================================================== #

class TestMemoryInfluencesReasoning(Phase6Base):

    def _ms(self, **kw):
        d = dict(mission_id="p6", target="http://target.ctf:8080", category="web")
        d.update(kw)
        return SharedMissionState(**d)

    def test_learned_knowledge_creates_candidate_but_never_evidence_sourced(self):
        # A retrieved memory generates a candidate, but it is tagged advisory
        # (source memory/playbook) — the coordinator only auto-injects EVIDENCE-sourced
        # candidates, so learned knowledge can never auto-execute (Part 7 / §32).
        gen = CandidateGenerator(memory_retriever=FakeMemory(
            [FakeRetrievedMemory("SSTI template injection", kind="playbook",
                                 strategy="render a malicious template", confidence=0.8)]))
        cands = [c for c in gen.generate(self._ms()) if c.source in ("memory", "playbook")]
        self.assertTrue(cands)
        self.assertTrue(all(c.source != "evidence" for c in cands))

    def test_current_evidence_outranks_a_historical_memory(self):
        # Two candidates for the same action: one strongly backed by CURRENT evidence,
        # one only by history. At low uncertainty (exploitation) evidence wins — current
        # evidence remains authoritative (Part 7 / criterion 9).
        evidence_backed = CandidateAction(
            action_type="web_exploit", objective="Exploit confirmed SQLi seen in the response.",
            source="evidence", evidence_support=0.85, success_probability=0.55, novelty=1.0)
        memory_backed = CandidateAction(
            action_type="web_exploit", objective="Adapt a historical SQLi technique.",
            target="other",  # different signature so it is not deduped/penalised as attempted
            source="memory", evidence_support=0.35, success_probability=0.9, novelty=0.85)
        scorer = ActionScorer()
        ranked = scorer.rank([memory_backed, evidence_backed], uncertainty=0.2)
        self.assertEqual(ranked[0].source, "evidence")

    def test_historical_success_lifts_and_poor_history_tempers_probability(self):
        # Same memory, two different mission contexts. A strong python-web track record
        # lifts the candidate's success_probability above the no-history baseline; a poor
        # windows track record pulls it below. History is ONE input, never the sole one.
        for _ in range(3):
            self._store(_exp("SQL injection union", category="web",
                             technologies=["python"], outcome="success"))
        for _ in range(3):
            self._store(_exp("SQL injection union", category="web",
                             technologies=["windows"], outcome="failed"))
        self._store(_exp("SQL injection union", category="web",
                         technologies=["windows"], outcome="success"))

        mem = [FakeRetrievedMemory("SQL injection union", confidence=0.6,
                                   success_rate=0.6, category="web")]
        baseline = CandidateGenerator(memory_retriever=FakeMemory(list(mem)))  # no stats
        withstats = CandidateGenerator(memory_retriever=FakeMemory(list(mem)),
                                       technique_stats=technique_stats)

        base_prob = [c for c in baseline.generate(self._ms(technologies=["python"]))
                     if c.source == "memory"][0].success_probability
        py_prob = [c for c in withstats.generate(self._ms(technologies=["python"]))
                   if c.source == "memory"][0].success_probability
        win_prob = [c for c in withstats.generate(self._ms(technologies=["windows"]))
                    if c.source == "memory"][0].success_probability

        self.assertGreater(py_prob, base_prob)     # strong history lifts
        self.assertLess(win_prob, base_prob)        # poor history tempers
        self.assertLess(py_prob, 1.0)               # never certainty

    def test_memory_candidate_carries_capability_for_the_gate(self):
        # A memory proposing an OCR technique must carry capability="ocr" so the existing
        # capability/privilege gate decides — learned knowledge never bypasses controls.
        gen = CandidateGenerator(memory_retriever=FakeMemory(
            [FakeRetrievedMemory("ocr the image to read the text", kind="playbook",
                                 strategy="tesseract image.png", confidence=0.8)]))
        sup = Supervisor("p6")
        ms = self._ms(category="forensics", target="/tmp/x.png")
        decision = sup.reason(ms, generator=gen, blocked_capabilities=["ocr"])
        # The blocked-capability candidate is dropped by the prefilter — not run.
        self.assertFalse(any(c.capability == "ocr" for c in decision.ranked))

# =========================================================================== #
# Part 13 — Observability of WHY learned knowledge affected a decision          #
# =========================================================================== #

class TestObservability(Phase6Base):

    def _ms(self, **kw):
        d = dict(mission_id="p6", target="http://target.ctf:8080", category="web")
        d.update(kw)
        return SharedMissionState(**d)

    def test_memory_candidate_records_structured_provenance(self):
        for _ in range(3):
            self._store(_exp("SQL injection union", category="web",
                             technologies=["python"], outcome="success"))
        gen = CandidateGenerator(
            memory_retriever=FakeMemory([FakeRetrievedMemory(
                "SQL injection union", confidence=0.7, success_rate=0.8,
                category="web", id="mem-123")]),
            technique_stats=technique_stats)
        c = [c for c in gen.generate(self._ms(technologies=["python"]))
             if c.source == "memory"][0]
        hs = c.historical_support
        self.assertEqual(hs["source"], "memory")
        self.assertIn("mem-123", hs["memory_ids"])
        self.assertEqual(hs["similar_challenges"], 3)         # 3 similar past challenges
        self.assertEqual(hs["contextual_success_rate"], 1.0)   # strong track record
        self.assertIn("applied_success_probability", hs)
        self.assertIn("context", hs)

    def test_provenance_serializes_into_reasoning_decision(self):
        # Use an SSTI technique (→ web_exploit) so the memory candidate is DISTINCT from
        # the generic recon/enum a bare web target already implies (no dedupe collision).
        for _ in range(2):
            self._store(_exp("SSTI template injection", category="web",
                             technologies=["python"], outcome="success"))
        gen = CandidateGenerator(
            memory_retriever=FakeMemory([FakeRetrievedMemory(
                "SSTI template injection", strategy="render malicious template",
                confidence=0.7, category="web", id="mem-ssti")]),
            technique_stats=technique_stats)
        sup = Supervisor("p6")
        decision = sup.reason(self._ms(technologies=["python"]), generator=gen)
        payload = decision.to_dict()
        # The full "why" is inspectable: every ranked candidate serialises its
        # historical_support + score_breakdown.
        mem_dicts = [c for c in payload["ranked"] if c["source"] == "memory"]
        self.assertTrue(mem_dicts)
        self.assertIn("historical_support", mem_dicts[0])
        self.assertIn("score_breakdown", mem_dicts[0])

    def test_basic_provenance_present_even_without_stats(self):
        gen = CandidateGenerator(memory_retriever=FakeMemory(
            [FakeRetrievedMemory("xss reflected", confidence=0.6, id="mem-x")]))
        c = [c for c in gen.generate(self._ms()) if c.source == "memory"][0]
        self.assertEqual(c.historical_support["source"], "memory")
        self.assertIn("mem-x", c.historical_support["memory_ids"])
        # No stats wired → no contextual fields fabricated.
        self.assertNotIn("contextual_success_rate", c.historical_support)

# =========================================================================== #
# Zero-regression guarantee — no stats wired == exact Phase-5 behaviour          #
# =========================================================================== #

class TestZeroRegressionFallback(Phase6Base):

    def _ms(self, **kw):
        d = dict(mission_id="p6", target="http://target.ctf:8080", category="web")
        d.update(kw)
        return SharedMissionState(**d)

    def test_no_stats_matches_phase5_formula(self):
        # With no technique_stats injected, success_probability is exactly the Phase-5
        # formula: clamp(0.35 + 0.4 * confidence * success_rate).
        gen = CandidateGenerator(memory_retriever=FakeMemory(
            [FakeRetrievedMemory("sqli union", confidence=0.9, success_rate=1.0, category="web")]))
        c = [c for c in gen.generate(self._ms()) if c.source == "memory"][0]
        expected = max(0.0, min(1.0, 0.35 + 0.4 * 0.9 * 1.0))
        self.assertAlmostEqual(c.success_probability, expected, places=6)

    def test_stats_present_but_no_matching_rows_falls_back(self):
        # Stats wired, but the DB has NO experience for this technique → fall back to the
        # exact same formula (contextual influence only kicks in with real evidence).
        gen = CandidateGenerator(
            memory_retriever=FakeMemory([FakeRetrievedMemory(
                "novel unheard technique xyz", confidence=0.5, success_rate=0.5, category="web")]),
            technique_stats=technique_stats)
        c = [c for c in gen.generate(self._ms()) if c.source == "memory"][0]
        expected = max(0.0, min(1.0, 0.35 + 0.4 * 0.5 * 0.5))
        self.assertAlmostEqual(c.success_probability, expected, places=6)

# =========================================================================== #
# Part 2 — Experience distinguishes success (fact) from failure (no fact)        #
# =========================================================================== #

class TestExperienceExtractionOutcome(Phase6Base):

    def test_failed_extraction_has_no_confirmed_vulnerability(self):
        from backend.knowledge.experience_extractor import experience_extractor
        board = _failed_web_board(challenge_id="p6-x-1", run_id="p6-xr-1")
        rec = experience_extractor.extract_from_board(board, flag="", outcome="failed")
        self.assertEqual(rec.outcome, "failed")
        self.assertEqual(rec.vulnerabilities, [])          # nothing confirmed
        self.assertEqual(rec.successful_techniques, [])
        self.assertLessEqual(rec.confidence, 0.5)

    def test_success_extraction_records_confirmed_vulnerability(self):
        from backend.knowledge.experience_extractor import experience_extractor
        board = _failed_web_board(challenge_id="p6-x-2", run_id="p6-xr-2")
        board.record_agent_step("agent_2",
                                command="curl -s 'http://hard.range.ctf:8080/greet?name={{7*7}}'",
                                output="Hello, 49", note="ssti reflected")
        board.flag_captured = "FLAG{redacted}"
        rec = experience_extractor.extract_from_board(board, flag="FLAG{redacted}", outcome="success")
        self.assertEqual(rec.outcome, "success")
        self.assertTrue(rec.vulnerabilities)               # a confirmed technique/fact
        self.assertGreater(rec.confidence, 0.5)

if __name__ == "__main__":
    unittest.main()
