"""
Phase 5 — Adaptive Autonomous Reasoning tests.

Covers the §34 checklist (fact/hypothesis separation, evidence confidence, failed-
action memory, duplicate suppression, candidate generation & scoring, information
gain, exploration/exploitation, failure classification, adaptive replanning, memory
retrieval & confidence, playbook adaptation, no-progress detection, stop conditions,
mission budget) plus the §29–§33 regressions and the §-integration path
(Agent → EvidenceBus → Supervisor → CandidateActions → Scheduler → MissionState →
Replanning).

Everything runs against the isolated unit-test database with LOCAL test doubles for
the provider / tool executor / specialist agent (the same pattern as
tests/test_phase4_swarm.py), so NO API key, network, or subprocess is required.
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
    AgentSessionModel, TrajectoryEventModel, SwarmMissionModel, SwarmTaskModel,
    SwarmEvidenceModel,
)

from backend.swarm import (
    AgentRole, Evidence, EvidenceType, EvidenceBus, Task, TaskStatus,
    SharedMissionState, Supervisor, SwarmLimits, SwarmCoordinator,
    # Phase 5 primitives
    Reliability, classify_reliability, Fact, Hypothesis, HypothesisStatus,
    FailedApproach, FailureClass, RecoveryHint, classify_failure, recovery_hint_for,
    CandidateAction, InformationGain, Risk, Cost, profile_for,
    ActionScorer, ScoreWeights, mission_uncertainty, information_gain_for,
    CandidateGenerator, technique_to_action_type,
    MissionBudget, ProgressLedger, StopCondition, evaluate_stop, knowledge_fingerprint,
    AgentResult,
)
from backend.swarm.dedup import action_signature, normalize_target
from backend.swarm import events as swarm_events

# --------------------------------------------------------------------------- #
# Local test doubles
# --------------------------------------------------------------------------- #

def _run(coro):
    return asyncio.run(coro)

class _R:
    """A minimal agent/task result stand-in for failure classification."""
    def __init__(self, status="FAILED", failure_category="", reason=""):
        self.status = status
        self.failure_category = failure_category
        self.reason = reason

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
                 success_rate=1.0, category="web"):
        self.technique = technique
        self.kind = kind
        self.strategy = strategy
        self.confidence = confidence
        self.success_rate = success_rate
        self.category = category

class ProgrammableAgent:
    """A SpecialistAgent double whose result is built by a per-mission responder.

    Records every (role, objective) it executes so a test can assert what ran and how
    often — the way we verify "A was not repeated" for the adaptive-planning regression.
    """
    def __init__(self, role, responder, executed):
        self.role = role
        self._responder = responder
        self._executed = executed

    async def execute(self, task, mission, bus, **kw):
        rv = self.role.value if hasattr(self.role, "value") else str(self.role)
        self._executed.append((rv, task.objective))
        return self._responder(rv, task)

class _CaptureEvents:
    def __init__(self):
        self.events = []

    def __enter__(self):
        self._orig = swarm_events.broadcast

        def rec(event, payload=None):
            self.events.append((event, payload or {}))
        swarm_events.broadcast = rec
        return self

    def __exit__(self, *a):
        swarm_events.broadcast = self._orig

    def names(self):
        return [e for e, _ in self.events]

class Phase5Base(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        db = SessionLocal()
        try:
            for model in (TrajectoryEventModel, AgentSessionModel, SwarmEvidenceModel,
                          SwarmTaskModel, SwarmMissionModel):
                db.query(model).delete()
            db.commit()
        finally:
            db.close()

    def _ms(self, **kw):
        defaults = dict(mission_id="m5", target="http://target.ctf:8080", category="web")
        defaults.update(kw)
        return SharedMissionState(**defaults)

    def _coord(self, **kw):
        defaults = dict(challenge_id="chal-p5", run_id=None, target="http://target.ctf:8080",
                        category="web", challenge_name="Phase5 Test", persist=True,
                        limits=SwarmLimits(max_task_retries=0))
        defaults.update(kw)
        return SwarmCoordinator(**defaults)

# =========================================================================== #
# §5 — Facts vs hypotheses (distinct; validation lifecycle)                    #
# =========================================================================== #

class TestFactsVsHypotheses(Phase5Base):

    def test_strong_evidence_becomes_fact_speculative_becomes_hypothesis(self):
        ms = self._ms()
        # DIRECT (from command output) → confirmed fact.
        ms.integrate_evidence(Evidence(mission_id="m5", evidence_type="service",
                                       title="OpenSSH 8.2", source="command", confidence=0.9))
        # SPECULATIVE (llm prose) → open hypothesis, NOT a fact.
        ms.integrate_evidence(Evidence(mission_id="m5", evidence_type="vulnerability",
                                       title="maybe SQLi", source="llm", confidence=0.3,
                                       tags=["speculative"]))
        facts = [f.statement for f in ms.get_facts()]
        hyps = [(h.statement, h.status) for h in ms.get_hypotheses()]
        self.assertTrue(any("OpenSSH 8.2" in f for f in facts))
        self.assertFalse(any("SQLi" in f for f in facts))          # speculation is NOT a fact
        self.assertTrue(any("SQLi" in s and st == "open" for s, st in hyps))

    def test_hypothesis_confirmation_promotes_to_fact(self):
        ms = self._ms()
        ms.add_hypothesis("The app is Flask", confidence=0.4)
        self.assertTrue(ms.get_hypotheses(open_only=True))
        ok = ms.confirm_hypothesis("The app is Flask", evidence_id="ev1")
        self.assertTrue(ok)
        self.assertFalse(ms.get_hypotheses(open_only=True))        # no longer open
        self.assertTrue(any("Flask" in f.statement for f in ms.get_facts()))  # promoted

    def test_hypothesis_never_overwrites_confirmed_fact(self):
        ms = self._ms()
        ms.add_fact("Server is nginx", reliability=Reliability.DIRECT.value)
        # A later speculative claim of the SAME statement must not shadow the fact.
        rec = ms.add_hypothesis("Server is nginx", confidence=0.2)
        self.assertIsNone(rec)
        self.assertEqual(len(ms.hypothesis_records), 0)
        self.assertEqual(len(ms.confirmed_facts), 1)

    def test_rejected_hypothesis_is_marked_not_deleted(self):
        ms = self._ms()
        ms.add_hypothesis("Vulnerable to SSTI", confidence=0.5)
        ms.reject_hypothesis("Vulnerable to SSTI", evidence_id="ev-neg")
        statuses = [h.status for h in ms.get_hypotheses()]
        self.assertIn(HypothesisStatus.REJECTED.value, statuses)
        self.assertFalse(ms.get_hypotheses(open_only=True))

# =========================================================================== #
# §6 — Evidence reliability / confidence                                       #
# =========================================================================== #

class TestEvidenceReliability(Phase5Base):

    def test_reliability_classification_rules(self):
        self.assertEqual(classify_reliability(source="command"), Reliability.DIRECT)
        self.assertEqual(classify_reliability(source="observation", confidence=0.6), Reliability.DERIVED)
        self.assertEqual(classify_reliability(source="supervisor"), Reliability.INFERRED)
        self.assertEqual(classify_reliability(source="llm"), Reliability.SPECULATIVE)
        self.assertEqual(classify_reliability(tags=["speculative"]), Reliability.SPECULATIVE)

    def test_reliability_weight_ordering(self):
        self.assertGreater(Reliability.DIRECT.weight, Reliability.DERIVED.weight)
        self.assertGreater(Reliability.DERIVED.weight, Reliability.INFERRED.weight)
        self.assertGreater(Reliability.INFERRED.weight, Reliability.SPECULATIVE.weight)

    def test_evidence_autoclassifies_and_is_strong(self):
        strong = Evidence(mission_id="m", evidence_type="endpoint", title="/admin",
                          source="command", confidence=0.9)
        weak = Evidence(mission_id="m", evidence_type="vulnerability", title="maybe RCE",
                        source="llm", confidence=0.3, tags=["speculative"])
        self.assertTrue(strong.is_strong)
        self.assertFalse(weak.is_strong)
        self.assertEqual(weak.reliability, Reliability.SPECULATIVE.value)

# =========================================================================== #
# §7 — Failed-approach memory (bounded, deduped)                               #
# =========================================================================== #

class TestFailedApproachMemory(Phase5Base):

    def test_record_and_query_failed_approach(self):
        ms = self._ms()
        sig = action_signature("directory_enum", "http://target.ctf/admin")
        added = ms.record_failed_approach(action="GET /admin", signature=sig, result="404",
                                          reason="not found", failure_class="NO_RESULT")
        self.assertTrue(added)
        self.assertTrue(ms.has_failed_action(sig))
        self.assertEqual(len(ms.get_failed_approaches()), 1)

    def test_failed_approach_dedup_by_signature(self):
        ms = self._ms()
        sig = action_signature("port_scan", "10.10.10.10")
        self.assertTrue(ms.record_failed_approach(action="nmap", signature=sig, result="down"))
        # Same signature again → not a new entry (refreshes reason instead).
        self.assertFalse(ms.record_failed_approach(action="nmap -sV", signature=sig, result="down2"))
        self.assertEqual(len(ms.get_failed_approaches()), 1)

    def test_failed_approach_storage_is_bounded(self):
        ms = self._ms()
        for i in range(400):
            ms.record_failed_approach(action=f"a{i}", signature=f"sig::{i}", result="x")
        # Bounded well under an unbounded 400 (the cap is 150).
        self.assertLessEqual(len(ms.failed_approaches), 150)

# =========================================================================== #
# §8 — Duplicate action suppression via normalized signatures                  #
# =========================================================================== #

class TestDuplicateSuppression(Phase5Base):

    def test_action_signature_normalizes_target(self):
        # scheme, default port and trailing slash are irrelevant to identity.
        self.assertEqual(action_signature("port_scan", "http://t.ctf:80/"),
                         action_signature("port_scan", "t.ctf"))
        self.assertEqual(normalize_target("HTTPS://Host:443/"), "host")

    def test_equivalent_commands_share_signature(self):
        # Whitespace / casing differences collapse (not raw string equality, §8).
        a = action_signature("port_scan", "10.0.0.1", "-sV  -p-")
        b = action_signature("port_scan", "10.0.0.1", "-sv -p-")
        self.assertEqual(a, b)

    def test_generator_dedupes_candidates(self):
        gen = CandidateGenerator()
        ms = self._ms()
        # Two artifact evidences with the same id → one candidate after dedup.
        ev = Evidence(mission_id="m5", evidence_type="artifact", title="cap.pcap",
                      artifact_id="cap.pcap", source="command")
        cands = gen.generate(ms, recent_evidence=[ev, ev], use_memory=False)
        sigs = [c.signature for c in cands]
        self.assertEqual(len(sigs), len(set(sigs)))                # no duplicate signatures

    def test_reason_filters_already_attempted_actions(self):
        sup = Supervisor("m5")
        ms = self._ms(category="misc", target="http://target.ctf")
        # First reasoning pass → pick a candidate, then mark its action attempted.
        d1 = sup.reason(ms, use_memory=False)
        self.assertIsNotNone(d1.selected)
        first_sig = d1.selected.signature
        ms.record_action_signature(first_sig)
        # Second pass with the attempted signature supplied → that action is not re-selected.
        d2 = sup.reason(ms, use_memory=False, attempted_signatures=[first_sig])
        self.assertNotEqual(d2.selected and d2.selected.signature, first_sig)

# =========================================================================== #
# §9/§16/§11 — Candidate generation                                            #
# =========================================================================== #

class TestCandidateGeneration(Phase5Base):

    def test_state_gap_generates_fingerprint_when_nothing_known(self):
        gen = CandidateGenerator()
        ms = self._ms(category="web", target="http://target.ctf")
        cands = gen.generate(ms, use_memory=False)
        types = {c.action_type for c in cands}
        self.assertTrue({"service_fingerprint", "directory_enum", "http_inspect"} & types)

    def test_known_vuln_generates_exploit_candidate(self):
        gen = CandidateGenerator()
        ms = self._ms(vulnerabilities=["CVE-2021-41773 path traversal"])
        cands = gen.generate(ms, use_memory=False)
        self.assertTrue(any(c.action_type == "vuln_exploit" for c in cands))

    def test_evidence_generates_reactive_candidate(self):
        gen = CandidateGenerator()
        ms = self._ms()
        ev = Evidence(mission_id="m5", evidence_type="technology",
                      title="Apache 2.4.49", related_technology="Apache 2.4.49", source="command")
        cands = gen.generate(ms, recent_evidence=[ev], use_memory=False)
        self.assertTrue(any(c.source == "evidence" and "apache" in c.objective.lower() for c in cands))

# =========================================================================== #
# §10 — Deterministic candidate scoring                                        #
# =========================================================================== #

class TestScoring(Phase5Base):

    def _cand(self, **kw):
        d = dict(action_type="directory_enum", objective="enumerate", capability="directory_enum")
        d.update(kw)
        return CandidateAction(**d)

    def test_score_is_deterministic_and_has_breakdown(self):
        scorer = ActionScorer()
        c1 = scorer.score(self._cand(), uncertainty=1.0)
        c2 = scorer.score(self._cand(), uncertainty=1.0)
        self.assertEqual(c1.score, c2.score)                       # deterministic
        self.assertIn("information_gain", c1.score_breakdown)      # inspectable (§37)
        self.assertIn("duplicate_penalty", c1.score_breakdown)

    def test_higher_information_gain_scores_higher(self):
        scorer = ActionScorer()
        hi = scorer.score(self._cand(information_gain=InformationGain.HIGH.value), uncertainty=1.0)
        lo = scorer.score(self._cand(information_gain=InformationGain.LOW.value), uncertainty=1.0)
        self.assertGreater(hi.score, lo.score)

    def test_duplicate_penalty_sinks_attempted_action(self):
        scorer = ActionScorer()
        c = self._cand()
        fresh = scorer.score(self._cand(), uncertainty=1.0).score
        dup = scorer.score(c, attempted_signatures=[c.signature], uncertainty=1.0).score
        self.assertLess(dup, fresh)

    def test_blocked_capability_is_penalised(self):
        scorer = ActionScorer()
        free = scorer.score(self._cand(), uncertainty=1.0).score
        blk = scorer.score(self._cand(), blocked_capabilities=["directory_enum"], uncertainty=1.0).score
        self.assertLess(blk, free)

    def test_rank_orders_best_first(self):
        scorer = ActionScorer()
        cands = [
            self._cand(action_type="credential_bruteforce", information_gain=InformationGain.LOW.value,
                       cost=Cost.HIGH.value, risk=Risk.HIGH.value, capability="auth_test"),
            self._cand(action_type="service_fingerprint", information_gain=InformationGain.HIGH.value,
                       cost=Cost.LOW.value, risk=Risk.LOW.value, capability="service_enumeration"),
        ]
        ranked = scorer.rank(cands, uncertainty=1.0)
        self.assertEqual(ranked[0].action_type, "service_fingerprint")  # info-seeking wins

# =========================================================================== #
# §11 — Information gain relative to current knowledge                         #
# =========================================================================== #

class TestInformationGain(Phase5Base):

    def test_fingerprint_high_gain_when_services_unknown_low_when_known(self):
        empty = self._ms()
        known = self._ms(services=["ssh 22"], technologies=["OpenSSH"])
        self.assertEqual(information_gain_for("service_fingerprint", empty), InformationGain.HIGH.value)
        self.assertEqual(information_gain_for("service_fingerprint", known), InformationGain.LOW.value)

    def test_enumeration_high_gain_when_surface_unknown(self):
        empty = self._ms()
        mapped = self._ms(endpoints=["/a", "/b", "/c"])
        self.assertEqual(information_gain_for("directory_enum", empty), InformationGain.HIGH.value)
        self.assertEqual(information_gain_for("directory_enum", mapped), InformationGain.LOW.value)

    def test_planner_prefers_information_seeking_over_bruteforce(self):
        # The reusable mechanism, not a hardcoded example: with high uncertainty, an
        # info-seeking action outranks a blind credential brute-force.
        gen = CandidateGenerator()
        scorer = ActionScorer()
        ms = self._ms(category="web", target="http://target.ctf")
        cands = gen.generate(ms, use_memory=False)
        ranked = scorer.rank(cands, uncertainty=mission_uncertainty(ms))
        top = ranked[0].action_type
        self.assertIn(top, ("service_fingerprint", "directory_enum", "http_inspect"))

# =========================================================================== #
# §12 — Exploration vs exploitation                                            #
# =========================================================================== #

class TestExplorationExploitation(Phase5Base):

    def test_uncertainty_high_when_nothing_known_low_with_vuln(self):
        self.assertGreater(mission_uncertainty(self._ms()), 0.6)
        exploit_ready = self._ms(vulnerabilities=["SQLi in /login"])
        self.assertLess(mission_uncertainty(exploit_ready), 0.5)

    def test_exploitation_bias_when_evidence_is_strong(self):
        # With low uncertainty, an evidence-backed exploit beats a generic scan.
        scorer = ActionScorer()
        exploit = CandidateAction(action_type="vuln_exploit", objective="exploit",
                                  capability="web_exploit", evidence_support=0.9,
                                  information_gain=InformationGain.MEDIUM.value,
                                  success_probability=0.6)
        scan = CandidateAction(action_type="service_fingerprint", objective="scan",
                               capability="service_enumeration", evidence_support=0.2,
                               information_gain=InformationGain.HIGH.value, success_probability=0.6)
        ranked = scorer.rank([scan, exploit], uncertainty=0.2)      # low uncertainty → exploit
        self.assertEqual(ranked[0].action_type, "vuln_exploit")

# =========================================================================== #
# §14 — Failure classification & recovery hints                                #
# =========================================================================== #

class TestFailureClassification(Phase5Base):

    def test_failure_taxonomy(self):
        self.assertEqual(classify_failure(_R(failure_category="COMMAND_NOT_FOUND")), FailureClass.TOOL_FAILURE)
        self.assertEqual(classify_failure(_R(failure_category="BLOCKED_CAPABILITY")), FailureClass.CAPABILITY_FAILURE)
        self.assertEqual(classify_failure(_R(failure_category="TARGET_MISMATCH")), FailureClass.TARGET_MISMATCH)
        self.assertEqual(classify_failure(_R(status="TIMEOUT")), FailureClass.TIMEOUT)
        self.assertEqual(classify_failure(_R(reason="connection refused")), FailureClass.NETWORK_FAILURE)
        self.assertEqual(classify_failure(_R(reason="401 unauthorized")), FailureClass.AUTH_FAILURE)
        self.assertEqual(classify_failure(_R(status="MAX_TURNS")), FailureClass.STRATEGY_FAILURE)
        self.assertEqual(classify_failure(_R(reason="provider exhausted")), FailureClass.PROVIDER_FAILURE)

    def test_recovery_hint_mapping(self):
        self.assertEqual(recovery_hint_for(FailureClass.TARGET_MISMATCH), RecoveryHint.FIX_TARGET)
        self.assertEqual(recovery_hint_for(FailureClass.CAPABILITY_FAILURE), RecoveryHint.RECOVER_CAPABILITY)
        self.assertEqual(recovery_hint_for(FailureClass.STRATEGY_FAILURE), RecoveryHint.REPLAN)
        self.assertEqual(recovery_hint_for(FailureClass.PROVIDER_FAILURE), RecoveryHint.ALTERNATIVE_PROVIDER)
        self.assertEqual(recovery_hint_for(FailureClass.NETWORK_FAILURE), RecoveryHint.RETRY)

    def test_supervisor_detailed_classification_backward_compatible(self):
        # The Phase-4 string API is preserved; the Phase-5 taxonomy is additive.
        sup = Supervisor("m")
        self.assertEqual(sup.classify_failure(_R(status="TIMEOUT")), "timeout")   # unchanged
        self.assertEqual(sup.classify_failure_detailed(_R(status="TIMEOUT")), FailureClass.TIMEOUT)

# =========================================================================== #
# §18/§19 — Memory retrieval (bounded, advisory) & confidence                  #
# =========================================================================== #

class TestMemoryRetrieval(Phase5Base):

    def test_memory_produces_advisory_candidate_with_confidence(self):
        mem = FakeMemory([FakeRetrievedMemory("file upload extension mismatch", kind="experience",
                                              strategy="upload .php as .jpg", confidence=0.9,
                                              success_rate=1.0, category="web")])
        gen = CandidateGenerator(memory_retriever=mem)
        ms = self._ms(category="web")
        cands = gen.generate(ms, use_memory=True)
        mem_cands = [c for c in cands if c.source == "memory"]
        self.assertEqual(mem.calls, 1)                             # used the retriever (bounded)
        self.assertTrue(mem_cands)
        c = mem_cands[0]
        self.assertEqual(c.action_type, "web_exploit")            # technique mapped
        # §19 — high historical confidence lifts success_probability but never to certainty.
        self.assertGreater(c.success_probability, 0.5)
        self.assertLess(c.success_probability, 1.0)

    def test_low_confidence_memory_scores_lower_than_high(self):
        gen_hi = CandidateGenerator(memory_retriever=FakeMemory(
            [FakeRetrievedMemory("sqli", strategy="union select", confidence=0.95, success_rate=1.0)]))
        gen_lo = CandidateGenerator(memory_retriever=FakeMemory(
            [FakeRetrievedMemory("sqli", strategy="union select", confidence=0.3, success_rate=0.4)]))
        ms = self._ms(category="web")
        hi = [c for c in gen_hi.generate(ms) if c.source == "memory"][0]
        lo = [c for c in gen_lo.generate(ms) if c.source == "memory"][0]
        self.assertGreater(hi.success_probability, lo.success_probability)

# =========================================================================== #
# §20/§21 — Playbook adaptation (not blind execution)                          #
# =========================================================================== #

class TestPlaybookAdaptation(Phase5Base):

    def test_playbook_technique_adapted_to_current_target(self):
        # A SQLi playbook (distinct from the generic recon/enum a bare web target already
        # implies) — its prose is mapped to an action type and adapted to THIS target.
        mem = FakeMemory([FakeRetrievedMemory("SSTI template injection", kind="playbook",
                                              strategy="render a malicious template", confidence=0.8)])
        gen = CandidateGenerator(memory_retriever=mem)
        ms = self._ms(category="web", target="http://victim.ctf")
        cands = [c for c in gen.generate(ms) if c.source == "playbook"]
        self.assertTrue(cands)
        c = cands[0]
        self.assertEqual(c.action_type, "web_exploit")            # mapped from prose (§20)
        self.assertEqual(c.target, "http://victim.ctf")           # adapted to THIS target
        self.assertIn("ssti", c.objective.lower())

    def test_playbook_needing_blocked_capability_is_not_proposed(self):
        # §21 — a playbook whose capability is unavailable here must not be blindly run.
        mem = FakeMemory([FakeRetrievedMemory("ocr the image", kind="playbook",
                                              strategy="tesseract image.png", confidence=0.8)])
        gen = CandidateGenerator(memory_retriever=mem)
        sup = Supervisor("m5")
        ms = self._ms(category="forensics", target="/tmp/x.png")
        decision = sup.reason(ms, generator=gen, blocked_capabilities=["ocr"])
        self.assertFalse(any(c.capability == "ocr" for c in decision.ranked))

# =========================================================================== #
# §26 — No-progress detection                                                  #
# =========================================================================== #

class TestNoProgressDetection(Phase5Base):

    def test_knowledge_fingerprint_reflects_growth(self):
        ms = self._ms()
        fp0 = knowledge_fingerprint(ms)
        ms.endpoints.append("/admin")
        fp1 = knowledge_fingerprint(ms)
        self.assertNotEqual(fp0, fp1)

    def test_ledger_flags_stagnation_after_limit(self):
        ledger = ProgressLedger(stagnation_limit=3)
        ms = self._ms()
        # Step 1 makes progress; then 3 no-gain steps → stagnant.
        ms.endpoints.append("/a")
        self.assertTrue(ledger.record(ms))
        for _ in range(3):
            self.assertFalse(ledger.record(ms))
        self.assertTrue(ledger.is_stagnant())

    def test_ledger_resets_on_new_progress(self):
        ledger = ProgressLedger(stagnation_limit=3)
        ms = self._ms()
        ledger.record(ms)
        ledger.record(ms)                                          # no progress
        self.assertEqual(ledger.stagnant_steps, 1)
        ms.vulnerabilities.append("SQLi")                          # new knowledge
        ledger.record(ms)
        self.assertEqual(ledger.stagnant_steps, 0)                 # reset

# =========================================================================== #
# §27 — Mission budget                                                         #
# =========================================================================== #

class TestMissionBudget(Phase5Base):

    def test_unbounded_by_default(self):
        b = MissionBudget()
        for _ in range(100):
            b.record_agent_call()
        done, _ = b.exhausted()
        self.assertFalse(done)                                     # 0 cap = unbounded
        self.assertEqual(b.pressure(), 0.0)

    def test_exhaustion_and_pressure(self):
        b = MissionBudget(max_agent_calls=3)
        b.record_agent_call(); b.record_agent_call()
        self.assertFalse(b.exhausted()[0])
        self.assertAlmostEqual(b.pressure(), 2 / 3, places=3)
        b.record_agent_call()
        self.assertTrue(b.exhausted()[0])

    def test_budget_roundtrips(self):
        b = MissionBudget(max_tool_executions=10)
        b.record_tool_executions(4)
        b2 = MissionBudget.from_dict(b.to_dict())
        self.assertEqual(b2.tool_executions, 4)
        self.assertEqual(b2.max_tool_executions, 10)

# =========================================================================== #
# §25 — Stop conditions                                                        #
# =========================================================================== #

class TestStopConditions(Phase5Base):

    def test_flag_verified_wins(self):
        ms = self._ms(verified_flag="picoCTF{x}")
        cond, _ = evaluate_stop(ms)
        self.assertEqual(cond, StopCondition.FLAG_VERIFIED)
        self.assertEqual(cond.final_status, "COMPLETED")

    def test_budget_exhausted_stop(self):
        b = MissionBudget(max_agent_calls=1)
        b.record_agent_call()
        cond, _ = evaluate_stop(self._ms(), budget=b)
        self.assertEqual(cond, StopCondition.MISSION_BUDGET_EXHAUSTED)
        self.assertEqual(cond.final_status, "FAILED")

    def test_no_progress_stop(self):
        ledger = ProgressLedger(stagnation_limit=2)
        ms = self._ms()
        ledger.record(ms); ledger.record(ms); ledger.record(ms)
        cond, _ = evaluate_stop(ms, ledger=ledger)
        self.assertEqual(cond, StopCondition.NO_PROGRESS)

    def test_capability_blocked_stop(self):
        cond, _ = evaluate_stop(self._ms(), all_capabilities_blocked=True)
        self.assertEqual(cond, StopCondition.CAPABILITY_BLOCKED)

    def test_target_blocked_stop(self):
        cond, _ = evaluate_stop(self._ms(), target_blocked=True)
        self.assertEqual(cond, StopCondition.TARGET_BLOCKED)

    def test_no_stop_while_productive(self):
        cond, _ = evaluate_stop(self._ms(), has_open_work=True)
        self.assertEqual(cond, StopCondition.NONE)

# =========================================================================== #
# §24 — Supervisor central reasoning                                           #
# =========================================================================== #

class TestSupervisorReasoning(Phase5Base):

    def test_reason_returns_ranked_decision_and_records_candidates(self):
        sup = Supervisor("m5")
        ms = self._ms(category="web", target="http://target.ctf")
        decision = sup.reason(ms, use_memory=False)
        self.assertTrue(decision.has_action)
        self.assertTrue(decision.ranked)
        self.assertEqual(decision.ranked[0].signature, decision.selected.signature)
        # §37 — the ranked candidates are stashed on the mission for observability.
        self.assertTrue(ms.candidate_actions)

    def test_reason_mode_explore_vs_exploit(self):
        sup = Supervisor("m5")
        explore = sup.reason(self._ms(target="http://t.ctf"), use_memory=False)
        self.assertEqual(explore.mode, "explore")                  # nothing known yet
        exploit = sup.reason(self._ms(target="http://t.ctf", vulnerabilities=["SQLi in /login"]),
                             use_memory=False)
        self.assertEqual(exploit.mode, "exploit")                  # strong evidence present

# =========================================================================== #
# §29–§33 — Reasoning regressions                                              #
# =========================================================================== #

class TestReasoningRegressions(Phase5Base):

    def test_29_binary_digits_recognizes_ocr_then_does_not_repeat(self):
        # binary → decode → JPEG → image likely holds hidden text → OCR is the required
        # next capability. Recognise it; once it is known-unavailable, do NOT repeat it.
        gen = CandidateGenerator()
        ms = self._ms(category="forensics", target="/tmp/secret.jpg",
                      objective="Extract the hidden text from the image to reveal the flag",
                      artifacts=["secret.jpg"])
        img = Evidence(mission_id="m5", evidence_type="artifact", title="secret.jpg",
                       artifact_id="secret.jpg", source="command")
        cands = gen.generate(ms, recent_evidence=[img], use_memory=False)
        ocr = next((c for c in cands if c.action_type == "ocr_extract"), None)
        self.assertIsNotNone(ocr, "should recognise OCR as the required next capability")
        self.assertEqual(ocr.capability, "ocr")

        # OCR turns out to be unavailable → record it blocked/failed, then re-reason.
        ms.record_failed_approach(action="ocr the image", signature=ocr.signature,
                                  capability="ocr", result="BLOCKED",
                                  failure_class=FailureClass.CAPABILITY_FAILURE.value)
        sup = Supervisor("m5")
        decision = sup.reason(ms, recent_evidence=[img], use_memory=False,
                              blocked_capabilities=["ocr"])
        self.assertFalse(any(c.capability == "ocr" for c in decision.ranked))  # not repeated
        self.assertTrue(any(c.action_type == "artifact_analysis" for c in decision.ranked))  # adapt

    async def test_30_flag_hunters_target_mismatch_blocks_without_endless_guessing(self):
        # source code → the challenge needs LIVE interaction. If the provided target is a
        # static file, that is a terminal TARGET_MISMATCH (block once), NOT endless retry.
        coord = self._coord(target="https://challenge-files.picoctf.net/x/source.py",
                            category="pwn")
        coord.scheduler.add(Task(mission_id=coord.mission_id, role="pwn",
                                 objective="Connect to the service and send the winning input",
                                 target_type="LIVE_TCP"))
        coord.mission.status = "RUNNING"
        await coord.run()
        t = coord.scheduler.all()[0]
        self.assertEqual(t.status, TaskStatus.FAILED.value)
        self.assertIn("TARGET_MISMATCH", t.failure_reason)
        # Recorded as a failed approach so it is never retried (no endless guessing).
        self.assertTrue(any(fa.failure_class == "TARGET_MISMATCH"
                            for fa in coord.mission.get_failed_approaches()))
        self.assertEqual(t.retry_count, 0)

    def test_30b_live_target_yields_interactive_candidate(self):
        # Target-type awareness: a live TCP target surfaces an interactive/service action.
        gen = CandidateGenerator()
        ms = self._ms(category="pwn", target="nc target.ctf 1337", target_type="LIVE_TCP")
        cands = gen.generate(ms, use_memory=False)
        self.assertTrue(any(c.action_type in ("service_fingerprint", "interactive_probe",
                                              "port_scan") for c in cands))

    async def test_31_action_A_fails_B_reveals_evidence_C_justified(self):
        # A (blind web probe) → nothing; B (recon) → reveals a technology; C (targeted at
        # that technology) becomes justified. A must not be repeated.
        coord = self._coord(category="web", target="http://target.ctf")
        executed = []

        def responder(role, task):
            obj = task.objective.lower()
            if role == "recon":
                ev = [Evidence(mission_id=task.mission_id, evidence_type="technology",
                               title="CustomCMS 3.1", related_technology="CustomCMS 3.1",
                               source="command", confidence=0.9)]
                return AgentResult(task_id=task.id, role=role, status="COMPLETED", evidence=ev,
                                   reason="fingerprinted CustomCMS 3.1")
            if "customcms" in obj:
                return AgentResult(task_id=task.id, role=role, status="COMPLETED",
                                   reason="exploited CustomCMS")
            return AgentResult(task_id=task.id, role=role, status="FAILED",
                               reason="no result", failure_category="NO_RESULT")

        coord.agent_factory = lambda role: ProgrammableAgent(role, responder, executed)
        coord.scheduler.add(Task(mission_id=coord.mission_id, role="recon",
                                 objective="Fingerprint the server", priority=90))
        coord.scheduler.add(Task(mission_id=coord.mission_id, role="web",
                                 objective="Blindly probe the endpoint '/A'", priority=50))
        await coord.run()
        web_objs = [o for (r, o) in executed if r == "web"]
        self.assertTrue(any("customcms" in o.lower() for o in web_objs),
                        f"C (CustomCMS-targeted) should run; executed: {executed}")
        self.assertEqual(sum(1 for o in web_objs if "'/A'" in o or "/a" in o.lower()), 1)  # A once

    async def test_33_adaptive_planning_discards_A_prioritizes_internal(self):
        # Initial plan enumerates endpoint A → 404. robots.txt reveals /internal → the
        # planner must pursue /internal and NOT keep re-running A.
        coord = self._coord(category="web", target="http://target.ctf")
        executed = []

        def responder(role, task):
            obj = task.objective.lower()
            if role == "recon":
                ev = [Evidence(mission_id=task.mission_id, evidence_type="endpoint",
                               title="/internal", related_endpoint="http://target.ctf/internal",
                               source="command", confidence=0.9)]
                return AgentResult(task_id=task.id, role=role, status="COMPLETED", evidence=ev,
                                   reason="robots.txt reveals /internal")
            if "internal" in obj:
                return AgentResult(task_id=task.id, role=role, status="COMPLETED",
                                   reason="explored /internal")
            return AgentResult(task_id=task.id, role=role, status="FAILED",
                               reason="404", failure_category="NO_RESULT")

        coord.agent_factory = lambda role: ProgrammableAgent(role, responder, executed)
        coord.scheduler.add(Task(mission_id=coord.mission_id, role="recon",
                                 objective="Read robots.txt", priority=90))
        coord.scheduler.add(Task(mission_id=coord.mission_id, role="web",
                                 objective="Enumerate the endpoint '/A'", priority=50))
        await coord.run()
        web_objs = [o for (r, o) in executed if r == "web"]
        self.assertTrue(any("internal" in o.lower() for o in web_objs),
                        f"/internal should be pursued; executed: {executed}")
        self.assertEqual(sum(1 for o in web_objs if "'/A'" in o), 1)  # A not repeated
        self.assertTrue(any(fa.result == "FAILED" for fa in coord.mission.get_failed_approaches()))

# =========================================================================== #
# Integration path + §32 memory-does-not-auto-execute + §37 observability      #
# =========================================================================== #

class TestReasoningIntegration(Phase5Base):

    async def test_full_reasoning_loop_agent_to_replanning(self):
        # Agent → EvidenceBus → Supervisor → CandidateActions → Scheduler →
        # ExecutionResult → MissionState → Replanning, end to end.
        coord = self._coord(category="web", target="http://target.ctf")
        executed = []

        def responder(role, task):
            if role == "recon":
                ev = [Evidence(mission_id=task.mission_id, evidence_type="technology",
                               title="nginx 1.18", related_technology="nginx 1.18",
                               source="command", confidence=0.9)]
                return AgentResult(task_id=task.id, role=role, status="COMPLETED",
                                   evidence=ev, facts=["technology: nginx 1.18"],
                                   reason="recon done", tool_executions=2)
            return AgentResult(task_id=task.id, role=role, status="COMPLETED",
                               reason="done", tool_executions=1)

        coord.agent_factory = lambda role: ProgrammableAgent(role, responder, executed)
        coord.scheduler.add(Task(mission_id=coord.mission_id, role="recon",
                                 objective="Recon the host", priority=90))
        res = await coord.run()
        # Evidence became a fact in shared state; a follow-up task was created from it.
        self.assertTrue(any("nginx" in f.statement.lower() for f in coord.mission.get_facts()))
        self.assertTrue(any("nginx" in o.lower() for (_r, o) in executed if _r != "recon")
                        or coord.scheduler.total_count() > 1)
        # Budget accounted for the agent calls + tool executions.
        self.assertGreaterEqual(coord.budget.agent_calls, 1)
        self.assertGreaterEqual(coord.budget.tool_executions, 2)

    async def test_32_memory_is_not_auto_executed_without_evidence(self):
        # A strong historical experience is retrieved & presented (advisory), but is NOT
        # turned into an executed task unless CURRENT evidence justifies it.
        mem = FakeMemory([FakeRetrievedMemory("upload .php as .jpg (extension mismatch)",
                                              kind="experience", strategy="upload bypass",
                                              confidence=0.95, success_rate=1.0, category="web")])
        gen = CandidateGenerator(memory_retriever=mem)
        coord = self._coord(category="web", target="http://target.ctf",
                            candidate_generator=gen)
        executed = []

        def responder(role, task):
            # recon produces a benign endpoint (triggers a replan) but nothing about uploads.
            if role == "recon":
                ev = [Evidence(mission_id=task.mission_id, evidence_type="endpoint",
                               title="/status", related_endpoint="http://target.ctf/status",
                               source="command", confidence=0.9)]
                return AgentResult(task_id=task.id, role=role, status="COMPLETED", evidence=ev,
                                   reason="found /status")
            return AgentResult(task_id=task.id, role=role, status="COMPLETED", reason="done")

        coord.agent_factory = lambda role: ProgrammableAgent(role, responder, executed)
        coord.scheduler.add(Task(mission_id=coord.mission_id, role="recon",
                                 objective="Recon", priority=90))
        await coord.run()
        # The memory technique was RETRIEVED and presented as a candidate (advisory)…
        cand_types = [c.get("source") for c in coord.mission.candidate_actions]
        self.assertIn("memory", cand_types)
        # …but it was NOT auto-executed (no upload task ran) because no evidence justified it.
        self.assertFalse(any("upload" in o.lower() or "extension mismatch" in o.lower()
                             for (_r, o) in executed))

    async def test_37_snapshot_exposes_reasoning_state(self):
        coord = self._coord(category="web", target="http://target.ctf")
        snap = coord.snapshot()
        self.assertIn("reasoning", snap)
        r = snap["reasoning"]
        for key in ("uncertainty", "budget", "progress", "stop_condition",
                    "facts", "open_hypotheses", "failed_approaches"):
            self.assertIn(key, r)

    async def test_37_reasoning_decision_event_emitted(self):
        coord = self._coord(category="web", target="http://target.ctf")

        def responder(role, task):
            if role == "recon":
                ev = [Evidence(mission_id=task.mission_id, evidence_type="technology",
                               title="Apache 2.4.49", related_technology="Apache 2.4.49",
                               source="command", confidence=0.9)]
                return AgentResult(task_id=task.id, role=role, status="COMPLETED", evidence=ev,
                                   reason="found apache")
            return AgentResult(task_id=task.id, role=role, status="COMPLETED", reason="done")

        coord.agent_factory = lambda role: ProgrammableAgent(role, responder, [])
        coord.scheduler.add(Task(mission_id=coord.mission_id, role="recon",
                                 objective="Recon", priority=90))
        with _CaptureEvents() as cap:
            await coord.run()
        self.assertIn(swarm_events.REASONING_DECISION, cap.names())

if __name__ == "__main__":
    unittest.main()
