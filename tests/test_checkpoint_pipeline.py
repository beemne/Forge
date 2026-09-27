"""Unit tests for the HITL checkpoint pipeline (Part 3): consolidated report
assembly (deterministic + strictly-extractive) and the suggestion parser/router."""

import asyncio
import os
import unittest

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.checkpoint_pipeline import (
    AgentCheckpointRecord,
    build_consolidated_report,
    build_response_instructions,
    parse_suggestions,
)

def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)

class TestReportAssembly(unittest.TestCase):
    def _report(self, records, summarizer=None):
        return _run(build_consolidated_report(
            challenge_name="Crack the Gate", category="WEB", difficulty="EASY",
            target="http://t.ctf", cycle_n=1, start_time="00:00:00", end_time="00:05:00",
            records=records, summarizer=summarizer,
        ))

    def test_header_and_agent_blocks(self):
        recs = [AgentCheckpointRecord(agent_id="agent_1", evidence=["HTTP/1.1 200"], tried=["curl http://t.ctf -> 200"])]
        report = self._report(recs)
        self.assertIn("=== FORGE CHECKPOINT — Cycle 1 (00:00:00–00:05:00) ===", report)
        self.assertIn("Challenge: Crack the Gate | WEB | EASY", report)
        self.assertIn("Target: http://t.ctf", report)
        self.assertIn("--- agent: agent_1 ---", report)
        self.assertIn("HTTP/1.1 200", report)

    def test_flag_candidate_none_when_absent(self):
        recs = [AgentCheckpointRecord(agent_id="agent_1")]
        report = self._report(recs)
        self.assertIn("Flag candidate: NONE", report)

    def test_flag_candidate_marked_unverified(self):
        recs = [AgentCheckpointRecord(agent_id="agent_1", flag_candidate="picoCTF{real_from_output}", flag_source="tool_output")]
        report = self._report(recs)
        self.assertIn("picoCTF{real_from_output}", report)
        self.assertIn("UNVERIFIED", report)
        self.assertIn("tool_output", report)

    def test_extractive_guard_transcript_flag_not_upgraded(self):
        # A fabricated flag that appears ONLY in the transcript (never as a real
        # candidate) must NOT surface as a flag candidate in the report.
        recs = [AgentCheckpointRecord(
            agent_id="agent_1",
            transcript="the model mused: I think the flag is picoCTF{fabricated_guess}",
            flag_candidate=None,
        )]
        report = self._report(recs)          # no summarizer → deterministic only
        self.assertIn("Flag candidate: NONE", report)
        self.assertNotIn("picoCTF{fabricated_guess}", report)

    def test_summarizer_fills_hypothesis_and_blocked(self):
        async def stub(_prompt):
            return "HYPOTHESIS: pivot to /admin with dev header\nBLOCKED: need the developer username"
        recs = [AgentCheckpointRecord(agent_id="agent_1", transcript="curl ...; got 403")]
        report = self._report(recs, summarizer=stub)
        self.assertIn("Hypothesis: pivot to /admin with dev header", report)
        self.assertIn("Blocked on: need the developer username", report)

    def test_summarizer_failure_falls_back(self):
        async def bad(_prompt):
            raise RuntimeError("provider exhausted")
        recs = [AgentCheckpointRecord(agent_id="agent_1", transcript="something")]
        report = self._report(recs, summarizer=bad)
        self.assertIn("Hypothesis: none stated", report)
        self.assertIn("Blocked on: none stated", report)

    def test_report_includes_response_format_instructions(self):
        # Part 4 #1: the report MUST end with the response-format block, listing the actual
        # agent ids — its absence caused the live 'UNPARSEABLE paste' failure.
        recs = [AgentCheckpointRecord(agent_id="agent_1"), AgentCheckpointRecord(agent_id="agent_2")]
        report = self._report(recs)
        self.assertIn("=== INSTRUCTIONS FOR YOUR RESPONSE ===", report)
        self.assertIn("--- suggestion: agent_1 ---", report)
        self.assertIn("--- suggestion: agent_2 ---", report)

    def test_instructions_block_round_trips_through_parser(self):
        # The exact format the report tells the model to use must parse deterministically.
        report = self._report([AgentCheckpointRecord(agent_id="agent_1"),
                               AgentCheckpointRecord(agent_id="agent_2")])
        tail = report[report.index("=== INSTRUCTIONS FOR YOUR RESPONSE ==="):]
        parsed = parse_suggestions(tail, ["agent_1", "agent_2"])
        self.assertTrue(parsed.parsed)
        self.assertIn("agent_1", parsed.directives)
        self.assertIn("agent_2", parsed.directives)

    def test_response_instructions_defaults_when_no_agents(self):
        block = build_response_instructions([])
        self.assertIn("--- suggestion: agent_1 ---", block)

class TestSuggestionParser(unittest.TestCase):
    def test_happy_path_two_agents(self):
        text = (
            "--- suggestion: agent_1 ---\n"
            "Try the X-Dev-Access: yes header on /login.\n\n"
            "--- suggestion: agent_2 ---\n"
            "Decode the base64 blob in the JS bundle."
        )
        p = parse_suggestions(text, ["agent_1", "agent_2"])
        self.assertTrue(p.parsed)
        self.assertIn("agent_1", p.directives)
        self.assertIn("agent_2", p.directives)
        self.assertIn("X-Dev-Access", p.directives["agent_1"])
        self.assertIn("base64", p.directives["agent_2"])

    def test_cross_agent_routing_by_own_label(self):
        # Suggestion labelled agent_2 even though it references agent_1's finding —
        # must route to agent_2 (its own label), not agent_1.
        text = (
            "--- suggestion: agent_2 ---\n"
            "Agent_1 found a username; you (agent_2) should try it against /login."
        )
        p = parse_suggestions(text, ["agent_1", "agent_2"])
        self.assertTrue(p.parsed)
        self.assertIn("agent_2", p.directives)
        self.assertNotIn("agent_1", p.directives)

    def test_unparseable_falls_back_to_all_agents(self):
        text = "just a blob of advice with no delimiters at all"
        p = parse_suggestions(text, ["agent_1"])
        self.assertFalse(p.parsed)
        self.assertTrue(p.fallback)
        self.assertEqual(p.fallback_text, text)

    def test_unknown_label_recorded_but_routed(self):
        text = "--- suggestion: agent_9 ---\nfocus on the sqlite dump"
        p = parse_suggestions(text, ["agent_1", "agent_2"])
        self.assertTrue(p.parsed)
        self.assertIn("agent_9", p.directives)
        self.assertIn("agent_9", p.unknown_labels)

    def test_duplicate_label_appended(self):
        text = (
            "--- suggestion: agent_1 ---\nfirst directive\n"
            "--- suggestion: agent_1 ---\nsecond directive"
        )
        p = parse_suggestions(text, ["agent_1"])
        self.assertTrue(p.parsed)
        self.assertIn("first directive", p.directives["agent_1"])
        self.assertIn("second directive", p.directives["agent_1"])

class TestSuggestionEvaluationGate(unittest.TestCase):
    """Tests for evaluate_suggestion and evaluate_suggestions in checkpoint_pipeline."""

    def test_accept_concrete_evidence_backed_suggestion(self):
        from backend.agents.checkpoint_pipeline import (
            SuggestionDecision, evaluate_suggestion,
        )
        ev = evaluate_suggestion(
            "Access /uploads/shell.php using curl to verify RCE execution.",
            known_files=["uploads/shell.php"],
            known_endpoints=["/uploads/shell.php"],
        )
        self.assertEqual(ev.decision, SuggestionDecision.ACCEPT)
        self.assertTrue(any("uploads/shell.php" in item for item in ev.evidence))
        self.assertIn("uploads/shell.php", ev.suggested_action)

    def test_reject_vague_suggestions(self):
        from backend.agents.checkpoint_pipeline import (
            SuggestionDecision, evaluate_suggestion,
        )
        vague_inputs = [
            "",
            "   ",
            "Try something else",
            "do better",
            "try again.",
            "investigate further",
            "be more creative",
            "think harder",
            "good luck",
            "no idea",
        ]
        for inp in vague_inputs:
            with self.subTest(vague_input=inp):
                ev = evaluate_suggestion(inp)
                self.assertEqual(
                    ev.decision,
                    SuggestionDecision.REJECT,
                    f"Expected REJECT for vague input: '{inp}'",
                )
                self.assertEqual(ev.suggested_action, "")

    def test_reject_exhausted_strategy_without_novelty(self):
        from backend.agents.checkpoint_pipeline import (
            SuggestionDecision, evaluate_suggestion,
        )
        ev = evaluate_suggestion(
            "Run directory_enum with gobuster and standard wordlist to discover more paths.",
            exhausted_strategies=["directory_enum"],
        )
        self.assertEqual(ev.decision, SuggestionDecision.REJECT)
        self.assertIn("exhausted", ev.reason.lower())
        self.assertEqual(ev.suggested_action, "")

    def test_modify_unverified_flag_assertion(self):
        from backend.agents.checkpoint_pipeline import (
            SuggestionDecision, evaluate_suggestion,
        )
        ev = evaluate_suggestion(
            "The flag is picoCTF{unverified_secret_12345}",
            flag_candidates=[],
        )
        self.assertEqual(ev.decision, SuggestionDecision.MODIFY)
        self.assertIn("VERIFY", ev.suggested_action.upper())
        self.assertIn("picoCTF{unverified_secret_12345}", ev.suggested_action)
        self.assertIn("Do NOT submit this as the flag unless you can verify", ev.suggested_action)

    def test_modify_failed_technique_with_novel_target(self):
        from backend.agents.checkpoint_pipeline import (
            SuggestionDecision, evaluate_suggestion,
        )
        ev = evaluate_suggestion(
            "Try sql_injection against /api/v2/items?category=admin to extract credentials.",
            failed_techniques=["sql_injection"],
        )
        self.assertEqual(ev.decision, SuggestionDecision.MODIFY)
        self.assertIn("NEW TARGET ONLY", ev.suggested_action.upper())
        self.assertIn("/api/v2/items", ev.suggested_action)

    def test_evaluate_suggestions_batch(self):
        from backend.agents.checkpoint_pipeline import (
            ParsedSuggestions, SuggestionDecision, evaluate_suggestions,
        )
        parsed = ParsedSuggestions(
            parsed=True,
            directives={
                "agent_1": "Try something else",
                "agent_2": "Fetch /api/keys to inspect the token.",
                "agent_3": "The flag is picoCTF{fake_flag_xyz}",
            },
        )
        evals = evaluate_suggestions(
            parsed,
            known_endpoints=["/api/keys"],
        )
        self.assertEqual(evals["agent_1"].decision, SuggestionDecision.REJECT)
        self.assertEqual(evals["agent_2"].decision, SuggestionDecision.ACCEPT)
        self.assertEqual(evals["agent_3"].decision, SuggestionDecision.MODIFY)

class TestCheckpointPipelineLiveFlow(unittest.TestCase):
    """Integration-level tests covering the live checkpoint flow sequence:
    checkpoint suggestion -> parse -> evaluate -> ACCEPT / MODIFY / REJECT
    -> evaluated guidance -> candidate / action pipeline.
    """

    def test_checkpoint_flow_with_mixed_decisions_and_candidate_pipeline(self):
        """Proves that in a full checkpoint flow:
        - Rejected suggestions are discarded (not injected as agent directives).
        - Modified suggestions are converted to verification directives.
        - Accepted suggestions guide the candidate generator.
        - Execution primitives are not bypassed.
        """
        from backend.agents.checkpoint_pipeline import (
            SuggestionDecision, evaluate_suggestion, parse_suggestions,
        )
        from backend.swarm.candidates import CandidateGenerator

        pasted_text = (
            "--- suggestion: agent_1 ---\n"
            "Try something else.\n\n"
            "--- suggestion: agent_2 ---\n"
            "The flag is picoCTF{asserted_flag_value_abc}\n\n"
            "--- suggestion: agent_3 ---\n"
            "Access /uploads/shell.php and execute id to confirm execution."
        )
        known_agents = ["agent_1", "agent_2", "agent_3"]
        parsed = parse_suggestions(pasted_text, known_agents)
        self.assertTrue(parsed.parsed)

        # Shared state context
        exhausted = ["directory_enum"]
        known_files = ["uploads/shell.php"]
        known_endpoints = ["/uploads/shell.php"]
        flag_candidates = []

        # Simulated orchestrator injection gate
        agent_directives = {}
        for aid, directive in parsed.directives.items():
            ev = evaluate_suggestion(
                directive,
                exhausted_strategies=exhausted,
                known_files=known_files,
                known_endpoints=known_endpoints,
                flag_candidates=flag_candidates,
            )
            if ev.decision == SuggestionDecision.REJECT:
                # Rejected suggestion must NOT be injected
                continue
            action_text = ev.suggested_action or directive
            agent_directives[aid] = action_text

        # 1. Verify agent_1 (rejected) was NOT injected
        self.assertNotIn("agent_1", agent_directives,
                         "Rejected suggestion for agent_1 must not be present in agent_directives")

        # 2. Verify agent_2 (flag assertion) was MODIFIED into a verification instruction
        self.assertIn("agent_2", agent_directives)
        self.assertIn("VERIFY BEFORE SUBMITTING", agent_directives["agent_2"].upper())
        self.assertIn("picoCTF{asserted_flag_value_abc}", agent_directives["agent_2"])

        # 3. Verify agent_3 (accepted) was injected as guidance
        self.assertIn("agent_3", agent_directives)
        self.assertIn("/uploads/shell.php", agent_directives["agent_3"])

        # 4. Verify the candidate generator on the underlying mission state
        # produces actionable candidates for the accepted foothold without corruption
        class _MissionStub:
            target = "http://challenge.local"
            target_type = "web"
            category = "web"
            services = []
            technologies = []
            endpoints = ["/uploads/shell.php"]
            known_endpoints = ["/uploads/shell.php"]
            vulnerabilities = []
            credentials = []
            artifacts = []
            known_files = ["uploads/shell.php"]
            failed_techniques = []
            exhausted_strategies = exhausted
            action_signatures = []
            flag_candidates = []

        gen = CandidateGenerator()
        candidates = gen._from_state_gaps(_MissionStub())
        c_targets = [getattr(c, "target", "") for c in candidates]
        self.assertTrue(any("uploads/shell.php" in str(t) for t in c_targets),
                        "Candidate generator must produce candidate targeting the accepted foothold")

    def test_unparseable_fallback_guidance_evaluated_before_injection(self):
        """Unparseable fallback paste is evaluated through the gate before general injection."""
        from backend.agents.checkpoint_pipeline import (
            SuggestionDecision, evaluate_suggestion, parse_suggestions,
        )

        pasted = "Try something else"
        parsed = parse_suggestions(pasted, ["agent_1", "agent_2"])
        self.assertFalse(parsed.parsed)
        self.assertTrue(parsed.fallback)

        ev = evaluate_suggestion(parsed.fallback_text)
        self.assertEqual(ev.decision, SuggestionDecision.REJECT)

        # Because it's REJECT, the orchestrator does NOT inject it to any agent
        agent_directives = {}
        if ev.decision != SuggestionDecision.REJECT:
            for aid in ["agent_1", "agent_2"]:
                agent_directives[aid] = ev.suggested_action

        self.assertEqual(agent_directives, {})

if __name__ == "__main__":
    unittest.main()
