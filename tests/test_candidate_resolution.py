"""
Tests for Candidate Resolution and Verification Pipeline.

Validates:
1. A generic answer candidate can be created from evidence.
2. Challenge semantics affect whether a candidate resolves.
3. A flag-shaped candidate is NOT automatically accepted merely because of its format.
4. A non-flag answer can be resolved.
5. A strong candidate reaches the resolver instead of remaining only a log message.
6. A successfully resolved answer causes the correct existing completion/termination behavior.
7. An unresolved candidate does not falsely mark the challenge solved.
8. Weak/noisy conflicting evidence does not automatically destroy a strong candidate.
9. The live competition execution path actually exercises the resolver.
10. Backward compatibility with FlagVerifier / FlagVerdict.
11. Regression Tests A through H (Traditional flag, username, hash, number, distractor, vision, no fake verification, termination).
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agent_runtime.verifier import (
    AnswerCandidate,
    AnswerResolver,
    AnswerSource,
    AnswerStatus,
    AnswerType,
    AnswerVerdict,
    FlagSource,
    FlagStatus,
    FlagVerdict,
    FlagVerifier,
    VerifierAgent,
    infer_expected_answer_type,
)
from backend.agents.swarm_orchestrator import SwarmOrchestrator
from backend.agents.swarm_state import SwarmBlackboard
from backend.database.models import ChallengeModel, EvidenceModel, RunModel
from backend.database.session import SessionLocal, init_db
from backend.swarm.coordinator import SwarmCoordinator

class TestCandidateResolution(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        init_db()
        self.resolver = AnswerResolver()
        self.verifier_agent = VerifierAgent(resolver=self.resolver)

    def test_01_generic_answer_candidate_creation_from_evidence(self):
        """1. A generic answer candidate can be created from evidence with full metadata."""
        candidate = AnswerCandidate(
            value="picoCTF{valid_evidence_candidate_123}",
            answer_type=AnswerType.FLAG,
            source=AnswerSource.VISION_READ,
            confidence=0.95,
            worker_id="agent-0",
            evidence={"derived_path": "/tmp/reconstructed.jpg", "tool": "vision_read"},
            task_context={"description": "Find the flag inside the image", "category": "forensics"},
            provenance={"worker_id": "agent-0", "action_succeeded": True},
        )
        d = candidate.to_dict()
        self.assertEqual(d["value"], "picoCTF{valid_evidence_candidate_123}")
        self.assertEqual(d["source"], "vision_read")
        self.assertEqual(d["evidence"]["tool"], "vision_read")
        self.assertEqual(d["status"], "FLAG_CANDIDATE")

    def test_02_challenge_semantics_affect_resolution(self):
        """2. Challenge semantics affect whether a candidate resolves."""
        # When challenge asks for a username:
        user_cand = AnswerCandidate(
            value="admin_user_99",
            source=AnswerSource.TOOL_OUTPUT,
            worker_id="agent-0",
            task_context={"description": "What is the hidden username?", "category": "web"},
            provenance={"action_succeeded": True},
        )
        verdict_user = self.resolver.resolve(user_cand)
        self.assertEqual(verdict_user.answer_type, AnswerType.USERNAME)
        self.assertTrue(verdict_user.is_resolved)

        # When challenge asks for a SHA256 hash:
        hash_cand = AnswerCandidate(
            value="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            source=AnswerSource.TOOL_OUTPUT,
            worker_id="agent-0",
            task_context={"description": "Calculate the sha256 hash of the payload", "category": "crypto"},
            provenance={"action_succeeded": True},
        )
        verdict_hash = self.resolver.resolve(hash_cand)
        self.assertEqual(verdict_hash.answer_type, AnswerType.HASH)
        self.assertTrue(verdict_hash.is_resolved)

    def test_03_flag_candidate_not_accepted_for_non_flag_question(self):
        """3. A flag-shaped candidate is NOT automatically accepted when the challenge asks for a username or hash."""
        flag_for_username = AnswerCandidate(
            value="picoCTF{not_a_username_at_all}",
            source=AnswerSource.TOOL_OUTPUT,
            worker_id="agent-0",
            task_context={"description": "What is the hidden admin username?", "category": "web"},
            provenance={"action_succeeded": True},
        )
        verdict = self.resolver.resolve(flag_for_username)
        self.assertEqual(verdict.status, AnswerStatus.REJECTED)
        self.assertFalse(verdict.is_resolved)
        self.assertIn("username", verdict.reasons[0].lower())

    def test_04_non_flag_answer_can_be_resolved(self):
        """4. A non-flag answer (e.g. port number or secret key) can be resolved."""
        port_cand = AnswerCandidate(
            value="8080",
            source=AnswerSource.TOOL_OUTPUT,
            worker_id="agent-0",
            task_context={"description": "Which port number is running the hidden service?", "category": "recon"},
            provenance={"action_succeeded": True},
        )
        verdict = self.resolver.resolve(port_cand)
        self.assertEqual(verdict.answer_type, AnswerType.NUMBER)
        self.assertTrue(verdict.is_resolved)
        self.assertEqual(verdict.candidate, "8080")

    async def test_05_strong_candidate_reaches_resolver_and_verifies_on_blackboard(self):
        """5. A strong candidate reaches the resolver instead of remaining only a log message.

        Simulates the exact Binary Digits regression:
        Derived JPEG analyzed via vision_read -> candidate recorded on SwarmBlackboard ->
        resolver marks it verified/resolved -> board.flag_captured is set.
        """
        board = SwarmBlackboard(
            challenge_id="ch-binary-digits",
            run_id="run-binary-digits",
            target_scope="local",
            description="Reconstruct the JPEG image from binary digits and find the flag.",
            category="forensics",
        )
        board.verifier_agent.router = None  # deterministic-only for unit tests

        # Candidate produced by vision_read on reconstructed artifact
        await board.record_flag_candidate(
            candidate="picoCTF{binary_digits_reconstructed_jpeg_flag}",
            worker_id="agent-recon",
            source="vision_read",
            evidence={"derived_path": "/tmp/derived_3be06099e5cf.jpg", "tool": "vision_read"},
        )

        # MUST be captured and flagged immediately
        self.assertEqual(board.flag_captured, "picoCTF{binary_digits_reconstructed_jpeg_flag}")
        self.assertTrue(board.flag_event.is_set())
        self.assertEqual(len(board.flag_candidates), 1)
        self.assertEqual(board.flag_candidates[0]["status"], "RESOLVED")

    async def test_06_resolved_answer_causes_completion_termination(self):
        """6. A successfully resolved answer causes swarm completion/termination."""
        board = SwarmBlackboard(
            challenge_id="ch-solve-test",
            run_id="run-solve-test",
            target_scope="local",
            description="Extract the flag from the binary.",
            category="rev",
        )
        board.verifier_agent.router = None  # deterministic-only for unit tests

        await board.record_flag_candidate(
            candidate="FLAG{solved_via_tool_output}",
            worker_id="worker-1",
            source="tool_output",
            evidence={"command": "strings target_bin"},
        )

        self.assertEqual(board.flag_captured, "FLAG{solved_via_tool_output}")
        self.assertTrue(board.flag_event.is_set())

    async def test_07_unresolved_candidate_does_not_falsely_solve_challenge(self):
        """7. An unresolved candidate (e.g. LLM prose claim or placeholder) does not mark challenge solved."""
        board = SwarmBlackboard(
            challenge_id="ch-unresolved-test",
            run_id="run-unresolved-test",
            target_scope="local",
            description="Find the flag.",
            category="crypto",
        )
        board.verifier_agent.router = None  # deterministic-only for unit tests

        # Model prose claim (unverified)
        await board.record_flag_candidate(
            candidate="picoCTF{model_speculation_only}",
            worker_id="agent-0",
            source="llm_prose",
        )

        # Stays candidate, NOT solved
        self.assertIsNone(board.flag_captured)
        self.assertFalse(board.flag_event.is_set())
        self.assertEqual(len(board.flag_candidates), 1)
        self.assertEqual(board.flag_candidates[0]["status"], "FLAG_CANDIDATE")

        # Placeholder rejected completely
        await board.record_flag_candidate(
            candidate="picoCTF{...}",
            worker_id="agent-0",
            source="llm_prose",
        )
        self.assertIsNone(board.flag_captured)

    async def test_08_noisy_conflicting_evidence_does_not_destroy_strong_candidate(self):
        """8. Weak/noisy conflicting evidence does not destroy a strong candidate."""
        board = SwarmBlackboard(
            challenge_id="ch-noise-test",
            run_id="run-noise-test",
            target_scope="local",
            description="Find the flag.",
            category="web",
        )
        board.verifier_agent.router = None  # deterministic-only for unit tests

        # 1. High confidence evidence candidate
        await board.record_flag_candidate(
            candidate="picoCTF{real_verified_flag}",
            worker_id="agent-recon",
            source="tool_output",
        )
        self.assertEqual(board.flag_captured, "picoCTF{real_verified_flag}")

        # 2. Subsequent noisy or broken candidate from another agent
        await board.record_flag_candidate(
            candidate="picoCTF{fake_noisy_hallucination}",
            worker_id="agent-noisy",
            source="llm_prose",
        )

        # Verified flag is preserved
        self.assertEqual(board.flag_captured, "picoCTF{real_verified_flag}")

    def test_09_competition_harness_path_exercises_resolver(self):
        """9. The competition execution path exercises the AnswerResolver."""
        from backend.agents.orchestrator_loop import orchestrator_loop
        db = SessionLocal()
        try:
            ch = ChallengeModel(name="Live Test Challenge", category="web", description="Find the flag")
            db.add(ch)
            db.commit()

            run = RunModel(challenge_id=ch.id, status="RUNNING", current_phase="recon", current_agent="orchestrator")
            db.add(run)
            db.commit()

            res = asyncio.run(orchestrator_loop.execute_run_step(db, run.id))
            self.assertEqual(res["status"], "RUNNING")

            # Check that evidence was recorded in DB
            evs = db.query(EvidenceModel).filter(EvidenceModel.challenge_id == ch.id).all()
            self.assertGreater(len(evs), 0)

            # Test resolver on orchestrator_loop
            verdict = orchestrator_loop.answer_resolver.assess(
                "picoCTF{live_harness_test}",
                source="tool_output",
                description=ch.description,
                category=ch.category,
            )
            self.assertTrue(verdict.is_resolved)
        finally:
            db.close()

    def test_10_backward_compatibility_with_flag_verifier(self):
        """10. Existing FlagVerifier and FlagVerdict interfaces remain compatible."""
        fv = FlagVerifier()
        verdict = fv.assess("picoCTF{compat_test_1234}", source=FlagSource.TOOL_OUTPUT)
        self.assertTrue(verdict.is_resolved)
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.candidate, "picoCTF{compat_test_1234}")

    # =========================================================================
    # FOCUSED REGRESSION TESTS (A - H)
    # =========================================================================

    def test_A_traditional_flag(self):
        """Test A — traditional flag: Evidence contains a flag-shaped answer."""
        verdict = self.verifier_agent.verify_sync(
            "FLAG{synthetic_traditional_flag_abc}",
            task_context={"description": "Find the secret flag in the service", "category": "web"},
            source=AnswerSource.TOOL_OUTPUT,
            action_succeeded=True,
        )
        self.assertTrue(verdict.is_resolved)
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.answer_type, AnswerType.FLAG)
        self.assertEqual(verdict.candidate, "FLAG{synthetic_traditional_flag_abc}")

    def test_B_username_answer(self):
        """Test B — username: Challenge asks for username; evidence has NO flag string."""
        evidence_text = "Login successful for user: shadow_operator_42"
        extracted = self.resolver.extract_candidates(
            evidence_text,
            task_context={"description": "What username is exposed on the dashboard?", "category": "web"},
            source=AnswerSource.TOOL_OUTPUT,
        )
        self.assertGreater(len(extracted), 0)
        user_cand = extracted[0]
        self.assertEqual(user_cand.value, "shadow_operator_42")

        verdict = self.verifier_agent.verify_sync(
            user_cand,
            task_context={"description": "What username is exposed on the dashboard?", "category": "web"},
        )
        self.assertTrue(verdict.is_resolved)
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.answer_type, AnswerType.USERNAME)
        self.assertEqual(verdict.candidate, "shadow_operator_42")

    def test_C_hash_answer(self):
        """Test C — hash: Challenge asks for SHA256; evidence contains hash, NO flag format."""
        sha256_val = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
        evidence_text = f"Result payload hash computed: {sha256_val}"
        extracted = self.resolver.extract_candidates(
            evidence_text,
            task_context={"description": "What is the SHA256 hash of the payload?", "category": "crypto"},
            source=AnswerSource.TOOL_OUTPUT,
        )
        self.assertGreater(len(extracted), 0)
        hash_cand = extracted[0]
        self.assertEqual(hash_cand.value, sha256_val)

        verdict = self.verifier_agent.verify_sync(
            hash_cand,
            task_context={"description": "What is the SHA256 hash of the payload?", "category": "crypto"},
        )
        self.assertTrue(verdict.is_resolved)
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.answer_type, AnswerType.HASH)
        self.assertEqual(verdict.candidate, sha256_val)

    def test_D_number_answer(self):
        """Test D — number: Challenge asks for port/number; evidence has number, NO flag format."""
        evidence_text = "Hidden backdoor listening on port: 31337"
        extracted = self.resolver.extract_candidates(
            evidence_text,
            task_context={"description": "What port number is open for the admin service?", "category": "recon"},
            source=AnswerSource.TOOL_OUTPUT,
        )
        self.assertGreater(len(extracted), 0)
        num_cand = extracted[0]
        self.assertEqual(num_cand.value, "31337")

        verdict = self.verifier_agent.verify_sync(
            num_cand,
            task_context={"description": "What port number is open for the admin service?", "category": "recon"},
        )
        self.assertTrue(verdict.is_resolved)
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.answer_type, AnswerType.NUMBER)
        self.assertEqual(verdict.candidate, "31337")

    def test_E_flag_shaped_distractor(self):
        """Test E — flag-shaped distractor: Evidence has flag format that does NOT answer challenge question."""
        # Challenge specifically asks for a username
        distractor_flag = "picoCTF{this_is_a_distractor_not_a_username}"
        verdict = self.verifier_agent.verify_sync(
            distractor_flag,
            task_context={"description": "What username is exposed in the database dump?", "category": "web"},
            source=AnswerSource.TOOL_OUTPUT,
        )
        # Verifier must NOT blindly accept flag when username is asked
        self.assertEqual(verdict.status, AnswerStatus.REJECTED)
        self.assertFalse(verdict.is_resolved)

    async def test_F_vision_reconstructed_evidence(self):
        """Test F — vision/reconstructed evidence: reaches the exact same resolver/verifier pipeline."""
        board = SwarmBlackboard(
            challenge_id="ch-vision-pipeline",
            run_id="run-vision-pipeline",
            target_scope="local",
            description="Reconstruct image and find flag.",
            category="forensics",
        )
        board.verifier_agent.router = None  # deterministic-only for unit tests
        await board.record_flag_candidate(
            candidate="FLAG{vision_reconstructed_pipeline_test}",
            worker_id="agent-recon",
            source="vision_read",
            evidence={"tool": "vision_read", "derived_path": "/tmp/test.png"},
        )
        self.assertEqual(board.flag_captured, "FLAG{vision_reconstructed_pipeline_test}")
        self.assertTrue(board.flag_event.is_set())
        self.assertEqual(len(board.flag_candidates), 1)
        self.assertEqual(board.flag_candidates[0]["status"], "RESOLVED")

    def test_G_no_fake_verification(self):
        """Test G — no fake verification: Strong evidence candidate without authoritative check is RESOLVED, not VERIFIED."""
        # Normal evidence source without authoritative submission/check
        verdict = self.verifier_agent.verify_sync(
            "FLAG{strong_evidence_candidate}",
            task_context={"description": "Find the flag", "category": "crypto"},
            source=AnswerSource.TOOL_OUTPUT,
            action_succeeded=True,
            authoritative=False,
        )
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertTrue(verdict.is_resolved)
        self.assertFalse(verdict.is_verified)  # Meaningful distinction preserved!

        # Explicitly authoritative verification
        auth_verdict = self.verifier_agent.verify_sync(
            "FLAG{strong_evidence_candidate}",
            task_context={"description": "Find the flag", "category": "crypto"},
            source=AnswerSource.TOOL_OUTPUT,
            action_succeeded=True,
            authoritative=True,
        )
        self.assertEqual(auth_verdict.status, AnswerStatus.VERIFIED)
        self.assertTrue(auth_verdict.is_verified)

    async def test_H_termination_on_resolved_answer(self):
        """Test H — termination: Live orchestration stops unnecessary additional iterations once answer is resolved."""
        board = SwarmBlackboard(
            challenge_id="ch-termination-test",
            run_id="run-termination-test",
            target_scope="local",
            description="Find the flag.",
            category="pwn",
        )
        board.verifier_agent.router = None  # deterministic-only for unit tests
        self.assertFalse(board.flag_event.is_set())
        self.assertIsNone(board.flag_captured)

        # Resolving an answer sets flag_captured and triggers flag_event
        await board.record_flag_candidate(
            candidate="FLAG{termination_confirmed}",
            worker_id="agent-pwn",
            source="tool_output",
        )
        self.assertEqual(board.flag_captured, "FLAG{termination_confirmed}")
        self.assertTrue(board.flag_event.is_set())

    async def test_12_verifier_agent_llm_structured_evaluation(self):
        """12. VerifierAgent with ModelRouter parses structured LLM output and validates evidence."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "RESOLVE", "confidence": 0.96, "answer_type": "flag", "is_distractor": false, "reasoning": "Output matches binary stdout exactly.", "evidence_soundness": "Valid execution output."}'
        )

        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{valid_flag_from_llm_verified}",
            source=AnswerSource.TOOL_OUTPUT,
            command="./vuln",
            action_succeeded=True,
            task_context={"description": "Exploit binary to get flag", "category": "pwn"},
        )

        mock_router.route_request.assert_awaited_once()
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.confidence, 0.96)
        self.assertTrue(any("Output matches binary stdout" in r for r in verdict.reasons))

    async def test_13_verifier_agent_trust_boundary_prose_cannot_verify(self):
        """13. Trust Boundary: Model prose cannot be promoted to RESOLVED even if LLM verifier recommends RESOLVE."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "RESOLVE", "confidence": 0.99, "answer_type": "flag", "is_distractor": false, "reasoning": "Model sounds very confident."}'
        )

        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{hallucinated_prose_flag}",
            source=AnswerSource.LLM_PROSE,
            task_context={"description": "Find the flag", "category": "crypto"},
        )

        # Deterministic trust gate strictly overrides LLM recommendation
        self.assertEqual(verdict.status, AnswerStatus.CANDIDATE)
        self.assertLessEqual(verdict.confidence, 0.5)

    async def test_14_verifier_agent_trust_boundary_cannot_fake_authoritative_verified(self):
        """14. Trust Boundary: LLM cannot declare VERIFIED status without authoritative=True."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "RESOLVE", "confidence": 1.0, "answer_type": "flag", "is_distractor": false, "reasoning": "Confirmed 100%."}'
        )

        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "FLAG{evidence_based_flag}",
            source=AnswerSource.TOOL_OUTPUT,
            authoritative=False,
            task_context={"description": "Find the flag", "category": "web"},
        )

        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertFalse(verdict.is_verified)

    async def test_15_verifier_agent_llm_distractor_rejection(self):
        """15. VerifierAgent correctly rejects candidate when LLM flags it as a distractor."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "REJECT", "confidence": 0.05, "answer_type": "flag", "is_distractor": true, "reasoning": "This is a decoy string embedded in comments."}'
        )

        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "FLAG{decoy_flag_in_comment}",
            source=AnswerSource.TOOL_OUTPUT,
            task_context={"description": "Find the real admin username", "category": "web"},
        )

        self.assertEqual(verdict.status, AnswerStatus.REJECTED)
        self.assertLessEqual(verdict.confidence, 0.1)

    async def test_16_verifier_agent_graceful_fallback_on_router_failure(self):
        """16. Graceful fallback: If router fails/times out, VerifierAgent falls back cleanly to deterministic assessment."""
        mock_router = AsyncMock()
        mock_router.route_request.side_effect = TimeoutError("Router request timed out")

        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{fallback_verified_flag}",
            source=AnswerSource.TOOL_OUTPUT,
            command="cat flag.txt",
            action_succeeded=True,
            task_context={"description": "Read flag.txt", "category": "misc"},
        )

        # Successfully falls back to deterministic AnswerResolver
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertGreaterEqual(verdict.confidence, 0.8)

    # =========================================================================
    # Explicit Coverage for 12 Integration Requirements
    # =========================================================================

    async def test_req01_real_tool_output_normal_picoctf_flag(self):
        """Req 1: Real tool output containing a normal picoCTF flag."""
        output = "Connected to server.\nAuthenticating...\nHere is your key: picoCTF{standard_flag_1234_alpha}\nGoodbye!"
        cands = self.resolver.extract_candidates(
            output,
            task_context={"description": "Find the flag", "category": "web"},
            source=AnswerSource.TOOL_OUTPUT,
        )
        self.assertTrue(any(c.value == "picoCTF{standard_flag_1234_alpha}" for c in cands))
        verdict = await self.verifier_agent.verify(
            cands[0],
            task_context={"description": "Find the flag", "category": "web"},
            command="curl http://target/flag",
            action_succeeded=True,
        )
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.answer_type, AnswerType.FLAG)

    async def test_req02_real_tool_output_non_flag_username(self):
        """Req 2: Real tool output containing a non-flag answer such as a username."""
        output = "Admin panel loaded.\nLogged in as user: admin_svc_account_99\nSession active."
        task_ctx = {"description": "Find the admin username on the server", "category": "web"}
        cands = self.resolver.extract_candidates(output, task_context=task_ctx, source=AnswerSource.TOOL_OUTPUT)
        self.assertTrue(len(cands) > 0)
        verdict = await self.verifier_agent.verify(
            cands[0],
            task_context=task_ctx,
            command="curl http://target/profile",
            action_succeeded=True,
        )
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.answer_type, AnswerType.USERNAME)

    async def test_req03_flag_shaped_distractor_when_challenge_asks_for_username(self):
        """Req 3: A flag-shaped distractor when the challenge asks for a username/hash/number."""
        task_ctx = {"description": "What is the database administrator username?", "category": "web"}
        verdict = await self.verifier_agent.verify(
            "picoCTF{fake_distractor_flag}",
            task_context=task_ctx,
            source=AnswerSource.TOOL_OUTPUT,
            command="cat /etc/passwd",
            action_succeeded=True,
        )
        self.assertEqual(verdict.status, AnswerStatus.REJECTED)
        self.assertLessEqual(verdict.confidence, 0.1)

    async def test_req04_llm_verifier_returning_resolve(self):
        """Req 4: LLM verifier returning RESOLVE produces RESOLVED."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "RESOLVE", "confidence": 0.95, "answer_type": "flag", "is_distractor": false, "reasoning": "Direct match from command output."}'
        )
        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{agent4_resolve_test}",
            source=AnswerSource.TOOL_OUTPUT,
            command="./solve",
            action_succeeded=True,
            task_context={"description": "Find the flag", "category": "pwn"},
        )
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertEqual(verdict.confidence, 0.95)
        self.assertFalse(verdict.is_verified)

    async def test_req05_llm_verifier_returning_reject(self):
        """Req 5: LLM verifier returning REJECT produces REJECTED."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "REJECT", "confidence": 0.0, "answer_type": "flag", "is_distractor": true, "reasoning": "Decoy string in comments."}'
        )
        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{decoy_flag}",
            source=AnswerSource.TOOL_OUTPUT,
            command="cat app.js",
            action_succeeded=True,
            task_context={"description": "Find the flag", "category": "web"},
        )
        self.assertEqual(verdict.status, AnswerStatus.REJECTED)
        self.assertLessEqual(verdict.confidence, 0.1)

    async def test_req06_llm_verifier_returning_needs_more_evidence(self):
        """Req 6: LLM verifier returning NEEDS_MORE_EVIDENCE preserves CANDIDATE."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "NEEDS_MORE_EVIDENCE", "confidence": 0.5, "answer_type": "flag", "is_distractor": false, "reasoning": "String looks like a flag fragment, need full decrypt."}'
        )
        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{partial_fragment_only}",
            source=AnswerSource.TOOL_OUTPUT,
            command="strings bin",
            action_succeeded=True,
            task_context={"description": "Find the flag", "category": "rev"},
        )
        self.assertEqual(verdict.status, AnswerStatus.CANDIDATE)
        self.assertLessEqual(verdict.confidence, 0.6)

    async def test_req07_malformed_llm_json_fallback(self):
        """Req 7: Malformed LLM JSON falls back gracefully to deterministic assessment."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content="This is not valid JSON at all!"
        )
        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{valid_flag_despite_bad_json}",
            source=AnswerSource.TOOL_OUTPUT,
            command="cat flag.txt",
            action_succeeded=True,
            task_context={"description": "Find flag", "category": "misc"},
        )
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)

    async def test_req08_router_provider_failure_or_refusal(self):
        """Req 8: Router/provider failure or refusal falls back cleanly to deterministic assessment."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content="",
            is_refusal=True,
            refusal_reason="Model refused safety policy"
        )
        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{flag_after_refusal}",
            source=AnswerSource.TOOL_OUTPUT,
            command="cat flag.txt",
            action_succeeded=True,
            task_context={"description": "Find flag", "category": "misc"},
        )
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)

    async def test_req09_llm_prose_candidate_cannot_become_verified(self):
        """Req 9: LLM prose candidate cannot become VERIFIED."""
        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "RESOLVE", "confidence": 1.0, "answer_type": "flag", "is_distractor": false, "reasoning": "Asserted in prose."}'
        )
        agent = VerifierAgent(resolver=self.resolver, router=mock_router)
        verdict = await agent.verify(
            "picoCTF{prose_unsupported_flag}",
            source=AnswerSource.LLM_PROSE,
            task_context={"description": "Find flag", "category": "misc"},
        )
        self.assertEqual(verdict.status, AnswerStatus.CANDIDATE)
        self.assertFalse(verdict.is_verified)
        self.assertLessEqual(verdict.confidence, 0.5)

    async def test_req10_resolved_is_not_reported_as_verified(self):
        """Req 10: RESOLVED is not reported as VERIFIED."""
        verdict = await self.verifier_agent.verify(
            "picoCTF{resolved_evidence_flag}",
            source=AnswerSource.TOOL_OUTPUT,
            command="cat flag.txt",
            action_succeeded=True,
            authoritative=False,
            task_context={"description": "Find flag", "category": "misc"},
        )
        self.assertEqual(verdict.status, AnswerStatus.RESOLVED)
        self.assertFalse(verdict.is_verified)
        self.assertTrue(verdict.is_resolved)

    async def test_req11_agent4_invoked_by_live_runtime_verification_path(self):
        """Req 11: Agent #4 is actually invoked by the live runtime verification path."""
        from backend.agent_runtime import AgentRuntime, MissionState, session_manager
        from backend.agent_runtime.action import Action, ActionType, ExecResult
        from backend.agent_runtime.decision import ProviderCompletion

        class MockToolExec:
            async def execute(self, action, **kwargs):
                return ExecResult(
                    status="SUCCESS",
                    stdout="Flag captured: picoCTF{live_runtime_flag_999}\n",
                    stderr="",
                    exit_code=0,
                    command=action.display(),
                )

        mock_router = AsyncMock()
        mock_router.route_request.return_value = AsyncMock(
            content='{"verdict": "RESOLVE", "confidence": 0.98, "answer_type": "flag", "is_distractor": false, "reasoning": "Real command output contains valid flag."}',
            is_refusal=False,
        )

        class MockProviderGateway:
            def __init__(self, router):
                self.router = router

            async def complete(self, **kwargs):
                return ProviderCompletion(
                    content="cat /flag.txt\nstrategy: exfil\nobjective: read flag",
                    provider_name="mock_provider",
                    model_name="mock_model",
                )

        gateway = MockProviderGateway(mock_router)
        runtime = AgentRuntime(tool_executor=MockToolExec(), provider_gateway=gateway)

        session = session_manager.create(
            agent_id="test_worker_1",
            engine="forge",
            target_scope="http://target:8080",
            category="web",
            description="Find the flag on the web server",
        )

        result = await runtime.run(session, max_turns=2)

        # Agent #4 router MUST have been awaited
        mock_router.route_request.assert_awaited()
        self.assertEqual(result.status, "COMPLETED")
        # RESOLVED from tool evidence → verified_flag is set in RunResult (enables coordinator flag flow)
        # The distinction between RESOLVED and VERIFIED is via AnswerStatus, not whether verified_flag is set
        self.assertIsNotNone(result.verified_flag)
        self.assertEqual(result.verified_flag, "picoCTF{live_runtime_flag_999}")

    async def test_req12_authoritative_verification_only_path_producing_verified(self):
        """Req 12: Authoritative verification is the only path producing VERIFIED."""
        # Non-authoritative
        non_auth = await self.verifier_agent.verify(
            "picoCTF{test_auth_flag}",
            source=AnswerSource.TOOL_OUTPUT,
            action_succeeded=True,
            authoritative=False,
            task_context={"description": "Find flag", "category": "web"},
        )
        self.assertEqual(non_auth.status, AnswerStatus.RESOLVED)
        self.assertFalse(non_auth.is_verified)

        # Authoritative
        auth = await self.verifier_agent.verify(
            "picoCTF{test_auth_flag}",
            source=AnswerSource.TOOL_OUTPUT,
            action_succeeded=True,
            authoritative=True,
            task_context={"description": "Find flag", "category": "web"},
        )
        self.assertEqual(auth.status, AnswerStatus.VERIFIED)
        self.assertTrue(auth.is_verified)
        self.assertEqual(auth.confidence, 1.0)

if __name__ == "__main__":
    unittest.main()
