"""Unit tests for FORGE Swarm Architecture, Blackboard, Circuit Breaker, and Keep-Awake Engine."""

import unittest
import asyncio
import os

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.engine.keep_awake import keep_awake_manager
from backend.agents.swarm_orchestrator import SwarmOrchestrator
from backend.agents.swarm_state import SwarmBlackboard, SwarmTask
from backend.agents.swarm_helpers import _is_meaningful_header, _decode_artifacts
from backend.providers.quota_manager import quota_manager

class TestSwarmEngine(unittest.TestCase):

    def test_keep_awake_lifecycle(self):
        initial_holds = keep_awake_manager.active_holds_count
        
        # Acquire hold 1
        keep_awake_manager.acquire("Test Challenge Run 1")
        self.assertEqual(keep_awake_manager.active_holds_count, initial_holds + 1)
        self.assertTrue(keep_awake_manager.is_active)

        # Acquire hold 2
        keep_awake_manager.acquire("Test Challenge Run 2")
        self.assertEqual(keep_awake_manager.active_holds_count, initial_holds + 2)

        # Release hold 2
        keep_awake_manager.release("Test Challenge Run 2 finished")
        self.assertEqual(keep_awake_manager.active_holds_count, initial_holds + 1)

        # Release hold 1
        keep_awake_manager.release("Test Challenge Run 1 finished")
        self.assertEqual(keep_awake_manager.active_holds_count, initial_holds)

    def test_session_circuit_breaker(self):
        test_provider = "test_rate_limited_provider"
        self.assertFalse(quota_manager.is_blacklisted_for_session(test_provider))

        # Blacklist provider
        quota_manager.blacklist_for_session(test_provider, "HTTP 429 Rate Limit")
        self.assertTrue(quota_manager.is_blacklisted_for_session(test_provider))

        # Clean reset
        quota_manager.reset_session_blacklists()
        self.assertFalse(quota_manager.is_blacklisted_for_session(test_provider))

    def test_swarm_blackboard_lifecycle(self):
        async def run_blackboard_flow():
            board = SwarmBlackboard("test_ch_1", "test_run_1", "http://127.0.0.1:8000")

            # 1. Add tasks
            t1 = await board.add_task("RECON", "Scan web ports", priority=5)
            t2 = await board.add_task("CODE_AUDIT", "Deobfuscate JS script", priority=3)
            self.assertEqual(len(board.task_pool), 2)

            # 2. Claim recon task
            claimed = await board.claim_task("worker_recon", ["RECON"])
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.task_id, t1.task_id)
            self.assertEqual(claimed.status, "CLAIMED")
            self.assertEqual(claimed.claimed_by, "worker_recon")

            # 3. Complete task with discoveries
            await board.complete_task(
                claimed.task_id,
                result="Found /admin",
                discoveries={"endpoints": ["/admin", "/login"], "headers": {"X-Dev-Access": "yes"}}
            )
            self.assertEqual(claimed.status, "COMPLETED")
            self.assertIn("/admin", board.discovered_endpoints)
            self.assertEqual(board.extracted_headers.get("X-Dev-Access"), "yes")

            # 4. Record flag and verify event set
            self.assertFalse(board.flag_event.is_set())
            await board.record_flag("picoCTF{test_swarm_flag_123}", "worker_exploit")
            self.assertTrue(board.flag_event.is_set())
            self.assertEqual(board.flag_captured, "picoCTF{test_swarm_flag_123}")

        asyncio.run(run_blackboard_flow())

    def test_header_validation_rejects_llm_noise(self):
        """Regression for the 'Crack the Gate 1' run: LLM placeholder/prose header
        suggestions must be rejected, while concrete headers pass."""
        # Placeholder / prose shapes that previously got injected repeatedly.
        self.assertFalse(_is_meaningful_header("X-Forwarded-For", "127.0.0.1; [malicious payload]"))
        self.assertFalse(_is_meaningful_header("X-Forwarded-For", "127.0.0.1**"))
        self.assertFalse(_is_meaningful_header("X-Dev-Access", "<the value>"))
        self.assertFalse(_is_meaningful_header("Bad Header", "x"))          # space in name
        self.assertFalse(_is_meaningful_header("X", "line1\nline2"))         # multi-line
        # Concrete, injectable headers are accepted.
        self.assertTrue(_is_meaningful_header("X-Dev-Access", "yes"))
        self.assertTrue(_is_meaningful_header("X-Forwarded-For", "127.0.0.1"))  # legit technique, tried once via dedup

    def test_rot13_comment_decodes_to_header(self):
        """The exact ROT13 HTML comment from the challenge must deterministically
        decode to the winning 'X-Dev-Access: yes' directive."""
        comment = 'ABGR: Wnpx - grzcbenel olcnff: hfr urnqre "K-Qri-Npprff: lrf"'
        decodes = _decode_artifacts(comment)
        self.assertTrue(any(d["scheme"] == "rot13" for d in decodes))
        self.assertTrue(any("X-Dev-Access" in d["decoded"] and "yes" in d["decoded"] for d in decodes),
                        f"ROT13 decode did not surface the header directive: {decodes}")

    def test_decoded_directive_queues_one_deduped_exploit(self):
        """_apply_decoded_directives must turn the decoded hint into exactly one
        prioritized EXPLOIT task, store the concrete header, and never duplicate it
        on re-analysis (the loop the original run fell into)."""
        async def flow():
            board = SwarmBlackboard("t_ch", "t_run", "http://amiable-citadel.picoctf.net:60068/")
            orch = SwarmOrchestrator()
            comment = 'ABGR: Wnpx - grzcbenel olcnff: hfr urnqre "K-Qri-Npprff: lrf"'

            acted = await orch._apply_decoded_directives(comment, board, "worker_code_crypto")
            self.assertTrue(acted)
            self.assertEqual(board.extracted_headers.get("X-Dev-Access"), "yes")
            exploit_tasks = [t for t in board.task_pool.values() if t.category == "EXPLOIT"]
            self.assertEqual(len(exploit_tasks), 1)
            self.assertEqual(exploit_tasks[0].metadata.get("header_value"), "yes")

            # Re-analyzing the same artifact must NOT enqueue a duplicate injection.
            await orch._apply_decoded_directives(comment, board, "worker_code_crypto")
            exploit_tasks = [t for t in board.task_pool.values() if t.category == "EXPLOIT"]
            self.assertEqual(len(exploit_tasks), 1)

        asyncio.run(flow())

    def test_header_injection_is_deduped(self):
        """note_exploit_header queues an injection exactly once per (name,value),
        even across placeholder-annotated variants of the same value."""
        async def flow():
            board = SwarmBlackboard("t_ch", "t_run", "http://target.ctf/")
            self.assertTrue(await board.note_exploit_header("X-Forwarded-For", "127.0.0.1 (for bypass)", "w"))
            # Same concrete value again (verbatim + annotated) -> no new task.
            self.assertFalse(await board.note_exploit_header("X-Forwarded-For", "127.0.0.1", "w"))
            self.assertFalse(await board.note_exploit_header("X-Forwarded-For", "127.0.0.1**", "w"))
            exploit_tasks = [t for t in board.task_pool.values() if t.category == "EXPLOIT"]
            self.assertEqual(len(exploit_tasks), 1)
            self.assertEqual(board.extracted_headers.get("X-Forwarded-For"), "127.0.0.1")

        asyncio.run(flow())

if __name__ == "__main__":
    unittest.main()
