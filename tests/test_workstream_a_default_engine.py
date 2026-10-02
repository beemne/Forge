"""Workstream A — default-engine depth:
  A3: passive rate-limit header capture for Gemini + Cloudflare.
  A4: shared cross-mission failed-approach record (both engines write; recall for prompts).
  A5: per-run token budget on the blackboard engine (clean wind-down).
"""
import os
import sys
import asyncio
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db, SessionLocal
from backend.database.models import FailedApproachModel
from backend.providers.quota_manager import quota_manager
from backend.providers.real_providers import GeminiProvider, CloudflareProvider
from backend.knowledge.failed_approaches import (
    record_failed_approach, recall_failed_approaches, recall_block,
)


class TestA3RateLimitCapture(unittest.TestCase):

    def test_gemini_records_ratelimit_snapshot(self):
        provider = GeminiProvider(api_key="k")

        async def fake_post(url, headers, payload, timeout=30.0):
            return ({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
                    {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "5"})

        with patch.object(provider, "_post_json", side_effect=fake_post):
            resp = asyncio.run(provider.generate_response("hi"))
        self.assertFalse(resp.is_refusal)
        snap = quota_manager.get_ratelimit_snapshot("gemini")
        self.assertIsNotNone(snap)
        self.assertAlmostEqual(quota_manager.get_headroom("gemini"), 0.05, places=3)

    def test_cloudflare_records_ratelimit_snapshot(self):
        provider = CloudflareProvider(api_key="k", account_id="acct")

        async def fake_post(url, headers, payload, timeout=30.0):
            return ({"result": {"response": "ok"}},
                    {"x-ratelimit-limit-requests": "50", "x-ratelimit-remaining-requests": "25"})

        with patch.object(provider, "_post_json", side_effect=fake_post):
            resp = asyncio.run(provider.generate_response("hi"))
        self.assertFalse(resp.is_refusal)
        self.assertAlmostEqual(quota_manager.get_headroom("cloudflare"), 0.5, places=3)


class TestA4SharedFailedApproaches(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self._cat = "pytest_a4_cat"
        self._purge()

    def tearDown(self):
        self._purge()

    def _purge(self):
        db = SessionLocal()
        try:
            db.query(FailedApproachModel).filter(FailedApproachModel.category == self._cat).delete()
            db.commit()
        finally:
            db.close()

    def test_record_increments_and_recall(self):
        record_failed_approach(self._cat, "curl -s <URL> :: 403", "blocked")
        record_failed_approach(self._cat, "curl -s <URL> :: 403", "still blocked")  # same -> count 2
        record_failed_approach(self._cat, "sqlmap <URL> :: timeout", "timeout")
        rows = {sig: count for sig, count, _ in recall_failed_approaches(self._cat)}
        self.assertEqual(rows.get("curl -s <URL> :: 403"), 2)
        self.assertEqual(rows.get("sqlmap <URL> :: timeout"), 1)

    def test_recall_block_formats_and_empty_is_blank(self):
        record_failed_approach(self._cat, "idor on /api/user", "no change")
        block = recall_block(self._cat)
        self.assertIn("CROSS-MISSION DEAD-ENDS", block)
        self.assertIn("idor on /api/user", block)
        self.assertEqual(recall_block("category_that_has_nothing_xyz"), "")


class TestA5TokenBudget(unittest.TestCase):

    def test_blackboard_has_token_budget_fields(self):
        from backend.agents.swarm_state import SwarmBlackboard
        board = SwarmBlackboard("c1", "r1", "http://t")
        self.assertEqual(board.tokens_used, 0)
        self.assertTrue(hasattr(board, "max_tokens"))

    def test_budget_exhaustion_condition(self):
        from backend.agents.swarm_state import SwarmBlackboard
        board = SwarmBlackboard("c1", "r1", "http://t")
        board.max_tokens = 100
        board.tokens_used += 60
        self.assertFalse(board.max_tokens and board.tokens_used >= board.max_tokens)
        board.tokens_used += 50            # 110 >= 100 -> wind-down fires in the worker loop
        self.assertTrue(board.max_tokens and board.tokens_used >= board.max_tokens)

    def test_run_swarm_accepts_max_tokens(self):
        import inspect
        from backend.agents.swarm_orchestrator import SwarmOrchestrator
        sig = inspect.signature(SwarmOrchestrator.run_swarm)
        self.assertIn("max_tokens", sig.parameters)


if __name__ == "__main__":
    unittest.main()
