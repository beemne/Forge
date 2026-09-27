"""Unit tests for passive rate-limit awareness and provider-response hardening (Part 4 #5).

Covers:
- Scope inference from the reset window — the Groq "1,000 RPM" mislabel is really a
  DAILY bucket (reset measured in hours), and must be classified 'day', not 'minute'.
- Duration parsing across compound/bare forms.
- Headroom / near-ceiling queries on the quota manager.
- GLM max_tokens floor (_max_tokens_floor) so reasoning tokens can't starve output.
"""

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

from backend.providers.rate_limits import (
    parse_ratelimit_headers,
    parse_duration_seconds,
    classify_scope,
)
from backend.providers.quota_manager import AgentRouterQuotaManager
from backend.providers.real_providers import _max_tokens_floor

class TestDurationParsing(unittest.TestCase):
    def test_compound_and_bare(self):
        self.assertAlmostEqual(parse_duration_seconds("2m59.56s"), 179.56, places=2)
        self.assertAlmostEqual(parse_duration_seconds("1h30m"), 5400.0, places=2)
        self.assertAlmostEqual(parse_duration_seconds("882ms"), 0.882, places=3)
        self.assertAlmostEqual(parse_duration_seconds("60"), 60.0, places=2)
        self.assertEqual(parse_duration_seconds(""), None)
        self.assertEqual(parse_duration_seconds(None), None)

class TestScopeClassification(unittest.TestCase):
    def test_scope_boundaries(self):
        self.assertEqual(classify_scope(30), "minute")
        self.assertEqual(classify_scope(90), "minute")
        self.assertEqual(classify_scope(600), "hour")
        self.assertEqual(classify_scope(6 * 3600), "day")
        self.assertEqual(classify_scope(None), "unknown")

class TestGroqDayScopeMislabel(unittest.TestCase):
    def test_groq_daily_bucket_not_labeled_per_minute(self):
        # Groq free tier: a big request allowance whose reset is HOURS away -> it's a
        # per-DAY bucket. The friendly "1,000 RPM" reading is the bug this guards against.
        headers = {
            "x-ratelimit-limit-requests": "14400",
            "x-ratelimit-remaining-requests": "812",
            "x-ratelimit-reset-requests": "6h30m",
        }
        snap = parse_ratelimit_headers("groq", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.scope_requests, "day")          # NOT "minute"
        self.assertAlmostEqual(snap.request_headroom_fraction(), 812 / 14400, places=4)

    def test_true_per_minute_bucket(self):
        headers = {
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-remaining-requests": "29",
            "x-ratelimit-reset-requests": "2s",
        }
        snap = parse_ratelimit_headers("some_provider", headers)
        self.assertEqual(snap.scope_requests, "minute")

    def test_no_ratelimit_headers_returns_none(self):
        self.assertIsNone(parse_ratelimit_headers("x", {"content-type": "application/json"}))

class TestHeadroomQueries(unittest.TestCase):
    def test_near_ceiling_and_headroom(self):
        qm = AgentRouterQuotaManager()
        qm.record_ratelimit_snapshot("groq", parse_ratelimit_headers("groq", {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "50",     # 5% left
            "x-ratelimit-reset-requests": "12h",
        }))
        self.assertAlmostEqual(qm.get_headroom("groq"), 0.05, places=4)
        self.assertTrue(qm.is_near_ceiling("groq", frac=0.1))
        # Unobserved provider: no data -> never penalized.
        self.assertIsNone(qm.get_headroom("never_seen"))
        self.assertFalse(qm.is_near_ceiling("never_seen"))

class TestGlmMaxTokensFloor(unittest.TestCase):
    def test_glm_floored(self):
        self.assertGreaterEqual(_max_tokens_floor("z-ai/glm-5.3-flash"), 2048)
        self.assertGreaterEqual(_max_tokens_floor("glm-5.3"), 2048)

    def test_non_glm_no_floor(self):
        self.assertEqual(_max_tokens_floor("qwen/qwen3.8-27b"), 0)
        self.assertEqual(_max_tokens_floor("deepseek/deepseek-chat"), 0)

if __name__ == "__main__":
    unittest.main()
