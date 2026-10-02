"""Workstream A1/A2: universal circuit breaker with proportional cooldowns,
(provider, model, capability) granularity, and persistence across restart."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db, SessionLocal
from backend.database.models import ProviderBreakerModel
from backend.providers.quota_manager import AgentRouterQuotaManager, quota_manager


class TestCircuitBreaker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_cooldown_tiers_are_proportional(self):
        qm = AgentRouterQuotaManager()
        cases = [
            ("brk_gone", "404 model not found", qm.COOLDOWN_GONE),
            ("brk_quota", "402 insufficient quota", qm.COOLDOWN_QUOTA),
            ("brk_auth", "403 forbidden permission denied", qm.COOLDOWN_AUTH),
            ("brk_rl", "429 rate limit", qm.COOLDOWN_RATE_LIMIT),
        ]
        for key, reason, expected in cases:
            qm.blacklist_for_session(key, reason, persist=False)
            expiry, _ = qm._session_blacklisted[key]
            self.assertAlmostEqual(expiry - time.time(), expected, delta=5,
                                   msg=f"{reason} -> wrong cooldown tier")

    def test_triple_key_granularity(self):
        qm = AgentRouterQuotaManager()
        qm.trip_breaker("nvidia", "deepseek/x", "recon", reason="410 gone", )
        self.assertTrue(qm.is_tripped("nvidia", "deepseek/x", "recon"))
        # A different model/capability on the same provider is NOT tripped.
        self.assertFalse(qm.is_tripped("nvidia", "other/model", "web_analysis"))
        # A different provider is unaffected.
        self.assertFalse(qm.is_tripped("groq", "y", "recon"))

    def test_bare_provider_trip_blocks_every_capability(self):
        qm = AgentRouterQuotaManager()
        qm.blacklist_for_session("cloudflare", "402 quota", persist=False)
        self.assertTrue(qm.is_tripped("cloudflare", capability="recon"))
        self.assertTrue(qm.is_tripped("cloudflare", capability="code_analysis"))

    def test_breaker_state_survives_restart(self):
        key = "pytest_breaker_prov"
        self._cleanup(key)
        # Trip with persistence on (writes a row to provider_breakers).
        quota_manager.blacklist_for_session(key, "402 quota exhausted", persist=True)

        # A brand-new manager (a simulated process restart) starts clean...
        fresh = AgentRouterQuotaManager()
        self.assertFalse(fresh.is_blacklisted_for_session(key))
        # ...then restores the still-active breaker from the DB.
        loaded = fresh.load_persisted_breakers()
        self.assertGreaterEqual(loaded, 1)
        self.assertTrue(fresh.is_blacklisted_for_session(key))
        self._cleanup(key)

    def _cleanup(self, key: str):
        quota_manager._session_blacklisted.pop(key.lower(), None)
        db = SessionLocal()
        try:
            db.query(ProviderBreakerModel).filter(ProviderBreakerModel.key == key.lower()).delete()
            db.commit()
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
