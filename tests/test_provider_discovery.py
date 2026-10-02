"""Workstream B: live provider/model catalog discovery + dynamic routing via breaker.

Network is fully mocked — no real provider quota is spent (the brief's cost constraint).
"""
import os
import sys
import asyncio
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db
from backend.providers.router import model_router
from backend.providers.quota_manager import quota_manager
from backend.providers.discovery import DiscoveryService


class _FakeResp:
    def __init__(self, status_code=200, payload=None, raise_exc=None):
        self.status_code = status_code
        self._payload = payload or {}
        self._raise = raise_exc

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, resp=None, raise_exc=None):
        self._resp = resp
        self._raise = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        if self._raise:
            raise self._raise
        return self._resp


class _FakeProvider:
    def __init__(self, name, base_url, default_model):
        self.name = name
        self.base_url = base_url
        self.api_key = "k"
        self.default_model = default_model
        self.is_paid = False
        self.extra_headers = {}

    async def is_available(self):
        return True


class TestProviderDiscovery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def _run_with_providers(self, providers, resp=None, raise_exc=None, verify=False):
        svc = DiscoveryService()
        original = model_router.providers
        try:
            model_router.providers = {p.name: p for p in providers}
            client_factory = lambda *a, **k: _FakeClient(resp=resp, raise_exc=raise_exc)
            with patch("httpx.AsyncClient", client_factory):
                return asyncio.run(svc.discover_all(verify=verify)), svc
        finally:
            model_router.providers = original

    def test_catalog_probe_records_health(self):
        prov = _FakeProvider("groq_probe_test", "https://api.groq.test/openai/v1", "qwen/qwen3.8-27b")
        payload = {"data": [{"id": "qwen/qwen3.8-27b"}, {"id": "other/model"}]}
        report, svc = self._run_with_providers([prov], resp=_FakeResp(200, payload))
        h = svc.health_for("groq_probe_test")
        self.assertIsNotNone(h)
        self.assertTrue(h.reachable)
        self.assertTrue(h.auth_ok)
        self.assertEqual(h.health_status, "healthy")
        self.assertIn("qwen/qwen3.8-27b", h.catalog_models)
        self.assertTrue(h.default_model_present)
        self.assertIn("groq_probe_test", report["providers"])

    def test_default_model_missing_trips_breaker(self):
        prov = _FakeProvider("nvidia_probe_test", "https://nv.test/v1", "deepseek/gone-model")
        payload = {"data": [{"id": "some/other-model"}]}  # default model NOT present
        _report, svc = self._run_with_providers([prov], resp=_FakeResp(200, payload))
        h = svc.health_for("nvidia_probe_test")
        self.assertFalse(h.default_model_present)
        # Dynamic routing (B3): the missing default model is circuit-broken.
        self.assertTrue(quota_manager.is_tripped("nvidia_probe_test", "deepseek/gone-model"))

    def test_unreachable_provider_marked_unavailable(self):
        prov = _FakeProvider("dead_probe_test", "https://dead.test/v1", "x")
        _report, svc = self._run_with_providers([prov], raise_exc=ConnectionError("no route"))
        h = svc.health_for("dead_probe_test")
        self.assertEqual(h.health_status, "unavailable")
        self.assertFalse(h.reachable)
        self.assertTrue(quota_manager.is_blacklisted_for_session("dead_probe_test"))

    def test_auth_failure_marks_unavailable(self):
        prov = _FakeProvider("badkey_probe_test", "https://x.test/v1", "x")
        _report, svc = self._run_with_providers([prov], resp=_FakeResp(401, {}))
        h = svc.health_for("badkey_probe_test")
        self.assertFalse(h.auth_ok)
        self.assertEqual(h.health_status, "unavailable")

    def tearDown(self):
        # Clear any breaker entries this test created so it doesn't leak into others.
        for k in ("nvidia_probe_test", "dead_probe_test",
                  "nvidia_probe_test::deepseek/gone-model"):
            quota_manager._session_blacklisted.pop(k.lower(), None)


if __name__ == "__main__":
    unittest.main()
