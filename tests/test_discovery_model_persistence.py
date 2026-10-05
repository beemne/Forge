"""Workstream B2: discovery persists per-model state (present_in_catalog, callable,
context_length, last_probe_ts) to the models table. Network fully mocked."""
import os
import sys
import asyncio
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db, SessionLocal
from backend.database.models import ModelConfigModel
from backend.providers.router import model_router
from backend.providers.discovery import DiscoveryService


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
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


class TestDiscoveryModelPersistence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_per_model_rows_reflect_catalog(self):
        # "groq" is in MODEL_PROVIDER_MAP (qwen/qwen3.8-27b, openai/gpt-oss-120b, groq/compound).
        prov = _FakeProvider("groq", "https://api.groq.test/openai/v1", "qwen/qwen3.8-27b")
        payload = {"data": [{"id": "qwen/qwen3.8-27b"}]}  # only qwen present in the live catalog
        svc = DiscoveryService()
        original = model_router.providers
        try:
            model_router.providers = {"groq": prov}
            with patch("httpx.AsyncClient", lambda *a, **k: _FakeClient(_FakeResp(200, payload))):
                asyncio.run(svc.discover_all(verify=False))
        finally:
            model_router.providers = original

        db = SessionLocal()
        try:
            qwen = (db.query(ModelConfigModel)
                    .filter_by(provider_name="groq", model_name="qwen/qwen3.8-27b").first())
            self.assertIsNotNone(qwen)
            self.assertTrue(qwen.present_in_catalog)
            self.assertGreater(qwen.context_length, 1024)   # real window persisted, not the 8192 default pin
            self.assertGreater(qwen.last_probe_ts, 0)

            gptoss = (db.query(ModelConfigModel)
                      .filter_by(provider_name="groq", model_name="openai/gpt-oss-120b").first())
            self.assertIsNotNone(gptoss)
            self.assertFalse(gptoss.present_in_catalog)     # advertised in MAP, absent from live catalog
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
