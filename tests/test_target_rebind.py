"""Tests for dynamic target address re-binding.

Validates:
1. PUT /targets/{id}/address updates current_address AND address_history lineage.
2. Past evidence and runs are NOT deleted on rebind.
3. Challenge chat executes rebind on /rebind <addr> and natural-language requests.
4. Response body includes challenge_id and address_history.
"""

import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.main import app
from backend.database.session import SessionLocal, init_db
from backend.database.models import (
    ChallengeModel,
    TargetProfileModel,
    EvidenceModel,
    RunModel,
    CheckpointModel,
    FindingModel,
    ReportModel,
    ToolExecutionModel,
    AgentStateModel,
    ChatMessageModel,
)
from backend.providers.base import ProviderResponse

_WIPE_ORDER = [
    AgentStateModel, CheckpointModel, ToolExecutionModel,
    ChatMessageModel, EvidenceModel, FindingModel,
    ReportModel, RunModel, TargetProfileModel, ChallengeModel,
]

def _wipe(db):
    for m in _WIPE_ORDER:
        db.query(m).delete()
    db.commit()

class TestTargetRebindEndpoint(unittest.TestCase):
    """Core rebind mechanics via PUT /targets/{id}/address."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()
        _wipe(self.db)
        ch = ChallengeModel(id="ch-rb-1", name="Rebind", category="WEB", status="RUNNING")
        self.db.add(ch)
        tgt = TargetProfileModel(
            id="tgt-rb-1",
            challenge_id="ch-rb-1",
            current_address="10.10.10.10",
            hostname="init.ctf",
            address_history=["10.10.10.10"],
            verification_status="verified_network",
        )
        self.db.add(tgt)
        self.db.add(RunModel(id="run-rb-1", challenge_id="ch-rb-1", status="RUNNING"))
        self.db.add(EvidenceModel(
            challenge_id="ch-rb-1", agent="recon",
            evidence_type="http_response", source="curl",
            content="HTTP 200 from initial",
        ))
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def _rebind(self, addr):
        return self.client.put(
            "/api/targets/tgt-rb-1/address",
            json={"new_address": addr},
        )

    def test_updates_address_and_preserves_lineage(self):
        r = self._rebind("10.10.14.25")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["current_address"], "10.10.14.25")
        self.assertIn("10.10.10.10", d["address_history"], "Old address must remain in lineage")
        self.assertIn("10.10.14.25", d["address_history"], "New address must be in lineage")
        self.assertEqual(d["address_history"][-1], "10.10.14.25")

    def test_response_includes_challenge_id(self):
        r = self._rebind("192.168.1.1")
        self.assertEqual(r.json()["challenge_id"], "ch-rb-1")

    def test_verification_status_is_address_updated(self):
        r = self._rebind("172.16.0.5")
        self.assertEqual(r.json()["verification_status"], "address_updated")

    def test_evidence_not_deleted_on_rebind(self):
        self._rebind("10.10.14.25")
        ev = self.db.query(EvidenceModel).filter(
            EvidenceModel.challenge_id == "ch-rb-1"
        ).all()
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].content, "HTTP 200 from initial")

    def test_runs_not_deleted_on_rebind(self):
        self._rebind("10.10.14.25")
        runs = self.db.query(RunModel).filter(
            RunModel.challenge_id == "ch-rb-1"
        ).all()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].id, "run-rb-1")

    def test_target_stays_linked_to_challenge(self):
        self._rebind("10.10.14.25")
        self.db.expire_all()
        tgt = self.db.query(TargetProfileModel).filter(
            TargetProfileModel.id == "tgt-rb-1"
        ).first()
        self.assertEqual(tgt.challenge_id, "ch-rb-1")

    def test_empty_address_rejected(self):
        r = self.client.put(
            "/api/targets/tgt-rb-1/address",
            json={"new_address": "  "},
        )
        self.assertEqual(r.status_code, 400)

    def test_nonexistent_target_is_404(self):
        r = self.client.put(
            "/api/targets/nope/address",
            json={"new_address": "10.0.0.1"},
        )
        self.assertEqual(r.status_code, 404)

    def test_sequential_rebinds_accumulate_history(self):
        for addr in ["10.10.14.25", "10.10.14.50", "10.10.14.99"]:
            self._rebind(addr)
        d = self._rebind("10.10.14.200").json()
        for addr in ["10.10.10.10", "10.10.14.25", "10.10.14.50", "10.10.14.99", "10.10.14.200"]:
            self.assertIn(addr, d["address_history"])

class TestRebindViaChat(unittest.TestCase):
    """Chat endpoint rebinds target on /rebind command and natural language."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()
        _wipe(self.db)
        ch = ChallengeModel(id="ch-crbind-1", name="ChatRebind", category="WEB", status="RUNNING")
        self.db.add(ch)
        self.db.add(TargetProfileModel(
            id="tgt-crbind-1",
            challenge_id="ch-crbind-1",
            current_address="10.0.0.1",
            hostname="old.ctf",
            address_history=["10.0.0.1"],
            verification_status="verified_network",
        ))
        self.db.add(EvidenceModel(
            challenge_id="ch-crbind-1", agent="recon",
            evidence_type="banner", source="nmap",
            content="SSH banner captured",
        ))
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def _msg(self, content):
        return self.client.post(
            "/api/challenges/ch-crbind-1/messages",
            json={"content": content},
        )

    def test_slash_rebind_updates_target(self):
        r = self._msg("/rebind 10.10.14.77")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()["assistant_message"]["content"]
        self.assertIn("10.10.14.77", body)
        self.assertIn("10.0.0.1", body)
        self.db.expire_all()
        tgt = self.db.query(TargetProfileModel).filter(
            TargetProfileModel.challenge_id == "ch-crbind-1"
        ).first()
        self.assertEqual(tgt.current_address, "10.10.14.77")
        self.assertIn("10.0.0.1", tgt.address_history)

    def test_natural_language_rebind(self):
        r = self._msg("rebind the target address to 192.168.55.10")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("192.168.55.10", r.json()["assistant_message"]["content"])
        self.db.expire_all()
        tgt = self.db.query(TargetProfileModel).filter(
            TargetProfileModel.challenge_id == "ch-crbind-1"
        ).first()
        self.assertEqual(tgt.current_address, "192.168.55.10")

    def test_chat_rebind_preserves_evidence(self):
        self._msg("/rebind 10.10.14.77")
        ev = self.db.query(EvidenceModel).filter(
            EvidenceModel.challenge_id == "ch-crbind-1"
        ).all()
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].content, "SSH banner captured")

    @patch("backend.providers.router.model_router.route_request", new_callable=AsyncMock)
    def test_normal_message_routes_to_llm(self, mock_route):
        mock_route.return_value = ProviderResponse(
            provider_name="test", model_name="test",
            content="Port 80 is open.", prompt_tokens=5, completion_tokens=5,
            is_refusal=False,
        )
        r = self._msg("What ports are open?")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Port 80", r.json()["assistant_message"]["content"])
        mock_route.assert_called_once()

class TestExtractRebindAddress(unittest.TestCase):
    """Unit tests for the _extract_rebind_address NLP helper."""

    def setUp(self):
        from backend.api.routes.challenges import _extract_rebind_address
        self.extract = _extract_rebind_address

    def test_slash_rebind_ip(self):
        self.assertEqual(self.extract("/rebind 10.10.14.25"), "10.10.14.25")

    def test_slash_rebind_url(self):
        self.assertEqual(self.extract("/rebind http://target.local:8080"), "http://target.local:8080")

    def test_slash_rebind_target_variant(self):
        self.assertEqual(self.extract("/rebind_target 10.10.14.25"), "10.10.14.25")

    def test_natural_full_phrase(self):
        self.assertEqual(self.extract("rebind the target address to 10.0.0.5"), "10.0.0.5")

    def test_natural_update_ip(self):
        self.assertEqual(self.extract("update the target ip to 192.168.1.1"), "192.168.1.1")

    def test_natural_change_domain(self):
        self.assertEqual(self.extract("change target domain to new.ctf"), "new.ctf")

    def test_non_rebind_returns_none(self):
        self.assertIsNone(self.extract("What is the current challenge status?"))

    def test_empty_returns_none(self):
        self.assertIsNone(self.extract(""))

if __name__ == "__main__":
    unittest.main()
