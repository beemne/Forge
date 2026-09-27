import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

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
    ChatMessageModel,
    FindingModel,
    TargetProfileModel,
    RunModel,
    ToolExecutionModel,
    AgentStateModel,
    CheckpointModel,
    EvidenceModel,
    ReportModel,
)
from backend.providers.base import ProviderResponse
import os

class TestChallengePersistentChat(unittest.TestCase):
    """Test suite for persistent challenge chat and live state awareness."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()
        try:
            self.db.query(AgentStateModel).delete()
            self.db.query(CheckpointModel).delete()
            self.db.query(ToolExecutionModel).delete()
            self.db.query(ChatMessageModel).delete()
            self.db.query(EvidenceModel).delete()
            self.db.query(FindingModel).delete()
            self.db.query(ReportModel).delete()
            self.db.query(RunModel).delete()
            self.db.query(TargetProfileModel).delete()
            self.db.query(ChallengeModel).delete()
            self.db.commit()
        finally:
            self.db.close()
            self.db = SessionLocal()

    def tearDown(self):
        if hasattr(self, "db") and self.db:
            self.db.close()

    def test_get_messages_empty_and_not_found(self):
        # 404 on nonexistent challenge
        resp = self.client.get("/api/challenges/nonexistent-id/messages")
        self.assertEqual(resp.status_code, 404)

        # 200 on existing challenge with no messages
        ch = ChallengeModel(
            id="ch-chat-test-1",
            name="Chat Test Challenge",
            category="WEB",
            difficulty="EASY",
            status="QUEUED",
        )
        self.db.add(ch)
        self.db.commit()

        resp = self.client.get("/api/challenges/ch-chat-test-1/messages")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    @patch("backend.providers.router.model_router.route_request", new_callable=AsyncMock)
    def test_post_message_persists_history_and_uses_context(self, mock_route_request):
        mock_route_request.return_value = ProviderResponse(
            provider_name="test_provider",
            model_name="test_model",
            content="Based on the scan findings, port 80 and 443 are open.",
            prompt_tokens=10,
            completion_tokens=15,
            is_refusal=False,
        )

        ch = ChallengeModel(
            id="ch-chat-test-2",
            name="SQLi Challenge",
            category="WEB",
            difficulty="MEDIUM",
            status="RUNNING",
            progress=40,
            description="Exploit the login portal",
        )
        self.db.add(ch)
        self.db.add(FindingModel(
            challenge_id="ch-chat-test-2",
            agent="web_agent",
            title="SQL Injection in /login",
            severity="HIGH",
            endpoint="/login",
        ))
        self.db.commit()

        # Post a message
        payload = {"content": "What have we found so far?"}
        resp = self.client.post("/api/challenges/ch-chat-test-2/messages", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("user_message", data)
        self.assertIn("assistant_message", data)
        self.assertEqual(data["user_message"]["content"], "What have we found so far?")
        self.assertEqual(data["assistant_message"]["content"], "Based on the scan findings, port 80 and 443 are open.")

        # Verify route_request was called with system_instruction containing live challenge context
        mock_route_request.assert_called_once()
        call_kwargs = mock_route_request.call_args[1]
        self.assertIn("SQLi Challenge", call_kwargs["system_instruction"])
        self.assertIn("SQL Injection in /login", call_kwargs["system_instruction"])
        self.assertIn("RUNNING", call_kwargs["system_instruction"])

        # Verify GET /messages returns both turns in order
        get_resp = self.client.get("/api/challenges/ch-chat-test-2/messages")
        self.assertEqual(get_resp.status_code, 200)
        msgs = get_resp.json()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")

    def test_post_message_validation_errors(self):
        # Empty message
        ch = ChallengeModel(id="ch-chat-test-3", name="Validation Test", category="REV")
        self.db.add(ch)
        self.db.commit()

        resp = self.client.post("/api/challenges/ch-chat-test-3/messages", json={"content": "  "})
        self.assertEqual(resp.status_code, 422)

        # Nonexistent challenge
        resp = self.client.post("/api/challenges/bad-id/messages", json={"content": "hello"})
        self.assertEqual(resp.status_code, 404)
