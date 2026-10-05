"""Workstream F: chat-driven challenge creation — conversational missing-field
collection, category/difficulty normalization, and the '+' multi-target convention."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from fastapi.testclient import TestClient
from backend.main import app
from backend.database.session import init_db
from backend.utils.challenge_normalize import (
    normalize_category, normalize_difficulty, normalize_targets,
)


class TestChallengeNormalization(unittest.TestCase):

    def test_category_normalization(self):
        self.assertEqual(normalize_category("binary exploitation"), "pwn")
        self.assertEqual(normalize_category("Reverse Engineering"), "rev")
        self.assertEqual(normalize_category("Web"), "web")
        self.assertEqual(normalize_category("cryptography"), "crypto")
        self.assertIsNone(normalize_category("   "))

    def test_difficulty_normalization(self):
        self.assertEqual(normalize_difficulty("insane"), "INSANE")
        self.assertEqual(normalize_difficulty("beginner"), "EASY")
        self.assertEqual(normalize_difficulty("HARD"), "HARD")
        self.assertEqual(normalize_difficulty(""), "MEDIUM")           # default
        self.assertEqual(normalize_difficulty(None, default="MEDIUM"), "MEDIUM")

    def test_multi_target_joined_with_plus(self):
        self.assertEqual(normalize_targets("http://x:8080, /path/a.pcap, 10.10.14.23"),
                         "http://x:8080 + /path/a.pcap + 10.10.14.23")
        self.assertEqual(normalize_targets("http://x:8080\nnc host 9000"),
                         "http://x:8080 + nc host 9000")
        self.assertEqual(normalize_targets("a + a + b"), "a + b")       # dedupe, order kept
        self.assertEqual(normalize_targets(""), "")


class TestConversationalCreationFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def test_incremental_missing_field_collection(self):
        # 1. Start a session.
        start = self.client.post("/api/challenges/chat-session")
        self.assertEqual(start.status_code, 200)
        sid = start.json()["session_id"]

        # 2. Provide ONLY the name — the bot must ask for the still-missing fields and NOT fail.
        r1 = self.client.post(f"/api/challenges/chat-session/{sid}/message",
                              json={"challenge_name": "Impossible Password"})
        self.assertEqual(r1.status_code, 200)
        body1 = r1.json()
        self.assertEqual(body1["step"], 1)                 # still collecting
        self.assertIn("category", body1.get("awaiting", []))
        self.assertIn("difficulty", body1.get("awaiting", []))
        self.assertNotIn("name", body1.get("awaiting", []))  # name already captured

        # 3. Provide the remaining fields (free-text category + difficulty) -> advance.
        r2 = self.client.post(f"/api/challenges/chat-session/{sid}/message",
                              json={"challenge_type": "binary exploitation", "difficulty": "insane"})
        self.assertEqual(r2.status_code, 200)
        body2 = r2.json()
        self.assertEqual(body2["step"], 2)                 # all required fields gathered
        self.assertIn("pwn", body2["bot_message"])         # category normalized
        self.assertIn("INSANE", body2["bot_message"])      # difficulty normalized

    def test_pure_chat_free_text_slot_filling(self):
        """Operator answers each prompt with a plain free-text message (no form fields)."""
        sid = self.client.post("/api/challenges/chat-session").json()["session_id"]

        r_name = self.client.post(f"/api/challenges/chat-session/{sid}/message",
                                  json={"message": "Matrix Breakout"})
        self.assertEqual(r_name.json()["step"], 1)
        self.assertIn("category", r_name.json().get("awaiting", []))

        # Unrecognized category -> re-ask category only, never fail.
        r_bad = self.client.post(f"/api/challenges/chat-session/{sid}/message",
                                 json={"message": "something vague"})
        self.assertEqual(r_bad.status_code, 200)
        self.assertIn("category", r_bad.json().get("awaiting", []))

        r_cat = self.client.post(f"/api/challenges/chat-session/{sid}/message",
                                 json={"message": "reverse engineering"})
        self.assertIn("difficulty", r_cat.json().get("awaiting", []))

        r_diff = self.client.post(f"/api/challenges/chat-session/{sid}/message",
                                  json={"message": "hard"})
        body = r_diff.json()
        self.assertEqual(body["step"], 2)
        self.assertIn("rev", body["bot_message"])          # category normalized from free text
        self.assertIn("HARD", body["bot_message"])         # difficulty normalized from free text


if __name__ == "__main__":
    unittest.main()
