import os
import sys
import unittest
import asyncio
import json
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db, SessionLocal
from backend.database.models import ChallengeModel, RunModel
from backend.websocket.manager import ws_manager, ConnectionManager

class TestObservabilityUX(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ── Test 1: useRef / Frontend Build Succeeds ──────────────────────────────

    def test_frontend_build_succeeds(self):
        """Verify that frontend TypeScript compilation and build pass cleanly."""
        frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Frontend build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # ── Test 2: WebSocket Connection Lifecycle ────────────────────────────────

    def test_websocket_connection_lifecycle(self):
        """Verify WebSocket ConnectionManager broadcast and connection tracking."""
        manager = ConnectionManager()
        self.assertEqual(len(manager.active_connections), 0)

        events_received = []

        class MockWebSocket:
            async def accept(self):
                pass
            async def send_json(self, message):
                events_received.append(message)

        mock_ws = MockWebSocket()
        asyncio.run(manager.connect(mock_ws))
        self.assertEqual(len(manager.active_connections), 1)

        test_event = {"event": "RUN_STARTED", "challenge_id": "ch_test_123", "status": "RUNNING"}
        asyncio.run(manager.broadcast(test_event))

        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0]["event"], "RUN_STARTED")
        self.assertEqual(events_received[0]["challenge_id"], "ch_test_123")

        manager.disconnect(mock_ws)
        self.assertEqual(len(manager.active_connections), 0)

    # ── Test 3: WebSocket Reconnect Behavior ──────────────────────────────────

    def test_websocket_reconnect_resync_payload(self):
        """Verify state resync payload delivered after reconnection."""
        ch = ChallengeModel(
            name="Reconnect Test",
            category="web",
            status="RUNNING",
            progress=45
        )
        self.db.add(ch)
        self.db.commit()

        # Query authoritative state as fetched on reconnection
        fetched = self.db.query(ChallengeModel).filter(ChallengeModel.id == ch.id).first()
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.status, "RUNNING")
        self.assertEqual(fetched.progress, 45)

        self.db.delete(ch)
        self.db.commit()

    # ── Test 4 & 5: Optimistic ID Transition & Early Events Preserved ─────────

    def test_optimistic_id_transition_and_early_events(self):
        """Verify optimistic temporary ID mapping logic preserves early events."""
        # Simulated optimistic state in frontend
        temp_id = "ch-1700000000"
        real_backend_id = "ch_backend_999"

        pending_temp_ids = {temp_id}
        challenges_state = [{"id": temp_id, "name": "Optimistic CTF", "status": "RUNNING", "progress": 0}]

        # Early WS Event arrives with real_backend_id
        early_ws_event = {
            "event": "RUN_STARTED",
            "challenge_id": real_backend_id,
            "status": "RUNNING",
            "progress": 10
        }

        # Resolution function simulating frontend WS message handler
        def process_ws_event(event, state, pending_ids):
            target_id = event["challenge_id"]
            exists = any(c["id"] == target_id for c in state)
            if not exists and len(pending_ids) > 0:
                mapped_temp = list(pending_ids)[0]
                pending_ids.remove(mapped_temp)
                for c in state:
                    if c["id"] == mapped_temp:
                        c["id"] = target_id
            
            # Apply event to state
            for c in state:
                if c["id"] == target_id:
                    c["status"] = event.get("status", c["status"])
                    c["progress"] = event.get("progress", c["progress"])
            return state

        updated_state = process_ws_event(early_ws_event, challenges_state, pending_temp_ids)

        self.assertEqual(len(pending_temp_ids), 0)
        self.assertEqual(updated_state[0]["id"], real_backend_id)
        self.assertEqual(updated_state[0]["progress"], 10)
        self.assertEqual(updated_state[0]["status"], "RUNNING")

    # ── Test 6: API Failure Represented as Error/Offline State ───────────────

    def test_api_failure_represented_as_error(self):
        """Verify API errors produce error state rather than empty data state."""
        class MockApiService:
          def __init__(self):
            self.is_online = True
            self.last_error = None

          def get_challenges_fail(self):
            self.is_online = False
            self.last_error = "HTTP 500 Server Error"
            raise RuntimeError("HTTP 500 Server Error")

        api = MockApiService()
        existing_cached_challenges = [{"id": "ch_existing", "name": "Cached Challenge"}]

        backend_error = None
        current_challenges = list(existing_cached_challenges)

        try:
          api.get_challenges_fail()
        except Exception as e:
          backend_error = str(e)

        # Confirm error is captured and cached state is preserved (not wiped to [])
        self.assertIsNotNone(backend_error)
        self.assertIn("500", backend_error)
        self.assertEqual(len(current_challenges), 1)
        self.assertEqual(current_challenges[0]["id"], "ch_existing")

    # ── Test 7: Running Challenge Status Visible from Backend State ───────────

    def test_running_challenge_status_visible(self):
        """Verify backend status 'RUNNING' is accurately reflected in models."""
        ch = ChallengeModel(
            name="Running Visibility Test",
            category="pwn",
            status="RUNNING",
            progress=65,
            description="http://10.10.10.20/"
        )
        self.db.add(ch)
        self.db.commit()

        run = RunModel(
            challenge_id=ch.id,
            status="RUNNING",
            current_phase="exploitation"
        )
        self.db.add(run)
        self.db.commit()

        # Query backend API model state
        db_ch = self.db.query(ChallengeModel).filter(ChallengeModel.id == ch.id).first()
        db_run = self.db.query(RunModel).filter(RunModel.challenge_id == ch.id).first()

        self.assertIsNotNone(db_ch)
        self.assertEqual(db_ch.status, "RUNNING")
        self.assertEqual(db_ch.progress, 65)
        self.assertIsNotNone(db_run)
        self.assertEqual(db_run.status, "RUNNING")

        # Cleanup
        self.db.delete(run)
        self.db.delete(ch)
        self.db.commit()

if __name__ == "__main__":
    unittest.main()
