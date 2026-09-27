"""Unit tests for swarm budget/resume helpers (Part 4 #3 and #4).

#3 — the per-agent wall clock must EXCLUDE checkpoint-pause time.
#4 — pause/resume must carry each agent's actual reasoning history, not just counters.
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

from backend.agents.swarm_helpers import _effective_elapsed_minutes
from backend.agents.swarm_state import SwarmBlackboard

class TestEffectiveElapsedMinutes(unittest.TestCase):
    def test_no_pause(self):
        self.assertAlmostEqual(_effective_elapsed_minutes(1000.0, 1000.0 + 300, 0.0), 5.0, places=3)

    def test_pause_is_excluded(self):
        # The exact production scenario: 8.5 wall-clock minutes, 3.5 of them a checkpoint
        # pause -> only 5.0 working minutes should count against a 5-minute budget.
        started, now, paused = 1000.0, 1000.0 + 510, 210.0
        self.assertAlmostEqual(_effective_elapsed_minutes(started, now, paused), 5.0, places=3)

    def test_never_negative(self):
        self.assertEqual(_effective_elapsed_minutes(1000.0, 1000.0 + 60, 999.0), 0.0)

class TestResumeRehydratesContext(unittest.TestCase):
    def _board(self):
        return SwarmBlackboard("chal-test", "run-test", "http://t.ctf")

    def test_transcripts_and_history_round_trip(self):
        # Populate a board as if a run happened, snapshot it, then rehydrate a fresh board.
        b1 = self._board()
        b1.agent_ids = ["agent_1", "agent_2"]
        b1.record_agent_step("agent_1", command="curl -s http://t.ctf/login", output='{"error":"User not found."}')
        b1.record_agent_step("agent_2", command="curl -s http://t.ctf/", output="<!-- dev note -->")
        snapshot = b1._build_mission_plan()["blackboard_state"]

        self.assertIn("agent_transcripts", snapshot)
        self.assertIn("execution_history", snapshot)

        b2 = self._board()
        counts = b2.load_snapshot(snapshot)
        self.assertGreater(counts["transcript_lines"], 0)
        # The agent's actual prior step must be visible in its history context after resume,
        # so it does not re-run the same initial recon from scratch.
        ctx = b2.build_history_context("agent_1")
        self.assertIn("Your recent steps", ctx)
        self.assertIn("curl -s http://t.ctf/login", ctx)
        # Shared execution history carried over too.
        self.assertTrue(any(h.get("agent") == "agent_2" for h in b2.execution_history))

    def test_empty_snapshot_is_safe(self):
        b = self._board()
        counts = b.load_snapshot(None)
        self.assertEqual(counts["transcript_lines"], 0)

if __name__ == "__main__":
    unittest.main()
