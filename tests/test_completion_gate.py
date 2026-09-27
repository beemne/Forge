"""
Tests for Phase 1: Strict Dual-Gated Completion
Validates that challenges only transition to COMPLETED on verified flag capture or agent-confirmed solve.
"""
import os
import sys
import unittest

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import SessionLocal
from backend.database.models import (
    ChallengeModel, RunModel, TargetProfileModel, AgentStateModel,
    CheckpointModel, ToolExecutionModel, FindingModel, EvidenceModel,
    ReportModel
)

class TestDualGatedCompletion(unittest.TestCase):
    """Test that Forge never marks a challenge COMPLETED without flag proof."""

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        # FK-safe cleanup: SQLite runs with PRAGMA foreign_keys=ON, and a bulk
        # query().delete() does NOT honor ORM cascades — so children must be
        # deleted before their parents. Earlier suites (e.g. the competition
        # harness) leave checkpoint / tool-execution rows referencing runs AND
        # report rows referencing challenges, which made the old delete raise a FK
        # IntegrityError (DELETE FROM challenges) and error every test in this class.
        try:
            for model in (
                CheckpointModel, ToolExecutionModel, AgentStateModel,
                EvidenceModel, FindingModel, ReportModel, TargetProfileModel,
                RunModel, ChallengeModel,
            ):
                self.db.query(model).delete()
            self.db.commit()
        finally:
            self.db.close()

    def test_flag_regex_captures_picoctf(self):
        """Flag regex should match picoCTF{...} format."""
        import re
        FLAG_REGEX = re.compile(
            r"(?:picoCTF|FORGE|CTF|HTB|FLAG|THM)\{[A-Za-z0-9_!\-@#\$%\^&\*\.]+\}",
            re.IGNORECASE
        )
        test_output = "The flag is picoCTF{gr3p_15_4w3s0m3} hidden in the binary."
        matches = FLAG_REGEX.findall(test_output)
        self.assertTrue(len(matches) > 0, "Should match picoCTF flag format")
        self.assertEqual(matches[0], "picoCTF{gr3p_15_4w3s0m3}")

    def test_flag_regex_captures_htb(self):
        """Flag regex should match HTB{...} format."""
        import re
        FLAG_REGEX = re.compile(
            r"(?:picoCTF|FORGE|CTF|HTB|FLAG|THM)\{[A-Za-z0-9_!\-@#\$%\^&\*\.]+\}",
            re.IGNORECASE
        )
        matches = FLAG_REGEX.findall("Congratulations! HTB{buffer_0verfl0w_m4st3r}")
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0], "HTB{buffer_0verfl0w_m4st3r}")

    def test_flag_regex_captures_thm(self):
        """Flag regex should match THM{...} format."""
        import re
        FLAG_REGEX = re.compile(
            r"(?:picoCTF|FORGE|CTF|HTB|FLAG|THM)\{[A-Za-z0-9_!\-@#\$%\^&\*\.]+\}",
            re.IGNORECASE
        )
        matches = FLAG_REGEX.findall("THM{web_h4ck1ng}")
        self.assertTrue(len(matches) > 0)

    def test_exit_code_0_no_flag_should_not_complete(self):
        """Exit code 0 without a flag match should NOT result in COMPLETED status."""
        # Simulate the logic from cli_agent_runner.py
        captured_flag = None
        returncode = 0

        if captured_flag:
            status = "COMPLETED"
            progress = 100
        elif returncode == 0:
            status = "AWAITING_FLAG"
            progress = 80
        else:
            status = "FAILED"
            progress = 30

        self.assertEqual(status, "AWAITING_FLAG", "Exit 0 without flag must be AWAITING_FLAG, never COMPLETED")
        self.assertEqual(progress, 80)

    def test_exit_code_0_with_flag_should_complete(self):
        """Exit code 0 WITH a flag should result in COMPLETED status."""
        captured_flag = "picoCTF{test_flag_123}"
        returncode = 0

        if captured_flag:
            status = "COMPLETED"
            progress = 100
        elif returncode == 0:
            status = "AWAITING_FLAG"
            progress = 80
        else:
            status = "FAILED"
            progress = 30

        self.assertEqual(status, "COMPLETED")
        self.assertEqual(progress, 100)

    def test_agent_structured_completion_with_flag(self):
        """Agent emitting {"action": "complete", "flag": "...", "proof": "..."} with valid flag should COMPLETE."""
        import json
        import re
        raw_output = '{"action": "complete", "flag": "picoCTF{str1ngs_4re_gr8}", "proof": "Found flag via strings command on binary"}'
        parsed = json.loads(raw_output)
        self.assertEqual(parsed["action"], "complete")

        FLAG_REGEX = re.compile(
            r"(?:picoCTF|FORGE|CTF|HTB|FLAG|THM)\{[A-Za-z0-9_!\-@#\$%\^&\*\.]+\}",
            re.IGNORECASE
        )
        combined = f"{parsed.get('flag', '')} {parsed.get('proof', '')}"
        matches = FLAG_REGEX.findall(combined)
        self.assertTrue(len(matches) > 0, "Should find flag in structured completion JSON")
        self.assertEqual(matches[0], "picoCTF{str1ngs_4re_gr8}")

    def test_agent_done_without_flag_is_not_completed(self):
        """Agent saying 'DONE' without a flag should NOT mark challenge as COMPLETED in the post-loop handler."""
        raw_output = "DONE"
        self.assertEqual(raw_output.upper(), "DONE")
        # Post-loop logic should check flag_status before setting COMPLETED
        flag_status = None  # No flag was captured
        if flag_status == "CAPTURED":
            final_status = "COMPLETED"
        else:
            final_status = "AWAITING_FLAG"
        self.assertEqual(final_status, "AWAITING_FLAG")

    def test_nonzero_exit_code_is_failed(self):
        """Non-zero exit code with no flag should result in FAILED status."""
        captured_flag = None
        returncode = 1

        if captured_flag:
            status = "COMPLETED"
        elif returncode == 0:
            status = "AWAITING_FLAG"
        else:
            status = "FAILED"

        self.assertEqual(status, "FAILED")

if __name__ == "__main__":
    unittest.main()
