"""Regression test for the flag brace-guard (SSTI1 false positive).

`picoCTF{{{flag}}}` was an SSTI *payload the agent echoed* (`[GET] 'picoCTF{{{flag}}}' -> 405`),
not a real flag — yet the old body class `[^}]{4,}` matched `picoCTF{{{flag}` and auto-captured
it. The body now excludes braces ([^{}]), so template/format artifacts can't match while real
flags (clean bodies) still do.
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

from backend.agents.swarm_orchestrator import FLAG_REGEX, FALSE_FLAG_PATTERNS

class TestFlagBraceGuard(unittest.TestCase):
    def test_template_artifact_rejected(self):
        self.assertIsNone(FLAG_REGEX.search("picoCTF{{{flag}}}"))
        self.assertIsNone(FLAG_REGEX.search("picoCTF{{flag}}"))
        # The exact SSTI1 payload-echo line that caused the false capture.
        self.assertIsNone(FLAG_REGEX.search("[GET] 'picoCTF{{{flag}}}' ... -> Status: 405"))

    def test_real_flags_still_match(self):
        for real in (
            "picoCTF{brut4_f0rc4_0d39383f}",
            "picoCTF{s3t_s3ss10n_3xp1rat10n5_77b6684a}",
            "picoCTF{rs4_k3y_1n_1mg_0a64c2f9}",
        ):
            m = FLAG_REGEX.search(f"prefix text {real} trailing")
            self.assertIsNotNone(m, real)
            self.assertEqual(m.group(0), real)
            self.assertFalse(FALSE_FLAG_PATTERNS.search(real), real)

if __name__ == "__main__":
    unittest.main()
