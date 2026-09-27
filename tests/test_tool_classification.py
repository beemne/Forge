"""Unit tests for local vs network execution-failure classification (Part 4 #2).

A command that never reached the target (unquoted path with a space -> [Errno 2],
Python SyntaxError, permission denied, missing dependency) must be flagged as an
execution failure in a LOCAL category, so the swarm aborts after a couple of hits
instead of re-running the same broken invocation against a rate-limited provider chain.
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

from backend.tools.manager import classify_tool_execution, LOCAL_EXEC_CATEGORIES

class TestLocalExecutionFailures(unittest.TestCase):
    def test_errno2_no_such_file_is_file_not_found(self):
        # The exact production signature: `python solve_agent_1.py` split on a space in
        # ".../WEB/EASY/Old Sessions" -> python can't open "solve_agent_1.py".
        r = classify_tool_execution("python", 2,
                                    "", "python: can't open file 'Sessions/solve.py': [Errno 2] No such file or directory")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "FILE_NOT_FOUND")
        self.assertIn("FILE_NOT_FOUND", LOCAL_EXEC_CATEGORIES)

    def test_python_syntax_error(self):
        r = classify_tool_execution("python", 1, "", "  File \"solve.py\", line 3\n    SyntaxError: invalid syntax")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "SYNTAX_ERROR")

    def test_permission_denied(self):
        r = classify_tool_execution("bash", 126, "", "bash: ./run.sh: Permission denied")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "PERMISSION_DENIED")

    def test_missing_dependency(self):
        r = classify_tool_execution("python", 1, "", "ModuleNotFoundError: No module named 'pwn'")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "MISSING_DEP")

class TestNetworkFailuresStillWork(unittest.TestCase):
    def test_dns_error_preserved(self):
        r = classify_tool_execution("curl", 6, "", "curl: (6) Could not resolve host: bad.ctf")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "DNS_ERROR")

    def test_connection_refused_preserved(self):
        r = classify_tool_execution("curl", 7, "", "curl: (7) Failed to connect")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "CONNECTION_REFUSED")

class TestTargetResponsesNotFlagged(unittest.TestCase):
    def test_normal_200_is_not_a_failure(self):
        r = classify_tool_execution("curl", 0, "HTTP/1.1 200 OK\n{\"success\":false}", "")
        self.assertFalse(r["execution_failure"])
        self.assertIsNone(r["failure_category"])

    def test_target_saying_user_not_found_is_not_local_failure(self):
        # A real target response (even a rejection) must NOT be a local exec failure —
        # it reached the target and is a legitimate signal for the agent to reason about.
        r = classify_tool_execution("curl", 0, '{"error":"User not found."}', "")
        self.assertFalse(r["execution_failure"])

class TestZeroExitOutputIsNeverAFailure(unittest.TestCase):
    """Regression: a command that exited 0 succeeded — failure phrases in its OUTPUT
    are data, not diagnostics (observed with an 11MB heap-dump that contained the
    string 'no such file or directory' yet exited 0 and held the flag)."""

    def test_heapdump_body_with_enoent_text_not_flagged(self):
        stdout = ('200 11252517\n/tmp/hd.bin: ASCII text\npicoCTF{Pat!3nt_15_Th3_K3y}\n'
                  '{"err":"ENOENT: no such file or directory, open \'/app/x\'"}')
        r = classify_tool_execution("curl", 0, stdout, "")
        self.assertFalse(r["execution_failure"])
        self.assertIsNone(r["failure_category"])

    def test_strings_output_mentioning_permission_denied_not_flagged(self):
        r = classify_tool_execution("strings", 0, "log: Permission denied while opening /etc/shadow", "")
        self.assertFalse(r["execution_failure"])

    def test_grep_hit_on_modulenotfound_text_not_flagged(self):
        # grep succeeds (exit 0) and prints a line containing the phrase — not a dep failure.
        r = classify_tool_execution("grep", 0, "README: handle ModuleNotFoundError: No module named 'x'", "")
        self.assertFalse(r["execution_failure"])

    def test_real_failure_with_nonzero_exit_still_flagged(self):
        # The guard is exit-code scoped: a genuine bad invocation (non-zero) is unchanged.
        r = classify_tool_execution("python", 2, "", "python: can't open file 'x.py': [Errno 2] No such file or directory")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "FILE_NOT_FOUND")

    def test_unknown_exit_still_scanned(self):
        # exit_code None (unknown) must still be scanned — preserves the -1/None path.
        r = classify_tool_execution("python", None, "", "ModuleNotFoundError: No module named 'pwn'")
        self.assertTrue(r["execution_failure"])
        self.assertEqual(r["failure_category"], "MISSING_DEP")

if __name__ == "__main__":
    unittest.main()
