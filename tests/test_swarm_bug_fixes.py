import os
import sys
import unittest
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.swarm_helpers import (
    _normalize_recon_target,
    _normalize_failure_signature,
)
from backend.agents.swarm_state import SwarmBlackboard
from backend.tools.manager import classify_tool_execution

class TestSwarmBugFixes(unittest.TestCase):

    # ── Bug 1: Recon Cache Normalization ─────────────────────────────────────

    def test_recon_cache_normalization_formatting_flags(self):
        # Formatting flags (-s, -v, -i, -k, -L) stripped, same key generated
        cmd1 = "curl http://target.local:8080/api"
        cmd2 = "curl -s -v -k -L http://target.local:8080/api/"
        key1 = _normalize_recon_target(cmd1)
        key2 = _normalize_recon_target(cmd2)
        self.assertIsNotNone(key1)
        self.assertEqual(key1, key2)

    def test_recon_cache_preserves_auth_state(self):
        # Unauthenticated request vs request with auth header must NOT collapse
        cmd_unauth = "curl -s http://target.local/admin"
        cmd_auth = "curl -s -H 'Authorization: Bearer token123' http://target.local/admin"
        cmd_cookie = "curl -s -b 'session=xyz' http://target.local/admin"
        cmd_post = "curl -s -X POST -d 'user=admin' http://target.local/admin"

        key_unauth = _normalize_recon_target(cmd_unauth)
        key_auth = _normalize_recon_target(cmd_auth)
        key_cookie = _normalize_recon_target(cmd_cookie)
        key_post = _normalize_recon_target(cmd_post)

        self.assertNotEqual(key_unauth, key_auth)
        self.assertNotEqual(key_unauth, key_cookie)
        self.assertNotEqual(key_unauth, key_post)
        self.assertNotEqual(key_auth, key_cookie)

    def test_recon_cache_pipeline_differentiation(self):
        cmd1 = "curl -s http://standard-pizzas.picoctf.net:59328/"
        cmd2 = "curl -s http://standard-pizzas.picoctf.net:59328/ | cat"
        cmd3 = "curl -s http://standard-pizzas.picoctf.net:59328/ | grep -oE 'name=\"[^\"]*\"|<input[^>]*>'"
        cmd4 = "curl -s http://standard-pizzas.picoctf.net:59328/ -o /tmp/idx.html; grep -oE 'name=\"[^\"]*\"|action=\"[^\"]*\"'"

        k1 = _normalize_recon_target(cmd1)
        k2 = _normalize_recon_target(cmd2)
        k3 = _normalize_recon_target(cmd3)
        k4 = _normalize_recon_target(cmd4)

        keys = [k1, k2, k3, k4]
        self.assertEqual(len(set(keys)), 4, "All four pipeline variations must produce distinct cache keys")

        # Verify genuine duplicate commands return identical keys
        cmd_a = "curl -s http://standard-pizzas.picoctf.net:59328/"
        cmd_b = "curl -s http://standard-pizzas.picoctf.net:59328/"
        self.assertEqual(_normalize_recon_target(cmd_a), _normalize_recon_target(cmd_b))

        cmd_c = "curl -s http://standard-pizzas.picoctf.net:59328/ | grep -oE 'name=\"[^\"]*\"'"
        cmd_d = "curl -s http://standard-pizzas.picoctf.net:59328/ | grep -oE 'name=\"[^\"]*\"'"
        self.assertEqual(_normalize_recon_target(cmd_c), _normalize_recon_target(cmd_d))

    def test_recon_cache_file_and_nmap(self):
        key_cat = _normalize_recon_target("cat /etc/passwd")
        key_cat2 = _normalize_recon_target("cat -n /etc/passwd")
        self.assertEqual(key_cat, key_cat2)

        key_nmap = _normalize_recon_target("nmap -sV 10.10.10.5")
        key_nmap2 = _normalize_recon_target("nmap 10.10.10.5")
        self.assertEqual(key_nmap, key_nmap2)

    # ── Bug 3: Failure Signatures with Target Host Scoping ─────────────────────

    def test_failure_signature_target_scoping(self):
        # Identical error text on Host A vs Host B should produce DIFFERENT signatures
        sig_a = _normalize_failure_signature("curl http://hostA.ctf/secret", "CONNECTION_REFUSED", "curl: (7) Failed to connect to hostA.ctf port 80: Connection refused")
        sig_b = _normalize_failure_signature("curl http://hostB.ctf/secret", "CONNECTION_REFUSED", "curl: (7) Failed to connect to hostB.ctf port 80: Connection refused")
        self.assertNotEqual(sig_a, sig_b)
        self.assertTrue(sig_a.startswith("hosta.ctf:"))
        self.assertTrue(sig_b.startswith("hostb.ctf:"))

    def test_failure_signature_normalizes_volatile_data(self):
        # Hex addresses and line numbers should be normalized out
        err1 = "Traceback (most recent call last):\n  File 'solve.py', line 42, in <module>\n    print(0x7ffff7a00000)"
        err2 = "Traceback (most recent call last):\n  File 'solve.py', line 88, in <module>\n    print(0x7ffff7b12345)"
        sig1 = _normalize_failure_signature("python solve.py", "SYNTAX_ERROR", err1)
        sig2 = _normalize_failure_signature("python solve.py", "SYNTAX_ERROR", err2)
        self.assertEqual(sig1, sig2)

    # ── Bug 4: First-Occurrence Terminal Failure Classification ──────────────

    def test_sudo_password_classified_as_permission_denied(self):
        res1 = classify_tool_execution("bash", 1, "", "sudo: a password is required")
        res2 = classify_tool_execution("bash", 1, "", "sudo: no tty present and no askpass program specified")
        self.assertTrue(res1["execution_failure"])
        self.assertEqual(res1["failure_category"], "PERMISSION_DENIED")
        self.assertTrue(res2["execution_failure"])
        self.assertEqual(res2["failure_category"], "PERMISSION_DENIED")

    # ── Blackboard Persistence Snapshot ──────────────────────────────────────

    def test_blackboard_snapshot_roundtrip(self):
        board1 = SwarmBlackboard("chal-1", "run-1", "http://target.local")
        board1.recon_cache["web:GET:http://target.local"] = "cached data"
        board1.failure_signatures["target.local:403:forbidden"] = 3
        board1.blocked_failure_sigs.add("target.local:403:forbidden")
        board1.blocked_capabilities.add("sudo")

        mission_plan = board1._build_mission_plan()
        snapshot = mission_plan["blackboard_state"]

        board2 = SwarmBlackboard("chal-1", "run-1", "http://target.local")
        board2.load_snapshot(snapshot)

        self.assertEqual(board2.recon_cache.get("web:GET:http://target.local"), "cached data")
        self.assertEqual(board2.failure_signatures.get("target.local:403:forbidden"), 3)
        self.assertIn("target.local:403:forbidden", board2.blocked_failure_sigs)
        self.assertIn("sudo", board2.blocked_capabilities)

    # ── Bug 3 End-to-End Integration Pre/Post Check Test ─────────────────────

    def test_bug3_precheck_postcheck_integration_skip(self):
        """Integration test for Bug 3:
        1. Runs failing command 3 times through real pre-check + post-check flow.
        2. Asserts command shape is added to blocked_failure_sigs.
        3. Runs command a 4th time and asserts it is skipped via pre-check (execute_tool not called).
        """
        from unittest.mock import AsyncMock, patch
        from backend.agents.swarm_orchestrator import SwarmOrchestrator
        from backend.agents.swarm_helpers import _normalize_command_shape

        board = SwarmBlackboard("chal-int-1", "run-int-1", "http://target.local")
        orchestrator = SwarmOrchestrator()

        cmd = "python solve_exploit.py --target http://target.local/admin"
        cmd_shape = _normalize_command_shape(cmd)

        mock_tool_res = AsyncMock()
        mock_tool_res.exit_code = 1
        mock_tool_res.stdout = ""
        mock_tool_res.stderr = "Traceback (most recent call last):\n  File 'solve.py', line 10\n    ConnectionRefusedError: [Errno 111] Connection refused"
        mock_tool_res.execution_failure = True
        mock_tool_res.failure_category = "CONNECTION_REFUSED"

        iter_n = 0
        def fake_route_request(*args, **kwargs):
            nonlocal iter_n
            iter_n += 1
            cmd_variant = f"python solve_exploit.py --target http://target.local/admin?v={iter_n}"
            resp = AsyncMock()
            resp.is_refusal = False
            resp.content = f"I will run the exploit script:\n```bash\n{cmd_variant}\n```"
            resp.model_name = "test-model"
            return resp

        exec_count = 0
        async def fake_execute_tool(*args, **kwargs):
            nonlocal exec_count
            exec_count += 1
            return mock_tool_res

        with patch("backend.agents.swarm_orchestrator.model_router.route_request", side_effect=fake_route_request), \
             patch("backend.agents.swarm_orchestrator.privilege_manager.evaluate_privilege_ex", return_value=(True, None)), \
             patch("backend.agents.swarm_orchestrator.tool_manager.execute_tool", side_effect=fake_execute_tool):

            board.max_iterations = 4
            asyncio.run(orchestrator._agent_worker("agent_1", board, ".", "code_execution"))

            self.assertEqual(exec_count, 3, "Tool should have executed exactly 3 times across 4 iterations (4th iteration skipped by pre-check)")
            self.assertIn(cmd_shape, board.blocked_failure_sigs, f"Command shape '{cmd_shape}' must be in blocked_failure_sigs")
            self.assertTrue(any("[SIGNATURE BLOCKED]" in line for line in board.agent_transcripts.get("agent_1", [])))

    # ── Bug Fix: Challenge Log-Path Race Prevention & Line Migration ───────────

    def test_challenge_log_path_race_prevention_and_migration(self):
        import uuid
        from backend.utils.challenge_paths import (
            register_challenge_log_path,
            resolve_challenge_log_path,
            forget_challenge_log_path,
            logs_base,
        )
        from backend.agents.swarm_helpers import _append_to_challenge_log

        ch_id = f"test_race_{uuid.uuid4().hex[:8]}"

        try:
            # 1. Early write with zero metadata (resolves to provisional flat path)
            _append_to_challenge_log(ch_id, "worker1", "EARLY_LOG_LINE_1")

            # 2. Later call register_challenge_log_path with full metadata
            final_path = register_challenge_log_path(
                ch_id, "HackTheBox", "Web", "Easy", "MyRaceChallenge"
            )

            # 3. Late write after registration
            _append_to_challenge_log(ch_id, "worker2", "LATE_LOG_LINE_2")

            # Assert final_path exists and contains BOTH lines
            self.assertTrue(os.path.isfile(final_path), f"Final log path '{final_path}' should exist")
            with open(final_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("EARLY_LOG_LINE_1", content, "Early log line 1 must be migrated to final file")
            self.assertIn("LATE_LOG_LINE_2", content, "Late log line 2 must be present in final file")

            # Assert provisional flat file was cleaned up / migrated
            flat_file = os.path.join(logs_base(), f"challenge_{ch_id}.log")
            self.assertFalse(os.path.exists(flat_file), f"Provisional flat log file '{flat_file}' should no longer exist after migration")

        finally:
            # Cleanup test files if present
            resolved = resolve_challenge_log_path(ch_id)
            forget_challenge_log_path(ch_id)
            if os.path.isfile(resolved):
                try:
                    os.remove(resolved)
                except OSError:
                    pass

if __name__ == "__main__":
    unittest.main()
