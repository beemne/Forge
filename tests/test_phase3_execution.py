"""
tests/test_phase3_execution.py

Phase 3 — Tool Runtime / Execution Layer tests.

All tests run on Windows without requiring Linux, Kali/Parrot, Docker, or any
real network connectivity.  Real subprocess execution is either mocked or uses
trivially available system commands.
"""
import asyncio
import hashlib
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Allow running as `py -m unittest discover tests` from the project root ──
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine synchronously in tests (Python 3.10+ compatible)."""
    return asyncio.run(coro)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ExecutionRequest / ExecutionResult models
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionModels(unittest.TestCase):

    def test_execution_request_defaults(self):
        from backend.execution.base import ExecutionRequest
        req = ExecutionRequest(command="echo hello")
        self.assertEqual(req.command, "echo hello")
        self.assertEqual(req.timeout_seconds, 120)
        self.assertEqual(req.capability, "")
        self.assertIsNone(req.cwd)

    def test_execution_result_succeeded(self):
        from backend.execution.base import ExecutionResult, STATUS_SUCCESS, STATUS_FAILED
        r = ExecutionResult(status=STATUS_SUCCESS, exit_code=0)
        self.assertTrue(r.succeeded)
        r2 = ExecutionResult(status=STATUS_FAILED, exit_code=1)
        self.assertFalse(r2.succeeded)
        r3 = ExecutionResult(status=STATUS_SUCCESS, execution_failure=True)
        self.assertFalse(r3.succeeded)

    def test_to_tool_execution_result_bridge(self):
        from backend.execution.base import ExecutionResult, STATUS_SUCCESS
        r = ExecutionResult(
            status=STATUS_SUCCESS, stdout="hello", stderr="",
            exit_code=0, duration_ms=100.0,
            tool_name="curl", capability="web_testing",
            command="curl http://example.com",
        )
        ter = r.to_tool_execution_result()
        self.assertEqual(ter.tool_name, "curl")
        self.assertEqual(ter.capability, "web_testing")
        self.assertEqual(ter.status, STATUS_SUCCESS)
        self.assertEqual(ter.stdout, "hello")
        self.assertEqual(ter.exit_code, 0)

    def test_to_exec_result_bridge(self):
        from backend.execution.base import ExecutionResult, STATUS_FAILED
        r = ExecutionResult(
            status=STATUS_FAILED, stdout="", stderr="err",
            exit_code=1, tool_name="nmap", capability="network_scan",
        )
        er = r.to_exec_result()
        self.assertEqual(er.tool_name, "nmap")
        self.assertEqual(er.status, STATUS_FAILED)
        self.assertFalse(er.succeeded)

# ─────────────────────────────────────────────────────────────────────────────
# 2. WordlistResolver
# ─────────────────────────────────────────────────────────────────────────────

class TestWordlistResolver(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_env_var_override(self):
        from backend.execution.wordlist import WordlistResolver
        wl_file = os.path.join(self._tmp, "custom.txt")
        with open(wl_file, "w") as f:
            f.write("admin\nlogin\n")
        with patch.dict(os.environ, {"FORGE_WORDLIST": wl_file}):
            resolver = WordlistResolver(workspace_root=self._tmp)
            result = resolver.resolve()
        self.assertEqual(result, wl_file)

    def test_workspace_local_wordlist(self):
        from backend.execution.wordlist import WordlistResolver
        wl_file = os.path.join(self._tmp, "common.txt")
        with open(wl_file, "w") as f:
            f.write("admin\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORGE_WORDLIST", None)
            resolver = WordlistResolver(workspace_root=self._tmp)
            result = resolver.resolve()
        self.assertEqual(result, wl_file)

    def test_fallback_generation(self):
        from backend.execution.wordlist import WordlistResolver
        # Point at a clean temp dir with no wordlist
        clean_dir = os.path.join(self._tmp, "empty")
        os.makedirs(clean_dir, exist_ok=True)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORGE_WORDLIST", None)
            resolver = WordlistResolver(workspace_root=clean_dir)
            # On Windows & Linux system paths won't exist in a tempdir
            with patch("backend.execution.wordlist._LINUX_CANDIDATES", []):
                with patch("backend.execution.wordlist._WINDOWS_CANDIDATES", []):
                    result = resolver.resolve()
        self.assertTrue(os.path.isfile(result))
        with open(result) as f:
            contents = f.read()
        self.assertIn("admin", contents)

    def test_fallback_is_idempotent(self):
        """Calling resolve() twice does not recreate the file."""
        from backend.execution.wordlist import WordlistResolver
        clean_dir = os.path.join(self._tmp, "idem")
        os.makedirs(clean_dir, exist_ok=True)
        os.environ.pop("FORGE_WORDLIST", None)
        resolver = WordlistResolver(workspace_root=clean_dir)
        with patch("backend.execution.wordlist._LINUX_CANDIDATES", []):
            with patch("backend.execution.wordlist._WINDOWS_CANDIDATES", []):
                p1 = resolver.resolve()
                mtime1 = os.path.getmtime(p1)
                p2 = resolver.resolve()
                mtime2 = os.path.getmtime(p2)
        self.assertEqual(p1, p2)
        self.assertEqual(mtime1, mtime2)

# ─────────────────────────────────────────────────────────────────────────────
# 3. ArtifactStore
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactStore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _write(self, name: str, content: bytes) -> str:
        path = os.path.join(self._tmp, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_record_existing_file(self):
        from backend.execution.artifact_store import ArtifactStore, sha256_file
        store = ArtifactStore()
        content = b"picoCTF{test_artifact}"
        path = self._write("flag.txt", content)
        rec = store.record(path, session_id="s1", agent_id="a1", command="cat flag.txt")
        self.assertEqual(rec.session_id, "s1")
        self.assertEqual(rec.size, len(content))
        expected_sha = hashlib.sha256(content).hexdigest()
        self.assertEqual(rec.sha256, expected_sha)

    def test_record_missing_file_no_crash(self):
        from backend.execution.artifact_store import ArtifactStore
        store = ArtifactStore()
        rec = store.record("/nonexistent/path.bin", session_id="s2")
        self.assertIsNone(rec.sha256)
        self.assertEqual(rec.size, 0)

    def test_for_session_filter(self):
        from backend.execution.artifact_store import ArtifactStore
        store = ArtifactStore()
        p1 = self._write("a.txt", b"a")
        p2 = self._write("b.txt", b"b")
        store.record(p1, session_id="sess_A")
        store.record(p2, session_id="sess_B")
        results = store.for_session("sess_A")
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].path.endswith("a.txt"))

    def test_sha256_standalone(self):
        from backend.execution.artifact_store import sha256_file
        content = b"forge_test_hash"
        path = self._write("hash_test.bin", content)
        result = sha256_file(path)
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(result, expected)

    def test_sha256_missing_file_returns_none(self):
        from backend.execution.artifact_store import sha256_file
        result = sha256_file("/does/not/exist.bin")
        self.assertIsNone(result)

# ─────────────────────────────────────────────────────────────────────────────
# 4. ProcessManager
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessManager(unittest.TestCase):

    def test_successful_echo(self):
        from backend.execution.process_manager import ProcessManager
        pm = ProcessManager()
        # Use a command available on both Windows and Linux
        cmd = "echo hello"
        stdout, stderr, exit_code = _run(pm.run(cmd, timeout_seconds=10))
        self.assertEqual(exit_code, 0)
        self.assertIn("hello", stdout)

    def test_failed_command(self):
        from backend.execution.process_manager import ProcessManager
        pm = ProcessManager()
        cmd = "exit 1" if sys.platform == "win32" else "sh -c 'exit 1'"
        stdout, stderr, exit_code = _run(pm.run(cmd, timeout_seconds=5))
        self.assertNotEqual(exit_code, 0)

    def test_timeout_kills_process(self):
        from backend.execution.process_manager import ProcessManager
        pm = ProcessManager()
        # ping with many packets — will run longer than the timeout
        if sys.platform == "win32":
            cmd = "ping -n 100 127.0.0.1"
        else:
            cmd = "sleep 100"
        stdout, stderr, exit_code = _run(pm.run(cmd, timeout_seconds=1))
        self.assertEqual(exit_code, -1)
        self.assertIn("timed out", stderr)

    def test_active_count_zero_after_run(self):
        from backend.execution.process_manager import ProcessManager
        pm = ProcessManager()
        _run(pm.run("echo done", timeout_seconds=5))
        self.assertEqual(pm.active_count(), 0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. LocalBackend — tool availability
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalBackend(unittest.TestCase):

    def test_missing_tool_returns_missing_tool_status(self):
        """A tool not on PATH returns MISSING_TOOL, not an OS error."""
        from backend.execution.backends.local import LocalBackend
        from backend.execution.base import ExecutionRequest, STATUS_MISSING_TOOL
        backend = LocalBackend()
        req = ExecutionRequest(
            command="__forge_definitely_not_installed__ --help",
            tool_name="__forge_definitely_not_installed__",
            capability="web_fuzzing",
            timeout_seconds=5,
        )
        result = _run(backend.execute(req))
        self.assertEqual(result.status, STATUS_MISSING_TOOL)
        self.assertTrue(result.execution_failure)
        self.assertEqual(result.failure_category, "COMMAND_NOT_FOUND")

    def test_echo_command_succeeds(self):
        """A real command (echo) executes and returns SUCCESS."""
        from backend.execution.backends.local import LocalBackend
        from backend.execution.base import ExecutionRequest, STATUS_SUCCESS
        backend = LocalBackend()
        req = ExecutionRequest(command="echo phase3", timeout_seconds=10)
        result = _run(backend.execute(req))
        self.assertEqual(result.status, STATUS_SUCCESS)
        self.assertIn("phase3", result.stdout)
        self.assertEqual(result.exit_code, 0)

    def test_python_normalisation_windows(self):
        """On Windows, python3 is rewritten to python if python3 is absent."""
        from backend.execution.backends.local import LocalBackend
        backend = LocalBackend()

        with patch("backend.execution.backends.local.shutil.which") as mock_which:
            # Simulate: python3 absent, python present
            def _which(name):
                return "C:\\Python\\python.exe" if name == "python" else None
            mock_which.side_effect = _which

            normalised = backend._normalise_command("python3 solve.py")
            self.assertTrue(
                normalised.startswith("python"),
                f"Expected python prefix, got: {normalised}"
            )

    def test_python_normalisation_linux(self):
        """On Linux, python3 is preferred over python."""
        from backend.execution.backends.local import LocalBackend
        backend = LocalBackend()

        with patch("backend.execution.backends.local.shutil.which") as mock_which:
            def _which(name):
                return "/usr/bin/python3" if name == "python3" else None
            mock_which.side_effect = _which
            with patch("backend.execution.backends.local._IS_WINDOWS", False):
                normalised = backend._normalise_command("python3 solve.py")
        self.assertIn("python3", normalised)

    def test_backend_kind_is_local(self):
        from backend.execution.backends.local import LocalBackend
        self.assertEqual(LocalBackend.kind, "local")

# ─────────────────────────────────────────────────────────────────────────────
# 6. _resolve_python
# ─────────────────────────────────────────────────────────────────────────────

class TestResolvePython(unittest.TestCase):

    def test_returns_a_string_or_none(self):
        from backend.execution.backends.local import _resolve_python
        result = _resolve_python()
        # Must be None or a non-empty string — never crash
        self.assertTrue(result is None or isinstance(result, str))
        if result:
            self.assertGreater(len(result), 0)

    def test_windows_prefers_python(self):
        with patch("backend.execution.backends.local._IS_WINDOWS", True):
            with patch("backend.execution.backends.local.shutil.which") as w:
                w.side_effect = lambda n: "C:\\py.exe" if n == "python" else None
                from backend.execution.backends.local import _resolve_python
                result = _resolve_python()
        self.assertEqual(result, "python")

    def test_linux_prefers_python3(self):
        with patch("backend.execution.backends.local._IS_WINDOWS", False):
            with patch("backend.execution.backends.local.shutil.which") as w:
                w.side_effect = lambda n: "/usr/bin/python3" if n == "python3" else None
                from backend.execution.backends.local import _resolve_python
                result = _resolve_python()
        self.assertEqual(result, "python3")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ExecutionService
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionService(unittest.TestCase):

    def test_run_command_returns_execution_result(self):
        from backend.execution.service import ExecutionService
        from backend.execution.base import ExecutionResult
        svc = ExecutionService()
        result = _run(svc.run_command("echo forge_service", timeout_seconds=10))
        self.assertIsInstance(result, ExecutionResult)
        self.assertIn("forge_service", result.stdout)

    def test_register_custom_backend(self):
        from backend.execution.service import ExecutionService
        from backend.execution.base import ExecutionRequest, ExecutionResult, STATUS_SUCCESS
        svc = ExecutionService()

        class FakeBackend:
            async def execute(self, req):
                return ExecutionResult(status=STATUS_SUCCESS, stdout="fake_out", command=req.command)

        svc.register_backend("fake", FakeBackend())
        req = ExecutionRequest(command="whatever")
        result = _run(svc.execute(req, backend_kind="fake"))
        self.assertEqual(result.stdout, "fake_out")

    def test_unknown_backend_falls_back_to_local(self):
        from backend.execution.service import ExecutionService
        svc = ExecutionService()
        backend = svc.get_backend("nonexistent_backend_xyz")
        # Should silently fall back to local
        from backend.execution.backends.local import LocalBackend
        self.assertIsInstance(backend, LocalBackend)

    def test_execute_propagates_session_id(self):
        from backend.execution.service import ExecutionService
        from backend.execution.base import ExecutionRequest, ExecutionResult, STATUS_SUCCESS
        svc = ExecutionService()

        captured = {}

        class CapturingBackend:
            async def execute(self, req):
                captured["session_id"] = req.session_id
                return ExecutionResult(status=STATUS_SUCCESS)

        svc.register_backend("capture", CapturingBackend())
        req = ExecutionRequest(command="test", session_id="sess_xyz")
        _run(svc.execute(req, backend_kind="capture"))
        self.assertEqual(captured["session_id"], "sess_xyz")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ToolManager backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestToolManagerBackwardCompat(unittest.TestCase):
    """
    ToolManager's public API must still return ToolExecutionResult.
    We mock the execution_service so no real subprocesses are spawned.
    """

    def _make_mock_exec_result(self, stdout="", stderr="", exit_code=0, status="SUCCESS"):
        from backend.execution.base import ExecutionResult
        return ExecutionResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            command="mocked_command",
            tool_name="curl",
            capability="web_testing",
        )

    def test_execute_raw_command_returns_tool_execution_result(self):
        from backend.tools.manager import ToolManager, ToolExecutionResult
        mgr = ToolManager()
        mock_result = self._make_mock_exec_result(stdout="HTTP/1.1 200 OK")

        async def fake_run_command(*args, **kwargs):
            return mock_result

        with patch("backend.tools.manager.execution_service") as mock_svc:
            mock_svc.run_command = fake_run_command
            result = _run(mgr.execute_raw_command("curl http://example.com"))

        self.assertIsInstance(result, ToolExecutionResult)
        self.assertEqual(result.stdout, "HTTP/1.1 200 OK")
        self.assertEqual(result.status, "SUCCESS")

    def test_execute_capability_missing_tool(self):
        """When no tool is registered for a capability, MISSING_TOOL is returned."""
        from backend.tools.manager import ToolManager, ToolExecutionResult
        mgr = ToolManager()
        result = _run(mgr.execute_capability("__unknown_capability_xyz__", "10.10.14.1"))
        self.assertIsInstance(result, ToolExecutionResult)
        self.assertEqual(result.status, "MISSING_TOOL")

# ─────────────────────────────────────────────────────────────────────────────
# 9. ToolRegistry — no hardcoded Linux paths
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRegistryNoPaths(unittest.TestCase):

    def test_ffuf_args_template_has_no_usr_share(self):
        from backend.tools.registry import tool_registry
        ffuf = tool_registry.tools.get("ffuf")
        self.assertIsNotNone(ffuf, "ffuf should be registered")
        self.assertNotIn("/usr/share", ffuf.args_template,
                         "ffuf args_template must not hardcode /usr/share paths")

    def test_all_tools_no_hardcoded_linux_paths(self):
        from backend.tools.registry import tool_registry
        for name, tool in tool_registry.tools.items():
            self.assertNotIn("/usr/share", tool.args_template,
                             f"{name}.args_template contains hardcoded /usr/share path")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Workspace isolation (unchanged, verifying regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkspaceSafetyRegression(unittest.TestCase):

    def test_project_root_is_protected(self):
        from backend.utils.workspace import is_deletable_working_dir, PROJECT_ROOT
        self.assertFalse(is_deletable_working_dir(PROJECT_ROOT))

    def test_legitimate_challenge_dir_is_safe(self):
        from backend.utils.workspace import CTF_WORKSPACE_ROOT, is_deletable_working_dir
        challenge_dir = os.path.join(CTF_WORKSPACE_ROOT, "_workspaces", "web", "test_challenge")
        self.assertTrue(is_deletable_working_dir(challenge_dir))

    def test_resolve_safe_dir_for_dot_input(self):
        from backend.utils.workspace import resolve_safe_working_dir, CTF_WORKSPACE_ROOT
        result = resolve_safe_working_dir(".", "chall_001", category="web")
        self.assertTrue(result.startswith(CTF_WORKSPACE_ROOT),
                        f"Expected path inside CTF workspace, got: {result}")
        self.assertNotEqual(os.path.abspath("."), result)

# ─────────────────────────────────────────────────────────────────────────────
# 11. ExecutionResult → ExecResult → AgentRuntime compat
# ─────────────────────────────────────────────────────────────────────────────

class TestExecResultIntegration(unittest.TestCase):

    def test_execution_result_to_exec_result_succeeded(self):
        from backend.execution.base import ExecutionResult, STATUS_SUCCESS
        from backend.agent_runtime.action import ExecResult
        er = ExecutionResult(
            status=STATUS_SUCCESS, stdout="flag found", exit_code=0,
            tool_name="curl", capability="web_testing",
        )
        exec_res = er.to_exec_result()
        self.assertIsInstance(exec_res, ExecResult)
        self.assertTrue(exec_res.succeeded)
        self.assertEqual(exec_res.stdout, "flag found")

    def test_execution_result_to_exec_result_failure(self):
        from backend.execution.base import ExecutionResult, STATUS_FAILED
        from backend.agent_runtime.action import ExecResult
        er = ExecutionResult(
            status=STATUS_FAILED, stderr="connection refused", exit_code=7,
            execution_failure=True, failure_category="CONNECTION_REFUSED",
            tool_name="curl", capability="web_testing",
        )
        exec_res = er.to_exec_result()
        self.assertFalse(exec_res.succeeded)
        self.assertEqual(exec_res.failure_category, "CONNECTION_REFUSED")

    def test_exec_result_from_tool_result_still_works(self):
        """ExecResult.from_tool_result() continues working with ToolExecutionResult."""
        from backend.tools.manager import ToolExecutionResult
        from backend.agent_runtime.action import ExecResult
        ter = ToolExecutionResult(
            tool_name="nmap", capability="network_scanning",
            command="nmap 10.10.14.1", status="SUCCESS",
            stdout="Host is up", exit_code=0,
        )
        er = ExecResult.from_tool_result(ter)
        self.assertEqual(er.tool_name, "nmap")
        self.assertTrue(er.succeeded)

# ─────────────────────────────────────────────────────────────────────────────
# 12. CapabilityReport — unchanged Phase 2 behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCapabilityReportRegression(unittest.TestCase):

    def test_has_tool_case_insensitive(self):
        from backend.agent_runtime.execution_backend import CapabilityReport
        cap = CapabilityReport(os="linux", available_tools=["nmap", "ffuf"])
        self.assertTrue(cap.has_tool("NMAP"))
        self.assertFalse(cap.has_tool("gobuster"))

    def test_satisfies_os(self):
        from backend.agent_runtime.execution_backend import CapabilityReport
        cap = CapabilityReport(os="windows")
        self.assertTrue(cap.satisfies_os("windows"))
        self.assertFalse(cap.satisfies_os("linux"))
        self.assertTrue(cap.satisfies_os("any"))
        self.assertTrue(cap.satisfies_os(None))

    def test_fit_score_linux_missing_tools(self):
        from backend.agent_runtime.execution_backend import CapabilityReport
        cap = CapabilityReport(os="linux", available_tools=["nmap"])
        score = cap.fit_score({"required_os": "linux", "tools": ["nmap", "ffuf", "gobuster"]})
        # 1 of 3 tools present → score < 1.0 but > 0
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Execution-layer status endpoint + operator-terminal delegation (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def _multipart_available() -> bool:
    """backend.api.routes defines Form/UploadFile endpoints that need
    python-multipart at import time; the execution layer itself does not, so the
    handler test below is guarded on this rather than coupling the whole suite
    to the FastAPI surface."""
    import importlib.util
    return importlib.util.find_spec("multipart") is not None

class TestExecutionStatusEndpoint(unittest.TestCase):
    """GET /execution/status (criterion #8) and the operator terminal delegating
    to the shared ExecutionService instead of spawning its own subprocess
    (criterion #11).

    The status payload's data sources are verified directly so the execution-layer
    suite stays decoupled from the full FastAPI surface; the HTTP handlers are
    exercised whenever that surface is importable here.
    """

    def test_execution_status_payload_sources(self):
        # Exactly what GET /execution/status aggregates — verified via the real
        # Phase 3 singletons so the endpoint always has well-formed data to return.
        from backend.execution.service import execution_service
        from backend.execution.process_manager import process_manager
        from backend.execution.artifact_store import artifact_store
        from backend.agent_runtime.execution_backend import execution_backend

        backends = execution_service.list_backends()
        self.assertEqual(backends["default"], "local")
        self.assertIn("local", backends["registered"])

        env = execution_backend.capabilities().to_dict()
        for key in ("os", "backend_kind", "available_tools", "available_python_libs"):
            self.assertIn(key, env)

        self.assertIsInstance(process_manager.active_count(), int)
        self.assertIsInstance(process_manager.active_pids(), list)
        self.assertIsInstance(artifact_store.all_records(), list)

    @unittest.skipUnless(
        _multipart_available(),
        "backend.api.routes needs python-multipart (Form/UploadFile endpoints)")
    def test_status_and_terminal_handlers(self):
        from backend.api.routes import execution as routes
        from backend.api.routes.execution import (
            TerminalExecuteRequest, get_execution_status, execute_terminal_command)
        from backend.execution.base import ExecutionResult

        # GET /execution/status returns environment/backend/process/artifact info.
        status = get_execution_status()
        for key in ("environment", "backends", "processes", "artifacts"):
            self.assertIn(key, status)
        self.assertIn("local", status["backends"]["registered"])
        self.assertIn("os", status["environment"])

        # POST /terminal/execute delegates to ExecutionService (no local subprocess).
        fake = ExecutionResult(status="SUCCESS", stdout="root\n", exit_code=0, command="whoami")
        with patch.object(routes.execution_service, "run_command",
                          AsyncMock(return_value=fake)) as mock_run, \
             patch.object(routes.ws_manager, "broadcast", AsyncMock()):
            payload = _run(execute_terminal_command(
                TerminalExecuteRequest(command="whoami")))

        mock_run.assert_awaited_once()
        self.assertEqual(mock_run.await_args.args[0], "whoami")
        self.assertEqual(payload["event"], "LOG_OUTPUT")
        self.assertEqual(payload["output"], "root\n")
        self.assertEqual(payload["exit_code"], 0)

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
