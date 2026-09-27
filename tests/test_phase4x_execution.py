"""
Phase 4.x — Execution Capability Expansion tests (§27 checklist).

Covers: scripted stdin (Tier 1), persistent interactive execution (Tier 2),
interactive timeout + cleanup, checkpoint-safety, target-type detection + mismatch,
capability discovery + alternatives + missing-capability recovery, controlled
(privilege-gated) acquisition, OCR capability + Binary-Digits regression, the swarm
pre-dispatch capability/target gate (no infinite retry) + Flag-Hunters regression,
platform behaviour, and trajectory integration.

Everything runs on Windows without Kali/Parrot/Docker/network.  Real subprocesses
use the current interpreter (``sys.executable``) so they work regardless of how the
shell's ``python`` alias is configured; capability/acquisition tests use LOCAL
doubles so nothing is installed and no host is mutated.
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db

_IS_WINDOWS = sys.platform == "win32"
PY = sys.executable  # the real interpreter (avoids the Windows Store 'python' alias)

def _run(coro):
    return asyncio.run(coro)

# A tiny interactive challenge: prints a prompt, reads a line, prints the flag on
# the control string "RETURN 0" (the Flag Hunters shape), else prints "nope".
INTERACTIVE_CHILD = (
    "import sys\n"
    "print('Enter command:', flush=True)\n"
    "line = sys.stdin.readline().strip()\n"
    "print('picoCTF{flag_hunters_interactive}' if line == 'RETURN 0' else 'nope', flush=True)\n"
)
# A child that echoes each line it receives, forever (for multi-turn dialogue).
ECHO_CHILD = (
    "import sys\n"
    "for line in sys.stdin:\n"
    "    print('echo:' + line.strip(), flush=True)\n"
)
# A child that just sleeps (for max-lifetime enforcement).
SLEEP_CHILD = "import time\ntime.sleep(30)\n"
# A child that prints one line and exits immediately (already-dead cleanup tests).
QUICK_CHILD = "print('bye', flush=True)\n"

def _write_child(body: str, name: str = "challenge.py") -> str:
    d = tempfile.mkdtemp(prefix="forge4x_")
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    return p

# --------------------------------------------------------------------------- #
# Capability / acquisition test doubles (no host mutation, deterministic)
# --------------------------------------------------------------------------- #

class FakeCapability:
    def __init__(self, name, status, available, provider="", alternatives=None,
                 reason="", action="replan", acq=False):
        self.name = name
        self.status = status
        self.available = available
        self.provider = provider
        self.alternatives = alternatives or []
        self.providers_checked = ["providerA", "providerB"]
        self.backend_supported = True
        self.installable = True
        self.acquisition_possible = acq
        self.recommended_action = action
        self.reason = reason or f"{name} is {status}"

    def to_dict(self):
        return dict(self.__dict__)

    def observability_line(self):
        return (f"CAPABILITY_{self.status} capability={self.name} "
                f"providers_checked={self.providers_checked} "
                f"acquisition_possible={self.acquisition_possible} "
                f"recommended_action={self.recommended_action}")

class FakeCapabilityService:
    def __init__(self, mapping):
        self.mapping = mapping
        self.discover_calls = []

    def discover(self, name, use_cache=True):
        self.discover_calls.append(name)
        return self.mapping.get(name) or FakeCapability(name, "BLOCKED", False, reason="unknown")

    def is_available(self, name):
        return self.discover(name).available

    def refresh(self, name=None):
        pass

class FakeAcquisitionPlanner:
    def __init__(self, result):
        self._result = result
        self.acquire_calls = []

    async def acquire(self, capability, *, provider=None, agent="", installer=None,
                      privilege_decider=None):
        self.acquire_calls.append(capability)
        return self._result

# ─────────────────────────────────────────────────────────────────────────────
# 1. Tier-1 scripted stdin
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptedStdin(unittest.TestCase):

    def test_execution_service_stdin(self):
        from backend.execution.service import execution_service
        p = _write_child(INTERACTIVE_CHILD)
        r = _run(execution_service.run_command(f'"{PY}" "{p}"', stdin="RETURN 0\n",
                                               timeout_seconds=30))
        self.assertEqual(r.status, "SUCCESS")
        self.assertIn("picoCTF{flag_hunters_interactive}", r.stdout)

    def test_scripted_stdin_wrong_input(self):
        from backend.execution.service import execution_service
        p = _write_child(INTERACTIVE_CHILD)
        r = _run(execution_service.run_command(f'"{PY}" "{p}"', stdin="WRONG\n",
                                               timeout_seconds=30))
        self.assertIn("nope", r.stdout)

    def test_tool_manager_execute_raw_command_stdin(self):
        from backend.tools.manager import tool_manager
        p = _write_child(INTERACTIVE_CHILD)
        r = _run(tool_manager.execute_raw_command(f'"{PY}" "{p}"', stdin="RETURN 0\n",
                                                  timeout_seconds=30))
        self.assertEqual(r.status, "SUCCESS")
        self.assertIn("picoCTF{flag_hunters_interactive}", r.stdout)

    def test_no_stdin_is_unchanged_behaviour(self):
        # A command with no stdin still runs as a normal one-shot (regression guard).
        from backend.execution.service import execution_service
        r = _run(execution_service.run_command(f'"{PY}" -c "print(2+2)"', timeout_seconds=30))
        self.assertEqual(r.status, "SUCCESS")
        self.assertIn("4", r.stdout)

    def test_runtime_action_stdin_path(self):
        # The AgentRuntime's RealToolExecutor threads Action.stdin through to the process.
        from backend.agent_runtime.runtime import RealToolExecutor
        from backend.agent_runtime.action import Action, ActionType

        async def _auto_approve_gate(cmd, **kwargs):
            # This test exercises stdin plumbing, not the approval policy, so it injects
            # an operator who approves. RealToolExecutor's default is the real gate.
            return True, "approve", None

        p = _write_child(INTERACTIVE_CHILD)
        ex = RealToolExecutor(approval_gate=_auto_approve_gate)
        res = _run(ex.execute(Action(type=ActionType.COMMAND, command=f'"{PY}" "{p}"',
                                     stdin="RETURN 0\n"), timeout_seconds=30))
        self.assertIn("picoCTF{flag_hunters_interactive}", res.stdout)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Tier-2 persistent interactive execution
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractiveSession(unittest.TestCase):

    def tearDown(self):
        from backend.execution.interactive import interactive_manager
        _run(interactive_manager.close_all())

    def test_start_read_send_read_close(self):
        from backend.execution.interactive import interactive_manager

        async def scenario():
            sess = await interactive_manager.open(f'"{PY}" "{_write_child(INTERACTIVE_CHILD)}"',
                                                  idle_timeout=1.0, read_timeout=8)
            prompt = await sess.read(until="Enter command:")
            self.assertTrue(prompt.matched)
            self.assertIn("Enter command:", prompt.data)
            resp = await sess.send_and_read("RETURN 0", until="picoCTF")
            self.assertTrue(resp.matched)
            self.assertIn("picoCTF{flag_hunters_interactive}", resp.data)
            await sess.close()
            self.assertFalse(sess.is_alive())
        _run(scenario())

    def test_multi_turn_dialogue(self):
        from backend.execution.interactive import interactive_manager

        async def scenario():
            sess = await interactive_manager.open(f'"{PY}" "{_write_child(ECHO_CHILD)}"',
                                                  idle_timeout=1.0, read_timeout=8)
            r1 = await sess.send_and_read("alpha", until="echo:alpha")
            self.assertIn("echo:alpha", r1.data)
            r2 = await sess.send_and_read("beta", until="echo:beta")
            self.assertIn("echo:beta", r2.data)
            await sess.close()
        _run(scenario())

    def test_cleanup_no_orphans(self):
        from backend.execution.interactive import interactive_manager
        from backend.execution.process_manager import process_manager

        async def scenario():
            sess = await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"',
                                                  idle_timeout=0.5, read_timeout=2)
            pid = sess.pid
            self.assertIsNotNone(pid)
            self.assertIn(pid, process_manager.active_pids())
            await sess.close()
            self.assertFalse(sess.is_alive())
            self.assertNotIn(pid, process_manager.active_pids())
            self.assertEqual(interactive_manager.count(), 0)  # self-removed on close
        _run(scenario())

    def test_session_limit_enforced(self):
        from backend.execution.interactive import InteractiveSessionManager

        async def scenario():
            mgr = InteractiveSessionManager(max_sessions=1)
            s1 = await mgr.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            with self.assertRaises(RuntimeError):
                await mgr.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            n = await mgr.close_all()
            self.assertGreaterEqual(n, 1)
        _run(scenario())

    def test_trajectory_events_emitted(self):
        from backend.execution.interactive import (
            interactive_manager, EVENT_START, EVENT_SEND, EVENT_READ, EVENT_CLOSE)
        events = []

        async def scenario():
            sess = await interactive_manager.open(
                f'"{PY}" "{_write_child(INTERACTIVE_CHILD)}"', idle_timeout=1.0, read_timeout=8,
                recorder=lambda et, payload: events.append(et))
            await sess.read(until="Enter command:")
            await sess.send_and_read("RETURN 0", until="picoCTF")
            await sess.close()
        _run(scenario())
        for expected in (EVENT_START, EVENT_SEND, EVENT_READ, EVENT_CLOSE):
            self.assertIn(expected, events)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Interactive timeout — cannot hang forever
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractiveTimeout(unittest.TestCase):

    def tearDown(self):
        from backend.execution.interactive import interactive_manager
        _run(interactive_manager.close_all())

    def test_idle_read_returns_without_hanging(self):
        from backend.execution.interactive import interactive_manager, IO_TIMEOUT

        async def scenario():
            # A sleeping child produces no output — read must return on the idle gap,
            # not block the loop forever.
            sess = await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"',
                                                  idle_timeout=0.4, read_timeout=1.0)
            res = await sess.read()
            self.assertTrue(res.timed_out)
            self.assertEqual(res.status, IO_TIMEOUT)
            self.assertTrue(sess.is_alive())  # idle ≠ dead
            await sess.close()
        _run(scenario())

    def test_max_lifetime_enforced_and_cleaned_up(self):
        from backend.execution.interactive import interactive_manager, EVENT_TIMEOUT

        async def scenario():
            events = []
            sess = await interactive_manager.open(
                f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3, read_timeout=1.0,
                max_lifetime=0.3, recorder=lambda et, p: events.append(et))
            await asyncio.sleep(0.5)  # exceed the max lifetime
            res = await sess.read()   # next op must terminate + clean up the session
            self.assertTrue(res.timed_out)
            self.assertFalse(sess.is_alive())
            self.assertIn(EVENT_TIMEOUT, events)
        _run(scenario())

# ─────────────────────────────────────────────────────────────────────────────
# 4. Checkpoint safety — restartable spec, never a live handle
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractiveCheckpoint(unittest.TestCase):

    def tearDown(self):
        from backend.execution.interactive import interactive_manager
        _run(interactive_manager.close_all())

    def test_spec_is_serializable_without_handles(self):
        from backend.execution.interactive import interactive_manager, InteractiveSessionSpec

        async def scenario():
            sess = await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"',
                                                  target="nc host 1", idle_timeout=0.3)
            spec = sess.spec()
            d = spec.to_dict()
            json.dumps(d)  # must be JSON-serializable
            self.assertEqual(spec.state, "restartable")
            self.assertNotIn("_proc", d)
            self.assertNotIn("proc", d)
            self.assertEqual(d["command"], sess.command)
            self.assertIn(PY, d["command"])
            # round-trip
            restored = InteractiveSessionSpec.from_dict(d)
            self.assertEqual(restored.session_key, spec.session_key)
            self.assertEqual(restored.target, "nc host 1")
            await sess.close()
            self.assertEqual(sess.spec().state, "closed")
        _run(scenario())

    def test_mark_all_stale_terminates_and_does_not_orphan(self):
        # Phase 4.x hardening §2/§8 (Test A): the synchronous resume cleanup must
        # TERMINATE each still-live tracked process, not merely clear bookkeeping.
        # Previously this orphaned live processes and left stale ProcessManager rows.
        from backend.execution.interactive import interactive_manager
        from backend.execution.process_manager import process_manager

        async def scenario():
            await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            live = interactive_manager.all()
            pids = [s.pid for s in live]
            self.assertTrue(all(p in process_manager.active_pids() for p in pids))

            specs = interactive_manager.mark_all_stale()
            self.assertEqual(len(specs), 2)
            self.assertTrue(all(s.state == "restartable" for s in specs))
            self.assertEqual(interactive_manager.count(), 0)          # registry cleared
            # No orphans: every previously-tracked process is terminated + unregistered.
            for s, pid in zip(live, pids):
                self.assertFalse(s.is_alive())
                self.assertNotIn(pid, process_manager.active_pids())
        _run(scenario())

# ─────────────────────────────────────────────────────────────────────────────
# 5. Target-type detection
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetDetection(unittest.TestCase):

    def setUp(self):
        from backend.execution.targets import target_detector, TargetType
        self.det = target_detector
        self.T = TargetType

    def test_static_source_url(self):
        t = self.det.detect("https://challenge-files.picoctf.net/static/x/source.py")
        self.assertEqual(t.type, self.T.STATIC_FILE)

    def test_live_http(self):
        self.assertEqual(self.det.detect("http://target:5000").type, self.T.LIVE_HTTP)
        self.assertEqual(self.det.detect("https://ctf.example.com/login").type, self.T.LIVE_HTTP)

    def test_live_tcp_nc(self):
        t = self.det.detect("nc challenge.picoctf.net 12345")
        self.assertEqual(t.type, self.T.LIVE_TCP)
        self.assertEqual(t.host, "challenge.picoctf.net")
        self.assertEqual(t.port, 12345)

    def test_live_tcp_hostport(self):
        self.assertEqual(self.det.detect("challenge.example.com:1337").type, self.T.LIVE_TCP)

    def test_local_script(self):
        p = _write_child(INTERACTIVE_CHILD)
        self.assertEqual(self.det.detect(p).type, self.T.LOCAL_SCRIPT)

    def test_local_artifact(self):
        d = tempfile.mkdtemp()
        art = os.path.join(d, "capture.pcap")
        with open(art, "wb") as f:
            f.write(b"\xd4\xc3\xb2\xa1")
        self.assertEqual(self.det.detect(art).type, self.T.LOCAL_ARTIFACT)

    def test_remote_service_bare_ip(self):
        self.assertEqual(self.det.detect("10.10.14.23").type, self.T.REMOTE_SERVICE)

    def test_unknown_is_conservative(self):
        t = self.det.detect("just some words that are not a target")
        self.assertEqual(t.type, self.T.UNKNOWN)

    def test_multi_target_plus_delimiter(self):
        # FORGE joins multiple targets with '+' (no_demo_data rule #4).
        spec = "http://target.ctf:8080 + /tmp/artifact.pcap + 10.10.14.23"
        targets = self.det.detect_multi(spec)
        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[0].type, self.T.LIVE_HTTP)
        self.assertEqual(targets[2].type, self.T.REMOTE_SERVICE)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Target mismatch
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetMismatch(unittest.TestCase):

    def setUp(self):
        from backend.execution.targets import target_detector, TargetType
        self.det = target_detector
        self.T = TargetType

    def test_live_required_static_provided(self):
        provided = self.det.detect("https://challenge-files.picoctf.net/x/source.py")
        mism = self.det.classify_mismatch(self.T.LIVE_TCP, provided)
        self.assertIsNotNone(mism)
        self.assertEqual(mism.expected, self.T.LIVE_TCP)
        self.assertEqual(mism.observed, self.T.STATIC_FILE)
        self.assertEqual(mism.recommended_action, "request_live_target")

    def test_no_mismatch_when_both_live(self):
        provided = self.det.detect("http://target:5000")
        self.assertIsNone(self.det.classify_mismatch(self.T.LIVE_TCP, provided))

    def test_no_mismatch_when_unknown(self):
        provided = self.det.detect("http://target:5000")
        self.assertIsNone(self.det.classify_mismatch(self.T.UNKNOWN, provided))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Capability discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestCapabilityDiscovery(unittest.TestCase):

    def test_installed_capability(self):
        from backend.execution.capabilities import CapabilityService, AVAILABLE
        cap = CapabilityService().discover("python_exec")
        self.assertTrue(cap.available)
        self.assertEqual(cap.status, AVAILABLE)
        self.assertTrue(cap.provider)

    def test_unknown_capability_backend_unsupported(self):
        from backend.execution.capabilities import CapabilityService, UNKNOWN
        cap = CapabilityService().discover("teleportation")
        self.assertFalse(cap.available)
        self.assertEqual(cap.status, UNKNOWN)
        self.assertFalse(cap.backend_supported)
        self.assertEqual(cap.recommended_action, "replan")

    def test_alternative_provider_selected_when_preferred_absent(self):
        # Registry: preferred provider missing, an alternative present → ALTERNATIVE_AVAILABLE.
        from backend.execution.capabilities import (
            CapabilityService, ProviderSpec, ProviderKind, ALTERNATIVE_AVAILABLE)
        registry = {"demo_cap": [
            ProviderSpec("ghost_tool", ProviderKind.TOOL, binary="definitely_not_installed_xyz"),
            ProviderSpec("py_socket", ProviderKind.BUILTIN),   # builtin ⇒ present
        ]}
        cap = CapabilityService(registry=registry).discover("demo_cap")
        self.assertTrue(cap.available)
        self.assertEqual(cap.status, ALTERNATIVE_AVAILABLE)
        self.assertEqual(cap.provider, "py_socket")

    def test_blocked_capability_when_nothing_available_or_acquirable(self):
        from backend.execution.capabilities import (
            CapabilityService, ProviderSpec, ProviderKind, BLOCKED)

        class NoAcqPlanner:
            def can_acquire(self, spec):
                return False

        registry = {"hard_cap": [ProviderSpec("ghost", ProviderKind.TOOL, binary="nope_not_here_xyz")]}
        svc = CapabilityService(registry=registry)
        with patch("backend.execution.acquisition.acquisition_planner", NoAcqPlanner()):
            cap = svc.discover("hard_cap")
        self.assertFalse(cap.available)
        self.assertEqual(cap.status, BLOCKED)
        self.assertEqual(cap.recommended_action, "replan")

    def test_alternatives_listing(self):
        from backend.execution.capabilities import CapabilityService, ProviderSpec, ProviderKind
        registry = {"multi": [
            ProviderSpec("b1", ProviderKind.BUILTIN),
            ProviderSpec("b2", ProviderKind.BUILTIN),
        ]}
        alts = CapabilityService(registry=registry).alternatives("multi")
        self.assertIn("b1", alts)
        self.assertIn("b2", alts)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Controlled acquisition (privilege-gated; nothing installed for real)
# ─────────────────────────────────────────────────────────────────────────────

class TestControlledAcquisition(unittest.TestCase):

    def test_plan_pylib_is_user_level_when_pip_present(self):
        from backend.execution.acquisition import AcquisitionPlanner, AcquisitionMethod
        from backend.execution.capabilities import ProviderSpec, ProviderKind
        planner = AcquisitionPlanner()
        spec = ProviderSpec("pytesseract", ProviderKind.PYLIB, module="pytesseract",
                            install_recipe="pip install pytesseract")
        with patch.object(planner, "_pip", return_value="pip"):
            plan = planner.plan_for_spec(spec)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.method, AcquisitionMethod.USER_LEVEL)
        self.assertIn("--user", plan.command)
        self.assertEqual(plan.privilege_level, "PRIVILEGED")

    def test_tool_not_acquirable_without_package_manager(self):
        from backend.execution.acquisition import AcquisitionPlanner, AcquisitionMethod
        from backend.execution.capabilities import ProviderSpec, ProviderKind
        planner = AcquisitionPlanner()
        spec = ProviderSpec("nmap", ProviderKind.TOOL, binary="nmap")
        with patch.object(planner, "detect_package_manager", return_value=""):
            plan = planner.plan_for_spec(spec)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.method, AcquisitionMethod.UNAVAILABLE)

    def test_acquire_denied_when_not_approved(self):
        from backend.execution.acquisition import AcquisitionPlanner
        from backend.execution.capabilities import ProviderSpec, ProviderKind
        planner = AcquisitionPlanner()
        registry_specs = [ProviderSpec("pytesseract", ProviderKind.PYLIB, module="pytesseract",
                                       install_recipe="pip install pytesseract")]
        with patch("backend.execution.capabilities.capability_service") as cs, \
             patch.object(planner, "_pip", return_value="pip"):
            cs.provider_specs.return_value = registry_specs
            res = _run(planner.acquire("ocr", privilege_decider=lambda plan: False))
        self.assertTrue(res.requested)
        self.assertFalse(res.approved)
        self.assertFalse(res.executed)   # nothing runs without approval

    def test_acquire_approved_runs_via_injected_installer(self):
        from backend.execution.acquisition import AcquisitionPlanner
        from backend.execution.capabilities import ProviderSpec, ProviderKind

        class OKExec:
            succeeded = True
            status = "SUCCESS"
            stdout = "Successfully installed pytesseract"
            stderr = ""

        installer_calls = []

        async def fake_installer(command):
            installer_calls.append(command)
            return OKExec()

        planner = AcquisitionPlanner()
        specs = [ProviderSpec("pytesseract", ProviderKind.PYLIB, module="pytesseract",
                              install_recipe="pip install pytesseract")]
        with patch("backend.execution.capabilities.capability_service") as cs, \
             patch.object(planner, "_pip", return_value="pip"):
            cs.provider_specs.return_value = specs
            res = _run(planner.acquire("ocr", privilege_decider=lambda plan: True,
                                       installer=fake_installer))
        self.assertTrue(res.approved)
        self.assertTrue(res.executed)
        self.assertTrue(res.success)
        self.assertEqual(len(installer_calls), 1)
        self.assertIn("pip install --user", installer_calls[0])

# ─────────────────────────────────────────────────────────────────────────────
# 9. OCR capability + Binary Digits regression
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRCapability(unittest.TestCase):

    def _fake_image(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "digits.jpg")
        with open(p, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 32)  # JPEG magic + filler
        return p

    def test_binary_digits_ocr_blocked_is_structured_not_retried(self):
        # Scenario: binary → JPEG generated, but no OCR provider available.  FORGE must
        # return a STRUCTURED blocked result (replan) — not retry tesseract 40 times.
        from backend.execution.ocr import OCRService, OCR_BLOCKED
        fake_caps = FakeCapabilityService({
            "ocr": FakeCapability("ocr", "BLOCKED", False, reason="no OCR provider",
                                  action="replan")})
        svc = OCRService(capability_service=fake_caps)
        img = self._fake_image()
        res = _run(svc.extract_text(img))
        self.assertEqual(res.status, OCR_BLOCKED)
        self.assertTrue(res.blocked)
        self.assertEqual(res.recommended_action, "replan")
        self.assertEqual(res.provider, "")
        # Deterministic single discovery — not an unbounded retry loop.
        self.assertEqual(fake_caps.discover_calls.count("ocr"), 1)

    def test_ocr_available_extracts_text_via_tesseract(self):
        from backend.execution.ocr import OCRService, OCR_OK

        class OKExec:
            succeeded = True
            status = "SUCCESS"
            stdout = "picoCTF{ocr_capability}"
            stderr = ""

        fake_caps = FakeCapabilityService({
            "ocr": FakeCapability("ocr", "AVAILABLE", True, provider="tesseract",
                                  action="execute")})
        svc = OCRService(capability_service=fake_caps)
        img = self._fake_image()
        with patch("backend.execution.service.execution_service.run_command",
                   new=AsyncMock(return_value=OKExec())):
            res = _run(svc.extract_text(img))
        self.assertEqual(res.status, OCR_OK)
        self.assertEqual(res.provider, "tesseract")
        self.assertIn("picoCTF{ocr_capability}", res.text)

    def test_ocr_missing_image(self):
        from backend.execution.ocr import OCRService, OCR_NO_INPUT
        svc = OCRService(capability_service=FakeCapabilityService({}))
        res = _run(svc.extract_text("/no/such/file.jpg"))
        self.assertEqual(res.status, OCR_NO_INPUT)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Swarm integration — capability/target gate, no infinite retry
# ─────────────────────────────────────────────────────────────────────────────

class TestSwarmCapabilityGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def _coord(self, *, capability_service=None, acquisition_planner=None, target="",
               agent_probe=None):
        from backend.swarm.coordinator import SwarmCoordinator
        from backend.swarm.limits import SwarmLimits

        def probe_factory(role):
            if agent_probe is not None:
                agent_probe.append(role)
            raise AssertionError("A blocked task must NOT spawn an agent.")

        return SwarmCoordinator(
            target=target, persist=False, limits=SwarmLimits(max_total_tasks=40),
            capability_service=capability_service, acquisition_planner=acquisition_planner,
            agent_factory=probe_factory)

    def test_blocked_capability_no_retry_and_suppressed(self):
        from backend.swarm.tasks import Task, TaskStatus
        from backend.swarm.evidence import EvidenceType
        fake_caps = FakeCapabilityService({
            "ocr": FakeCapability("ocr", "BLOCKED", False, reason="no OCR provider here",
                                  action="replan")})
        spawned = []
        coord = self._coord(capability_service=fake_caps, agent_probe=spawned)

        # Two tasks needing the same blocked capability.
        for i in range(2):
            coord._add_task(Task(mission_id=coord.mission_id, role="forensics",
                                 objective=f"Analyze image {i}", required_capabilities=["ocr"]),
                            origin="plan")
        coord.mission.status = "RUNNING"
        _run(coord._loop())

        tasks = coord.scheduler.all()
        self.assertEqual(len(tasks), 2)
        self.assertTrue(all(t.status == TaskStatus.FAILED.value for t in tasks))
        self.assertEqual(spawned, [])                       # no agent ever ran
        self.assertIn("ocr", coord._blocked_capabilities)   # remembered → suppressed
        # A structured capability evidence was published (deduped to one signature).
        self.assertGreaterEqual(len(coord.bus.by_type(EvidenceType.CAPABILITY.value)), 1)
        # Dead end recorded, and discovery was NOT called an unbounded number of times.
        self.assertTrue(coord.mission.dead_ends)
        self.assertLessEqual(fake_caps.discover_calls.count("ocr"), 2)

    def test_acquirable_but_not_approved_blocks_once(self):
        from backend.swarm.tasks import Task, TaskStatus
        from backend.execution.acquisition import AcquisitionResult
        fake_caps = FakeCapabilityService({
            "ocr": FakeCapability("ocr", "ACQUIRABLE", False, reason="acquirable via pip",
                                  action="request_acquisition", acq=True)})
        fake_acq = FakeAcquisitionPlanner(
            AcquisitionResult(capability="ocr", provider="pytesseract", requested=True,
                              approved=False, executed=False, reason="pending approval"))
        coord = self._coord(capability_service=fake_caps, acquisition_planner=fake_acq)
        for i in range(2):
            coord._add_task(Task(mission_id=coord.mission_id, role="forensics",
                                 objective=f"OCR the image {i}", required_capabilities=["ocr"]),
                            origin="plan")
        coord.mission.status = "RUNNING"
        _run(coord._loop())
        tasks = coord.scheduler.all()
        self.assertTrue(all(t.status == TaskStatus.FAILED.value for t in tasks))
        # Acquisition attempted exactly ONCE across both tasks (then remembered/suppressed).
        self.assertEqual(fake_acq.acquire_calls, ["ocr"])

    def test_available_capability_allows_dispatch(self):
        # When the capability IS available, the gate does not block — the agent runs.
        from backend.swarm.tasks import Task, TaskStatus
        fake_caps = FakeCapabilityService({
            "ocr": FakeCapability("ocr", "AVAILABLE", True, provider="tesseract", action="execute")})
        ran = []

        class DoneAgent:
            def __init__(self, role): self.role = role
            async def execute(self, task, mission, bus, **kw):
                from backend.swarm.agents import AgentResult
                ran.append(task.id)
                return AgentResult(task_id=task.id, role=getattr(self.role, "value", str(self.role)),
                                   status="COMPLETED", reason="done")

        from backend.swarm.coordinator import SwarmCoordinator
        from backend.swarm.limits import SwarmLimits
        coord = SwarmCoordinator(persist=False, limits=SwarmLimits(),
                                 capability_service=fake_caps,
                                 agent_factory=lambda role: DoneAgent(role))
        coord._add_task(Task(mission_id=coord.mission_id, role="forensics",
                             objective="OCR the image", required_capabilities=["ocr"]),
                        origin="plan")
        coord.mission.status = "RUNNING"
        _run(coord._loop())
        self.assertEqual(len(ran), 1)
        self.assertEqual(coord.scheduler.all()[0].status, TaskStatus.COMPLETED.value)

    def test_target_mismatch_blocks_task(self):
        from backend.swarm.tasks import Task, TaskStatus
        from backend.swarm.evidence import EvidenceType
        # Mission target is a static source file, but the task requires a live TCP service.
        coord = self._coord(target="https://challenge-files.picoctf.net/x/source.py")
        coord._add_task(Task(mission_id=coord.mission_id, role="pwn",
                             objective="Connect and exploit the service",
                             target_type="LIVE_TCP"), origin="plan")
        coord.mission.status = "RUNNING"
        _run(coord._loop())
        t = coord.scheduler.all()[0]
        self.assertEqual(t.status, TaskStatus.FAILED.value)
        self.assertIn("TARGET_MISMATCH", t.failure_reason)
        self.assertGreaterEqual(len(coord.bus.by_type(EvidenceType.TARGET_MISMATCH.value)), 1)

    def test_supervisor_abandons_blocked_and_mismatch(self):
        from backend.swarm.supervisor import Supervisor
        from backend.swarm.tasks import Task

        class R:
            def __init__(self, cat): self.status = "FAILED"; self.failure_category = cat; self.reason = ""

        sup = Supervisor("m")
        t = Task(mission_id="m", role="forensics", objective="x")
        self.assertEqual(sup.classify_failure(R("BLOCKED_CAPABILITY")), "blocked_capability")
        self.assertEqual(sup.classify_failure(R("TARGET_MISMATCH")), "target_mismatch")
        self.assertEqual(sup.decide_recovery(t, R("BLOCKED_CAPABILITY"), max_retries=2).action, "abandon")
        self.assertEqual(sup.decide_recovery(t, R("TARGET_MISMATCH"), max_retries=2).action, "abandon")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Platform behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestPlatformBehaviour(unittest.TestCase):

    def test_pty_gated_by_platform(self):
        from backend.execution.interactive import InteractiveSession
        sess = InteractiveSession("echo hi", use_pty=True)
        self.assertEqual(sess.use_pty, not _IS_WINDOWS)

    def test_pty_capability_matches_platform(self):
        from backend.execution.capabilities import CapabilityService
        cap = CapabilityService().discover("pty")
        self.assertEqual(cap.available, not _IS_WINDOWS)

    def test_normalise_command_handles_python(self):
        from backend.execution.interactive import _normalise_command
        out = _normalise_command("python3 solve.py")
        self.assertTrue(out.endswith("solve.py"))
        self.assertEqual(_normalise_command(""), "")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Trajectory integration for interactive events
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryInteractiveEvents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_interactive_events_record_into_trajectory(self):
        from backend.agent_runtime import session_manager, trajectory_store
        from backend.execution.interactive import EVENT_START, EVENT_SEND, EVENT_READ, EVENT_CLOSE
        sess = session_manager.create(objective="interactive", agent_id="pwn")
        # Wire an interactive recorder to the ONE trajectory system (no second store).
        def recorder(event_type, payload):
            trajectory_store.record(session_id=sess.id, event_type=event_type,
                                    agent_id="pwn", command=payload.get("data", ""),
                                    decision_summary=payload.get("preview", ""))
        for et in (EVENT_START, EVENT_SEND, EVENT_READ, EVENT_CLOSE):
            recorder(et, {"data": "RETURN 0", "preview": "picoCTF{...}"})
        recent = trajectory_store.get_recent(sess.id, n=10,
                                             event_types=[EVENT_START, EVENT_SEND, EVENT_READ, EVENT_CLOSE])
        kinds = {e.event_type for e in recent}
        self.assertIn(EVENT_START, kinds)
        self.assertIn(EVENT_CLOSE, kinds)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Phase 4.x HARDENING — interactive process lifecycle across checkpoint/resume
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractiveResumeLifecycle(unittest.TestCase):
    """Checkpoint/resume must never orphan or duplicate an interactive process."""

    def tearDown(self):
        from backend.execution.interactive import interactive_manager
        _run(interactive_manager.close_all())

    def test_reset_for_resume_terminates_no_orphan(self):
        # Test A: the async resume primitive terminates + reaps every tracked process.
        from backend.execution.interactive import interactive_manager
        from backend.execution.process_manager import process_manager

        async def scenario():
            await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            live = interactive_manager.all()
            pids = [s.pid for s in live]
            specs = await interactive_manager.reset_for_resume()
            self.assertEqual(len(specs), 2)
            self.assertTrue(all(sp.state == "restartable" for sp in specs))
            self.assertEqual(interactive_manager.count(), 0)
            for s, pid in zip(live, pids):
                self.assertFalse(s.is_alive())
                self.assertIsNotNone(s.returncode)          # child reaped → truly dead
                self.assertNotIn(pid, process_manager.active_pids())
        _run(scenario())

    def test_resume_does_not_duplicate_process(self):
        # Test C: after resume-cleanup + recreate there is exactly ONE live process
        # for the same logical session — never old + new.
        from backend.execution.interactive import interactive_manager
        from backend.execution.process_manager import process_manager

        async def scenario():
            cmd = f'"{PY}" "{_write_child(ECHO_CHILD)}"'
            s1 = await interactive_manager.open(cmd, session_key="dup-key", idle_timeout=0.3)
            pid1 = s1.pid
            # Interruption → resume cleanup terminates the old process.
            await interactive_manager.reset_for_resume()
            self.assertFalse(s1.is_alive())
            self.assertNotIn(pid1, process_manager.active_pids())
            # Recreate the same logical session (as a resume would, from its spec).
            s2 = await interactive_manager.open(cmd, session_key="dup-key", idle_timeout=0.3)
            pid2 = s2.pid
            self.assertNotEqual(pid1, pid2)                    # a genuinely new OS process
            self.assertTrue(s2.is_alive())
            self.assertEqual(interactive_manager.count(), 1)   # exactly one, not two
            self.assertIn(pid2, process_manager.active_pids())
            self.assertNotIn(pid1, process_manager.active_pids())
            await s2.close()
        _run(scenario())

    def test_cleanup_is_idempotent(self):
        # Test D: repeated cleanup must not crash and must leave a consistent state.
        from backend.execution.interactive import interactive_manager

        async def scenario():
            sess = await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            first = await interactive_manager.reset_for_resume()
            self.assertEqual(len(first), 1)
            second = await interactive_manager.reset_for_resume()      # nothing left
            self.assertEqual(second, [])
            self.assertEqual(interactive_manager.mark_all_stale(), [])  # sync variant too
            await sess.close()                                         # double close is safe
            await sess.close()
            self.assertEqual(interactive_manager.count(), 0)
        _run(scenario())

class TestInteractiveRegistryHygiene(unittest.TestCase):
    """The ProcessManager registry must reflect real process state — no stale rows."""

    def tearDown(self):
        from backend.execution.interactive import interactive_manager
        _run(interactive_manager.close_all())

    def test_checkpoint_snapshot_does_not_terminate(self):
        # Test B: taking a checkpoint spec/snapshot must NOT kill a resumable session.
        from backend.execution.interactive import interactive_manager
        from backend.execution.process_manager import process_manager

        async def scenario():
            sess = await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            pid = sess.pid
            specs = interactive_manager.snapshot_specs()
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0]["state"], "restartable")
            # The live process is untouched by checkpointing.
            self.assertTrue(sess.is_alive())
            self.assertEqual(interactive_manager.count(), 1)
            self.assertIn(pid, process_manager.active_pids())
            await sess.close()
        _run(scenario())

    def test_self_exited_process_is_unregistered(self):
        # Test E: a process that exits on its own must be cleaned out of the registry,
        # and cleanup on an already-dead process must be graceful (no crash).
        from backend.execution.interactive import interactive_manager
        from backend.execution.process_manager import process_manager

        async def scenario():
            sess = await interactive_manager.open(f'"{PY}" "{_write_child(QUICK_CHILD)}"',
                                                  idle_timeout=0.5, read_timeout=2)
            pid = sess.pid
            r = await sess.read()                       # drains 'bye' then hits EOF
            self.assertTrue(r.eof or "bye" in r.data)
            for _ in range(40):                         # let the child watcher set returncode
                if not sess.is_alive():
                    break
                await asyncio.sleep(0.05)
            interactive_manager._reap()
            self.assertNotIn(pid, process_manager.active_pids())    # no stale entry left
            self.assertEqual(interactive_manager.count(), 0)
            await sess.close()                          # already dead → must not raise
        _run(scenario())

    def test_terminate_pid_only_acts_on_tracked_processes(self):
        # Security (§12): terminate_pid must NEVER signal an untracked/foreign PID.
        from backend.execution.process_manager import process_manager
        self.assertFalse(process_manager.terminate_pid(2_147_480_000))  # not tracked → no-op

class TestAppShutdownInteractiveCleanup(unittest.TestCase):
    """Test I — tracked interactive processes must not survive a controlled shutdown."""

    def test_close_all_terminates_tracked_sessions(self):
        # main.py's @app.on_event("shutdown") handler calls exactly this close_all().
        from backend.execution.interactive import interactive_manager
        from backend.execution.process_manager import process_manager

        async def scenario():
            await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            await interactive_manager.open(f'"{PY}" "{_write_child(SLEEP_CHILD)}"', idle_timeout=0.3)
            live = interactive_manager.all()
            pids = [s.pid for s in live]
            closed = await interactive_manager.close_all(reason="app_shutdown")
            self.assertGreaterEqual(closed, 2)
            self.assertEqual(interactive_manager.count(), 0)
            for s, pid in zip(live, pids):
                self.assertFalse(s.is_alive())
                self.assertIsNotNone(s.returncode)          # truly terminated
                self.assertNotIn(pid, process_manager.active_pids())
        _run(scenario())

class TestTaskMetadataResume(unittest.TestCase):
    """Phase 4.x hardening §5/§6 — coordination metadata + gates survive resume."""

    @classmethod
    def setUpClass(cls):
        init_db()

    def _cleanup(self, mission_id):
        from backend.database.session import SessionLocal
        from backend.database.models import SwarmTaskModel, SwarmMissionModel
        db = SessionLocal()
        try:
            db.query(SwarmTaskModel).filter(SwarmTaskModel.mission_id == mission_id).delete()
            db.query(SwarmMissionModel).filter(SwarmMissionModel.id == mission_id).delete()
            db.commit()
        finally:
            db.close()

    def test_task_metadata_persists_and_reloads(self):
        # Test F: required_capabilities + target_type survive a save/reload round-trip,
        # and a stale RUNNING task is reset for re-dispatch WITHOUT losing that metadata.
        import uuid as _uuid
        from backend.swarm.tasks import Task, TaskStatus
        from backend.swarm.scheduler import TaskScheduler
        mid = str(_uuid.uuid4())
        try:
            t = Task(mission_id=mid, role="pwn", objective="Exploit the service",
                     required_capabilities=["interactive_tcp"], target_type="LIVE_TCP",
                     status=TaskStatus.RUNNING.value)
            t.save()
            sched = TaskScheduler(persist=False)
            n = sched.load(mid)
            self.assertEqual(n, 1)
            reloaded = sched.get(t.id)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.required_capabilities, ["interactive_tcp"])
            self.assertEqual(reloaded.target_type, "LIVE_TCP")
            # stale in-flight task is re-dispatchable (never left RUNNING), metadata intact
            self.assertIn(reloaded.status, (TaskStatus.PENDING.value, TaskStatus.READY.value))
            self.assertNotEqual(reloaded.status, TaskStatus.RUNNING.value)
        finally:
            self._cleanup(mid)

    def test_capability_gate_after_resume(self):
        # Test G: a resumed task whose required capability is unavailable is still
        # blocked at the pre-dispatch gate — no agent is spawned.
        import uuid as _uuid
        from backend.swarm.coordinator import SwarmCoordinator
        from backend.swarm.limits import SwarmLimits
        from backend.swarm.tasks import Task, TaskStatus
        from backend.swarm.evidence import EvidenceType
        mid = str(_uuid.uuid4())
        fake_caps = FakeCapabilityService({
            "ocr": FakeCapability("ocr", "BLOCKED", False, reason="no OCR provider here",
                                  action="replan")})

        def probe(role):
            raise AssertionError("A blocked task must NOT spawn an agent after resume.")

        try:
            c1 = SwarmCoordinator(persist=True, mission_id=mid, capability_service=fake_caps,
                                  limits=SwarmLimits(max_total_tasks=10), agent_factory=probe)
            c1._add_task(Task(mission_id=mid, role="forensics", objective="Analyze image",
                              required_capabilities=["ocr"]), origin="plan")
            c1._save_mission()

            # A brand-new coordinator resumes the same mission purely from persistence.
            c2 = SwarmCoordinator(persist=True, mission_id=mid, capability_service=fake_caps,
                                  limits=SwarmLimits(max_total_tasks=10), agent_factory=probe)
            c2._load_persisted()
            reloaded = c2.scheduler.all()
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded[0].required_capabilities, ["ocr"])   # survived resume
            c2.mission.status = "RUNNING"
            _run(c2._loop())
            t = c2.scheduler.all()[0]
            self.assertEqual(t.status, TaskStatus.FAILED.value)
            self.assertIn("ocr", c2._blocked_capabilities)
            self.assertGreaterEqual(len(c2.bus.by_type(EvidenceType.CAPABILITY.value)), 1)
        finally:
            self._cleanup(mid)

    def test_target_gate_after_resume(self):
        # Test H: a resumed task requiring a live target against a static artifact is
        # still reported as TARGET_MISMATCH — not attempted.
        import uuid as _uuid
        from backend.swarm.coordinator import SwarmCoordinator
        from backend.swarm.limits import SwarmLimits
        from backend.swarm.tasks import Task, TaskStatus
        from backend.swarm.evidence import EvidenceType
        mid = str(_uuid.uuid4())
        target = "https://challenge-files.picoctf.net/static/x/source.py"

        def probe(role):
            raise AssertionError("A mismatched task must NOT spawn an agent after resume.")

        try:
            c1 = SwarmCoordinator(target=target, persist=True, mission_id=mid,
                                  limits=SwarmLimits(max_total_tasks=10), agent_factory=probe)
            c1._add_task(Task(mission_id=mid, role="pwn", objective="Connect and exploit",
                              target_type="LIVE_TCP"), origin="plan")
            c1._save_mission()

            c2 = SwarmCoordinator(target=target, persist=True, mission_id=mid,
                                  limits=SwarmLimits(max_total_tasks=10), agent_factory=probe)
            c2._load_persisted()
            reloaded = c2.scheduler.all()
            self.assertEqual(reloaded[0].target_type, "LIVE_TCP")          # survived resume
            self.assertEqual(c2.mission.target, target)
            c2.mission.status = "RUNNING"
            _run(c2._loop())
            t = c2.scheduler.all()[0]
            self.assertEqual(t.status, TaskStatus.FAILED.value)
            self.assertIn("TARGET_MISMATCH", t.failure_reason)
            self.assertGreaterEqual(len(c2.bus.by_type(EvidenceType.TARGET_MISMATCH.value)), 1)
        finally:
            self._cleanup(mid)

if __name__ == "__main__":
    unittest.main()
