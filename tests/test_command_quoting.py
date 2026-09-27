"""
tests/test_command_quoting.py

Shell-injection hardening for registered tool argument templates.

A capability request carries a target (IP, hostname or URL) that originates from
challenge data, and the constructed command is handed to a shell
(ExecutionService.run_command -> asyncio.create_subprocess_shell).  If that target is
interpolated into `args_template` verbatim, a `;`, `|`, backtick or `$(...)` inside it
executes as a separate command.  `format_tool_args()` must therefore shell-quote every
value it substitutes, while leaving the literals already written into the template
(`-sV -F`, `-i -s`, `-n 8`, `-e`) untouched.

The strongest assertion here is on `shlex.split(command)`: it reproduces what the shell
would treat as separate words, so we can prove the payload arrives as ONE argument
rather than being split on a separator.
"""

import asyncio
import os
import shlex
import string
import sys
import unittest
from unittest.mock import AsyncMock, patch

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

from backend.tools import manager as manager_module
from backend.tools.manager import ToolManager, format_tool_args
from backend.tools.registry import tool_registry

MALICIOUS_TARGET = "10.0.0.1; rm -rf /tmp/test_marker"

class _AllInstalled(dict):
    """Environment stub: reports every registered tool binary as installed."""

    def get(self, key, default=None):
        return {"installed": True}

class _FakeExecResult:
    stdout = ""
    stderr = ""
    exit_code = 0
    status = "SUCCESS"

def _run(coro):
    return asyncio.run(coro)

class TestFormatToolArgsUnit(unittest.TestCase):
    """Direct unit tests of the interpolation helper (the smallest testable unit)."""

    def test_malicious_target_is_single_quoted_argument(self):
        args = format_tool_args("{target} -sV -F", target=MALICIOUS_TARGET)
        self.assertEqual(args, f"'{MALICIOUS_TARGET}' -sV -F")
        self.assertEqual(shlex.split(args), [MALICIOUS_TARGET, "-sV", "-F"])

    def test_template_literals_stay_unquoted(self):
        # Only the `.format()` keyword values may be quoted — the flags written into the
        # template itself must remain real flags, not a quoted blob.
        args = format_tool_args("{target} -sV -F", target="10.0.0.1")
        self.assertEqual(args, "10.0.0.1 -sV -F")
        self.assertNotIn("'-sV -F'", args)

    def test_wordlist_placeholder_is_quoted_too(self):
        # `{wordlist}` is the second external placeholder; a resolved path with a space
        # (or a hostile one) must not split into extra arguments either.
        args = format_tool_args(
            "-u {target}/FUZZ -w {wordlist}",
            target="http://10.0.0.1",
            wordlist="/tmp/my wordlists/common.txt",
        )
        self.assertEqual(args, "-u http://10.0.0.1/FUZZ -w '/tmp/my wordlists/common.txt'")
        self.assertEqual(
            shlex.split(args),
            ["-u", "http://10.0.0.1/FUZZ", "-w", "/tmp/my wordlists/common.txt"],
        )

    def test_benign_target_is_byte_identical_to_unquoted(self):
        # shlex.quote() must be a no-op for the ordinary values this path sees, so the
        # existing tool invocations are unchanged.
        for benign in ("10.0.0.1", "target.ctf", "http://10.0.0.1:8080/path", "/tmp/artifact.bin"):
            with self.subTest(benign=benign):
                self.assertEqual(
                    format_tool_args("{target} -sV -F", target=benign),
                    f"{benign} -sV -F",
                )

    def test_every_registry_target_template_quotes_a_payload(self):
        # Guards against a new tool being registered with an unquoted template.
        checked = 0
        for name, tool in tool_registry.tools.items():
            fields = [
                field_name
                for _, field_name, _, _ in string.Formatter().parse(tool.args_template)
                if field_name
            ]
            if "target" not in fields:
                continue
            checked += 1
            with self.subTest(tool=name):
                # Supply every placeholder the template declares, with the malicious value
                # going to {target}.
                values = {f: "benign" for f in fields}
                values["target"] = MALICIOUS_TARGET
                rendered = format_tool_args(tool.args_template, **values)
                argv = shlex.split(rendered)
                # The payload must ride inside exactly one argument. Some templates embed
                # the placeholder in a larger token (`{target}/FUZZ`), so check containment
                # per token rather than demanding an exact token match.
                carrying = [tok for tok in argv if MALICIOUS_TARGET in tok]
                self.assertEqual(len(carrying), 1, f"{name}: payload not a single argument: {argv}")
                self.assertNotIn(";", argv, f"{name}: separator survived as its own token: {argv}")
                self.assertNotIn("rm", argv, f"{name}: payload split into separate tokens: {argv}")
        self.assertGreater(checked, 0, "no {target} templates found to check")

class TestExecuteCapabilityQuoting(unittest.TestCase):
    """End-to-end through execute_capability: assert on the shell-bound command."""

    def _capture_command(self, capability, target, wordlist="/usr/share/wordlists/dirb/common.txt"):
        mgr = ToolManager()
        with patch.object(
            manager_module.environment_detector,
            "detect_environment",
            return_value={"installed_tools": _AllInstalled()},
        ), patch.object(
            manager_module.execution_service,
            "run_command",
            new=AsyncMock(return_value=_FakeExecResult()),
        ), patch(
            "backend.execution.wordlist.WordlistResolver.resolve",
            return_value=wordlist,
        ):
            result = _run(mgr.execute_capability(capability=capability, target=target))
        return result.command

    def test_nmap_target_payload_is_one_argument(self):
        command = self._capture_command("recon", MALICIOUS_TARGET)  # nmap is first for recon
        self.assertTrue(command.startswith("nmap "), command)
        argv = shlex.split(command)
        self.assertIn(MALICIOUS_TARGET, argv)
        self.assertNotIn(";", argv)
        self.assertNotIn("rm", argv)
        self.assertIn("-sV", argv)
        self.assertIn("-F", argv)

    def test_file_analysis_target_payload_is_one_argument(self):
        command = self._capture_command("file_analysis", MALICIOUS_TARGET)  # strings
        self.assertTrue(command.startswith("strings "), command)
        argv = shlex.split(command)
        self.assertIn(MALICIOUS_TARGET, argv)
        self.assertNotIn(";", argv)
        self.assertIn("-n", argv)
        self.assertIn("8", argv)

    def test_pipe_and_backtick_payloads_are_neutralised(self):
        for payload in (
            "10.0.0.1 | tee /tmp/test_marker",
            "10.0.0.1`id`",
            "10.0.0.1 && rm -rf /tmp/test_marker",
            "10.0.0.1$(id)",
        ):
            with self.subTest(payload=payload):
                argv = shlex.split(self._capture_command("file_analysis", payload))
                self.assertIn(payload, argv)
                self.assertNotIn("|", argv)
                self.assertNotIn("&&", argv)

    def test_ffuf_quotes_target_and_wordlist(self):
        # ffuf builds its args through the registry template, so both the target-derived
        # URL and the resolved wordlist path must be quoted.
        command = self._capture_command(
            "web_testing", MALICIOUS_TARGET, wordlist="/tmp/hostile wordlist.txt"
        )
        self.assertTrue(command.startswith("ffuf "), command)
        argv = shlex.split(command)
        self.assertIn("10.0.0.1; rm -rf /tmp/test_marker/FUZZ", argv)
        self.assertIn("/tmp/hostile wordlist.txt", argv)
        self.assertIn("200,301,302,401,403", argv)

    def test_benign_targets_produce_unchanged_commands(self):
        # Regression guard: quoting must not alter the ordinary invocations.
        self.assertEqual(
            self._capture_command("recon", "http://10.0.0.1:8080"),
            "nmap 10.0.0.1 -sV -F -p 8080",
        )
        self.assertEqual(
            self._capture_command("file_analysis", "target.ctf"),
            "strings -n 8 target.ctf",
        )
        self.assertEqual(
            self._capture_command("web_testing", "http://10.0.0.1"),
            "ffuf -u http://10.0.0.1/FUZZ -w /usr/share/wordlists/dirb/common.txt"
            " -mc 200,301,302,401,403 -s",
        )

if __name__ == "__main__":
    unittest.main()
