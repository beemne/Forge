"""Unit tests for the flexible full-context agent prompt template (Part 1)."""

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

from backend.agents.agent_prompt import AgentContext, build_agent_prompt, make_context_from_env
from backend.agents.artifact_classifier import ClassificationResult

class TestPromptBuilder(unittest.TestCase):
    def _ctx(self, **kw):
        base = dict(
            platform="PicoCTF", challenge_name="Crack the Gate", category="WEB",
            difficulty="EASY", description="Log in as the developer.",
            target_url="http://target.ctf:8080", detected_os="Kali Linux",
            tool_inventory="curl, nmap, ffuf", python_libs="requests",
            working_directory="/work", max_iterations=25, max_minutes=12,
            flag_pattern="picoCTF{...}|flag{...}",
        )
        base.update(kw)
        return AgentContext(**base)

    def test_budget_line_reflects_config(self):
        system, user = build_agent_prompt(self._ctx())
        self.assertIn("25 tool-call iterations", system)
        self.assertIn("12 minutes", system)
        self.assertIn("BUDGET_EXHAUSTED", system)

    def test_flag_is_validation_filter_not_target(self):
        system, _ = build_agent_prompt(self._ctx())
        self.assertIn("VALIDATION FILTER", system)
        self.assertIn("picoCTF{...}", system)
        # The wording must forbid constructing a string to match.
        self.assertIn("Never construct a string to fit this pattern", system)

    def test_full_context_in_user_prompt(self):
        _, user = build_agent_prompt(self._ctx())
        for token in ("Crack the Gate", "WEB", "http://target.ctf:8080", "Kali Linux", "curl, nmap, ffuf"):
            self.assertIn(token, user)

    def test_binary_mode_block_present_when_classified_binary(self):
        clf = ClassificationResult(
            is_binary=True, reason="octet-stream", artifact_type="elf",
            recommended_tools=["file", "objdump", "radare2"], safe_file_path="/work/artifact.bin",
        )
        ctx = self._ctx(attached_file_paths=["/work/artifact.bin"], artifact_classification=clf)
        _, user = build_agent_prompt(ctx)
        self.assertIn("BINARY ARTIFACT MODE", user)
        self.assertIn("/work/artifact.bin", user)
        self.assertIn("objdump", user)
        # Must forbid re-download and mandate byte-safe tooling.
        self.assertIn("Do NOT re-download", user)
        self.assertIn("open(path, 'rb')", user)

    def test_no_binary_block_for_web(self):
        _, user = build_agent_prompt(self._ctx())
        self.assertNotIn("BINARY ARTIFACT MODE", user)

    def test_injected_directive_surfaced(self):
        ctx = self._ctx(injected_directive="Try the X-Dev-Access: yes header on /login.")
        _, user = build_agent_prompt(ctx)
        self.assertIn("OPERATOR DIRECTIVE", user)
        self.assertIn("X-Dev-Access", user)

class TestContextFactory(unittest.TestCase):
    def test_make_context_from_env_maps_installed_only(self):
        env = {
            "distro": "Parrot OS",
            "installed_tools": {"curl": {"installed": True}, "nmap": {"installed": False}, "ffuf": {"installed": True}},
            "installed_python_libs": {"requests": True, "pwn": False},
        }
        ctx = make_context_from_env(
            env_info=env, challenge_name="c", platform="p", category="REV", difficulty="EASY",
            description="d", target_url="", working_directory="/w",
            max_iterations=40, max_minutes=30, flag_pattern="flag{...}",
        )
        self.assertIn("curl", ctx.tool_inventory)
        self.assertIn("ffuf", ctx.tool_inventory)
        self.assertNotIn("nmap", ctx.tool_inventory)          # not installed → excluded
        self.assertIn("requests", ctx.python_libs)
        self.assertNotIn("pwn", ctx.python_libs)
        self.assertEqual(ctx.detected_os, "Parrot OS")

if __name__ == "__main__":
    unittest.main()
