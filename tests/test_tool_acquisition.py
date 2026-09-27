import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.config import settings
from backend.database.session import init_db
from backend.environment.detector import environment_detector
from backend.execution.acquisition import (
    AcquisitionMethod,
    AcquisitionPlanner,
    acquisition_planner,
)
from backend.privilege.gate import SHARED_PENDING_APPROVALS
from backend.tools.manager import (
    ToolManager,
    find_tool_binary,
    refresh_environment_path,
)
from backend.tools.registry import ToolMetadata, ToolRegistry, tool_registry

class TestToolAcquisitionAndResolution(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        init_db()
        SHARED_PENDING_APPROVALS.clear()
        self.tool_manager = ToolManager()

    def test_system_audit_reuses_tool_registry(self):
        """Audit on /system should reuse ToolRegistry's arsenal tools without duplication."""
        audit = environment_detector.perform_requirements_audit()
        self.assertIn("tool_requirements", audit)
        req_names = [t["name"] for t in audit["tool_requirements"]]
        self.assertIn("nmap", req_names)
        self.assertIn("ffuf", req_names)
        self.assertIn("sqlmap", req_names)
        self.assertIn("cyberchef", req_names)
        self.assertIn("volatility3", req_names)
        self.assertIn("claude", req_names)
        self.assertIn("codex", req_names)
        self.assertGreaterEqual(len(req_names), 25)

    def test_dynamic_plan_for_known_and_arbitrary_tools(self):
        """Planner dynamically generates commands for both registered tools and arbitrary tools."""
        planner = AcquisitionPlanner()

        # 1. Known tool from registry
        plan_nmap = planner.plan_for_tool_name("nmap")
        self.assertTrue(plan_nmap.feasible)
        self.assertIn("nmap", plan_nmap.command)

        # 2. Arbitrary python package
        plan_pip = planner.plan_for_tool_name("pip:beautifulsoup4")
        self.assertTrue(plan_pip.feasible)
        self.assertIn("install", plan_pip.command)
        self.assertIn("beautifulsoup4", plan_pip.command)

        # 3. Arbitrary npm package
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}" if x == "npm" else None):
            plan_npm = planner.plan_for_tool_name("npm:semgrep")
            self.assertTrue(plan_npm.feasible)
            self.assertIn("npm install -g semgrep", plan_npm.command)

        # 4. Arbitrary tool name with no prefix on host with apt-get
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}" if x == "apt-get" else None), \
             patch("backend.execution.acquisition._IS_WINDOWS", False):
            plan_custom = planner.plan_for_tool_name("trufflehog")
            self.assertTrue(plan_custom.feasible)
            self.assertIn("apt-get install -y trufflehog", plan_custom.command)

    async def test_tool_install_auto_approved_and_registered(self):
        """When AUTO_APPROVE_PRIVILEGED=True, install executes, dynamically registers in ToolRegistry, and is usable."""
        mock_exec_res = MagicMock()
        mock_exec_res.succeeded = True
        mock_exec_res.status = "SUCCESS"
        mock_exec_res.stdout = "Successfully installed mocktool"
        mock_exec_res.stderr = ""
        mock_exec_res.exit_code = 0

        with patch.object(settings, "AUTO_APPROVE_PRIVILEGED", True), \
             patch("backend.execution.service.execution_service.run_command", new_callable=AsyncMock, return_value=mock_exec_res), \
             patch("backend.tools.manager.find_tool_binary", return_value="/usr/local/bin/mocktool"):

            res = await self.tool_manager.request_tool_install("mocktool")
            self.assertEqual(res.status, "SUCCESS")
            self.assertEqual(res.tool_name, "mocktool")
            self.assertIn("Successfully installed", res.stdout)

            # Check that mocktool is now in ToolRegistry
            registered = tool_registry.get_tool("mocktool")
            self.assertIsNotNone(registered)
            self.assertEqual(registered.binary, "/usr/local/bin/mocktool")
            self.assertIn("mocktool", registered.capabilities)

            # Check that get_tools_for_capability finds it immediately
            candidates = tool_registry.get_tools_for_capability("mocktool")
            self.assertTrue(any(c.tool_name == "mocktool" for c in candidates))

    async def test_tool_install_denied_in_manual_mode(self):
        """When in manual mode and unapproved, install returns CAPABILITY_GAP failure."""
        with patch.object(settings, "AUTO_APPROVE_PRIVILEGED", False), \
             patch.object(settings, "FORGE_APPROVAL_MODE", "manual"), \
             patch("backend.privilege.gate.require_approval", new_callable=AsyncMock, return_value=(False, "deny", None)):

            res = await self.tool_manager.request_tool_install("sqlmap")
            self.assertEqual(res.status, "FAILED")
            self.assertEqual(res.failure_category, "CAPABILITY_GAP")
            self.assertIn("PRIVILEGE DENIED", res.stderr)

    async def test_execute_capability_triggers_install_capability(self):
        """Calling execute_capability with 'install_tool' invokes request_tool_install."""
        with patch.object(self.tool_manager, "request_tool_install", new_callable=AsyncMock) as mock_install:
            mock_install.return_value = MagicMock(status="SUCCESS")
            await self.tool_manager.execute_capability(
                capability="install_tool",
                target="ncat"
            )
            mock_install.assert_called_once()

    async def test_warning_when_binary_requires_restart(self):
        """When install succeeds with 0 but binary is not visible in PATH, log clear warning."""
        mock_exec_res = MagicMock()
        mock_exec_res.succeeded = True
        mock_exec_res.status = "SUCCESS"
        mock_exec_res.stdout = "Installed tool_needing_restart"
        mock_exec_res.stderr = ""
        mock_exec_res.exit_code = 0

        with patch.object(settings, "AUTO_APPROVE_PRIVILEGED", True), \
             patch("backend.execution.service.execution_service.run_command", new_callable=AsyncMock, return_value=mock_exec_res), \
             patch("backend.tools.manager.find_tool_binary", return_value=None):

            res = await self.tool_manager.request_tool_install("tool_needing_restart")
            self.assertEqual(res.status, "SUCCESS")
            self.assertIn("restart may be required", res.stdout)
            self.assertIn("restart may be required", res.stderr)

if __name__ == "__main__":
    unittest.main()
