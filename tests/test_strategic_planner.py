import unittest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock

# Ensure backend can be imported and isolated test database is used per Rule 5
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.strategic_planner import StrategicPlanner, strategic_planner

class TestStrategicPlanner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.planner = StrategicPlanner()

    def test_fallback_plan_generation(self):
        plan_tasks = self.planner._generate_fallback_plan(
            category="Web",
            target="http://ctf.example.com/login",
            challenge_name="Web SQL Injection CTF"
        )
        self.assertIsInstance(plan_tasks, list)
        self.assertGreater(len(plan_tasks), 3)
        self.assertEqual(plan_tasks[0]["status"], "IN_PROGRESS")
        self.assertEqual(plan_tasks[0]["phase"], "RECON")

    @patch("backend.agents.strategic_planner.model_router.route_request")
    async def test_generate_initial_plan(self, mock_route):
        mock_response = MagicMock()
        mock_response.content = '{"summary": "Pwn strategy", "tasks": [{"id": "task-1", "phase": "RECON", "title": "Checksec", "tool": "checksec", "reasoning": "Audit protections"}]}'
        mock_response.model = "gpt-4o"
        mock_route.return_value = mock_response

        plan = await self.planner.generate_initial_plan(
            challenge_id="test-chal-1",
            challenge_name="Pwn Buffer Overflow",
            category="Pwn",
            difficulty="Medium",
            target="/tmp/vuln"
        )
        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["challenge_id"], "test-chal-1")
        self.assertIn("tasks", plan)
        self.assertGreater(len(plan["tasks"]), 0)

    def test_progress_calculation(self):
        plan = {
            "tasks": [
                {"id": "t1", "status": "COMPLETED"},
                {"id": "t2", "status": "IN_PROGRESS"},
                {"id": "t3", "status": "PENDING"},
                {"id": "t4", "status": "PENDING"}
            ]
        }
        pct = self.planner.calculate_progress(plan)
        self.assertGreaterEqual(pct, 10)
        self.assertLessEqual(pct, 95)
        
        # Test flag captured override
        full_pct = self.planner.calculate_progress(plan, flag_captured=True)
        self.assertEqual(full_pct, 100)

    def test_task_advancement(self):
        plan = {
            "tasks": [
                {"id": "t1", "phase": "RECON", "status": "IN_PROGRESS", "output_summary": ""},
                {"id": "t2", "phase": "SURFACE_ANALYSIS", "status": "PENDING", "output_summary": ""},
                {"id": "t3", "phase": "EXPLOITATION", "status": "PENDING", "output_summary": ""}
            ]
        }
        updated_plan = self.planner.update_task_progress(
            mission_plan=plan,
            turn=2,
            executed_command="nmap -sV -sC -p 80 target.ctf",
            output_snippet="Port 80 open Apache 2.4",
            is_success=True
        )
        self.assertEqual(updated_plan["tasks"][0]["status"], "COMPLETED")
        self.assertEqual(updated_plan["tasks"][1]["status"], "IN_PROGRESS")

    @patch("backend.agents.strategic_planner.model_router.route_request")
    async def test_review_and_adapt_plan(self, mock_route):
        mock_response = MagicMock()
        mock_response.content = '{"diagnosis": "Cloudflare WAF Blocked", "pivot_strategy": "Use tamper script", "new_tasks": [{"id": "pivot-1", "phase": "EXPLOITATION", "title": "Tamper Bypass", "tool": "python_script", "reasoning": "Header manipulation", "status": "IN_PROGRESS"}]}'
        mock_response.model = "deepseek-r1"
        mock_route.return_value = mock_response

        plan = {
            "tasks": [
                {"id": "t1", "phase": "RECON", "status": "COMPLETED"},
                {"id": "t2", "phase": "SURFACE_ANALYSIS", "status": "IN_PROGRESS"},
                {"id": "t3", "phase": "EXPLOITATION", "status": "PENDING"}
            ],
            "strategic_reviews": []
        }
        adapted_plan = await self.planner.review_and_adapt_plan(
            challenge_name="WAF Protected Web App",
            category="Web",
            target="http://waf.example.com",
            mission_plan=plan,
            stuck_reason="Repetitive command execution on blocked endpoint",
            recent_history=["curl -s http://waf.example.com/api", "curl -s http://waf.example.com/api"],
            primary_model="Gemini 2.5 Pro"
        )
        self.assertIn("strategic_reviews", adapted_plan)
        self.assertGreater(len(adapted_plan["strategic_reviews"]), 0)
        review = adapted_plan["strategic_reviews"][0]
        self.assertIn("diagnosis", review)
        self.assertIn("pivot_strategy", review)

if __name__ == "__main__":
    unittest.main()
