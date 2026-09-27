"""
Tests for persistent interactive execution exposed to the autonomous agent and ToolManager.

Verifies:
1. Deterministic local fake interactive child process (multi-turn dialogue).
2. Generic interactive session opening reaches ExecutionService.open_interactive() and returns session key and initial prompt.
3. ToolManager.execute_capability and execute_raw_command route interactive commands correctly.
4. Sending input to the SAME session and reading responses.
5. send_and_read capability and command execution.
6. Closing sessions, process cleanup, and unregistration from ProcessManager.
7. Integration with Action, RealToolExecutor, and AgentRuntime action types.
8. One-shot command execution continues working unchanged.
9. Privilege and safety constraints preserved (no hardcoded CTF host/port/exploit).
"""

import asyncio
import os
import sys
import tempfile
import unittest

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.tools.manager import tool_manager, ToolExecutionResult
from backend.execution.service import execution_service
from backend.execution.interactive import interactive_manager
from backend.execution.process_manager import process_manager
from backend.agent_runtime.action import Action, ActionType, ExecResult
from backend.agent_runtime.runtime import RealToolExecutor
from backend.agent_runtime.decision import DecisionEngine

PY = sys.executable or "python"

def _run(coro):
    return asyncio.run(coro)

# ── Local deterministic interactive child fixture ─────────────────────────── #
FAKE_DIALOGUE_CHILD = """\
import sys

sys.stdout.write("WELCOME_CHALLENGE_BANNER\\nPlease identify: ")
sys.stdout.flush()

line1 = sys.stdin.readline().strip()
if line1 == "CHALLENGER_ONE":
    sys.stdout.write("STAGE_ONE_CLEARED\\nEnter secret code: ")
    sys.stdout.flush()
    line2 = sys.stdin.readline().strip()
    if line2 == "CODE_42":
        sys.stdout.write("STAGE_TWO_CLEARED\\nResult: FLAG{interactive_dialogue_success}\\n")
        sys.stdout.flush()
    else:
        sys.stdout.write("FAILED_STAGE_TWO\\n")
        sys.stdout.flush()
else:
    sys.stdout.write("REJECTED_STAGE_ONE\\n")
    sys.stdout.flush()
"""

class TestInteractiveAgentIntegration(unittest.TestCase):

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.script_path = os.path.join(self._temp_dir.name, "fake_server.py")
        with open(self.script_path, "w", encoding="utf-8") as f:
            f.write(FAKE_DIALOGUE_CHILD)

    def tearDown(self):
        _run(interactive_manager.close_all(reason="test_teardown"))
        self._temp_dir.cleanup()

    # ---------------------------------------------------------------------- #
    # 1. ToolManager.execute_raw_command with interactive primitives
    # ---------------------------------------------------------------------- #

    def test_raw_command_full_interactive_lifecycle(self):
        """Verify open -> read -> send -> read -> send_and_read -> close via raw command syntax."""
        async def scenario():
            cmd_start = f'interactive_open "{PY}" "{self.script_path}"'
            open_res: ToolExecutionResult = await tool_manager.execute_raw_command(cmd_start)

            self.assertEqual(open_res.status, "SUCCESS")
            self.assertEqual(open_res.tool_name, "interactive_open")
            self.assertEqual(open_res.capability, "interactive_open")
            self.assertIn("[SESSION:", open_res.stdout)
            self.assertIn("WELCOME_CHALLENGE_BANNER", open_res.stdout)
            self.assertIn("Please identify:", open_res.stdout)

            # Extract session key from output
            sess_key = open_res.stdout.split("[SESSION:")[1].split("]")[0].strip()
            self.assertTrue(bool(sess_key))

            # Verify session is tracked in interactive_manager and process_manager
            sess = interactive_manager.get(sess_key)
            self.assertIsNotNone(sess)
            self.assertTrue(sess.is_alive())
            self.assertIn(sess.pid, process_manager.active_pids())

            # Send first stage response
            send_res: ToolExecutionResult = await tool_manager.execute_raw_command(
                f"interactive_send {sess_key} CHALLENGER_ONE"
            )
            self.assertEqual(send_res.status, "SUCCESS")

            # Read intermediate output
            read_res: ToolExecutionResult = await tool_manager.execute_raw_command(
                f"interactive_read {sess_key}"
            )
            self.assertEqual(read_res.status, "SUCCESS")
            self.assertIn("STAGE_ONE_CLEARED", read_res.stdout)
            self.assertIn("Enter secret code:", read_res.stdout)

            # Send second stage response and read result in one step
            snr_res: ToolExecutionResult = await tool_manager.execute_raw_command(
                f"interactive_send_and_read {sess_key} CODE_42"
            )
            self.assertEqual(snr_res.status, "SUCCESS")
            self.assertIn("STAGE_TWO_CLEARED", snr_res.stdout)
            self.assertIn("FLAG{interactive_dialogue_success}", snr_res.stdout)

            # Close the session
            close_res: ToolExecutionResult = await tool_manager.execute_raw_command(
                f"interactive_close {sess_key}"
            )
            self.assertEqual(close_res.status, "SUCCESS")
            self.assertIn("Closed", close_res.stdout)
            self.assertIsNone(interactive_manager.get(sess_key))
            self.assertNotIn(sess.pid, process_manager.active_pids())

        _run(scenario())

    # ---------------------------------------------------------------------- #
    # 2. ToolManager.execute_capability structured routing
    # ---------------------------------------------------------------------- #

    def test_execute_capability_interactive_methods(self):
        """Verify execute_capability direct calls for interactive primitives."""
        async def scenario():
            cmd = f'"{PY}" "{self.script_path}"'
            open_res = await tool_manager.execute_capability("interactive_open", target=cmd)
            self.assertEqual(open_res.status, "SUCCESS")
            self.assertIn("WELCOME_CHALLENGE_BANNER", open_res.stdout)

            sess_key = open_res.stdout.split("[SESSION:")[1].split("]")[0].strip()

            # send_and_read directly via capability
            snr_res = await tool_manager.execute_capability(
                "interactive_send_and_read", target=sess_key, extra_args="CHALLENGER_ONE"
            )
            self.assertEqual(snr_res.status, "SUCCESS")
            self.assertIn("STAGE_ONE_CLEARED", snr_res.stdout)

            # close via capability
            close_res = await tool_manager.execute_capability("interactive_close", target=sess_key)
            self.assertEqual(close_res.status, "SUCCESS")

        _run(scenario())

    # ---------------------------------------------------------------------- #
    # 3. ToolManager.execute_tool with dictionary params
    # ---------------------------------------------------------------------- #

    def test_execute_tool_structured_params(self):
        """Verify execute_tool universal dispatcher routes interactive tool calls."""
        async def scenario():
            cmd = f'"{PY}" "{self.script_path}"'
            open_res = await tool_manager.execute_tool(
                "interactive_open", {"command": cmd}
            )
            self.assertEqual(open_res.status, "SUCCESS")
            sess_key = open_res.stdout.split("[SESSION:")[1].split("]")[0].strip()

            # Using tool_name="interactive" with action="send_and_read"
            snr_res = await tool_manager.execute_tool(
                "interactive",
                {"action": "send_and_read", "session_key": sess_key, "data": "CHALLENGER_ONE"}
            )
            self.assertEqual(snr_res.status, "SUCCESS")
            self.assertIn("STAGE_ONE_CLEARED", snr_res.stdout)

            # Close via execute_tool
            close_res = await tool_manager.execute_tool(
                "interactive_close", {"session_key": sess_key}
            )
            self.assertEqual(close_res.status, "SUCCESS")

        _run(scenario())

    # ---------------------------------------------------------------------- #
    # 4. RealToolExecutor and AgentRuntime Action integration
    # ---------------------------------------------------------------------- #

    def test_real_tool_executor_with_agent_action(self):
        """Verify RealToolExecutor executes Action(type=COMMAND) and Action(type=TOOL_CALL)."""
        async def scenario():
            executor = RealToolExecutor(tool_manager=tool_manager)

            # ActionType.COMMAND
            act_open = Action(
                type=ActionType.COMMAND,
                command=f'interactive_open "{PY}" "{self.script_path}"'
            )
            res_open: ExecResult = await executor.execute(act_open)
            self.assertEqual(res_open.status, "SUCCESS")
            self.assertTrue(res_open.succeeded)
            self.assertIn("WELCOME_CHALLENGE_BANNER", res_open.stdout)

            sess_key = res_open.stdout.split("[SESSION:")[1].split("]")[0].strip()

            # ActionType.TOOL_CALL
            act_snr = Action(
                type=ActionType.TOOL_CALL,
                capability="interactive_send_and_read",
                tool_args={"target": sess_key, "extra_args": "CHALLENGER_ONE"}
            )
            res_snr: ExecResult = await executor.execute(act_snr)
            self.assertEqual(res_snr.status, "SUCCESS")
            self.assertIn("STAGE_ONE_CLEARED", res_snr.stdout)

            # Close via COMMAND
            act_close = Action(
                type=ActionType.COMMAND,
                command=f"interactive_close {sess_key}"
            )
            res_close: ExecResult = await executor.execute(act_close)
            self.assertEqual(res_close.status, "SUCCESS")

        _run(scenario())

    # ---------------------------------------------------------------------- #
    # 5. One-shot command preservation
    # ---------------------------------------------------------------------- #

    def test_one_shot_commands_unaffected(self):
        """Verify standard one-shot commands still execute normally through ExecutionService."""
        async def scenario():
            res = await tool_manager.execute_raw_command(f'{PY} -c "print(12345 * 2)"')
            self.assertEqual(res.status, "SUCCESS")
            self.assertIn("24690", res.stdout.strip())
            self.assertEqual(res.capability, "custom_command")
        _run(scenario())

    # ---------------------------------------------------------------------- #
    # 6. Autonomous Interactive-Pivot Multi-Turn Dialogue Integration
    # ---------------------------------------------------------------------- #

    def test_autonomous_interactive_dialogue_pivot_e2e(self):
        """Verify autonomous decision/action layer selects persistent interaction,
        retains the session ID across turns, completes sequential dialogue, and cleans up.
        """
        async def scenario():
            from backend.agent_runtime import AgentRuntime, session_manager
            from backend.agent_runtime.decision import ProviderCompletion

            # Track session key across turns in scripted model gateway
            extracted_session_key = []

            class ScriptedDialogueProvider:
                def __init__(self, script_path):
                    self.script_path = script_path
                    self.turn = 0

                async def complete(self, *, prompt, system_instruction="", capability="general_reasoning",
                                   urgency="normal", reasoning_depth="fast"):
                    # Turn 0: Model observes interactive target requirement and chooses interactive_open
                    if self.turn == 0:
                        self.turn += 1
                        return ProviderCompletion(
                            content=f'interactive_open "{PY}" "{self.script_path}"',
                            provider_name="test_scripted_gw",
                            model_name="test_dialogue_model"
                        )

                    # Turn 1: Model sees initial prompt, extracts session key, sends Stage 1 input
                    if self.turn == 1:
                        self.turn += 1
                        # Extract session key from prompt / state block.
                        # ContextBuilder renders "Active interactive sessions: <key>" in
                        # the state block; also check raw "[SESSION: <key>]" from stdout
                        # in case the trajectory includes it.
                        import re as _re
                        _sess_m = _re.search(r'\[SESSION:\s*([A-Za-z0-9_\-]+)\]', prompt)
                        if not _sess_m:
                            _sess_m = _re.search(r'interactive sessions?:\s*([A-Za-z0-9_\-]+)', prompt, _re.IGNORECASE)
                        if _sess_m:
                            extracted_session_key.append(_sess_m.group(1))
                        key = extracted_session_key[-1] if extracted_session_key else "sess-fail"
                        return ProviderCompletion(
                            content=f"interactive_send {key} CHALLENGER_ONE",
                            provider_name="test_scripted_gw",
                            model_name="test_dialogue_model"
                        )

                    # Turn 2: Read response after Stage 1
                    if self.turn == 2:
                        self.turn += 1
                        key = extracted_session_key[-1] if extracted_session_key else "sess-fail"
                        return ProviderCompletion(
                            content=f"interactive_read {key}",
                            provider_name="test_scripted_gw",
                            model_name="test_dialogue_model"
                        )

                    # Turn 3: Send Stage 2 secret code to the SAME session
                    if self.turn == 3:
                        self.turn += 1
                        key = extracted_session_key[-1]
                        return ProviderCompletion(
                            content=f"interactive_send {key} CODE_42",
                            provider_name="test_scripted_gw",
                            model_name="test_dialogue_model"
                        )

                    # Turn 4: Read final flag from session
                    if self.turn == 4:
                        self.turn += 1
                        key = extracted_session_key[-1]
                        return ProviderCompletion(
                            content=f"interactive_read {key}",
                            provider_name="test_scripted_gw",
                            model_name="test_dialogue_model"
                        )

                    # Turn 5: Close interactive session cleanly
                    if self.turn == 5:
                        self.turn += 1
                        key = extracted_session_key[-1]
                        return ProviderCompletion(
                            content=f"interactive_close {key}",
                            provider_name="test_scripted_gw",
                            model_name="test_dialogue_model"
                        )

                    return ProviderCompletion(
                        content="Done.",
                        provider_name="test_scripted_gw",
                        model_name="test_dialogue_model"
                    )

            provider = ScriptedDialogueProvider(self.script_path)
            tool_executor = RealToolExecutor(tool_manager=tool_manager)
            runtime = AgentRuntime(
                tool_executor=tool_executor,
                provider_gateway=provider,
                learn_on_completion=False,
            )

            sess = session_manager.create(
                challenge_id="interactive_dialogue_chal",
                objective="Interact with dialogue challenge service and extract flag",
                target_scope="local_interactive",
                agent_id="interactive_specialist",
                engine="runtime"
            )

            result = await runtime.run(session=sess, max_turns=6)

            # 1. Verify multi-turn execution completed
            self.assertIn(result.status, ("COMPLETED", "MAX_TURNS"))
            self.assertGreaterEqual(len(extracted_session_key), 1)
            sess_key = extracted_session_key[0]

            # 2. Verify session was retained across turns and final flag observed
            # RunResult carries flag_candidates directly (no .session attribute).
            self.assertIn("FLAG{interactive_dialogue_success}", result.flag_candidates)

            # 3. Verify session was cleanly closed and no orphan process remains
            self.assertIsNone(interactive_manager.get(sess_key))

        _run(scenario())

    # ---------------------------------------------------------------------- #
    # 7. Decision parser recognizes interactive commands
    # ---------------------------------------------------------------------- #

    def test_decision_parser_recognizes_interactive_commands(self):
        """Verify DecisionEngine parses interactive commands without skipping them as prose."""
        parser = DecisionEngine(gateway=None)
        raw_model_output = (
            "I need to interact with the challenge dialogue process.\n"
            "interactive_open nc 127.0.0.1 1337"
        )
        decision, action, malformed = parser.parse(raw_model_output)
        self.assertFalse(malformed)
        self.assertEqual(action.type, ActionType.COMMAND)
        self.assertEqual(action.command, "interactive_open nc 127.0.0.1 1337")

        # Test interactive_send_and_read
        raw_snr = (
            "Sending response to session.\n"
            "interactive_send_and_read abc-123 my_response"
        )
        decision, action, malformed = parser.parse(raw_snr)
        self.assertFalse(malformed)
        self.assertEqual(action.type, ActionType.COMMAND)
        self.assertEqual(action.command, "interactive_send_and_read abc-123 my_response")

    # ---------------------------------------------------------------------- #
    # 8. Invalid/stale session handling
    # ---------------------------------------------------------------------- #

    def test_invalid_session_returns_clean_failure(self):
        """Verify operating on a non-existent or closed session returns a structured failure without crashing."""
        async def scenario():
            res = await tool_manager.execute_raw_command("interactive_send non_existent_key test")
            self.assertEqual(res.status, "FAILED")
            self.assertTrue(res.execution_failure)
            self.assertEqual(res.failure_category, "SESSION_NOT_FOUND")

            res_read = await tool_manager.execute_raw_command("interactive_read non_existent_key")
            self.assertEqual(res_read.status, "FAILED")
            self.assertTrue(res_read.execution_failure)

            res_close = await tool_manager.execute_raw_command("interactive_close non_existent_key")
            self.assertEqual(res_close.status, "FAILED")
        _run(scenario())
