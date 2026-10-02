"""
Execution layer base types.

ExecutionRequest is the structured input from the intelligence layer.
ExecutionResult is the structured output from a backend.

Bridge methods (to_tool_execution_result / to_exec_result) convert to the
existing ToolExecutionResult and ExecResult models so all callers keep working
without change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from backend.agent_runtime.action import ExecResult
    from backend.tools.manager import ToolExecutionResult

# ── Status constants ──────────────────────────────────────────────────────── #
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_MISSING_TOOL = "MISSING_TOOL"
STATUS_KILLED = "KILLED"
STATUS_CANCELLED = "CANCELLED"
# Phase 4.x — capability/target-level outcomes surfaced by the execution layer.
STATUS_BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
STATUS_TARGET_MISMATCH = "TARGET_MISMATCH"


@dataclass
class ExecutionRequest:
    """
    Structured description of what the intelligence layer wants to execute.

    The intelligence layer fills in *what* it wants (capability, command); the
    execution layer decides *how* it runs (backend, OS, binary path, shell).
    """
    command: str
    capability: str = ""
    tool_name: str = ""
    cwd: Optional[str] = None
    timeout_seconds: int = 120
    env: Optional[Dict[str, str]] = None
    session_id: str = ""
    agent_id: str = ""
    action_id: str = ""
    privilege_level: str = "SAFE"
    canonical_target: Optional[str] = None
    # Tier-1 interactive execution (Phase 4.x §4): predetermined input written once to
    # the process's stdin after spawn (e.g. ``printf 'RETURN 0\n' | python challenge.py``
    # expressed structurally). None keeps the ordinary one-shot behaviour unchanged.
    stdin: Optional[str] = None


@dataclass
class ExecutionResult:
    """
    Full execution result returned by a backend.

    Extends ToolExecutionResult with backend provenance, workspace, and
    artifact tracking. Bridge methods convert to the existing result models
    so callers that already use ToolExecutionResult / ExecResult need no changes.
    """
    status: str = STATUS_SUCCESS
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    duration_ms: float = 0.0
    command: str = ""
    tool_name: str = ""
    capability: str = ""
    backend: str = "local"
    cwd: str = ""
    session_id: str = ""
    agent_id: str = ""
    execution_failure: bool = False
    failure_category: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status == STATUS_SUCCESS and not self.execution_failure

    def to_tool_execution_result(self) -> "ToolExecutionResult":
        """Bridge to the existing ToolExecutionResult used by ToolManager callers."""
        from backend.tools.manager import ToolExecutionResult  # lazy — avoids circular import
        return ToolExecutionResult(
            tool_name=self.tool_name or "unknown",
            capability=self.capability or "custom_command",
            command=self.command,
            status=self.status,
            stdout=self.stdout,
            stderr=self.stderr,
            exit_code=self.exit_code,
            duration_ms=self.duration_ms,
            execution_failure=self.execution_failure,
            failure_category=self.failure_category,
        )

    def to_exec_result(self) -> "ExecResult":
        """Bridge to the ExecResult used by AgentRuntime callers."""
        from backend.agent_runtime.action import ExecResult  # lazy — avoids circular import
        return ExecResult(
            tool_name=self.tool_name or "raw_cmd",
            capability=self.capability or "custom_command",
            command=self.command,
            status=self.status,
            stdout=self.stdout,
            stderr=self.stderr,
            exit_code=self.exit_code,
            duration_ms=self.duration_ms,
            execution_failure=self.execution_failure,
            failure_category=self.failure_category,
        )
