"""
LocalBackend — platform-aware subprocess execution backend.

Detects whether FORGE is running on Windows or Linux/macOS and routes
execution accordingly.  Callers never inspect sys.platform — they call
LocalBackend.execute() and receive a structured ExecutionResult.

Binary resolution:
- Always uses shutil.which() to locate a tool; never assumes a hardcoded path.
- On Windows, also checks for .exe / .cmd / .bat variants automatically via
  pathext handling in shutil.which().
- If the binary is not found, returns ExecutionResult(status=MISSING_TOOL)
  immediately — no silent fallback to a different tool.

python / python3:
- On Windows  shutil.which('python3') may be absent; falls back to 'python'.
- On Linux    shutil.which('python') may point to python2; prefers 'python3'.
  The backend resolves this at execute() time for PYTHON_EXEC capability.
"""
from __future__ import annotations

import logging
import shutil
import sys
from typing import Optional

from backend.execution.base import (
    ExecutionRequest,
    ExecutionResult,
    STATUS_FAILED,
    STATUS_MISSING_TOOL,
    STATUS_SUCCESS,
    STATUS_TIMEOUT,
)
from backend.execution.process_manager import process_manager

# classify_tool_execution is imported lazily inside execute() to break the
# import cycle: local.py → manager.py → execution_service → backends/local.py

logger = logging.getLogger("forge.execution.local")

_IS_WINDOWS = sys.platform == "win32"


def _resolve_python() -> Optional[str]:
    """Return the best python binary for the current OS."""
    if _IS_WINDOWS:
        # Windows: 'python' is usually the py-launcher alias; 'python3' may not exist.
        for candidate in ("python", "python3", "py"):
            p = shutil.which(candidate)
            if p:
                return candidate
    else:
        # Linux / macOS: prefer python3 to avoid Python 2.
        for candidate in ("python3", "python"):
            p = shutil.which(candidate)
            if p:
                return candidate
    return None


class LocalBackend:
    """
    Executes commands on the local host (Windows or Linux/macOS).

    This is the only backend FORGE uses today.  Future backends
    (LinuxVMBackend, DockerBackend, SSHBackend) will implement the same
    interface so AgentRuntime never needs to change.
    """

    kind: str = "local"

    # ------------------------------------------------------------------ #

    def capabilities(self) -> dict:
        """Thin wrapper — the authoritative CapabilityReport is in execution_backend.py."""
        from backend.agent_runtime.execution_backend import LocalExecutionBackend
        return LocalExecutionBackend().capabilities().to_dict()

    async def execute(self, req: ExecutionRequest) -> ExecutionResult:
        """
        Execute *req* on the local host and return a structured ExecutionResult.

        The command string is taken as-is from req.command.  The backend
        resolves the python binary if the command starts with 'python3' or
        'python' and normalises it for the current OS.
        """
        command = self._normalise_command(req.command)
        first = command.split()[0] if command.strip() else ""

        # Do not pre-check shutil.which here — shell built-ins (echo, cd, dir on
        # Windows) are valid commands that have no filesystem binary.  If a tool
        # genuinely does not exist, the subprocess exit code plus stderr text will
        # trigger COMMAND_NOT_FOUND via classify_tool_execution, and we then map
        # that to STATUS_MISSING_TOOL below.
        logger.info(f"[LocalBackend] execute cwd={req.cwd!r}: {command[:200]}")

        stdout, stderr, exit_code = await process_manager.run(
            command,
            cwd=req.cwd,
            timeout_seconds=req.timeout_seconds,
            env=req.env,
            session_id=req.session_id,
            agent_id=req.agent_id,
            backend=self.kind,
            input_data=req.stdin,
        )

        timed_out = (
            exit_code == -1
            and f"timed out after {req.timeout_seconds}" in stderr
        )
        if timed_out:
            status = STATUS_TIMEOUT
        elif exit_code == 0:
            status = STATUS_SUCCESS
        else:
            status = STATUS_FAILED

        from backend.tools.manager import classify_tool_execution  # lazy — breaks import cycle
        classification = classify_tool_execution(
            first, exit_code, stdout, stderr
        )

        # Promote status to MISSING_TOOL when the classifier confirms the binary
        # was not found — this keeps the structured contract even for shell
        # built-ins (echo, cd) that have no filesystem binary and pass through
        # the subprocess unchanged.
        if classification["failure_category"] == "COMMAND_NOT_FOUND":
            status = STATUS_MISSING_TOOL

        return ExecutionResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            command=command,
            tool_name=req.tool_name or first,
            capability=req.capability,
            backend=self.kind,
            cwd=req.cwd or "",
            session_id=req.session_id,
            agent_id=req.agent_id,
            execution_failure=classification["execution_failure"],
            failure_category=classification["failure_category"],
        )

    # ------------------------------------------------------------------ #

    def _normalise_command(self, command: str) -> str:
        """
        Rewrite platform-incompatible python invocations.

        Replaces 'python3 ...' with the correct local python binary so FORGE
        runs on Windows without manual PATH adjustments.
        """
        stripped = command.strip()
        if not stripped:
            return stripped

        parts = stripped.split(None, 1)
        first = parts[0]

        if first in ("python3", "python"):
            resolved = _resolve_python()
            if resolved and resolved != first:
                rest = parts[1] if len(parts) > 1 else ""
                return f"{resolved} {rest}".strip()

        return stripped
