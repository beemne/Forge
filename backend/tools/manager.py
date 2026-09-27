import os
import re
import sys
import difflib
from urllib.parse import urlparse
import asyncio
import time
import shlex
import shutil
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from backend.tools.registry import tool_registry, ToolMetadata
from backend.environment.detector import environment_detector
from backend.execution.service import execution_service

logger = logging.getLogger("forge.tools")


def refresh_environment_path() -> None:
    """Refreshes standard binary directories in os.environ['PATH'] so newly installed tools are immediately found."""
    current_path = os.environ.get("PATH", "")
    paths = current_path.split(os.pathsep)
    extra_dirs = []

    # Python scripts directory
    py_dir = os.path.dirname(sys.executable)
    py_scripts = os.path.join(py_dir, "Scripts") if sys.platform == "win32" else os.path.join(py_dir, "bin")
    if os.path.exists(py_scripts) and py_scripts not in paths:
        extra_dirs.append(py_scripts)

    # User local bin / cargo / go / npm dirs
    home = os.path.expanduser("~")
    user_candidates = [
        os.path.join(home, ".local", "bin"),
        os.path.join(home, ".cargo", "bin"),
        os.path.join(home, "go", "bin"),
        os.path.join(home, "AppData", "Roaming", "npm"),
        os.path.join(home, "AppData", "Local", "Programs", "Python", f"Python{sys.version_info.major}{sys.version_info.minor}", "Scripts"),
        "C:\\ProgramData\\chocolatey\\bin",
    ]
    for cand in user_candidates:
        if os.path.exists(cand) and cand not in paths:
            extra_dirs.append(cand)

    if extra_dirs:
        os.environ["PATH"] = os.pathsep.join(extra_dirs + paths)


def find_tool_binary(binary_name: str) -> Optional[str]:
    """Finds a tool binary on the host PATH, including platform-specific extensions."""
    refresh_environment_path()
    resolved = shutil.which(binary_name)
    if not resolved and sys.platform == "win32":
        for ext in [".exe", ".cmd", ".bat", ".ps1"]:
            cand = shutil.which(f"{binary_name}{ext}")
            if cand:
                return cand
    return resolved


class ToolExecutionResult(BaseModel):
    tool_name: str
    capability: str
    command: str
    status: str # SUCCESS, FAILED, TIMEOUT, MISSING_TOOL
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    duration_ms: float = 0.0
    execution_failure: bool = False       # True if DNS failure, connection refused, or timeout
    failure_category: Optional[str] = None # DNS_ERROR, TIMEOUT, CONNECTION_REFUSED, COMMAND_NOT_FOUND


def format_tool_args(template: str, **values: str) -> str:
    """
    Interpolate externally-sourced values into a registered tool's args template.

    The values substituted here come from outside this module's control: the challenge
    target (an IP, hostname or URL) and the runtime-resolved wordlist path. The resulting
    argument string is handed to a shell (see ExecutionService.run_command ->
    create_subprocess_shell), so an unquoted `;`, `|`, backtick or `$(...)` inside a
    target would execute as a separate command instead of being scanned as a hostname.
    Every keyword value is therefore passed through shlex.quote() so that it reaches the
    shell as a single literal argument.

    Only the `.format()` keyword values are quoted. Literals already written into the
    template itself (`-sV -F`, `-i -s`, `-n 8`, `-e`) must stay unquoted so the tool's
    flags remain flags.
    """
    return template.format(**{key: shlex.quote(str(value)) for key, value in values.items()})


def sanitize_and_correct_command_target(command: str, canonical_target_url: Optional[str] = None) -> str:
    """
    Detects and auto-corrects typo'd hostnames in generated CLI commands
    (e.g., 'amiable-citidel' -> 'amiable-citadel') before subprocess execution.
    """
    if not command or not canonical_target_url:
        return command

    try:
        parsed_canonical = urlparse(canonical_target_url if "://" in canonical_target_url else f"http://{canonical_target_url}")
        canonical_host = (parsed_canonical.hostname or canonical_target_url.split(":")[0]).replace("http://", "").replace("https://", "").strip("/")
        if not canonical_host or len(canonical_host) < 4:
            return command

        corrected_cmd = command
        # Extract potential host tokens in command
        tokens = re.findall(r"https?://([a-zA-Z0-9_\-\.]+)|([a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,})", command)
        flat_tokens = [t[0] or t[1] for t in tokens if (t[0] or t[1])]

        for token in flat_tokens:
            token_clean = token.split(":")[0].strip()
            if token_clean.lower() != canonical_host.lower() and len(token_clean) >= 4:
                ratio = difflib.SequenceMatcher(None, token_clean.lower(), canonical_host.lower()).ratio()
                if 0.75 <= ratio < 1.0:
                    logger.warning(f"[AUTO-CORRECT HOST] Correcting typo'd host '{token_clean}' -> '{canonical_host}' in command: {command[:80]}")
                    corrected_cmd = corrected_cmd.replace(token_clean, canonical_host)
        return corrected_cmd
    except Exception as e:
        logger.debug(f"Target correction skip: {e}")
        return command


def classify_tool_execution(tool_name: str, exit_code: Optional[int], stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Classifies tool exit codes and output into target responses vs execution-level failures.

    Two families of execution failure are distinguished:
    - NETWORK-level (the command left the host but never got a target response):
      curl exit 6 (DNS), exit 7 (connection refused), exit 28 (timeout).
    - LOCAL-level (the command never ran / never reached the target at all): a bad
      invocation such as an unquoted path with a space (`[Errno 2] No such file or
      directory`), a Python SyntaxError, a permission error, or a missing dependency.
      These must NOT be retried identically — the caller aborts after a couple of hits
      (see LOCAL_EXEC_CATEGORIES) instead of burning the iteration/time budget.
    """
    if exit_code == 0:
        return {"execution_failure": False, "failure_category": None}

    combined = ((stdout or "") + " " + (stderr or "")).lower()

    # ── LOCAL-level failures first: unambiguous text signatures win over exit-code
    #    heuristics (a bad invocation can share exit_code -1 with a subprocess error). ──
    if ("[errno 2]" in combined or "no such file or directory" in combined
            or "can't open file" in combined or "cannot open file" in combined):
        return {"execution_failure": True, "failure_category": "FILE_NOT_FOUND"}
    if ("syntaxerror" in combined or "invalid syntax" in combined
            or "unterminated string" in combined or "unexpected eof while parsing" in combined
            or "unexpected token" in combined):
        return {"execution_failure": True, "failure_category": "SYNTAX_ERROR"}
    if ("[errno 13]" in combined or "permission denied" in combined
            or "not permitted" in combined or "sudo: a password is required" in combined
            or "sudo: no tty present" in combined or "a password is required" in combined
            or "no tty present" in combined):
        return {"execution_failure": True, "failure_category": "PERMISSION_DENIED"}
    if ("modulenotfounderror" in combined or "no module named" in combined
            or "importerror" in combined):
        return {"execution_failure": True, "failure_category": "MISSING_DEP"}
    if ("no scheme supplied" in combined or "invalid url" in combined
            or "missing scheme" in combined or "unknown url type" in combined):
        return {"execution_failure": True, "failure_category": "INVALID_URL"}

    # ── NETWORK-level failures (left the host, no usable target response) ──────────────
    if exit_code in [6] or "could not resolve host" in combined or "name or service not known" in combined:
        return {"execution_failure": True, "failure_category": "DNS_ERROR"}
    if exit_code in [7] or "connection refused" in combined or "failed to connect" in combined:
        return {"execution_failure": True, "failure_category": "CONNECTION_REFUSED"}
    if exit_code in [28, -1] or "operation timed out" in combined or "timed out after" in combined:
        return {"execution_failure": True, "failure_category": "TIMEOUT"}
    if "command not found" in combined or "is not recognized as an internal or external command" in combined:
        return {"execution_failure": True, "failure_category": "COMMAND_NOT_FOUND"}

    return {"execution_failure": False, "failure_category": None}


# Failure categories that mean the command never reached the target (a local invocation
# problem, not a target response). The swarm aborts an agent after a couple of these in a
# row rather than re-running the same broken command against a shared, rate-limited chain.
# COMMAND_NOT_FOUND is included: a missing binary is a local problem, not a target signal.
LOCAL_EXEC_CATEGORIES = frozenset({
    "FILE_NOT_FOUND", "SYNTAX_ERROR", "PERMISSION_DENIED", "MISSING_DEP", "COMMAND_NOT_FOUND",
    "INVALID_URL", "INTERPRETER_ASSUMPTION",
})


class ToolManager:
    """Resolves agent capability requests to concrete installed tool executions."""

    async def request_tool_install(
        self,
        tool_or_capability: str,
        preferred_pm: Optional[str] = None,
        agent_id: str = "capability_manager",
        challenge_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> ToolExecutionResult:
        """Dynamically plans and requests privilege-approved installation of a tool on the host."""
        start_time = time.time()
        from backend.execution.acquisition import acquisition_planner
        plan = acquisition_planner.plan_for_tool_name(tool_or_capability, preferred_pm=preferred_pm)

        if not plan.feasible or not plan.command:
            elapsed_ms = (time.time() - start_time) * 1000
            return ToolExecutionResult(
                tool_name="package_installer",
                capability="install_tool",
                command="",
                status="FAILED",
                stderr=f"No feasible install recipe for '{tool_or_capability}': {plan.reason}",
                exit_code=1,
                duration_ms=elapsed_ms,
                execution_failure=True,
                failure_category="INSTALL_FAILED"
            )

        # Route through unified privilege approval gate
        from backend.privilege.gate import require_approval, SHARED_PENDING_APPROVALS
        from backend.websocket.manager import ws_manager

        approved, decision, sudo_password = await require_approval(
            cmd=plan.command,
            agent_id=agent_id,
            pending_approvals=SHARED_PENDING_APPROVALS,
            broadcast_fn=ws_manager.broadcast,
            challenge_id=challenge_id,
            run_id=run_id,
            # Without this the operator sees only a bare `apt-get install -y ffuf` and
            # cannot tell an acquire-on-demand install from any other privileged command.
            # The plan already knows the capability and the provider, so surface them.
            context={
                "request_kind": "tool_install",
                "capability": tool_or_capability,
                "provider": plan.provider,
                "method": plan.method.value,
                "install_command": plan.command,
                "reason": plan.reason,
            },
        )

        if not approved:
            elapsed_ms = (time.time() - start_time) * 1000
            return ToolExecutionResult(
                tool_name="package_installer",
                capability="install_tool",
                command=plan.command,
                status="FAILED",
                stderr=f"[PRIVILEGE DENIED] Installation of '{tool_or_capability}' via '{plan.command}' was not approved.",
                exit_code=-1,
                duration_ms=elapsed_ms,
                execution_failure=True,
                failure_category="CAPABILITY_GAP"
            )

        logger.info(f"Approved tool installation executing: {plan.command}")
        exec_res = await execution_service.run_command(
            plan.command,
            timeout_seconds=300,
            capability="install_tool",
            tool_name="installer"
        )
        elapsed_ms = (time.time() - start_time) * 1000

        if not exec_res.succeeded and exec_res.status != "SUCCESS":
            return ToolExecutionResult(
                tool_name="package_installer",
                capability="install_tool",
                command=plan.command,
                status="FAILED",
                stdout=exec_res.stdout,
                stderr=f"Installation failed: {exec_res.stderr}",
                exit_code=exec_res.exit_code or 1,
                duration_ms=elapsed_ms,
                execution_failure=True,
                failure_category="INSTALL_FAILED"
            )

        # Refresh PATH and capability state
        refresh_environment_path()
        try:
            from backend.execution.capabilities import capability_service
            capability_service.refresh(tool_or_capability)
        except Exception:
            pass

        resolved_bin = find_tool_binary(plan.provider)
        if resolved_bin:
            tool_registry.register_dynamic_tool(
                tool_name=plan.provider,
                binary=resolved_bin,
                capabilities=[tool_or_capability, plan.provider],
                installation_recipe=plan.command
            )
            logger.info(f"Successfully installed and registered tool '{plan.provider}' at {resolved_bin}")
            return ToolExecutionResult(
                tool_name=plan.provider,
                capability="install_tool",
                command=plan.command,
                status="SUCCESS",
                stdout=f"Successfully installed '{plan.provider}' at {resolved_bin}.\n{exec_res.stdout}",
                stderr="",
                exit_code=0,
                duration_ms=elapsed_ms,
                execution_failure=False,
                failure_category=None
            )
        else:
            warn_msg = f"Installation command succeeded, but binary '{plan.provider}' is not yet visible in host PATH. A process or terminal restart may be required to refresh system environment variables."
            logger.warning(warn_msg)
            tool_registry.register_dynamic_tool(
                tool_name=plan.provider,
                binary=plan.provider,
                capabilities=[tool_or_capability, plan.provider],
                installation_recipe=plan.command
            )
            return ToolExecutionResult(
                tool_name=plan.provider,
                capability="install_tool",
                command=plan.command,
                status="SUCCESS",
                stdout=f"{exec_res.stdout}\n[WARNING] {warn_msg}",
                stderr=warn_msg,
                exit_code=0,
                duration_ms=elapsed_ms,
                execution_failure=False,
                failure_category=None
            )

    async def execute_capability(
        self,
        capability: str,
        target: str,
        extra_args: Optional[str] = None,
        cwd: Optional[str] = None,
        agent_id: str = "agent",
        challenge_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> ToolExecutionResult:
        start_time = time.time()
        parsed_target = target.replace("+", ",").split(",")[0].strip()

        # Tool installation / acquisition capability
        if capability in ("install_tool", "acquire_tool"):
            tool_target = parsed_target or (extra_args or "").strip()
            return await self.request_tool_install(
                tool_or_capability=tool_target,
                agent_id=agent_id,
                challenge_id=challenge_id,
                run_id=run_id
            )

        # Direct model-powered capabilities like vision_read
        if capability == "vision_read":
            target_path = parsed_target
            if cwd and not os.path.isabs(target_path) and not os.path.exists(target_path):
                alt_path = os.path.join(cwd, target_path)
                if os.path.exists(alt_path):
                    target_path = alt_path

            if not target_path or not os.path.isfile(target_path):
                elapsed_ms = (time.time() - start_time) * 1000
                return ToolExecutionResult(
                    tool_name="vision_read",
                    capability=capability,
                    command=f"vision_read {target}",
                    status="FAILED",
                    stderr=f"Target image file not found: {target_path}",
                    exit_code=1,
                    duration_ms=elapsed_ms,
                    execution_failure=True,
                    failure_category="FILE_NOT_FOUND"
                )

            from backend.providers.router import model_router
            vision_prompt = (
                "Extract any readable text visible in this image. Report it EXACTLY as "
                "it appears, character for character, with no paraphrasing. If any text "
                "resembles a CTF flag format (e.g. flag{...}, picoCTF{...}, or similar), "
                "quote it verbatim first before anything else."
            )
            resp = await model_router.route_request(
                prompt=vision_prompt,
                capability="vision_read",
                image_path=target_path
            )
            elapsed_ms = (time.time() - start_time) * 1000
            if resp.is_refusal:
                refusal_msg = resp.refusal_reason or resp.content or "Model refusal for vision_read"
                return ToolExecutionResult(
                    tool_name="vision_read",
                    capability=capability,
                    command=f"vision_read {target_path}",
                    status="FAILED",
                    stderr=refusal_msg,
                    exit_code=1,
                    duration_ms=elapsed_ms,
                    execution_failure=True,
                    failure_category="MODEL_REFUSAL"
                )
            else:
                return ToolExecutionResult(
                    tool_name="vision_read",
                    capability=capability,
                    command=f"vision_read {target_path}",
                    status="SUCCESS",
                    stdout=resp.content,
                    stderr="",
                    exit_code=0,
                    duration_ms=elapsed_ms,
                    execution_failure=False,
                    failure_category=None
                )

        # Persistent interactive process capabilities
        if capability in ("interactive_open", "interactive_start"):
            cmd_to_run = (target or "").strip()
            if extra_args:
                cmd_to_run = f"{cmd_to_run} {extra_args}".strip() if cmd_to_run else extra_args.strip()
            if not cmd_to_run:
                elapsed_ms = (time.time() - start_time) * 1000
                return ToolExecutionResult(
                    tool_name="interactive_open",
                    capability="interactive_open",
                    command="interactive_open",
                    status="FAILED",
                    stderr="No command specified to start interactive session.",
                    exit_code=1,
                    duration_ms=elapsed_ms,
                    execution_failure=True,
                    failure_category="SYNTAX_ERROR",
                )

            exec_cwd = cwd if (cwd and os.path.exists(cwd)) else None
            try:
                sess = await execution_service.open_interactive(
                    cmd_to_run,
                    cwd=exec_cwd,
                    target=target,
                )
                read_res = await sess.read(timeout=3.0, idle_timeout=1.0)
                elapsed_ms = (time.time() - start_time) * 1000
                banner = read_res.data if read_res.data else "(Process started, waiting for input)"
                stdout_text = f"[SESSION: {sess.session_key}]\n{banner}"
                return ToolExecutionResult(
                    tool_name="interactive_open",
                    capability="interactive_open",
                    command=f"interactive_open {cmd_to_run}",
                    status="SUCCESS",
                    stdout=stdout_text,
                    stderr="",
                    exit_code=0 if sess.is_alive() else (sess.returncode or 0),
                    duration_ms=elapsed_ms,
                    execution_failure=False,
                    failure_category=None,
                )
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.warning(f"Failed to open interactive session for '{cmd_to_run}': {e}")
                return ToolExecutionResult(
                    tool_name="interactive_open",
                    capability="interactive_open",
                    command=f"interactive_open {cmd_to_run}",
                    status="FAILED",
                    stderr=f"Failed to open interactive session: {e}",
                    exit_code=1,
                    duration_ms=elapsed_ms,
                    execution_failure=True,
                    failure_category="EXECUTION_ERROR",
                )

        if capability == "interactive_send":
            session_key = (target or "").strip().strip('"').strip("'")
            data = extra_args if extra_args is not None else ""
            from backend.execution.interactive import interactive_manager
            sess = interactive_manager.get(session_key)
            elapsed_ms = (time.time() - start_time) * 1000
            if not sess or not sess.is_alive():
                return ToolExecutionResult(
                    tool_name="interactive_send",
                    capability="interactive_send",
                    command=f"interactive_send {session_key}",
                    status="FAILED",
                    stderr=f"Interactive session '{session_key}' not found or already closed.",
                    exit_code=1,
                    duration_ms=elapsed_ms,
                    execution_failure=True,
                    failure_category="SESSION_NOT_FOUND",
                )
            sent = await sess.send(data)
            elapsed_ms = (time.time() - start_time) * 1000
            if not sent:
                return ToolExecutionResult(
                    tool_name="interactive_send",
                    capability="interactive_send",
                    command=f"interactive_send {session_key}",
                    status="FAILED",
                    stderr="Failed to send data to interactive session.",
                    exit_code=1,
                    duration_ms=elapsed_ms,
                    execution_failure=True,
                    failure_category="SEND_FAILED",
                )
            return ToolExecutionResult(
                tool_name="interactive_send",
                capability="interactive_send",
                command=f"interactive_send {session_key}",
                status="SUCCESS",
                stdout=f"[SESSION: {session_key}] Sent {len(data)} characters.",
                stderr="",
                exit_code=0,
                duration_ms=elapsed_ms,
                execution_failure=False,
                failure_category=None,
            )

        if capability == "interactive_read":
            session_key = (target or "").strip().strip('"').strip("'")
            until_marker = extra_args.strip() if extra_args else None
            from backend.execution.interactive import interactive_manager
            sess = interactive_manager.get(session_key)
            elapsed_ms = (time.time() - start_time) * 1000
            if not sess:
                return ToolExecutionResult(
                    tool_name="interactive_read",
                    capability="interactive_read",
                    command=f"interactive_read {session_key}",
                    status="FAILED",
                    stderr=f"Interactive session '{session_key}' not found or already closed.",
                    exit_code=1,
                    duration_ms=elapsed_ms,
                    execution_failure=True,
                    failure_category="SESSION_NOT_FOUND",
                )
            read_res = await sess.read(until=until_marker)
            elapsed_ms = (time.time() - start_time) * 1000
            status = "SUCCESS" if (read_res.ok or read_res.data) else ("TIMEOUT" if read_res.timed_out else "FAILED")
            return ToolExecutionResult(
                tool_name="interactive_read",
                capability="interactive_read",
                command=f"interactive_read {session_key}",
                status=status,
                stdout=read_res.data,
                stderr="" if (read_res.ok or read_res.data) else ("Read timed out with no data" if read_res.timed_out else "Read failed/EOF"),
                exit_code=0 if (sess.is_alive() or read_res.data) else (sess.returncode or 0),
                duration_ms=elapsed_ms,
                execution_failure=False if (read_res.ok or read_res.data) else read_res.timed_out,
                failure_category="TIMEOUT" if (read_res.timed_out and not read_res.data) else None,
            )

        if capability == "interactive_send_and_read":
            session_key = (target or "").strip().strip('"').strip("'")
            data = extra_args if extra_args is not None else ""
            from backend.execution.interactive import interactive_manager
            sess = interactive_manager.get(session_key)
            elapsed_ms = (time.time() - start_time) * 1000
            if not sess or not sess.is_alive():
                return ToolExecutionResult(
                    tool_name="interactive_send_and_read",
                    capability="interactive_send_and_read",
                    command=f"interactive_send_and_read {session_key}",
                    status="FAILED",
                    stderr=f"Interactive session '{session_key}' not found or already closed.",
                    exit_code=1,
                    duration_ms=elapsed_ms,
                    execution_failure=True,
                    failure_category="SESSION_NOT_FOUND",
                )
            read_res = await sess.send_and_read(data)
            elapsed_ms = (time.time() - start_time) * 1000
            status = "SUCCESS" if (read_res.ok or read_res.data) else ("TIMEOUT" if read_res.timed_out else "FAILED")
            return ToolExecutionResult(
                tool_name="interactive_send_and_read",
                capability="interactive_send_and_read",
                command=f"interactive_send_and_read {session_key}",
                status=status,
                stdout=read_res.data,
                stderr="" if (read_res.ok or read_res.data) else ("Read timed out with no data" if read_res.timed_out else "Read failed/EOF"),
                exit_code=0 if (sess.is_alive() or read_res.data) else (sess.returncode or 0),
                duration_ms=elapsed_ms,
                execution_failure=False if (read_res.ok or read_res.data) else read_res.timed_out,
                failure_category="TIMEOUT" if (read_res.timed_out and not read_res.data) else None,
            )

        if capability == "interactive_close":
            session_key = (target or "").strip().strip('"').strip("'")
            from backend.execution.interactive import interactive_manager
            closed = await interactive_manager.close(session_key)
            elapsed_ms = (time.time() - start_time) * 1000
            return ToolExecutionResult(
                tool_name="interactive_close",
                capability="interactive_close",
                command=f"interactive_close {session_key}",
                status="SUCCESS" if closed else "FAILED",
                stdout=f"[SESSION: {session_key}] Closed." if closed else "",
                stderr="" if closed else f"Interactive session '{session_key}' not found or already closed.",
                exit_code=0 if closed else 1,
                duration_ms=elapsed_ms,
                execution_failure=not closed,
                failure_category=None if closed else "SESSION_NOT_FOUND",
            )

        # 1. Resolve candidate tools for capability
        candidate_tools = tool_registry.get_tools_for_capability(capability)
        if not candidate_tools:
            # Check dynamic acquisition plan for this capability/tool name
            from backend.execution.acquisition import acquisition_planner
            plan = acquisition_planner.plan_for_tool_name(capability)
            recipe_hint = f" Resolved install recipe: `{plan.command}`" if plan.command else ""
            return ToolExecutionResult(
                tool_name="none",
                capability=capability,
                command="",
                status="MISSING_TOOL",
                stderr=f"No approved tool registered for capability '{capability}'.{recipe_hint} Request installation with 'install_tool {capability}'."
            )

        # 2. Check host environment for installed tool
        env_tools = environment_detector.detect_environment().get("installed_tools", {})
        selected_tool: Optional[ToolMetadata] = None

        for tool in candidate_tools:
            if find_tool_binary(tool.binary) or env_tools.get(tool.binary, {}).get("installed"):
                selected_tool = tool
                break

        if not selected_tool:
            first_candidate = candidate_tools[0]
            from backend.execution.acquisition import acquisition_planner
            plan = acquisition_planner.plan_for_tool_name(first_candidate.tool_name)
            recipe = plan.command or first_candidate.installation_recipe
            return ToolExecutionResult(
                tool_name=first_candidate.tool_name,
                capability=capability,
                command="",
                status="MISSING_TOOL",
                stderr=f"Tool '{first_candidate.tool_name}' required for capability '{capability}' is not installed. Trusted install recipe: `{recipe}`. Request installation with 'install_tool {first_candidate.tool_name}'."
            )

        # 3. Construct safe execution command string & sanitize target format for specific tools
        target_port = None

        if parsed_target.startswith("http://") or parsed_target.startswith("https://"):
            u = urlparse(parsed_target)
            host_only = u.hostname or parsed_target
            target_port = u.port
            base_url = f"{u.scheme}://{u.netloc}"
        else:
            host_only = parsed_target.split(":")[0]
            base_url = parsed_target

        if selected_tool.tool_name == "nmap":
            target_for_cmd = host_only
            extra_port = f" -p {target_port}" if target_port else ""
            raw_args = format_tool_args(selected_tool.args_template, target=target_for_cmd) + extra_port
        elif selected_tool.tool_name == "ffuf":
            from backend.execution.wordlist import wordlist_resolver  # lazy — avoids circular import
            clean_url = base_url.rstrip("/")
            wl_path = wordlist_resolver.resolve("web_common")
            # Both values are external — the URL derives from the challenge target and the
            # wordlist is resolved at runtime — so both go through the template's quoted
            # substitution rather than being pasted straight into the command.
            raw_args = format_tool_args(
                selected_tool.args_template, target=clean_url, wordlist=wl_path
            ) + " -mc 200,301,302,401,403 -s"
        else:
            raw_args = format_tool_args(selected_tool.args_template, target=parsed_target)

        if extra_args:
            raw_args += f" {extra_args}"

        full_command = f"{selected_tool.binary} {raw_args}"
        full_command = sanitize_and_correct_command_target(full_command, target)
        logger.info(f"Executing tool '{selected_tool.tool_name}' (cwd={cwd}): {full_command}")

        # 4. Delegate subprocess execution to ExecutionService (Phase 3)
        exec_cwd = cwd if (cwd and os.path.exists(cwd)) else None
        _exec = await execution_service.run_command(
            full_command,
            cwd=exec_cwd,
            timeout_seconds=selected_tool.timeout_seconds,
            capability=capability,
            tool_name=selected_tool.tool_name,
        )
        stdout = _exec.stdout
        stderr = _exec.stderr
        exit_code = _exec.exit_code
        status = _exec.status

        elapsed_ms = (time.time() - start_time) * 1000
        classification = classify_tool_execution(selected_tool.tool_name, exit_code, stdout, stderr)

        return ToolExecutionResult(
            tool_name=selected_tool.tool_name,
            capability=capability,
            command=full_command,
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=elapsed_ms,
            execution_failure=classification["execution_failure"],
            failure_category=classification["failure_category"]
        )

    async def execute_raw_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout_seconds: int = 120,
        canonical_target: Optional[str] = None,
        stdin: Optional[str] = None
    ) -> ToolExecutionResult:
        start_time = time.time()
        raw_cmd = sanitize_and_correct_command_target(command.strip(), canonical_target)
        logger.info(f"Executing raw CLI command (cwd={cwd}): {raw_cmd}")

        # Model capability intercept: vision_read
        if raw_cmd.startswith("vision_read ") or raw_cmd.startswith("vision_read\t") or raw_cmd == "vision_read":
            target_file = raw_cmd.split(None, 1)[1].strip().strip('"').strip("'") if " " in raw_cmd or "\t" in raw_cmd else ""
            return await self.execute_capability(capability="vision_read", target=target_file, cwd=cwd)

        # Model capability intercept: interactive execution primitives
        if raw_cmd.startswith("interactive_open ") or raw_cmd.startswith("interactive_open\t") or raw_cmd == "interactive_open":
            cmd_arg = raw_cmd.split(None, 1)[1].strip() if " " in raw_cmd or "\t" in raw_cmd else ""
            return await self.execute_capability(capability="interactive_open", target=cmd_arg, cwd=cwd)

        if raw_cmd.startswith("interactive_send ") or raw_cmd.startswith("interactive_send\t"):
            parts = raw_cmd.split(None, 2)
            sess_key = parts[1].strip() if len(parts) > 1 else ""
            data_arg = parts[2] if len(parts) > 2 else ""
            return await self.execute_capability(capability="interactive_send", target=sess_key, extra_args=data_arg, cwd=cwd)

        if raw_cmd.startswith("interactive_read ") or raw_cmd.startswith("interactive_read\t") or raw_cmd == "interactive_read":
            parts = raw_cmd.split(None, 2)
            sess_key = parts[1].strip() if len(parts) > 1 else ""
            until_arg = parts[2].strip() if len(parts) > 2 else ""
            return await self.execute_capability(capability="interactive_read", target=sess_key, extra_args=until_arg, cwd=cwd)

        if raw_cmd.startswith("interactive_send_and_read ") or raw_cmd.startswith("interactive_send_and_read\t"):
            parts = raw_cmd.split(None, 2)
            sess_key = parts[1].strip() if len(parts) > 1 else ""
            data_arg = parts[2] if len(parts) > 2 else ""
            return await self.execute_capability(capability="interactive_send_and_read", target=sess_key, extra_args=data_arg, cwd=cwd)

        # interactive_close must be intercepted here too: it is a FORGE-internal
        # primitive with no filesystem binary, so without this branch the raw command
        # fell through to ExecutionService and died with COMMAND_NOT_FOUND/MISSING_TOOL,
        # leaving the session (and its child process) alive. Every other interactive
        # primitive is intercepted above; this one was simply missing.
        if raw_cmd.startswith("interactive_close ") or raw_cmd.startswith("interactive_close\t") or raw_cmd == "interactive_close":
            sess_key = raw_cmd.split(None, 1)[1].strip() if " " in raw_cmd or "\t" in raw_cmd else ""
            return await self.execute_capability(capability="interactive_close", target=sess_key, cwd=cwd)

        # Tool installation / acquisition intercept
        if raw_cmd.startswith("install_tool ") or raw_cmd.startswith("install_tool\t") or raw_cmd == "install_tool":
            tool_arg = raw_cmd.split(None, 1)[1].strip() if " " in raw_cmd or "\t" in raw_cmd else ""
            return await self.request_tool_install(tool_or_capability=tool_arg)

        if raw_cmd.startswith("acquire_tool ") or raw_cmd.startswith("acquire_tool\t") or raw_cmd == "acquire_tool":
            tool_arg = raw_cmd.split(None, 1)[1].strip() if " " in raw_cmd or "\t" in raw_cmd else ""
            return await self.request_tool_install(tool_or_capability=tool_arg)

        # Delegate subprocess execution to ExecutionService (Phase 3). When *stdin* is
        # supplied it is fed to the process once (Tier-1 scripted interactive, Phase 4.x §4).
        exec_cwd = cwd if (cwd and os.path.exists(cwd)) else None
        _exec = await execution_service.run_command(
            raw_cmd,
            cwd=exec_cwd,
            timeout_seconds=timeout_seconds,
            capability="custom_command",
            tool_name=os.path.basename(raw_cmd.split()[0]) if raw_cmd.strip() else "raw_cmd",
            stdin=stdin,
        )
        stdout = _exec.stdout
        stderr = _exec.stderr
        exit_code = _exec.exit_code
        status = _exec.status

        elapsed_ms = (time.time() - start_time) * 1000
        first_word = raw_cmd.split()[0] if raw_cmd else "raw_cmd"
        binary_name = os.path.basename(first_word)
        classification = classify_tool_execution(binary_name, exit_code, stdout, stderr)

        return ToolExecutionResult(
            tool_name=binary_name,
            capability="custom_command",
            command=raw_cmd,
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=elapsed_ms,
            execution_failure=classification["execution_failure"],
            failure_category=classification["failure_category"]
        )

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: int = 60,
        working_directory: Optional[str] = None,
        canonical_target: Optional[str] = None,
        stdin: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Universal tool invocation supporting CLI/bash commands and registered capabilities.

        *stdin* is an optional string written once to the subprocess's stdin (e.g. a
        sudo password for ``sudo -S`` invocations).  It is NOT stored, logged, or
        embedded into any command string — it is passed exclusively via the stdin pipe.
        """
        if tool_name in ["bash", "sh", "cli", "terminal", "command", "raw"]:
            cmd = params.get("command") or params.get("cmd") or ""
            return await self.execute_raw_command(
                command=cmd,
                cwd=working_directory,
                timeout_seconds=timeout,
                canonical_target=canonical_target,
                stdin=stdin,
            )
        elif tool_name in ["interactive_open", "interactive_send", "interactive_read", "interactive_send_and_read", "interactive_close", "interactive"]:
            action = tool_name
            if tool_name == "interactive":
                action = params.get("action") or "open"
                if not action.startswith("interactive_"):
                    action = f"interactive_{action}"
            target = params.get("target") or params.get("command") or params.get("cmd") or params.get("session_key") or params.get("session_id") or ""
            extra_args = params.get("extra_args") or params.get("data") or params.get("input") or params.get("until") or params.get("args") or ""
            return await self.execute_capability(
                capability=action,
                target=str(target),
                extra_args=str(extra_args) if extra_args else None,
                cwd=working_directory
            )
        else:
            target = params.get("target") or params.get("url") or params.get("ip") or params.get("file_path") or params.get("image_path") or params.get("path") or canonical_target or ""
            extra_args = params.get("extra_args") or params.get("args") or ""
            return await self.execute_capability(
                capability=tool_name,
                target=target,
                extra_args=extra_args,
                cwd=working_directory
            )


tool_manager = ToolManager()
