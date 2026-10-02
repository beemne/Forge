"""
FORGE Agent Runtime — the central agent execution loop (Step 2).

`AgentRuntime.run()` drives one mission session through the canonical cycle:

    command → stdout/stderr → observation → state change → decision → next action

for every turn, persisting each step to the trajectory so the mission survives a
crash, a pause, or a provider swap. It ties together the SessionManager,
TrajectoryStore, ObservationEngine, DecisionEngine, RecoveryEngine,
RepetitionDetector, FlagVerifier and ContextBuilder.

The loop is provider-agnostic (Step 8): it only ever talks to a
:class:`~backend.agent_runtime.decision.ProviderGateway`, and a provider failure
fails over WITHOUT touching session state. It is also executor-agnostic: it depends
only on the :class:`~backend.agent_runtime.action.ToolExecutor` protocol, so tests
inject a scripted executor (no subprocess, no network).

Termination conditions: verified flag · explicit completion (budget) · mission
timeout · cancellation · unrecoverable failure · max turns.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from backend.agent_runtime.action import Action, ActionType, ActionValidator, ExecResult, ToolExecutor
from backend.privilege.gate import require_approval, SHARED_PENDING_APPROVALS
from backend.websocket.manager import ws_manager
from backend.agent_runtime.context import ContextBuilder
from backend.agent_runtime.decision import DecisionEngine, ProviderGateway, RouterProviderGateway
from backend.agent_runtime.observation import ObservationEngine
from backend.agent_runtime.recovery import RecoveryEngine, RecoveryStrategy
from backend.agent_runtime.repetition import RepetitionDetector, RepetitionKind
from backend.agent_runtime.session import AgentSession, SessionManager, session_manager
from backend.agent_runtime.trajectory import TrajectoryStore, trajectory_store
from backend.agent_runtime.verifier import (
    FlagVerifier, AnswerStatus, VerifierAgent, AnswerCandidate, AnswerSource,
)

logger = logging.getLogger("forge.agent_runtime.runtime")



@dataclass
class RunResult:
    session_id: str
    status: str                    # COMPLETED | FAILED | CANCELLED | PAUSED | TIMEOUT | MAX_TURNS
    verified_flag: Optional[str] = None
    outcome: Optional[str] = None  # success | failure | incomplete
    turns: int = 0
    reason: str = ""
    flag_candidates: List[str] = field(default_factory=list)


# ── Generic target/URL resolution (Task 3) ─────────────────────────────────── #
# Resolve a possibly-relative URL reference against the challenge's canonical target.
# Purely mechanical (urljoin) — no challenge-specific rules, no hardcoded endpoints.

def resolve_target_url(reference: str, canonical_target: Optional[str]) -> str:
    """Resolve *reference* against *canonical_target*.

    - empty reference               → the canonical target (or "")
    - reference already absolute     → returned unchanged (scheme present)
    - no canonical target            → reference returned unchanged
    - relative reference             → joined onto the canonical target, preserving
                                       its scheme, host, non-default port and path prefix.
    A leading-'/' reference resolves against the host root; a bare relative reference
    resolves *under* the canonical target treated as a directory.
    """
    from urllib.parse import urljoin
    ref = (reference or "").strip()
    base = (canonical_target or "").strip()
    if not ref:
        return base
    if "://" in ref:
        return ref
    if not base:
        return ref
    if "://" not in base:
        base = "http://" + base
    if ref.startswith("/"):
        return urljoin(base, ref)
    dir_base = base if base.endswith("/") else base + "/"
    return urljoin(dir_base, ref)


# ── Pre-execution generated-Python consistency check (Tasks 2, 3, 8) ────────── #
# Catches a handful of GENERIC, reliably-detectable mistakes before a mission turn is
# spent running a script that is guaranteed to fail. All checks are AST-based (never
# fragile source regexes) and conservative: only unambiguous cases fire.
#
#   * INTERPRETER_ASSUMPTION — a Python-2-only string codec (str.encode('base64'), …)
#     that always raises LookupError on a Python 3 host.
#   * FILE_NOT_FOUND         — the script open()s a relative local file for READING
#     that does not exist and is not created by an earlier write in the same script.
#   * INVALID_URL            — a scheme-less URL literal passed to requests/httpx/urlopen
#     that always raises "No scheme supplied".
#
# Anything not reliably detectable statically (arbitrary bytes/str misuse, dynamic
# paths, non-literal URLs) is deliberately left to execution feedback + recovery.

# Python-2-only text/binary transform codecs (invalid for str.encode/str.decode in py3).
_PY2_TEXT_CODECS = {
    "base64", "base64_codec", "hex", "hex_codec", "rot13", "rot_13", "uu", "uu_codec",
    "zlib", "zlib_codec", "bz2", "bz2_codec", "quopri", "quopri_codec", "string_escape",
}
# HTTP entrypoints whose first positional argument is a URL. Kept tight (known modules
# only) so a scheme-less first arg is UNAMBIGUOUSLY a bad URL — never a dict.get() etc.
_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "request"}
_HTTP_MODULES = {"requests", "httpx"}


def _py_str_literal(node: Any) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _py_is_write_mode(mode: str) -> bool:
    m = (mode or "").lower()
    return any(c in m for c in "wax") or "+" in m


def _py_is_relative_local(path: str) -> bool:
    """True only for a workspace-relative path — absolute/home/drive/URL refs are
    target- or host-side and are never flagged as a missing local artifact."""
    if not path or "://" in path:
        return False
    if path.startswith(("/", "\\", "~")):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", path):
        return False
    return True


def _fail(category: str, message: str) -> "ExecResult":
    return ExecResult(status="FAILED", stderr=f"Pre-execution check: {message}",
                      exit_code=-1, execution_failure=True, failure_category=category)


def analyze_python_script(script: str, *, cwd: Optional[str] = None,
                          canonical_target: Optional[str] = None) -> Optional["ExecResult"]:
    """Return a structured failure ExecResult for a reliably-detectable pre-exec issue,
    or None when the script is clear to run. Syntax errors are the caller's concern."""
    try:
        tree = ast.parse(script or "")
    except SyntaxError:
        return None  # caller reports SYNTAX_ERROR separately

    written: set = set()
    read_paths: List[str] = []
    py2_codec: Optional[str] = None
    bad_url: Optional[str] = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func

        # open(path[, mode]) — track write-mode targets, collect read-mode relative paths.
        if isinstance(f, ast.Name) and f.id == "open" and node.args:
            p = _py_str_literal(node.args[0])
            if p is not None:
                mode = _py_str_literal(node.args[1]) if len(node.args) > 1 else "r"
                for kw in node.keywords:
                    if kw.arg == "mode":
                        mode = _py_str_literal(kw.value) or mode
                mode = mode or "r"
                norm = p[2:] if p.startswith("./") else p
                if _py_is_write_mode(mode):
                    written.add(norm)
                elif _py_is_relative_local(p):
                    read_paths.append(norm)
            continue

        # x.encode('base64') / x.decode('hex') — Python-2-only string codec.
        if isinstance(f, ast.Attribute) and f.attr in ("encode", "decode") and node.args:
            codec = _py_str_literal(node.args[0])
            if codec and codec.lower().replace("-", "_") in _PY2_TEXT_CODECS and py2_codec is None:
                py2_codec = f"{f.attr}('{codec}')"
            continue

        # requests.post('upload.php') / httpx.get(...) / urlopen('upload.php') — scheme-less URL.
        if bad_url is None:
            method = recv = None
            if isinstance(f, ast.Attribute):
                method, recv = f.attr, (f.value.id if isinstance(f.value, ast.Name) else None)
            elif isinstance(f, ast.Name):
                method = f.id
            arg0 = _py_str_literal(node.args[0]) if node.args else None
            if arg0 and "://" not in arg0:
                if (recv in _HTTP_MODULES and method in _HTTP_METHODS) or method in ("urlopen", "urlretrieve"):
                    bad_url = arg0

    if py2_codec is not None:
        return _fail("INTERPRETER_ASSUMPTION",
                     f"the script calls .{py2_codec}, a Python-2-only string codec that raises "
                     f"LookupError on this Python 3 host. Use the base64 / binascii / codecs modules "
                     f"instead (e.g. base64.b64encode(...), codecs.encode(data, '...')).")

    check_dir = cwd if (cwd and os.path.isdir(cwd)) else "."
    missing = [p for p in dict.fromkeys(read_paths)
               if p not in written and not os.path.exists(os.path.join(check_dir, p))]
    if missing:
        return _fail("FILE_NOT_FOUND",
                     f"the script reads local file(s) that do not exist in the workspace and are "
                     f"not created by an earlier step: {', '.join(missing[:5])}. Create, download or "
                     f"generate the artifact first, or use a technique that does not need it. Do NOT "
                     f"re-run the same script unchanged.")

    if bad_url is not None:
        resolved = resolve_target_url(bad_url, canonical_target)
        hint = (f" Use the absolute URL '{resolved}'." if resolved and "://" in resolved
                else " Use the full target URL including scheme and host (e.g. http://<host>/...).")
        return _fail("INVALID_URL",
                     f"an HTTP request used the scheme-less URL '{bad_url}', which raises "
                     f"'No scheme supplied'.{hint}")

    return None


class RealToolExecutor:
    """Production executor: routes actions through FORGE's existing ToolManager.

    COMMAND      → tool_manager.execute_raw_command
    PYTHON_SCRIPT→ writes solve.py in the workspace, then runs it (raw command)
    TOOL_CALL    → tool_manager.execute_capability

    EVERY process-spawning path is gated by the shared operator-approval gate
    (``backend/privilege/gate.py``) immediately BEFORE its execution call, so an agent
    cannot run a command — least of all a DANGEROUS one — without explicit operator
    approval. A denied/timed-out command returns a normal FAILED ``ExecResult`` so the
    reasoning loop can pivot, instead of raising.

    ``approval_gate`` is a test seam, not a policy switch: it defaults to ``None``,
    which resolves to the REAL :func:`~backend.privilege.gate.require_approval` at call
    time. A misconfigured/mis-wired caller therefore gets MORE security, never less.
    Only tests that exercise real subprocess plumbing inject an explicit auto-approving
    stub, exactly as they already inject their own ``tool_manager``.
    """

    # Capabilities that spawn a process from a caller-supplied string. Unlike a fixed
    # binary (nmap, curl), the capability name itself is NOT what runs — `target` IS the
    # command line — so it must be classified and gated as a command.
    _SHELL_CAPABILITIES = frozenset({"interactive_open", "interactive_start"})

    def __init__(self, tool_manager=None, approval_gate=None):
        if tool_manager is None:
            from backend.tools.manager import tool_manager as tm
            tool_manager = tm
        self.tool_manager = tool_manager
        self._approval_gate = approval_gate

    async def _gate(self, cmd: str) -> Optional[ExecResult]:
        """Gate *cmd*. Returns None when approved, else a blocked ExecResult."""
        gate = self._approval_gate or require_approval
        approved, decision, _sudo_pw = await gate(
            cmd=cmd,
            agent_id="agent_runtime",
            pending_approvals=SHARED_PENDING_APPROVALS,
            broadcast_fn=ws_manager.broadcast,
            challenge_id=None,
            run_id=None,
        )
        if approved:
            return None
        timed_out = decision is None
        return ExecResult(
            command=cmd,
            status="FAILED",
            stdout="",
            stderr=(
                f"[PRIVILEGE DENIED] Operator did not approve this command "
                f"({'no response before the approval window closed' if timed_out else 'denied'}). "
                f"The command was NOT executed (capability gap). Preserving original intent for privilege escalation."
            ),
            exit_code=-1,
            execution_failure=True,
            failure_category="CAPABILITY_GAP",
        )

    async def execute(self, action: Action, *, cwd: Optional[str] = None,
                      timeout_seconds: int = 120, canonical_target: Optional[str] = None) -> ExecResult:
        if action.type == ActionType.COMMAND:
            blocked = await self._gate(action.command)
            if blocked is not None:
                blocked.command = blocked.command or action.display()
                return blocked
            r = await self.tool_manager.execute_raw_command(
                action.command, cwd=cwd, timeout_seconds=timeout_seconds,
                canonical_target=canonical_target, stdin=(action.stdin or None))
            return ExecResult.from_tool_result(r)

        if action.type == ActionType.PYTHON_SCRIPT:
            import ast
            try:
                ast.parse(action.script)
            except SyntaxError as se:
                return ExecResult(
                    command=action.display(),
                    status="FAILED",
                    stderr=f"SyntaxError in generated Python script at line {se.lineno}, col {se.offset}: {se.msg}\n  {se.text or ''}",
                    exit_code=-1,
                    execution_failure=True,
                    failure_category="SYNTAX_ERROR",
                )
            # Pre-execution consistency check (missing local artifact / scheme-less URL /
            # Python-2-only codec). Returns a structured failure so the reasoning loop gets
            # actionable evidence instead of wasting a turn on a guaranteed runtime crash.
            issue = analyze_python_script(action.script, cwd=cwd, canonical_target=canonical_target)
            if issue is not None:
                issue.command = issue.command or action.display()
                return issue
            script_path = os.path.join(cwd or ".", "solve.py")
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(action.script)
            except Exception as e:
                return ExecResult(status="FAILED", stderr=f"Failed to write solve.py: {e}",
                                  exit_code=-1, execution_failure=True, failure_category="IO")
            # Use sys.executable to guarantee the current Python interpreter is used.
            # On Windows, shutil.which("python") may point to a zero-byte Store stub.
            py_bin = sys.executable
            if not py_bin or not os.path.exists(py_bin):
                from backend.execution.backends.local import _resolve_python
                resolved = _resolve_python()
                py_bin = resolved or "python"
            py_cmd = f'"{py_bin}" "{script_path}"'
            # Gate the command that will ACTUALLY run (interpreter + script path), not the
            # model's raw script text — that is what reaches the shell.
            blocked = await self._gate(py_cmd)
            if blocked is not None:
                blocked.command = action.display()
                return blocked
            r = await self.tool_manager.execute_raw_command(
                py_cmd, cwd=cwd, timeout_seconds=timeout_seconds,
                canonical_target=canonical_target, stdin=(action.stdin or None))
            return ExecResult.from_tool_result(r)

        if action.type == ActionType.TOOL_CALL:
            args = dict(action.tool_args or {})
            target = args.pop("target", None) or canonical_target or ""
            capability = action.capability or action.tool_name
            if capability in self._SHELL_CAPABILITIES:
                # `target`/extra_args ARE the command line this capability spawns.
                extra = args.get("extra_args")
                spawned = (target or "").strip()
                if extra:
                    spawned = f"{spawned} {extra}".strip() if spawned else str(extra).strip()
                blocked = await self._gate(spawned)
                if blocked is not None:
                    blocked.command = action.display()
                    return blocked
            r = await self.tool_manager.execute_capability(
                capability=capability, target=target,
                cwd=cwd, **args)
            return ExecResult.from_tool_result(r)

        return ExecResult(status="SUCCESS", stdout="", command=action.display())


class AgentRuntime:
    """Central execution loop. Provider- and executor-agnostic by construction."""

    def __init__(
        self,
        tool_executor: ToolExecutor,
        provider_gateway: Optional[ProviderGateway] = None,
        *,
        sessions: SessionManager = session_manager,
        trajectory: TrajectoryStore = trajectory_store,
        observation_engine: Optional[ObservationEngine] = None,
        context_builder: Optional[ContextBuilder] = None,
        verifier: Optional[FlagVerifier] = None,
        verifier_agent: Optional[VerifierAgent] = None,
        execution_backend: Optional[Any] = None,
        learn_on_completion: bool = True,
    ):
        self.tool_executor = tool_executor
        self.gateway = provider_gateway or RouterProviderGateway()
        self.sessions = sessions
        self.trajectory = trajectory
        self.observations = observation_engine or ObservationEngine()
        self.context = context_builder or ContextBuilder()
        self.verifier = verifier or FlagVerifier()
        router = getattr(self.gateway, "router", None) or self.gateway
        self.verifier_agent = verifier_agent or VerifierAgent(resolver=self.verifier, router=router)
        self.decider = DecisionEngine(self.gateway)
        self.learn_on_completion = learn_on_completion

        # (Step 19) The intelligence layer is OS-independent; the execution backend
        # REPORTS what can actually run here so prompts and skills are environment-aware
        # (Step 12) instead of assuming a Linux host. Detection is best-effort/non-fatal.
        self._backend = execution_backend
        self._caps = None
        self._detected_os = "Linux"
        self._tool_inventory = ""
        self._python_libs = ""
        try:
            if self._backend is None:
                from backend.agent_runtime.execution_backend import execution_backend as _eb
                self._backend = _eb
            self._caps = self._backend.capabilities()
            # Report the EXECUTION environment's OS, capitalised for the prompt template.
            self._detected_os = (self._caps.os or "linux").capitalize()
            self._tool_inventory = self._caps.tool_inventory_str()
            self._python_libs = self._caps.python_libs_str()
            # Give the context builder the capability report so retrieved memory is
            # ranked/annotated by what is actually runnable here (§8, §12).
            if getattr(self.context, "_capabilities", None) is None:
                self.context._capabilities = self._caps
        except Exception:
            pass

    # ------------------------------------------------------------------ #

    async def run(
        self,
        session: AgentSession,
        *,
        max_turns: Optional[int] = None,
        max_seconds: Optional[float] = None,
        cwd: Optional[str] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        timeout_seconds_per_command: int = 120,
        checkpoint_every: int = 5,
    ) -> RunResult:
        """Execute the mission loop for *session* until a terminal condition."""
        try:
            return await self._run_loop(
                session,
                max_turns=max_turns,
                max_seconds=max_seconds,
                cwd=cwd,
                cancel_check=cancel_check,
                timeout_seconds_per_command=timeout_seconds_per_command,
                checkpoint_every=checkpoint_every,
            )
        finally:
            await self._cleanup_interactive_sessions(session.state)

    async def _run_loop(
        self,
        session: AgentSession,
        *,
        max_turns: Optional[int] = None,
        max_seconds: Optional[float] = None,
        cwd: Optional[str] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        timeout_seconds_per_command: int = 120,
        checkpoint_every: int = 5,
    ) -> RunResult:
        state = session.state
        recovery = RecoveryEngine()
        repetition = RepetitionDetector()
        validator = ActionValidator()

        deadline = (time.time() + max_seconds) if max_seconds else None
        recovery_directive = ""
        memory_context, memory_ids = "", []
        cross_session_failures = ""
        all_retrieved_ids: List[str] = []   # accumulated across the mission for post-run feedback (§12/§14)
        provider_fail_streak = 0
        malformed_streak = 0
        turns = 0

        self.sessions.save(session, status="RUNNING")

        self._record(session, "PLAN", strategy=state.phase,
                     decision_summary=f"Objective: {state.current_objective}")

        while True:


            if cancel_check and cancel_check():
                self.sessions.pause(session)
                return self._result(session, "CANCELLED", turns, "Cancelled by operator/kill-switch.")
            # ── Termination: timeout ──
            if deadline and time.time() >= deadline:
                self.sessions.complete(session, outcome="incomplete")
                self._learn(session, "failure", all_retrieved_ids)
                return self._result(session, "TIMEOUT", turns, "Mission time budget exhausted.")
            # ── Termination: explicit turn bound (only when the caller supplies one) ──
            # max_turns=None keeps FORGE's no-auto-stop lifecycle: runs end on flag
            # capture, the operator kill switch, or the time budget -- never merely
            # because a counter filled up. A caller that passes a number opts into a
            # hard bound, which matters because neither COMPLETE path below can end
            # the loop on its own: a provider that only ever repeats a flag assertion
            # or BUDGET_EXHAUSTED would otherwise spin here indefinitely.
            # Status is MAX_TURNS (not INCOMPLETE): it is the value in this module's
            # status contract, and swarm/reasoning.py + swarm/supervisor.py classify
            # failures by matching that literal string.
            if max_turns is not None and turns >= max_turns:
                self.sessions.complete(session, outcome="incomplete")
                self._learn(session, "failure", all_retrieved_ids)
                return self._result(session, "MAX_TURNS", turns,
                                    f"Turn limit reached ({max_turns}).")

            turns += 1

            # ── (Step 3) retrieve relevant context (bounded, refreshed periodically) ──
            if turns == 1 or turns % 4 == 0:
                memory_context, memory_ids = self.context.retrieve_memory(state)
                self._record_retrieval(session, memory_ids)
                for mid in memory_ids:
                    if mid not in all_retrieved_ids:
                        all_retrieved_ids.append(mid)
                # (§5/§9) cross-session dead-ends from OTHER missions with similar conditions.
                try:
                    cross_session_failures = self.context.recall_cross_session_failures(
                        state, exclude_session=session.id)
                except Exception:
                    cross_session_failures = ""

            recent = self.trajectory.get_recent(
                session.id, n=8, event_types=["COMMAND", "OBSERVATION", "REPLAN", "RECOVERY"])
            system_instruction, user_prompt = self.context.build(
                state=state, latest_observation=None, recent_events=recent,
                recovery_directive=recovery_directive, memory_context=memory_context,
                tool_inventory=self._tool_inventory, python_libs=self._python_libs,
                detected_os=self._detected_os, cross_session_failures=cross_session_failures,
                # Budget/compact against the model the last turn used (Workstream D). Empty on
                # turn 1 (budgeting disabled); a mid-run model switch is picked up next turn,
                # so the budget always tracks the model actually in use (D3 fallback-aware).
                model_name=getattr(session, "model_name", "") or "",
            )

            # ── (Step 4) ask model for next action ──
            depth = "deep" if (recovery_directive or state.phase in ("exploit", "escalate")) else "fast"
            result = await self.decider.decide(
                system_instruction=system_instruction, user_prompt=user_prompt,
                reasoning_depth=depth)
            session.add_tokens(result.completion.prompt_tokens, result.completion.completion_tokens)
            session.provider_name = result.completion.provider_name or session.provider_name
            session.model_name = result.completion.model_name or session.model_name

            # ── Provider failure → fail over WITHOUT losing the session (Step 8) ──
            if result.provider_failed:
                provider_fail_streak += 1
                plan = recovery.diagnose(provider_failed=True)
                self._record(session, "RECOVERY", result="SWITCH_PROVIDER",
                             decision_summary=plan.directive, strategy="provider_failover")
                if provider_fail_streak >= 3:
                    self.sessions.fail(session, "All providers exhausted repeatedly.")
                    return self._result(session, "FAILED", turns,
                                        "Provider layer exhausted — session preserved for resume.")
                recovery_directive = plan.directive
                continue
            provider_fail_streak = 0

            # ── Malformed model output → ask for a single clean action ──
            if result.malformed or result.action is None:
                malformed_streak += 1
                plan = recovery.diagnose(malformed_response=True)
                self._record(session, "RECOVERY", result="MALFORMED",
                             decision_summary=plan.directive, strategy="reprompt")
                if malformed_streak >= 4:
                    self.sessions.fail(session, "Model repeatedly produced unparseable output.")
                    return self._result(session, "FAILED", turns, "Unparseable model output.")
                recovery_directive = plan.directive
                continue
            malformed_streak = 0

            action = result.action
            decision = result.decision
            if decision.objective:
                state.set_objective(decision.objective)

            # ── (Step 5) validate the action (validation separate from execution) ──
            vr = validator.validate(action)
            if not vr.ok:
                self._record(session, "RECOVERY", result="INVALID", action_type=action.type.value,
                             command=action.display(),
                             decision_summary=f"Rejected action: {vr.reason}", strategy="validation")
                state.record_failure(action.display(), vr.reason)
                recovery_directive = f"Your previous action was rejected ({vr.reason}). Choose a valid, different action."
                continue

            # ── Explicit completion paths ──
            if action.type == ActionType.COMPLETE:
                if action.reason == "budget_exhausted":
                    recovery_directive = "Budget restrictions are lifted. Formulate your next hypothesis or alternative attack vector and proceed."
                    continue
                # A model-asserted flag is a CANDIDATE only (never auto-verified from prose).
                task_ctx = {
                    "description": getattr(state, "description", "") or getattr(state, "current_objective", ""),
                    "challenge_name": getattr(state, "challenge_name", ""),
                    "category": getattr(state, "category", ""),
                    "flag_pattern": getattr(state, "flag_format", ""),
                    "target_scope": getattr(state, "target", ""),
                }
                verdict = await self.verifier_agent.verify(
                    action.command,
                    source=AnswerSource.LLM_PROSE,
                    task_context=task_ctx,
                )
                if verdict.status == AnswerStatus.CANDIDATE:
                    if verdict.candidate not in state.flag_candidates:
                        state.flag_candidates.append(verdict.candidate)
                    self._record(session, "FLAG_CANDIDATE", result="CANDIDATE",
                                 decision_summary=f"Model asserted flag (unverified): {verdict.candidate}")
                    recovery_directive = ("You asserted a flag but it was NOT observed in real command "
                                          "output. Run the exact command whose output contains it so it "
                                          "can be verified — do not just restate it.")
                    continue
                self._record(session, "RECOVERY", result="REJECTED",
                             decision_summary=f"Rejected flag assertion: {verdict.reasons}")
                recovery_directive = "That is not a valid flag. Continue the investigation."
                continue

            # ── (Step 12) pre-exec repetition: never blindly repeat ──
            pre_kind = repetition.classify(action.display())
            if pre_kind == RepetitionKind.EXACT_REPEAT:
                plan = recovery.diagnose(
                    repetition=repetition.observe(action.display(), failed=True,
                                                  failure_category="repeat", novel=False))
                self._record(session, "REPLAN", result="EXACT_REPEAT", command=action.display(),
                             decision_summary=plan.directive, strategy="repetition_guard")
                state.record_dead_end(f"Repeated: {action.display()[:80]}")
                recovery_directive = plan.directive
                repetition.reset_streaks()
                continue

            # ── (Step 6/DECISION) record the decision, then execute ──
            self._record(session, "DECISION", action_type=action.type.value, command=action.display(),
                         decision_summary=(decision.reason or "")[:400], strategy=decision.strategy,
                         provider=session.provider_name, model=session.model_name,
                         prompt_tokens=result.completion.prompt_tokens,
                         completion_tokens=result.completion.completion_tokens)
            state.record_command(action.display())

            exec_result = await self.tool_executor.execute(
                action, cwd=cwd, timeout_seconds=timeout_seconds_per_command,
                canonical_target=(state.scope[0] if state.scope else None))

            # ── (Step 7) capture the complete result (COMMAND event folds in OUTPUT) ──
            self._record(session, "COMMAND", action_type=action.type.value,
                         command=exec_result.command or action.display(),
                         tool_name=exec_result.tool_name, stdout=exec_result.stdout,
                         stderr=exec_result.stderr, exit_code=exec_result.exit_code,
                         duration_ms=exec_result.duration_ms, result=exec_result.status)

            # ── (Step 8) create observation (evidence only) ──
            obs = self.observations.observe(exec_result, state)
            if obs.summary or obs.errors:
                self._record(session, "OBSERVATION", result=exec_result.status,
                             observation=obs.to_dict(), decision_summary=obs.summary)

            # ── Flag / Answer verification from REAL tool output ──
            candidates_to_eval: List[AnswerCandidate | str] = []
            if getattr(obs, "answer_candidates", None):
                candidates_to_eval.extend(obs.answer_candidates)
            if getattr(obs, "flag_candidates", None):
                for fc in obs.flag_candidates:
                    if not any(
                        (isinstance(c, AnswerCandidate) and c.value == fc) or c == fc
                        for c in candidates_to_eval
                    ):
                        candidates_to_eval.append(fc)

            task_ctx = {
                "description": getattr(state, "description", "") or getattr(state, "current_objective", ""),
                "challenge_name": getattr(state, "challenge_name", ""),
                "category": getattr(state, "category", ""),
                "flag_pattern": getattr(state, "flag_format", ""),
                "target_scope": getattr(state, "target", ""),
            }

            for cand in candidates_to_eval:
                val = cand.value if isinstance(cand, AnswerCandidate) else str(cand)
                if not val or not val.strip():
                    continue

                verdict = await self.verifier_agent.verify(
                    cand,
                    task_context=task_ctx,
                    command=exec_result.command or action.display(),
                    action_succeeded=exec_result.succeeded,
                    evidence={"stdout": exec_result.stdout, "stderr": exec_result.stderr},
                    authoritative=False,
                )

                if verdict.status in (AnswerStatus.VERIFIED, AnswerStatus.RESOLVED):
                    state.set_verified_flag(verdict.candidate)
                    if verdict.candidate not in state.flag_candidates:
                        state.flag_candidates.append(verdict.candidate)
                    state.record_success(decision.strategy or action.display())
                    event_type = "FLAG_VERIFIED" if verdict.status == AnswerStatus.VERIFIED else "ANSWER_RESOLVED"
                    self._record(session, event_type, result=verdict.status.value,
                                 decision_summary=f"Answer {verdict.status.value} from command output: {verdict.candidate}")
                    self.sessions.complete(session, outcome="success", verified_flag=verdict.candidate)
                    self._learn(session, "success", all_retrieved_ids)
                    return self._result(session, "COMPLETED", turns,
                                        f"Answer {verdict.status.value} from real command output.")


                elif verdict.status == AnswerStatus.CANDIDATE:
                    if verdict.candidate not in state.flag_candidates:
                        state.flag_candidates.append(verdict.candidate)
                    self._record(session, "FLAG_CANDIDATE", result=AnswerStatus.CANDIDATE.value,
                                 decision_summary=f"Candidate (unverified): {verdict.candidate} — {verdict.reasons}")

                elif verdict.status == AnswerStatus.REJECTED:
                    self._record(session, "FLAG_REJECTED", result=AnswerStatus.REJECTED.value,
                                 decision_summary=f"Rejected candidate {verdict.candidate}: {verdict.reasons}")


            # ── (Step 9) update state; (STATE_UPDATE) record the delta ──
            delta = state.apply_observation(obs)
            if not delta.is_empty():
                changed = []
                if delta.new_endpoints:
                    changed.append(f"{len(delta.new_endpoints)} endpoint(s)")
                if delta.new_vulnerabilities:
                    changed.append(f"{len(delta.new_vulnerabilities)} vuln(s)")
                if delta.new_technologies:
                    changed.append(f"{len(delta.new_technologies)} tech")
                if delta.new_credentials:
                    changed.append(f"{len(delta.new_credentials)} cred(s)")
                if delta.new_flag_candidates:
                    changed.append(f"{len(delta.new_flag_candidates)} flag candidate(s)")
                self._record(session, "STATE_UPDATE", state_delta=delta.to_dict(),
                             decision_summary=", ".join(changed) or "state advanced")

            # ── (Step 11/12) evaluate progress + repetition (post-exec) ──
            rep_report = repetition.observe(
                action.display(), failed=not exec_result.succeeded,
                failure_category=exec_result.failure_category or ("fail" if not exec_result.succeeded else None),
                novel=obs.novelty)

            # ── (Step 13/14) diagnose + recover / replan ──
            plan = recovery.diagnose(exec_result=exec_result, observation=obs,
                                     repetition=rep_report, command=action.command)
            if plan.strategy == RecoveryStrategy.CONTINUE:
                recovery_directive = ""
                if exec_result.succeeded and obs.novelty:
                    state.record_success(decision.strategy or action.display()[:80])
            elif plan.strategy == RecoveryStrategy.ABORT:
                self._record(session, "RECOVERY", result="ABORT",
                             decision_summary=plan.directive, strategy="abort")
                self.sessions.fail(session, plan.directive)
                self._learn(session, "failure", all_retrieved_ids)
                return self._result(session, "FAILED", turns, "Unrecoverable: " + plan.directive)
            elif plan.strategy == RecoveryStrategy.REPLAN:
                self._record(session, "REPLAN", result=plan.category.value,
                             decision_summary=plan.directive, strategy="replan")
                state.record_failure(action.display(), plan.category.value)
                recovery_directive = plan.directive
                repetition.reset_streaks()
            else:
                self._record(session, "RECOVERY", result=plan.category.value,
                             decision_summary=plan.directive, strategy=plan.strategy.value)
                if not exec_result.succeeded:
                    state.record_failure(action.display(), plan.category.value)
                if exec_result.failure_category == "CAPABILITY_GAP":
                    cap_name = action.capability or action.tool_name or (action.command.split()[0] if action.command else "")
                    state.record_capability_gap(
                        action=action.display(),
                        capability=cap_name,
                        target=state.target,
                        reason=exec_result.stderr[:200] if exec_result.stderr else "Privilege denial (capability gap)",
                        privilege_level="PRIVILEGED",
                    )
                recovery_directive = plan.directive

            # ── Persist state every turn; checkpoint periodically (crash-safe resume) ──
            session.last_sequence = self.trajectory.next_sequence(session.id) - 1
            self.sessions.save(session)
            if checkpoint_every and turns % checkpoint_every == 0:
                self.sessions.checkpoint(session, last_action=action.display()[:120])

    # ------------------------------------------------------------------ #


    async def _cleanup_interactive_sessions(self, state: Any) -> None:
        try:
            from backend.execution.interactive import interactive_manager
            for sk in list(getattr(state, "interactive_sessions", []) or []):
                try:
                    await interactive_manager.close(sk, reason="session_ended")
                except Exception:
                    pass
        except Exception:
            pass


    def _record(self, session: AgentSession, event_type: str, **kw) -> None:
        self.trajectory.record(
            session_id=session.id, event_type=event_type, run_id=session.run_id,
            challenge_id=session.challenge_id, agent_id=session.agent_id, **kw)

    def _record_retrieval(self, session: AgentSession, memory_ids: List[str]) -> None:
        if not memory_ids:
            return
        try:
            from backend.knowledge.experience_memory import experience_memory
            experience_memory.record_retrieval(memory_ids, run_id=session.run_id,
                                               challenge_id=session.challenge_id)
        except Exception:
            pass

    def _learn(self, session: AgentSession, outcome: str, retrieved_ids: List[str]) -> None:
        """Distil the finished mission into a generalized experience (Phase 2, §4-6, §20).

        Non-fatal: a learning failure never changes the RunResult. Only runs on a truly
        terminal outcome — a resumable PAUSE (operator stop) is NOT learned from, because
        the mission may still continue and be learned from at its real end.
        """
        if not self.learn_on_completion:
            return
        try:
            from backend.agent_runtime.learning import runtime_learner
            runtime_learner.learn_from_session(
                session, outcome=outcome, retrieved_memory_ids=retrieved_ids)
        except Exception:
            pass

    def _result(self, session: AgentSession, status: str, turns: int, reason: str) -> RunResult:
        return RunResult(
            session_id=session.id, status=status, verified_flag=session.verified_flag,
            outcome=session.outcome, turns=turns, reason=reason,
            flag_candidates=list(session.state.flag_candidates),
        )
