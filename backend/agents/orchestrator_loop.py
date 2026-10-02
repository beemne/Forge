import asyncio
import json
import re
import os
import sys
import time
import logging
import uuid
from backend.utils.time import utcnow
from typing import Dict, Any, Optional, Set
from sqlalchemy.orm import Session

from backend.database.session import SessionLocal
from backend.providers.router import model_router
from backend.tools.manager import tool_manager, ToolExecutionResult
from backend.privilege.gate import require_approval, SHARED_PENDING_APPROVALS
from backend.checkpoints.manager import checkpoint_manager
from backend.environment.detector import environment_detector
from backend.database.models import RunModel, ChallengeModel, TargetProfileModel, EvidenceModel, FindingModel
from backend.websocket.manager import ws_manager
from backend.api.runner import workflow_runner
from backend.reporting.generator import report_generator
from backend.knowledge.playbook_vault import playbook_vault
from backend.recon.turbo_recon import turbo_recon
from backend.agents.stream_condenser import stream_condenser
from backend.agents.strategic_planner import strategic_planner
from backend.agent_runtime.verifier import (
    AnswerResolver, AnswerCandidate, AnswerSource, VerifierAgent, FLAG_REGEX,
    FALSE_FLAG_PATTERNS,
)

logger = logging.getLogger("forge.orchestrator")

# Common import-name to pip-package-name mappings
IMPORT_TO_PACKAGE_MAP = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "Crypto": "pycryptodome",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "flask_unsign": "flask-unsign",
    "pwn": "pwntools",
}

def _privilege_denied_result(command: str, decision: Optional[str]) -> ToolExecutionResult:
    """Build a normal FAILED result for a command the operator did NOT approve.

    The command is never executed; the reasoning loop receives a structured failure
    (rather than an exception) so it can pivot instead of crashing.
    """
    timed_out = decision is None
    return ToolExecutionResult(
        tool_name="raw_cmd",
        capability="custom_command",
        command=command,
        status="FAILED",
        stdout="",
        stderr=(
            f"[PRIVILEGE {'TIMEOUT' if timed_out else 'DENIED'}] Operator did not approve "
            f"this command ({'no response before the approval window closed' if timed_out else 'denied'}). "
            f"The command was NOT executed. Pivot to an approach that does not require "
            f"privilege elevation."
        ),
        exit_code=-1,
        duration_ms=0.0,
        execution_failure=True,
        failure_category="PRIVILEGE_DENIED",
    )


class AutonomousOrchestrator:
    """Central autonomous investigation ReAct 1-Command Cycle loop."""

    def __init__(self):
        # Tracks pending install requests: request_id -> asyncio.Event
        self._install_events: dict[str, asyncio.Event] = {}
        self._install_results: dict[str, bool] = {}
        # Tracks pending root/privilege requests: request_id -> asyncio.Event
        self._root_events: dict[str, asyncio.Event] = {}
        self._root_results: dict[str, dict] = {}
        self.answer_resolver = AnswerResolver()
        self.verifier_agent = VerifierAgent(resolver=self.answer_resolver, router=model_router)

        # ── Agent-runtime facade (Step 5) ──────────────────────────────────────
        # The new HERMES-inspired runtime owns canonical session/trajectory state.
        # These are lazily bound so importing the orchestrator never forces the
        # runtime stack to load; the legacy run_autonomous_loop below is preserved
        # verbatim for backward compatibility, and delegates only via run_with_runtime.
        self._runtime = None
        self._session_manager = None

    # ------------------------------------------------------------------ #
    # Agent-runtime facade (Step 5). The orchestrator is becoming a thin
    # compatibility surface over backend.agent_runtime. This entrypoint runs a
    # mission through the canonical runtime (session + trajectory owned by FORGE,
    # provider-agnostic) while the legacy run_autonomous_loop remains available.
    # ------------------------------------------------------------------ #

    def _ensure_runtime(self):
        if self._runtime is None:
            from backend.agent_runtime import AgentRuntime, RealToolExecutor, session_manager
            self._session_manager = session_manager
            self._runtime = AgentRuntime(tool_executor=RealToolExecutor())
        return self._runtime

    async def run_with_runtime(
        self,
        run_id: str,
        challenge_id: str,
        target: str,
        *,
        resume_session_id: Optional[str] = None,
        max_turns: int = 40,
        max_minutes: int = 0,
    ):
        """Run a mission via the new AgentRuntime, returning its RunResult.

        FORGE owns the session/trajectory; a crash is resumable via resume_session_id.
        Challenge metadata is read from the DB so the runtime prompt matches the swarm's.
        Non-fatal to the legacy path — callers may still use run_autonomous_loop.
        """
        runtime = self._ensure_runtime()
        sm = self._session_manager

        db = SessionLocal()
        try:
            ch = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
            meta = {
                "challenge_name": getattr(ch, "name", "") or challenge_id,
                "category": getattr(ch, "category", "") or "",
                "difficulty": getattr(ch, "difficulty", "") or "",
                "platform": getattr(ch, "platform_name", "") or "",
                "description": getattr(ch, "description", "") or "",
                "working_directory": getattr(ch, "working_directory", "") or "",
            }
        finally:
            db.close()

        if resume_session_id:
            session = sm.restore(resume_session_id) or sm.create(
                challenge_id=challenge_id, run_id=run_id, target_scope=target,
                agent_id="orchestrator", engine="orchestrator", **{
                    k: v for k, v in meta.items() if k != "working_directory"})
        else:
            session = sm.create(
                challenge_id=challenge_id, run_id=run_id, target_scope=target,
                agent_id="orchestrator", engine="orchestrator", **{
                    k: v for k, v in meta.items() if k != "working_directory"})

        return await runtime.run(
            session,
            max_turns=max_turns,
            max_seconds=(max_minutes * 60) if max_minutes else None,
            cwd=meta.get("working_directory") or None,
            cancel_check=lambda: workflow_runner.is_kill_switch_active(run_id),
        )

    def resolve_install_request(self, request_id: str, success: bool):
        """Called by the API route after user approves and pip install completes."""
        self._install_results[request_id] = success
        event = self._install_events.get(request_id)
        if event:
            event.set()

    def resolve_root_request(self, request_id: str, success: bool, tool_res: Optional[Any] = None, message: str = ""):
        """Called by the API route when operator approves or denies root execution."""
        self._root_results[request_id] = {
            "success": success,
            "tool_res": tool_res,
            "message": message
        }
        event = self._root_events.get(request_id)
        if event:
            event.set()

    @staticmethod
    def _snapshot_dir(dir_path: str) -> dict:
        """Returns a mapping of relative file paths to their size in bytes."""
        snapshot = {}
        if not dir_path or not os.path.exists(dir_path):
            return snapshot
        try:
            for root, _, files in os.walk(dir_path):
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, dir_path)
                    try:
                        snapshot[rel_path] = os.path.getsize(full_path)
                    except Exception:
                        pass
        except Exception:
            pass
        return snapshot

    @staticmethod
    def _diff_dir_snapshots(before: dict, after: dict) -> list:
        """Computes a human-readable list of file additions, modifications, and deletions."""
        changes = []
        for path, size in after.items():
            if path not in before:
                changes.append(f"+ Created: {path} ({size} bytes)")
            elif before[path] != size:
                changes.append(f"~ Modified: {path} ({before[path]} -> {size} bytes)")
        for path in before:
            if path not in after:
                changes.append(f"- Deleted: {path}")
        return changes

    async def run_autonomous_loop(self, run_id: str, challenge_id: str, target: str):
        """Asynchronous background 1-command ping-pong loop processing CTF challenge target."""
        logger.info(f"Starting 1-command ReAct autonomous loop for run {run_id}, challenge {challenge_id}, target {target}")

        env_info = environment_detector.detect_environment()
        os_distro = env_info.get("distro") or env_info.get("os", "Linux (Parrot OS)")
        installed_tools = [name for name, meta in env_info.get("installed_tools", {}).items() if meta.get("installed")]
        tools_str = ", ".join(installed_tools) if installed_tools else "nmap, ffuf, curl, python3, gdb, strings, objdump"
        
        # Audit importable Python libraries for solver scripts
        installed_py_libs = [lib for lib, active in env_info.get("installed_python_libs", {}).items() if active]
        py_libs_str = ", ".join(installed_py_libs) if installed_py_libs else "requests, beautifulsoup4, flask-unsign, cryptography"
        
        for req_lib in ["requests", "bs4", "flask_unsign", "cryptography"]:
            if not env_info.get("installed_python_libs", {}).get(req_lib, True):
                logger.warning(f"[SOLVER LIB WARNING] Advertised library '{req_lib}' is NOT importable in host Python runtime ({sys.executable}).")

        history_summary = []
        normalized_history = []
        max_turns = 1000  # Restrictions removed; runs continue until flag capture or operator stop
        
        # Structured State Memory Object
        state_memory = {
            "discovered_endpoints": set(),
            "observed_cookies": set(),
            "headers_found": set(),
            "detected_technologies": set(),
            "repetition_warnings": 0,
            "last_tool_output": ""
        }
        attempted_playbook_ids: Set[str] = set()
        active_playbook: Optional[Any] = None

        # ── Pre-flight Mission Planning Phase ──
        db_init: Session = SessionLocal()
        mission_plan = None
        try:
            ch_obj = db_init.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
            if ch_obj:
                if not ch_obj.started_at:
                    ch_obj.started_at = utcnow()
                
                # Check existing mission plan or generate new pre-flight plan
                if ch_obj.mission_plan and isinstance(ch_obj.mission_plan, dict) and ch_obj.mission_plan.get("tasks"):
                    mission_plan = ch_obj.mission_plan
                else:
                    cached_recon = turbo_recon.get_cached_recon(challenge_id)
                    recon_summary = cached_recon.get("raw_summary", "") if cached_recon else ""
                    
                    mission_plan = await strategic_planner.generate_initial_plan(
                        challenge_id=challenge_id,
                        challenge_name=ch_obj.name,
                        category=ch_obj.category,
                        difficulty=ch_obj.difficulty,
                        target=target,
                        description=ch_obj.description,
                        turbo_recon_summary=recon_summary
                    )
                    ch_obj.mission_plan = mission_plan
                    db_init.commit()

                await ws_manager.broadcast({
                    "event": "PLAN_GENERATED",
                    "challenge_id": challenge_id,
                    "run_id": run_id,
                    "plan": mission_plan
                })
        except Exception as plan_err:
            logger.error(f"Pre-flight mission plan initialization error: {plan_err}")
        finally:
            db_init.close()

        for turn in range(1, max_turns + 1):
            if workflow_runner.is_cancelled(run_id):
                logger.warning(f"Run {run_id} cancelled by Kill Switch.")
                await ws_manager.broadcast({
                    "event": "RUN_CANCELLED",
                    "run_id": run_id,
                    "challenge_id": challenge_id
                })
                return

            db: Session = SessionLocal()
            try:
                run = db.query(RunModel).filter(RunModel.id == run_id).first()
                challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
                if not run or not challenge or run.status in ["PAUSED", "CANCELLED"]:
                    logger.info(f"Run {run_id} terminated or paused externally.")
                    return

                # Calculate dynamic task-based progress %
                dynamic_progress = strategic_planner.calculate_progress(
                    mission_plan,
                    flag_captured=(challenge.flag_status == "CAPTURED" or challenge.status == "COMPLETED")
                )
                challenge.progress = max(challenge.progress, dynamic_progress)

                # Record live operation uptime / duration
                if challenge.started_at:
                    duration_sec = int((utcnow() - challenge.started_at).total_seconds())
                    challenge.duration_seconds = max(challenge.duration_seconds or 0, duration_sec)

                run.current_phase = "recon" if turn <= 3 else ("web" if turn <= 10 else "exploitation")
                run.current_agent = "orchestrator"
                db.commit()

                await ws_manager.broadcast({
                    "event": "PROGRESS_UPDATED",
                    "challenge_id": challenge_id,
                    "run_id": run_id,
                    "phase": run.current_phase,
                    "progress": challenge.progress,
                    "duration_seconds": challenge.duration_seconds
                })

                # Format Structured State Memory for Prompt Injection
                memory_json = json.dumps({
                    "discovered_endpoints": list(state_memory["discovered_endpoints"])[:10],
                    "observed_cookies": list(state_memory["observed_cookies"])[:5],
                    "headers_found": list(state_memory["headers_found"])[:5],
                    "loop_warning": state_memory["repetition_warnings"] > 0
                }, indent=2)

                # Two-Stage Retrieval from Playbook Vault
                playbook_context = ""
                try:
                    if turn == 1:
                        # Stage A (Pre-recon coarse retrieval): coarse category methodology query
                        matching_pbs = playbook_vault.search_playbooks(
                            query=f"{challenge.category} standard methodology",
                            category=challenge.category,
                            excluded_ids=attempted_playbook_ids,
                            top_k=1
                        )
                    else:
                        # Stage B (Post-recon primary retrieval): build query from real observed evidence
                        obs_headers = " ".join(state_memory["headers_found"])
                        obs_tech = " ".join(state_memory["detected_technologies"])
                        obs_endpoints = " ".join(state_memory["discovered_endpoints"])
                        recon_artifacts = f"{obs_headers}\n{obs_tech}\n{obs_endpoints}\n{state_memory['last_tool_output'][:1000]}"

                        cand_tags = list(state_memory["detected_technologies"])
                        if challenge.category:
                            cand_tags.append(challenge.category.lower())

                        evidence_query = f"{challenge.category} {obs_tech} {obs_headers}"
                        matching_pbs = playbook_vault.search_playbooks(
                            query=evidence_query,
                            category=challenge.category,
                            recon_artifacts=recon_artifacts,
                            candidate_tags=cand_tags,
                            excluded_ids=attempted_playbook_ids,
                            top_k=2
                        )

                    if matching_pbs:
                        active_playbook = matching_pbs[0]
                        pb_snippets = []
                        for pb in matching_pbs:
                            pb_snippets.append(
                                f"### Playbook: {pb.id} ({pb.category})\n"
                                f"Tags: {', '.join(pb.tags)}\n"
                                f"Trigger Signatures: {', '.join(pb.trigger_signatures) if pb.trigger_signatures else 'None'}\n"
                                f"Template:\n{pb.exploit_template[:400]}"
                            )
                        playbook_context = "\n\n--- FEW-SHOT CTF EXPLOIT PLAYBOOKS ---\n" + "\n\n".join(pb_snippets)
                    else:
                        active_playbook = None
                except Exception as pb_err:
                    logger.debug(f"Playbook vault retrieval skip: {pb_err}")

                # Fetch pre-warmed turbo recon on initial turn
                turbo_recon_context = ""
                if turn == 1:
                    cached_recon = turbo_recon.get_cached_recon(challenge_id)
                    if cached_recon and cached_recon.get("raw_summary"):
                        turbo_recon_context = f"\n\n--- PRE-WARMED TURBO RECON (INSTANT) ---\n{cached_recon['raw_summary']}"

                # Construct transparent ReAct System Prompt with Educational CTF Lab Context
                system_instruction = (
                    f"You are FORGE Autonomous CTF Security Assistant operating in an authorized educational lab environment on {os_distro}.\n"
                    f"SYSTEM DIRECTIVE: Analyze the target scope rigorously and rely exclusively on verified command outputs. Zero mock/fake flags.\n"
                    f"Target Scope: {target}\n"
                    f"Challenge Name: {challenge.name} | Category: {challenge.category} | Difficulty: {challenge.difficulty} | Platform: {challenge.platform_name}\n"
                    f"Working Directory: {challenge.working_directory}\n"
                    f"Installed Analysis Tools & Libraries: {tools_str}, python3, {py_libs_str}\n\n"
                    f"ACTION SELECTION MODES:\n"
                    f"Mode A (CLI Command): Output a SINGLE executable bash/shell command line.\n"
                    f"Mode B (Python Solver Script): Output a complete Python script inside ```python ... ``` blocks. FORGE will save it to `solve.py` and run `{sys.executable} solve.py` automatically.\n"
                    f"Mode C (Structured Completion): Output JSON {{\"action\": \"complete\", \"flag\": \"<flag>\", \"proof\": \"<explanation>\"}} if solve is verified.\n\n"
                    f"STRICT RULES:\n"
                    f"1. Output ONLY the bash command line, the ```python ... ``` solver script block, or structured completion JSON.\n"
                    f"2. If you discover the flag (e.g. picoCTF{{...}}, FLAG{{...}}, HTB{{...}}), reply EXACTLY: FLAG: <captured_flag>\n"
                    f"3. If mission is complete, provide the verified flag or exact solve payload.\n"
                    f"4. Multi-target inputs are joined using '+' sign. Process targets accordingly."
                    f"{playbook_context}"
                )

                history_context = "\n".join(history_summary[-6:]) if history_summary else "No commands executed yet."
                
                # Check for Strategic Repetition Override
                repetition_warning_prompt = ""
                if state_memory["repetition_warnings"] > 0:
                    repetition_warning_prompt = (
                        "\n⚠️ STRATEGIC OVERRIDE ALERT: Your recent actions are repeating similar commands on the target without progress.\n"
                        "YOU MUST PIVOT METHODOLOGY IMMEDIATELY!\n"
                        "- Do NOT repeat curl requests with similar cookies or endpoints.\n"
                        "- Use Mode B to write a Python script using requests.Session(), flask-unsign, pwntools, or bs4.\n"
                        "- Inspect HTML comments, script tags, or unusual HTTP response headers.\n"
                    )

                prompt = (
                    f"--- STRUCTURED TARGET MEMORY STATE ---\n{memory_json}\n"
                    f"{turbo_recon_context}\n"
                    f"--- RECENT STEP HISTORY ---\n{history_context}\n"
                    f"{repetition_warning_prompt}\n"
                    f"--- CURRENT TURN #{turn} ---\n"
                    f"Analyze the target memory, turbo recon, and step history. Issue the NEXT single command OR Python solver script to execute on '{target}'."
                )

                # Query AI Router with Speed-Tiered Routing
                capability = "code_analysis" if state_memory["repetition_warnings"] > 0 else ("general_reasoning" if turn == 1 else "web_testing")
                speed_tier = "deep" if challenge.category in ["reverse", "pwn", "crypto"] and turn > 5 else "fast"
                llm_start_time = time.time()
                llm_response = await model_router.route_request(
                    prompt=prompt,
                    capability=capability,
                    system_instruction=system_instruction,
                    speed_tier=speed_tier
                )
                llm_duration_ms = int((time.time() - llm_start_time) * 1000)

                raw_ai_output = (llm_response.content or "").strip()
                model_used = getattr(llm_response, "model", getattr(llm_response, "provider_name", "model_router"))

                # Broadcast AI Prompt Transparency Event to Frontend
                await ws_manager.broadcast({
                    "event": "AI_PROMPT_TRANSPARENCY",
                    "challenge_id": challenge_id,
                    "run_id": run_id,
                    "turn": turn,
                    "system_instruction": system_instruction,
                    "prompt": prompt,
                    "raw_response": raw_ai_output,
                    "model": model_used
                })

                await ws_manager.broadcast({
                    "event": "AI_DECISION",
                    "challenge_id": challenge_id,
                    "run_id": run_id,
                    "agent": "ORCHESTRATOR",
                    "goal": f"Turn #{turn} Action Selection ({os_distro})",
                    "capability": capability,
                    "model": model_used,
                    "result": raw_ai_output[:300],
                    "confidence": 95
                })

                # Check if model emitted structured completion JSON: {"action": "complete", ...}
                try:
                    if "{" in raw_ai_output and "}" in raw_ai_output:
                        json_match = re.search(r"\{.*\}", raw_ai_output, re.DOTALL)
                        if json_match:
                            parsed_action = json.loads(json_match.group(0))
                            if isinstance(parsed_action, dict) and parsed_action.get("action") == "complete":
                                cand_flag = parsed_action.get("flag", "")
                                cand_proof = parsed_action.get("proof", "")
                                found_in_json = re.findall(r"(?:picoCTF|FORGE|CTF|HTB|FLAG|THM)\{[A-Za-z0-9_!\-@#\$%\^&\*\.]+\}", f"{cand_flag} {cand_proof}", re.IGNORECASE)
                                if found_in_json:
                                    flag_str = found_in_json[0]
                                    await self._record_flag_capture(db, challenge, run, challenge_id, run_id, flag_str, f"Turn #{turn} Agent Structured Solve (Proof: {cand_proof[:100]})", history_summary)
                                    break
                                elif cand_proof and len(cand_proof) > 20:
                                    # Verified proof of exploit completion
                                    challenge.status = "COMPLETED"
                                    challenge.progress = 100
                                    if run:
                                        run.status = "COMPLETED"
                                    db.commit()
                                    report_generator.generate_readme(db, challenge_id)
                                    await ws_manager.broadcast({
                                        "event": "RUN_COMPLETED",
                                        "challenge_id": challenge_id,
                                        "run_id": run_id,
                                        "status": "COMPLETED",
                                        "proof": cand_proof
                                    })
                                    break
                except Exception:
                    pass

                # Check if model returned Flag directly in response
                direct_flags = re.findall(r"(?:picoCTF|FORGE|CTF|HTB|FLAG|THM)\{[A-Za-z0-9_!\-@#\$%\^&\*\.]+\}", raw_ai_output, re.IGNORECASE)
                if direct_flags or raw_ai_output.startswith("FLAG:"):
                    flag_str = direct_flags[0] if direct_flags else raw_ai_output.replace("FLAG:", "").strip()
                    await self._record_flag_capture(db, challenge, run, challenge_id, run_id, flag_str, f"Turn #{turn} AI Reasoning", history_summary)
                    break

                if raw_ai_output.upper() == "DONE":
                    logger.info(f"AI declared mission completed without explicit flag on turn #{turn}.")
                    break

                # Extract Python script block OR single CLI command line
                cmd_line = ""
                is_python_script = False
                script_code = ""
                py_match = re.search(r"```python\s*(.*?)\s*```", raw_ai_output, re.DOTALL)
                if not py_match:
                    py_match = re.search(r"```(?:sh|bash)?\s*(import\s+.*|from\s+.*)\s*```", raw_ai_output, re.DOTALL)

                if py_match or raw_ai_output.startswith("import ") or raw_ai_output.startswith("from "):
                    is_python_script = True
                    script_code = py_match.group(1) if py_match else raw_ai_output
                    solve_file_path = os.path.join(challenge.working_directory, "solve.py")
                    try:
                        with open(solve_file_path, "w", encoding="utf-8") as sf:
                            sf.write(script_code)
                        cmd_line = f'"{sys.executable}" solve.py'
                        logger.info(f"Synthesized Python solver script `solve.py` for turn #{turn} (interpreter: {sys.executable})")
                    except Exception as sf_err:
                        logger.error(f"Failed to save solve.py: {sf_err}")
                        cmd_line = f'"{sys.executable}" -c ' + json.dumps(script_code)
                else:
                    cmd_line = self._clean_cli_command(raw_ai_output)

                # Sanitize nmap target if AI passes http:// or port
                if cmd_line.startswith("nmap"):
                    clean_host = target.replace("http://", "").replace("https://", "").split("/")[0].split(":")[0]
                    cmd_line = re.sub(r"https?://[^\s]+", clean_host, cmd_line)

                # Normalized Anti-Repetition Loop Guard
                if is_python_script and script_code:
                    import hashlib
                    norm_cmd = "python_script:" + hashlib.md5(script_code.strip().encode("utf-8")).hexdigest()
                else:
                    norm_cmd = self._normalize_command(cmd_line)

                if normalized_history and normalized_history.count(norm_cmd) >= 1:
                    state_memory["repetition_warnings"] += 1
                    logger.warning(f"Normalized loop detected for `{norm_cmd[:30]}` (Count={normalized_history.count(norm_cmd)}). Forcing pivot.")
                    
                    # Trigger Multi-Model Strategic Review
                    if mission_plan:
                        try:
                            mission_plan = await strategic_planner.review_and_adapt_plan(
                                challenge_name=challenge.name,
                                category=challenge.category,
                                target=target,
                                mission_plan=mission_plan,
                                stuck_reason=f"Repetition deadlock on command `{norm_cmd[:40]}`",
                                recent_history=history_summary,
                                primary_model=model_used
                            )
                            challenge.mission_plan = mission_plan
                            db.commit()
                            if mission_plan.get("strategic_reviews"):
                                await ws_manager.broadcast({
                                    "event": "STRATEGIC_REVIEW_TRIGGERED",
                                    "challenge_id": challenge_id,
                                    "run_id": run_id,
                                    "review": mission_plan["strategic_reviews"][-1],
                                    "plan": mission_plan
                                })
                                await ws_manager.broadcast({
                                    "event": "PLAN_UPDATED",
                                    "challenge_id": challenge_id,
                                    "run_id": run_id,
                                    "plan": mission_plan
                                })
                        except Exception as rev_err:
                            logger.error(f"Strategic review trigger error: {rev_err}")

                    if "login" in cmd_line:
                        cmd_line = f"curl -i -s -c cookies.txt {target.rstrip('/')}/register"
                    elif "nmap" in cmd_line:
                        clean_host = target.replace("http://", "").replace("https://", "").split("/")[0].split(":")[0]
                        cmd_line = f"nmap -sV -F {clean_host}"
                    else:
                        cmd_line = f"curl -i -L {target}"
                    norm_cmd = self._normalize_command(cmd_line)
                
                normalized_history.append(norm_cmd)

                if not cmd_line:
                    cmd_line = f"curl -s -L {target}" if target.startswith("http") else f"nmap -F {target}"

                # Snapshot working directory before command execution
                dir_before = self._snapshot_dir(challenge.working_directory)
                cmd_start_time = time.time()

                # ── Privilege gate: operator approval required before ANY execution ──
                # Shared with the swarm path (backend/privilege/gate.py). Timeout, deny,
                # a missing decision or a broadcast failure all resolve to approved=False,
                # in which case execute_raw_command is never reached at all.
                _approved, _decision, _sudo_pw = await require_approval(
                    cmd=cmd_line,
                    agent_id="orchestrator_loop",
                    pending_approvals=SHARED_PENDING_APPROVALS,
                    broadcast_fn=ws_manager.broadcast,
                    challenge_id=challenge_id,
                    run_id=run_id,
                )

                if not _approved:
                    logger.warning(f"[PRIVILEGE] ReAct loop command not approved ({_decision}): {cmd_line[:120]}")
                    tool_res = _privilege_denied_result(cmd_line, _decision)
                    cmd_duration_ms = int((time.time() - cmd_start_time) * 1000)
                    stdout_text = ""
                    stderr_text = tool_res.stderr[:1000]
                    log_output = stderr_text
                else:
                    # Execute Command inside working directory
                    tool_res = await tool_manager.execute_raw_command(
                        command=cmd_line,
                        cwd=challenge.working_directory,
                        timeout_seconds=120
                    )
                    cmd_duration_ms = int((time.time() - cmd_start_time) * 1000)

                    stdout_text = tool_res.stdout[:3000] if tool_res.stdout else ""
                    stderr_text = tool_res.stderr[:1000] if tool_res.stderr else ""
                    log_output = stdout_text or stderr_text or f"[Return Code {tool_res.exit_code}] Execution finished with no output."

                # ── ImportError Auto-Install Detection ──
                if is_python_script and tool_res.exit_code != 0:
                    combined_err = f"{stdout_text}\n{stderr_text}"
                    import_err_match = re.search(
                        r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]([^'\"]+)['\"]",
                        combined_err
                    )
                    if import_err_match:
                        missing_import = import_err_match.group(1).split('.')[0]
                        pip_package = IMPORT_TO_PACKAGE_MAP.get(missing_import, missing_import)
                        request_id = str(uuid.uuid4())
                        error_snippet = combined_err.strip()[-300:]

                        logger.info(f"Detected missing module '{missing_import}' (pip: {pip_package}). Requesting user install approval.")

                        # Send install request popup to frontend
                        await ws_manager.broadcast({
                            "event": "PACKAGE_INSTALL_REQUEST",
                            "request_id": request_id,
                            "challenge_id": challenge_id,
                            "challenge_name": challenge.name,
                            "package_name": pip_package,
                            "import_name": missing_import,
                            "error_snippet": error_snippet,
                            "timestamp": utcnow().strftime("%H:%M:%S")
                        })

                        # Wait for user decision (up to 120s)
                        install_event = asyncio.Event()
                        self._install_events[request_id] = install_event
                        try:
                            await asyncio.wait_for(install_event.wait(), timeout=120)
                            if self._install_results.get(request_id, False):
                                logger.info(f"Package '{pip_package}' installed. Retrying solver script.")
                                # Re-run the same solver script — gated exactly like the
                                # first attempt; a retry is a NEW execution, not a free pass.
                                _r_approved, _r_decision, _r_sudo_pw = await require_approval(
                                    cmd=cmd_line,
                                    agent_id="orchestrator_loop",
                                    pending_approvals=SHARED_PENDING_APPROVALS,
                                    broadcast_fn=ws_manager.broadcast,
                                    challenge_id=challenge_id,
                                    run_id=run_id,
                                )
                                cmd_retry_start = time.time()
                                if not _r_approved:
                                    logger.warning(f"[PRIVILEGE] Solver retry not approved ({_r_decision}): {cmd_line[:120]}")
                                    tool_res = _privilege_denied_result(cmd_line, _r_decision)
                                else:
                                    tool_res = await tool_manager.execute_raw_command(
                                        command=cmd_line,
                                        cwd=challenge.working_directory,
                                        timeout_seconds=120
                                    )
                                cmd_duration_ms += int((time.time() - cmd_retry_start) * 1000)
                                stdout_text = tool_res.stdout[:3000] if tool_res.stdout else ""
                                stderr_text = tool_res.stderr[:1000] if tool_res.stderr else ""
                                log_output = stdout_text or stderr_text or f"[Return Code {tool_res.exit_code}] Re-execution finished."
                            else:
                                logger.warning(f"Package install for '{pip_package}' failed or was rejected.")
                        except asyncio.TimeoutError:
                            logger.warning(f"Install approval for '{pip_package}' timed out after 120s. Continuing.")
                        finally:
                            self._install_events.pop(request_id, None)
                            self._install_results.pop(request_id, None)

                # ── Root / Privilege Elevation Detection ──
                requires_root_pattern = r"(?:Permission denied|Operation not permitted|You must be root|need root privileges|requires root|must be run as root|sudo:\s*a password is required|superuser privileges|EACCES)"
                combined_output = f"{stdout_text}\n{stderr_text}"
                needs_root_elevation = (
                    cmd_line.strip().startswith("sudo ") or
                    (tool_res.exit_code != 0 and re.search(requires_root_pattern, combined_output, re.I))
                )

                if needs_root_elevation and not is_python_script:
                    # Auto-elevate with sudo if challenge.requires_root is enabled
                    if getattr(challenge, "requires_root", False):
                        logger.info(f"Challenge has requires_root=True enabled. Auto-elevating command with sudo: {cmd_line}")
                        sudo_cmd = cmd_line if cmd_line.startswith("sudo ") else f"sudo {cmd_line}"
                        # Auto-elevation is still an agent-originated privileged execution —
                        # `requires_root` is a challenge property, NOT an operator approval,
                        # so it must pass the same gate as every other command.
                        _s_approved, _s_decision, _s_sudo_pw = await require_approval(
                            cmd=sudo_cmd,
                            agent_id="orchestrator_loop",
                            pending_approvals=SHARED_PENDING_APPROVALS,
                            broadcast_fn=ws_manager.broadcast,
                            challenge_id=challenge_id,
                            run_id=run_id,
                        )
                        cmd_elev_start = time.time()
                        if not _s_approved:
                            logger.warning(f"[PRIVILEGE] Auto-elevation not approved ({_s_decision}): {sudo_cmd[:120]}")
                            tool_res = _privilege_denied_result(sudo_cmd, _s_decision)
                        else:
                            tool_res = await tool_manager.execute_raw_command(
                                command=sudo_cmd,
                                cwd=challenge.working_directory,
                                timeout_seconds=120
                            )
                        cmd_duration_ms += int((time.time() - cmd_elev_start) * 1000)
                        stdout_text = tool_res.stdout[:3000] if tool_res.stdout else ""
                        stderr_text = tool_res.stderr[:1000] if tool_res.stderr else ""
                        log_output = stdout_text or stderr_text or f"[Return Code {tool_res.exit_code}] Auto-elevated root execution completed."
                    else:
                        root_request_id = str(uuid.uuid4())
                        root_reason = "This command requires elevated root / superuser privileges to execute."
                        error_snippet = (stderr_text or stdout_text).strip()[-300:]
                        logger.info(f"Command '{cmd_line}' requires root/sudo privileges. Requesting user approval [ID: {root_request_id}].")

                        await ws_manager.broadcast({
                            "event": "ROOT_PERMISSION_REQUEST",
                            "request_id": root_request_id,
                            "challenge_id": challenge_id,
                            "challenge_name": challenge.name,
                            "command": cmd_line,
                            "reason": root_reason,
                            "error_snippet": error_snippet,
                            "timestamp": utcnow().strftime("%H:%M:%S")
                        })

                        root_event = asyncio.Event()
                        self._root_events[root_request_id] = root_event
                        try:
                            await asyncio.wait_for(root_event.wait(), timeout=120)
                            res = self._root_results.get(root_request_id, {})
                            if res.get("success"):
                                logger.info(f"Root permission approved for '{cmd_line}'.")
                                if res.get("tool_res"):
                                    tool_res = res["tool_res"]
                                else:
                                    sudo_cmd = cmd_line if cmd_line.startswith("sudo ") else f"sudo {cmd_line}"
                                    # The ROOT_PERMISSION_REQUEST approval above covers the
                                    # operator's elevation consent, but this is still a
                                    # distinct sudo execution originating from the agent, so
                                    # it passes the same per-command gate (re-asking is
                                    # redundant, never unsafe — there is no default-allow).
                                    _e_approved, _e_decision, _e_sudo_pw = await require_approval(
                                        cmd=sudo_cmd,
                                        agent_id="orchestrator_loop",
                                        pending_approvals=SHARED_PENDING_APPROVALS,
                                        broadcast_fn=ws_manager.broadcast,
                                        challenge_id=challenge_id,
                                        run_id=run_id,
                                    )
                                    cmd_elev_start = time.time()
                                    if not _e_approved:
                                        logger.warning(f"[PRIVILEGE] Elevated execution not approved ({_e_decision}): {sudo_cmd[:120]}")
                                        tool_res = _privilege_denied_result(sudo_cmd, _e_decision)
                                    else:
                                        tool_res = await tool_manager.execute_raw_command(
                                            command=sudo_cmd,
                                            cwd=challenge.working_directory,
                                            timeout_seconds=120
                                        )
                                    cmd_duration_ms += int((time.time() - cmd_elev_start) * 1000)
                                stdout_text = tool_res.stdout[:3000] if tool_res.stdout else ""
                                stderr_text = tool_res.stderr[:1000] if tool_res.stderr else ""
                                log_output = stdout_text or stderr_text or f"[Return Code {tool_res.exit_code}] Elevated execution completed."
                            else:
                                logger.warning(f"Root permission was rejected/denied by operator for '{cmd_line}'.")
                                stdout_text = ""
                                stderr_text = f"[ACCESS DENIED] Root/sudo permission was rejected by the operator for: {cmd_line}. Please pivot to an unprivileged user-space alternative."
                                log_output = stderr_text
                        except asyncio.TimeoutError:
                            logger.warning(f"Root approval request timed out for '{cmd_line}'.")
                            stderr_text = f"[TIMEOUT] Root permission request timed out after 120s. Please pivot to an unprivileged user-space alternative."
                            log_output = stderr_text
                        finally:
                            self._root_events.pop(root_request_id, None)
                            self._root_results.pop(root_request_id, None)

                # Snapshot working directory after execution and compute filesystem deltas
                dir_after = self._snapshot_dir(challenge.working_directory)
                dir_changes = self._diff_dir_snapshots(dir_before, dir_after)

                # Update Structured State Memory from Command Output
                state_memory["last_tool_output"] = stdout_text or log_output
                self._update_state_memory(state_memory, stdout_text or log_output)

                # Gate 2: Intermediate Expected Outcome Signature Verification
                if active_playbook and active_playbook.expected_outcome_signatures:
                    raw_combined_output = (stdout_text or "") + " " + (stderr_text or "")
                    outcome_matched = False
                    for outcome_sig in active_playbook.expected_outcome_signatures:
                        if re.search(outcome_sig, raw_combined_output, re.IGNORECASE):
                            outcome_matched = True
                            break

                    if not outcome_matched and (tool_res.exit_code != 0 or "error" in raw_combined_output.lower()):
                        logger.info(f"[ANTI-HALLUCINATION GATE 2] Active playbook '{active_playbook.id}' failed expected outcome verification. Adding to failure exclusion set.")
                        attempted_playbook_ids.add(active_playbook.id)
                        active_playbook = None

                # Append Full AI Conversation, Timings & Telemetry to Dedicated Challenge Log File
                from backend.utils.challenge_paths import resolve_challenge_log_path
                ch_log_path = resolve_challenge_log_path(challenge_id)
                now_str = utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                try:
                    with open(ch_log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{now_str} UTC] TURN #{turn} (Model: {model_used} | LLM Latency: {llm_duration_ms}ms | Exec Duration: {cmd_duration_ms}ms)\n")
                        f.write(f"  --- SYSTEM INSTRUCTION SENT TO AI ---\n{system_instruction}\n\n")
                        f.write(f"  --- PROMPT / RECENT HISTORY SENT TO AI ---\n{prompt}\n\n")
                        f.write(f"  --- RAW AI RESPONSE RECEIVED ---\n{raw_ai_output}\n\n")
                        f.write(f"  --- EXECUTED COMMAND ---\n{cmd_line}\n\n")
                        f.write(f"  --- EXECUTION TELEMETRY ---\n")
                        f.write(f"    Exit Code: {tool_res.exit_code} | Duration: {cmd_duration_ms}ms | Working Dir: {challenge.working_directory}\n")
                        if dir_changes:
                            f.write(f"    Filesystem Changes:\n")
                            for chg in dir_changes:
                                f.write(f"      {chg}\n")
                        f.write(f"  --- STDOUT / STDERR OUTPUT ---\n{log_output}\n")
                        f.write(f"================================================================================\n\n")
                except Exception as log_err:
                    logger.warning(f"Failed to append to challenge log file: {log_err}")

                # Broadcast terminal log to frontend
                await ws_manager.broadcast({
                    "event": "LOG_OUTPUT",
                    "challenge_id": challenge_id,
                    "run_id": run_id,
                    "command": cmd_line,
                    "output": log_output,
                    "exit_code": tool_res.exit_code,
                    "timestamp": utcnow().strftime("%H:%M:%S")
                })

                # Distill output via Stream Condenser for history summary & prompt context
                condensed_output = stream_condenser.condense_output(tool_res.tool_name, stdout_text or log_output)
                history_summary.append(f"Turn #{turn} Command: `{cmd_line}`\nOutput snippet:\n{condensed_output[:600]}")

                # Update Mission Todo List Task Progress
                if mission_plan:
                    try:
                        mission_plan = strategic_planner.update_task_progress(
                            mission_plan=mission_plan,
                            turn=turn,
                            executed_command=cmd_line,
                            output_snippet=condensed_output,
                            is_success=(tool_res.exit_code == 0)
                        )
                        challenge.mission_plan = mission_plan
                        new_progress = strategic_planner.calculate_progress(mission_plan, flag_captured=(challenge.flag_status == "CAPTURED"))
                        challenge.progress = max(challenge.progress, new_progress)
                        db.commit()

                        await ws_manager.broadcast({
                            "event": "PLAN_UPDATED",
                            "challenge_id": challenge_id,
                            "run_id": run_id,
                            "plan": mission_plan
                        })
                        await ws_manager.broadcast({
                            "event": "PROGRESS_UPDATED",
                            "challenge_id": challenge_id,
                            "run_id": run_id,
                            "phase": run.current_phase,
                            "progress": challenge.progress,
                            "duration_seconds": challenge.duration_seconds
                        })
                    except Exception as plan_up_err:
                        logger.error(f"Task progress update error: {plan_up_err}")

                # Save Evidence
                ev = EvidenceModel(
                    challenge_id=challenge_id,
                    agent="orchestrator",
                    evidence_type="command_output",
                    source=tool_res.tool_name,
                    content=f"### COMMAND\n`{cmd_line}`\n\n### STDOUT\n{stdout_text}\n\n### STDERR\n{stderr_text}",
                    confidence=0.95
                )
                db.add(ev)
                db.commit()

                # Check for REAL Flag / Answer pattern in STDOUT/STDERR via AnswerResolver & VerifierAgent
                combined_output = f"{stdout_text}\n{stderr_text}"
                task_ctx = {
                    "description": getattr(challenge, "description", "") or "",
                    "challenge_name": getattr(challenge, "name", "") or "",
                    "category": getattr(challenge, "category", "") or "",
                    "flag_pattern": getattr(challenge, "flag_pattern", "") or "",
                    "target_scope": target,
                }
                
                # 1. Extract generic and specialized candidates
                candidates = self.answer_resolver.extract_candidates(
                    combined_output, task_context=task_ctx, source=AnswerSource.TOOL_OUTPUT
                )

                # 2. Add any direct FLAG_REGEX matches if not already present
                seen_cand_vals = {c.value for c in candidates}
                for m in FLAG_REGEX.finditer(combined_output):
                    fval = m.group(0).strip()
                    if fval not in seen_cand_vals and not FALSE_FLAG_PATTERNS.search(fval):
                        candidates.append(AnswerCandidate(
                            value=fval,
                            source=AnswerSource.TOOL_OUTPUT,
                            task_context=task_ctx,
                        ))
                        seen_cand_vals.add(fval)

                # 3. Evaluate each candidate through VerifierAgent
                for cand_obj in candidates:
                    verdict = await self.verifier_agent.verify(
                        cand_obj,
                        task_context=task_ctx,
                        command=cmd_line,
                        action_succeeded=(tool_res.exit_code == 0),
                        evidence={"stdout": stdout_text, "stderr": stderr_text},
                    )
                    if verdict.is_verified or verdict.is_resolved:
                        flag_str = verdict.candidate
                        await self._record_flag_capture(db, challenge, run, challenge_id, run_id, flag_str, f"Turn #{turn} Tool `{cmd_line}` Output", history_summary)
                        break
                if challenge.flag_status == "CAPTURED":
                    break


                # State Checkpoint
                checkpoint_manager.create_checkpoint(
                    db=db,
                    run_id=run_id,
                    current_phase=run.current_phase,
                    current_agent="orchestrator",
                    last_action=f"turn_{turn}_{tool_res.tool_name}",
                    state_snapshot={"target": target, "turn": turn}
                )

            except Exception as e:
                import traceback
                logger.error(f"Error during turn #{turn} of run {run_id}: {str(e)}\n{traceback.format_exc()}")
                try:
                    from backend.utils.challenge_paths import resolve_challenge_log_path
                    ch_log_path = resolve_challenge_log_path(challenge_id)
                    with open(ch_log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{utcnow().strftime('%H:%M:%S')}] TURN #{turn} ERROR: {str(e)}\n{traceback.format_exc()}\n")
                except Exception:
                    pass
            finally:
                db.close()

            await asyncio.sleep(2.0)

        # Check Final State: Dual-Gated Strict Completion Rule
        db: Session = SessionLocal()
        try:
            run = db.query(RunModel).filter(RunModel.id == run_id).first()
            challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
            if challenge:
                if challenge.started_at:
                    duration_sec = int((utcnow() - challenge.started_at).total_seconds())
                    challenge.duration_seconds = max(challenge.duration_seconds or 0, duration_sec)

                if challenge.flag_status == "CAPTURED" or challenge.status == "COMPLETED":
                    challenge.status = "COMPLETED"
                    challenge.progress = 100
                    challenge.completed_at = utcnow()
                    if run:
                        run.status = "COMPLETED"
                    
                    if challenge.mission_plan and isinstance(challenge.mission_plan, dict) and "tasks" in challenge.mission_plan:
                        for t in challenge.mission_plan["tasks"]:
                            t["status"] = "COMPLETED"
                        challenge.mission_plan["status"] = "COMPLETED"
                        await ws_manager.broadcast({
                            "event": "PLAN_UPDATED",
                            "challenge_id": challenge_id,
                            "run_id": run_id,
                            "plan": challenge.mission_plan
                        })

                    output_dir = getattr(challenge, 'working_directory', '') or 'reports'
                    report_generator.generate_readme(db, challenge_id, output_dir=output_dir)
                    await ws_manager.broadcast({
                        "event": "RUN_COMPLETED",
                        "challenge_id": challenge_id,
                        "run_id": run_id,
                        "status": "COMPLETED",
                        "flag": challenge.flag,
                        "duration_seconds": challenge.duration_seconds
                    })
                else:
                    # Turn cycle completed WITHOUT capturing a flag -> AWAITING_FLAG (NEVER COMPLETED)
                    challenge.status = "AWAITING_FLAG"
                    challenge.progress = min(challenge.progress, 80)
                    if run:
                        run.status = "AWAITING_FLAG"
                    logger.info(f"[Orchestrator] Run completed cycle without flag match. Setting status to AWAITING_FLAG.")
                    await ws_manager.broadcast({
                        "event": "RUN_AWAITING_FLAG",
                        "challenge_id": challenge_id,
                        "run_id": run_id,
                        "status": "AWAITING_FLAG",
                        "duration_seconds": challenge.duration_seconds,
                        "message": "Turn cycle finished without flag match. Ready for operator review or next attack vector."
                    })
            db.commit()
        finally:
            db.close()

    async def _record_flag_capture(self, db: Session, challenge, run, challenge_id: str, run_id: str, flag_str: str, source: str, history_summary: Optional[list] = None):
        """Helper to record flag capture, mark challenge COMPLETED, and trigger Self-Learning Flywheel."""
        challenge.flag_status = "CAPTURED"
        challenge.flag = flag_str
        challenge.status = "COMPLETED"
        challenge.progress = 100
        challenge.completed_at = utcnow()
        if challenge.started_at:
            duration_sec = int((utcnow() - challenge.started_at).total_seconds())
            challenge.duration_seconds = max(challenge.duration_seconds or 0, duration_sec)

        if run:
            run.status = "COMPLETED"

        # Complete all mission plan tasks
        if challenge.mission_plan and isinstance(challenge.mission_plan, dict) and "tasks" in challenge.mission_plan:
            for t in challenge.mission_plan["tasks"]:
                t["status"] = "COMPLETED"
            challenge.mission_plan["status"] = "COMPLETED"
            await ws_manager.broadcast({
                "event": "PLAN_UPDATED",
                "challenge_id": challenge_id,
                "run_id": run_id,
                "plan": challenge.mission_plan
            })

        db.commit()

        finding = FindingModel(
            challenge_id=challenge_id,
            agent="orchestrator",
            title=f"REAL Flag Extracted via {source}",
            description=f"Successfully extracted valid flag: {flag_str}",
            vulnerability_class="Flag Extraction",
            severity="CRITICAL",
            verified=True,
            confidence=1.0
        )
        db.add(finding)
        db.commit()

        # Generate markdown report
        output_dir = getattr(challenge, 'working_directory', '') or 'reports'
        report_generator.generate_readme(db, challenge_id, output_dir=output_dir)

        # Trigger Self-Learning Flywheel Synthesis in Background
        try:
            winning_cmds = list(history_summary or [])
            playbook_vault.synthesize_from_run(
                challenge_id=challenge_id,
                challenge_title=challenge.name,
                category=challenge.category,
                target_endpoint=getattr(challenge, "target", "") or source,
                winning_payload=source,
                winning_commands=winning_cmds,
                flag=flag_str
            )
        except Exception as flywheel_err:
            logger.debug(f"[Flywheel] Background synthesis skip: {flywheel_err}")

        await ws_manager.broadcast({
            "event": "FLAG_CAPTURED",
            "challenge_id": challenge_id,
            "flag": flag_str,
            "source": source,
            "duration_seconds": challenge.duration_seconds
        })

    async def execute_run_step(self, db: Session, run_id: str) -> Dict[str, Any]:
        # Check Kill Switch
        run = db.query(RunModel).filter(RunModel.id == run_id).first()
        current_agent_name = run.current_agent if run else "recon"

        if workflow_runner.is_cancelled(run_id):
            logger.warning(f"Run {run_id} halted by Kill Switch.")
            if run:
                run.status = "CANCELLED"
                db.commit()
            return {"status": "CANCELLED", "agent": current_agent_name, "reason": "Kill switch activated"}

        if not run or run.status in ["PAUSED", "COMPLETED", "CANCELLED"]:
            return {"status": run.status if run else "NOT_FOUND", "agent": current_agent_name}

        target = db.query(TargetProfileModel).filter(TargetProfileModel.challenge_id == run.challenge_id).first()
        target_addr = target.current_address if target else "127.0.0.1"

        capability = "network_scanning" if run.current_phase == "recon" else "web_testing"

        # Record step evidence
        step_evidence = EvidenceModel(
            challenge_id=run.challenge_id,
            agent=current_agent_name,
            evidence_type="step_execution",
            source=capability,
            content=f"Executed phase '{run.current_phase}' step for agent '{current_agent_name}' on target '{target_addr}'.",
            confidence=0.95,
        )
        db.add(step_evidence)

        # Create State Checkpoint
        checkpoint_manager.create_checkpoint(
            db=db,
            run_id=run_id,
            current_phase=run.current_phase,
            current_agent=current_agent_name,
            last_action="executed_step",
            state_snapshot={"target": target_addr}
        )

        if run.current_phase == "recon":
            run.current_phase = "web"
        db.commit()
        return {
            "status": run.status,
            "agent": current_agent_name,
            "capability": capability,
            "tool_status": "SUCCESS"
        }

    def _clean_cli_command(self, raw_output: str) -> str:
        """Extracts and sanitizes a single executable command line from potentially noisy LLM output."""
        text = raw_output.strip()
        
        # 1. Look for backticked command in markdown: `cmd` or ```bash\ncmd\n```
        fenced_match = re.search(r"```(?:bash|sh|cmd|powershell)?\s*\n?([^\n`]+)\n?```", text, re.IGNORECASE)
        if fenced_match:
            cmd = fenced_match.group(1).strip()
            if cmd:
                return cmd

        inline_backtick = re.search(r"`([^`\n]+)`", text)
        if inline_backtick:
            cmd = inline_backtick.group(1).strip()
            if any(cmd.startswith(prefix) for prefix in ["curl", "nmap", "ffuf", "gobuster", "python", "py ", "nikto", "sqlmap", "hydra", "tshark", "strings", "file ", "checksec", "cat ", "ls ", "nc ", "netcat", "ping", "dir"]):
                return cmd

        # 2. Iterate through lines and strip label headers
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines:
            cleaned = re.sub(r"^(?:Mode\s+[A-Z]\s*(?:\([^)]*\))?\s*[:\-]|Command\s*[:\-]|Action\s*[:\-]|Execute\s*[:\-]|Bash\s*[:\-]|Shell\s*[:\-])\s*", "", line, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"^`|`$", "", cleaned).strip()
            if not cleaned:
                continue
            if any(cleaned.startswith(tool) for tool in ["curl", "nmap", "ffuf", "gobuster", "python", "py ", "nikto", "sqlmap", "hydra", "tshark", "tcpdump", "strings", "file ", "checksec", "cat ", "ls ", "nc ", "netcat", "ping", "dir", "wpscan", "ghidra", "r2", "radare2", "volatility", "john", "hashcat", "sublist3r", "rustscan", "masscan"]):
                return cleaned
            if not cleaned.lower().startswith(("given ", "i will ", "we can ", "let's ", "this command", "to ", "first, ", "next, ")):
                return cleaned

        return lines[0] if lines else ""

    def _normalize_command(self, command: str) -> str:
        """Normalize command string by removing dynamic cookies, timestamps, and redundant whitespace."""
        cmd = command.strip()
        cmd = re.sub(r"session=[a-zA-Z0-9_\-\.%]+", "session=COOKIESUB", cmd)
        cmd = re.sub(r"Cookie:\s*['\"][^'\"]+['\"]", "Cookie: COOKIESUB", cmd, flags=re.IGNORECASE)
        cmd = re.sub(r"\?t=\d+", "?t=TIMESTAMP", cmd)
        cmd = re.sub(r"\s+", " ", cmd)
        return cmd.lower()

    def _update_state_memory(self, state_memory: Dict[str, Any], output_text: str):
        """Parse command stdout to update structured target state memory."""
        if not output_text:
            return

        # Extract Discovered URLs / Endpoints
        urls = re.findall(r"https?://[^\s\"'>]+", output_text)
        for u in urls:
            if len(u) < 120:
                state_memory["discovered_endpoints"].add(u)

        # Extract Cookies
        cookies = re.findall(r"Set-Cookie:\s*([^;\r\n]+)", output_text, re.IGNORECASE)
        for c in cookies:
            state_memory["observed_cookies"].add(c.strip())

        # Extract Server / Tech Headers
        headers = re.findall(r"(?:Server|X-Powered-By|X-Framework|X-AspNet-Version):\s*([^\r\n]+)", output_text, re.IGNORECASE)
        for h in headers:
            clean_h = h.strip()
            state_memory["headers_found"].add(clean_h)
            for part in re.split(r"[\s/\-_,;]+", clean_h):
                if len(part) > 2 and not part.isdigit():
                    state_memory["detected_technologies"].add(part.lower())

        # Extract Technology clues from HTML / output
        tech_indicators = {
            "jinja2": ["jinja", "werkzeug", "{{", "}}", "render_template_string"],
            "flask": ["flask", "werkzeug"],
            "express": ["express", "connect.sid"],
            "nodejs": ["node.js", "nodejs", "npm"],
            "php": ["php", "phpsessid", "x-powered-by: php"],
            "django": ["csrftoken", "django"],
            "jwt": ["bearer eyj", "jwt", "eyJ"],
            "graphql": ["graphql", "schema", "query {"],
            "mysql": ["mysql", "mariadb"],
            "sqlite": ["sqlite3", "sqlite"],
            "postgres": ["postgresql", "pg_"]
        }
        output_lower = output_text.lower()
        for tech_tag, markers in tech_indicators.items():
            if any(m.lower() in output_lower for m in markers):
                state_memory["detected_technologies"].add(tech_tag)

orchestrator_loop = AutonomousOrchestrator()

