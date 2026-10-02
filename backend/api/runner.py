import asyncio
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("forge.runner")

class WorkflowRunner:
    """Manages execution state, kill switch, and active autonomous run loops."""

    def __init__(self):
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        self.kill_switches: Dict[str, bool] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    def is_kill_switch_active(self, run_id: Optional[str] = None) -> bool:
        if run_id:
            return self.kill_switches.get(run_id, False) or self.kill_switches.get("__global__", False)
        return self.kill_switches.get("__global__", False)

    def is_cancelled(self, run_id: str) -> bool:
        return self.is_kill_switch_active(run_id)

    def start_run(
        self,
        run_id: str,
        challenge_id: str,
        target: str,
        engine_type: Optional[str] = None,
        model: Optional[str] = None,
        resume: bool = False
    ):
        self.active_runs[run_id] = {
            "run_id": run_id,
            "challenge_id": challenge_id,
            "target": target,
            "status": "RUNNING",
            "current_phase": "recon",
            "current_agent": engine_type or "auto"
        }
        self.kill_switches[run_id] = False

        # Engine selection. The legacy blackboard swarm remains the DEFAULT so all
        # existing behavior/tests are unchanged. Phase 4 adds the coordinated engine
        # (supervisor + specialist AgentRuntime agents + evidence bus), selectable via
        # engine_type in {coord, team, supervisor, coordinated, swarm_coord}.
        engine = (engine_type or "swarm").strip().lower()
        coordinated = engine in ("coord", "team", "supervisor", "coordinated", "swarm_coord")

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            from backend.database.session import SessionLocal
            from backend.database.models import ChallengeModel
            from backend.utils.workspace import resolve_safe_working_dir

            db = SessionLocal()
            ch = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
            category = ch.category if ch else "WEB"
            difficulty = ch.difficulty if ch else "EASY"
            challenge_name = ch.name if ch else ""
            platform = ch.platform_name if ch else ""
            description = ch.description if ch else ""
            # Per-run config (budget / flag pattern / uploaded artifacts / instance
            # timer) is stashed in mission_plan.run_config at creation time.
            run_config = (ch.mission_plan or {}).get("run_config", {}) if ch else {}
            # Never let a missing/"." working_directory resolve to the project root —
            # resolve to a safe path strictly inside the CTF workspace.
            workdir = resolve_safe_working_dir(
                ch.working_directory if ch else "",
                challenge_id,
                category,
                ch.name if ch else "",
            )
            os.makedirs(workdir, exist_ok=True)
            db.close()

            # Ensure mirrored challenge log path is registered BEFORE dispatching runner tasks
            try:
                from backend.utils.challenge_paths import register_challenge_log_path
                register_challenge_log_path(
                    challenge_id, platform, category, difficulty, challenge_name or challenge_id
                )
            except Exception as reg_err:
                logger.error(f"[WorkflowRunner] Log path registration failed for {challenge_id}: {reg_err}", exc_info=True)

            if coordinated:
                # Phase 4 coordinated swarm — builds ABOVE AgentRuntime/ExecutionService.
                from backend.swarm import SwarmCoordinator, SwarmLimits

                limits = SwarmLimits(
                    task_timeout_seconds=int(run_config.get("task_timeout", 0) or 0),
                    max_turns_per_task=int(run_config.get("max_turns_per_task", 12) or 12),
                    max_concurrent_agents=int(run_config.get("max_concurrent_agents", 3) or 3),
                )
                coordinator = SwarmCoordinator(
                    run_id=run_id, challenge_id=challenge_id, target=target,
                    category=category, difficulty=difficulty, challenge_name=challenge_name,
                    platform=platform, description=description,
                    flag_format=run_config.get("flag_pattern", ""),
                    workspace_root=workdir, limits=limits, enable_report=True,
                    kill_switch=lambda: self.is_kill_switch_active(run_id),
                )
                task = loop.create_task(coordinator.run(resume=resume))
                selected_engine = "swarm_coord"
            else:
                # Full replacement: every legacy run dispatches the flexible-agent swarm.
                from backend.agents.swarm_orchestrator import swarm_orchestrator

                task = loop.create_task(
                    swarm_orchestrator.run_swarm(
                        run_id=run_id,
                        challenge_id=challenge_id,
                        target_scope=target,
                        working_directory=workdir,
                        category=category,
                        difficulty=difficulty,
                        resume=resume,
                        challenge_name=challenge_name,
                        platform=platform,
                        description=description,
                        flag_pattern=run_config.get("flag_pattern", ""),
                        max_iterations=int(run_config.get("max_iterations", 0) or 0),
                        max_minutes=int(run_config.get("max_minutes", 0) or 0),
                        max_tokens=int(run_config.get("max_tokens", 0) or 0),
                        attached_file_paths=run_config.get("attached_file_paths", []),
                        instance_expiry_ts=run_config.get("instance_expiry_ts"),
                    )
                )
                selected_engine = "swarm"

            # Attach error callback so unhandled exceptions surface in logs
            def _on_task_done(t: asyncio.Task):
                if t.cancelled():
                    logger.info(f"Run task {run_id} was cancelled.")
                    return
                exc = t.exception()
                if exc:
                    import traceback as tb
                    logger.error(f"Run task {run_id} crashed with unhandled exception: {exc}\n{''.join(tb.format_exception(type(exc), exc, exc.__traceback__))}")
                    # Write to challenge log file for visibility
                    try:
                        from backend.agents.swarm_helpers import _append_to_challenge_log
                        _append_to_challenge_log(challenge_id, "runner", f"FATAL: {exc}")
                    except Exception:
                        pass

            task.add_done_callback(_on_task_done)
            self.tasks[run_id] = task
            logger.info(f"Started workflow run {run_id} using engine '{selected_engine}' for target {target}")

        except RuntimeError as e:
            import traceback as tb
            logger.error(f"Failed to create run task for run {run_id}: {e}\n{tb.format_exc()}")
        except Exception as e:
            import traceback as tb
            logger.error(f"Unexpected error starting run task for run {run_id}: {e}\n{tb.format_exc()}")

    def activate_kill_switch(self, run_id: Optional[str] = None):
        """Emergency Kill Switch - immediately halts autonomous operations."""
        if run_id:
            self.kill_switches[run_id] = True
            if run_id in self.active_runs:
                self.active_runs[run_id]["status"] = "CANCELLED"
            if run_id in self.tasks and not self.tasks[run_id].done():
                self.tasks[run_id].cancel()
            logger.warning(f"KILL SWITCH ACTIVATED FOR RUN {run_id}")
        else:
            # Universal kill switch
            self.kill_switches["__global__"] = True
            for rid in list(self.kill_switches.keys()):
                self.kill_switches[rid] = True
                if rid in self.active_runs:
                    self.active_runs[rid]["status"] = "CANCELLED"
                if rid in self.tasks and not self.tasks[rid].done():
                    self.tasks[rid].cancel()
            logger.warning("UNIVERSAL KILL SWITCH ACTIVATED - ALL RUNS HALTED.")

    def reset(self):
        """Reset runner state for clean test isolation."""
        self.active_runs.clear()
        self.kill_switches.clear()
        for task in list(self.tasks.values()):
            if not task.done():
                task.cancel()
        self.tasks.clear()

workflow_runner = WorkflowRunner()
