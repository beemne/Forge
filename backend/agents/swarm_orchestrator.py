"""FORGE Autonomous Swarm Intelligence Orchestrator.
Dispatches specialized parallel agent workers (Recon, Code Audit, Exploit Solver) on a shared blackboard.
Features:
- Dynamic claim-and-solve task pool
- Real-time shared target memory (endpoints, headers, decoded secrets)
- Deduplication filter
- Zero-retry Quota Circuit Breaker
- Verified flag discovery (tool output only, never LLM prose)
"""

import asyncio
import hashlib
import re
import os
import time
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from backend.database.session import SessionLocal
from backend.database.models import RunModel, ChallengeModel, ToolExecutionModel, CheckpointModel, TrajectoryEventModel
from backend.providers.router import model_router
from backend.tools.manager import tool_manager, LOCAL_EXEC_CATEGORIES
from backend.privilege.classify import classify_command_privilege
from backend.privilege.gate import require_approval, SHARED_PENDING_APPROVALS
# Part of this module's patchable surface: tests patch
# backend.agents.swarm_orchestrator.privilege_manager, so it must stay importable
# here even though gating now flows through classify_command_privilege/require_approval.
from backend.privilege.manager import privilege_manager  # noqa: F401
from backend.websocket.manager import ws_manager
from backend.engine.keep_awake import keep_awake_manager
from backend.reporting.generator import report_generator
from backend.knowledge.memory_retriever import memory_retriever
from backend.knowledge.experience_memory import experience_memory
from backend.knowledge.experience_extractor import experience_extractor
from backend.recon.turbo_recon import turbo_recon
from backend.recon.web_forms import describe_form, extract_forms, verify_template_probe
from backend.config import settings
from backend.environment.detector import environment_detector
from backend.agents.agent_prompt import AgentContext, build_agent_prompt, make_context_from_env
from backend.agents.artifact_acquisition import acquire_artifacts
from backend.agents import checkpoint_pipeline
from backend.agents.checkpoint_pipeline import (
    SuggestionDecision,
)
from backend.agent_runtime.verifier import (
    AnswerSource, FLAG_REGEX, FALSE_FLAG_PATTERNS,
)
from backend.agents.response_profiler import (
    generate_post_exploitation_probes,
)


logger = logging.getLogger("forge.swarm")

# ── Relocated modules (re-exported here so existing imports keep working) ────
from backend.agents.swarm_helpers import (
    _HEADER_HINT_RE,
    _append_to_challenge_log,
    _decode_artifacts,
    _effective_elapsed_minutes,
    _force_pivot_if_needed,
    _normalize_command_shape,
    _normalize_failure_signature,
    _normalize_recon_target,
    _update_strategy_state,
)
from backend.agents.swarm_state import SwarmBlackboard


def _capability_gap_retry_context(
    board: SwarmBlackboard, bin_name: str, priv_level: str
) -> Optional[Dict[str, Any]]:
    """Describe ``bin_name`` as a retry of an already-recorded capability gap, if it is one.

    When the operator denies a command, the gate records a capability gap and the recovery
    policy later re-issues the same command (``supervisor.decide_recovery`` returns "retry"
    for ``capability_gap``).  Without this, that retry reaches the operator as a second,
    context-free card for a request they already refused, indistinguishable from a new one.

    Returns the context dict for the newest matching gap, or None when this command is not
    a retry.  Presentation metadata only — it is never consulted when deciding anything.
    """
    if not bin_name:
        return None
    for gap in reversed(board.capability_gaps):
        if gap.get("capability") == bin_name:
            return {
                "request_kind": "capability_gap_retry",
                "capability": bin_name,
                "target": gap.get("target", ""),
                "denied_reason": gap.get("reason", ""),
                "previous_decision": gap.get("decision"),
                "privilege_level": gap.get("privilege_level", priv_level),
            }
    return None


class SwarmOrchestrator:
    """Manages the lifecycle, workers, and blackboard of an active parallel swarm."""

    def __init__(self):
        self.active_swarms: Dict[str, SwarmBlackboard] = {}

    async def run_swarm(
        self,
        run_id: str,
        challenge_id: str,
        target_scope: str,
        working_directory: str,
        category: str = "WEB",
        difficulty: str = "EASY",
        resume: bool = False,
        challenge_name: str = "",
        platform: str = "",
        description: str = "",
        flag_pattern: str = "",
        max_iterations: int = 0,
        max_minutes: int = 0,
        max_tokens: int = 0,
        attached_file_paths: Optional[List[str]] = None,
        instance_expiry_ts: Optional[float] = None,
    ):
        """Dispatch N general-purpose full-context agents on the target (flexible-agent
        engine; no fixed recon/crypto/exploit roles or hardcoded task checklist)."""
        # Register the mirrored challenge-log path FIRST before any log writes, so
        # every append (including swarm start) lands under logs/<Platform>/<Category>/
        # <Difficulty>/<Name>/ instead of a flat file.
        try:
            from backend.utils.challenge_paths import register_challenge_log_path
            register_challenge_log_path(challenge_id, platform, category, difficulty,
                                        challenge_name or challenge_id)
        except Exception as exc:
            logger.error(f"[SwarmOrchestrator] Failed to register challenge log path for challenge '{challenge_id}': {exc}", exc_info=True)

        logger.info(f"[SwarmOrchestrator] 🚀 Starting flexible-agent swarm for Challenge '{challenge_id}' on '{target_scope}' (resume={resume})")
        _append_to_challenge_log(challenge_id, "orchestrator", f"Swarm {'resuming' if resume else 'starting'} | target={target_scope} | category={category} | difficulty={difficulty}")

        # Engage OS Keep-Awake lock
        keep_awake_manager.acquire(reason=f"Swarm Challenge {challenge_id}")

        board = SwarmBlackboard(challenge_id, run_id, target_scope)
        self.active_swarms[run_id] = board

        # Challenge context → baked into every agent's full-context prompt.
        board.challenge_name = challenge_name or challenge_id
        board.platform = platform
        board.category = category
        board.difficulty = difficulty
        board.description = description
        if flag_pattern:
            board.flag_pattern = flag_pattern
        if max_iterations:
            board.max_iterations = max_iterations
        if max_minutes:
            board.max_minutes = max_minutes
        if max_tokens:
            board.max_tokens = max_tokens
        board.attached_file_paths = list(attached_file_paths or [])
        board.instance_expiry_ts = instance_expiry_ts

        db = SessionLocal()
        try:
            # Update Run status in DB
            run_obj = db.query(RunModel).filter(RunModel.id == run_id).first()
            if run_obj:
                run_obj.status = "RUNNING"
                run_obj.started_at = datetime.now(timezone.utc)
                db.commit()

            # Pre-warmed Turbo Recon (Initial seed — non-fatal)
            try:
                turbo_data = await turbo_recon.start_turbo_recon(challenge_id, target_scope, category.lower(), working_directory=working_directory)
            except Exception as recon_err:
                logger.warning(f"[SwarmOrchestrator] Turbo recon seed failed (non-fatal): {recon_err}")
                _append_to_challenge_log(challenge_id, "orchestrator", f"Turbo recon skipped: {recon_err}")
                turbo_data = turbo_recon.get_cached_recon(challenge_id)
            if turbo_data:
                for ep in turbo_data.get("endpoints", []):
                    board.discovered_endpoints.add(ep)
                for h_k, h_v in turbo_data.get("headers", {}).items():
                    board.extracted_headers[h_k] = str(h_v)
                _append_to_challenge_log(challenge_id, "orchestrator", f"Turbo recon seeded: {len(board.discovered_endpoints)} endpoints, {len(board.extracted_headers)} headers")

            # Resume: rehydrate prior blackboard state + task pool so the swarm
            # continues aware of earlier work instead of repeating it. Non-fatal —
            # a rehydrate failure just falls through to a fresh start.
            if resume:
                try:
                    ch_row = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
                    mp = (ch_row.mission_plan if ch_row else None) or {}
                    snapshot = mp.get("blackboard_state")
                    prior_tasks = mp.get("tasks")
                    rows = (db.query(ToolExecutionModel.command)
                            .join(RunModel, ToolExecutionModel.run_id == RunModel.id)
                            .filter(RunModel.challenge_id == challenge_id).all())
                    prior_cmds = [r[0] for r in rows if r and r[0]]
                    counts = board.load_snapshot(snapshot, prior_cmds, prior_tasks)
                    # Phase 7 (STEP 2/5) — if the target changed since the snapshot, keep
                    # the current one authoritative and drop stale host-bound endpoints so
                    # a respawned/edited instance address can't silently drive execution.
                    recon = board.reconcile_target((snapshot or {}).get("target_scope", ""))
                    if recon.get("changed"):
                        _append_to_challenge_log(
                            challenge_id, "orchestrator",
                            f"[TARGET CHANGED] '{recon['previous']}' -> '{recon['authoritative']}' — "
                            f"dropped {recon['dropped_endpoints']} stale endpoint(s); current target "
                            f"is authoritative")
                        await ws_manager.broadcast({
                            "event": "TARGET_CHANGED", "challenge_id": challenge_id, "run_id": run_id,
                            "old_target": recon["previous"], "new_target": recon["authoritative"],
                            "dropped_endpoints": recon["dropped_endpoints"],
                            "stale_hosts": recon.get("stale_hosts", []),
                        })
                    _append_to_challenge_log(
                        challenge_id, "orchestrator",
                        f"Resuming from prior progress: {counts['endpoints']} endpoints, "
                        f"{counts['commands']} prior commands, {counts['transcript_lines']} restored "
                        f"transcript lines (agents resume their reasoning thread, not just counters)"
                    )
                    logger.info(f"[SwarmOrchestrator] Resuming challenge '{challenge_id}' with rehydrated state: {counts}")
                except Exception as rehydrate_err:
                    logger.warning(f"[SwarmOrchestrator] Resume rehydrate failed (starting fresh): {rehydrate_err}")

            # ── Artifact acquisition (Part 2) — deterministic, before any agent ──
            # Binary-safe: downloads/uploads never pass through tool_manager's
            # text-decoding capture. If binary, the prompt flips to BINARY MODE.
            try:
                board.env_info = environment_detector.detect_environment()
            except Exception as env_err:
                logger.warning(f"[SwarmOrchestrator] Env detect failed (non-fatal): {env_err}")
                board.env_info = {}
            try:
                targets = [t.strip() for t in target_scope.split("+") if t.strip()]
                manifest = await acquire_artifacts(targets, working_directory, board.attached_file_paths)
                for note in manifest.notes:
                    _append_to_challenge_log(challenge_id, "orchestrator", f"[artifact] {note}")
                for p in manifest.saved_paths:
                    if p not in board.attached_file_paths:
                        board.attached_file_paths.append(p)
                if manifest.primary is not None:
                    board.artifact_classification = manifest.primary
                    _append_to_challenge_log(
                        challenge_id, "orchestrator",
                        f"BINARY ARTIFACT MODE engaged: {manifest.primary.artifact_type} "
                        f"@ {manifest.primary.safe_file_path or '(header-only)'}")
            except Exception as art_err:
                logger.warning(f"[SwarmOrchestrator] Artifact acquisition failed (non-fatal): {art_err}")

            # ── Encoded-artifact pre-scan — deterministic, before any agent ──────
            # If an attached artifact is itself an ENCODED file (e.g. a text file of
            # ASCII 0/1 that reconstructs to a JPEG), rebuild + preserve + escalate it
            # now so it enters the pipeline as a first-class analyzable input.
            try:
                await self._scan_attached_for_encoded(board, working_directory)
            except Exception as recon_err:
                logger.warning(f"[SwarmOrchestrator] Encoded-artifact pre-scan failed (non-fatal): {recon_err}")

            # ── Memory retrieval phase (§6) — ONE shared retrieval before agents ──
            # OBSERVE → RETRIEVE MEMORY: use the seeded recon + challenge context to
            # pull a few relevant past experiences / reference playbooks into a compact,
            # reference-only context shared by every agent. Agents still reason and
            # decide what to run (§8, §22). Fast + non-fatal (§17).
            try:
                evidence_parts = [board.description or ""]
                evidence_parts.extend(sorted(board.discovered_endpoints)[:15])
                evidence_parts.extend(f"{k}: {v}" for k, v in list(board.extracted_headers.items())[:12])
                if board.artifact_classification is not None:
                    evidence_parts.append(f"binary artifact {board.artifact_classification.artifact_type}")
                evidence_text = "\n".join(p for p in evidence_parts if p)
                technologies = experience_extractor._detect_technologies(evidence_text)
                mem_context, memories = memory_retriever.retrieve_and_format(
                    evidence=evidence_text,
                    category=board.category,
                    technologies=technologies,
                    query=f"{board.category} {board.challenge_name}",
                    top_k=int(getattr(settings, "MEMORY_RETRIEVAL_TOP_K", 6) or 6),
                )
                board.memory_context = mem_context
                board.retrieved_memory_ids = [m.id for m in memories if m.kind == "experience" and m.id]
                if memories:
                    experience_memory.record_retrieval(board.retrieved_memory_ids, run_id, challenge_id)
                    _append_to_challenge_log(
                        challenge_id, "orchestrator",
                        f"[MEMORY] Retrieved {len(memories)} relevant memories "
                        f"({len(board.retrieved_memory_ids)} FORGE experiences) for shared agent context")
                    await ws_manager.broadcast({
                        "event": "MEMORY_RETRIEVED", "challenge_id": challenge_id, "run_id": run_id,
                        "count": len(memories), "experiences": len(board.retrieved_memory_ids),
                        "techniques": [m.technique for m in memories][:8],
                    })
                else:
                    _append_to_challenge_log(challenge_id, "orchestrator",
                                             "[MEMORY] No relevant prior experience found (cold start)")
            except Exception as mem_err:
                logger.warning(f"[SwarmOrchestrator] Memory retrieval failed (non-fatal): {mem_err}")

            # ── Spawn N general-purpose agents (config-driven; not roles) ──────────
            pool_size = self._resolve_pool_size(board.env_info)
            board.agent_ids = [f"agent_{i+1}" for i in range(pool_size)]
            # Capabilities rotate only for provider diversity/throughput across the
            # free-tier chain — every agent gets the SAME full-context prompt.
            cap_cycle = ["web_analysis", "code_analysis", "general_reasoning",
                         "reverse_engineering", "fast_reasoning"]
            agents = [
                self._agent_worker(aid, board, working_directory, cap_cycle[i % len(cap_cycle)])
                for i, aid in enumerate(board.agent_ids)
            ]

            # Run agents concurrently until flag capture, all-budget-exhaustion, or
            # cancellation. ensure_future (not create_task) wraps the gather Future so
            # it can sit in asyncio.wait() alongside the flag + checkpoint tasks.
            worker_task = asyncio.ensure_future(asyncio.gather(*agents, return_exceptions=True))
            flag_task = asyncio.create_task(board.flag_event.wait())
            checkpoint_task = asyncio.create_task(self._checkpoint_coordinator(board, working_directory))

            _append_to_challenge_log(challenge_id, "orchestrator", f"{pool_size} flexible-agent workers + checkpoint coordinator dispatched")

            # Wait for flag event, all agents finishing, or coordinator exit.
            done, pending = await asyncio.wait(
                [flag_task, worker_task, checkpoint_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Stop everything and release any coordinator hard-wait so it can exit.
            board.is_stopped = True
            board.checkpoint_response_event.set()
            for p in pending:
                p.cancel()

            # Check if workers raised exceptions
            if worker_task in done:
                try:
                    results = worker_task.result()
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"[SwarmOrchestrator] Worker {i} raised exception: {result}")
                            _append_to_challenge_log(challenge_id, f"worker_{i}", f"EXCEPTION: {result}")
                except Exception:
                    pass

            # Finalize status in Database
            run_obj = db.query(RunModel).filter(RunModel.id == run_id).first()
            ch_obj = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()

            if board.flag_captured:
                if run_obj:
                    run_obj.status = "COMPLETED"
                    run_obj.completed_at = datetime.now(timezone.utc)
                if ch_obj:
                    ch_obj.status = "SOLVED"
                    ch_obj.flag = board.flag_captured
                    ch_obj.progress = 100
                    ch_obj.completed_at = datetime.now(timezone.utc)
                db.commit()
                await board._persist_progress_if_due(force=True)
                _append_to_challenge_log(challenge_id, "orchestrator", f"🏁 SOLVED! Flag: {board.flag_captured}")
                logger.info(f"[SwarmOrchestrator] 🏁 Swarm SOLVED challenge '{challenge_id}'! Flag: {board.flag_captured}")
                await ws_manager.broadcast({
                    "event": "RUN_COMPLETED",
                    "challenge_id": challenge_id,
                    "run_id": run_id,
                    "flag": board.flag_captured
                })

                # LEARN → STORE: distill this solve into generalized experience (§3, §19).
                await self._learn_from_run(board, outcome="success")
            elif board.pause_requested:
                # Operator paused — persist a resume snapshot and finalize as PAUSED
                # (resumable), never FAILED. Progress and blackboard state are kept.
                if run_obj:
                    run_obj.status = "PAUSED"
                if ch_obj:
                    ch_obj.status = "PAUSED"
                db.commit()
                await board._persist_progress_if_due(force=True)
                _append_to_challenge_log(challenge_id, "orchestrator", "Swarm PAUSED by operator — state saved for resume")
                logger.info(f"[SwarmOrchestrator] Swarm PAUSED for '{challenge_id}' (resumable)")
                await ws_manager.broadcast({
                    "event": "RUN_PAUSED",
                    "challenge_id": challenge_id,
                    "run_id": run_id
                })
            else:
                # If a reconstructed artifact was never analyzed, say so explicitly rather
                # than reporting a clean "no flag" — the run stalled with pending work, and
                # the derived artifact remains on the board for the next resume to pick up
                # (requirement #11/#12/#14).
                if board.has_unanalyzed_derived() and not board.stall_reason:
                    pending = [os.path.basename(d.get("derived_path") or "")
                               for d in board.derived_artifacts
                               if d.get("derived_path") and not d.get("analyzed")]
                    board.stall_reason = (
                        f"DERIVED_ANALYSIS_PENDING — reconstructed artifact(s) not yet analyzed: "
                        f"{', '.join(pending[:5])}")
                stall_msg = board.stall_reason or "Swarm finished without a verified flag (task pool exhausted)"
                if run_obj:
                    run_obj.status = "FAILED"
                    run_obj.completed_at = datetime.now(timezone.utc)
                if ch_obj:
                    ch_obj.status = "FAILED"
                    ch_obj.completed_at = datetime.now(timezone.utc)
                db.commit()
                await board._persist_progress_if_due(force=True)
                candidates_summary = ", ".join([c["flag"] for c in board.flag_candidates]) if board.flag_candidates else "none"
                _append_to_challenge_log(challenge_id, "orchestrator", f"Swarm finished without verified flag. {stall_msg} Candidates: {candidates_summary}")
                logger.info(f"[SwarmOrchestrator] Swarm ended without flag for '{challenge_id}': {stall_msg}")
                await ws_manager.broadcast({
                    "event": "RUN_STALLED",
                    "challenge_id": challenge_id,
                    "run_id": run_id,
                    "reason": stall_msg,
                    "candidates": [c["flag"] for c in board.flag_candidates]
                })

                # LEARN → STORE: a failed/stalled run is still experience (Part 8). Record
                # what was attempted and why it stalled so future missions avoid the same
                # dead ends, and down-weight any retrieved memory that did not help.
                await self._learn_from_run(board, outcome="failed")

            # Generate structured report
            try:
                report_generator.generate_final_report(challenge_id=challenge_id, run_id=run_id, db=db)
            except Exception as e:
                logger.warning(f"[SwarmOrchestrator] Report generation skip: {e}")

        except Exception as e:
            logger.error(f"[SwarmOrchestrator] Swarm execution exception: {e}\n{traceback.format_exc()}")
            _append_to_challenge_log(challenge_id, "orchestrator", f"FATAL EXCEPTION: {e}\n{traceback.format_exc()}")
            # Update run status to ERROR so it's visible in the UI
            try:
                run_obj = db.query(RunModel).filter(RunModel.id == run_id).first()
                if run_obj:
                    run_obj.status = "ERROR"
                    run_obj.completed_at = datetime.now(timezone.utc)
                    db.commit()
            except Exception:
                pass
        finally:
            db.close()
            self.active_swarms.pop(run_id, None)
            # Release OS Keep-Awake lock
            keep_awake_manager.release(reason=f"Swarm Challenge {challenge_id} Ended")
            _append_to_challenge_log(challenge_id, "orchestrator", "Swarm shutdown complete")

    async def request_pause(self, challenge_id: str) -> bool:
        """Gracefully pause every active swarm for a challenge.

        Signals workers to stop after their current step (no hard cancel, so no
        orphaned tasks) and force-persists a resume snapshot. run_swarm then
        finalizes the run as PAUSED. Returns True if an active swarm was found.
        """
        paused_any = False
        for board in list(self.active_swarms.values()):
            if board.challenge_id == challenge_id and not board.is_stopped:
                board.pause_requested = True
                board.is_stopped = True
                # Release a coordinator that is hard-waiting at a checkpoint so it
                # can observe is_stopped and exit cleanly instead of hanging.
                board.checkpoint_response_event.set()
                try:
                    await board._persist_progress_if_due(force=True)
                except Exception:
                    pass
                paused_any = True
                _append_to_challenge_log(challenge_id, "orchestrator", "Pause requested by operator — workers finishing current step")
                logger.info(f"[SwarmOrchestrator] Pause requested for challenge '{challenge_id}'")
        return paused_any

    def _resolve_pool_size(self, env_info: Dict[str, Any]) -> int:
        """How many general-purpose agents to spawn. Config AGENT_POOL_SIZE (int),
        or the string 'auto' -> min(cpu_cores-1, 4). Clamped to [1, 6]."""
        raw = getattr(settings, "AGENT_POOL_SIZE", 3)
        if isinstance(raw, str) and raw.strip().lower() == "auto":
            cores = int(env_info.get("cpu_cores") or 2)
            size = min(max(cores - 1, 1), 4)
        else:
            try:
                size = int(raw)
            except (TypeError, ValueError):
                size = 3
        return max(1, min(size, 6))

    def _build_agent_context(self, board: "SwarmBlackboard", workdir: str, agent_id: str,
                             *, memory_override: Optional[str] = None,
                             history_override: Optional[str] = None) -> AgentContext:
        """Full challenge context for one agent this turn (identical template for all;
        only history_context + injected_directive differ per agent).

        memory_override / history_override let the caller substitute compacted versions
        when the assembled prompt would exceed the model's context window (Workstream D).
        """
        pivot = board.pivot_directive if board.pivot_directive != "__pending__" else ""
        if memory_override is not None:
            memory_context = memory_override
        else:
            # Fold the shared cross-mission dead-end recall into memory_context (A4) so the
            # default engine avoids rediscovering approaches that failed in prior missions.
            memory_context = board.memory_context or ""
            try:
                from backend.knowledge.failed_approaches import recall_block
                xmission = recall_block(board.category)
                if xmission:
                    memory_context = (memory_context + "\n\n" + xmission).strip() if memory_context else xmission
            except Exception:
                pass
        history_context = history_override if history_override is not None else board.build_history_context(agent_id)
        return make_context_from_env(
            env_info=board.env_info or {},
            challenge_name=board.challenge_name,
            platform=board.platform,
            category=board.category,
            difficulty=board.difficulty,
            description=board.description,
            target_url=board.target_scope,
            working_directory=workdir,
            max_iterations=board.max_iterations,
            max_minutes=board.max_minutes,
            flag_pattern=board.flag_pattern,
            attached_file_paths=list(board.attached_file_paths),
            artifact_classification=board.artifact_classification,
            history_context=history_context,
            injected_directive=board.agent_directives.get(agent_id, ""),
            memory_context=memory_context,
            exhausted_strategies=list(board.exhausted_strategies),
            pivot_directive=pivot,
        )

    def _build_budgeted_prompt(self, board: "SwarmBlackboard", workdir: str, agent_id: str,
                               model_name: str):
        """Assemble (system_instruction, user_prompt) and, when *model_name* is known and
        the prompt would exceed ~80% of that model's context window, compact EXTRACTIVELY
        (Workstream D): first drop supplementary memory recall, then trim the OLDEST tail
        of history (recent step transcripts) while keeping the protected head — discovered
        state, decoded secrets, flag candidates, exhausted strategies, pivot directive,
        and pending derived-artifact analysis all sit at the top of build_history_context.
        Returns (system_instruction, user_prompt, compacted: bool).
        """
        from backend.agent_runtime.context_budget import estimate_tokens, budget_tokens, model_window

        ctx = self._build_agent_context(board, workdir, agent_id)
        system_instruction, user_prompt = build_agent_prompt(ctx)
        if not model_name:
            return system_instruction, user_prompt, False

        budget = budget_tokens(model_name)
        if estimate_tokens(system_instruction) + estimate_tokens(user_prompt) <= budget:
            return system_instruction, user_prompt, False

        # 1) Drop supplementary recall (least valuable).
        ctx = self._build_agent_context(board, workdir, agent_id, memory_override="")
        system_instruction, user_prompt = build_agent_prompt(ctx)
        if estimate_tokens(system_instruction) + estimate_tokens(user_prompt) <= budget:
            self._note_compaction(board, agent_id, model_name, dropped_memory=True, trimmed_history=False)
            return system_instruction, user_prompt, True

        # 2) Trim the history tail (oldest step transcripts) to fit, keeping the protected head.
        full_history = board.build_history_context(agent_id)
        overhead = (estimate_tokens(system_instruction) + estimate_tokens(user_prompt)
                    - estimate_tokens(full_history))
        hist_token_budget = max(500, budget - overhead)
        trimmed = full_history[: hist_token_budget * 4]
        if len(trimmed) < len(full_history):
            trimmed += "\n…[older step history trimmed to fit the model context window]"
        ctx = self._build_agent_context(board, workdir, agent_id,
                                        memory_override="", history_override=trimmed)
        system_instruction, user_prompt = build_agent_prompt(ctx)
        self._note_compaction(board, agent_id, model_name, dropped_memory=True, trimmed_history=True)
        return system_instruction, user_prompt, True

    def _note_compaction(self, board, agent_id, model_name, *, dropped_memory, trimmed_history):
        """Observability for a default-engine compaction (D4): agent step + WS event."""
        from backend.agent_runtime.context_budget import model_window
        note = (f"[CONTEXT COMPACTED] model={model_name} window={model_window(model_name)} "
                f"dropped_memory={dropped_memory} trimmed_history={trimmed_history} "
                f"(protected state/flags/evidence preserved)")
        try:
            board.record_agent_step(agent_id, note=note)
        except Exception:
            pass
        try:
            import asyncio
            from backend.websocket.manager import ws_manager
            asyncio.get_running_loop().create_task(ws_manager.broadcast({
                "type": "CONTEXT_COMPACTED",
                "data": {"challenge_id": board.challenge_id, "agent_id": agent_id,
                         "model": model_name, "dropped_memory": dropped_memory,
                         "trimmed_history": trimmed_history},
            }))
        except Exception:
            pass

    async def _agent_worker(self, agent_id: str, board: "SwarmBlackboard", workdir: str, capability: str):
        """One general-purpose, full-context agent running a budget-bounded ReAct
        loop. No fixed role: it reasons about THIS challenge from the shared
        blackboard context and decides its own next command / solver / flag."""
        logger.info(f"[Agent] {agent_id} started (capability route: {capability}).")
        _append_to_challenge_log(board.challenge_id, agent_id, f"Agent started (provider route: {capability})")
        board.agent_started_ts[agent_id] = time.time()
        board.agent_iterations[agent_id] = 0
        board.agent_paused_seconds[agent_id] = 0.0
        board.agent_local_fail_streak[agent_id] = 0
        await board.update_worker_state(agent_id, status="RUNNING", current_capability=capability,
                                        selected_model="auto (capability chain)", current_task="Booting agent")
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5
        consecutive_duplicates = 0
        MAX_CONSECUTIVE_DUPLICATES = 3

        while not board.flag_captured and not board.is_stopped:
            # Hard-pause at a checkpoint: idle until the operator response resumes us.
            # Time spent idle here is accumulated and subtracted from the work budget so a
            # multi-minute operator pause never counts against the M working-minute clock.
            if board.checkpoint_pause:
                pause_started = time.time()
                await board.update_worker_state(agent_id, status="IDLE", current_task="Paused at operator checkpoint")
                while board.checkpoint_pause and not board.is_stopped and not board.flag_captured:
                    await asyncio.sleep(1.0)
                board.agent_paused_seconds[agent_id] = (
                    board.agent_paused_seconds.get(agent_id, 0.0) + (time.time() - pause_started)
                )
                continue

            # Instance-expiry hard wind-down — the CTF platform kills the target on its own
            # timer (as short as ~15 min). Stop and report best findings a buffer before that
            # deadline rather than being cut off mid-command.
            if board.instance_expiry_ts:
                buffer = int(getattr(settings, "INSTANCE_WINDDOWN_BUFFER_SECONDS", 30))
                if time.time() >= board.instance_expiry_ts - buffer:
                    reason = "WINDING_DOWN: platform instance expiry approaching — reporting best findings"
                    board.record_agent_step(agent_id, note=reason)
                    _append_to_challenge_log(board.challenge_id, agent_id, reason)
                    await board.update_worker_state(agent_id, status="DONE", current_task=reason)
                    if not board.stall_reason:
                        board.stall_reason = "Platform instance expiry reached"
                    break

            # Telemetry tracking for iterations and working minutes (restrictions removed)
            elapsed_min = _effective_elapsed_minutes(
                board.agent_started_ts[agent_id], time.time(),
                board.agent_paused_seconds.get(agent_id, 0.0),
            )
            iters = board.agent_iterations.get(agent_id, 0)

            try:
                # Budget/compact against the model the agent last used (Workstream D):
                # empty on the first turn (no compaction), then tracks the model actually
                # in use so a long run never overflows that model's context window.
                last_model = board.agent_last_model.get(agent_id, "")
                system_instruction, user_prompt, _compacted = self._build_budgeted_prompt(
                    board, workdir, agent_id, last_model)
                await board.update_worker_state(agent_id, status="ANALYZING",
                                                current_task=f"Deciding next action (iter {iters + 1})")

                resp = await model_router.route_request(
                    prompt=user_prompt,
                    system_instruction=system_instruction,
                    capability=capability,
                )

                if resp.is_refusal:
                    consecutive_errors += 1
                    _append_to_challenge_log(board.challenge_id, agent_id, f"Provider exhausted/refusal: {resp.refusal_reason[:100]}")
                    await board.update_worker_state(agent_id, status="IDLE", failures=consecutive_errors,
                                                    current_task=f"Provider error: {resp.refusal_reason[:80]}")
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        await asyncio.sleep(30)
                        consecutive_errors = 0
                    else:
                        await asyncio.sleep(5)
                    continue  # provider errors do not consume the iteration budget

                content = resp.content or ""
                model_name = getattr(resp, "model_name", capability)
                # Remember the model this agent is actually using so next turn's prompt is
                # budgeted/compacted against the right context window (Workstream D).
                board.agent_last_model[agent_id] = model_name or ""

                # Per-run token budget (A5): accumulate REAL usage from the response and wind
                # the run down cleanly when the budget is exhausted — rather than looping until
                # some unrelated limit trips. 0 = unlimited (default). Surfaced via stall_reason.
                board.tokens_used += int(getattr(resp, "prompt_tokens", 0) or 0) + int(getattr(resp, "completion_tokens", 0) or 0)
                if board.max_tokens and board.tokens_used >= board.max_tokens:
                    reason = f"Token budget exhausted ({board.tokens_used}/{board.max_tokens} tokens) — winding run down."
                    board.record_agent_step(agent_id, note=reason)
                    _append_to_challenge_log(board.challenge_id, agent_id, reason)
                    await board.update_worker_state(agent_id, status="DONE", current_task=reason)
                    if not board.stall_reason:
                        board.stall_reason = reason
                    break

                # ── Parse STRATEGY: tag (required from agent per system prompt Rule 7) ──
                strategy_label = "unknown"
                strategy_match = re.search(r"^STRATEGY:\s*(\w+)", content, re.MULTILINE | re.IGNORECASE)
                if strategy_match:
                    strategy_label = strategy_match.group(1).strip().lower()
                else:
                    board.record_agent_step(
                        agent_id,
                        note="[FORMAT WARNING] Missing STRATEGY: <label> line. Prefix your response with 'STRATEGY: <strategy_name>' to categorize your action.",
                    )
                    await board.update_worker_state(
                        agent_id,
                        status="RUNNING",
                        current_task="Restating action with valid STRATEGY: <label> line",
                    )

                # ── Parse the agent's single action per the prompt output contract ──
                if re.search(r"\bBUDGET_EXHAUSTED\b", content):
                    board.record_agent_step(agent_id, note=f"Model reported BUDGET_EXHAUSTED: {content[:200]}")
                    await board.update_worker_state(agent_id, status="DONE", current_task="Budget exhausted (model-reported)")
                    break

                # Explicit FLAG: line is an LLM claim → candidate only, never auto-captured.
                flag_line = re.search(r"FLAG:\s*(\S+)", content)
                if flag_line:
                    cand = flag_line.group(1).strip()
                    if FLAG_REGEX.search(cand) and not FALSE_FLAG_PATTERNS.search(cand):
                        await board.record_flag_candidate(cand, agent_id, "llm_reported")
                        # NOTE: the verified/rejected log entry is written inside
                        # record_flag_candidate() once the verifier returns its verdict.
                        # Do NOT add a duplicate record_agent_step here (Secondary Bug 3).

                # A Python solver block -> write solve.py and run it byte-safely.
                py_match = re.search(r"```python\s*\n(.*?)\n```", content, re.DOTALL)
                if py_match:
                    script = py_match.group(1)
                    import ast
                    try:
                        ast.parse(script)
                    except SyntaxError as se:
                        err_msg = f"SyntaxError in generated Python script (line {se.lineno}, col {se.offset}): {se.msg}"
                        board.record_agent_step(agent_id, note=err_msg)
                        _append_to_challenge_log(board.challenge_id, agent_id, err_msg)
                        board.agent_iterations[agent_id] = iters + 1
                        await board.update_worker_state(agent_id, status="RUNNING", current_task=f"Repairing syntax error: {se.msg}")
                        continue
                    # Content-hash the filename so a REVISED script actually runs (its command
                    # string differs), while a byte-identical retry still dedups. Without this,
                    # every revised solver reused one filename -> one command string -> silently
                    # skipped by the dedup set.
                    script_hash = hashlib.md5(script.encode("utf-8", "replace")).hexdigest()[:8]
                    solver_path = os.path.join(workdir, f"solve_{agent_id}_{script_hash}.py")
                    try:
                        with open(solver_path, "w", encoding="utf-8") as fh:
                            fh.write(script)
                    except OSError as werr:
                        werr_msg = f"Could not write solver: {werr}"
                        board.record_agent_step(agent_id, note=werr_msg)
                        _append_to_challenge_log(board.challenge_id, agent_id, werr_msg)
                        board.agent_iterations[agent_id] = iters + 1
                        continue
                    # Quote the path — the workspace path can contain spaces (e.g.
                    # .../WEB/EASY/Old Sessions), which otherwise splits into [Errno 2].
                    cmd = f'python "{solver_path}"'
                else:
                    cmd = self._extract_command(content)
                    if not cmd:
                        board.record_agent_step(agent_id, note="No executable command produced this turn")
                        _append_to_challenge_log(board.challenge_id, agent_id, "[NO PROGRESS] No executable command produced this turn")
                        board.agent_iterations[agent_id] = iters + 1
                        await board.update_worker_state(agent_id, status="RUNNING",
                                                        current_task="No command produced; re-planning")
                        await asyncio.sleep(1.0)
                        continue

                try:
                    await ws_manager.broadcast({
                        "event": "AI_DECISION", "challenge_id": board.challenge_id, "agent": agent_id,
                        "goal": f"{board.category} challenge next step", "capability": capability,
                        "result": cmd[:250], "confidence": 90, "model": model_name,
                    })
                except Exception:
                    pass

                try:
                    db = SessionLocal()
                    try:
                        dec_row = TrajectoryEventModel(
                            session_id=board.run_id or f"swarm-{board.challenge_id}",
                            run_id=board.run_id,
                            challenge_id=board.challenge_id,
                            agent_id=agent_id,
                            event_type="AI_DECISION",
                            action_type="command",
                            command=cmd[:2000],
                            tool_name=capability or "swarm",
                            decision_summary=f"{board.category} challenge next step",
                            strategy=f"Model: {model_name}",
                            result=cmd[:250],
                            model=model_name
                        )
                        db.add(dec_row)
                        db.commit()
                    except Exception as e:
                        logger.debug(f"[SwarmOrchestrator] AI decision persist skip: {e}")
                    finally:
                        db.close()
                except Exception:
                    pass


                board.agent_iterations[agent_id] = iters + 1

                if cmd in board.executed_commands_dedup:
                    board.record_agent_step(agent_id, command=cmd, note="skipped (already executed by the swarm)")
                    _append_to_challenge_log(board.challenge_id, agent_id, f"[DUPLICATE SKIPPED] {cmd[:150]}")
                    consecutive_duplicates += 1
                    if consecutive_duplicates >= MAX_CONSECUTIVE_DUPLICATES:
                        _append_to_challenge_log(
                            board.challenge_id, agent_id,
                            f"[DUPLICATE LOOP] {consecutive_duplicates} identical duplicates in a row — forcing pivot."
                        )
                        asyncio.create_task(_force_pivot_if_needed(board, agent_id, strategy_label))
                        consecutive_duplicates = 0
                    await asyncio.sleep(0.5)
                    continue

                # Bug 4 Pre-check: Blocked capabilities (sudo / missing binaries)
                cmd_tokens = cmd.strip().split()
                bin_name = os.path.basename(cmd_tokens[0]).lower() if cmd_tokens else ""
                if ("sudo" in board.blocked_capabilities and cmd.strip().startswith("sudo ")) or (f"cmd:{bin_name}" in board.blocked_capabilities):
                    board.record_agent_step(agent_id, command=cmd, note=f"[CAPABILITY BLOCKED] Command '{cmd[:80]}' uses a capability/tool blocked on first failure.")
                    await asyncio.sleep(0.5)
                    continue

                # Bug 1 Pre-check: Cross-agent recon deduplication (auth-preserving)
                recon_key = _normalize_recon_target(cmd)
                if recon_key and recon_key in board.recon_cache:
                    cached = board.recon_cache[recon_key]
                    board.record_agent_step(agent_id, command=cmd, output=cached, note="[RECON CACHE HIT] Result already fetched by another agent.")
                    _append_to_challenge_log(board.challenge_id, agent_id, f"[RECON CACHE HIT] {cmd[:100]}")
                    await asyncio.sleep(0.5)
                    continue

                # Bug 3 Pre-check: Blocked failure signature / command shape check
                cmd_shape = _normalize_command_shape(cmd)
                if cmd_shape in board.blocked_failure_sigs:
                    board.record_agent_step(agent_id, command=cmd, note=f"[SIGNATURE BLOCKED] Command shape '{cmd_shape}' is blocked after repeated failures.")
                    await asyncio.sleep(0.5)
                    continue

                # ── Strategy gate pre-check ────────────────────────────────────────────
                # Must run AFTER the dedup check so an exhausted-strategy command that is also a
                # duplicate still hits the (cheaper) dedup guard first.
                if strategy_label in board.exhausted_strategies:
                    _append_to_challenge_log(
                        board.challenge_id, agent_id,
                        f"[STRATEGY BLOCKED] '{strategy_label}' is already exhausted — "
                        f"forcing pivot (this command will not execute)."
                    )
                    board.record_agent_step(
                        agent_id, command=cmd,
                        note=f"[STRATEGY BLOCKED] Strategy '{strategy_label}' exhausted; skipping execution."
                    )
                    asyncio.create_task(_force_pivot_if_needed(board, agent_id, strategy_label))
                    await asyncio.sleep(0.5)
                    continue

                # ── Privilege gate ─────────────────────────────────────────────────────
                # classify/approve/wait/reconcile all live in the shared gate so this
                # path cannot drift from the other agent command paths that use it.
                # `bin_name` is deliberately recomputed here (NOT reused from the
                # lower-cased pre-check above) because the failure-blocking code below
                # keys board.blocked_capabilities on this exact value.
                bin_name = os.path.basename(cmd.strip().split()[0]) if cmd.strip() else ""
                priv_level = classify_command_privilege(cmd, bin_name)

                # ── Capability-gap retry tagging ──────────────────────────────────────
                # A denied command is recorded as a capability gap and later re-issued by
                # the recovery policy (swarm/supervisor.decide_recovery returns "retry" for
                # capability_gap).  That retry arrives here as an ordinary command, so the
                # operator would be shown a second, context-free card for something they
                # already refused — with no way to tell it apart from a fresh request.
                # Attach the prior denial so the card can say what it is.  This is
                # presentation metadata ONLY: the gate below classifies and decides the
                # command exactly as it would without it, and `context` cannot widen
                # execution (nothing in require_approval reads it).
                _retry_context = _capability_gap_retry_context(board, bin_name, priv_level)

                approved, decision, _sudo_pw_for_exec = await require_approval(
                    cmd=cmd,
                    agent_id=agent_id,
                    pending_approvals=board.pending_approvals,
                    broadcast_fn=ws_manager.broadcast,
                    challenge_id=board.challenge_id,
                    run_id=board.run_id,
                    context=_retry_context,
                )
                # req_sudo is only consulted inside the sudo-stdin block below, which
                # additionally requires a password the gate returns only on approval —
                # so recomputing it here is behavior-identical to the previous inline gate.
                req_sudo = bool(re.search(r"\bsudo\b", cmd))

                if not approved:
                    _sudo_pw_for_exec = None  # discard on deny/timeout
                    status_str = "timed out" if decision is None else "denied"
                    intent = {
                        "agent_id": agent_id,
                        "capability": bin_name or (cmd.split()[0] if cmd.strip() else ""),
                        "target": board.target_scope,
                        "command": cmd,
                        "privilege_level": priv_level,
                        "strategy": strategy_label,
                        "decision": decision,
                        "reason": f"Privilege check {status_str} ({priv_level})",
                    }
                    board.record_capability_gap(intent)
                    _append_to_challenge_log(
                        board.challenge_id,
                        agent_id,
                        f"[CAPABILITY_GAP] Privilege {status_str.upper()} (level={priv_level}) for capability '{intent['capability']}' on target '{intent['target']}'. Intent preserved."
                    )
                    board.record_agent_step(
                        agent_id,
                        command=cmd,
                        note=f"[CAPABILITY_GAP] Privilege {status_str.upper()} for '{intent['capability']}'. Preserving intent for approval escalation.",
                    )
                    await asyncio.sleep(0.5)
                    continue

                board.executed_commands_dedup.add(cmd)
                consecutive_duplicates = 0
                _append_to_challenge_log(board.challenge_id, agent_id, f"Executing: {cmd[:200]}")

                # ── Sudo stdin injection (security: password never in command string) ────
                _stdin_data: Optional[str] = None
                _logged_cmd = cmd
                if req_sudo and approved and _sudo_pw_for_exec:
                    # Strip leading 'sudo' tokens and rebuild as 'sudo -S' so the
                    # password is read from stdin, never from a command-line argument.
                    _stripped = cmd.strip()
                    if _stripped.startswith("sudo "):
                        _inner_cmd = _stripped[5:].lstrip()
                        # Remove any -S that is already present to avoid duplication
                        if _inner_cmd.startswith("-S "):
                            _inner_cmd = _inner_cmd[3:].lstrip()
                    else:
                        _inner_cmd = _stripped
                    # Logged command shows the sudo form WITHOUT the password.
                    _logged_cmd = f"sudo -S {_inner_cmd}"
                    cmd = _logged_cmd  # use rewritten form for execution
                    _stdin_data = f"{_sudo_pw_for_exec}\n"
                    # Wipe the local password variable now that we've used it to
                    # build stdin; the password exists only inside _stdin_data for
                    # the duration of the subprocess call below.
                    _sudo_pw_for_exec = None

                try:
                    res = await tool_manager.execute_tool(
                        "bash", {"command": cmd}, timeout=25,
                        working_directory=workdir, canonical_target=board.target_scope,
                        stdin=_stdin_data,
                    )
                finally:
                    # Ensure the password bytes leave scope regardless of outcome.
                    _stdin_data = None
                output = res.stdout or res.stderr or ""
                if getattr(res, "execution_failure", False):
                    _append_to_challenge_log(board.challenge_id, agent_id,
                                             f"Execution failure ({getattr(res, 'failure_category', 'UNKNOWN')}): {res.stderr[:200]}")

                await board.record_tool_execution(agent_id, cmd, res, privilege_level=priv_level, approved=approved)
                board.record_agent_step(agent_id, command=cmd, output=output)

                # ── Deterministic web-surface facts: forms + probe delivery ───────────
                # Record the form field names this response actually exposes, so no agent
                # ever has to guess one. An invented field name silently no-ops against a
                # form that redirects regardless — the exact way the SSTI1 run burned its
                # whole budget on payloads the template never saw.
                for form in extract_forms(output, base_url=board.target_scope):
                    if form not in board.observed_forms:
                        board.observed_forms.append(form)
                        action = form.get("action") or ""
                        if action and action not in board.discovered_endpoints:
                            board.discovered_endpoints.add(action)
                        _append_to_challenge_log(
                            board.challenge_id, agent_id,
                            f"🧾 Form discovered — {describe_form(form)}")

                # A self-checking payload that never renders means the injection is not
                # reaching the sink — NOT that the technique failed. Surfacing it (in the
                # challenge log AND the shared agent transcript) lets the swarm correct
                # course instead of concluding the approach is dead.
                if not getattr(res, "execution_failure", False):
                    probe = verify_template_probe(cmd, output)
                    if probe and probe.get("delivered") is False:
                        probe_notice = (
                            f"[PROBE] self-checking payload {probe['payload']} did not render "
                            f"(expected {probe['expected']}): {probe['reason']}. If this is a "
                            f"reflection-based injection then the payload is not reaching the sink — "
                            f"confirm the form field name and follow redirects before treating this "
                            f"approach as failed."
                        )
                        _append_to_challenge_log(board.challenge_id, agent_id, probe_notice)
                        board.record_agent_step(agent_id, note=probe_notice)

                # Bug 1 Store: populate recon cache on successful execution
                if recon_key and getattr(res, "exit_code", 1) == 0 and output.strip():
                    board.recon_cache[recon_key] = output[:1000]

                # Bug 4 Post-check: First-occurrence terminal failure blocking
                fail_cat = getattr(res, "failure_category", None)
                if getattr(res, "execution_failure", False):
                    if fail_cat == "PERMISSION_DENIED" and "sudo" in (cmd + " " + output).lower():
                        board.blocked_capabilities.add("sudo")
                        board.record_agent_step(agent_id, note="[CAPABILITY BLOCKED] sudo requires password/tty; blocked for remaining run.")
                    elif fail_cat == "COMMAND_NOT_FOUND" and bin_name:
                        board.blocked_capabilities.add(f"cmd:{bin_name}")
                        board.record_agent_step(agent_id, note=f"[CAPABILITY BLOCKED] Binary '{bin_name}' not found; blocked for remaining run.")

                    # Bug 3 Post-check: Track failure signature & block command shape on repeat limit
                    sig_key = _normalize_failure_signature(cmd, fail_cat or "EXEC_FAIL", res.stderr or output)
                    board.failure_signatures[sig_key] = board.failure_signatures.get(sig_key, 0) + 1
                    if board.failure_signatures[sig_key] >= 3:
                        if cmd_shape not in board.blocked_failure_sigs:
                            board.blocked_failure_sigs.add(cmd_shape)
                            board.record_agent_step(agent_id, note=f"[FAILURE REPEAT LIMIT] Command shape '{cmd_shape}' hit threshold (3); blocking repeated attempts.")
                            # Cross-mission record (A4): share this dead-end with future
                            # missions of the same category (advisory recall, not a hard ban).
                            try:
                                from backend.knowledge.failed_approaches import record_failed_approach
                                record_failed_approach(board.category, cmd_shape, fail_cat or "EXEC_FAIL")
                            except Exception:
                                pass

                # Local execution failures (Errno 2 / SyntaxError / permission / missing dep)
                # never reached the target — a broken invocation, not a target response. Don't
                # let identical retries burn the shared free-tier budget: inject a corrective
                # note and abort the agent after LOCAL_EXEC_MAX_RETRIES consecutive hits.
                fail_cat = getattr(res, "failure_category", None)
                if getattr(res, "execution_failure", False) and fail_cat in LOCAL_EXEC_CATEGORIES:
                    board.agent_local_fail_streak[agent_id] = board.agent_local_fail_streak.get(agent_id, 0) + 1
                    streak = board.agent_local_fail_streak[agent_id]
                    board.record_agent_step(agent_id, note=(
                        f"LOCAL EXECUTION ERROR ({fail_cat}) — this command never reached the target; "
                        f"it is a local invocation problem, not a target response. Fix the invocation "
                        f"(e.g. quote any path with a space: python \"my dir/solve.py\") or try a different "
                        f"approach. Do NOT re-run the same command."))
                    max_local = int(getattr(settings, "LOCAL_EXEC_MAX_RETRIES", 2))
                    if streak >= max_local:
                        blocker = (f"ABORTING agent after {streak} consecutive local execution failures "
                                   f"({fail_cat}) — blocked on local tooling; reporting instead of retrying.")
                        board.record_agent_step(agent_id, note=blocker)
                        _append_to_challenge_log(board.challenge_id, agent_id, blocker)
                        await board.update_worker_state(agent_id, status="DONE", current_task=blocker)
                        if not board.stall_reason:
                            board.stall_reason = f"An agent was blocked on a local tooling error ({fail_cat})"
                        break
                else:
                    # Command reached the target (or a non-local failure) — reset the streak.
                    board.agent_local_fail_streak[agent_id] = 0

                await board.update_worker_state(
                    agent_id, status="RUNNING", last_tool=cmd[:160],
                    last_result=(output[:200] or f"[Exit {res.exit_code}] no output"),
                    commands_run=board.worker_states.get(agent_id, {}).get("commands_run", 0) + 1,
                )
                _append_to_challenge_log(board.challenge_id, agent_id, f"Output ({len(output)} bytes, exit={res.exit_code}): {output[:300]}")

                try:
                    await ws_manager.broadcast({
                        "event": "LOG_OUTPUT", "challenge_id": board.challenge_id, "run_id": board.run_id,
                        "command": cmd,
                        "output": output[:3000] if output else f"[Exit Code {res.exit_code}] Execution complete with no output.",
                        "exit_code": res.exit_code, "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    })
                except Exception:
                    pass

                # Flags/leads come ONLY from real tool output — never from LLM prose.
                self._check_tool_output_for_flags(output, board, agent_id)
                self._check_tool_output_for_rejections(output, board, agent_id)
                await self._apply_decoded_directives(output, board, agent_id)

                # ── Empirical Baseline Profiling, Anomaly Detection & Actionable Preemption ──
                # Check if this command handled any pending preemption
                for p in board.actionable_preemptions:
                    if not p.get("handled"):
                        targ = p.get("normalized_target") or p.get("raw_value") or ""
                        raw_val = p.get("raw_value") or ""
                        if (targ and targ in cmd) or (raw_val and raw_val in cmd):
                            p["handled"] = True
                            _append_to_challenge_log(
                                board.challenge_id, agent_id,
                                f"✓ Followed up on actionable preemption artifact: {targ}"
                            )

                anom_result = board.response_profiler.profile_and_evaluate(
                    command_or_target=cmd,
                    output=output,
                    status_code=getattr(res, "exit_code", None),
                    base_url=board.target_scope,
                )
                if anom_result.is_anomalous:
                    reasons_str = "; ".join(anom_result.reasons)
                    _append_to_challenge_log(
                        board.challenge_id, agent_id,
                        f"[ANOMALOUS_RESPONSE] Diverged from baseline ({len(output)} bytes, score={anom_result.score:.2f}): {reasons_str}"
                    )
                    try:
                        await ws_manager.broadcast({
                            "event": "ANOMALOUS_RESPONSE",
                            "challenge_id": board.challenge_id,
                            "run_id": board.run_id,
                            "agent": agent_id,
                            "command": cmd[:200],
                            "score": anom_result.score,
                            "reasons": anom_result.reasons,
                            "candidates": [c.to_dict() for c in anom_result.candidate_artifacts],
                        })
                    except Exception:
                        pass

                # If candidate artifacts were extracted:
                for cand in anom_result.candidate_artifacts:
                    targ = cand.normalized_target or cand.raw_value
                    if targ and targ not in board.seen_candidate_targets:
                        board.seen_candidate_targets.add(targ)
                        if cand.artifact_type in ("PATH", "ENDPOINT", "URL"):
                            board.discovered_endpoints.add(targ)
                        probes = generate_post_exploitation_probes(targ, base_url=board.target_scope)
                        preempt_item = {
                            "raw_value": cand.raw_value,
                            "artifact_type": cand.artifact_type,
                            "normalized_target": targ,
                            "source_command": cmd,
                            "reasons": anom_result.reasons,
                            "suggested_probes": probes,
                            "handled": False,
                        }
                        board.actionable_preemptions.append(preempt_item)
                        _append_to_challenge_log(
                            board.challenge_id, agent_id,
                            f"⚡ Actionable candidate artifact discovered: {targ} ({cand.artifact_type}) — prioritizing immediate verification"
                        )

                # ── Strategy evidence fingerprint + stale accounting ───────────────────
                # Runs on EVERY clean execution (exit=0 or not) because a script that
                # exits cleanly and prints "no flag found" is the exact failure mode we
                # need to catch.  _update_strategy_state returns True when the strategy
                # was just exhausted; in that case kick off the pivot immediately.
                just_exhausted = _update_strategy_state(board, agent_id, strategy_label, output, res)
                if just_exhausted:
                    asyncio.create_task(_force_pivot_if_needed(board, agent_id, strategy_label))

                # Encoded-artifact reconstruction: if this output is really an encoded
                # file (e.g. ASCII 0/1 that reconstructs to a JPEG), rebuild it, preserve
                # it as evidence, and escalate its analysis — never conclude "no flag"
                # with an un-analyzed artifact still on the board.
                await self._reconstruct_and_escalate(
                    output, board, agent_id, workdir, origin_label=f"tool_output:{(cmd or '')[:60]}")
                if "<!--" in output:
                    for c in re.findall(r"<!--(.*?)-->", output, re.DOTALL):
                        c_clean = c.strip()
                        if len(c_clean) > 3:
                            await self._apply_decoded_directives(c_clean, board, agent_id)
                for ep in re.findall(r'href=["\'](/[^"\']+)["\']', output):
                    board.discovered_endpoints.add(ep)

                consecutive_errors = 0
                await board._broadcast_blackboard()

            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"[{agent_id}] Error #{consecutive_errors}: {e}")
                _append_to_challenge_log(board.challenge_id, agent_id, f"Error #{consecutive_errors}: {e}")
                await board.update_worker_state(agent_id, status="IDLE", failures=consecutive_errors,
                                                current_task=f"Error: {str(e)[:100]}")
                await asyncio.sleep(min(3 * consecutive_errors, 15))

        await board.update_worker_state(agent_id, status="DONE", current_task="Agent finished")

    # ── HITL checkpoint coordinator (hard pause & wait) ─────────────────────────

    def _make_summarizer(self):
        """Async callable for the strictly-extractive narrative pass. Returns
        None on any failure so the report still renders deterministically.

        Routed through the general_reasoning chain (NOT Gemini): Gemini is reserved
        for vision + final report authoring (Workstream C), and its safety filters
        refuse often enough on red-team context to make it unreliable mid-run."""
        async def _summarize(prompt: str) -> Optional[str]:
            try:
                resp = await model_router.route_request(
                    prompt=prompt, capability="general_reasoning")
                if resp and not resp.is_refusal:
                    return resp.content
            except Exception:
                return None
            return None
        return _summarize

    def _wind_down_for_expiry(self, board: "SwarmBlackboard"):
        """Stop the run cleanly as the platform instance-expiry deadline nears. Agents
        self-stop on the same deadline (see _agent_worker); this signals run_swarm to
        finalize with best findings instead of the coordinator hard-waiting on a paste."""
        board.stall_reason = board.stall_reason or "Platform instance expiry reached"
        _append_to_challenge_log(board.challenge_id, "checkpoint",
                                 "Instance expiry within wind-down buffer — stopping run with best findings")
        board.is_stopped = True
        board.checkpoint_response_event.set()

    async def _checkpoint_coordinator(self, board: "SwarmBlackboard", workdir: str):
        """Every CHECKPOINT_INTERVAL_SECONDS (or as the instance timer nears expiry,
        or immediately on a verified flag), hard-pause all agents, emit ONE
        consolidated report, and wait for the operator's pasted guidance."""
        _append_to_challenge_log(board.challenge_id, "checkpoint", "Checkpoint coordinator started")
        while not board.flag_captured and not board.is_stopped:
            interval = int(getattr(settings, "CHECKPOINT_INTERVAL_SECONDS", 300))
            buffer = int(getattr(settings, "INSTANCE_WINDDOWN_BUFFER_SECONDS", 30))
            # Too close to instance expiry to run an interactive (operator-paste) checkpoint
            # — wind down instead of hard-waiting for a paste that can't complete in time.
            if board.instance_expiry_ts and (board.instance_expiry_ts - time.time()) <= buffer:
                self._wind_down_for_expiry(board)
                break
            if board.instance_expiry_ts:
                secs_left = board.instance_expiry_ts - time.time()
                if secs_left > 0:
                    interval = min(interval, max(10, int(secs_left - buffer)))
            try:
                await asyncio.wait_for(board.flag_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            if board.flag_captured or board.is_stopped:
                break
            # Crossed into the expiry buffer while waiting → wind down, don't hard-wait.
            if board.instance_expiry_ts and (board.instance_expiry_ts - time.time()) <= buffer:
                self._wind_down_for_expiry(board)
                break
            try:
                await self._run_checkpoint_cycle(board, workdir)
            except Exception as e:
                logger.warning(f"[checkpoint] cycle error (non-fatal): {e}")
                _append_to_challenge_log(board.challenge_id, "checkpoint", f"Cycle error (non-fatal): {e}")
                board.checkpoint_pause = False
                board.checkpoint_active = False
        _append_to_challenge_log(board.challenge_id, "checkpoint", "Checkpoint coordinator stopped")

    async def _run_checkpoint_cycle(self, board: "SwarmBlackboard", workdir: str):
        cycle_start = board.cycle_window_start_ts
        board.checkpoint_pause = True
        board.checkpoint_active = True
        board.cycle_n += 1
        board.checkpoint_response_event.clear()
        board.latest_pasted_response = None
        _append_to_challenge_log(board.challenge_id, "checkpoint", f"=== Checkpoint cycle {board.cycle_n}: pausing agents ===")
        # Passive provider-headroom summary (from real response headers; no extra calls).
        try:
            from backend.providers.quota_manager import quota_manager as _qm
            _append_to_challenge_log(board.challenge_id, "checkpoint", f"Provider headroom: {_qm.ratelimit_summary()}")
        except Exception:
            pass
        await asyncio.sleep(2.0)  # brief quiesce so in-flight steps land

        records = board.snapshot_agent_records()
        start_str = datetime.fromtimestamp(cycle_start, tz=timezone.utc).strftime("%H:%M:%S")
        end_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        report = await checkpoint_pipeline.build_consolidated_report(
            challenge_name=board.challenge_name, category=board.category, difficulty=board.difficulty,
            target=board.target_scope, cycle_n=board.cycle_n, start_time=start_str, end_time=end_str,
            records=records, summarizer=self._make_summarizer(),
        )
        board.last_checkpoint_report = report

        db = SessionLocal()
        try:
            db.add(CheckpointModel(
                run_id=board.run_id,
                state_snapshot={"kind": "hitl_checkpoint", "cycle_n": board.cycle_n,
                                "report": report, "agent_ids": list(board.agent_ids)},
                last_successful_action=f"checkpoint_cycle_{board.cycle_n}",
                resumable=True,
            ))
            run_obj = db.query(RunModel).filter(RunModel.id == board.run_id).first()
            if run_obj:
                run_obj.status = "WAITING_FOR_USER"
            ch_obj = db.query(ChallengeModel).filter(ChallengeModel.id == board.challenge_id).first()
            if ch_obj:
                ch_obj.status = "WAITING_FOR_USER"
            db.commit()
        except Exception as e:
            logger.debug(f"[checkpoint] persist skip: {e}")
        finally:
            db.close()

        await ws_manager.broadcast({
            "event": "CHECKPOINT_REACHED", "challenge_id": board.challenge_id, "run_id": board.run_id,
            "cycle": board.cycle_n, "report": report,
        })
        _append_to_challenge_log(board.challenge_id, "checkpoint",
                                 f"Report emitted for cycle {board.cycle_n}; awaiting operator paste")

        # Bounded wait for operator's pasted external-model response.
        # Concurrently monitors operator response, flag capture, and cancellation/pause
        # while respecting CHECKPOINT_TIMEOUT_SECONDS and instance wind-down buffer.
        timeout = float(getattr(settings, "CHECKPOINT_TIMEOUT_SECONDS", 30))
        buffer = float(getattr(settings, "INSTANCE_WINDDOWN_BUFFER_SECONDS", 30))
        if board.instance_expiry_ts:
            secs_left = board.instance_expiry_ts - time.time() - buffer
            if secs_left > 0:
                timeout = min(timeout, max(0.1, secs_left))
            else:
                self._wind_down_for_expiry(board)
                board.checkpoint_pause = False
                board.checkpoint_active = False
                return

        resp_wait = asyncio.create_task(board.checkpoint_response_event.wait())
        flag_wait = asyncio.create_task(board.flag_event.wait())
        timed_out = False
        try:
            if timeout > 0:
                done, pending = await asyncio.wait(
                    [resp_wait, flag_wait],
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    timed_out = True
            else:
                timed_out = True
        except asyncio.CancelledError:
            board.checkpoint_pause = False
            board.checkpoint_active = False
            raise
        except Exception:
            timed_out = True
        finally:
            resp_wait.cancel()
            flag_wait.cancel()

        if board.is_stopped or board.flag_captured:
            board.checkpoint_pause = False
            board.checkpoint_active = False
            return

        if board.instance_expiry_ts and (board.instance_expiry_ts - time.time()) <= buffer:
            self._wind_down_for_expiry(board)
            board.checkpoint_pause = False
            board.checkpoint_active = False
            return

        pasted = board.latest_pasted_response
        if timed_out or not pasted:
            _append_to_challenge_log(
                board.challenge_id, "checkpoint",
                f"No operator guidance received within timeout ({timeout:.1f}s) — resuming autonomous execution"
            )
            await ws_manager.broadcast({
                "event": "CHECKPOINT_RESUMED", "challenge_id": board.challenge_id, "run_id": board.run_id,
                "cycle": board.cycle_n, "routed": [], "timeout": True,
            })
        else:
            pasted_str = pasted or ""
            parsed = checkpoint_pipeline.parse_suggestions(pasted_str, board.agent_ids)

            # -- Pull state for evaluation from the shared mission (best-effort) --
            # We use the board's own fields where available; the suggestion evaluator
            # gracefully handles None/empty iterables.
            _ms = getattr(board, "mission_state", None)
            _exh = list(board.exhausted_strategies) if hasattr(board, "exhausted_strategies") else []
            _fail = list(getattr(_ms, "failed_techniques", []) or []) if _ms else []
            _eps = list(getattr(_ms, "endpoints", []) or getattr(_ms, "known_endpoints", []) or []) if _ms else []
            _files = list(getattr(_ms, "known_files", []) or []) if _ms else []
            _flags = list(getattr(_ms, "flag_candidates", []) or []) if _ms else []

            def _evaluate_and_inject(aid: str, directive: str) -> None:
                """Evaluate one directive; inject only if ACCEPTED or MODIFIED."""
                ev = checkpoint_pipeline.evaluate_suggestion(
                    directive,
                    exhausted_strategies=_exh,
                    failed_techniques=_fail,
                    known_endpoints=_eps,
                    known_files=_files,
                    flag_candidates=_flags,
                )
                if ev.decision == SuggestionDecision.REJECT:
                    _append_to_challenge_log(
                        board.challenge_id, "checkpoint",
                        f"[SUGGESTION_REJECTED:{aid}] {ev.reason} | original: {directive[:120]}",
                    )
                    return  # Do NOT inject rejected suggestions
                action_text = ev.suggested_action or directive
                prev = board.agent_directives.get(aid, "")
                if ev.decision == SuggestionDecision.MODIFY:
                    _append_to_challenge_log(
                        board.challenge_id, "checkpoint",
                        f"[SUGGESTION_MODIFIED:{aid}] {ev.reason}",
                    )
                board.agent_directives[aid] = (prev + "\n\n" + action_text).strip() if prev else action_text

            if parsed.parsed:
                async with board._lock:
                    for aid, directive in parsed.directives.items():
                        _evaluate_and_inject(aid, directive)
                injected = [aid for aid in parsed.directives if board.agent_directives.get(aid)]
                _append_to_challenge_log(board.challenge_id, "checkpoint",
                                         f"Routed evaluated directives to: {', '.join(sorted(parsed.directives.keys()))}"
                                         + (f" | {parsed.note}" if parsed.note else ""))
                await ws_manager.broadcast({
                    "event": "CHECKPOINT_RESUMED", "challenge_id": board.challenge_id, "run_id": board.run_id,
                    "cycle": board.cycle_n, "routed": sorted(parsed.directives.keys()),
                    "unknown_labels": parsed.unknown_labels,
                })
            else:
                if parsed.fallback and parsed.fallback_text:
                    async with board._lock:
                        for aid in board.agent_ids:
                            _evaluate_and_inject(aid, "[GENERAL GUIDANCE] " + parsed.fallback_text)
                _append_to_challenge_log(board.challenge_id, "checkpoint",
                                         f"UNPARSEABLE paste — {parsed.note} Applied as general guidance to all agents.")
                await ws_manager.broadcast({
                    "event": "CHECKPOINT_PARSE_ERROR", "challenge_id": board.challenge_id, "run_id": board.run_id,
                    "cycle": board.cycle_n, "note": parsed.note, "applied_as_general": bool(parsed.fallback_text),
                })

        # Refresh each agent's budget for the new cycle. The operator just re-authorized
        # continuation at the checkpoint, so directives get a fresh iteration/minute window
        # instead of dying instantly on a budget that was already spent before the pause
        # (the checkpoint-interval == budget trap). The dedup set is deliberately NOT reset,
        # so agents still never re-run an identical command.
        now_ts = time.time()
        for aid in board.agent_ids:
            board.agent_started_ts[aid] = now_ts
            board.agent_paused_seconds[aid] = 0.0
            board.agent_iterations[aid] = 0
            board.agent_local_fail_streak[aid] = 0
        _append_to_challenge_log(
            board.challenge_id, "checkpoint",
            f"Budget refreshed for {len(board.agent_ids)} agents on resume (fresh cycle window)")

        board.checkpoint_pause = False
        board.checkpoint_active = False
        board.latest_pasted_response = None
        db = SessionLocal()
        try:
            run_obj = db.query(RunModel).filter(RunModel.id == board.run_id).first()
            if run_obj:
                run_obj.status = "RUNNING"
            ch_obj = db.query(ChallengeModel).filter(ChallengeModel.id == board.challenge_id).first()
            if ch_obj:
                ch_obj.status = "RUNNING"
            db.commit()
        except Exception:
            pass
        finally:
            db.close()
        board.cycle_window_start_ts = time.time()
        _append_to_challenge_log(board.challenge_id, "checkpoint", f"=== Checkpoint cycle {board.cycle_n}: agents resumed ===")

    async def submit_checkpoint_response(self, challenge_id: str, text: str) -> Dict[str, Any]:
        """Deliver the operator's pasted external-model response to a waiting swarm.
        Called by the API. Parsing/injection happens in the coordinator on wake."""
        for board in list(self.active_swarms.values()):
            if board.challenge_id == challenge_id and board.checkpoint_active:
                board.latest_pasted_response = text or ""
                preview = checkpoint_pipeline.parse_suggestions(text or "", board.agent_ids)
                board.checkpoint_response_event.set()
                return {
                    "accepted": True, "parsed": preview.parsed,
                    "routed": sorted(preview.directives.keys()),
                    "fallback": preview.fallback, "unknown_labels": preview.unknown_labels,
                    "note": preview.note,
                }
        return {"accepted": False, "reason": "No active checkpoint is awaiting a response for this challenge."}

    async def submit_approval_response(self, request_id: str, decision: str, sudo_password: Optional[str] = None) -> Dict[str, Any]:
        """Deliver operator's approve/deny decision for a pending privileged command.

        When *decision* is 'approve' and the pending entry has requires_sudo=True,
        *sudo_password* is stored in-memory on the pending entry dict ONLY — it is
        never logged, never persisted, never broadcast.  The worker loop retrieves it
        once, wipes it from the dict before execution, and discards it after the
        subprocess completes.
        """
        if decision not in ("approve", "deny"):
            return {"accepted": False, "reason": "Invalid decision. Must be 'approve' or 'deny'."}

        for board in list(self.active_swarms.values()):
            if request_id in board.pending_approvals:
                entry = board.pending_approvals[request_id]
                entry["decision"] = decision
                # Store sudo_password in-memory only; only when approving a sudo command.
                if decision == "approve" and entry.get("requires_sudo"):
                    entry["sudo_password"] = sudo_password  # may be None if not a sudo command
                entry["event"].set()
                return {"accepted": True}

        # Non-swarm callers (legacy ReAct loop, agent_runtime RealToolExecutor) own no
        # board, so their requests live in the shared registry. Same uuid4 keyspace, so
        # there is no ambiguity with the board lookup above.
        shared_entry = SHARED_PENDING_APPROVALS.get(request_id)
        if shared_entry is not None:
            shared_entry["decision"] = decision
            if decision == "approve" and shared_entry.get("requires_sudo"):
                shared_entry["sudo_password"] = sudo_password
            shared_entry["event"].set()
            return {"accepted": True}

        return {"accepted": False, "reason": "No pending approval with this request_id."}

    def _extract_command(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"```(?:bash|sh)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = [l.strip() for l in text.strip().split("\n") if l.strip().startswith(("curl", "python", "ffuf", "nmap", "sqlmap"))]
        return lines[0] if lines else None

    def _check_tool_output_for_rejections(self, text: str, board: SwarmBlackboard, worker_id: str):
        """Detect target rejection signals (e.g., 'User not found', 'Invalid token')."""
        if not text:
            return
        text_lower = text.lower()
        if "user not found" in text_lower or "unknown user" in text_lower or "no such user" in text_lower or "user does not exist" in text_lower:
            board.last_target_rejection = "USER_NOT_FOUND"
            _append_to_challenge_log(board.challenge_id, worker_id, "ℹ Target rejected username ('User not found'). Triggering candidate entity priority refill.")
        elif "invalid token" in text_lower or "token expired" in text_lower:
            board.last_target_rejection = "INVALID_TOKEN"

    async def _reconstruct_and_escalate(self, text: str, board: "SwarmBlackboard", worker_id: str,
                                        workdir: str, origin_label: str = ""):
        """Deterministically reconstruct an encoded artifact from ``text`` and escalate it.

        If ``text`` contains an encoded representation of another file (ASCII binary,
        hex, base64), rebuild the real bytes, scan any readable rendering for a flag,
        preserve a recognized/opaque file as a derived artifact (provenance sidecar),
        register it as shared evidence + an analyzable input, and attach tesseract-free
        visual metadata for images. Never executes the artifact (requirement #15).
        """
        if not text or len(text) < 64:
            return
        try:
            from backend.agents.artifact_reconstruction import (
                reconstruct_from_text, persist_derived_artifact, is_image_type,
                inspect_image, attempt_ocr,
            )
        except Exception as exc:
            logger.debug("[reconstruct] module unavailable: %s", exc)
            return

        input_hash = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        async with board._lock:
            if input_hash in board.processed_recon_inputs:
                return
            board.processed_recon_inputs.add(input_hash)

        try:
            outcome = reconstruct_from_text(text, origin_label=origin_label)
        except Exception as exc:
            logger.debug("[reconstruct] failed: %s", exc)
            return
        if outcome is None:
            return

        task_context = {
            "description": board.description,
            "challenge_name": board.challenge_name,
            "category": board.category,
            "flag_pattern": board.flag_pattern,
            "target_scope": board.target_scope,
        }

        # Any readable rendering is scanned for candidates with explicit provenance.
        if outcome.decoded_text:
            extracted = board.answer_resolver.extract_candidates(
                outcome.decoded_text, task_context=task_context, source=AnswerSource.RECONSTRUCTED_ARTIFACT
            )
            for cand in extracted:
                await board.record_flag_candidate(cand.value, worker_id, "reconstructed_artifact")
            m = FLAG_REGEX.search(outcome.decoded_text)
            if m and not FALSE_FLAG_PATTERNS.search(m.group(0)):
                await board.record_flag_candidate(m.group(0).strip(), worker_id, "reconstructed_artifact")

        if not outcome.should_persist:
            return

        derived_path = persist_derived_artifact(outcome, workdir)
        newly = await board.register_derived_artifact(outcome, derived_path, worker_id)
        if not newly or not derived_path:
            return

        # Visual metadata (tesseract-free) for image artifacts, so a vision-capable
        # agent knows what it is looking at. OCR is strictly best-effort.
        if is_image_type(outcome.artifact_type):
            meta = inspect_image(derived_path)
            ocr = attempt_ocr(derived_path)
            for d in board.derived_artifacts:
                if d.get("derived_path") == derived_path:
                    d["visual"] = meta
                    d["ocr_available"] = bool(ocr.get("available"))
                    break
            if ocr.get("available") and ocr.get("text"):
                extracted_ocr = board.answer_resolver.extract_candidates(
                    ocr["text"], task_context=task_context, source=AnswerSource.RECONSTRUCTED_ARTIFACT_OCR
                )
                for cand in extracted_ocr:
                    await board.record_flag_candidate(cand.value, worker_id, "reconstructed_artifact_ocr")
                m2 = FLAG_REGEX.search(ocr["text"])
                if m2 and not FALSE_FLAG_PATTERNS.search(m2.group(0)):
                    await board.record_flag_candidate(m2.group(0).strip(), worker_id, "reconstructed_artifact_ocr")
            # Escalate image reading to vision_read capability
            await self.analyze_derived_artifact(board, derived_path, worker_id)

    async def analyze_derived_artifact(
        self, board: "SwarmBlackboard", derived_path: str, worker_id: str
    ) -> Optional[str]:
        """Analyze a derived artifact via vision_read (or applicable tool) and record candidate flags/answers."""
        if not derived_path or not os.path.isfile(derived_path):
            return None

        task_context = {
            "description": board.description,
            "challenge_name": board.challenge_name,
            "category": board.category,
            "flag_pattern": board.flag_pattern,
            "target_scope": board.target_scope,
        }

        # Execute vision_read capability on derived_path
        res = await tool_manager.execute_capability("vision_read", target=derived_path)
        if res.status == "SUCCESS" and res.stdout:
            # Check stdout for generic and flag candidates
            extracted = board.answer_resolver.extract_candidates(
                res.stdout, task_context=task_context, source=AnswerSource.VISION_READ
            )
            for cand in extracted:
                await board.record_flag_candidate(
                    cand.value, worker_id, "vision_read",
                    evidence={"derived_path": derived_path, "tool": "vision_read", "stdout": res.stdout[:500]}
                )
            for m in FLAG_REGEX.finditer(res.stdout):
                cand_str = m.group(0).strip()
                if not FALSE_FLAG_PATTERNS.search(cand_str):
                    await board.record_flag_candidate(
                        cand_str, worker_id, "vision_read",
                        evidence={"derived_path": derived_path, "tool": "vision_read", "stdout": res.stdout[:500]}
                    )
            board.mark_derived_analyzed(derived_path)
            return res.stdout
        elif res.stderr:
            logger.info(f"[{worker_id}] vision_read analysis notice for {derived_path}: {res.stderr}")
            if res.failure_category == "MODEL_REFUSAL" or "PAID_MODEL_ALLOWED" in res.stderr:
                _append_to_challenge_log(board.challenge_id, worker_id, f"ℹ vision_read: {res.stderr}")
        return None

    async def _scan_attached_for_encoded(self, board: "SwarmBlackboard", workdir: str):
        """One-shot scan of attached artifacts for an encoded representation of a file.

        Handles the case where the CHALLENGE FILE ITSELF (not a command's output) is a
        wall of ASCII bits / hex / base64 that reconstructs into another file type —
        e.g. a forensics artifact provided as text. Text-like files only, size-capped,
        and skips anything already carrying a binary magic (real binaries are handled by
        the existing acquisition path) and the forge_derived output dir.
        """
        for path in list(board.attached_file_paths):
            try:
                if not path or "forge_derived" in path.replace("\\", "/"):
                    continue
                if not os.path.isfile(path):
                    continue
                if os.path.getsize(path) > 8 * 1024 * 1024:      # 8 MB cap
                    continue
                with open(path, "rb") as fh:
                    head = fh.read(16)
                # Skip files that already ARE a known binary type.
                from backend.agents.artifact_classifier import _classify_magic
                label, _ = _classify_magic(head)
                if label not in ("text", "unknown"):
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read(9 * 1024 * 1024)
                await self._reconstruct_and_escalate(
                    content, board, "orchestrator", workdir,
                    origin_label=f"attached:{os.path.basename(path)}")
            except Exception as exc:
                logger.debug("[reconstruct] attached scan skip %s: %s", path, exc)

    def _check_tool_output_for_flags(self, text: str, board: SwarmBlackboard, worker_id: str):
        """Scan ONLY tool/command output for flag and answer patterns. Never call this on LLM prose."""
        if not text:
            return
        task_context = {
            "description": board.description,
            "challenge_name": board.challenge_name,
            "category": board.category,
            "flag_pattern": board.flag_pattern,
            "target_scope": board.target_scope,
        }
        extracted = board.answer_resolver.extract_candidates(
            text, task_context=task_context, source=AnswerSource.TOOL_OUTPUT
        )
        for cand in extracted:
            asyncio.create_task(board.record_flag_candidate(cand.value, worker_id, "tool_output"))

        match = FLAG_REGEX.search(text)
        if match:
            candidate = match.group(0).strip()
            if not FALSE_FLAG_PATTERNS.search(candidate):
                asyncio.create_task(board.record_flag_candidate(candidate, worker_id, "tool_output"))

    async def _apply_decoded_directives(self, raw: str, board: SwarmBlackboard, worker_id: str) -> bool:
        """Deterministically decode an artifact (HTML comment, task text) and act
        on concrete leads (headers, candidate usernames, flags).

        - Caches hash of input to prevent 15x duplicate decode loops on unchanged input.
        - Harvests candidate usernames and queues targeted exploit tasks.
        - Extracts header directives (e.g. 'use header "X-Dev-Access: yes"').
        """
        if not raw or not raw.strip():
            return False

        import hashlib
        raw_hash = hashlib.md5(raw.strip().encode("utf-8")).hexdigest()
        async with board._lock:
            if raw_hash in board.processed_decode_hashes:
                return False
            board.processed_decode_hashes.add(raw_hash)

        task_context = {
            "description": board.description,
            "challenge_name": board.challenge_name,
            "category": board.category,
            "flag_pattern": board.flag_pattern,
            "target_scope": board.target_scope,
        }

        acted = False
        try:
            for d in _decode_artifacts(raw):
                decoded = d["decoded"]
                _append_to_challenge_log(board.challenge_id, worker_id, f"🔓 {d['scheme']} decode: {decoded[:160]}")
                # Flags/answers hidden via an encoding inside a real captured artifact.
                extracted = board.answer_resolver.extract_candidates(
                    decoded, task_context=task_context, source=AnswerSource.DECODED_ARTIFACT
                )
                for cand in extracted:
                    await board.record_flag_candidate(cand.value, worker_id, "decoded_artifact")
                    acted = True

                for fm in FLAG_REGEX.finditer(decoded):
                    cand = fm.group(0).strip()
                    if not FALSE_FLAG_PATTERNS.search(cand):
                        await board.record_flag_candidate(cand, worker_id, "decoded_artifact")
                        acted = True

                # Harvest candidate usernames (e.g. 'NOTE: Jack', 'username: Jack', 'dev: Jack', 'account: Jack')
                user_matches = re.findall(r"\b(?:user(?:name)?|developer|account|NOTE)\s*[:=\-]\s*['\"]?([A-Za-z0-9_\-\.]{3,24})", decoded, re.IGNORECASE)
                for u in user_matches:
                    u_clean = u.strip().strip("'\".,;:()")
                    if u_clean.lower() not in ["not", "the", "found", "error", "true", "false", "null", "undefined", "header", "temporary", "bypass", "access"]:
                        if u_clean not in board.candidate_usernames:
                            board.candidate_usernames.add(u_clean)
                            _append_to_challenge_log(board.challenge_id, worker_id, f"👤 Candidate username discovered: {u_clean}")
                            logger.info(f"[SwarmBlackboard] Candidate username discovered by {worker_id}: {u_clean}")

                # Header directives only when the decode explicitly names one.
                if re.search(r"\bheader\b", decoded, re.IGNORECASE):
                    for hm in _HEADER_HINT_RE.finditer(decoded):
                        if await board.note_exploit_header(hm.group(1), hm.group(2), worker_id):
                            acted = True
        except Exception as e:
            logger.debug(f"[{worker_id}] decode-directives skip: {e}")

        return acted

    async def _learn_from_run(self, board: "SwarmBlackboard", outcome: str = "success"):
        """LEARN → STORE (§3, §5, §12, §19; Phase 6 Part 8).

        Distil the finished run into a single GENERALIZED experience, store it, and
        run the feedback loop for any memories that were retrieved this mission. This
        fires on BOTH a solve and a genuine failure/stall: a failed run is still
        experience (what was tried, why it stalled) that future missions use to avoid
        dead ends. The experience layer is the one learning entry point; a *proven*
        experience is promoted into the Playbook Vault by ExperienceMemory itself (§10)
        — a failed one never is (it cannot meet the promotion bar). Deterministic +
        non-fatal — a learning failure can never break run completion.
        """
        try:
            flag = board.flag_captured or ""
            # A failed run only teaches something if it actually attempted work; storing
            # an empty no-op failure would be noise (and violates the no-demo-data rule).
            if outcome != "success" and not (getattr(board, "execution_history", None) or []):
                return
            record = experience_extractor.extract_from_board(board, flag=flag, outcome=outcome)
            exp_id = experience_memory.store(record)
            if exp_id:
                _append_to_challenge_log(
                    board.challenge_id, "orchestrator",
                    f"[MEMORY] Learned {outcome} experience {exp_id}: '{record.technique}' "
                    f"({len(record.successful_attack_chain)}-step chain, "
                    f"{len(record.failed_techniques)} failed approaches recorded)")
                try:
                    await ws_manager.broadcast({
                        "event": "MEMORY_LEARNED", "challenge_id": board.challenge_id,
                        "run_id": board.run_id, "experience_id": exp_id,
                        "technique": record.technique, "outcome": outcome,
                    })
                except Exception:
                    pass
            # Feedback loop (§12): memories retrieved this mission are reinforced by the
            # outcome — positively on a solve, negatively on a run that did not capture
            # the flag, so a memory that led nowhere loses confidence. Failure is
            # CONTEXTUAL, not a blacklist: success_rate/confidence decay but the memory
            # is never deleted or blocked (Part 8).
            solved = (outcome == "success")
            for mid in board.retrieved_memory_ids:
                try:
                    experience_memory.record_feedback(
                        mid, success=solved,
                        note=("Retrieved during a solved run" if solved
                              else "Retrieved during a run that did not capture the flag"),
                        run_id=board.run_id, challenge_id=board.challenge_id)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[SwarmOrchestrator] Experience learning failed (non-fatal): {e}")

# Global Singleton
swarm_orchestrator = SwarmOrchestrator()
