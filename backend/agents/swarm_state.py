"""FORGE swarm shared state: ``SwarmTask`` and ``SwarmBlackboard``.

``SwarmBlackboard`` is the shared in-memory (and persisted) state every parallel
swarm worker reads from and writes to.

Relocated verbatim from ``backend/agents/swarm_orchestrator.py`` — no behavior
change; the orchestrator re-imports these names.
"""

import asyncio
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from backend.database.session import SessionLocal
from backend.database.models import ChallengeModel, ToolExecutionModel
from backend.websocket.manager import ws_manager
from backend.config import settings
from backend.agent_runtime.verifier import (
    AnswerResolver, AnswerCandidate, AnswerStatus, VerifierAgent,
)
from backend.agents.response_profiler import ResponseProfiler
from backend.agents.checkpoint_pipeline import AgentCheckpointRecord
from backend.recon.web_forms import describe_form
from backend.agents.swarm_helpers import (
    _append_to_challenge_log,
    _is_meaningful_header,
)


logger = logging.getLogger("forge.swarm")


class SwarmTask:
    def __init__(self, task_id: str, category: str, description: str, priority: int = 1, metadata: Optional[Dict] = None):
        self.task_id = task_id
        self.category = category # RECON, CODE_AUDIT, CRYPTO_DECODE, EXPLOIT, PWN
        self.description = description
        self.priority = priority
        self.metadata = metadata or {}
        self.claimed_by: Optional[str] = None
        self.status = "PENDING" # PENDING, CLAIMED, COMPLETED, FAILED
        self.result: Optional[str] = None

class SwarmBlackboard:
    """Shared in-memory and synchronized state across all parallel swarm workers."""
    def __init__(
        self,
        challenge_id: str,
        run_id: str,
        target_scope: str,
        *,
        description: str = "",
        category: str = "WEB",
        difficulty: str = "EASY",
        challenge_name: str = "",
        flag_pattern: str = "",
    ):
        self.challenge_id = challenge_id
        self.run_id = run_id
        self.target_scope = target_scope
        self.discovered_endpoints: Set[str] = set()
        self.extracted_headers: Dict[str, str] = {}
        self.observed_cookies: Dict[str, str] = {}
        # Forms read off the target's own HTML (action/method/field names). Field
        # names are FACTS from the markup, never guessed by an agent: an invented
        # field name silently no-ops against a form that redirects regardless.
        self.observed_forms: List[Dict[str, Any]] = []
        self.deobfuscated_secrets: List[Dict[str, str]] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.executed_commands_dedup: Set[str] = set()
        self.recon_cache: Dict[str, str] = {}
        self.failure_signatures: Dict[str, int] = {}
        self.blocked_failure_sigs: Set[str] = set()
        self.blocked_capabilities: Set[str] = set()
        self.task_pool: Dict[str, SwarmTask] = {}
        self.flag_captured: Optional[str] = None
        self.flag_candidates: List[Dict[str, str]] = []  # Unverified candidates
        self.flag_event = asyncio.Event()
        self.is_stopped = False
        # Set true only by an operator Pause so run_swarm finalizes as PAUSED
        # (resumable) rather than FAILED. Distinct from is_stopped, which also
        # trips on flag capture / kill switch.
        self.pause_requested = False
        # ── Strategy-exhaustion tracking (PRIMARY fix for "No FA" budget burn) ──
        # Keyed by strategy label (snake_case, from the agent's STRATEGY: line).
        # Shared across ALL agents on the blackboard — agent_1's stale result
        # counts against agent_3's budget for the same strategy class.
        self.strategy_attempts: Dict[str, int] = {}           # total turns per label
        self.strategy_stale_counts: Dict[str, int] = {}       # consecutive stale turns
        self.strategy_evidence_fingerprints: Dict[str, Set[str]] = {}  # distinct evidence hashes
        self.exhausted_strategies: Set[str] = set()           # labels that hit the limit
        self.pivot_directive: str = ""                        # injected after a forced pivot
        self.pivot_reviews_in_flight: Set[str] = set()        # labels currently being reviewed
        self.reviewed_pivot_strategies: Set[str] = set()      # labels whose pivot review has completed
        # Stall / refill / persistence state
        self.stall_reason: Optional[str] = None
        self.worker_states: Dict[str, Dict[str, Any]] = {}
        self.refill_count = 0
        self.max_refills = 6
        self.refresh_interval = 15  # seconds between task-pool refill checks
        self.last_activity_ts = time.time()
        self.last_persist_ts = 0.0
        self.persist_interval = 3.0
        self.started_ts = time.time()
        # Signatures ("name: value") of header injections already queued, so the
        # same lead is never re-added to the task pool (kills the injection loop).
        self.tried_header_signatures: Set[str] = set()
        # Candidate Entities Tracking (Usernames, tokens, and decode deduplication)
        self.candidate_usernames: Set[str] = set()
        self.candidate_tokens: Set[str] = set()
        self.tested_user_combinations: Set[str] = set()
        self.processed_decode_hashes: Set[str] = set()
        self.last_target_rejection: Optional[str] = None
        self._lock = asyncio.Lock()
        self._pivot_lock = asyncio.Lock()

        # ── Experience memory (retrieved ONCE per mission, shared by all agents) ──
        # memory_context is a compact, reference-only prompt block; retrieved_memory_ids
        # feeds the post-solve feedback loop (§6, §8, §12).
        self.memory_context: str = ""
        self.retrieved_memory_ids: List[str] = []

        # ── Unified flexible-agent challenge context ────────────────────────────
        # Populated by run_swarm before agents start; baked into every agent's
        # full-context prompt via build_agent_prompt(). No per-role framing.
        self.challenge_name: str = challenge_name
        self.platform: str = ""
        self.category: str = category or "WEB"
        self.difficulty: str = difficulty or "EASY"
        self.description: str = description
        self.flag_pattern: str = flag_pattern or getattr(settings, "DEFAULT_FLAG_PATTERNS",
                                                         "picoCTF{...}|FLAG{...}|flag{...}|HTB{...}|CTF{...}")
        self.max_iterations: int = getattr(settings, "AGENT_MAX_ITERATIONS", 40)
        self.max_minutes: int = getattr(settings, "AGENT_MAX_MINUTES", 30)
        # Per-run token budget (A5): cumulative prompt+completion tokens across the run.
        # 0 = unlimited. When exceeded the run winds down cleanly via stall_reason.
        self.max_tokens: int = int(getattr(settings, "AGENT_MAX_TOKENS", 0) or 0)
        self.tokens_used: int = 0
        self.attached_file_paths: List[str] = []
        self.artifact_classification = None            # ClassificationResult | None
        self.env_info: Dict[str, Any] = {}
        self.answer_resolver = AnswerResolver()
        router = None
        try:
            from backend.providers.router import model_router
            router = model_router
        except Exception:
            pass
        self.verifier_agent = VerifierAgent(resolver=self.answer_resolver, router=router)

        # ── Encoded-artifact reconstruction & escalation (deterministic) ────────
        # When a tool output or attached artifact turns out to be an ENCODED file
        # (e.g. a wall of ASCII 0/1 that is really a JPEG), the reconstructed file is
        # preserved here as evidence and escalated for analysis — so the swarm never
        # concludes "no flag" while a rebuilt artifact sits un-analyzed.
        self.derived_artifacts: List[Dict[str, Any]] = []   # provenance records
        self.reconstruction_hashes: Set[str] = set()        # output_sha256 dedup
        self.processed_recon_inputs: Set[str] = set()        # input_sha256 fast-skip

        # ── Empirical response profiling, anomaly detection & actionable preemption ──
        self.response_profiler = ResponseProfiler()
        self.actionable_preemptions: List[Dict[str, Any]] = []
        self.seen_candidate_targets: Set[str] = set()

        # ── Per-agent live state (arbitrary N agents, not fixed roles) ──────────
        self.agent_ids: List[str] = []
        self.agent_transcripts: Dict[str, List[str]] = {}   # agent_id -> transcript lines
        self.agent_directives: Dict[str, str] = {}          # agent_id -> injected directive
        self.agent_iterations: Dict[str, int] = {}          # agent_id -> tool-call count
        self.agent_last_model: Dict[str, str] = {}          # agent_id -> last model used (context budgeting)
        self.agent_started_ts: Dict[str, float] = {}        # agent_id -> wall-clock start
        # Wall-clock seconds each agent spent idle at a checkpoint pause. Subtracted from
        # elapsed so a 3.5-min operator pause never counts against a 5-min work budget.
        self.agent_paused_seconds: Dict[str, float] = {}    # agent_id -> accumulated pause secs
        # Consecutive LOCAL execution failures (command never reached the target) per agent.
        # Reset the moment a command actually reaches the target; caps unbounded identical retries.
        self.agent_local_fail_streak: Dict[str, int] = {}   # agent_id -> streak count

        # ── HITL checkpoint state (hard pause & wait) ───────────────────────────
        # checkpoint_pause is a RESUMABLE wait (distinct from is_stopped /
        # pause_requested): agents idle at a checkpoint and resume on operator paste.
        self.checkpoint_pause = False
        self.checkpoint_active = False
        self.checkpoint_response_event = asyncio.Event()
        self.latest_pasted_response: Optional[str] = None
        self.cycle_n = 0
        self.last_checkpoint_report: str = ""
        self.instance_expiry_ts: Optional[float] = None     # epoch secs, or None
        self.cycle_window_start_ts: float = time.time()     # start of the current report window

        # ── Command Privilege Approvals (Per-Request HITL Gate) ────────────────
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        # Preserved intent for capability gaps (privilege denials)
        self.capability_gaps: List[Dict[str, Any]] = []

    async def add_task(self, category: str, description: str, priority: int = 1, metadata: Optional[Dict] = None) -> SwarmTask:
        async with self._lock:
            # Check for duplicate description in pending/claimed tasks
            for t in self.task_pool.values():
                if t.description == description and t.status in ["PENDING", "CLAIMED"]:
                    return t
            t_id = f"task_{uuid.uuid4().hex[:8]}"
            task = SwarmTask(t_id, category, description, priority, metadata)
            self.task_pool[t_id] = task
            await self._broadcast_blackboard()
            return task

    async def claim_task(self, worker_id: str, categories: List[str]) -> Optional[SwarmTask]:
        async with self._lock:
            # Sort by priority descending
            pending = [t for t in self.task_pool.values() if t.status == "PENDING" and t.category in categories]
            if not pending:
                return None
            pending.sort(key=lambda t: t.priority, reverse=True)
            chosen = pending[0]
            chosen.status = "CLAIMED"
            chosen.claimed_by = worker_id
            await self._broadcast_blackboard()
            return chosen

    async def complete_task(self, task_id: str, result: str, discoveries: Optional[Dict] = None):
        async with self._lock:
            self.last_activity_ts = time.time()
            task = self.task_pool.get(task_id)
            if task:
                task.status = "COMPLETED"
                task.result = result

            if discoveries:
                for ep in discoveries.get("endpoints", []):
                    if ep not in self.discovered_endpoints:
                        self.discovered_endpoints.add(ep)
                        asyncio.create_task(ws_manager.broadcast({
                            "event": "EVIDENCE_CAPTURED",
                            "challenge_id": self.challenge_id,
                            "evidence_id": f"ev-{uuid.uuid4().hex[:8]}",
                            "type": "endpoint_discovered",
                            "source": "swarm_recon",
                            "description": f"Discovered Endpoint: {ep}"
                        }))
                for k, v in discoveries.get("headers", {}).items():
                    self.extracted_headers[k] = v
                for k, v in discoveries.get("cookies", {}).items():
                    self.observed_cookies[k] = v
                for sec in discoveries.get("secrets", []):
                    self.deobfuscated_secrets.append(sec)

            await self._broadcast_blackboard()

    async def record_flag(self, flag: str, worker_id: str):
        async with self._lock:
            if not self.flag_captured:
                self.flag_captured = flag
                self.last_activity_ts = time.time()
                self.flag_event.set()
                logger.info(f"[SwarmBlackboard] 🚩 FLAG CAPTURED BY WORKER {worker_id}: {flag}")
                _append_to_challenge_log(self.challenge_id, worker_id, f"🚩 FLAG CAPTURED: {flag}")
                await ws_manager.broadcast({
                    "event": "FLAG_CAPTURED",
                    "challenge_id": self.challenge_id,
                    "run_id": self.run_id,
                    "flag": flag,
                    "solver_worker": worker_id
                })

    async def record_flag_candidate(
        self,
        candidate: str,
        worker_id: str,
        source: str,
        evidence: Optional[Dict[str, Any]] = None,
        command: str = "",
        action_succeeded: bool = True,
    ):
        """Record an answer candidate, resolve against challenge semantics, and promote if verified/resolved."""
        # Phase 1 (lock): read board fields to build candidate object (fast, no I/O)
        async with self._lock:
            task_context = {
                "challenge_id": self.challenge_id,
                "challenge_name": self.challenge_name,
                "category": self.category,
                "difficulty": self.difficulty,
                "description": self.description,
                "flag_pattern": self.flag_pattern,
                "target_scope": self.target_scope,
            }
            cand_obj = AnswerCandidate(
                value=candidate,
                source=self.answer_resolver.normalize_source(source),
                worker_id=worker_id,
                evidence=evidence or {},
                task_context=task_context,
                provenance={"worker_id": worker_id, "command": command, "action_succeeded": action_succeeded},
            )

        # Phase 2 (unlocked): invoke Agent #4 async LLM-assisted verification.
        # Lock is released so other workers are not blocked during the LLM call.
        verdict = await self.verifier_agent.verify(cand_obj)

        # Phase 3 (lock): record the verdict (dedup, append, broadcast)
        async with self._lock:
            if verdict.status == AnswerStatus.REJECTED:
                logger.info(f"[SwarmBlackboard] Rejected false/invalid candidate: {candidate} (reasons: {verdict.reasons})")
                _append_to_challenge_log(self.challenge_id, worker_id, f"⚠ Rejected candidate: {candidate} ({'; '.join(verdict.reasons)})")
                return

            # Check if this candidate was already seen
            for existing in self.flag_candidates:
                if existing.get("flag") == verdict.candidate:
                    return

            self.flag_candidates.append({
                "flag": verdict.candidate,
                "worker": worker_id,
                "source": source,
                "status": verdict.status.value,
                "confidence": verdict.confidence,
                "reasons": verdict.reasons,
                "evidence": evidence or {},
            })
            _append_to_challenge_log(
                self.challenge_id, worker_id,
                f"🏳 Answer candidate from {source}: {verdict.candidate} [{verdict.status.value}, conf={verdict.confidence:.2f}]"
            )
            logger.info(f"[SwarmBlackboard] Answer candidate from {source} by {worker_id}: {verdict.candidate} [{verdict.status.value}]")

            await ws_manager.broadcast({
                "event": "FLAG_CANDIDATE",
                "challenge_id": self.challenge_id,
                "run_id": self.run_id,
                "candidate": verdict.candidate,
                "source": source,
                "worker": worker_id,
                "status": verdict.status.value,
                "confidence": verdict.confidence,
            })

        # If resolved or verified (e.g. from tool_output, vision_read, reconstructed_artifact, etc.):
        if verdict.is_verified or verdict.is_resolved:
            logger.info(f"[SwarmBlackboard] 🎯 Candidate verified/resolved: {verdict.candidate} via {source}")
            await self.record_flag(verdict.candidate, worker_id)

    def record_capability_gap(self, intent: Dict[str, Any]) -> None:
        """Preserve original intent when an action/capability is denied for privilege reasons."""
        if isinstance(intent, dict):
            self.capability_gaps.append(dict(intent))

    async def register_derived_artifact(self, outcome, derived_path: Optional[str],
                                        worker_id: str) -> bool:
        """Preserve a reconstructed artifact as shared evidence and escalate it.

        Deduplicates by the reconstructed bytes' sha256 so the same artifact is only
        registered once. Appends a persisted derived FILE to ``attached_file_paths`` so
        every subsequent agent sees it as an analyzable input. Broadcasts an
        ENCODED_ARTIFACT_RECONSTRUCTED event for UI observability (requirement #16).
        Returns True if newly registered.
        """
        if outcome is None:
            return False
        out_hash = getattr(outcome, "output_sha256", "") or ""
        async with self._lock:
            if out_hash and out_hash in self.reconstruction_hashes:
                return False
            if out_hash:
                self.reconstruction_hashes.add(out_hash)
            record = outcome.to_evidence_dict()
            record["derived_path"] = derived_path
            record["analyzed"] = False
            record["worker"] = worker_id
            self.derived_artifacts.append(record)
            # A persisted derived FILE becomes a first-class analyzable input.
            if derived_path and derived_path not in self.attached_file_paths:
                self.attached_file_paths.append(derived_path)
        _append_to_challenge_log(
            self.challenge_id, worker_id,
            f"🧩 Reconstructed {outcome.artifact_type} ({outcome.byte_count} bytes) "
            f"from {outcome.scheme} → {os.path.basename(derived_path) if derived_path else 'in-memory'} "
            f"[{outcome.state}]")
        logger.info("[SwarmBlackboard] Derived artifact registered by %s: %s (%s, %d bytes) path=%s",
                    worker_id, outcome.artifact_type, outcome.scheme, outcome.byte_count, derived_path)
        try:
            await ws_manager.broadcast({
                "event": "ENCODED_ARTIFACT_RECONSTRUCTED",
                "challenge_id": self.challenge_id,
                "run_id": self.run_id,
                "worker": worker_id,
                "scheme": outcome.scheme,
                "artifact_type": outcome.artifact_type,
                "byte_count": outcome.byte_count,
                "state": outcome.state,
                "derived_path": derived_path,
                "recommended_tools": list(outcome.recommended_tools),
            })
        except Exception:
            pass
        return True

    def has_unanalyzed_derived(self) -> bool:
        """True if a reconstructed FILE artifact still needs analysis.

        Used by the finish-without-flag path so the swarm does not terminate with
        "no flag" while a rebuilt artifact remains un-analyzed (requirement #11/#12/#14).
        """
        return any(d.get("derived_path") and not d.get("analyzed")
                   for d in self.derived_artifacts)

    def mark_derived_analyzed(self, derived_path: str):
        for d in self.derived_artifacts:
            if d.get("derived_path") == derived_path:
                d["analyzed"] = True

    async def note_exploit_header(self, name: str, value: str, worker_id: str) -> bool:
        """Validate a candidate exploit header and, if genuinely new and concrete,
        record it and queue a single injection task.

        Returns True only if an injection task was queued. Rejects LLM-placeholder
        shapes and de-dups by "name: value" signature, so a header can never spawn
        the same injection task twice (the root cause of the observed loop).
        """
        name = (name or "").strip()
        value = (value or "").strip().strip('"\'')
        # Cut trailing prose the model appends after a concrete value, e.g.
        # "127.0.0.1 (for bypassing IP restrictions)" -> "127.0.0.1".
        value = re.split(r"\s\(|\s--\s|\s//\s|\s#\s|\s{2,}", value, maxsplit=1)[0].strip().strip('"\'')
        if not _is_meaningful_header(name, value):
            _append_to_challenge_log(self.challenge_id, worker_id, f"⚠ Ignored non-actionable header suggestion: {name}: {value}")
            return False

        signature = f"{name.lower()}: {value}"
        async with self._lock:
            if signature in self.tried_header_signatures:
                return False
            self.tried_header_signatures.add(signature)
            self.extracted_headers[name] = value

        _append_to_challenge_log(self.challenge_id, worker_id, f"🔑 Actionable header queued: {name}: {value}")
        logger.info(f"[SwarmBlackboard] Actionable header queued by {worker_id}: {name}: {value}")
        await self.add_task(
            "EXPLOIT",
            f"Inject header '{name}: {value}' into the homepage, login, and API endpoints on {self.target_scope}",
            priority=6,
            metadata={"header_name": name, "header_value": value},
        )
        return True

    def save_snapshot(self) -> Dict[str, Any]:
        """Convenience alias for _build_mission_plan()."""
        return self._build_mission_plan()

    def _build_mission_plan(self) -> Dict[str, Any]:
        """Build the current mission plan for UI + DB persist.

        The claim-and-solve task pool was removed with the fixed-worker model, so
        the 'tasks' rows are now derived from live per-agent state (one row per
        active agent) — the frontend Todo/Mission Plan renders these unchanged.
        """
        plan_tasks = []
        for agent_id in (self.agent_ids or list(self.worker_states.keys())):
            ws_ = self.worker_states.get(agent_id, {})
            raw_status = (ws_.get("status") or "PENDING").upper()
            if self.flag_captured:
                row_status = "COMPLETED"
            elif raw_status in ("RUNNING", "ANALYZING", "EXECUTING"):
                row_status = "IN_PROGRESS"
            else:
                row_status = "PENDING"
            plan_tasks.append({
                "id": agent_id,
                "phase": "AGENT",
                "title": ws_.get("current_task") or "General-purpose CTF agent",
                "tool": "bash",
                "reasoning": f"Model: {ws_.get('selected_model', 'auto')} | iterations: {self.agent_iterations.get(agent_id, 0)}",
                "status": row_status,
                "output_summary": ws_.get("last_result", ""),
            })
        if self.checkpoint_active:
            status = "WAITING_FOR_USER"
        elif self.flag_captured:
            status = "COMPLETED"
        elif self.stall_reason:
            status = "STALLED"
        else:
            status = "IN_PROGRESS"
        return {
            "challenge_id": self.challenge_id,
            "status": status,
            "summary": f"Flexible agent swarm active on {self.target_scope}"
                       + (f" — awaiting operator checkpoint response (cycle {self.cycle_n})" if self.checkpoint_active else "")
                       + (f" — STALLED: {self.stall_reason}" if self.stall_reason else ""),
            "tasks": plan_tasks,
            "strategic_reviews": [],
            "stall_reason": self.stall_reason or "",
            # Resume snapshot — rehydrated by run_swarm(resume=True) so a paused
            # challenge continues aware of prior work instead of repeating it.
            "blackboard_state": {
                # Phase 7 (STEP 2/5) — the target this snapshot was taken against, so a
                # resume can detect a changed target and drop stale host-bound endpoints.
                "target_scope": self.target_scope,
                "discovered_endpoints": list(self.discovered_endpoints),
                "extracted_headers": dict(self.extracted_headers),
                "observed_cookies": dict(self.observed_cookies),
                # Form field names discovered from the target's HTML — durable across a
                # pause/resume so a resumed run keeps the real field names instead of
                # re-guessing them.
                "observed_forms": [dict(f) for f in self.observed_forms[-20:]],
                "deobfuscated_secrets": list(self.deobfuscated_secrets),
                "executed_commands": list(self.executed_commands_dedup),
                "flag_candidates": list(self.flag_candidates),
                "candidate_usernames": list(self.candidate_usernames),
                "agent_directives": dict(self.agent_directives),
                # Full per-agent reasoning history so a resumed run continues each agent's
                # thread instead of restarting recon from scratch (#4). Capped to keep the
                # snapshot small; outputs are already truncated by record_agent_step.
                "agent_transcripts": {aid: lines[-20:] for aid, lines in self.agent_transcripts.items()},
                "execution_history": [
                    {"agent": h.get("agent", ""), "command": h.get("command", ""),
                     "output": (h.get("output") or "")[:800], "ts": h.get("ts", ""), "note": h.get("note", "")}
                    for h in self.execution_history[-60:]
                ],
                "agent_ids": list(self.agent_ids),
                "cycle_n": self.cycle_n,
                # Encoded-artifact reconstruction evidence — so reconstructed derived
                # artifacts (and their analyzed/pending state) survive a pause/resume and
                # re-surface as escalation work instead of being silently forgotten.
                "derived_artifacts": [dict(d) for d in self.derived_artifacts[-40:]],
                "reconstruction_hashes": list(self.reconstruction_hashes),
                "processed_recon_inputs": list(self.processed_recon_inputs)[-200:],
                "recon_cache": dict(self.recon_cache),
                "failure_signatures": dict(self.failure_signatures),
                "blocked_failure_sigs": list(self.blocked_failure_sigs),
                "blocked_capabilities": list(self.blocked_capabilities),
                # Strategy-exhaustion state — survived across pause/resume so a
                # resumed run never re-spends budget on already-exhausted approaches.
                "strategy_attempts": dict(self.strategy_attempts),
                "strategy_stale_counts": dict(self.strategy_stale_counts),
                "strategy_evidence_fingerprints": {
                    k: list(v) for k, v in self.strategy_evidence_fingerprints.items()
                },
                "exhausted_strategies": list(self.exhausted_strategies),
                "pivot_directive": self.pivot_directive,
                "reviewed_pivot_strategies": list(self.reviewed_pivot_strategies),
                "response_profiler": self.response_profiler.to_dict(),
                "actionable_preemptions": [dict(p) for p in self.actionable_preemptions[-30:]],
                "capability_gaps": [dict(g) for g in self.capability_gaps[-50:]],
            },
        }

    def load_snapshot(
        self,
        snapshot: Optional[Dict[str, Any]],
        prior_commands: Optional[List[str]] = None,
        prior_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """Rehydrate blackboard state from a persisted mission_plan snapshot + prior
        tool commands so a resumed swarm continues from where it left off.

        Synchronous by design — call before any worker starts, so no lock/broadcast
        is needed. Returns counts for logging.
        """
        counts = {"endpoints": 0, "headers": 0, "commands": 0, "completed_tasks": 0,
                  "pending_tasks": 0, "transcript_lines": 0}
        snapshot = snapshot or {}

        for g in (snapshot.get("capability_gaps") or []):
            if isinstance(g, dict):
                self.capability_gaps.append(dict(g))

        for k, v in (snapshot.get("recon_cache") or {}).items():
            self.recon_cache[k] = v
        for k, v in (snapshot.get("failure_signatures") or {}).items():
            self.failure_signatures[k] = v
        for sig in (snapshot.get("blocked_failure_sigs") or []):
            self.blocked_failure_sigs.add(sig)
        for cap in (snapshot.get("blocked_capabilities") or []):
            self.blocked_capabilities.add(cap)
        # Strategy-exhaustion state — restore so a resumed run doesn't re-burn budget
        # on already-exhausted approaches (checkpoint-safe requirement).
        for k, v in (snapshot.get("strategy_attempts") or {}).items():
            self.strategy_attempts[k] = int(v)
        for k, v in (snapshot.get("strategy_stale_counts") or {}).items():
            self.strategy_stale_counts[k] = int(v)
        for k, vs in (snapshot.get("strategy_evidence_fingerprints") or {}).items():
            self.strategy_evidence_fingerprints.setdefault(k, set()).update(vs or [])
        for label in (snapshot.get("exhausted_strategies") or []):
            if label:
                self.exhausted_strategies.add(label)
        for label in (snapshot.get("reviewed_pivot_strategies") or []):
            if label:
                self.reviewed_pivot_strategies.add(label)
        self.pivot_directive = snapshot.get("pivot_directive") or ""

        if snapshot.get("response_profiler"):
            self.response_profiler.load_dict(snapshot.get("response_profiler"))
        for p in (snapshot.get("actionable_preemptions") or []):
            if isinstance(p, dict):
                self.actionable_preemptions.append(dict(p))
                targ = p.get("normalized_target") or p.get("raw_value")
                if targ:
                    self.seen_candidate_targets.add(targ)

        for ep in (snapshot.get("discovered_endpoints") or []):
            if ep:
                self.discovered_endpoints.add(ep)
        for k, v in (snapshot.get("extracted_headers") or {}).items():
            self.extracted_headers[k] = v
        for k, v in (snapshot.get("observed_cookies") or {}).items():
            self.observed_cookies[k] = v
        # Absent in snapshots written before forms were captured — an empty default keeps
        # older paused runs resumable.
        for form in (snapshot.get("observed_forms") or []):
            if isinstance(form, dict) and form.get("fields") and form not in self.observed_forms:
                self.observed_forms.append(dict(form))
        for sec in (snapshot.get("deobfuscated_secrets") or []):
            self.deobfuscated_secrets.append(sec)
        for cand in (snapshot.get("flag_candidates") or []):
            if cand not in self.flag_candidates:
                self.flag_candidates.append(cand)

        # Dedup set: prior executed commands from the snapshot + DB tool executions,
        # so workers never re-run a command already tried in an earlier run.
        for cmd in (snapshot.get("executed_commands") or []):
            if cmd:
                self.executed_commands_dedup.add(cmd)
        for cmd in (prior_commands or []):
            if cmd:
                self.executed_commands_dedup.add(cmd)

        # Restore candidate usernames + prior operator directives + checkpoint cycle
        # (the claim-and-solve task pool was removed with the flexible-agent model).
        for u in (snapshot.get("candidate_usernames") or []):
            if u:
                self.candidate_usernames.add(u)
        for aid, directive in (snapshot.get("agent_directives") or {}).items():
            if directive:
                self.agent_directives[aid] = directive

        # Restore each agent's actual reasoning history (#4): per-agent transcripts + the
        # shared execution history, so build_history_context shows "Your recent steps" and
        # agents resume their thread instead of re-running the same initial recon commands.
        for aid, lines in (snapshot.get("agent_transcripts") or {}).items():
            if lines:
                self.agent_transcripts.setdefault(aid, []).extend(list(lines))
        for h in (snapshot.get("execution_history") or []):
            if h:
                self.execution_history.append(h)
        for aid in (snapshot.get("agent_ids") or []):
            if aid and aid not in self.agent_ids:
                self.agent_ids.append(aid)

        try:
            self.cycle_n = int(snapshot.get("cycle_n") or 0)
        except (TypeError, ValueError):
            self.cycle_n = 0

        # Restore encoded-artifact reconstruction evidence so derived artifacts (and
        # whether they were already analyzed) survive a pause/resume: the escalation
        # in build_history_context re-surfaces any still-pending derived artifact, and
        # dedup sets prevent re-reconstructing/re-persisting the same bytes.
        for rec in (snapshot.get("derived_artifacts") or []):
            if not isinstance(rec, dict):
                continue
            self.derived_artifacts.append(dict(rec))
            out_sha = rec.get("output_sha256")
            if out_sha:
                self.reconstruction_hashes.add(out_sha)
            dpath = rec.get("derived_path")
            if dpath and dpath not in self.attached_file_paths:
                self.attached_file_paths.append(dpath)
        for h in (snapshot.get("reconstruction_hashes") or []):
            if h:
                self.reconstruction_hashes.add(h)
        for h in (snapshot.get("processed_recon_inputs") or []):
            if h:
                self.processed_recon_inputs.add(h)

        counts["endpoints"] = len(self.discovered_endpoints)
        counts["headers"] = len(self.extracted_headers)
        counts["commands"] = len(self.executed_commands_dedup)
        counts["transcript_lines"] = sum(len(v) for v in self.agent_transcripts.values())
        return counts

    def reconcile_target(self, persisted_target: str) -> Dict[str, Any]:
        """Phase 7 (STEP 2/5) — reconcile the authoritative target on resume.

        The blackboard is constructed with the CURRENT target (``self.target_scope``);
        this compares it against the target the resumed snapshot was taken against. If
        they differ, the current target stays authoritative (new commands already use
        it) and endpoints rehydrated from the OLD host are dropped so a stale address
        can neither mislead the agents nor be executed against. Non-fatal and
        deterministic. Returns a small summary for logging/broadcast.
        """
        try:
            from backend.swarm.target_reconciliation import reconcile_target, references_stale_host
            recon = reconcile_target(self.target_scope, persisted_target or "")
        except Exception:
            return {"changed": False, "dropped_endpoints": 0}
        if not recon.changed:
            return {"changed": False, "dropped_endpoints": 0}
        before = len(self.discovered_endpoints)
        self.discovered_endpoints = {
            ep for ep in self.discovered_endpoints
            if not references_stale_host(ep, recon.stale_hosts)
        }
        # A rejection signal captured against the old target no longer applies.
        self.last_target_rejection = ""
        return {
            "changed": True,
            "previous": recon.previous,
            "authoritative": recon.authoritative,
            "stale_hosts": recon.stale_hosts,
            "dropped_endpoints": before - len(self.discovered_endpoints),
        }

    def _compute_progress(self) -> int:
        """Progress: 100 on flag capture, else a soft function of agent activity
        (distinct executed commands) capped at 90 — no fixed task pool to measure."""
        if self.flag_captured:
            return 100
        commands = len(self.executed_commands_dedup)
        return min(90, commands * 4)

    def _build_agent_states(self) -> List[Dict[str, Any]]:
        """Live worker fleet state for /api/agents and AGENT_UPDATE events."""
        states = []
        runtime_seconds = int(time.time() - self.started_ts)
        for worker_id, ws_ in self.worker_states.items():
            states.append({
                **ws_,
                "worker_id": worker_id,
                "challenge_id": self.challenge_id,
                "run_id": self.run_id,
                "runtime_seconds": runtime_seconds
            })
        return states

    async def update_worker_state(self, worker_id: str, **fields):
        """Update a worker's live telemetry row (status, current task, counts...)."""
        async with self._lock:
            state = self.worker_states.setdefault(worker_id, {
                "status": "IDLE",
                "current_task": "",
                "current_capability": "",
                "selected_model": "",
                "last_tool": "",
                "last_result": "",
                "commands_run": 0,
                "failures": 0,
                "last_activity": time.time(),
            })
            state.update(fields)
            state["last_activity"] = time.time()
            self.last_activity_ts = time.time()

    async def record_tool_execution(self, worker_id: str, command: str, res, privilege_level: str = "SAFE", approved: bool = True):
        """Persist a swarm tool execution to ToolExecutionModel so terminal history survives restarts/refreshes."""
        try:
            db = SessionLocal()
            try:
                status = "SUCCESS" if res.exit_code == 0 else ("FAILED" if res.status != "TIMEOUT" else "TIMEOUT")
                exec_row = ToolExecutionModel(
                    run_id=self.run_id,
                    agent=worker_id,
                    tool_name=getattr(res, "tool_name", "bash") or "bash",
                    capability="swarm_" + (worker_id or "worker"),
                    command=(command or "")[:2000],
                    privilege_level=privilege_level,
                    approved=approved,
                    status=status,
                    stdout=(getattr(res, "stdout", "") or "")[:4000],
                    stderr=(getattr(res, "stderr", "") or "")[:4000],
                    exit_code=res.exit_code,
                    duration_ms=float(getattr(res, "duration_ms", 0.0) or 0.0)
                )
                db.add(exec_row)
                db.commit()
            except Exception as e:
                logger.debug(f"[SwarmBlackboard] Tool execution persist skip: {e}")
            finally:
                db.close()
        except Exception:
            pass

    def record_agent_step(self, agent_id: str, command: str = "", output: str = "", note: str = ""):
        """Append one agent step to its transcript + shared execution history.

        Feeds both build_history_context() (cross-agent awareness in the prompt)
        and snapshot_agent_records() (the checkpoint report). Verbatim — no
        paraphrasing, so the report's factual fields stay un-fabricated.
        """
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        parts = [f"[{ts}]"]
        if command:
            parts.append(f"CMD: {command}")
        if output:
            parts.append(f"OUT: {output[:600]}")
        if note:
            parts.append(note)
        self.agent_transcripts.setdefault(agent_id, []).append(" ".join(parts))
        self.execution_history.append({
            "agent": agent_id, "command": command,
            "output": (output or "")[:2000], "ts": ts, "note": note,
        })

    def build_history_context(self, agent_id: str) -> str:
        """Shared blackboard state + this agent's and peers' recent steps, injected
        into the full-context prompt each turn (replaces the old task-pool routing)."""
        lines: List[str] = []
        if self.discovered_endpoints:
            lines.append("Discovered endpoints: " + ", ".join(sorted(self.discovered_endpoints)[:15]))
        if self.extracted_headers:
            lines.append("Known / exploit headers: " + "; ".join(f"{k}: {v}" for k, v in list(self.extracted_headers.items())[:10]))
        if self.observed_cookies:
            lines.append("Cookies: " + "; ".join(f"{k}={v}" for k, v in list(self.observed_cookies.items())[:8]))
        if self.observed_forms:
            # Field names are read off the target's own HTML. Sending anything else
            # no-ops silently on a form that redirects regardless of what it received.
            lines.append("Known web forms (USE THESE EXACT FIELD NAMES — a name the form does "
                         "not expose is silently ignored by the target):")
            for form in self.observed_forms[:5]:
                lines.append("  • " + describe_form(form))
        if self.deobfuscated_secrets:
            lines.append("Decoded secrets: " + "; ".join(str(s)[:120] for s in self.deobfuscated_secrets[:6]))
        if self.candidate_usernames:
            lines.append("Candidate usernames discovered (try these before generic 'admin'): " + ", ".join(sorted(self.candidate_usernames)))
        if self.candidate_tokens:
            lines.append("Candidate tokens: " + ", ".join(list(self.candidate_tokens)[:6]))
        if self.last_target_rejection:
            lines.append(f"Last target rejection signal: {self.last_target_rejection} — reconcile this with the evidence above (e.g. a 'user not found' means try a discovered username).")
        if self.flag_candidates:
            lines.append("Unverified flag candidates so far (MUST be reproduced from real output before accepting): "
                         + ", ".join(c.get("flag", "") for c in self.flag_candidates[:5]))
        # Strategy-exhaustion awareness: every agent sees the global ban list and
        # pivot directive so no worker independently rediscovers the dead end.
        if self.exhausted_strategies:
            lines.append(
                "[STRATEGY GATE] Exhausted approaches — do NOT repeat these: "
                + ", ".join(sorted(self.exhausted_strategies))
            )
        if self.pivot_directive and self.pivot_directive != "__pending__":
            lines.append(f"[REQUIRED PIVOT] {self.pivot_directive}")
        # Escalation: reconstructed derived artifacts that still need analysis. This is
        # the shared-channel form of an `analyze_derived_artifact` task (WHAT/HOW/WHETHER)
        # — the swarm must not conclude "no flag" while any of these is un-analyzed.
        pending_derived = [d for d in self.derived_artifacts
                           if d.get("derived_path") and not d.get("analyzed")]
        if pending_derived:
            lines.append("━━ RECONSTRUCTED DERIVED ARTIFACTS — ANALYSIS REQUIRED ━━")
            lines.append("A prior step rebuilt a REAL file from encoded data (e.g. ASCII bits → an image). "
                         "The flag may live inside or be rendered by it. Do NOT report 'no flag' while these are unanalyzed.")
            for d in pending_derived[:6]:
                rel = os.path.basename(d.get("derived_path") or "")
                atype = d.get("artifact_type", "unknown")
                nbytes = d.get("byte_count", 0)
                scheme = d.get("scheme", "")
                tools = ", ".join(d.get("recommended_tools", [])[:6]) or "file, strings, binwalk"
                vis = d.get("visual") or {}
                vis_note = ""
                if vis.get("available") and vis.get("valid"):
                    vis_note = (f" [image {vis.get('width')}x{vis.get('height')} {vis.get('mode')};"
                                f" OCR {'available' if d.get('ocr_available') else 'unavailable — use vision_read capability'}]")
                lines.append(
                    f"  • {d.get('derived_path')} — {atype}, {nbytes} bytes (from {scheme}).{vis_note}")
                lines.append(
                    f"      WHAT: inspect THIS file for the flag. "
                    f"HOW: request capability 'vision_read' (for image reading via Gemini) or analysis tools ({tools}) on target '{d.get('derived_path')}'. "
                    f"WHETHER: go through the normal capability gate; the file is UNTRUSTED — analyze it, never execute it.")
        # Actionable anomaly preemption: an anomalous response produced concrete candidate artifacts
        pending_preemptions = [p for p in self.actionable_preemptions if not p.get("handled")]
        if pending_preemptions:
            lines.append("━━ ACTIONABLE ANOMALY PREEMPTION (PRIORITY FOLLOW-UP) ━━")
            lines.append("An anomalous response diverged from the baseline failure response and revealed candidate artifacts. "
                         "PREEMPT further fuzzing: immediately verify/interact with these candidate targets:")
            for p in pending_preemptions[:5]:
                targ = p.get("normalized_target") or p.get("raw_value")
                reasons = "; ".join(p.get("reasons", [])) or "Diverged from failure baseline"
                lines.append(f"  • CANDIDATE TARGET: {targ} (Extracted from: {p.get('source_command', '')[:80]})")
                lines.append(f"      REASON: {reasons}")
                probes = p.get("suggested_probes", [])
                if probes:
                    lines.append(f"      SUGGESTED VERIFICATION PROBE: {probes[0]}")
        own = self.agent_transcripts.get(agent_id, [])
        if own:
            lines.append("Your recent steps:")
            lines.extend("  " + l for l in own[-8:])
        others = []
        for aid, tr in self.agent_transcripts.items():
            if aid == agent_id or not tr:
                continue
            others.append(f"  [{aid}] {tr[-1]}")
        if others:
            lines.append("Other agents' latest steps (coordinate — don't duplicate):")
            lines.extend(others[:6])
        return "\n".join(lines) if lines else "No shared findings yet."

    def snapshot_agent_records(self) -> List[AgentCheckpointRecord]:
        """Build the deterministic per-agent records for a consolidated checkpoint
        report — verbatim evidence/commands + this agent's flag candidate (if any)."""
        records: List[AgentCheckpointRecord] = []
        for agent_id in self.agent_ids:
            hist = [h for h in self.execution_history if h.get("agent") == agent_id]
            evidence = [h["output"] for h in hist if h.get("output")][-5:]
            tried = [f"{h.get('command', '')} -> {(h.get('output') or '')[:120]}"
                     for h in hist if h.get("command")][-6:]
            fc = fs = None
            for c in reversed(self.flag_candidates):
                if c.get("worker") == agent_id:
                    fc, fs = c.get("flag"), c.get("source")
                    break
            records.append(AgentCheckpointRecord(
                agent_id=agent_id,
                evidence=evidence,
                tried=tried,
                transcript="\n".join(self.agent_transcripts.get(agent_id, [])),
                flag_candidate=fc,
                flag_source=fs,
            ))
        return records

    async def _persist_progress_if_due(self, force: bool = False):
        """Throttled DB persist of mission plan (task statuses) + progress so the UI is reload-safe."""
        now = time.time()
        if not force and (now - self.last_persist_ts) < self.persist_interval:
            return
        self.last_persist_ts = now
        try:
            db = SessionLocal()
            try:
                ch = db.query(ChallengeModel).filter(ChallengeModel.id == self.challenge_id).first()
                if ch:
                    ch.mission_plan = self._build_mission_plan()
                    ch.progress = self._compute_progress()
                    db.commit()
            except Exception as e:
                logger.debug(f"[SwarmBlackboard] Progress persist skip: {e}")
            finally:
                db.close()
        except Exception:
            pass

    async def _broadcast_blackboard(self):
        try:
            total_tasks = len(self.task_pool)
            completed_tasks = sum(1 for t in self.task_pool.values() if t.status == "COMPLETED")
            progress = self._compute_progress()
            mission_plan = self._build_mission_plan()

            # Broadcast plan update so frontend Todo List / Mission Plan renders live
            await ws_manager.broadcast({
                "event": "PLAN_UPDATED",
                "challenge_id": self.challenge_id,
                "run_id": self.run_id,
                "plan": mission_plan
            })

            # Broadcast progress update
            await ws_manager.broadcast({
                "event": "PROGRESS_UPDATED",
                "challenge_id": self.challenge_id,
                "progress": progress
            })

            # Broadcast live worker fleet state
            await ws_manager.broadcast({
                "event": "AGENT_UPDATE",
                "challenge_id": self.challenge_id,
                "run_id": self.run_id,
                "agents": self._build_agent_states()
            })

            # Broadcast swarm blackboard stats
            await ws_manager.broadcast({
                "event": "SWARM_BLACKBOARD_UPDATE",
                "challenge_id": self.challenge_id,
                "run_id": self.run_id,
                "endpoints_count": len(self.discovered_endpoints),
                "headers": self.extracted_headers,
                "secrets_count": len(self.deobfuscated_secrets),
                "pending_tasks": sum(1 for t in self.task_pool.values() if t.status == "PENDING"),
                "completed_tasks": completed_tasks,
                "flag_captured": bool(self.flag_captured),
                "flag_candidates_count": len(self.flag_candidates)
            })

            # Throttled DB persist keeps the todo list / progress reload-safe
            await self._persist_progress_if_due()
        except Exception:
            pass
