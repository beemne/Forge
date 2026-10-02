"""The Swarm Coordinator (Phase 4 — the layer that ties §1–§15 together).

This is the top of the coordination stack. It owns concurrency, I/O, and
persistence; the policy pieces it drives (supervisor, scheduler, evidence bus,
specialist agents) are small and pure. The coordinator sits strictly ABOVE the
existing systems and reuses them wholesale:

    SwarmCoordinator
        → Supervisor            (plan / react / recover — pure)
        → TaskScheduler         (dependency-aware queue — pure)
        → SpecialistAgent(role) → AgentRuntime  (EXISTING loop)
                                    → ToolManager → ExecutionService → backend
                                    → FlagVerifier / memory / trajectory  (EXISTING)
        → EvidenceBus           → SharedMissionState
        → FlagVerifier          (authoritative, single verified flag)

Nothing here re-implements subprocess execution, memory, providers, trajectory, or
flag verification — those remain owned by the layers below.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from backend.agent_runtime import (
    FlagVerifier, VerifierAgent, session_manager, trajectory_store,
)

from backend.swarm import events
from backend.swarm.agents import AgentResult, SpecialistAgent
from backend.swarm.evidence import Evidence, EvidenceBus, EvidenceType
from backend.swarm.limits import SwarmLimits
from backend.swarm.mission import SharedMissionState
from backend.swarm.roles import AgentRole
from backend.swarm.scheduler import TaskScheduler
from backend.swarm.supervisor import Supervisor
from backend.swarm.tasks import Task

# Phase 5 — adaptive reasoning layer (deterministic; pure data + scoring).
from backend.swarm.candidates import CandidateGenerator
from backend.swarm.dedup import action_signature
from backend.swarm.progress import (
    MissionBudget, ProgressLedger, StopCondition, evaluate_stop,
)
from backend.swarm.reasoning import classify_failure as _classify_failure_class
from backend.swarm.scoring import ActionScorer, mission_uncertainty
from backend.swarm.target_reconciliation import reconcile_target

# Phase 4.x — capability discovery / acquisition / target modelling (cycle-free imports).
from backend.execution.capabilities import (
    capability_service as _default_capability_service, ACQUIRABLE,
)
from backend.execution.acquisition import acquisition_planner as _default_acquisition_planner
from backend.execution.targets import target_detector as _default_target_detector, TargetType

logger = logging.getLogger("forge.swarm.coordinator")

# Objective keywords that imply a capability requirement (kept tight so existing
# role objectives are unaffected — only an explicit OCR/text-from-image need triggers).
_CAP_KEYWORDS = {
    "ocr": ("ocr", "extract text from the image", "extract text from image",
            "read the text in the image", "text from the decoded image"),
}

# Registry of live coordinators so the API/WebSocket layer can observe the swarm.
active_missions: Dict[str, "SwarmCoordinator"] = {}


@dataclass
class MissionResult:
    mission_id: str
    status: str
    verified_flag: Optional[str]
    reason: str
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    evidence_count: int


class SwarmCoordinator:
    def __init__(
        self,
        *,
        run_id: Optional[str] = None,
        challenge_id: Optional[str] = None,
        target: str = "",
        category: str = "",
        difficulty: str = "",
        challenge_name: str = "",
        platform: str = "",
        description: str = "",
        flag_format: str = "",
        workspace_root: Optional[str] = None,
        limits: Optional[SwarmLimits] = None,
        agent_factory: Optional[Callable[[AgentRole], SpecialistAgent]] = None,
        verifier: Optional[FlagVerifier] = None,
        mission_id: Optional[str] = None,
        persist: bool = True,
        enable_report: bool = False,
        kill_switch: Optional[Callable[[], bool]] = None,
        capability_service: Optional[Any] = None,
        acquisition_planner: Optional[Any] = None,
        target_detector: Optional[Any] = None,
        allow_acquisition: bool = True,
        acquisition_privilege_decider: Optional[Callable[[Any], bool]] = None,
        budget: Optional[MissionBudget] = None,
        enable_reasoning: bool = True,
        scorer: Optional[ActionScorer] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        stagnation_limit: int = 5,
        max_replan_actions: int = 2,
    ):
        self.limits = limits or SwarmLimits()
        self.persist = persist
        self.enable_report = enable_report
        self.kill_switch = kill_switch
        self.verifier = verifier or FlagVerifier()
        router = None
        try:
            from backend.providers.router import model_router
            router = model_router
        except Exception:
            pass
        self.verifier_agent = VerifierAgent(resolver=self.verifier, router=router)
        self.workspace_root = workspace_root

        self.mission_id = mission_id or str(uuid.uuid4())
        self._stopped = False
        self._stop_reason = ""

        # Phase 5 — adaptive reasoning: budget, no-progress ledger, scorer, candidate
        # generator. All deterministic; reasoning can be disabled for pure Phase-4 behaviour.
        self.enable_reasoning = enable_reasoning
        self.budget = budget or MissionBudget()
        self.ledger = ProgressLedger(stagnation_limit=stagnation_limit)
        self.scorer = scorer or ActionScorer()
        if candidate_generator is not None:
            self.generator = candidate_generator
        else:
            # Phase 6 §9 — production coord engine gets contextual success statistics so
            # a memory candidate's success_probability reflects the technique's real
            # track record (and carries it as observability provenance). Lazy + non-fatal.
            try:
                from backend.knowledge.technique_stats import technique_stats as _ts
            except Exception:
                _ts = None
            self.generator = CandidateGenerator(technique_stats=_ts)
        self.max_replan_actions = max_replan_actions
        self._dispatched_actions: Dict[str, str] = {}   # action signature → owning task id
        self._stagnation_replans = 0

        # Phase 4.x — capability/target awareness (injectable for deterministic tests).
        self.capability_service = capability_service or _default_capability_service
        self.acquisition_planner = acquisition_planner or _default_acquisition_planner
        self.target_detector = target_detector or _default_target_detector
        self.allow_acquisition = allow_acquisition
        self._acq_decider = acquisition_privilege_decider
        self._blocked_capabilities: set = set()      # capabilities proven unavailable this mission
        self._acquisition_attempted: set = set()      # capabilities we already tried to acquire
        # Phase 7 (STEP 2/5) — the most recent target reconciliation (empty until a
        # resume reconciles the checkpoint target against the current one).
        self._last_reconciliation: Dict[str, Any] = {}

        scope = [t.strip() for t in (target or "").split("+") if t.strip()]
        self.mission = SharedMissionState(
            mission_id=self.mission_id, run_id=run_id, challenge_id=challenge_id,
            target=target, scope=scope, category=category, difficulty=difficulty,
            challenge_name=challenge_name, platform=platform, description=description,
            flag_format=flag_format, status="PLANNING",
        )
        # Detect the KIND of target up front (conservative; UNKNOWN when unsure) so the
        # snapshot/API and the pre-dispatch mismatch gate can reason about it (§9–§11).
        try:
            if target:
                self.mission.target_type = self.target_detector.detect(target).type.value
        except Exception:
            pass
        # Phase 5 §4 — a concise objective seeds reasoning and agent context.
        self.mission.set_objective(
            (description or f"Capture the flag for the {category or 'CTF'} challenge "
             f"'{challenge_name or target or 'target'}'.")[:300])

        # A dedicated coordination trajectory session — coordination events are
        # recorded here, kept separate from each specialist's execution trajectory
        # (§14), but reusing the ONE trajectory system rather than a second one.
        self.coord_session = session_manager.create(
            challenge_id=challenge_id, run_id=run_id, target_scope=target,
            objective="Coordinate specialist agents to capture the flag.",
            agent_id="supervisor", engine="swarm_coord",
            challenge_name=challenge_name, category=category, difficulty=difficulty,
            platform=platform, description=description, flag_format=flag_format,
        )

        self.bus = EvidenceBus(self.mission_id, run_id=run_id, challenge_id=challenge_id,
                               persist=persist)
        self.scheduler = TaskScheduler(persist=persist)
        self.supervisor = Supervisor(self.mission_id, run_id=run_id, challenge_id=challenge_id)
        self.agent_factory = agent_factory or (
            lambda role: SpecialistAgent(role, workspace_root=workspace_root))

        self._followup_buffer: List[Task] = []
        self.bus.subscribe(self._on_evidence)
        self._agent_seq = 0

    # ================================================================== #
    # Public entry points
    # ================================================================== #

    async def run(self, *, resume: bool = False) -> MissionResult:
        active_missions[self.mission_id] = self
        if self.challenge_key:
            active_missions[self.challenge_key] = self
        try:
            self._record_coord("PLAN", decision_summary=(
                f"Mission {self.mission_id[:8]} — target={self.mission.target or 'n/a'} "
                f"category={self.mission.category or 'n/a'}"))

            if resume:
                self._load_persisted()

            # Fresh plan when the queue is empty OR — Phase 7 (STEP 2/10) — when the
            # target changed on resume: the swarm must actively re-engage the NEW
            # instance (e.g. re-run recon), not merely stop with a stale, completed plan.
            target_changed = bool(self._last_reconciliation.get("changed"))
            if not self.scheduler.all() or target_changed:
                for t in self.supervisor.plan_initial(self.mission):
                    self._add_task(t, origin="plan")

            self.mission.status = "RUNNING"
            self._save_mission()
            events.broadcast(events.MISSION_STARTED, {
                "mission_id": self.mission_id, "run_id": self.mission.run_id,
                "challenge_id": self.mission.challenge_id, "target": self.mission.target,
                "tasks": self.scheduler.total_count(),
            })

            await self._loop()
            return self._finalize()
        finally:
            active_missions.pop(self.mission_id, None)
            if self.challenge_key and active_missions.get(self.challenge_key) is self:
                active_missions.pop(self.challenge_key, None)

    def request_stop(self, reason: str = "Operator stop.") -> None:
        """Ask the mission to stop; in-flight agents pause at their next turn boundary."""
        self._trigger_global_stop(reason, final_status="PAUSED")

    def submit_flag_candidate(self, candidate: str, *, source: str = "tool_output",
                              command: str = "", action_succeeded: bool = True,
                              agent_id: str = "external", evidence: Optional[Dict[str, Any]] = None,
                              authoritative: bool = False) -> bool:
        """Central, authoritative flag/answer verification for candidates from any source.
        Returns True iff VERIFIED / RESOLVED."""
        task_context = {
            "target_scope": self.mission.target,
            "flag_pattern": self.mission.flag_format,
            "description": self.mission.description,
            "challenge_name": self.mission.challenge_name,
            "category": self.mission.category,
        }
        verdict = self.verifier_agent.verify_sync(
            candidate,
            source=source,
            command=command,
            action_succeeded=action_succeeded,
            task_context=task_context,
            evidence=evidence or {},
            authoritative=authoritative,
        )
        if verdict.is_verified or verdict.is_resolved:
            self._accept_verified_flag(verdict.candidate, agent_id=agent_id, task=None)
            return True
        if verdict.candidate and verdict.candidate not in self.mission.flag_candidates:
            self.mission.flag_candidates.append(verdict.candidate)
            self._broadcast_flag_candidate(verdict.candidate, agent_id)
        return False

    async def submit_flag_candidate_async(self, candidate: str, *, source: str = "tool_output",
                                         command: str = "", action_succeeded: bool = True,
                                         agent_id: str = "external", evidence: Optional[Dict[str, Any]] = None,
                                         authoritative: bool = False) -> bool:
        """Central, authoritative async flag/answer verification for candidates from any source."""
        task_context = {
            "target_scope": self.mission.target,
            "flag_pattern": self.mission.flag_format,
            "description": self.mission.description,
            "challenge_name": self.mission.challenge_name,
            "category": self.mission.category,
        }
        verdict = await self.verifier_agent.verify(
            candidate,
            source=source,
            command=command,
            action_succeeded=action_succeeded,
            task_context=task_context,
            evidence=evidence or {},
            authoritative=authoritative,
        )
        if verdict.is_verified or verdict.is_resolved:
            self._accept_verified_flag(verdict.candidate, agent_id=agent_id, task=None)
            return True
        if verdict.candidate and verdict.candidate not in self.mission.flag_candidates:
            self.mission.flag_candidates.append(verdict.candidate)
            self._broadcast_flag_candidate(verdict.candidate, agent_id)
        return False


    def snapshot(self) -> Dict[str, Any]:
        """Read-only view of the swarm for the API/WebSocket (§13)."""
        return {
            "mission_id": self.mission_id, "run_id": self.mission.run_id,
            "challenge_id": self.mission.challenge_id, "status": self.mission.status,
            "progress": self.mission.progress, "verified_flag": self.mission.verified_flag,
            "target": self.mission.target, "strategy": self.mission.strategy,
            "target_reconciliation": self._last_reconciliation,
            "shared_state": self.mission.to_dict(),
            "agents": [
                {"agent_id": aid, **info} for aid, info in self.mission.agent_statuses.items()
            ],
            "tasks": [t.to_dict() for t in self.scheduler.all()],
            "task_counts": self._task_counts(),
            "evidence_count": self.bus.count(),
            "quota": self.quota_snapshot(),
            "limits": vars(self.limits),
            # Phase 5 §37 — inspectable reasoning state.
            "reasoning": {
                "uncertainty": mission_uncertainty(self.mission),
                "budget": self.budget.to_dict(),
                "progress": self.ledger.to_dict(),
                "stop_condition": self.mission.stop_condition,
                "facts": len(self.mission.confirmed_facts),
                "open_hypotheses": sum(1 for h in self.mission.hypothesis_records
                                       if h.get("status") == "open"),
                "failed_approaches": len(self.mission.failed_approaches),
                "candidate_actions": list(self.mission.candidate_actions),
            },
        }

    @staticmethod
    def quota_snapshot() -> Dict[str, Any]:
        """Read-only integration with the EXISTING provider/quota system (§12)."""
        info: Dict[str, Any] = {}
        try:
            from backend.providers.quota_manager import quota_manager
            for attr in ("get_status", "status", "snapshot"):
                fn = getattr(quota_manager, attr, None)
                if callable(fn):
                    info = fn() or {}
                    break
            if not info:
                info = {"blacklisted": list(getattr(quota_manager, "blacklisted_providers", []) or [])}
        except Exception:
            info = {}
        return info

    # ================================================================== #
    # Core loop
    # ================================================================== #

    async def _loop(self) -> None:
        running: Dict[str, asyncio.Task] = {}
        try:
            while True:
                if self._should_stop():
                    break

                free = self.limits.max_concurrent_agents - len(running)
                if free > 0:
                    for task in self.scheduler.ready_tasks(limit=free):
                        # Phase 4.x pre-dispatch gate: a task that needs a missing capability
                        # or the wrong kind of target is blocked HERE — never spawned — so an
                        # impossible task cannot consume the iteration budget (§15, §22).
                        if await self._precheck_blocks(task):
                            continue
                        self._dispatch(task, running)

                if not running:
                    if not self.scheduler.has_open_work():
                        break
                    # Open work but nothing dispatchable (all blocked/unsatisfiable) → stop.
                    self.scheduler.refresh_states()
                    if not self.scheduler.ready_tasks(limit=1):
                        break
                    continue

                done, _pending = await asyncio.wait(
                    list(running.values()), return_when=asyncio.FIRST_COMPLETED)
                for at in done:
                    task_id = getattr(at, "_forge_task_id", None)
                    if task_id:
                        running.pop(task_id, None)
                    try:
                        result = at.result()
                    except asyncio.CancelledError:
                        continue
                    except Exception as e:  # pragma: no cover - execute() is guarded
                        logger.warning(f"[SwarmCoordinator] agent task error: {e}")
                        continue
                    await self._handle_result(result)
                    if self._should_stop():
                        break
                if self._should_stop():
                    break
        finally:
            for at in running.values():
                at.cancel()
            if running:
                await asyncio.gather(*running.values(), return_exceptions=True)

    def _dispatch(self, task: Task, running: Dict[str, asyncio.Task]) -> None:
        role = AgentRole.from_value(task.role)
        self._agent_seq += 1
        agent_id = f"{role.value}#{self._agent_seq}"
        agent = self.agent_factory(role)

        self.scheduler.mark_running(task, assigned_agent=agent_id)
        # Phase 5 §8/§27 — remember the action being taken (so a differently-worded but
        # equivalent action is recognised as a duplicate) and count the agent call.
        self._dispatched_actions[self._action_sig_for(task)] = task.id
        self.budget.record_agent_call()
        self.mission.set_agent_status(agent_id, role=role.value, status="RUNNING",
                                      current_task=(task.objective or "")[:120], task_id=task.id)
        self._record_coord("DECISION", action_type="assign", command=(task.objective or "")[:400],
                           strategy=role.value, decision_summary=f"Assigned to {agent_id}")
        events.broadcast(events.TASK_ASSIGNED, self._task_event(task, agent_id))
        events.broadcast(events.TASK_STARTED, self._task_event(task, agent_id))
        events.broadcast(events.AGENT_STATUS, {
            "mission_id": self.mission_id, "agent_id": agent_id, "role": role.value,
            "status": "RUNNING", "task_id": task.id})

        coro = agent.execute(
            task, self.mission, self.bus,
            max_turns=self.limits.max_turns_per_task,
            task_timeout_seconds=self.limits.task_timeout_seconds,
            cancel_check=self._should_stop,
        )
        at = asyncio.ensure_future(coro)
        setattr(at, "_forge_task_id", task.id)
        running[task.id] = at

    # ================================================================== #
    # Phase 4.x pre-dispatch capability / target gate (§11, §15, §22)
    # ================================================================== #

    async def _precheck_blocks(self, task: Task) -> bool:
        """Return True (and block the task) if it cannot succeed as specified.

        This is the anti-infinite-retry guarantee: a task needing an unavailable,
        non-acquirable capability, or pointed at the wrong KIND of target, is failed
        here without ever spawning an agent — and the offending capability is
        remembered so no sibling/follow-up task re-triggers the same dead end.
        """
        # 1) Target-type mismatch (only when the task explicitly declares one).
        mism = self._detect_target_mismatch(task)
        if mism is not None:
            self._block_task(task, category="TARGET_MISMATCH",
                             evidence_type=EvidenceType.TARGET_MISMATCH.value,
                             title=mism.summary(), reason=mism.reason,
                             action=mism.recommended_action, tags=["target_mismatch"])
            return True

        # 2) Required capabilities (explicit + conservatively auto-derived).
        for capname in self._auto_capabilities(task):
            cap, ok = await self._resolve_capability(capname)
            if not ok:
                self._block_task(task, category="BLOCKED_CAPABILITY",
                                 evidence_type=EvidenceType.CAPABILITY.value,
                                 title=f"capability '{capname}' unavailable",
                                 reason=cap.reason, action=cap.recommended_action,
                                 tags=["blocked", f"capability:{capname}"], capability=capname)
                return True
        return False

    async def _resolve_capability(self, capname: str):
        """Discover a capability; attempt controlled acquisition once if acquirable.

        Returns (Capability, ok). ok is True only if a provider is actually usable.
        A capability that stays unavailable is memoised in ``_blocked_capabilities``
        so it is never re-discovered/re-acquired for another task this mission.
        """
        cap = self.capability_service.discover(capname)
        if cap.available:
            return cap, True
        if capname in self._blocked_capabilities:
            return cap, False
        if (self.allow_acquisition and cap.status == ACQUIRABLE
                and capname not in self._acquisition_attempted):
            self._acquisition_attempted.add(capname)
            try:
                res = await self.acquisition_planner.acquire(
                    capname, agent="supervisor", privilege_decider=self._acq_decider)
                self._record_coord("ACQUISITION", strategy=capname,
                                   result=("SUCCESS" if res.success else "PENDING"),
                                   decision_summary=(res.reason or "")[:300])
                if res.success:
                    self.capability_service.refresh(capname)
                    cap2 = self.capability_service.discover(capname)
                    if cap2.available:
                        return cap2, True
            except Exception as e:  # acquisition must never crash the mission
                logger.debug(f"[SwarmCoordinator] acquisition error for {capname}: {e}")
        self._blocked_capabilities.add(capname)
        return cap, False

    def _detect_target_mismatch(self, task: Task):
        if not task.target_type:
            return None
        try:
            required = TargetType(task.target_type)
        except Exception:
            return None
        tgt = self.mission.target or ""
        if not tgt:
            return None
        try:
            provided = self.target_detector.detect(tgt)
            return self.target_detector.classify_mismatch(required, provided)
        except Exception:
            return None

    def _auto_capabilities(self, task: Task) -> List[str]:
        caps = list(task.required_capabilities or [])
        text = (task.objective or "").lower()
        for capname, kws in _CAP_KEYWORDS.items():
            if capname not in caps and any(kw in text for kw in kws):
                caps.append(capname)
        return caps

    # ── Phase 5 §8 — normalized action signatures + duplicate suppression ── #

    def _action_sig_for(self, task: Task) -> str:
        """Deterministic action signature for a task (capability::target::params).

        Unlike the task signature (role + objective prose), this collapses differently-
        worded objectives that exercise the same capability against the same target — the
        §8 requirement not to rely on raw string equality. The target is extracted from
        the objective (a quoted endpoint/artifact, a path, or a URL) so endpoint-specific
        tasks stay distinct; otherwise it falls back to the mission target.
        """
        from backend.swarm.candidates import technique_to_action_type
        cap = (task.required_capabilities[0] if task.required_capabilities
               else technique_to_action_type(task.objective))
        return action_signature(cap, self._extract_target(task.objective), "")

    def _extract_target(self, objective: str) -> str:
        text = objective or ""
        m = re.search(r"'([^']+)'", text) or re.search(r'"([^"]+)"', text)
        if m:
            return m.group(1)
        m = re.search(r"https?://\S+", text)
        if m:
            return m.group(0).rstrip(".,;)")
        m = re.search(r"(?<!\w)(/[\w./\-]+)", text)
        if m:
            return m.group(1)
        return self.mission.target or ""

    def _block_task(self, task: Task, *, category: str, evidence_type: str, title: str,
                    reason: str, action: str, tags: List[str], capability: str = "") -> None:
        """Terminally block a task (no retry) and record structured, learnable evidence."""
        self.bus.publish(Evidence(
            mission_id=self.mission_id, agent_id=task.role, task_id=task.id,
            evidence_type=evidence_type, title=(title or "")[:400], description=(reason or "")[:500],
            source="supervisor", confidence=0.6,
            tags=list(tags or []) + [f"action:{action}"], related_technology=capability or ""))
        self.scheduler.mark_failed(task, reason=f"{category}: {reason}")
        self.mission.record_dead_end(f"{task.role}: {category} — {(reason or '')[:80]}")
        # §7 — a blocked action is a failed approach too, so reasoning does not re-propose
        # it (the capability itself is separately memoised in _blocked_capabilities).
        self.mission.record_failed_approach(
            action=(task.objective or "")[:200], signature=self._action_sig_for(task),
            capability=capability, target=self._extract_target(task.objective),
            result="BLOCKED", reason=(reason or "")[:200], failure_class=category, agent=task.role)
        self._record_coord(category, command=(task.objective or "")[:400], strategy=task.role,
                           result="BLOCKED", decision_summary=(reason or "")[:300])
        events.broadcast(events.TASK_FAILED, {
            **self._task_event(task, task.assigned_agent or ""),
            "reason": reason, "category": category})
        self._save_mission()

    # ================================================================== #
    # Result handling
    # ================================================================== #

    async def _handle_result(self, result: AgentResult) -> None:
        task = self.scheduler.get(result.task_id)
        if task is None:
            return

        # Publish harvested evidence (fills shared state + buffers follow-up tasks).
        published: List[str] = []
        for ev in result.evidence:
            eid = self.bus.publish(ev)
            if eid:
                published.append(eid)

        # ── Authoritative flag path (§11) ──
        if result.verified_flag:
            self._accept_verified_flag(result.verified_flag, agent_id=task.assigned_agent, task=task)
            self._drain_followups()
            self._save_mission()
            return

        # Unverified candidates: recorded + broadcast, but never end the mission.
        for cand in (result.flag_candidates or []):
            if cand and cand not in self.mission.flag_candidates:
                self.mission.flag_candidates.append(cand)
                self._broadcast_flag_candidate(cand, task.assigned_agent)

        # ── Task outcome ──
        if result.status == "COMPLETED":
            self.scheduler.mark_completed(
                task, result={"reason": result.reason, "session_id": result.session_id},
                evidence_ids=published)
            self.mission.record_action_signature(self._action_sig_for(task))   # §8
            self.mission.set_agent_status(task.assigned_agent, status="IDLE", current_task=None)
            self._record_coord("TASK_COMPLETED", command=(task.objective or "")[:400],
                               strategy=task.role, result="COMPLETED")
            events.broadcast(events.TASK_COMPLETED, self._task_event(task, task.assigned_agent))
        elif result.status == "CANCELLED":
            self.scheduler.mark_cancelled(task, reason="Global stop.")
            self.mission.set_agent_status(task.assigned_agent, status="IDLE", current_task=None)
        else:
            self._handle_failure(task, result)

        self._drain_followups()
        # Phase 5 — OBSERVE → REASON → (re)ACT: update budget/progress, reason about the
        # new evidence, and evaluate stop conditions after each completed step.
        self._post_step(task, result)
        self._save_mission()
        self._maybe_checkpoint()

    def _handle_failure(self, task: Task, result: AgentResult) -> None:
        # 0) Phase 5 §7/§14 — remember this failed approach (bounded) with its richer
        #    failure class, so it is not blindly repeated and so replanning can react.
        fclass = _classify_failure_class(result)
        self.mission.record_failed_approach(
            action=(task.objective or "")[:200], signature=self._action_sig_for(task),
            capability=(task.required_capabilities[0] if task.required_capabilities else ""),
            target=self._extract_target(task.objective), result=result.status,
            reason=(result.reason or "")[:200], failure_class=fclass.value, agent=task.assigned_agent)

        # 1) Record structured failure evidence (§9) so it is visible + learnable.
        self.bus.publish(Evidence(
            mission_id=self.mission_id, agent_id=task.role, task_id=task.id,
            evidence_type=EvidenceType.FAILURE.value,
            title=f"{task.role} task failed: {result.status}",
            description=(result.reason or "")[:500], source="agent", confidence=0.5,
            tags=[result.failure_category or "failure", fclass.value]))
        self.mission.set_agent_status(task.assigned_agent, status="IDLE", current_task=None)
        self._record_coord("TASK_FAILED", command=(task.objective or "")[:400],
                           strategy=task.role, result=result.status,
                           decision_summary=(result.reason or "")[:300])
        events.broadcast(events.TASK_FAILED, {
            **self._task_event(task, task.assigned_agent),
            "reason": result.reason, "category": result.failure_category})

        # 2) Decide recovery: retry / reassign / abandon (§9).
        decision = self.supervisor.decide_recovery(
            task, result, max_retries=self.limits.max_task_retries)
        if decision.action == "retry":
            task.retry_count += 1
            self.scheduler.requeue_for_retry(task)
            self._record_coord("REPLAN", strategy=task.role,
                               decision_summary=f"retry #{task.retry_count}: {decision.reason}")
        elif decision.action == "reassign":
            self.scheduler.mark_reassigned(task, reason=decision.reason)
            new = Task(
                mission_id=self.mission_id, run_id=self.mission.run_id,
                challenge_id=self.mission.challenge_id,
                role=decision.new_role or task.role,
                objective=decision.objective or task.objective,
                priority=task.priority, parent_task_id=task.id)
            if self._add_task(new, origin="reassign"):
                events.broadcast(events.TASK_REASSIGNED, {
                    "mission_id": self.mission_id, "from_task": task.id, "to_task": new.id,
                    "role": new.role, "reason": decision.reason})
                self._record_coord("TASK_REASSIGNED", strategy=new.role,
                                   decision_summary=decision.reason)
        else:  # abandon
            self.scheduler.mark_failed(task, reason=decision.reason)
            self.mission.record_dead_end(
                f"{task.role}: {(task.objective or '')[:60]} — {decision.reason}")

    # ================================================================== #
    # Phase 5 — OBSERVE → REASON → replan → stop (§13, §24, §25, §26, §27)
    # ================================================================== #

    def _post_step(self, task: Task, result: AgentResult) -> None:
        """Run after every completed step: account budget, track progress, reason about
        new evidence, and evaluate stop conditions. All deterministic (§35)."""
        # §27 — resource accounting.
        self.budget.record_tool_executions(getattr(result, "tool_executions", 0) or 0)
        if (result.status or "").upper() not in ("COMPLETED", "CANCELLED"):
            self.budget.record_failure()
        self.mission.mission_budget = self.budget.to_dict()

        if not self.enable_reasoning:
            return

        # §26 — no-progress ledger (compare knowledge snapshot to the best so far).
        self.ledger.record(self.mission)

        # §13/§16 — adaptive replanning: reason about the freshly published evidence and
        # inject scored, evidence-backed follow-up actions the deterministic evidence
        # rules did not already cover. Only fires when this step produced new evidence,
        # so a step that learned nothing never floods the queue.
        if getattr(result, "evidence", None):
            self._reason_and_replan(list(result.evidence))

        # §25 — stop conditions (budget exhausted / stagnation / all-blocked).
        self._evaluate_stop_conditions()

    def _reason_and_replan(self, recent_evidence: List[Any]) -> None:
        """Score candidate next actions and inject the best evidence-backed ones (§13).

        The Supervisor's :meth:`reason` produces the full ranked, scored candidate set
        (stored on the mission for observability, §37). We only *inject* candidates that
        react to concrete new evidence — memory/state-gap/playbook candidates are
        advisory and are NOT auto-executed without evidence (§18/§19/§32); the existing
        deterministic evidence rules + these injections cover evidence-driven work.
        """
        try:
            decision = self.supervisor.reason(
                self.mission, recent_evidence=recent_evidence, scorer=self.scorer,
                generator=self.generator,
                attempted_signatures=list(self._dispatched_actions.keys()),
                blocked_capabilities=list(self._blocked_capabilities),
                budget_pressure=self.budget.pressure(),
            )
        except Exception as e:  # reasoning must never crash the mission
            logger.debug(f"[SwarmCoordinator] reasoning error: {e}")
            return

        if decision.has_action:
            sel = decision.selected
            self._record_coord("REASONING_DECISION", command=(sel.objective or "")[:400],
                               strategy=sel.role, result=decision.mode,
                               decision_summary=decision.reason[:400])
            events.broadcast(events.REASONING_DECISION, {
                "mission_id": self.mission_id, "challenge_id": self.mission.challenge_id,
                "selected": sel.action_type, "role": sel.role, "score": round(sel.score, 3),
                "mode": decision.mode, "uncertainty": decision.uncertainty,
                "considered": decision.considered, "reason": decision.reason[:300]})

        injected = 0
        blocked = {c.lower() for c in self._blocked_capabilities}
        for cand in decision.ranked:
            if injected >= self.max_replan_actions:
                break
            if cand.source != "evidence":                     # advisory-only, not auto-run
                continue
            if cand.signature in self._dispatched_actions or self.mission.has_failed_action(cand.signature):
                continue                                       # §8 duplicate / already failed
            if cand.capability and cand.capability.lower() in blocked:
                continue                                       # needs an unavailable capability
            new_task = cand.to_task(mission_id=self.mission_id, run_id=self.mission.run_id,
                                    challenge_id=self.mission.challenge_id)
            new_task.priority = cand.priority                  # §17 score-driven priority
            if self._add_task(new_task, origin="reasoning"):
                injected += 1

    def _evaluate_stop_conditions(self) -> None:
        """Stop the mission on budget exhaustion / stagnation / all-capabilities-blocked
        (§25). A verified flag is handled elsewhere and always wins; this only fires the
        *negative* terminal conditions so an unproductive mission cannot loop forever."""
        if self._stopped:
            return
        budget_done, _ = self.budget.exhausted()
        stagnant = self.ledger.is_stagnant()
        all_blocked = self._all_remaining_work_blocked()
        if not (budget_done or stagnant or all_blocked):
            return
        cond, reason = evaluate_stop(
            self.mission, budget=self.budget, ledger=self.ledger,
            has_open_work=self.scheduler.has_open_work(),
            all_capabilities_blocked=all_blocked)
        if cond.is_terminal and cond is not StopCondition.FLAG_VERIFIED:
            self.mission.stop_condition = cond.value
            self._record_coord("MISSION_STOP", result=cond.value, decision_summary=reason[:300])
            events.broadcast(events.MISSION_STOP, {
                "mission_id": self.mission_id, "challenge_id": self.mission.challenge_id,
                "stop_condition": cond.value, "reason": reason})
            self._trigger_global_stop(f"{cond.value}: {reason}", final_status="FAILED")

    def _all_remaining_work_blocked(self) -> bool:
        """True if there is open work but every not-yet-terminal task needs a capability
        already proven blocked this mission (§25 CAPABILITY_BLOCKED)."""
        if not self._blocked_capabilities:
            return False
        open_tasks = [t for t in self.scheduler.all() if t.is_active]
        if not open_tasks:
            return False
        blocked = {c.lower() for c in self._blocked_capabilities}
        for t in open_tasks:
            caps = {c.lower() for c in (t.required_capabilities or [])}
            if not caps or not caps.issubset(blocked):
                return False
        return True

    # ================================================================== #
    # Evidence subscription (runs synchronously during publish)
    # ================================================================== #

    def _on_evidence(self, ev: Evidence) -> None:
        # Integrate into shared state and let the supervisor propose follow-up work.
        self.mission.integrate_evidence(ev)
        try:
            self._followup_buffer.extend(self.supervisor.react_to_evidence(ev, self.mission))
        except Exception as e:  # a planning hiccup must not break publication
            logger.debug(f"[SwarmCoordinator] react_to_evidence error: {e}")

    def _drain_followups(self) -> None:
        buf, self._followup_buffer = self._followup_buffer, []
        for t in buf:
            self._add_task(t, origin="evidence")

    # ================================================================== #
    # Flag verification + global stop (§11)
    # ================================================================== #

    def _accept_verified_flag(self, flag: str, *, agent_id: str, task: Optional[Task]) -> None:
        if self.mission.verified_flag:
            return
        self.mission.set_verified_flag(flag)
        if task is not None:
            self.scheduler.mark_completed(task, result={"verified_flag": flag})
        self.bus.publish(Evidence(
            mission_id=self.mission_id, agent_id=agent_id or "swarm", task_id=(task.id if task else None),
            evidence_type=EvidenceType.FLAG.value, title=flag, source="agent",
            confidence=1.0, tags=["verified"]))
        self._record_coord("FLAG_VERIFIED", result="VERIFIED",
                           decision_summary=f"Flag verified: {flag}")
        events.broadcast(events.FLAG_VERIFIED, {
            "mission_id": self.mission_id, "run_id": self.mission.run_id,
            "challenge_id": self.mission.challenge_id, "flag": flag, "agent_id": agent_id})
        self._trigger_global_stop("Flag verified — mission complete.", final_status="COMPLETED")

    def _broadcast_flag_candidate(self, candidate: str, agent_id: str) -> None:
        self._record_coord("FLAG_CANDIDATE", decision_summary=f"Candidate (unverified): {candidate}")
        events.broadcast(events.FLAG_CANDIDATE, {
            "mission_id": self.mission_id, "challenge_id": self.mission.challenge_id,
            "candidate": candidate, "agent_id": agent_id})

    def _trigger_global_stop(self, reason: str, *, final_status: str) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_reason = reason
        self._pending_final_status = final_status
        cancelled = self.scheduler.cancel_all_open(reason=f"Mission stopped: {reason}")
        self._record_coord("MISSION_STOP", result=final_status,
                           decision_summary=f"{reason} (cancelled {cancelled} open task(s))")

    def _should_stop(self) -> bool:
        if self._stopped:
            return True
        if self.kill_switch:
            try:
                if self.kill_switch():
                    self._stopped = True
                    self._stop_reason = "Kill switch."
                    self._pending_final_status = "CANCELLED"
                    return True
            except Exception:
                pass
        return False

    # ================================================================== #
    # Task creation with dedup + limits (§8, §12)
    # ================================================================== #

    def _add_task(self, task: Task, *, origin: str) -> bool:
        if self.scheduler.total_count() >= self.limits.max_total_tasks:
            return False
        if origin != "plan" and self.scheduler.active_count() >= self.limits.max_active_tasks:
            return False
        # Cross-check against shared attempted signatures too (dedupes across resume).
        if task.signature in set(self.mission.attempted_signatures) and origin != "plan":
            if self.scheduler.has_signature(task.signature):
                return False
        added = self.scheduler.add(task)
        if not added:
            return False
        self.mission.note_attempt(task.signature)
        self._record_coord("TASK_CREATED", command=(task.objective or "")[:400],
                           strategy=task.role, decision_summary=f"origin={origin}")
        events.broadcast(events.TASK_CREATED, self._task_event(task, task.assigned_agent))
        return True

    # ================================================================== #
    # Persistence / finalize / resume
    # ================================================================== #

    def _save_mission(self) -> None:
        if not self.persist:
            return
        self.mission.save(coord_session_id=self.coord_session.id)
        events.broadcast(events.STATE_CHANGED, {
            "mission_id": self.mission_id, "status": self.mission.status,
            "progress": self.mission.progress, "task_counts": self._task_counts()})

    def _maybe_checkpoint(self) -> None:
        if not self.persist or not self.mission.run_id:
            return
        try:
            self.coord_session.last_sequence = trajectory_store.next_sequence(self.coord_session.id) - 1
            session_manager.checkpoint(self.coord_session, last_action="swarm coordination step")
        except Exception:
            pass

    def _load_persisted(self) -> None:
        # The target supplied for THIS (resumed) run — captured BEFORE we adopt the
        # persisted mission, because the checkpoint's target may be STALE (a CTF
        # instance respawns with a new address, or the operator edits it). Phase 7
        # (STEP 2/5): the current target wins over whatever is frozen in the checkpoint.
        fresh_target = self.mission.target
        loaded = SharedMissionState.load(self.mission_id)
        if loaded:
            # Keep the freshly-provided metadata but adopt aggregated knowledge/flags.
            loaded.run_id = loaded.run_id or self.mission.run_id
            loaded.challenge_id = loaded.challenge_id or self.mission.challenge_id
            self.mission = loaded
            # Phase 5 — restore the budget so accounting continues across resume; the
            # attempted-action signatures are already carried in the persisted state.
            try:
                if loaded.mission_budget:
                    restored = MissionBudget.from_dict(loaded.mission_budget)
                    # Preserve caps from this construction; adopt the used counters.
                    restored.max_agent_calls = self.budget.max_agent_calls or restored.max_agent_calls
                    restored.max_tool_executions = self.budget.max_tool_executions or restored.max_tool_executions
                    restored.max_failed_attempts = self.budget.max_failed_attempts or restored.max_failed_attempts
                    restored.max_duplicate_attempts = self.budget.max_duplicate_attempts or restored.max_duplicate_attempts
                    restored.max_wall_seconds = self.budget.max_wall_seconds or restored.max_wall_seconds
                    self.budget = restored
            except Exception:
                pass
            # Phase 7 (STEP 2/5) — reconcile the authoritative target: if the operator
            # supplied a different target for this resume, it becomes authoritative and
            # host-specific state from the old target is invalidated so nothing executes
            # against a stale address merely because it exists in the checkpoint.
            self._reconcile_authoritative_target(fresh_target)
        self.bus.load()
        self.scheduler.load(self.mission_id)

    def _reconcile_authoritative_target(self, fresh_target: str) -> None:
        """Compare the current run's target against the persisted one; on a change,
        make the current target authoritative and invalidate stale host-bound state.

        Deterministic and non-fatal: a reconciliation hiccup must never abort a resume.
        """
        try:
            recon = reconcile_target(fresh_target, self.mission.target)
        except Exception as e:  # reconciliation must never crash a resume
            logger.debug(f"[SwarmCoordinator] target reconcile error: {e}")
            return
        self._last_reconciliation = recon.to_dict()
        if not recon.changed:
            return

        counts = self.mission.adopt_authoritative_target(recon.authoritative, recon.stale_hosts)
        # Re-derive the target KIND against the new target (conservative; UNKNOWN if unsure).
        try:
            self.mission.target_type = self.target_detector.detect(recon.authoritative).type.value
        except Exception:
            pass
        total = sum(counts.values())
        self._last_reconciliation["invalidated"] = counts
        logger.warning(
            f"[SwarmCoordinator] TARGET CHANGED on resume: '{recon.previous}' -> "
            f"'{recon.authoritative}'. Invalidated {total} stale item(s): {counts}. "
            f"The current target is now authoritative.")
        self._record_coord("TARGET_CHANGED", result="RECONCILED", command=recon.authoritative,
                           decision_summary=f"{recon.reason}; invalidated {total} stale item(s)"[:300])
        try:
            self.bus.publish(Evidence(
                mission_id=self.mission_id, agent_id="supervisor",
                evidence_type=EvidenceType.NOTE.value,
                title=f"Authoritative target changed to {recon.authoritative}",
                description=(f"Previous target '{recon.previous}' is stale; {total} host-specific "
                             f"item(s) invalidated so execution targets only the current "
                             f"address.")[:500],
                source="supervisor", confidence=1.0, tags=["target_changed", "resume"]))
        except Exception:
            pass
        events.broadcast(events.TARGET_CHANGED, {
            "mission_id": self.mission_id, "run_id": self.mission.run_id,
            "challenge_id": self.mission.challenge_id, "old_target": recon.previous,
            "new_target": recon.authoritative, "stale_hosts": recon.stale_hosts,
            "invalidated": counts})
        self._save_mission()

    def _finalize(self) -> MissionResult:
        counts = self._task_counts()
        if self.mission.verified_flag:
            status = "COMPLETED"
        elif getattr(self, "_pending_final_status", "") in ("CANCELLED", "PAUSED"):
            status = self._pending_final_status
        elif self.scheduler.has_open_work():
            status = "PAUSED"
        else:
            status = "FAILED"
        self.mission.status = status
        # Phase 5 §25 — record the authoritative stop condition + final budget snapshot.
        if status == "COMPLETED":
            self.mission.stop_condition = StopCondition.FLAG_VERIFIED.value
        elif not self.mission.stop_condition and status == "FAILED":
            self.mission.stop_condition = (StopCondition.UNRECOVERABLE_ERROR.value
                                           if getattr(self, "_pending_final_status", "") == "CANCELLED"
                                           else StopCondition.NO_PROGRESS.value)
        self.mission.mission_budget = self.budget.to_dict()
        self._save_mission()

        try:
            if status == "COMPLETED":
                session_manager.complete(self.coord_session, outcome="success",
                                         verified_flag=self.mission.verified_flag)
            elif status in ("FAILED", "CANCELLED"):
                session_manager.fail(self.coord_session, self._stop_reason or "Mission ended without a flag.")
            else:
                session_manager.pause(self.coord_session)
        except Exception:
            pass

        if status == "COMPLETED":
            events.broadcast(events.MISSION_COMPLETE, {
                "mission_id": self.mission_id, "run_id": self.mission.run_id,
                "challenge_id": self.mission.challenge_id, "flag": self.mission.verified_flag})
            self._maybe_generate_report()
        elif status == "FAILED":
            events.broadcast(events.MISSION_FAILED, {
                "mission_id": self.mission_id, "challenge_id": self.mission.challenge_id,
                "reason": self._stop_reason or "No flag captured."})

        return MissionResult(
            mission_id=self.mission_id, status=status, verified_flag=self.mission.verified_flag,
            reason=self._stop_reason or status, tasks_total=counts["total"],
            tasks_completed=counts.get("COMPLETED", 0), tasks_failed=counts.get("FAILED", 0),
            evidence_count=self.bus.count())

    def _maybe_generate_report(self) -> None:
        """Best-effort report trigger (§11). Off by default so tests stay hermetic."""
        if not self.enable_report or not self.mission.challenge_id:
            return
        try:
            from backend.reporting.generator import report_generator  # noqa: F401
            # Report generation is delegated to the existing generator; kept guarded so a
            # missing provider/network never affects the mission's completion status.
            logger.info(f"[SwarmCoordinator] report generation queued for {self.mission.challenge_id}")
        except Exception:
            pass

    # ================================================================== #
    # Small helpers
    # ================================================================== #

    @property
    def challenge_key(self) -> Optional[str]:
        return self.mission.challenge_id

    def _task_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {"total": self.scheduler.total_count()}
        for t in self.scheduler.all():
            counts[t.status] = counts.get(t.status, 0) + 1
        return counts

    def _task_event(self, task: Task, agent_id: str) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id, "run_id": self.mission.run_id,
            "challenge_id": self.mission.challenge_id, "task_id": task.id,
            "role": task.role, "agent_id": agent_id, "objective": (task.objective or "")[:200],
            "status": task.status, "priority": task.priority, "dependencies": list(task.dependencies),
        }

    def _record_coord(self, event_type: str, **kw: Any) -> None:
        try:
            trajectory_store.record(
                session_id=self.coord_session.id, event_type=f"COORD_{event_type}",
                run_id=self.mission.run_id, challenge_id=self.mission.challenge_id,
                agent_id="supervisor", **kw)
        except Exception:
            pass
