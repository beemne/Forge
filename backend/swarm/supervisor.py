"""The Supervisor (Phase 4 §1, §9) — deterministic planning & recovery brain.

The supervisor decides *what work remains* and *who should do it*. It is
intentionally pure: every method takes state and returns proposed tasks or a
recovery decision, with no side effects. That makes it fully unit-testable with no
LLM and no database, and it keeps the coordinator (which owns concurrency, I/O and
persistence) small.

The supervisor does NOT perform tool actions itself — it delegates by producing
:class:`~backend.swarm.tasks.Task` objects for specialist agents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.swarm.tasks import Task
from backend.swarm.roles import (
    AgentRole, profile, roles_for_category, roles_activated_by,
)
from backend.swarm.reasoning import (
    CandidateAction, FailureClass, classify_failure as _classify_failure_class,
)
from backend.swarm.scoring import ActionScorer, mission_uncertainty
from backend.swarm.candidates import CandidateGenerator

_VERSION_RE = re.compile(r"\d+\.\d+")
# A "networked" target looks like a URL or host:port / IP — recon (nmap/http) helps.
_NETWORK_RE = re.compile(r"(https?://|:\d{2,5}\b|\b\d{1,3}(?:\.\d{1,3}){3}\b|\.[a-z]{2,})", re.I)

# Roles whose work genuinely benefits from recon output when the target is networked.
_RECON_DEPENDENT = {AgentRole.WEB, AgentRole.PWN}


@dataclass
class RecoveryDecision:
    action: str                 # "retry" | "reassign" | "abandon"
    reason: str
    new_role: Optional[str] = None
    objective: Optional[str] = None


@dataclass
class ReasoningDecision:
    """The Supervisor's answer to "what should happen next, and why?" (§24, §37).

    Fully inspectable: it carries the selected action, the ranked alternatives it beat,
    the plain-language reason, and the uncertainty/mode that shaped the choice — so the
    decision can be logged to the trajectory and shown in reports.
    """

    selected: Optional[CandidateAction] = None
    ranked: List[CandidateAction] = field(default_factory=list)
    reason: str = ""
    uncertainty: float = 1.0
    mode: str = "explore"           # "explore" (reduce uncertainty) | "exploit" (act on evidence)
    considered: int = 0

    @property
    def has_action(self) -> bool:
        return self.selected is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "ranked": [c.to_dict() for c in self.ranked],
            "reason": self.reason, "uncertainty": self.uncertainty,
            "mode": self.mode, "considered": self.considered,
        }


class Supervisor:
    def __init__(self, mission_id: str, *, run_id: Optional[str] = None,
                 challenge_id: Optional[str] = None):
        self.mission_id = mission_id
        self.run_id = run_id
        self.challenge_id = challenge_id

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #

    def plan_initial(self, mission_state: Any) -> List[Task]:
        """Produce the initial task set for a challenge — activated roles, with the
        recon→exploit dependency wired where it applies (the Apache example: a
        version-specific web task waits on recon identifying the service)."""
        target = getattr(mission_state, "target", "") or ""
        category = getattr(mission_state, "category", "") or ""
        description = getattr(mission_state, "description", "") or ""
        artifacts = getattr(mission_state, "artifacts", []) or []

        networked = bool(_NETWORK_RE.search(target)) and not target.strip().startswith("/")
        activated: List[AgentRole] = []
        for r in roles_for_category(category) + roles_activated_by(f"{description} {category}"):
            if r not in activated:
                activated.append(r)
        # Forensics is implied by attached artifacts even if the category didn't say so.
        if artifacts and AgentRole.FORENSICS not in activated:
            activated.append(AgentRole.FORENSICS)

        tasks: List[Task] = []
        recon_task: Optional[Task] = None
        if networked and (AgentRole.RECON in activated or any(r in _RECON_DEPENDENT for r in activated)):
            recon_task = self._make_task(AgentRole.RECON, profile(AgentRole.RECON).objective,
                                         mission_state, priority=90)
            tasks.append(recon_task)

        for role in activated:
            if role is AgentRole.RECON:
                continue
            deps = [recon_task.id] if (recon_task and role in _RECON_DEPENDENT) else []
            prio = 70 if role in _RECON_DEPENDENT else 60
            tasks.append(self._make_task(role, profile(role).objective, mission_state,
                                         priority=prio, dependencies=deps,
                                         parent_task_id=(recon_task.id if deps else None)))
        if not tasks:
            # Nothing matched — default to a recon sweep so the mission still does work.
            tasks.append(self._make_task(AgentRole.RECON, profile(AgentRole.RECON).objective,
                                         mission_state, priority=80))
        return tasks

    # ------------------------------------------------------------------ #
    # React to evidence → follow-up tasks (deterministic rules)
    # ------------------------------------------------------------------ #

    def react_to_evidence(self, ev: Any, mission_state: Any) -> List[Task]:
        """Given a freshly published piece of evidence, propose follow-up tasks.

        The coordinator deduplicates by signature and enforces limits, so this may
        freely propose; identical leads collapse to one task.
        """
        etype = getattr(ev, "evidence_type", "note")
        proposals: List[Task] = []

        if etype in ("service", "technology"):
            tech = (getattr(ev, "related_technology", "") or getattr(ev, "title", "")
                    or getattr(ev, "description", "")).strip()
            if tech:
                if _VERSION_RE.search(tech):
                    # A versioned service justifies targeted vulnerability research (the
                    # Apache 2.4.49 example): high priority, version carried in the objective.
                    proposals.append(self._make_task(
                        AgentRole.WEB,
                        f"Research and exploit known vulnerabilities for the detected service '{tech}'. Extract the flag.",
                        mission_state, priority=85))
                else:
                    # A generic technology still warrants targeted web investigation.
                    proposals.append(self._make_task(
                        AgentRole.WEB,
                        f"Investigate the detected technology '{tech}' for known vulnerabilities, "
                        f"misconfigurations, and default credentials.",
                        mission_state, priority=68))
        elif etype == "endpoint":
            ep = (getattr(ev, "related_endpoint", "") or getattr(ev, "title", "")).strip()
            if ep:
                proposals.append(self._make_task(
                    AgentRole.WEB,
                    f"Enumerate and test the discovered endpoint '{ep}' for access-control and injection flaws.",
                    mission_state, priority=72))
        elif etype == "vulnerability":
            vuln = (getattr(ev, "related_vulnerability", "") or getattr(ev, "title", "")).strip()
            if vuln:
                proposals.append(self._make_task(
                    AgentRole.WEB,
                    f"Develop and run an exploit for the identified vulnerability: {vuln}. Extract the flag.",
                    mission_state, priority=88))
        elif etype == "credential":
            cred = (getattr(ev, "title", "") or getattr(ev, "description", "")).strip()
            if cred:
                proposals.append(self._make_task(
                    AgentRole.WEB,
                    f"Authenticate using discovered credentials ({cred}) and access protected resources.",
                    mission_state, priority=80))
        elif etype in ("artifact", "file"):
            title_str = getattr(ev, 'title', '') or getattr(ev, 'artifact_id', '') or getattr(ev, 'description', '')
            cat = (getattr(mission_state, "category", "") or "").lower()
            if cat == "web" or any(k in title_str.lower() for k in (".php", ".phtml", ".jsp", ".asp", ".aspx", "uploads/", "/uploads")):
                proposals.append(self._make_task(
                    AgentRole.WEB,
                    f"Access and exploit the discovered web artifact/path '{title_str}'. Extract the flag.",
                    mission_state, priority=80))
            else:
                proposals.append(self._make_task(
                    AgentRole.FORENSICS,
                    f"Analyze the discovered artifact '{title_str}'.",
                    mission_state, priority=65))
        return proposals

    # ------------------------------------------------------------------ #
    # Failure classification + recovery decision (§9)
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_failure(result: Any) -> str:
        """Map an agent result to a failure category the recovery policy understands."""
        status = (getattr(result, "status", "") or "").upper()
        cat = (getattr(result, "failure_category", "") or "").upper()
        reason = (getattr(result, "reason", "") or "").lower()

        if status == "CANCELLED":
            return "cancelled"
        if cat in ("CAPABILITY_GAP", "PRIVILEGE_DENIED") or "capability_gap" in reason or "privilege_denied" in reason or ("privilege" in reason and "denied" in reason):
            return "capability_gap"
        # Phase 4.x — a capability that is unavailable-and-not-acquirable, or a target of
        # the wrong KIND, must NOT be retried: retrying cannot change the outcome and only
        # burns the mission budget. Classify them so decide_recovery abandons immediately.
        if cat == "BLOCKED_CAPABILITY" or "blocked_capability" in reason:
            return "blocked_capability"
        if cat == "TARGET_MISMATCH" or "target_mismatch" in reason:
            return "target_mismatch"
        if status == "TIMEOUT":
            return "timeout"
        if cat in ("COMMAND_NOT_FOUND", "MISSING_TOOL", "MISSING_DEPENDENCY"):
            return "missing_dependency"
        if cat == "NETWORK" or "network" in reason or "connection" in reason:
            return "network"
        if cat == "PERMISSION" or "permission" in reason or "denied" in reason:
            return "permission"
        if "provider" in reason or "exhausted" in reason:
            return "provider"
        if status == "MAX_TURNS" or "no_progress" in reason or cat == "NO_PROGRESS":
            return "no_progress"
        return "execution"

    def decide_recovery(self, task: Task, result: Any, *, max_retries: int) -> RecoveryDecision:
        category = self.classify_failure(result)

        if category == "cancelled":
            return RecoveryDecision("abandon", "Task cancelled by global stop.")

        if category == "capability_gap":
            return RecoveryDecision(
                "retry", "Capability gap (privilege denied); request privilege escalation via approval pipeline.")

        # Phase 4.x — impossible-as-specified failures: retrying is futile. Abandon so the
        # mission records a dead end and moves on instead of re-dispatching (§15, §22).
        if category == "blocked_capability":
            return RecoveryDecision(
                "abandon", "A required capability is unavailable and cannot be acquired here; "
                           "recording a dead end rather than retrying.")
        if category == "target_mismatch":
            return RecoveryDecision(
                "abandon", "The provided target is the wrong kind for this task; a correct "
                           "target is required — abandoning to avoid wasted iterations.")

        # Transient categories are worth a bounded retry.
        if category in ("provider", "timeout", "network") and task.retry_count < max_retries:
            return RecoveryDecision("retry", f"Transient failure ({category}); retrying.")

        # A missing tool/dependency: reassign with an explicit instruction to use an
        # available alternative (do not just crash the mission — §9 ffuf example).
        if category == "missing_dependency":
            alt = (f"{profile(AgentRole.from_value(task.role)).objective} "
                   f"NOTE: a required tool was unavailable last time — accomplish this "
                   f"using an alternative available method (e.g. curl/python instead of a "
                   f"missing scanner).")
            return RecoveryDecision("reassign", "Required tool unavailable; reassigning with an alternative method.",
                                    new_role=task.role, objective=alt)

        # No-progress: reassign to a different specialist lens if one is plausible.
        if category == "no_progress" and task.retry_count < max_retries:
            return RecoveryDecision("retry", "No progress; one more attempt with accumulated evidence.")

        return RecoveryDecision("abandon", f"Unrecoverable failure ({category}); recording dead end.")

    # ------------------------------------------------------------------ #
    # Phase 5 §10-13, §24 — the central "what should we try next?" decision
    # ------------------------------------------------------------------ #

    def reason(
        self,
        mission_state: Any,
        *,
        recent_evidence: Optional[List[Any]] = None,
        scorer: Optional[ActionScorer] = None,
        generator: Optional[CandidateGenerator] = None,
        attempted_signatures: Optional[List[str]] = None,
        available_capabilities: Optional[List[str]] = None,
        blocked_capabilities: Optional[List[str]] = None,
        exhausted_strategies: Optional[List[str]] = None,
        budget_pressure: float = 0.0,
        use_memory: bool = True,
    ) -> ReasoningDecision:
        """Decide the single best next action given everything currently known (§24).

        OBSERVE (mission_state + recent_evidence) → REASON (generate candidates, score
        them, balance exploration/exploitation) → return the top action with its full
        ranked alternatives and a plain reason. Pure: no side effects except stashing
        the ranked candidates on the mission_state for observability (§37).
        """
        scorer = scorer or ActionScorer()
        generator = generator or CandidateGenerator()

        candidates = generator.generate(mission_state, recent_evidence=recent_evidence,
                                         use_memory=use_memory)
        # §8/§21 — drop candidates that cannot succeed as specified:
        #   * one needing a capability already proven unavailable (do not re-propose it —
        #     the Binary Digits/OCR case: recognise it once, then stop repeating), and
        #   * one whose action was already attempted/failed, UNLESS fresh evidence
        #     justifies a retry (a capability-blocked action is never "justified"), and
        #   * one whose strategy class is exhausted across the swarm unless fresh evidence justifies it.
        attempted = set(attempted_signatures or [])
        failed = set(getattr(mission_state, "action_signatures", []) or [])
        blocked = {c.lower() for c in (blocked_capabilities or [])}
        exhausted = {s.lower() for s in (exhausted_strategies or getattr(mission_state, "exhausted_strategies", []) or [])}

        prefiltered = []
        for c in candidates:
            if c.capability and c.capability.lower() in blocked:
                continue
            strat_key = (getattr(c, "strategy", "") or c.action_type or "").lower()
            already = c.signature in attempted or c.signature in failed or (strat_key and strat_key in exhausted)
            evidence_backed = c.source in ("evidence",) or c.evidence_support >= 0.7
            if already and not evidence_backed:
                continue
            prefiltered.append(c)
        candidates = prefiltered or candidates   # never end up with nothing to consider

        uncertainty = mission_uncertainty(mission_state)
        ranked = scorer.rank(
            candidates,
            attempted_signatures=attempted | failed,
            available_capabilities=available_capabilities,
            blocked_capabilities=blocked_capabilities,
            exhausted_strategies=exhausted,
            uncertainty=uncertainty,
        )

        # §27 — under budget pressure, drop the lowest-value tail so we only spend the
        # remaining mission on high-value actions.
        if budget_pressure >= 0.8 and len(ranked) > 2:
            ranked = ranked[: max(2, len(ranked) // 2)]

        try:
            if hasattr(mission_state, "set_candidate_actions"):
                mission_state.set_candidate_actions(ranked)
        except Exception:
            pass

        selected = ranked[0] if ranked else None
        mode = "exploit" if uncertainty < 0.5 else "explore"
        reason = ""
        if selected:
            reason = (f"Selected '{selected.action_type}' (score {selected.score:.2f}, "
                      f"mode={mode}, uncertainty={uncertainty:.2f}): {selected.rationale}")
        return ReasoningDecision(selected=selected, ranked=ranked, reason=reason,
                                 uncertainty=uncertainty, mode=mode, considered=len(candidates))

    # ------------------------------------------------------------------ #
    # Phase 5 §14 — richer failure classification (the string API above is kept)
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_failure_detailed(result: Any) -> FailureClass:
        """Return the Phase 5 :class:`FailureClass` taxonomy for a result (§14)."""
        return _classify_failure_class(result)

    # ------------------------------------------------------------------ #

    def _make_task(self, role: AgentRole, objective: str, mission_state: Any, *,
                   priority: int = 50, dependencies: Optional[List[str]] = None,
                   parent_task_id: Optional[str] = None) -> Task:
        return Task(
            mission_id=self.mission_id, run_id=self.run_id, challenge_id=self.challenge_id,
            role=role.value, objective=objective, priority=priority,
            dependencies=list(dependencies or []), parent_task_id=parent_task_id,
        )
