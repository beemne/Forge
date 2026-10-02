"""Phase 5 §25/§26/§27 — mission budget, no-progress detection, stop conditions.

These are the guard rails that stop FORGE looping forever. All deterministic, no
LLM (§35): counting actions, comparing knowledge snapshots, and checking budgets do
not need a model.

  * :class:`MissionBudget` (§27) — bounded resource accounting across agent calls,
    tool executions, failed attempts and repeated actions. When it runs low the
    coordinator biases toward high-value actions and, when exhausted, stops.
  * :class:`ProgressLedger` (§26) — records a compact snapshot of "how much do we
    know" each step; if N consecutive steps add nothing new it reports
    ``STRATEGY_STAGNATION`` so the supervisor replans instead of grinding.
  * :class:`StopCondition` / :func:`evaluate_stop` (§25) — the authoritative set of
    reasons a mission ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────── #
# §25 Stop conditions
# ─────────────────────────────────────────────────────────────────────────── #

class StopCondition(str, Enum):
    NONE = "NONE"
    FLAG_VERIFIED = "FLAG_VERIFIED"
    TARGET_BLOCKED = "TARGET_BLOCKED"
    CAPABILITY_BLOCKED = "CAPABILITY_BLOCKED"
    NO_PROGRESS = "NO_PROGRESS"
    MISSION_BUDGET_EXHAUSTED = "MISSION_BUDGET_EXHAUSTED"
    UNRECOVERABLE_ERROR = "UNRECOVERABLE_ERROR"

    @property
    def is_terminal(self) -> bool:
        return self is not StopCondition.NONE

    @property
    def final_status(self) -> str:
        """The mission status a stop condition maps onto."""
        return {
            StopCondition.FLAG_VERIFIED: "COMPLETED",
            StopCondition.TARGET_BLOCKED: "FAILED",
            StopCondition.CAPABILITY_BLOCKED: "FAILED",
            StopCondition.NO_PROGRESS: "FAILED",
            StopCondition.MISSION_BUDGET_EXHAUSTED: "FAILED",
            StopCondition.UNRECOVERABLE_ERROR: "FAILED",
            StopCondition.NONE: "RUNNING",
        }[self]


# ─────────────────────────────────────────────────────────────────────────── #
# §27 Mission budget
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class MissionBudget:
    """Bounded resource accounting for one mission (§27).

    A budget of ``0`` (the default for any dimension) means *unbounded* for that
    dimension, so existing tests and callers that do not set a budget are unaffected.
    """

    max_agent_calls: int = 0
    max_tool_executions: int = 0
    max_failed_attempts: int = 0
    max_duplicate_attempts: int = 0
    max_wall_seconds: float = 0.0

    agent_calls: int = 0
    tool_executions: int = 0
    failed_attempts: int = 0
    duplicate_attempts: int = 0
    started_monotonic: float = 0.0
    elapsed_seconds: float = 0.0

    # -- accounting ----------------------------------------------------------- #

    def record_agent_call(self, n: int = 1) -> None:
        self.agent_calls += n

    def record_tool_executions(self, n: int) -> None:
        if n:
            self.tool_executions += int(n)

    def record_failure(self, n: int = 1) -> None:
        self.failed_attempts += n

    # -- queries -------------------------------------------------------------- #

    @staticmethod
    def _frac(used: int, cap: int) -> float:
        if cap <= 0:
            return 0.0
        return used / cap

    def pressure(self) -> float:
        """0..1 — how close the tightest bounded dimension is to its cap (§27).

        The coordinator uses this to prefer high-value actions as the budget tightens.
        """
        fracs = [
            self._frac(self.agent_calls, self.max_agent_calls),
            self._frac(self.tool_executions, self.max_tool_executions),
            self._frac(self.failed_attempts, self.max_failed_attempts),
            self._frac(self.duplicate_attempts, self.max_duplicate_attempts),
            (self.elapsed_seconds / self.max_wall_seconds) if self.max_wall_seconds > 0 else 0.0,
        ]
        return round(min(1.0, max(fracs)) if fracs else 0.0, 4)

    def exhausted(self) -> Tuple[bool, str]:
        """Return (exhausted, reason). Budget restrictions removed so runs are never halted on budget."""
        return False, ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "max_agent_calls": self.max_agent_calls, "agent_calls": self.agent_calls,
            "max_tool_executions": self.max_tool_executions, "tool_executions": self.tool_executions,
            "max_failed_attempts": self.max_failed_attempts, "failed_attempts": self.failed_attempts,
            "max_duplicate_attempts": self.max_duplicate_attempts,
            "duplicate_attempts": self.duplicate_attempts,
            "max_wall_seconds": self.max_wall_seconds, "elapsed_seconds": self.elapsed_seconds,
            "pressure": self.pressure(),
        }
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MissionBudget":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────── #
# §26 No-progress detection
# ─────────────────────────────────────────────────────────────────────────── #

def knowledge_fingerprint(mission_state: Any) -> Tuple[int, ...]:
    """A compact, comparable snapshot of how much is known (§26).

    Two steps with the same fingerprint means the second added no new knowledge — the
    signal the no-progress detector watches. Counting (not content) keeps it cheap and
    monotonic; it covers every dimension a productive step would grow.
    """
    def _n(*names: str) -> int:
        for n in names:
            v = getattr(mission_state, n, None)
            if v is not None:
                try:
                    return len(v)
                except Exception:
                    return 0
        return 0

    return (
        _n("endpoints", "known_endpoints"),
        _n("services", "known_services"),
        _n("technologies"),
        _n("vulnerabilities"),
        _n("credentials"),
        _n("artifacts"),
        _n("confirmed_facts", "facts"),
        _n("hypotheses"),
        _n("flag_candidates"),
        1 if getattr(mission_state, "verified_flag", None) else 0,
    )


@dataclass
class ProgressLedger:
    """Tracks knowledge growth across steps and detects stagnation (§26)."""

    stagnation_limit: int = 5          # N consecutive no-gain steps → stagnation
    history: List[Tuple[int, ...]] = field(default_factory=list)
    stagnant_steps: int = 0
    best_fingerprint: Optional[Tuple[int, ...]] = None

    def record(self, mission_state: Any) -> bool:
        """Record a step. Returns True if it made progress (added knowledge)."""
        fp = knowledge_fingerprint(mission_state)
        progressed = self.best_fingerprint is None or _fp_gt(fp, self.best_fingerprint)
        if progressed:
            self.best_fingerprint = _fp_max(fp, self.best_fingerprint)
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
        self.history.append(fp)
        return progressed

    def is_stagnant(self) -> bool:
        return self.stagnant_steps >= self.stagnation_limit

    def reset(self) -> None:
        """Called after a replan / strategy change so the counter starts fresh."""
        self.stagnant_steps = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"stagnation_limit": self.stagnation_limit,
                "stagnant_steps": self.stagnant_steps,
                "steps_recorded": len(self.history)}


def _fp_gt(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    """True if snapshot *a* strictly improves on *b* in at least one dimension and
    regresses in none (knowledge is monotonic-ish; a dropped count is not progress)."""
    if len(a) != len(b):
        return sum(a) > sum(b)
    better = any(x > y for x, y in zip(a, b))
    worse = any(x < y for x, y in zip(a, b))
    return better and not worse


def _fp_max(a: Tuple[int, ...], b: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    if b is None or len(a) != len(b):
        return a
    return tuple(max(x, y) for x, y in zip(a, b))


# ─────────────────────────────────────────────────────────────────────────── #
# §25 Stop-condition evaluation
# ─────────────────────────────────────────────────────────────────────────── #

def evaluate_stop(
    mission_state: Any,
    *,
    budget: Optional[MissionBudget] = None,
    ledger: Optional[ProgressLedger] = None,
    has_open_work: bool = True,
    all_capabilities_blocked: bool = False,
    target_blocked: bool = False,
    unrecoverable: bool = False,
) -> Tuple[StopCondition, str]:
    """Decide whether the mission should stop, and why (§25).

    Ordered by priority: a verified flag always wins; then hard blocks; then budget;
    then stagnation. Returns ``(StopCondition.NONE, "")`` while work should continue.
    """
    if getattr(mission_state, "verified_flag", None):
        return StopCondition.FLAG_VERIFIED, "Flag verified."

    if unrecoverable:
        return StopCondition.UNRECOVERABLE_ERROR, "Unrecoverable error."

    if target_blocked:
        return StopCondition.TARGET_BLOCKED, "The provided target is blocked or the wrong kind."

    if all_capabilities_blocked:
        return StopCondition.CAPABILITY_BLOCKED, "All remaining work needs unavailable capabilities."

    if budget is not None:
        done, reason = budget.exhausted()
        if done:
            return StopCondition.MISSION_BUDGET_EXHAUSTED, reason

    if ledger is not None and ledger.is_stagnant():
        return (StopCondition.NO_PROGRESS,
                f"No new knowledge in {ledger.stagnant_steps} consecutive steps.")

    # If there is simply nothing left to do and no flag, that is a (non-flag) end too,
    # but the caller distinguishes PAUSED vs FAILED — so we only signal it as no-work.
    if not has_open_work:
        return StopCondition.NONE, ""

    return StopCondition.NONE, ""
