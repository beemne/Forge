"""Phase 5 §10/§11 — deterministic candidate-action scoring & information gain.

The planner may propose many candidate next actions; this module ranks them with a
transparent, inspectable, fully-deterministic formula. No LLM decides the final
number — an LLM may *propose* a candidate (with prose fields), but the score that
selects it is computed here from labelled inputs, so it is testable and explainable
(§10, §35, §37).

    score =  w_ig   · information_gain
           + w_ev   · evidence_support
           + w_sp   · success_probability
           + w_nov  · novelty
           − w_cost · cost
           − w_risk · risk
           − duplicate_penalty
           − dependency_penalty          (a candidate whose capability is unavailable)

Every term is in [0, 1] before weighting; the weighted sum is returned together with
a per-term breakdown so the coordinator can log *why* an action was chosen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from backend.swarm.reasoning import (
    CandidateAction, Cost, InformationGain, Risk, profile_for,
)


@dataclass(frozen=True)
class ScoreWeights:
    """Tunable weights (defaults chosen so information-seeking beats blind exploits)."""

    information_gain: float = 1.0
    evidence_support: float = 0.9
    success_probability: float = 0.8
    novelty: float = 0.6
    cost: float = 0.5
    risk: float = 0.5
    duplicate_penalty: float = 1.5      # a near-duplicate action is heavily penalised (§8)
    dependency_penalty: float = 1.2     # an action needing an unavailable capability
    # Exploration vs exploitation (§12): how strongly to bias toward info-gain when the
    # mission is still uncertain, and toward exploitation when strong evidence exists.
    exploration_bonus: float = 0.5


def _gain_float(value: str) -> float:
    try:
        return InformationGain(value).value_float
    except Exception:
        return 0.5


def _cost_float(value: str) -> float:
    try:
        return Cost(value).value_float
    except Exception:
        return 0.45


def _risk_float(value: str) -> float:
    try:
        return Risk(value).value_float
    except Exception:
        return 0.4


class ActionScorer:
    """Scores :class:`CandidateAction` objects deterministically (§10)."""

    def __init__(self, weights: Optional[ScoreWeights] = None):
        self.weights = weights or ScoreWeights()

    # ------------------------------------------------------------------ #

    def score(
        self,
        action: CandidateAction,
        *,
        attempted_signatures: Optional[Iterable[str]] = None,
        available_capabilities: Optional[Iterable[str]] = None,
        blocked_capabilities: Optional[Iterable[str]] = None,
        exhausted_strategies: Optional[Iterable[str]] = None,
        uncertainty: float = 1.0,
    ) -> CandidateAction:
        """Compute and attach ``score`` + ``score_breakdown`` to *action*.

        ``uncertainty`` (0..1) is the mission's current uncertainty — high early
        (favours exploration / information gain), low once strong evidence exists
        (favours exploitation). See :func:`mission_uncertainty`.
        """
        w = self.weights
        attempted = set(attempted_signatures or [])
        available = set(c.lower() for c in (available_capabilities or []))
        blocked = set(c.lower() for c in (blocked_capabilities or []))
        exhausted = set(s.lower() for s in (exhausted_strategies or []))

        ig = _gain_float(action.information_gain)
        ev = _clamp(action.evidence_support)
        sp = _clamp(action.success_probability)
        nov = _clamp(action.novelty)
        cost = _cost_float(action.cost)
        risk = _risk_float(action.risk)

        # Duplicate suppression (§8): an already-attempted signature is strongly
        # penalised and its effective novelty collapses to 0.
        is_dup = bool(action.signature and action.signature in attempted)
        dup_pen = w.duplicate_penalty if is_dup else 0.0
        if is_dup:
            nov = 0.0

        # Strategy exhaustion penalty: a candidate using an exhausted strategy
        # is penalised unless fresh concrete evidence explicitly justifies it.
        strat_key = (getattr(action, "strategy", "") or action.action_type or "").lower()
        is_exhausted = bool(strat_key and strat_key in exhausted and action.source != "evidence" and ev < 0.7)
        if is_exhausted:
            dup_pen += w.duplicate_penalty
            nov = 0.0

        # Dependency availability (§10): a candidate needing a capability known to be
        # blocked is penalised so it sinks below anything runnable here-and-now.
        cap = (action.capability or "").lower()
        dep_pen = 0.0
        if cap:
            if cap in blocked:
                dep_pen = w.dependency_penalty
            elif available and cap not in available:
                dep_pen = w.dependency_penalty * 0.5   # unknown availability — mild caution

        # Exploration/exploitation blend (§11, §12): scale information-gain's reward by
        # how uncertain we still are, and evidence_support's reward by how certain.
        u = _clamp(uncertainty)
        ig_term = w.information_gain * ig * (0.5 + w.exploration_bonus * u)
        ev_term = w.evidence_support * ev * (0.5 + (1.0 - u) * 0.5)

        breakdown = {
            "information_gain": round(ig_term, 4),
            "evidence_support": round(ev_term, 4),
            "success_probability": round(w.success_probability * sp, 4),
            "novelty": round(w.novelty * nov, 4),
            "cost": round(-w.cost * cost, 4),
            "risk": round(-w.risk * risk, 4),
            "duplicate_penalty": round(-dup_pen, 4),
            "dependency_penalty": round(-dep_pen, 4),
        }
        total = round(sum(breakdown.values()), 4)
        action.score = total
        action.score_breakdown = breakdown
        action.priority = _score_to_priority(total)
        return action

    def rank(
        self,
        actions: List[CandidateAction],
        *,
        attempted_signatures: Optional[Iterable[str]] = None,
        available_capabilities: Optional[Iterable[str]] = None,
        blocked_capabilities: Optional[Iterable[str]] = None,
        exhausted_strategies: Optional[Iterable[str]] = None,
        uncertainty: float = 1.0,
    ) -> List[CandidateAction]:
        """Score every candidate and return them sorted best-first (stable)."""
        attempted = set(attempted_signatures or [])
        available = set(available_capabilities or [])
        blocked = set(blocked_capabilities or [])
        exhausted = set(exhausted_strategies or [])
        for a in actions:
            self.score(a, attempted_signatures=attempted, available_capabilities=available,
                       blocked_capabilities=blocked, exhausted_strategies=exhausted,
                       uncertainty=uncertainty)
        # Sort by score desc; ties keep insertion order (Python sort is stable).
        return sorted(actions, key=lambda a: a.score, reverse=True)


def _clamp(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _score_to_priority(score: float) -> int:
    """Map a score (roughly −2..+3) onto the scheduler's 0..99 priority band.

    Kept monotonic so a higher score always yields a higher scheduler priority; the
    exact mapping is not important, only the ordering (§17).
    """
    p = int(round(50 + score * 18))
    return max(1, min(99, p))


# ─────────────────────────────────────────────────────────────────────────── #
# §11 Information gain & §12 exploration/exploitation
# ─────────────────────────────────────────────────────────────────────────── #

def mission_uncertainty(mission_state: Any) -> float:
    """Estimate how much is still *unknown* (0 = fully understood, 1 = know nothing).

    Deterministic heuristic (§11/§12): each dimension of understanding that is still
    empty contributes to uncertainty. A mission with services, endpoints and a known
    vulnerability is far more certain than a bare target — so the scorer tilts from
    exploration toward exploitation as these fill in.
    """
    signals = [
        bool(getattr(mission_state, "services", None) or getattr(mission_state, "known_services", None)),
        bool(getattr(mission_state, "technologies", None)),
        bool(getattr(mission_state, "endpoints", None) or getattr(mission_state, "known_endpoints", None)),
        bool(getattr(mission_state, "vulnerabilities", None)),
        bool(getattr(mission_state, "credentials", None)),
        bool(getattr(mission_state, "confirmed_facts", None) or getattr(mission_state, "facts", None)),
    ]
    known = sum(1 for s in signals if s)
    # A strong exploitation signal (a known vuln or creds) drives uncertainty down hard.
    exploit_ready = bool(getattr(mission_state, "vulnerabilities", None)) or \
        bool(getattr(mission_state, "credentials", None))
    base = 1.0 - (known / len(signals))
    if exploit_ready:
        base = min(base, 0.35)
    return round(_clamp(base), 4)


def information_gain_for(action_type: str, mission_state: Any) -> str:
    """Return the information-gain label for an action *given current knowledge* (§11).

    The reusable mechanism (not a hardcoded example): an action that would reduce an
    *open* uncertainty is HIGH gain; if that dimension is already understood it drops
    to LOW. This is what makes the planner prefer fingerprinting/enumeration when the
    surface is unknown, and stop valuing them once it is known.
    """
    prof = profile_for(action_type)
    dim = prof.reduces_uncertainty_about
    base = prof.information_gain
    if not dim:
        return base

    filled = {
        "service": bool(getattr(mission_state, "services", None)
                        or getattr(mission_state, "known_services", None)
                        or getattr(mission_state, "technologies", None)),
        "surface": bool(getattr(mission_state, "endpoints", None)
                        or getattr(mission_state, "known_endpoints", None)),
        "vuln": bool(getattr(mission_state, "vulnerabilities", None)),
        "artifact": bool(getattr(mission_state, "artifacts", None)),
    }.get(dim, False)

    if filled and dim in ("service", "surface"):
        # The uncertainty this action addresses is already resolved → low value now.
        return InformationGain.LOW.value
    if not filled and dim == "vuln":
        # Can't exploit a vuln we don't have yet → this action can't pay off yet.
        return InformationGain.LOW.value
    return base
