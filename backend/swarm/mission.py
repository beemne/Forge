"""Shared Mission State (Phase 4 §7).

The central, shared, persistent aggregate of everything the swarm has learned —
distinct from each agent's *isolated* per-session ``MissionState``. Specialist
agents keep their own trajectories private; what they discover is promoted here
(via the Evidence Bus) so the whole team benefits without any agent receiving
another agent's entire conversation.

Persisted to the durable ``swarm_missions`` table so a mission resumes with all
aggregated knowledge intact.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from backend.utils.time import utcnow
from typing import Any, Dict, List, Optional

from backend.swarm.dedup import normalize_text
from backend.swarm.reasoning import (
    Fact, Hypothesis, HypothesisStatus, FailedApproach, Reliability,
)

# §4 — state must remain bounded. Caps on the Phase 5 reasoning collections so a long
# mission cannot grow the shared state without limit. Oldest entries are trimmed.
_MAX_FACTS = 120
_MAX_HYPOTHESES = 60
_MAX_FAILED_APPROACHES = 150
_MAX_ACTION_SIGNATURES = 400
_MAX_CANDIDATES = 40


def _add_unique(items: List[str], value: Optional[str]) -> bool:
    v = (value or "").strip()
    if v and v not in items:
        items.append(v)
        return True
    return False


def _bounded_append(items: List[Any], value: Any, cap: int) -> None:
    items.append(value)
    if len(items) > cap:
        del items[: len(items) - cap]


@dataclass
class SharedMissionState:
    """Serializable, resumable source of truth for a coordinated mission."""

    mission_id: str = ""
    run_id: Optional[str] = None
    challenge_id: Optional[str] = None

    # ── Challenge / target ──────────────────────────────────────────────────
    challenge_name: str = ""
    category: str = ""
    difficulty: str = ""
    platform: str = ""
    description: str = ""
    flag_format: str = ""
    target: str = ""
    target_type: str = ""          # Phase 4.x: detected TargetType of `target` (observability/§11)
    scope: List[str] = field(default_factory=list)

    # ── Aggregated knowledge (promoted from agents via evidence) ────────────
    endpoints: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    credentials: List[str] = field(default_factory=list)
    cookies: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    vulnerabilities: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)  # legacy free-text hypotheses (kept)

    # ── Objective (§4) ──────────────────────────────────────────────────────
    objective: str = ""

    # ── Phase 5 §5: facts vs hypotheses (structured, distinct) ──────────────
    # Stored as serialized dicts so to_dict/from_dict/JSON persistence need no custom
    # codec; use the typed accessors (get_facts / get_hypotheses / ...) to work with
    # them as dataclasses.
    confirmed_facts: List[Dict[str, Any]] = field(default_factory=list)
    hypothesis_records: List[Dict[str, Any]] = field(default_factory=list)

    # ── Phase 5 §7: failed-approach memory (structured, bounded) ────────────
    failed_approaches: List[Dict[str, Any]] = field(default_factory=list)
    # §8 normalized ACTION signatures already attempted (distinct from task sigs below).
    action_signatures: List[str] = field(default_factory=list)

    # ── Phase 5 §9/§37: last computed candidate actions (observability) ─────
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)

    # ── Phase 5 §27: mission budget snapshot + §25 stop reason ──────────────
    mission_budget: Dict[str, Any] = field(default_factory=dict)
    stop_condition: str = ""

    # ── Technique bookkeeping ───────────────────────────────────────────────
    attempted_techniques: List[str] = field(default_factory=list)
    successful_techniques: List[str] = field(default_factory=list)
    failed_techniques: List[str] = field(default_factory=list)
    dead_ends: List[str] = field(default_factory=list)
    exhausted_strategies: List[str] = field(default_factory=list)
    attempted_signatures: List[str] = field(default_factory=list)  # normalized cmd/task dedup

    # ── Flags ───────────────────────────────────────────────────────────────
    flag_candidates: List[str] = field(default_factory=list)
    verified_flag: Optional[str] = None

    # ── Coordination ────────────────────────────────────────────────────────
    strategy: str = ""
    phase: str = "recon"
    progress: int = 0
    status: str = "PLANNING"                       # PLANNING|RUNNING|PAUSED|COMPLETED|FAILED|CANCELLED
    agent_statuses: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # agent_id -> {...}
    evidence_ids: List[str] = field(default_factory=list)
    updated_at: str = ""

    # ------------------------------------------------------------------ #
    # Evidence integration (the only path knowledge enters shared state)
    # ------------------------------------------------------------------ #

    def integrate_evidence(self, ev: Any) -> None:
        """Merge one :class:`~backend.swarm.evidence.Evidence` into shared knowledge."""
        etype = getattr(ev, "evidence_type", "note")
        title = getattr(ev, "title", "") or ""
        desc = getattr(ev, "description", "") or ""
        label = (title or desc).strip()

        if getattr(ev, "id", ""):
            _add_unique(self.evidence_ids, ev.id)

        if etype == "endpoint":
            _add_unique(self.endpoints, getattr(ev, "related_endpoint", "") or label)
        elif etype == "service":
            _add_unique(self.services, label)
        elif etype == "technology":
            _add_unique(self.technologies, getattr(ev, "related_technology", "") or label)
        elif etype == "credential":
            _add_unique(self.credentials, label)
        elif etype == "vulnerability":
            _add_unique(self.vulnerabilities, getattr(ev, "related_vulnerability", "") or label)
        elif etype == "artifact":
            _add_unique(self.artifacts, getattr(ev, "artifact_id", None) or label)
        elif etype == "flag":
            _add_unique(self.flag_candidates, label)
        elif etype == "failure":
            _add_unique(self.failed_techniques, label)

        # Cross-reference fields can carry a lead regardless of the primary type.
        _add_unique(self.endpoints, getattr(ev, "related_endpoint", ""))
        _add_unique(self.technologies, getattr(ev, "related_technology", ""))
        _add_unique(self.vulnerabilities, getattr(ev, "related_vulnerability", ""))

        # Phase 5 §5/§6 — split the discovery into a FACT or a HYPOTHESIS by reliability.
        # Strong (DIRECT/DERIVED) evidence that names something concrete becomes a
        # confirmed fact; speculative evidence becomes an OPEN hypothesis to validate.
        # Failure/capability/mismatch evidence is bookkeeping, not a knowledge claim.
        if label and etype not in ("failure", "capability", "target_mismatch", "note"):
            reliability = getattr(ev, "reliability", "") or Reliability.DERIVED.value
            strong = reliability in (Reliability.DIRECT.value, Reliability.DERIVED.value)
            stmt = f"{etype}: {label}"
            if strong:
                self.add_fact(stmt, reliability=reliability,
                              confidence=float(getattr(ev, "confidence", 0.8) or 0.8),
                              source=getattr(ev, "source", ""), evidence_id=getattr(ev, "id", ""),
                              agent=getattr(ev, "agent_id", ""))
            else:
                self.add_hypothesis(stmt, reliability=reliability,
                                    confidence=float(getattr(ev, "confidence", 0.4) or 0.4),
                                    rationale=(getattr(ev, "description", "") or "")[:200],
                                    evidence_id=getattr(ev, "id", ""),
                                    agent=getattr(ev, "agent_id", ""))
        self._touch()

    # ── Explicit coordination mutators ──────────────────────────────── #

    def note_attempt(self, signature: str) -> None:
        _add_unique(self.attempted_signatures, signature)
        self._touch()

    def record_dead_end(self, note: str) -> None:
        _add_unique(self.dead_ends, note)
        self._touch()

    # ── Phase 7: authoritative target reconciliation on resume (STEP 2/5) ── #

    def adopt_authoritative_target(self, new_target: str, stale_hosts: List[str]) -> Dict[str, int]:
        """Make *new_target* authoritative and invalidate state tied to the OLD target.

        On resume with a changed target, host-specific knowledge and attempt history
        from the previous target must not silently drive execution against it. Only
        state that references a stale host is dropped — relative paths, techniques, and
        non-host facts stay useful and are kept. Attempt/failure signatures tied to a
        stale host are cleared too, so the swarm may legitimately RE-TRY equivalent
        actions against the new target instead of being suppressed by duplicate/
        failed-action memory. Flag candidates and the audit dead-end trail are never
        dropped. Returns per-field invalidation counts.
        """
        from backend.swarm.target_reconciliation import references_stale_host
        old = self.target
        self.target = new_target
        # target_type is re-derived by the coordinator against the new target.
        self.target_type = ""

        def _refs(obj: Any) -> bool:
            if isinstance(obj, dict):
                blob = " ".join(str(obj.get(k, "")) for k in
                                ("statement", "action", "target", "signature", "name",
                                 "value", "rationale", "endpoint", "title"))
            else:
                blob = str(obj)
            return references_stale_host(blob, stale_hosts)

        def _prune(items: List[Any]) -> tuple:
            kept, dropped = [], 0
            for it in items:
                if _refs(it):
                    dropped += 1
                else:
                    kept.append(it)
            return kept, dropped

        counts: Dict[str, int] = {}
        self.endpoints, counts["endpoints"] = _prune(self.endpoints)
        self.services, counts["services"] = _prune(self.services)
        self.artifacts, counts["artifacts"] = _prune(self.artifacts)
        self.confirmed_facts, counts["facts"] = _prune(self.confirmed_facts)
        self.hypothesis_records, counts["hypotheses"] = _prune(self.hypothesis_records)
        self.failed_approaches, counts["failed_approaches"] = _prune(self.failed_approaches)
        self.action_signatures, sig_a = _prune(self.action_signatures)
        self.attempted_signatures, sig_b = _prune(self.attempted_signatures)
        counts["signatures"] = sig_a + sig_b

        # Record the change as a DIRECT (operator-supplied) fact + an audit note so the
        # trajectory/report shows exactly when and why stale state was invalidated.
        if old and old != new_target:
            try:
                self.add_fact(
                    f"Authoritative target changed to '{new_target}' (was '{old}')",
                    reliability=Reliability.DIRECT.value, confidence=1.0, source="operator")
            except Exception:
                pass
            _add_unique(self.dead_ends,
                        f"TARGET CHANGED: '{old}' -> '{new_target}' — stale host state invalidated")
        self._touch()
        return counts

    # ── Phase 5 §5: facts / hypotheses ──────────────────────────────── #

    def add_fact(self, statement: str, *, reliability: str = Reliability.DERIVED.value,
                 confidence: float = 0.8, source: str = "", evidence_id: str = "",
                 agent: str = "") -> bool:
        """Record a CONFIRMED fact (deduped by normalized statement). Returns True if new."""
        stmt = (statement or "").strip()
        if not stmt:
            return False
        key = normalize_text(stmt)
        for f in self.confirmed_facts:
            if normalize_text(f.get("statement", "")) == key:
                return False
        _bounded_append(self.confirmed_facts, Fact(
            statement=stmt, reliability=reliability, confidence=confidence,
            source=source, evidence_id=evidence_id, agent=agent).to_dict(), _MAX_FACTS)
        self._touch()
        return True

    def add_hypothesis(self, statement: str, *, confidence: float = 0.4,
                       reliability: str = Reliability.SPECULATIVE.value, rationale: str = "",
                       evidence_id: str = "", agent: str = "") -> Optional[Dict[str, Any]]:
        """Record an OPEN hypothesis (deduped). Never overwrites a confirmed fact."""
        stmt = (statement or "").strip()
        if not stmt:
            return None
        key = normalize_text(stmt)
        # If this is already a confirmed fact, do not shadow it with a hypothesis.
        if any(normalize_text(f.get("statement", "")) == key for f in self.confirmed_facts):
            return None
        for h in self.hypothesis_records:
            if normalize_text(h.get("statement", "")) == key:
                if evidence_id and evidence_id not in h.get("supporting_evidence", []):
                    h.setdefault("supporting_evidence", []).append(evidence_id)
                    h["updated_at"] = datetime.now(timezone.utc).isoformat()
                return h
        rec = Hypothesis(statement=stmt, confidence=confidence, reliability=reliability,
                         rationale=rationale, agent=agent,
                         supporting_evidence=[evidence_id] if evidence_id else []).to_dict()
        _bounded_append(self.hypothesis_records, rec, _MAX_HYPOTHESES)
        self._touch()
        return rec

    def _set_hypothesis_status(self, statement: str, status: str,
                               *, evidence_id: str = "") -> bool:
        key = normalize_text(statement)
        for h in self.hypothesis_records:
            if normalize_text(h.get("statement", "")) == key:
                h["status"] = status
                h["updated_at"] = datetime.now(timezone.utc).isoformat()
                bucket = "supporting_evidence" if status == HypothesisStatus.CONFIRMED.value else "refuting_evidence"
                if evidence_id:
                    h.setdefault(bucket, []).append(evidence_id)
                self._touch()
                return True
        return False

    def confirm_hypothesis(self, statement: str, *, evidence_id: str = "",
                           promote_to_fact: bool = True) -> bool:
        """Validate an open hypothesis (§5). Optionally promote it to a confirmed fact."""
        ok = self._set_hypothesis_status(statement, HypothesisStatus.CONFIRMED.value,
                                         evidence_id=evidence_id)
        if ok and promote_to_fact:
            self.add_fact(statement, reliability=Reliability.DERIVED.value, confidence=0.85,
                          source="validation", evidence_id=evidence_id)
        return ok

    def reject_hypothesis(self, statement: str, *, evidence_id: str = "") -> bool:
        return self._set_hypothesis_status(statement, HypothesisStatus.REJECTED.value,
                                           evidence_id=evidence_id)

    def get_facts(self) -> List[Fact]:
        return [Fact.from_dict(d) for d in self.confirmed_facts]

    def get_hypotheses(self, *, open_only: bool = False) -> List[Hypothesis]:
        out = [Hypothesis.from_dict(d) for d in self.hypothesis_records]
        return [h for h in out if h.is_open] if open_only else out

    # ── Phase 5 §7/§8: failed-approach memory + action dedup ────────── #

    def record_failed_approach(self, *, action: str, signature: str = "", capability: str = "",
                               target: str = "", params: str = "", result: str = "",
                               reason: str = "", failure_class: str = "", agent: str = "") -> bool:
        """Remember a failed approach for THIS mission (§7). Deduped by signature."""
        sig = signature or ""
        if sig and any(fa.get("signature") == sig for fa in self.failed_approaches):
            # Already recorded — refresh the reason/result (latest wins) but do not grow.
            for fa in self.failed_approaches:
                if fa.get("signature") == sig:
                    fa["result"] = result or fa.get("result", "")
                    fa["reason"] = reason or fa.get("reason", "")
                    fa["failure_class"] = failure_class or fa.get("failure_class", "")
            return False
        _bounded_append(self.failed_approaches, FailedApproach(
            action=action, signature=sig, capability=capability, target=target, params=params,
            result=result, reason=reason, failure_class=failure_class, agent=agent).to_dict(),
            _MAX_FAILED_APPROACHES)
        if sig:
            self.record_action_signature(sig)
        self._touch()
        # Cross-mission record (A4): share this dead-end with the other engine / future
        # missions of the same category. Best-effort, advisory, never raises.
        try:
            from backend.knowledge.failed_approaches import record_failed_approach as _xmission_record
            _xmission_record(self.category, sig or action, reason or result or failure_class)
        except Exception:
            pass
        return True

    def has_failed_action(self, signature: str) -> bool:
        return bool(signature) and any(fa.get("signature") == signature for fa in self.failed_approaches)

    def get_failed_approaches(self) -> List[FailedApproach]:
        return [FailedApproach.from_dict(d) for d in self.failed_approaches]

    def record_action_signature(self, signature: str) -> None:
        """Record that an ACTION (capability::target::params) was attempted (§8)."""
        if signature and signature not in self.action_signatures:
            _bounded_append(self.action_signatures, signature, _MAX_ACTION_SIGNATURES)
            self._touch()

    def set_candidate_actions(self, candidates: List[Any]) -> None:
        """Store the latest ranked candidate actions for observability (§37). Bounded."""
        out: List[Dict[str, Any]] = []
        for c in (candidates or [])[:_MAX_CANDIDATES]:
            out.append(c.to_dict() if hasattr(c, "to_dict") else dict(c))
        self.candidate_actions = out
        self._touch()

    def set_objective(self, objective: str) -> None:
        if objective and objective != self.objective:
            self.objective = objective
            self._touch()

    def set_agent_status(self, agent_id: str, **fields: Any) -> None:
        cur = self.agent_statuses.get(agent_id, {})
        cur.update(fields)
        self.agent_statuses[agent_id] = cur
        self._touch()

    def set_verified_flag(self, flag: str) -> None:
        self.verified_flag = flag
        _add_unique(self.flag_candidates, flag)
        self.progress = 100
        self.status = "COMPLETED"
        self._touch()

    def recompute_progress(self) -> None:
        if self.verified_flag:
            self.progress = 100
            return
        score = 0
        if self.endpoints:
            score += 15
        if self.technologies:
            score += 10
        if self.services:
            score += 10
        if self.vulnerabilities:
            score += 25
        if self.credentials:
            score += 15
        if self.flag_candidates:
            score += 20
        self.progress = max(self.progress, min(score, 95))

    # ── Bounded summary for prompts / API ────────────────────────────── #

    def summary(self, max_items: int = 8) -> str:
        def _fmt(name: str, items: List[str]) -> str:
            if not items:
                return ""
            shown = items[:max_items]
            more = f" (+{len(items) - len(shown)} more)" if len(items) > len(shown) else ""
            return f"{name}: {', '.join(shown)}{more}"

        lines = [
            f"Target: {self.target or 'n/a'} | category={self.category or 'n/a'} | phase={self.phase}",
        ]
        if self.objective:
            lines.append(f"Objective: {self.objective}")
        # Phase 5 §22 — confirmed facts and open hypotheses are the most useful bounded
        # context for a specialist; failed approaches keep it from repeating dead ends.
        facts = [f.get("statement", "") for f in self.confirmed_facts]
        row = _fmt("Confirmed facts", facts)
        if row:
            lines.append(row)
        open_hyps = [h.get("statement", "") for h in self.hypothesis_records
                     if h.get("status") == "open"]
        row = _fmt("Open hypotheses", open_hyps)
        if row:
            lines.append(row)
        for name, items in (
            ("Endpoints", self.endpoints), ("Technologies", self.technologies),
            ("Services", self.services), ("Vulnerabilities", self.vulnerabilities),
            ("Credentials", self.credentials), ("Artifacts", self.artifacts),
            ("Dead ends", self.dead_ends), ("Flag candidates", self.flag_candidates),
        ):
            row = _fmt(name, items)
            if row:
                lines.append(row)
        failed = [f"{fa.get('action', '')} → {fa.get('result') or fa.get('failure_class', '')}"
                  for fa in self.failed_approaches]
        row = _fmt("Failed approaches (do not repeat without new evidence)", failed)
        if row:
            lines.append(row)
        if self.verified_flag:
            lines.append(f"VERIFIED FLAG: {self.verified_flag}")
        return "\n".join(lines)

    # ── Serialization / persistence ──────────────────────────────────── #

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SharedMissionState":
        data = data or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, *, coord_session_id: Optional[str] = None) -> None:
        try:
            from backend.database.session import SessionLocal
            from backend.database.models import SwarmMissionModel
            self.recompute_progress()
            db = SessionLocal()
            try:
                row = db.query(SwarmMissionModel).filter(SwarmMissionModel.id == self.mission_id).first()
                if row is None:
                    row = SwarmMissionModel(id=self.mission_id)
                    db.add(row)
                row.run_id = self.run_id
                row.challenge_id = self.challenge_id
                if coord_session_id:
                    row.coord_session_id = coord_session_id
                row.status = self.status
                row.strategy = self.strategy or ""
                row.progress = int(self.progress)
                row.shared_state = self.to_dict()
                row.verified_flag = self.verified_flag
                if self.status in ("COMPLETED", "FAILED", "CANCELLED"):
                    row.completed_at = utcnow()
                db.commit()
            finally:
                db.close()
        except Exception:
            pass

    @classmethod
    def load(cls, mission_id: str) -> Optional["SharedMissionState"]:
        try:
            from backend.database.session import SessionLocal
            from backend.database.models import SwarmMissionModel
            db = SessionLocal()
            try:
                row = db.query(SwarmMissionModel).filter(SwarmMissionModel.id == mission_id).first()
                if not row:
                    return None
                state = cls.from_dict(row.shared_state or {})
                state.mission_id = row.id
                state.run_id = row.run_id
                state.challenge_id = row.challenge_id
                state.status = row.status or state.status
                state.verified_flag = row.verified_flag or state.verified_flag
                return state
            finally:
                db.close()
        except Exception:
            return None

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
