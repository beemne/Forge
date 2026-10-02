"""
FORGE Experience Memory store.

The persistent, self-improving memory of what FORGE has actually done. It is the
FORGE-experience half of the knowledge system that sits alongside the external
:class:`PlaybookVault` (rule §10):

    KNOWLEDGE
      ├── External Knowledge   → PlaybookVault (writeups, repos, curated)
      └── FORGE Experience     → ExperienceMemory (this module)

Responsibilities:
* Persist generalized :class:`ExperienceRecord`s to the ``experiences`` /
  ``experience_attempts`` tables (source of truth) (§18).
* Maintain a fast in-memory SQLite FTS5 index for retrieval (mirrors the proven
  PlaybookVault approach — no new heavyweight dependency) (§6, §17).
* Track the learning-flywheel statistics — times_retrieved / used / successful /
  failed, success_rate, confidence, last_used (§12).
* Promote repeatedly-successful, high-confidence experiences into reusable
  Playbook Vault entries (§10).
* Emit WebSocket telemetry for memory events (§16).

Nothing here executes commands or touches the tool/privilege managers — memory
only records and recommends (rule §20).
"""

from __future__ import annotations

import re
import sqlite3
import logging
from datetime import datetime, timezone
from backend.utils.time import utcnow
from typing import Any, Dict, List, Optional

from backend.database.session import SessionLocal
from backend.database.models import ExperienceModel, ExperienceAttemptModel, MemoryUsageModel
from backend.knowledge.memory_models import ExperienceRecord, SOURCE_WEIGHTS

logger = logging.getLogger("forge.experience_memory")


def _broadcast_memory_event(event: str, experience_id: str, extra: Optional[Dict] = None):
    """Fire-and-forget WebSocket broadcast for memory events (§16). Never raises."""
    try:
        import asyncio
        from backend.websocket.manager import ws_manager
        payload = {
            "event": event,                    # MEMORY_CREATED | MEMORY_RETRIEVED | ...
            "experience_id": experience_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(ws_manager.broadcast(payload))
        except RuntimeError:
            pass  # No running loop (CLI/test) — skip broadcast silently.
    except Exception:
        pass


class ExperienceMemory:
    """Store + FTS index + learning statistics for FORGE experiences."""

    # Promotion thresholds (§10). A repeatedly-successful or very-high-confidence
    # experience graduates into the reusable Playbook Vault.
    PROMOTE_MIN_SUCCESSES = 2
    PROMOTE_MIN_CONFIDENCE = 0.9

    def __init__(self, vault: Any = None, auto_load: bool = True):
        # Lazy import to avoid an import cycle at module load; the global vault is
        # the default promotion sink, but tests can inject an isolated vault.
        if vault is None:
            from backend.knowledge.playbook_vault import playbook_vault
            vault = playbook_vault
        self.vault = vault
        self.db_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_fts()
        if auto_load:
            self.reload_index()

    # ------------------------------------------------------------------ #
    # FTS index
    # ------------------------------------------------------------------ #

    def _init_fts(self):
        with self.db_conn:
            self.db_conn.execute("DROP TABLE IF EXISTS experience_fts")
            self.db_conn.execute("""
                CREATE VIRTUAL TABLE experience_fts USING fts5(
                    id UNINDEXED,
                    category,
                    technique,
                    tags,
                    technologies,
                    keywords,
                    outcome UNINDEXED,
                    confidence UNINDEXED,
                    success_rate UNINDEXED,
                    times_successful UNINDEXED,
                    created_at UNINDEXED
                )
            """)

    def reload_index(self):
        """Rebuild the in-memory FTS index from the persistent experiences table."""
        self._init_fts()
        db = SessionLocal()
        try:
            rows = db.query(ExperienceModel).all()
            for exp in rows:
                self._index_experience(exp)
            logger.info(f"[ExperienceMemory] Indexed {len(rows)} experiences into FTS.")
        except Exception as e:
            logger.warning(f"[ExperienceMemory] reload_index skipped: {e}")
        finally:
            db.close()

    def _index_experience(self, exp: ExperienceModel):
        keywords = " ".join(filter(None, [
            exp.observed_conditions or "", exp.applicable_conditions or "",
            exp.generalized_strategy or "", " ".join(exp.vulnerabilities or []),
            " ".join(str(x) for x in (exp.success_indicators or [])),
            " ".join(getattr(exp, "required_tools", None) or []),
        ]))
        with self.db_conn:
            self.db_conn.execute("DELETE FROM experience_fts WHERE id = ?", (exp.id,))
            self.db_conn.execute(
                """INSERT INTO experience_fts(
                    id, category, technique, tags, technologies, keywords,
                    outcome, confidence, success_rate, times_successful, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    exp.id, exp.category or "", exp.technique or "",
                    " ".join(exp.tags or []), " ".join(exp.technologies or []), keywords,
                    exp.outcome or "success", float(exp.confidence or 0.0),
                    float(exp.success_rate or 0.0), int(exp.times_successful or 0),
                    (exp.created_at or utcnow()).isoformat(),
                ),
            )

    # ------------------------------------------------------------------ #
    # Store
    # ------------------------------------------------------------------ #

    def store(self, record: ExperienceRecord) -> Optional[str]:
        """Persist a generalized experience + its attempts, index it, broadcast, and
        auto-promote if it already meets the bar. Returns the experience id."""
        db = SessionLocal()
        try:
            exp = ExperienceModel(
                source=record.source or "forge_run",
                source_run_id=record.source_run_id,
                source_challenge_id=record.source_challenge_id,
                challenge_name=record.challenge_name or "",
                category=(record.category or "web").lower(),
                difficulty=record.difficulty or "MEDIUM",
                technique=record.technique or "solve methodology",
                tags=record.tags or [],
                target_characteristics=record.target_characteristics or {},
                initial_observations=record.initial_observations or "",
                observed_conditions=record.observed_conditions or "",
                applicable_conditions=record.applicable_conditions or "",
                discovered_endpoints=record.discovered_endpoints or [],
                technologies=record.technologies or [],
                vulnerabilities=record.vulnerabilities or [],
                successful_techniques=record.successful_techniques or [],
                failed_techniques=record.failed_techniques or [],
                commands_used=record.commands_used or [],
                important_tool_outputs=record.important_tool_outputs or [],
                successful_attack_chain=record.successful_attack_chain or [],
                verification_evidence=record.verification_evidence or "",
                success_indicators=record.success_indicators or [],
                prerequisites=record.prerequisites or [],
                generalized_strategy=record.generalized_strategy or "",
                detection_indicators=record.detection_indicators or {},
                required_os=(getattr(record, "required_os", None) or "any"),
                required_tools=getattr(record, "required_tools", None) or [],
                required_python_libs=getattr(record, "required_python_libs", None) or [],
                outcome=record.outcome or "success",
                confidence=float(record.confidence or 0.6),
                success_rate=1.0 if (record.outcome or "success") == "success" else 0.0,
                times_successful=1 if (record.outcome or "success") == "success" else 0,
                times_failed=0 if (record.outcome or "success") == "success" else 1,
            )
            db.add(exp)
            db.flush()  # assign exp.id
            for att in (record.attempts or []):
                db.add(ExperienceAttemptModel(
                    experience_id=exp.id, sequence=att.sequence, approach=att.approach,
                    technique=att.technique, outcome=att.outcome, reason=att.reason,
                    evidence=att.evidence,
                ))
            db.commit()
            exp_id = exp.id
            self._index_experience(exp)
            logger.info(f"[ExperienceMemory] Stored experience {exp_id} "
                        f"('{exp.technique}', outcome={exp.outcome}, conf={exp.confidence:.2f})")
            _broadcast_memory_event("MEMORY_CREATED", exp_id, {
                "technique": exp.technique, "category": exp.category, "outcome": exp.outcome,
            })
        except Exception as e:
            logger.error(f"[ExperienceMemory] store failed: {e}")
            db.rollback()
            db.close()
            return None
        finally:
            try:
                db.close()
            except Exception:
                pass

        # Promotion runs on its own session.
        try:
            self.maybe_promote(exp_id)
        except Exception as e:
            logger.debug(f"[ExperienceMemory] promotion check skipped: {e}")
        return exp_id

    # ------------------------------------------------------------------ #
    # Search + ranking
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str = "",
        category: Optional[str] = None,
        evidence: Optional[str] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 6,
        include_failures: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return the top-K most relevant experiences as dicts, ranked by relevance,
        proven success, confidence and provenance (§6, §7)."""
        terms = re.findall(r"[A-Za-z0-9_]{2,}", " ".join(filter(None, [query, evidence, " ".join(tags or [])])))
        # Drop ultra-generic stopwords that would match everything.
        stop = {"the", "and", "for", "http", "https", "com", "www", "with", "this", "that"}
        terms = [t for t in terms if t.lower() not in stop][:24]
        if not terms:
            return self._recent_fallback(category, top_k, include_failures)

        match_expr = " OR ".join(f'"{t}"' for t in terms)
        clauses = ["experience_fts MATCH ?"]
        params: List[Any] = [match_expr]
        if category:
            clauses.append("category = ?")
            params.append(category.lower())
        if not include_failures:
            clauses.append("outcome = 'success'")

        sql = f"""
            SELECT id, outcome, confidence, success_rate, times_successful, created_at,
                   bm25(experience_fts) AS rank
            FROM experience_fts
            WHERE {' AND '.join(clauses)}
            ORDER BY rank
            LIMIT 60
        """
        try:
            cur = self.db_conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
        except Exception as e:
            logger.warning(f"[ExperienceMemory] FTS search failed ('{query}'): {e}")
            return self._recent_fallback(category, top_k, include_failures)

        scored = []
        now = datetime.now(timezone.utc)
        for rid, outcome, conf, succ_rate, n_succ, created_at, raw_rank in rows:
            bm25_weight = 1.0 / (abs(raw_rank) + 0.001)
            src_weight = self._provenance_weight(outcome, int(n_succ or 0))
            recency = self._recency_factor(created_at, now)
            score = (bm25_weight
                     * (0.5 + 0.5 * float(conf or 0.0))
                     * (0.4 + 0.6 * float(succ_rate or 0.0))
                     * src_weight
                     * recency)
            scored.append((score, rid))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, rid in scored[:top_k]:
            row = self.get(rid)
            if row:
                row["_score"] = round(score, 4)
                results.append(row)
        return results

    @staticmethod
    def _provenance_weight(outcome: str, times_successful: int) -> float:
        if outcome == "success" and times_successful >= 2:
            return SOURCE_WEIGHTS["forge_repeated_success"]
        if outcome == "success":
            return SOURCE_WEIGHTS["forge_success"]
        return SOURCE_WEIGHTS["forge_failure"]

    @staticmethod
    def _recency_factor(created_at: str, now: datetime) -> float:
        """Mild recency preference — never dominates correctness. 1.0 (fresh) → 0.85 (old)."""
        try:
            ts = datetime.fromisoformat(created_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            return max(0.85, 1.0 - min(age_days, 60.0) / 400.0)
        except Exception:
            return 1.0

    def _recent_fallback(self, category, top_k, include_failures) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            q = db.query(ExperienceModel)
            if category:
                q = q.filter(ExperienceModel.category == category.lower())
            if not include_failures:
                q = q.filter(ExperienceModel.outcome == "success")
            rows = q.order_by(ExperienceModel.times_successful.desc(),
                              ExperienceModel.created_at.desc()).limit(top_k).all()
            return [self._row_to_dict(r) for r in rows]
        except Exception:
            return []
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Feedback loop (§12)
    # ------------------------------------------------------------------ #

    def record_retrieval(self, experience_ids: List[str], run_id: Optional[str] = None,
                         challenge_id: Optional[str] = None):
        """Bump times_retrieved and log a usage event for each retrieved memory."""
        if not experience_ids:
            return
        db = SessionLocal()
        try:
            for eid in experience_ids:
                exp = db.query(ExperienceModel).filter(ExperienceModel.id == eid).first()
                if not exp:
                    continue
                exp.times_retrieved = (exp.times_retrieved or 0) + 1
                db.add(MemoryUsageModel(experience_id=eid, run_id=run_id,
                                        challenge_id=challenge_id, event="retrieved"))
            db.commit()
        except Exception as e:
            logger.debug(f"[ExperienceMemory] record_retrieval skip: {e}")
            db.rollback()
        finally:
            db.close()

    def record_usage_event(self, experience_id: str, event: str, note: str = "",
                           run_id: Optional[str] = None, challenge_id: Optional[str] = None) -> bool:
        """Log a fine-grained memory-usage telemetry event WITHOUT changing stats (§14).

        Unlike :meth:`record_feedback` (which moves confidence/success_rate), this only
        appends a usage row so FORGE can later analyse "which memories actually help?" —
        e.g. event='contributed' (helped reach the flag) or 'contradicted' (real output
        disproved the memory's suggestion). Non-fatal.
        """
        if not experience_id or not event:
            return False
        db = SessionLocal()
        try:
            db.add(MemoryUsageModel(experience_id=experience_id, run_id=run_id,
                                    challenge_id=challenge_id, event=event, note=note[:500]))
            db.commit()
            return True
        except Exception as e:
            logger.debug(f"[ExperienceMemory] record_usage_event skip: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def record_feedback(self, experience_id: str, success: bool, note: str = "",
                        run_id: Optional[str] = None, challenge_id: Optional[str] = None) -> bool:
        """Record that a retrieved memory was USED and whether it helped (§12).

        Updates times_used / times_successful|failed, EWMA success_rate, confidence,
        and last_used, then re-checks promotion. Returns True on success."""
        db = SessionLocal()
        try:
            exp = db.query(ExperienceModel).filter(ExperienceModel.id == experience_id).first()
            if not exp:
                return False
            exp.times_used = (exp.times_used or 0) + 1
            if success:
                exp.times_successful = (exp.times_successful or 0) + 1
                exp.confidence = min(1.0, round((exp.confidence or 0.6) + 0.1, 3))
            else:
                exp.times_failed = (exp.times_failed or 0) + 1
                exp.confidence = max(0.1, round((exp.confidence or 0.6) - 0.08, 3))
            # EWMA success rate.
            alpha = 0.3
            target = 1.0 if success else 0.0
            exp.success_rate = round((1 - alpha) * float(exp.success_rate or 0.0) + alpha * target, 3)
            exp.last_used = utcnow()
            db.add(MemoryUsageModel(experience_id=experience_id, run_id=run_id, challenge_id=challenge_id,
                                    event="success" if success else "failure", note=note[:500]))
            db.commit()
            self._index_experience(exp)
            _broadcast_memory_event("MEMORY_UPDATED", experience_id, {
                "times_used": exp.times_used, "times_successful": exp.times_successful,
                "success_rate": exp.success_rate, "confidence": exp.confidence,
            })
        except Exception as e:
            logger.warning(f"[ExperienceMemory] record_feedback failed: {e}")
            db.rollback()
            return False
        finally:
            db.close()
        try:
            self.maybe_promote(experience_id)
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------ #
    # Promotion to Playbook Vault (§10)
    # ------------------------------------------------------------------ #

    def maybe_promote(self, experience_id: str) -> Optional[str]:
        """Promote a proven experience into a reusable Playbook Vault entry.

        Criteria: verified success AND (repeated success OR very-high confidence),
        not already promoted. Returns the new/linked playbook id, else None."""
        db = SessionLocal()
        try:
            exp = db.query(ExperienceModel).filter(ExperienceModel.id == experience_id).first()
            if not exp or exp.promoted_playbook_id:
                return exp.promoted_playbook_id if exp else None
            if exp.outcome != "success":
                return None
            eligible = (int(exp.times_successful or 0) >= self.PROMOTE_MIN_SUCCESSES
                        or float(exp.confidence or 0.0) >= self.PROMOTE_MIN_CONFIDENCE)
            if not eligible:
                return None

            pb_id = self._promote_row(exp)
            if pb_id:
                exp.promoted_playbook_id = pb_id
                db.commit()
                self._index_experience(exp)
                _broadcast_memory_event("MEMORY_PROMOTED", experience_id, {"playbook_id": pb_id})
                logger.info(f"[ExperienceMemory] Promoted experience {experience_id} → playbook {pb_id}")
            return pb_id
        except Exception as e:
            logger.warning(f"[ExperienceMemory] maybe_promote failed: {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def _promote_row(self, exp: ExperienceModel) -> Optional[str]:
        """Build a PlaybookSchema from a generalized experience and save it to the vault."""
        try:
            from backend.knowledge.playbook_vault import PlaybookSchema, CATEGORIES
            category = (exp.category or "web").lower()
            if category not in CATEGORIES:
                category = "web"
            template_lines = []
            if exp.generalized_strategy:
                template_lines.append(f"# Strategy: {exp.generalized_strategy}")
            template_lines.extend(exp.successful_attack_chain or [])
            template = "\n".join(template_lines) or f"# {exp.technique}"

            pb_id = f"exp-{category}-{(exp.id or '')[:8]}"
            pb = PlaybookSchema(
                id=pb_id,
                category=category,
                tags=list(dict.fromkeys((exp.tags or []) + ["forge_experience", "auto_learned"])),
                trigger_signatures=list(exp.technologies or [])[:8],
                exploit_template=template,
                expected_outcome_signatures=[str(s) for s in (exp.success_indicators or [])][:6],
                notes=(f"Promoted from a verified FORGE solve. Technique: {exp.technique}. "
                       f"Applicable when: {exp.applicable_conditions}"),
                source="auto_generated",
                confidence_score=min(1.0, float(exp.confidence or 0.6)),
                times_used=int(exp.times_successful or 0),
                success_rate=float(exp.success_rate or 1.0),
                is_promoted=True,
                is_sanitized=True,
            )
            self.vault.save_playbook(pb)
            return pb_id
        except Exception as e:
            logger.warning(f"[ExperienceMemory] _promote_row failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def get(self, experience_id: str, with_children: bool = False) -> Optional[Dict[str, Any]]:
        db = SessionLocal()
        try:
            exp = db.query(ExperienceModel).filter(ExperienceModel.id == experience_id).first()
            if not exp:
                return None
            d = self._row_to_dict(exp)
            if with_children:
                d["attempts"] = [
                    {"sequence": a.sequence, "approach": a.approach, "technique": a.technique,
                     "outcome": a.outcome, "reason": a.reason, "evidence": a.evidence}
                    for a in sorted(exp.attempts, key=lambda x: x.sequence or 0)
                ]
                d["usage_log"] = [
                    {"event": u.event, "run_id": u.run_id, "challenge_id": u.challenge_id,
                     "note": u.note, "created_at": u.created_at.isoformat() if u.created_at else None}
                    for u in sorted(exp.usages, key=lambda x: x.created_at or utcnow())
                ]
            return d
        finally:
            db.close()

    def list_experiences(self, limit: int = 100, category: Optional[str] = None,
                         outcome: Optional[str] = None) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            q = db.query(ExperienceModel)
            if category:
                q = q.filter(ExperienceModel.category == category.lower())
            if outcome:
                q = q.filter(ExperienceModel.outcome == outcome)
            rows = q.order_by(ExperienceModel.created_at.desc()).limit(limit).all()
            return [self._row_to_dict(r) for r in rows]
        finally:
            db.close()

    def get_stats(self) -> Dict[str, Any]:
        """Summary counts + leaderboards for the Memory UI (§15)."""
        db = SessionLocal()
        try:
            rows = db.query(ExperienceModel).all()
            total = len(rows)
            successful = sum(1 for r in rows if r.outcome == "success")
            failed = total - successful
            high_conf = sum(1 for r in rows if (r.confidence or 0) >= 0.8)
            recent = sorted(rows, key=lambda r: r.created_at or datetime.min, reverse=True)[:8]
            most_successful = sorted(rows, key=lambda r: (r.times_successful or 0, r.success_rate or 0),
                                     reverse=True)[:8]
            used = [r for r in rows if r.last_used]
            recently_used = sorted(used, key=lambda r: r.last_used, reverse=True)[:8]
            return {
                "total_memories": total,
                "successful_experiences": successful,
                "failed_experiences": failed,
                "high_confidence_techniques": high_conf,
                "promoted_playbooks": sum(1 for r in rows if r.promoted_playbook_id),
                "recent_techniques": [self._brief(r) for r in recent],
                "most_successful_techniques": [self._brief(r) for r in most_successful],
                "recently_used_memories": [self._brief(r) for r in recently_used],
            }
        finally:
            db.close()

    @staticmethod
    def _brief(r: ExperienceModel) -> Dict[str, Any]:
        return {
            "id": r.id, "technique": r.technique, "category": r.category, "outcome": r.outcome,
            "confidence": round(r.confidence or 0.0, 3), "success_rate": round(r.success_rate or 0.0, 3),
            "times_used": r.times_used or 0, "times_successful": r.times_successful or 0,
            "last_used": r.last_used.isoformat() if r.last_used else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }

    @staticmethod
    def _row_to_dict(exp: ExperienceModel) -> Dict[str, Any]:
        return {
            "id": exp.id,
            "source": exp.source,
            "source_run_id": exp.source_run_id,
            "source_challenge_id": exp.source_challenge_id,
            "challenge_name": exp.challenge_name,
            "category": exp.category,
            "difficulty": exp.difficulty,
            "technique": exp.technique,
            "tags": exp.tags or [],
            "target_characteristics": exp.target_characteristics or {},
            "initial_observations": exp.initial_observations,
            "observed_conditions": exp.observed_conditions,
            "applicable_conditions": exp.applicable_conditions,
            "discovered_endpoints": exp.discovered_endpoints or [],
            "technologies": exp.technologies or [],
            "vulnerabilities": exp.vulnerabilities or [],
            "successful_techniques": exp.successful_techniques or [],
            "failed_techniques": exp.failed_techniques or [],
            "commands_used": exp.commands_used or [],
            "important_tool_outputs": exp.important_tool_outputs or [],
            "successful_attack_chain": exp.successful_attack_chain or [],
            "verification_evidence": exp.verification_evidence,
            "success_indicators": exp.success_indicators or [],
            "prerequisites": exp.prerequisites or [],
            "generalized_strategy": exp.generalized_strategy,
            "detection_indicators": exp.detection_indicators or {},
            "required_os": getattr(exp, "required_os", None) or "any",
            "required_tools": getattr(exp, "required_tools", None) or [],
            "required_python_libs": getattr(exp, "required_python_libs", None) or [],
            "outcome": exp.outcome,
            "confidence": round(exp.confidence or 0.0, 3),
            "times_retrieved": exp.times_retrieved or 0,
            "times_used": exp.times_used or 0,
            "times_successful": exp.times_successful or 0,
            "times_failed": exp.times_failed or 0,
            "success_rate": round(exp.success_rate or 0.0, 3),
            "last_used": exp.last_used.isoformat() if exp.last_used else None,
            "promoted_playbook_id": exp.promoted_playbook_id,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
        }


experience_memory = ExperienceMemory()
