import logging
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from backend.database.models import AuditLogModel

logger = logging.getLogger("forge.privilege")

class PrivilegeManager:
    """Evaluates privilege requirements and maintains tamper-evident audit logs."""

    def get_approval_mode(self, challenge_id: Optional[str] = None, db: Optional[Session] = None) -> str:
        """Resolve approval mode ('auto' or 'manual') for a challenge, falling back to global settings."""
        from backend.config import settings
        if challenge_id and db:
            try:
                from backend.database.models import ChallengeModel
                ch = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
                if ch and ch.approval_mode:
                    c_mode = str(ch.approval_mode).strip().lower()
                    if c_mode in ("auto", "manual"):
                        return c_mode
            except Exception as e:
                logger.debug(f"get_approval_mode lookup failed: {e}")

        if getattr(settings, "AUTO_APPROVE_PRIVILEGED", False):
            return "auto"
        mode = str(getattr(settings, "FORGE_APPROVAL_MODE", "manual") or "").strip().lower()
        return mode if mode in ("auto", "manual") else "manual"

    def is_auto_approved(self, challenge_id: Optional[str] = None, db: Optional[Session] = None) -> bool:
        """True if the effective approval mode for this challenge (or global fallback) is 'auto'."""
        return self.get_approval_mode(challenge_id=challenge_id, db=db) == "auto"

    def evaluate_privilege(
        self,
        agent: str,
        tool_name: str,
        privilege_level: str,
        db: Session,
        challenge_id: Optional[str] = None,
    ) -> bool:
        """Determines if execution is permitted under security policy.

        Backward-compatible bare-bool API preserved for callers that only need the
        allow/deny answer (e.g. acquisition.py, competition harness, classification
        tests). Callers that participate in the async operator-approval workflow and
        must later reconcile the audit row with the REAL decision should instead use
        ``evaluate_privilege_ex()`` (to obtain the row id) plus
        ``record_privilege_decision()`` (to update it).
        """
        approved, _audit_log_id = self.evaluate_privilege_ex(
            agent=agent, tool_name=tool_name, privilege_level=privilege_level, db=db, challenge_id=challenge_id
        )
        return approved

    def evaluate_privilege_ex(
        self,
        agent: str,
        tool_name: str,
        privilege_level: str,
        db: Session,
        challenge_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Like :meth:`evaluate_privilege`, but also returns the AuditLogModel row id.

        SAFE                    -> (True, id).  The row is FINAL at write time.
        PRIVILEGED / DANGEROUS  -> (False, id). The row only records that a check was
                                   REQUESTED; the actual approve/deny outcome is decided
                                   later by the async operator-approval workflow, which
                                   MUST call :meth:`record_privilege_decision` with this
                                   id once the real decision is known. Without that
                                   follow-up, the row would falsely read ``approved=False``
                                   for a command that ultimately ran with operator consent.
        """
        approved = False

        if privilege_level == "SAFE":
            approved = True
        elif privilege_level in ["PRIVILEGED", "DANGEROUS"]:
            # Requires operator session approval (defaults to pending approval in automated flow)
            approved = False

        # Log audit trail
        audit_entry = AuditLogModel(
            agent=agent,
            action=f"execute_tool:{tool_name}",
            privilege_level=privilege_level,
            approved=approved,
            details={"tool": tool_name}
        )
        db.add(audit_entry)
        # Flush (not just commit) so the generate_uuid primary-key default is applied
        # and readable on the live instance BEFORE expire-on-commit could expire it.
        db.flush()
        audit_log_id = audit_entry.id
        db.commit()

        logger.info(f"Privilege check for agent '{agent}' requesting tool '{tool_name}' ({privilege_level}): approved={approved}")
        return approved, audit_log_id

    def record_privilege_decision(
        self,
        audit_log_id: Optional[str],
        approved: bool,
        db: Session
    ) -> bool:
        """Reconcile an existing AuditLogModel row with the FINAL operator decision.

        Called once the async approval workflow resolves (approve / deny / timeout) so
        the audit trail matches reality:
          * approve        -> flips the classification-time ``approved=False`` to True.
          * deny / timeout -> ``approved`` stays False; the row is already correct, so
                              this explicitly CONFIRMS it rather than assuming — no write
                              is issued when the stored value already matches.

        Returns True if a matching row was found (and left in the correct state), False
        if ``audit_log_id`` is missing or no such row exists. Never raises: any failure
        is logged and rolled back so audit reconciliation can never abort a live run.
        """
        if not audit_log_id:
            return False
        try:
            row = (
                db.query(AuditLogModel)
                .filter(AuditLogModel.id == audit_log_id)
                .first()
            )
            if row is None:
                logger.debug(f"record_privilege_decision: no audit row found for id={audit_log_id}")
                return False
            if bool(row.approved) != bool(approved):
                row.approved = approved
                db.commit()
                logger.info(f"Audit row {audit_log_id} reconciled to approved={approved}")
            else:
                # Already correct (the common deny/timeout case) — confirm, don't rewrite.
                logger.debug(f"Audit row {audit_log_id} already approved={approved}; no update needed")
            return True
        except Exception as e:  # pragma: no cover - defensive; never break a live run
            logger.debug(f"record_privilege_decision failed for id={audit_log_id}: {e}")
            try:
                db.rollback()
            except Exception:
                pass
            return False

privilege_manager = PrivilegeManager()
