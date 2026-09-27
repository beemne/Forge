"""
Phase 7 — Competition Hardening tests.

Focus: the genuine Phase-7 gaps closed on top of the mature Phase 1–6 system —

  * Authoritative TARGET RECONCILIATION on resume (STEP 2 / STEP 5 / STEP 10):
    a resumed mission must adopt the CURRENT target and never execute against a
    stale checkpoint target. Covered as a pure function, at the SharedMissionState
    level, at the coordinated-engine level (SwarmCoordinator._load_persisted), and
    at the default-engine level (SwarmBlackboard.reconcile_target).
  * Provider/quota OBSERVABILITY (STEP 4 / STEP 13): the quota manager exposes a
    real snapshot and the coordinator actually surfaces it.

Everything runs against the isolated unit-test database with NO API key, network, or
subprocess (same discipline as tests/test_phase4_swarm.py / test_phase5_reasoning.py).
"""
import os
import asyncio
import unittest

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.database.session import init_db, SessionLocal
from backend.database.models import (
    AgentSessionModel, TrajectoryEventModel, SwarmMissionModel, SwarmTaskModel,
    SwarmEvidenceModel,
)

from backend.swarm import (
    SharedMissionState, SwarmCoordinator, SwarmLimits, AgentRole, Task, AgentResult,
    reconcile_target, references_stale_host, hosts_of, TargetReconciliation,
)
from backend.swarm import events as swarm_events

# --------------------------------------------------------------------------- #
# Helpers / doubles
# --------------------------------------------------------------------------- #

def _run(coro):
    return asyncio.run(coro)

class ProgrammableAgent:
    """A SpecialistAgent double that records what ran and returns a scripted result."""
    def __init__(self, role, executed, status="COMPLETED"):
        self.role = role
        self._executed = executed
        self._status = status

    async def execute(self, task, mission, bus, **kw):
        rv = self.role.value if hasattr(self.role, "value") else str(self.role)
        self._executed.append((rv, task.objective))
        return AgentResult(task_id=task.id, role=rv, status=self._status,
                           session_id="s", verified_flag=None, flag_candidates=[],
                           evidence=[], reason="ok")

# --------------------------------------------------------------------------- #
# Helpers / doubles
# --------------------------------------------------------------------------- #

class _CaptureEvents:
    """Patch swarm_events.broadcast to record (event, payload) without a WS/loop."""
    def __init__(self):
        self.events = []

    def __enter__(self):
        self._orig = swarm_events.broadcast

        def rec(event, payload=None):
            self.events.append((event, payload or {}))
        swarm_events.broadcast = rec
        return self

    def __exit__(self, *a):
        swarm_events.broadcast = self._orig

    def names(self):
        return [e for e, _ in self.events]

class Phase7Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        db = SessionLocal()
        try:
            for model in (TrajectoryEventModel, AgentSessionModel, SwarmEvidenceModel,
                          SwarmTaskModel, SwarmMissionModel):
                db.query(model).delete()
            db.commit()
        finally:
            db.close()

# =========================================================================== #
# 1) reconcile_target — the pure decision (STEP 2 / STEP 5)                    #
# =========================================================================== #

class TestReconcilePure(Phase7Base):

    def test_unchanged_when_normalized_equal(self):
        # scheme + default port + trailing slash must not count as a change.
        r = reconcile_target("http://target.ctf:80/", "target.ctf")
        self.assertFalse(r.changed)
        self.assertEqual(r.stale_hosts, [])

    def test_empty_current_keeps_persisted(self):
        # A resume with no fresh target is a legitimate continuation — do NOT wipe.
        r = reconcile_target("", "10.10.14.1:1337")
        self.assertFalse(r.changed)
        self.assertEqual(r.authoritative, "10.10.14.1:1337")

    def test_ip_change_is_detected_with_stale_hosts(self):
        r = reconcile_target("http://10.10.14.2:8080", "http://10.10.14.1:8080")
        self.assertTrue(r.changed)
        self.assertEqual(r.authoritative, "http://10.10.14.2:8080")
        self.assertIn("10.10.14.1", r.stale_hosts)
        self.assertNotIn("10.10.14.2", r.stale_hosts)

    def test_hostname_change_is_detected(self):
        r = reconcile_target("http://box-b.htb", "http://box-a.htb")
        self.assertTrue(r.changed)
        self.assertTrue(any("box-a" in h for h in r.stale_hosts))

    def test_multi_target_plus_spec_reordering_is_not_a_change(self):
        r = reconcile_target("a.ctf + 10.0.0.5", "10.0.0.5 + a.ctf")
        self.assertFalse(r.changed)

    def test_multi_target_one_component_changed(self):
        r = reconcile_target("a.ctf + 10.0.0.6", "a.ctf + 10.0.0.5")
        self.assertTrue(r.changed)
        self.assertTrue(any("10.0.0.5" in h for h in r.stale_hosts))

    def test_references_stale_host_matches_old_endpoint_not_relative_path(self):
        r = reconcile_target("http://10.10.14.2", "http://10.10.14.1")
        self.assertTrue(references_stale_host("http://10.10.14.1/login", r.stale_hosts))
        self.assertFalse(references_stale_host("/login", r.stale_hosts))
        self.assertFalse(references_stale_host("nginx 1.18", r.stale_hosts))

    def test_reconciliation_serializes(self):
        r = reconcile_target("http://10.0.0.2", "http://10.0.0.1")
        d = r.to_dict()
        self.assertEqual(d["changed"], True)
        self.assertIn("stale_hosts", d)
        self.assertIsInstance(TargetReconciliation(changed=False, authoritative="x").to_dict(), dict)

# =========================================================================== #
# 2) SharedMissionState.adopt_authoritative_target (STEP 5)                    #
# =========================================================================== #

class TestMissionAdoptTarget(Phase7Base):

    def _mission(self):
        ms = SharedMissionState(mission_id="m7", target="http://10.10.14.1:8080", category="web")
        ms.endpoints = ["http://10.10.14.1:8080/login", "/dashboard",
                        "http://10.10.14.1:8080/api"]
        ms.services = ["10.10.14.1:22 OpenSSH", "nginx"]
        ms.technologies = ["nginx", "php"]
        ms.add_fact("service: OpenSSH 8.2 on 10.10.14.1:22", source="command")
        ms.add_fact("technology: nginx reverse proxy", source="command")
        ms.record_failed_approach(action="curl login", signature="web::10.10.14.1:8080/login::",
                                  target="http://10.10.14.1:8080/login", result="FAILED")
        return ms

    def test_prunes_host_bound_state_keeps_generic(self):
        ms = self._mission()
        r = reconcile_target("http://10.10.14.2:8080", ms.target)
        counts = ms.adopt_authoritative_target(r.authoritative, r.stale_hosts)
        self.assertEqual(ms.target, "http://10.10.14.2:8080")
        self.assertEqual(ms.target_type, "")  # re-derived by the coordinator later
        self.assertEqual(ms.endpoints, ["/dashboard"])          # relative path kept
        self.assertEqual(ms.services, ["nginx"])                # host-free service kept
        self.assertEqual(counts["endpoints"], 2)
        self.assertEqual(counts["services"], 1)
        self.assertEqual(counts["facts"], 1)                    # OpenSSH-on-old-IP dropped
        # generic tech fact survives
        self.assertTrue(any("nginx" in f.get("statement", "") for f in ms.confirmed_facts))

    def test_stale_failed_approach_cleared_so_reattempt_allowed(self):
        ms = self._mission()
        self.assertTrue(ms.has_failed_action("web::10.10.14.1:8080/login::"))
        r = reconcile_target("http://10.10.14.2:8080", ms.target)
        ms.adopt_authoritative_target(r.authoritative, r.stale_hosts)
        # The equivalent action against the NEW target must not be suppressed as "failed".
        self.assertFalse(ms.has_failed_action("web::10.10.14.1:8080/login::"))
        self.assertEqual(ms.failed_approaches, [])

    def test_records_change_as_fact_and_dead_end_note(self):
        ms = self._mission()
        r = reconcile_target("http://10.10.14.2:8080", ms.target)
        ms.adopt_authoritative_target(r.authoritative, r.stale_hosts)
        self.assertTrue(any("Authoritative target changed" in f.get("statement", "")
                            for f in ms.confirmed_facts))
        self.assertTrue(any("TARGET CHANGED" in d for d in ms.dead_ends))

    def test_flag_candidates_are_never_dropped(self):
        ms = self._mission()
        ms.flag_candidates = ["picoCTF{maybe_this_one}"]
        r = reconcile_target("http://10.10.14.2:8080", ms.target)
        ms.adopt_authoritative_target(r.authoritative, r.stale_hosts)
        self.assertIn("picoCTF{maybe_this_one}", ms.flag_candidates)

# =========================================================================== #
# 3) SwarmCoordinator resume reconciliation (STEP 2 / STEP 5 / STEP 10)        #
# =========================================================================== #

class TestCoordinatorResumeReconciliation(Phase7Base):

    def _persist_old_mission(self, mission_id, old_target):
        ms = SharedMissionState(mission_id=mission_id, run_id="r7", challenge_id="c7",
                                target=old_target, category="web", status="PAUSED")
        ms.endpoints = [f"{old_target}/login", "/robots.txt", f"{old_target}/admin"]
        ms.services = [f"{_host(old_target)}:22 OpenSSH"]
        ms.note_attempt("recon::" + old_target + "::")
        ms.save()
        return ms

    def _coord(self, mission_id, current_target):
        return SwarmCoordinator(
            run_id="r7", challenge_id="c7", target=current_target, category="web",
            challenge_name="Phase7 Resume", mission_id=mission_id, persist=True,
            limits=SwarmLimits(max_task_retries=0))

    def test_changed_target_becomes_authoritative_and_invalidates_stale(self):
        self._persist_old_mission("mres-1", "http://10.10.14.1:8080")
        coord = self._coord("mres-1", "http://10.10.14.9:8080")
        with _CaptureEvents() as cap:
            coord._load_persisted()
        # Current target wins; stale endpoints/services referencing the old host are gone.
        self.assertEqual(coord.mission.target, "http://10.10.14.9:8080")
        self.assertTrue(all("10.10.14.1" not in e for e in coord.mission.endpoints))
        self.assertIn("/robots.txt", coord.mission.endpoints)  # relative kept
        self.assertTrue(all("10.10.14.1" not in s for s in coord.mission.services))
        self.assertTrue(coord._last_reconciliation.get("changed"))
        self.assertIn(swarm_events.TARGET_CHANGED, cap.names())

    def test_same_target_resume_preserves_state_and_emits_no_change(self):
        self._persist_old_mission("mres-2", "http://10.10.14.1:8080")
        coord = self._coord("mres-2", "http://10.10.14.1:8080")
        with _CaptureEvents() as cap:
            coord._load_persisted()
        self.assertEqual(coord.mission.target, "http://10.10.14.1:8080")
        self.assertIn("http://10.10.14.1:8080/login", coord.mission.endpoints)  # preserved
        self.assertFalse(coord._last_reconciliation.get("changed"))
        self.assertNotIn(swarm_events.TARGET_CHANGED, cap.names())

    def test_resume_without_fresh_target_keeps_persisted(self):
        # The API sometimes resumes with no explicit target; the persisted one must
        # remain and its state must NOT be wiped.
        self._persist_old_mission("mres-3", "http://10.10.14.1:8080")
        coord = self._coord("mres-3", "")
        coord._load_persisted()
        self.assertEqual(coord.mission.target, "http://10.10.14.1:8080")
        self.assertIn("http://10.10.14.1:8080/login", coord.mission.endpoints)
        self.assertFalse(coord._last_reconciliation.get("changed"))

    def test_snapshot_surfaces_reconciliation(self):
        self._persist_old_mission("mres-4", "http://10.10.14.1:8080")
        coord = self._coord("mres-4", "http://10.10.14.9:8080")
        coord._load_persisted()
        snap = coord.snapshot()
        self.assertIn("target_reconciliation", snap)
        self.assertTrue(snap["target_reconciliation"].get("changed"))
        self.assertEqual(snap["target"], "http://10.10.14.9:8080")

# =========================================================================== #
# 4) Default engine — SwarmBlackboard.reconcile_target (STEP 2 / STEP 5)       #
# =========================================================================== #

class TestBlackboardReconcile(Phase7Base):

    def _board(self, target):
        from backend.agents.swarm_state import SwarmBlackboard
        b = SwarmBlackboard("c7", "r7", target)
        b.discovered_endpoints = {f"{target}/login", "/status", f"{target}/api"}
        return b

    def test_changed_target_drops_stale_endpoints(self):
        # Board constructed with the NEW target; the resumed snapshot had the OLD target.
        b = self._board("http://10.10.14.9:8080")
        b.discovered_endpoints = {"http://10.10.14.1:8080/login", "/status",
                                  "http://10.10.14.1:8080/api"}
        res = b.reconcile_target("http://10.10.14.1:8080")
        self.assertTrue(res["changed"])
        self.assertEqual(res["dropped_endpoints"], 2)
        self.assertEqual(b.discovered_endpoints, {"/status"})

    def test_unchanged_target_keeps_endpoints(self):
        b = self._board("http://10.10.14.1:8080")
        res = b.reconcile_target("http://10.10.14.1:8080")
        self.assertFalse(res["changed"])
        self.assertEqual(res["dropped_endpoints"], 0)
        self.assertEqual(len(b.discovered_endpoints), 3)

    def test_snapshot_persists_target_scope(self):
        b = self._board("http://10.10.14.1:8080")
        plan = b._build_mission_plan()
        self.assertEqual(plan["blackboard_state"].get("target_scope"), "http://10.10.14.1:8080")

# =========================================================================== #
# 5) Provider / quota observability (STEP 4 / STEP 13)                         #
# =========================================================================== #

class TestProviderObservability(Phase7Base):

    def test_quota_manager_snapshot_shape(self):
        from backend.providers.quota_manager import quota_manager
        snap = quota_manager.snapshot()
        for key in ("models", "session_blacklisted", "rate_limit_headroom",
                    "any_provider_blacklisted", "quota_exhausted_globally"):
            self.assertIn(key, snap)
        self.assertIsInstance(snap["session_blacklisted"], list)

    def test_blacklisted_provider_is_surfaced(self):
        from backend.providers.quota_manager import quota_manager
        try:
            quota_manager.blacklist_for_session("groq_test_p7", "402 quota exhausted (test)")
            snap = quota_manager.snapshot()
            self.assertTrue(snap["any_provider_blacklisted"])
            self.assertTrue(any(b.get("provider") == "groq_test_p7"
                                for b in snap["session_blacklisted"]))
        finally:
            quota_manager.reset_session_blacklists()

    def test_coordinator_quota_snapshot_is_non_empty(self):
        qs = SwarmCoordinator.quota_snapshot()
        self.assertTrue(qs)
        self.assertIn("models", qs)

def _host(spec: str) -> str:
    hs = sorted(hosts_of(spec))
    # pick the bare host token (shortest without scheme/port/path)
    bare = [h for h in hs if "/" not in h and ":" not in h]
    return bare[0] if bare else (hs[0] if hs else spec)

# =========================================================================== #
# 6) Re-plan-on-target-change recovery (STEP 2 / STEP 10) — full run() path    #
# =========================================================================== #

class TestReplanOnTargetChange(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        db = SessionLocal()
        try:
            for model in (TrajectoryEventModel, AgentSessionModel, SwarmEvidenceModel,
                          SwarmTaskModel, SwarmMissionModel):
                db.query(model).delete()
            db.commit()
        finally:
            db.close()

    def _seed_completed_mission(self, mission_id, old_target):
        """A resumable mission whose ONLY task already completed against the OLD target —
        so without re-planning the resumed swarm would have no open work to do."""
        pre = SwarmCoordinator(run_id="r7", challenge_id="c7", target=old_target,
                               category="web", mission_id=mission_id, persist=True,
                               limits=SwarmLimits(max_task_retries=0))
        t = Task(mission_id=mission_id, run_id="r7", challenge_id="c7", role="recon",
                 objective=f"Fingerprint {old_target}", priority=90)
        pre.scheduler.add(t)
        pre.scheduler.mark_completed(t, result={"reason": "done"})
        pre.mission.endpoints = [f"{old_target}/login"]
        pre.mission.note_attempt(t.signature)
        pre.mission.save()

    def _coord(self, mission_id, current_target, executed):
        c = SwarmCoordinator(run_id="r7", challenge_id="c7", target=current_target,
                             category="web", challenge_name="Phase7 Replan",
                             mission_id=mission_id, persist=True, enable_report=False,
                             limits=SwarmLimits(max_task_retries=0))
        c.agent_factory = lambda role: ProgrammableAgent(role, executed)
        return c

    async def test_changed_target_reengages_new_instance(self):
        self._seed_completed_mission("mrp-1", "http://10.10.14.1:8080")
        executed = []
        coord = self._coord("mrp-1", "http://10.10.14.9:8080", executed)
        with _CaptureEvents():
            await coord.run(resume=True)
        # The mission adopted the new target and re-planned (fresh recon dispatched),
        # instead of stopping with the stale completed plan.
        self.assertEqual(coord.mission.target, "http://10.10.14.9:8080")
        self.assertTrue(coord._last_reconciliation.get("changed"))
        self.assertTrue(executed, "resume with a changed target must re-dispatch work")
        # Nothing executed references the stale host.
        self.assertTrue(all("10.10.14.1" not in obj for _, obj in executed),
                        f"stale host must not be executed against; got {executed}")

    async def test_unchanged_target_does_not_replan_completed_mission(self):
        # Same target on resume + all work already done → no fresh plan, clean stop
        # (existing Phase-4/5 resume behaviour is preserved).
        self._seed_completed_mission("mrp-2", "http://10.10.14.1:8080")
        executed = []
        coord = self._coord("mrp-2", "http://10.10.14.1:8080", executed)
        await coord.run(resume=True)
        self.assertFalse(coord._last_reconciliation.get("changed"))
        self.assertEqual(executed, [])  # nothing re-dispatched; completed plan respected

def _tail_guard():  # pragma: no cover
    return True

if __name__ == "__main__":
    unittest.main()
