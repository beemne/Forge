"""
Phase 4 — Multi-agent swarm & coordination tests.

Covers the §18 checklist (supervisor, agents, tasks, scheduling, dependencies,
parallelism, isolation, evidence bus, shared state, dedup, failure/retry/reassign,
limits, memory integration, flag flow, global stop, checkpoint/resume, trajectory,
WS events, provider/quota integration) plus a Phase 1–3 regression guard.

Everything runs against the isolated unit-test database and uses LOCAL test doubles
for the model provider and the tool executor (the same pattern as
tests/test_agent_runtime.py), so NO API key, network, or subprocess is required.
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
from backend.agent_runtime import AgentRuntime, ExecResult, RealToolExecutor, trajectory_store
from backend.agent_runtime.decision import ProviderCompletion

from backend.swarm import (
    AgentRole, EvidenceBus, Evidence, EvidenceType, Task, TaskStatus, TaskScheduler,
    SharedMissionState, Supervisor, SpecialistAgent, SwarmLimits, SwarmCoordinator,
    roles_for_category,
)
from backend.swarm import events as swarm_events
from backend.swarm import dedup

# --------------------------------------------------------------------------- #
# Local test doubles (the spec's mock provider + mock tool executor)
# --------------------------------------------------------------------------- #

class ScriptedProvider:
    def __init__(self, responses, provider_name="stub", model_name="stub-model"):
        self.responses = responses
        self.provider_name = provider_name
        self.model_name = model_name
        self.calls = 0

    async def complete(self, *, prompt, system_instruction="", capability="general_reasoning",
                       urgency="normal", reasoning_depth="fast"):
        idx = self.calls
        self.calls += 1
        if callable(self.responses):
            content = self.responses(idx)
        elif idx < len(self.responses):
            content = self.responses[idx]
        else:
            content = self.responses[-1] if self.responses else None
        if content is None:
            return ProviderCompletion(is_refusal=True, refusal_reason="exhausted (simulated).")
        return ProviderCompletion(content=content, provider_name=self.provider_name,
                                  model_name=self.model_name, prompt_tokens=5, completion_tokens=3)

class ScriptedToolExecutor:
    def __init__(self, sequence=None, by_substring=None, default=None):
        self.sequence = list(sequence or [])
        self.by_substring = by_substring or {}
        self.default = default or ExecResult(status="SUCCESS", stdout="", exit_code=0)
        self.executed = []

    async def execute(self, action, *, cwd=None, timeout_seconds=120, canonical_target=None):
        cmd = action.display()
        self.executed.append(cmd)
        for key, res in self.by_substring.items():
            if key in cmd:
                return res
        if self.sequence:
            return self.sequence.pop(0)
        return self.default

class ConcurrencyProbeExecutor:
    """Records the peak number of agents executing simultaneously."""

    def __init__(self, shared, stdout=""):
        self.shared = shared
        self.stdout = stdout
        self.executed = []

    async def execute(self, action, *, cwd=None, timeout_seconds=120, canonical_target=None):
        self.executed.append(action.display())
        self.shared["cur"] += 1
        self.shared["max"] = max(self.shared["max"], self.shared["cur"])
        try:
            await asyncio.sleep(0.03)
        finally:
            self.shared["cur"] -= 1
        return ExecResult(command=action.display(), status="SUCCESS", exit_code=0, stdout=self.stdout)

def scripted_runtime(responses, *, by_substring=None, sequence=None, default=None, executor=None):
    ex = executor or ScriptedToolExecutor(by_substring=by_substring, sequence=sequence, default=default)
    return AgentRuntime(tool_executor=ex, provider_gateway=ScriptedProvider(responses),
                        learn_on_completion=False)

def factory_from(makers):
    """makers: dict role_value -> callable()->AgentRuntime. '*' is the fallback."""
    def factory(role):
        maker = makers.get(role.value) or makers.get("*")
        rt = maker() if maker else scripted_runtime(["BUDGET_EXHAUSTED: noop"])
        return SpecialistAgent(role, runtime=rt)
    return factory

class _CaptureEvents:
    """Context manager that captures all swarm WS broadcasts deterministically."""

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

class SwarmTestBase(unittest.IsolatedAsyncioTestCase):
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

    # helpers -------------------------------------------------------------- #

    def _coord(self, **kw):
        defaults = dict(challenge_id="chal-p4", run_id=None, target="http://target.ctf:8080",
                        category="web", challenge_name="Phase4 Test", persist=True,
                        limits=SwarmLimits(max_task_retries=0))
        defaults.update(kw)
        return SwarmCoordinator(**defaults)

    def _task(self, coord, role, objective, *, deps=None, priority=50):
        return Task(mission_id=coord.mission_id, run_id=coord.mission.run_id,
                    challenge_id=coord.mission.challenge_id, role=role.value if isinstance(role, AgentRole) else role,
                    objective=objective, priority=priority, dependencies=list(deps or []))

# =========================================================================== #
# 1. Supervisor + roles + task creation                                       #
# =========================================================================== #

class TestSupervisorAndRoles(SwarmTestBase):

    def test_01_roles_for_category(self):
        # recon is always activated; a crypto challenge activates the crypto specialist.
        self.assertIn(AgentRole.RECON, roles_for_category("crypto"))
        self.assertIn(AgentRole.CRYPTO, roles_for_category("crypto"))
        self.assertIn(AgentRole.WEB, roles_for_category("web"))

    def test_02_plan_initial_creates_recon_and_dependent_web(self):
        sup = Supervisor("m1")
        state = SharedMissionState(mission_id="m1", target="http://target.ctf:8080", category="web")
        tasks = sup.plan_initial(state)
        roles = [t.role for t in tasks]
        self.assertIn("recon", roles)
        self.assertIn("web", roles)
        # The Apache example: the web task WAITS on recon (dependency wired).
        recon = next(t for t in tasks if t.role == "recon")
        web = next(t for t in tasks if t.role == "web")
        self.assertIn(recon.id, web.dependencies)

    def test_03_plan_initial_forensics_from_artifacts(self):
        sup = Supervisor("m2")
        state = SharedMissionState(mission_id="m2", target="/tmp/capture.pcap", category="forensics",
                                   artifacts=["/tmp/capture.pcap"])
        tasks = sup.plan_initial(state)
        self.assertIn("forensics", [t.role for t in tasks])

    def test_04_react_to_versioned_service_creates_exploit_task(self):
        sup = Supervisor("m3")
        state = SharedMissionState(mission_id="m3")
        ev = Evidence(mission_id="m3", evidence_type=EvidenceType.TECHNOLOGY.value,
                      title="Apache httpd 2.4.49", related_technology="Apache httpd 2.4.49")
        proposals = sup.react_to_evidence(ev, state)
        self.assertTrue(proposals)
        self.assertEqual(proposals[0].role, "web")
        self.assertIn("2.4.49", proposals[0].objective)

    def test_05_react_to_endpoint_creates_web_task(self):
        sup = Supervisor("m4")
        ev = Evidence(mission_id="m4", evidence_type=EvidenceType.ENDPOINT.value,
                      title="/admin", related_endpoint="http://target.ctf/admin")
        proposals = sup.react_to_evidence(ev, SharedMissionState(mission_id="m4"))
        self.assertTrue(any("/admin" in p.objective for p in proposals))

# =========================================================================== #
# 2. Task scheduler: dependencies, priority, parallelism, dedup               #
# =========================================================================== #

class TestScheduler(SwarmTestBase):

    def test_06_dependencies_block_until_prereq_completes(self):
        sch = TaskScheduler(persist=False)
        a = Task(mission_id="m", role="recon", objective="scan")
        b = Task(mission_id="m", role="web", objective="exploit", dependencies=[a.id])
        sch.add(a)
        sch.add(b)
        ready = {t.id for t in sch.ready_tasks()}
        self.assertIn(a.id, ready)
        self.assertNotIn(b.id, ready)                 # blocked
        self.assertEqual(sch.get(b.id).status, TaskStatus.BLOCKED.value)
        sch.mark_running(a)
        sch.mark_completed(a)
        ready2 = {t.id for t in sch.ready_tasks()}
        self.assertIn(b.id, ready2)                   # unblocked after prereq completes

    def test_07_priority_ordering(self):
        sch = TaskScheduler(persist=False)
        lo = Task(mission_id="m", role="recon", objective="low", priority=10)
        hi = Task(mission_id="m", role="web", objective="high", priority=90)
        sch.add(lo)
        sch.add(hi)
        ordered = sch.ready_tasks()
        self.assertEqual(ordered[0].id, hi.id)        # highest priority first

    def test_08_parallel_independent_tasks_all_ready(self):
        sch = TaskScheduler(persist=False)
        ids = []
        for r in ("recon", "forensics", "crypto"):
            t = Task(mission_id="m", role=r, objective=f"{r} work")
            sch.add(t)
            ids.append(t.id)
        ready = {t.id for t in sch.ready_tasks()}
        self.assertEqual(ready, set(ids))             # all independent → all ready together

    def test_09_duplicate_signature_rejected(self):
        sch = TaskScheduler(persist=False)
        a = Task(mission_id="m", role="recon", objective="Scan the target for open ports")
        b = Task(mission_id="m", role="recon", objective="scan the   target for open PORTS")  # same sig
        self.assertTrue(sch.add(a))
        self.assertFalse(sch.add(b))                  # deduped (§8)
        self.assertEqual(sch.total_count(), 1)

    def test_10_unsatisfiable_dependency_cancels_dependent(self):
        sch = TaskScheduler(persist=False)
        a = Task(mission_id="m", role="recon", objective="scan")
        b = Task(mission_id="m", role="web", objective="exploit", dependencies=[a.id])
        sch.add(a)
        sch.add(b)
        sch.mark_running(a)
        sch.mark_failed(a, reason="boom")
        sch.refresh_states()
        self.assertEqual(sch.get(b.id).status, TaskStatus.CANCELLED.value)

# =========================================================================== #
# 3. Evidence bus + shared mission state                                      #
# =========================================================================== #

class TestEvidenceAndState(SwarmTestBase):

    def test_11_publish_dedup_and_consume(self):
        bus = EvidenceBus("m-ev", persist=False)
        e1 = Evidence(mission_id="m-ev", evidence_type=EvidenceType.SERVICE.value,
                      title="Apache httpd 2.4.49", agent_id="recon")
        e2 = Evidence(mission_id="m-ev", evidence_type=EvidenceType.SERVICE.value,
                      title="Apache httpd 2.4.49", agent_id="web")   # duplicate signature
        self.assertIsNotNone(bus.publish(e1))
        self.assertIsNone(bus.publish(e2))            # deduped
        self.assertEqual(bus.count(), 1)

    def test_12_relevant_for_role(self):
        bus = EvidenceBus("m-ev2", persist=False)
        bus.publish(Evidence(mission_id="m-ev2", evidence_type=EvidenceType.ENDPOINT.value,
                             title="http://t/login", tags=["login", "http"]))
        bus.publish(Evidence(mission_id="m-ev2", evidence_type=EvidenceType.ARTIFACT.value,
                             title="capture.pcap", tags=["pcap"]))
        web_leads = bus.relevant_for(AgentRole.WEB)
        fx_leads = bus.relevant_for(AgentRole.FORENSICS)
        self.assertTrue(any("login" in e.title for e in web_leads))
        self.assertTrue(any("pcap" in e.title for e in fx_leads))

    def test_13_subscriber_notified_on_new_evidence(self):
        bus = EvidenceBus("m-ev3", persist=False)
        seen = []
        bus.subscribe(lambda ev: seen.append(ev.title))
        bus.publish(Evidence(mission_id="m-ev3", evidence_type="note", title="hello"))
        bus.publish(Evidence(mission_id="m-ev3", evidence_type="note", title="hello"))  # dup
        self.assertEqual(seen, ["hello"])             # only new evidence notifies

    def test_14_shared_state_integrates_evidence(self):
        st = SharedMissionState(mission_id="m-st")
        st.integrate_evidence(Evidence(mission_id="m-st", evidence_type=EvidenceType.ENDPOINT.value,
                                       related_endpoint="http://t/admin", title="/admin"))
        st.integrate_evidence(Evidence(mission_id="m-st", evidence_type=EvidenceType.VULNERABILITY.value,
                                       related_vulnerability="SQLi in id param", title="SQLi"))
        self.assertIn("http://t/admin", st.endpoints)
        self.assertIn("SQLi in id param", st.vulnerabilities)
        st.recompute_progress()
        self.assertGreater(st.progress, 0)

    def test_15_evidence_persistence_and_reload(self):
        bus = EvidenceBus("m-persist", challenge_id="chal-p4", persist=True)
        bus.publish(Evidence(mission_id="m-persist", evidence_type=EvidenceType.ENDPOINT.value,
                             title="/secret", related_endpoint="/secret"))
        # A fresh bus over the same mission rehydrates from the durable table.
        fresh = EvidenceBus("m-persist", persist=True)
        loaded = fresh.load()
        self.assertGreaterEqual(loaded, 1)
        self.assertTrue(any(e.title == "/secret" for e in fresh.all()))

# =========================================================================== #
# 4. Failure classification + recovery decisions                              #
# =========================================================================== #

class TestRecoveryPolicy(SwarmTestBase):

    def _result(self, **kw):
        from backend.swarm.agents import AgentResult
        base = dict(task_id="t", role="recon", status="FAILED")
        base.update(kw)
        return AgentResult(**base)

    def test_16_classify_failures(self):
        sup = Supervisor("m")
        self.assertEqual(sup.classify_failure(self._result(status="TIMEOUT")), "timeout")
        self.assertEqual(sup.classify_failure(self._result(failure_category="network")), "network")
        self.assertEqual(sup.classify_failure(self._result(failure_category="missing_dependency")),
                         "missing_dependency")
        self.assertEqual(sup.classify_failure(self._result(status="MAX_TURNS")), "no_progress")
        self.assertEqual(sup.classify_failure(self._result(status="CANCELLED")), "cancelled")

    def test_17_retry_reassign_abandon(self):
        sup = Supervisor("m")
        t = Task(mission_id="m", role="web", objective="ffuf enum")
        # transient with retries remaining → retry
        d = sup.decide_recovery(t, self._result(failure_category="network"), max_retries=2)
        self.assertEqual(d.action, "retry")
        # missing tool → reassign with an alternative-method objective
        d2 = sup.decide_recovery(t, self._result(failure_category="missing_dependency"), max_retries=2)
        self.assertEqual(d2.action, "reassign")
        self.assertIn("alternative", (d2.objective or "").lower())
        # unrecoverable and no retries → abandon
        d3 = sup.decide_recovery(t, self._result(failure_category="execution"), max_retries=0)
        self.assertEqual(d3.action, "abandon")

# =========================================================================== #
# 5. SpecialistAgent isolation + AgentRuntime reuse (§2, §5, §14)             #
# =========================================================================== #

class TestSpecialistAgent(SwarmTestBase):

    def test_18_production_agent_uses_real_tool_executor(self):
        # Agents run through the EXISTING AgentRuntime → RealToolExecutor
        # (→ ToolManager → ExecutionService); no duplicate subprocess path.
        agent = SpecialistAgent(AgentRole.RECON)
        rt = agent._get_runtime()
        self.assertIsInstance(rt, AgentRuntime)
        self.assertIsInstance(rt.tool_executor, RealToolExecutor)

    async def test_19_isolated_session_trajectory_and_evidence(self):
        coord = self._coord()
        bus = coord.bus
        rt = scripted_runtime(
            ["curl -s http://target.ctf/", "BUDGET_EXHAUSTED: stop"],
            by_substring={"curl": ExecResult(command="curl -s http://target.ctf/", status="SUCCESS",
                                             exit_code=0, stdout='<a href="http://target.ctf/login">x</a>')})
        agent = SpecialistAgent(AgentRole.RECON, runtime=rt)
        task = self._task(coord, AgentRole.RECON, "recon the target")
        result = await agent.execute(task, coord.mission, bus, max_turns=4)

        # Isolated session with the role as its agent_id, and its OWN trajectory.
        self.assertTrue(result.session_id)
        events = trajectory_store.get_events(result.session_id)
        self.assertTrue(any(e.event_type == "COMMAND" for e in events))
        self.assertTrue(any(e.agent_id == "recon" for e in events))
        # Discovered an endpoint → surfaced as structured evidence for the team.
        self.assertTrue(any(e.evidence_type == EvidenceType.ENDPOINT.value for e in result.evidence))

    async def test_20_agent_seeded_with_bounded_context_not_full_db(self):
        coord = self._coord()
        # Shared state has knowledge from other agents.
        coord.mission.endpoints = [f"http://t/{i}" for i in range(50)]
        coord.mission.technologies = ["nginx"]
        rt = scripted_runtime(["BUDGET_EXHAUSTED: stop"])
        agent = SpecialistAgent(AgentRole.WEB, runtime=rt)
        task = self._task(coord, AgentRole.WEB, "test the app")
        result = await agent.execute(task, coord.mission, coord.bus, max_turns=2)
        from backend.agent_runtime import session_manager
        sess = session_manager.get(result.session_id)
        # Seeded a bounded slice (capped at 20), NOT all 50 endpoints.
        self.assertLessEqual(len(sess.state.known_endpoints), 20)
        self.assertIn("nginx", sess.state.technologies)

# =========================================================================== #
# 6. Coordinator: end-to-end, flag global stop, evidence→task, limits         #
# =========================================================================== #

class TestCoordinator(SwarmTestBase):

    async def test_21_flag_verified_triggers_global_stop(self):
        coord = self._coord(category="web")
        makers = {
            "recon": lambda: scripted_runtime(
                ["cat flag.txt"],
                by_substring={"cat": ExecResult(command="cat flag.txt", status="SUCCESS", exit_code=0,
                                                stdout="picoCTF{sw4rm_v3rified}")}),
            "*": lambda: scripted_runtime(["BUDGET_EXHAUSTED: noop"]),
        }
        coord.agent_factory = factory_from(makers)
        res = await coord.run()

        self.assertEqual(res.status, "COMPLETED")
        self.assertEqual(res.verified_flag, "picoCTF{sw4rm_v3rified}")
        # The dependent web task was cancelled by the global stop (§11).
        web = [t for t in coord.scheduler.all() if t.role == "web"]
        self.assertTrue(web and all(t.status == TaskStatus.CANCELLED.value for t in web))

    async def test_22_unverified_candidate_does_not_complete(self):
        coord = self._coord(category="crypto")
        # LLM-asserted flag (prose) must NOT verify or stop the mission.
        ok = coord.submit_flag_candidate("picoCTF{unproven}", source="llm", agent_id="crypto")
        self.assertFalse(ok)
        self.assertIsNone(coord.mission.verified_flag)
        self.assertIn("picoCTF{unproven}", coord.mission.flag_candidates)

    async def test_23_central_verification_from_tool_output(self):
        coord = self._coord()
        ok = coord.submit_flag_candidate("picoCTF{from_tool}", source="tool_output",
                                         command="cat flag", action_succeeded=True, agent_id="web")
        self.assertTrue(ok)
        self.assertEqual(coord.mission.verified_flag, "picoCTF{from_tool}")

    async def test_24_evidence_from_agent_spawns_followup_task(self):
        coord = self._coord(category="misc")
        # recon discovers the web server tech → supervisor should schedule a targeted
        # web investigation task referencing that technology (§1 delegation, §6 bus).
        makers = {
            "recon": lambda: scripted_runtime(
                ["curl -sI http://target.ctf/", "BUDGET_EXHAUSTED: stop"],
                by_substring={"curl": ExecResult(command="curl -sI http://target.ctf/", status="SUCCESS",
                                                 exit_code=0, stdout="Server: Apache/2.4.49 (Unix)")}),
            "*": lambda: scripted_runtime(["BUDGET_EXHAUSTED: noop"]),
        }
        coord.agent_factory = factory_from(makers)
        await coord.run()
        # A follow-up web task referencing the detected technology was created — one that
        # did NOT exist in the initial plan (whose web task is the generic profile objective).
        objs = [t.objective.lower() for t in coord.scheduler.all() if t.role == "web"]
        self.assertTrue(any("apache" in o for o in objs),
                        f"expected a technology-specific web task, got: {objs}")

    async def test_25_parallel_execution_of_independent_tasks(self):
        shared = {"cur": 0, "max": 0}
        coord = self._coord(category="misc", limits=SwarmLimits(max_concurrent_agents=2, max_task_retries=0))
        # Two independent tasks that run concurrently.
        coord.scheduler.add(self._task(coord, AgentRole.RECON, "recon sweep"))
        coord.scheduler.add(self._task(coord, AgentRole.FORENSICS, "inspect artifact"))

        def maker():
            return scripted_runtime(["echo hi", "BUDGET_EXHAUSTED: stop"],
                                    executor=ConcurrencyProbeExecutor(shared, stdout="ok"))
        coord.agent_factory = factory_from({"*": maker})
        await coord.run()
        self.assertGreaterEqual(shared["max"], 2)     # genuinely ran in parallel

    async def test_26_active_task_limit_enforced(self):
        coord = self._coord(limits=SwarmLimits(max_total_tasks=2, max_task_retries=0))
        self.assertTrue(coord._add_task(self._task(coord, AgentRole.RECON, "one"), origin="plan"))
        self.assertTrue(coord._add_task(self._task(coord, AgentRole.WEB, "two"), origin="plan"))
        # Third exceeds the total cap → refused.
        self.assertFalse(coord._add_task(self._task(coord, AgentRole.CRYPTO, "three"), origin="evidence"))

    async def test_27_duplicate_task_not_dispatched_twice(self):
        coord = self._coord()
        t1 = self._task(coord, AgentRole.RECON, "Scan the target for services")
        t2 = self._task(coord, AgentRole.RECON, "scan the target FOR services")  # same signature
        self.assertTrue(coord._add_task(t1, origin="plan"))
        self.assertFalse(coord._add_task(t2, origin="plan"))

# =========================================================================== #
# 7. Failure handling, retry, reassignment inside the coordinator             #
# =========================================================================== #

class TestCoordinatorRecovery(SwarmTestBase):

    async def test_28_failed_task_is_retried_then_succeeds(self):
        coord = self._coord(category="network", limits=SwarmLimits(max_task_retries=2,
                                                                    max_concurrent_agents=1))
        coord.scheduler.add(self._task(coord, AgentRole.RECON, "recon with retry"))
        state = {"n": 0}

        def recon_maker():
            state["n"] += 1
            if state["n"] == 1:
                return scripted_runtime(
                    ["curl http://down/", "BUDGET_EXHAUSTED: x"],
                    by_substring={"curl": ExecResult(command="curl http://down/", status="FAILED",
                                                     exit_code=7, stderr="Connection refused",
                                                     execution_failure=True, failure_category="NETWORK")})
            return scripted_runtime(
                ["cat /flag"],
                by_substring={"cat": ExecResult(command="cat /flag", status="SUCCESS", exit_code=0,
                                                stdout="picoCTF{retry_worked}")})
        coord.agent_factory = factory_from({"recon": recon_maker, "*": recon_maker})
        res = await coord.run()
        self.assertEqual(res.verified_flag, "picoCTF{retry_worked}")
        recon = next(t for t in coord.scheduler.all() if t.role == "recon" and t.retry_count > 0)
        self.assertGreaterEqual(recon.retry_count, 1)

    async def test_29_missing_tool_triggers_reassignment(self):
        coord = self._coord(limits=SwarmLimits(max_task_retries=0, max_total_tasks=3,
                                               max_concurrent_agents=1))
        coord.scheduler.add(self._task(coord, AgentRole.WEB, "enumerate with ffuf"))

        def maker():
            return scripted_runtime(
                ["ffuf -w list -u http://t/FUZZ", "BUDGET_EXHAUSTED: x"],
                by_substring={"ffuf": ExecResult(command="ffuf ...", status="FAILED", exit_code=127,
                                                 stderr="ffuf: command not found",
                                                 execution_failure=True, failure_category="COMMAND_NOT_FOUND")})
        coord.agent_factory = factory_from({"*": maker})
        await coord.run()
        statuses = [t.status for t in coord.scheduler.all()]
        objectives = [t.objective.lower() for t in coord.scheduler.all()]
        self.assertIn(TaskStatus.REASSIGNED.value, statuses)      # original was reassigned
        self.assertTrue(any("alternative" in o for o in objectives))  # new task uses another method

# =========================================================================== #
# 8. Checkpoint / resume, trajectory, WS events, quota                        #
# =========================================================================== #

class TestObservabilityAndResume(SwarmTestBase):

    async def test_30_checkpoint_resume_keeps_completed_work(self):
        mid = "mission-resume-1"
        # Simulate an interrupted mission persisted to the durable tables.
        a = self._coord(mission_id=mid, category="web")
        recon = self._task(a, AgentRole.RECON, "recon done")
        a.scheduler.add(recon)
        a.scheduler.mark_running(recon)
        a.scheduler.mark_completed(recon, result={"note": "done"})
        web = self._task(a, AgentRole.WEB, "finish exploit", deps=[recon.id])
        a.scheduler.add(web)
        a.mission.status = "PAUSED"
        a.mission.save(coord_session_id=a.coord_session.id)

        # Resume in a fresh coordinator; completed recon must NOT re-run.
        b = self._coord(mission_id=mid, category="web")
        b.agent_factory = factory_from({
            "web": lambda: scripted_runtime(
                ["cat /flag"],
                by_substring={"cat": ExecResult(command="cat /flag", status="SUCCESS", exit_code=0,
                                                stdout="picoCTF{resumed_ok}")}),
            "*": lambda: scripted_runtime(["BUDGET_EXHAUSTED: x"]),
        })
        res = await b.run(resume=True)
        self.assertEqual(res.verified_flag, "picoCTF{resumed_ok}")
        recon_after = next(t for t in b.scheduler.all() if t.role == "recon")
        self.assertEqual(recon_after.status, TaskStatus.COMPLETED.value)   # stayed completed

    async def test_31_websocket_events_emitted(self):
        with _CaptureEvents() as cap:
            coord = self._coord(category="web")
            coord.agent_factory = factory_from({
                "recon": lambda: scripted_runtime(
                    ["cat flag.txt"],
                    by_substring={"cat": ExecResult(command="cat flag.txt", status="SUCCESS", exit_code=0,
                                                    stdout="picoCTF{ws_events}")}),
                "*": lambda: scripted_runtime(["BUDGET_EXHAUSTED: x"]),
            })
            await coord.run()
        names = set(cap.names())
        for expected in (swarm_events.MISSION_STARTED, swarm_events.TASK_CREATED,
                         swarm_events.TASK_ASSIGNED, swarm_events.FLAG_VERIFIED,
                         swarm_events.MISSION_COMPLETE):
            self.assertIn(expected, names)

    async def test_32_coordination_events_recorded_separately_from_agents(self):
        coord = self._coord(category="web")
        coord.agent_factory = factory_from({
            "recon": lambda: scripted_runtime(
                ["cat flag.txt"],
                by_substring={"cat": ExecResult(command="cat flag.txt", status="SUCCESS", exit_code=0,
                                                stdout="picoCTF{traj}")}),
            "*": lambda: scripted_runtime(["BUDGET_EXHAUSTED: x"]),
        })
        await coord.run()
        # Coordination trajectory lives under the supervisor session, separate from
        # each specialist's execution trajectory — but in the ONE trajectory store.
        coord_events = trajectory_store.get_events(coord.coord_session.id)
        types = [e.event_type for e in coord_events]
        self.assertTrue(any(t.startswith("COORD_") for t in types))
        self.assertTrue(any(e.agent_id == "supervisor" for e in coord_events))

    async def test_33_persisted_rows_exist_after_mission(self):
        coord = self._coord(category="web")
        coord.agent_factory = factory_from({
            "recon": lambda: scripted_runtime(
                ["cat flag.txt"],
                by_substring={"cat": ExecResult(command="cat flag.txt", status="SUCCESS", exit_code=0,
                                                stdout="picoCTF{persisted}")}),
            "*": lambda: scripted_runtime(["BUDGET_EXHAUSTED: x"]),
        })
        await coord.run()
        db = SessionLocal()
        try:
            m = db.query(SwarmMissionModel).filter(SwarmMissionModel.id == coord.mission_id).first()
            tcount = db.query(SwarmTaskModel).filter(SwarmTaskModel.mission_id == coord.mission_id).count()
            ecount = db.query(SwarmEvidenceModel).filter(SwarmEvidenceModel.mission_id == coord.mission_id).count()
        finally:
            db.close()
        self.assertIsNotNone(m)
        self.assertEqual(m.status, "COMPLETED")
        self.assertGreaterEqual(tcount, 1)
        self.assertGreaterEqual(ecount, 1)

    def test_34_quota_snapshot_is_readonly_dict(self):
        # Integrates with the EXISTING provider/quota system (§12), never a new one.
        snap = SwarmCoordinator.quota_snapshot()
        self.assertIsInstance(snap, dict)

    async def test_35_snapshot_shape_for_api(self):
        coord = self._coord(category="web")
        coord.scheduler.add(self._task(coord, AgentRole.RECON, "recon"))
        snap = coord.snapshot()
        for key in ("mission_id", "status", "tasks", "task_counts", "evidence_count",
                    "agents", "quota", "limits", "shared_state"):
            self.assertIn(key, snap)

# =========================================================================== #
# 9. Phase 1–3 regression guard                                               #
# =========================================================================== #

class TestRegression(SwarmTestBase):

    def test_36_legacy_swarm_and_runtime_still_import(self):
        # Phase 4 must not break the legacy engine or the runtime layer.
        from backend.agents.swarm_orchestrator import swarm_orchestrator  # noqa: F401
        from backend.agent_runtime import AgentRuntime, session_manager    # noqa: F401
        self.assertTrue(callable(getattr(swarm_orchestrator, "run_swarm", None)))

    def test_37_normalize_command_dedup(self):
        self.assertEqual(dedup.normalize_command("nmap  -sV   TARGET"),
                         dedup.normalize_command("nmap -sv target".upper().lower()))

if __name__ == "__main__":
    unittest.main(verbosity=2)
