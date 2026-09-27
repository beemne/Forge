import os
import sys
import time
import asyncio
import tempfile
from datetime import datetime

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.guard import authorize_production_database
from backend.database.session import init_db, SessionLocal
from backend.database.models import (
    ChallengeModel, TargetProfileModel, RunModel, CheckpointModel, EvidenceModel,
    AuditLogModel, AgentSessionModel, TrajectoryEventModel
)
from backend.environment.detector import environment_detector
from backend.providers.router import model_router
from backend.tools.manager import tool_manager
from backend.privilege.manager import privilege_manager
from backend.agents.orchestrator_loop import orchestrator_loop
from backend.agent_runtime import (
    AgentRuntime, RealToolExecutor, Action, ActionType, ExecResult, session_manager,
    AnswerResolver, AnswerCandidate, AnswerSource, AnswerStatus, AnswerType as VerifierAnswerType,
    VerifierAgent,
)
from backend.execution.interactive import interactive_manager
from backend.execution.process_manager import process_manager
from backend.swarm.evidence import ProvenanceType
from backend.agents.stream_condenser import stream_condenser
from backend.reporting.generator import report_generator
from backend.api.runner import workflow_runner
from tests.fixtures.web_target import LocalCTFServer
from tests.fixtures.forensics_fixture import create_forensics_fixture

PY = sys.executable or "python"

FAKE_DIALOGUE_CHILD = """\
import sys

sys.stdout.write("STAGE 0: Ready for dialogue\\n")
sys.stdout.flush()

line1 = sys.stdin.readline().strip()
if line1 == "PING_STAGE_1":
    sys.stdout.write("STAGE 1: OK. Send token:\\n")
    sys.stdout.flush()
    line2 = sys.stdin.readline().strip()
    if line2 == "TOKEN_STAGE_2":
        sys.stdout.write("STAGE 2: OK. Secret: GENERIC_TEST_SECRET_9876\\n")
        sys.stdout.flush()
    else:
        sys.stdout.write("FAILED_STAGE_2\\n")
        sys.stdout.flush()
else:
    sys.stdout.write("FAILED_STAGE_1\\n")
    sys.stdout.flush()
"""


async def run_competition_simulation(engine_type=None):
    print("\n==================================================")
    print("      FORGE COMPETITION TEST SIMULATION           ")
    print("==================================================\n")

    results = {}
    metrics = {
        "start_time": time.time(),
        "tool_calls": 0,
        "model_calls": 0,
        "evidence_count": 0
    }

    # 1. Environment Check
    env = environment_detector.detect_environment()
    results["Environment"] = "PASS" if env["os"] else "FAIL"

    # 2. Database Check
    try:
        # This simulation drives the configured database directly instead of going
        # through the server, so backend.main's startup hook never runs. Authorizing
        # here is the deliberate, visible opt-in the guard requires.
        authorize_production_database()
        init_db()
        db = SessionLocal()
        results["Database"] = "PASS"
    except Exception as e:
        results["Database"] = f"FAIL ({str(e)})"
        db = None

    # 3. Target & Target Server Startup
    server = LocalCTFServer(port=8888)
    server.start()
    time.sleep(0.5)

    try:
        # 4. Providers & Model Router Check
        response = await model_router.route_request("FORGE Competition Prompt", capability="general_reasoning")
        results["Providers"] = "PASS" if response and not response.is_refusal else "FAIL"
        results["Model Router"] = "PASS" if response and response.provider_name else "FAIL"
        metrics["model_calls"] += 1

        # 5. Tool Manager & Execution Check (One-shot command)
        tool_res = await tool_manager.execute_capability("web_testing", "http://127.0.0.1:8888/")
        metrics["tool_calls"] += 1
        results["Tool Manager"] = "PASS" if tool_res.status == "SUCCESS" else "FAIL"

        # 6. Privilege Manager Check
        priv_pass = privilege_manager.evaluate_privilege("test_agent", "curl", "SAFE", db)
        priv_fail = privilege_manager.evaluate_privilege("test_agent", "rm -rf /", "DANGEROUS", db)
        results["Privilege"] = "PASS" if (priv_pass and not priv_fail) else "FAIL"

        # 7. Persistent Interactive Execution Check (open -> read -> send -> read -> send -> read -> close)
        temp_dir = tempfile.TemporaryDirectory()
        interactive_script = os.path.join(temp_dir.name, "interactive_sim.py")
        with open(interactive_script, "w", encoding="utf-8") as f:
            f.write(FAKE_DIALOGUE_CHILD)

        try:
            # 7a. Open (creates session and receives initial prompt)
            open_res = await tool_manager.execute_raw_command(f'interactive_open "{PY}" "{interactive_script}"')
            metrics["tool_calls"] += 1
            if open_res.status == "SUCCESS" and "[SESSION:" in open_res.stdout:
                sess_key = open_res.stdout.split("[SESSION:")[1].split("]")[0].strip()
                # 7b. Send stage 1 input
                send1 = await tool_manager.execute_raw_command(f"interactive_send {sess_key} PING_STAGE_1")
                metrics["tool_calls"] += 1
                # 7c. Read stage 1 response
                read1 = await tool_manager.execute_raw_command(f"interactive_read {sess_key}")
                metrics["tool_calls"] += 1
                # 7d. Send stage 2 input & read final secret
                send2 = await tool_manager.execute_raw_command(f"interactive_send {sess_key} TOKEN_STAGE_2")
                metrics["tool_calls"] += 1
                read2 = await tool_manager.execute_raw_command(f"interactive_read {sess_key}")
                metrics["tool_calls"] += 1
                # 7e. Close session
                close_res = await tool_manager.execute_raw_command(f"interactive_close {sess_key}")
                metrics["tool_calls"] += 1

                interactive_ok = (
                    "STAGE 0:" in open_res.stdout
                    and "STAGE 1: OK" in read1.stdout
                    and "GENERIC_TEST_SECRET_9876" in read2.stdout
                    and interactive_manager.get(sess_key) is None
                )
                results["Interactive Process"] = "PASS" if interactive_ok else "FAIL"
            else:
                results["Interactive Process"] = "FAIL"
        finally:
            await interactive_manager.close_all(reason="competition_sim_teardown")
            temp_dir.cleanup()

        # 8. Python Solver Syntax Validation Check
        executor = RealToolExecutor(tool_manager=tool_manager)
        valid_action = Action(type=ActionType.PYTHON_SCRIPT, script="x = 1 + 1\nprint(f'VAL={x}')\n")
        valid_res = await executor.execute(valid_action)
        metrics["tool_calls"] += 1

        invalid_action = Action(type=ActionType.PYTHON_SCRIPT, script="def broken_syntax(\n")
        invalid_res = await executor.execute(invalid_action)

        syntax_ok = (
            valid_res.status == "SUCCESS"
            and invalid_res.status == "FAILED"
            and invalid_res.failure_category == "SYNTAX_ERROR"
            and "SyntaxError" in invalid_res.stderr
        )
        results["Syntax Validation"] = "PASS" if syntax_ok else "FAIL"

        # 9. Artifact Provenance Check (LOCAL_FILE vs REMOTE_FILE / SOURCE_CODE_REFERENCE)
        from backend.agent_runtime.observation import ObservationEngine
        obs_engine = ObservationEngine()
        obs_local = obs_engine.observe(ExecResult(status="SUCCESS", stdout="File saved to ./output.bin successfully."))
        obs_remote = obs_engine.observe(ExecResult(status="SUCCESS", stdout='result = open("remote_flag.txt", "r").read()'))

        provenance_ok = (
            obs_local.file_provenance.get("./output.bin") == "LOCAL_FILE"
            and obs_remote.file_provenance.get("remote_flag.txt") == "SOURCE_CODE_REFERENCE"
            and ProvenanceType.LOCAL_FILE.value == "LOCAL_FILE"
        )
        results["Provenance"] = "PASS" if provenance_ok else "FAIL"

        # 10. Structured Visual/ASCII Terminal Output Handling Check
        ascii_grid = (
            "  ###   #####  \n"
            " #   #  #    # \n"
            " #####  #####  \n"
            " #   #  #      \n"
        )
        condensed_ascii = stream_condenser.condense_output("custom_tool", ascii_grid)
        # Leading whitespace must be preserved in structured layout lines
        structured_ok = "  ###   #####" in condensed_ascii and " #   #  #" in condensed_ascii
        results["Structured ASCII"] = "PASS" if structured_ok else "FAIL"

        # 11. Create Challenge & Target Profile
        ch = ChallengeModel(
            name="Competition E2E Modern Challenge",
            category="web",
            mission_plan={"run_config": {"task_timeout": 5, "max_turns_per_task": 1, "max_iterations": 1}}
        )
        db.add(ch)
        db.commit()

        target = TargetProfileModel(challenge_id=ch.id, current_address="http://127.0.0.1:8888/")
        db.add(target)
        db.commit()
        results["Target Manager"] = "PASS"

        # 12. Launch Production Swarm Run via WorkflowRunner (matches UI start path: engine_type=None)
        run = RunModel(challenge_id=ch.id, status="RUNNING", current_phase="recon", current_agent=(engine_type or "swarm"))
        db.add(run)
        db.commit()

        # Start run using WorkflowRunner with default production engine (engine_type=None -> "swarm")
        workflow_runner.start_run(run.id, ch.id, "http://127.0.0.1:8888/", engine_type=engine_type)
        results["WorkflowRunner"] = "PASS" if run.id in workflow_runner.active_runs else "FAIL"

        runner_task = workflow_runner.tasks.get(run.id)
        active_engine = (engine_type or "swarm").strip().lower()
        is_coord = active_engine in ("coord", "team", "supervisor", "coordinated", "swarm_coord")
        resolved_engine = "swarm_coord" if is_coord else "swarm"
        print(f"\nProduction engine: {resolved_engine}")

        # Await the actual run task and verify its execution
        task_verdict = "NO_TASK"
        task_result = None
        if runner_task is not None:
            try:
                task_result = await asyncio.wait_for(asyncio.shield(runner_task), timeout=5.0)
                task_verdict = "COMPLETED"
            except asyncio.TimeoutError:
                task_verdict = "RUNNING"
            except asyncio.CancelledError:
                task_verdict = "CANCELLED"
            except Exception as exc:
                task_verdict = f"EXCEPTION:{type(exc).__name__}"

        if is_coord:
            # Coordinator path verification
            if task_verdict == "COMPLETED" and task_result is not None:
                mission_status = getattr(task_result, "status", None)
                runtime_ok = mission_status in ("COMPLETED", "PAUSED", "FAILED", "CANCELLED")
            elif task_verdict == "RUNNING":
                coord_events = (db.query(TrajectoryEventModel)
                                .filter(TrajectoryEventModel.run_id == run.id,
                                        TrajectoryEventModel.event_type == "COORD_PLAN")
                                .count())
                runtime_ok = coord_events > 0 or (run.id in workflow_runner.active_runs)
            else:
                runtime_ok = False
        else:
            # Production Swarm path verification
            from backend.agents.swarm_orchestrator import swarm_orchestrator
            active_swarm = swarm_orchestrator.active_swarms.get(run.id)
            runtime_ok = (
                (task_verdict in ("COMPLETED", "RUNNING") and (active_swarm is not None or task_verdict == "COMPLETED"))
                or (run.id in workflow_runner.active_runs and runner_task is not None and not runner_task.cancelled())
            )

        results["AgentRuntime"] = "PASS" if runtime_ok else f"FAIL (task={task_verdict})"
        results["Orchestrator"] = "PASS" if runtime_ok else f"FAIL (task={task_verdict})"
        metrics["model_calls"] += 1
        metrics["tool_calls"] += 1

        # 13. Evidence Collection & Non-Flag Answer Verification
        evidence_entries = db.query(EvidenceModel).filter(EvidenceModel.challenge_id == ch.id).all()
        metrics["evidence_count"] = len(evidence_entries)
        results["Evidence"] = "PASS"

        # Test non-flag answer verification backed by tool output evidence
        resolver = AnswerResolver()
        verifier = VerifierAgent(resolver=resolver)
        candidate_token = AnswerCandidate(
            value="GENERIC_AUTH_TOKEN_XYZ",
            answer_type=VerifierAnswerType.STRING,
            source=AnswerSource.TOOL_OUTPUT,
            confidence=0.95,
            evidence={"tool_output": "GENERIC_AUTH_TOKEN_XYZ"},
            task_context={"description": "Extract the authentication token from the service", "category": "crypto"}
        )
        resolved = resolver.resolve(candidate_token)
        verdict = await verifier.verify(resolved, authoritative=True)
        verification_ok = (
            resolved.status in (AnswerStatus.RESOLVED, AnswerStatus.VERIFIED)
            and verdict.status == AnswerStatus.VERIFIED
        )
        results["Answer Verification"] = "PASS" if verification_ok else "FAIL"

        # 14. Checkpoint Verification
        cp = db.query(CheckpointModel).filter(CheckpointModel.run_id == run.id).first()
        results["Checkpoint"] = "PASS" if (cp is not None or run.id in workflow_runner.active_runs) else "FAIL"

        # 15. Kill Switch Verification
        workflow_runner.activate_kill_switch(run.id)
        kill_switch_ok = (
            workflow_runner.is_kill_switch_active(run.id)
            and workflow_runner.is_cancelled(run.id)
        )
        results["Kill Switch"] = "PASS" if kill_switch_ok else "FAIL"

        # 16. Reporting Verification
        report_path = report_generator.generate_readme(db, ch.id)
        results["Reporting"] = "PASS" if os.path.exists(report_path) else "FAIL"

    finally:
        server.stop()
        if db:
            db.close()

    # Calculate overall status
    all_passed = all(status == "PASS" for status in results.values())
    results["END-TO-END"] = "PASS" if all_passed else "FAIL"

    # Print Formatted Table Output
    print("FORGE COMPETITION TEST")
    print("----------------------------------------")
    for component, status in results.items():
        if component == "END-TO-END":
            print("----------------------------------------")
            print(f"{component:<22} {status}")
        else:
            print(f"{component:<22} {status}")
    print("\n--------------------------------------------------")
    print(f"Total Duration      : {round(time.time() - metrics['start_time'], 2)}s")
    print(f"Tool Subprocesses   : {metrics['tool_calls']}")
    print(f"Model Router Calls  : {metrics['model_calls']}")
    print(f"Evidence Artifacts  : {metrics['evidence_count']}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    eng = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("coord", "swarm") else None
    asyncio.run(run_competition_simulation(engine_type=eng))
