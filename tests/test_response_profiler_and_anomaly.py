"""Tests for FORGE Response Profiler, Statistical Anomaly Detection, Generic Artifact Extraction,
Priority Preemption, and Post-Exploitation Probing.
"""

import os

# Pinned above the first backend import on purpose. Importing backend used to repoint
# DATABASE_URL at production forge.db (load_dotenv override=True) at import time, so
# the pin had to be re-applied afterwards. It cannot any more: pydantic-settings gives
# real environment variables precedence over .env, and backend/database/guard.py
# refuses to build an engine for forge.db without an authorization that only the
# server's startup hook makes. Never point this at forge.db: other modules' tearDowns
# delete real rows.
os.environ["DATABASE_URL"] = "sqlite:///./test_forge.db"

from backend.agents.response_profiler import (
    EndpointBaseline,
    ResponseProfiler,
    extract_generic_artifacts,
    generate_post_exploitation_probes,
    _tokenize_structure,
)
from backend.agent_runtime.observation import ObservationEngine
from backend.agents.swarm_state import SwarmBlackboard

def test_endpoint_baseline_statistical_and_status_anomaly():
    baseline = EndpointBaseline(target_key="http://target.local/upload")

    # Initially not established
    is_anom, score, reasons = baseline.evaluate_divergence(length=50, status_code=200, min_samples=2)
    assert not is_anom
    assert baseline.sample_count == 0

    # Record 3 identical failure responses (e.g., 50 bytes, status 200)
    baseline.record(length=50, status_code=200, struct_hash="hash_a")
    baseline.record(length=50, status_code=200, struct_hash="hash_a")
    baseline.record(length=52, status_code=200, struct_hash="hash_a")

    assert baseline.sample_count == 3
    assert abs(baseline.mean_length - 50.66) < 0.1
    assert baseline.dominant_status == 200

    # Test 1: Another similar response -> NOT anomalous
    is_anom, score, reasons = baseline.evaluate_divergence(length=51, status_code=200, struct_hash="hash_a")
    assert not is_anom
    assert score < 0.35

    # Test 2: Status code change (e.g. 500 or 302) -> ANOMALOUS
    is_anom, score, reasons = baseline.evaluate_divergence(length=50, status_code=302, struct_hash="hash_a")
    assert is_anom
    assert score >= 0.35
    assert any("Status code changed" in r for r in reasons)

    # Test 3: Response length delta (e.g. 180 bytes vs 50 bytes) -> ANOMALOUS
    is_anom, score, reasons = baseline.evaluate_divergence(length=180, status_code=200, struct_hash="hash_a")
    assert is_anom
    assert score >= 0.35
    assert any("Response length 180 bytes diverges" in r for r in reasons)

    # Test 4: Structural hash change -> adds divergence score
    is_anom, score, reasons = baseline.evaluate_divergence(length=50, status_code=200, struct_hash="hash_b")
    assert score >= 0.3
    assert any("Structural skeleton hash" in r for r in reasons)

def test_generic_artifact_extraction_no_hardcoded_keywords():
    # Arbitrary text from a target response containing URLs, relative paths, directory endpoints, and tokens
    sample_text = (
        "File processed: /var/data/custom_dir/my_solver.phtml\n"
        "Reference ID: a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d\n"
        "See also: http://target.local:8080/view?id=42\n"
        "Relative location: static/media/avatar.png\n"
        "Folder: /internal/api/v2/\n"
    )

    extracted = extract_generic_artifacts(sample_text, base_url="http://target.local:8080/")

    types = {c.artifact_type for c in extracted}
    assert "URL" in types
    assert "PATH" in types
    assert "ENDPOINT" in types
    assert "TOKEN" in types

    raw_values = [c.raw_value for c in extracted]
    assert "http://target.local:8080/view?id=42" in raw_values
    assert "static/media/avatar.png" in raw_values or "avatar.png" in raw_values or any("avatar.png" in v for v in raw_values)
    assert "/var/data/custom_dir/my_solver.phtml" in raw_values or any("my_solver.phtml" in v for v in raw_values)
    assert "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d" in raw_values

def test_post_exploitation_probe_generation():
    probes_php = generate_post_exploitation_probes("http://target.local/uploads/shell.php")
    assert len(probes_php) >= 3
    assert any("cmd=" in p for p in probes_php)
    assert any("curl" in p for p in probes_php)

    probes_rel = generate_post_exploitation_probes("files/payload.phtml", base_url="http://target.local:5000/")
    assert any("http://target.local:5000/files/payload.phtml" in p for p in probes_rel)
    assert any("cat+/flag" in p for p in probes_rel)

def test_response_profiler_simulation_of_failure_and_exploit_anomaly():
    profiler = ResponseProfiler()
    cmd = 'curl -s -F "file=@payload.php" http://target.local/upload.php'

    # Step 1: Multiple identical failure responses (46 bytes error message)
    failure_output = "Sorry, there was an error uploading your file."
    for _ in range(3):
        res = profiler.profile_and_evaluate(cmd, failure_output, status_code=200, base_url="http://target.local/")
        assert not res.is_anomalous

    # Step 2: An anomalous response arrives containing a path and differing length
    anom_output = "File stored at: assets/uploads/custom_shell.php (size 140 bytes)"
    res_anom = profiler.profile_and_evaluate(cmd, anom_output, status_code=200, base_url="http://target.local/")

    assert res_anom.is_anomalous
    assert len(res_anom.candidate_artifacts) > 0
    candidate_targets = [c.normalized_target for c in res_anom.candidate_artifacts]
    assert any("assets/uploads/custom_shell.php" in t for t in candidate_targets)

def test_observation_engine_detects_anomaly_and_candidates():
    engine = ObservationEngine()
    target_url = "http://target.local:9000/"

    class FakeState:
        target = target_url
        known_endpoints = []
        known_services = []
        technologies = []
        known_files = []
        credentials = []
        vulnerabilities = []
        flag_candidates = []
        headers = {}
        cookies = {}

    class FakeResult:
        stdout = "Upload failed: file invalid."
        stderr = ""
        command = "curl http://target.local:9000/submit"
        exit_code = 0
        status = "SUCCESS"

    state = FakeState()
    # Baseline establishment
    engine.observe(FakeResult(), state)
    engine.observe(FakeResult(), state)

    # Anomalous result
    class AnomResult:
        stdout = "Upload succeeded: destination is /server/storage/uploaded_exploit.php"
        stderr = ""
        command = "curl http://target.local:9000/submit"
        exit_code = 0
        status = "SUCCESS"

    obs = engine.observe(AnomResult(), state)
    assert obs.anomalous_response is not None
    assert obs.novelty is True
    assert any("uploaded_exploit.php" in ep for ep in obs.new_endpoints)

def test_swarm_blackboard_actionable_preemption_and_history():
    board = SwarmBlackboard(
        challenge_id="chal-test-123",
        run_id="run-test-123",
        target_scope="http://example-ctf.local:8000/",
    )

    # Populate baseline
    cmd = "curl -s http://example-ctf.local:8000/upload.php"
    for _ in range(3):
        board.response_profiler.profile_and_evaluate(cmd, "Error uploading file", base_url=board.target_scope)

    # Evaluate anomaly with candidate path
    anom_output = "Success: /uploads/avatar_123.php"
    anom_res = board.response_profiler.profile_and_evaluate(cmd, anom_output, base_url=board.target_scope)
    assert anom_res.is_anomalous

    for cand in anom_res.candidate_artifacts:
        targ = cand.normalized_target or cand.raw_value
        if targ not in board.seen_candidate_targets:
            board.seen_candidate_targets.add(targ)
            probes = generate_post_exploitation_probes(targ, base_url=board.target_scope)
            board.actionable_preemptions.append({
                "raw_value": cand.raw_value,
                "artifact_type": cand.artifact_type,
                "normalized_target": targ,
                "source_command": cmd,
                "reasons": anom_res.reasons,
                "suggested_probes": probes,
                "handled": False,
            })

    # Verify history context renders priority preemption banner
    history_prompt = board.build_history_context("agent_1")
    assert "━━ ACTIONABLE ANOMALY PREEMPTION (PRIORITY FOLLOW-UP) ━━" in history_prompt
    assert "avatar_123.php" in history_prompt
    assert "SUGGESTED VERIFICATION PROBE:" in history_prompt

    # Test snapshot serialization and rehydration
    plan = board._build_mission_plan()
    snapshot = plan["blackboard_state"]
    assert "response_profiler" in snapshot
    assert "actionable_preemptions" in snapshot
    assert len(snapshot["actionable_preemptions"]) > 0

    new_board = SwarmBlackboard(
        challenge_id="chal-test-123",
        run_id="run-test-123-resumed",
        target_scope="http://example-ctf.local:8000/",
    )
    new_board.load_snapshot(snapshot)
    assert len(new_board.actionable_preemptions) == 1
    assert "avatar_123.php" in new_board.actionable_preemptions[0]["normalized_target"]
