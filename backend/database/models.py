import uuid
from backend.utils.time import utcnow
from sqlalchemy import (
    Column, String, Text, Boolean, Float, Integer, DateTime, ForeignKey, JSON, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class ProviderConfigModel(Base):
    __tablename__ = "providers"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False) # gemini, nvidia, cerebras, openrouter, hf, cloudflare, agentrouter, mistral, cohere, groq
    enabled = Column(Boolean, default=True)
    is_paid = Column(Boolean, default=False)
    api_key_configured = Column(Boolean, default=False)
    latency_ms = Column(Float, default=0.0)
    health_status = Column(String, default="healthy") # healthy, degraded, unavailable
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

class ModelConfigModel(Base):
    __tablename__ = "models"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    provider_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    capability = Column(String, nullable=False) # general_reasoning, code_analysis, recon, etc.
    context_length = Column(Integer, default=8192)
    cost_per_1k_input = Column(Float, default=0.0)
    cost_per_1k_output = Column(Float, default=0.0)
    enabled = Column(Boolean, default=True)

class ChallengeModel(Base):
    __tablename__ = "challenges"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    category = Column(String, default="general") # web, recon, forensics, crypto, pwn, rev
    difficulty = Column(String, default="MEDIUM") # EASY, MEDIUM, HARD, INSANE
    description = Column(Text, default="")
    working_directory = Column(String, default="")
    platform_name = Column(String, default="")
    status = Column(String, default="QUEUED") # QUEUED, RUNNING, PAUSED, COMPLETED, FAILED
    progress = Column(Integer, default=0)
    flag_status = Column(String, default="UNFOUND")
    flag = Column(String, nullable=True)
    requires_root = Column(Boolean, default=False)
    started_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, default=0)
    mission_plan = Column(JSON, default=dict)
    approval_mode = Column(String, nullable=True, default=None) # 'auto', 'manual', or None (inherit global)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    
    targets = relationship("TargetProfileModel", back_populates="challenge", cascade="all, delete-orphan")
    runs = relationship("RunModel", back_populates="challenge", cascade="all, delete-orphan")
    evidence = relationship("EvidenceModel", back_populates="challenge", cascade="all, delete-orphan")
    findings = relationship("FindingModel", back_populates="challenge", cascade="all, delete-orphan")
    messages = relationship("ChatMessageModel", back_populates="challenge", cascade="all, delete-orphan")

class TargetProfileModel(Base):
    __tablename__ = "targets"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    challenge_id = Column(String, ForeignKey("challenges.id"), nullable=False)
    current_address = Column(String, nullable=False) # IP, URL, domain
    hostname = Column(String, nullable=True)
    expected_services = Column(JSON, default=list) # e.g. ["http", "ssh"] or [{port, proto, service, version}]
    technologies = Column(JSON, default=list) # e.g. ["Linux", "HTTP"]
    address_history = Column(JSON, default=list) # e.g. ["127.0.0.1"]
    discovery_method = Column(String, default="FORGE Auto Ingest")
    verification_status = Column(String, default="unverified") # verified, changed, stale
    last_verified_at = Column(DateTime, default=utcnow)
    
    challenge = relationship("ChallengeModel", back_populates="targets")

class AgentStateModel(Base):
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    run_id = Column(String, ForeignKey("runs.id"), nullable=False)
    agent_name = Column(String, nullable=False)
    state_data = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

class RunModel(Base):
    __tablename__ = "runs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    challenge_id = Column(String, ForeignKey("challenges.id"), nullable=False)
    status = Column(String, default="QUEUED") # QUEUED, RUNNING, WAITING_FOR_TOOL, WAITING_FOR_USER, PAUSED, COMPLETED, CANCELLED
    current_phase = Column(String, default="recon")
    current_agent = Column(String, default="orchestrator")
    started_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    
    challenge = relationship("ChallengeModel", back_populates="runs")
    checkpoints = relationship("CheckpointModel", back_populates="run", cascade="all, delete-orphan")
    tool_executions = relationship("ToolExecutionModel", back_populates="run", cascade="all, delete-orphan")

class CheckpointModel(Base):
    __tablename__ = "checkpoints"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    run_id = Column(String, ForeignKey("runs.id"), nullable=False)
    state_snapshot = Column(JSON, nullable=False)
    last_successful_action = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)
    resumable = Column(Boolean, default=True)
    
    run = relationship("RunModel", back_populates="checkpoints")

class ToolExecutionModel(Base):
    __tablename__ = "tool_executions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    run_id = Column(String, ForeignKey("runs.id"), nullable=False)
    agent = Column(String, nullable=False)
    tool_name = Column(String, nullable=False)
    capability = Column(String, nullable=False)
    command = Column(Text, nullable=False)
    privilege_level = Column(String, default="SAFE")
    approved = Column(Boolean, default=True)
    status = Column(String, default="PENDING") # PENDING, EXECUTING, SUCCESS, FAILED, TIMEOUT, KILLED
    stdout = Column(Text, default="")
    stderr = Column(Text, default="")
    exit_code = Column(Integer, nullable=True)
    duration_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utcnow)
    
    run = relationship("RunModel", back_populates="tool_executions")

class FindingModel(Base):
    __tablename__ = "findings"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    challenge_id = Column(String, ForeignKey("challenges.id"), nullable=False)
    agent = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    vulnerability_class = Column(String, default="recon")
    severity = Column(String, default="HIGH")
    endpoint = Column(String, default="")
    verified = Column(Boolean, default=False)
    confidence = Column(Float, default=0.5)
    created_at = Column(DateTime, default=utcnow)
    
    challenge = relationship("ChallengeModel", back_populates="findings")

class EvidenceModel(Base):
    __tablename__ = "evidence"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    challenge_id = Column(String, ForeignKey("challenges.id"), nullable=False)
    agent = Column(String, nullable=False)
    evidence_type = Column(String, nullable=False) # http_response, command_output, banner, flag, screenshot
    source = Column(String, nullable=False)
    content = Column(Text, default="")
    file_path = Column(String, nullable=True)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=utcnow)
    
    challenge = relationship("ChallengeModel", back_populates="evidence")

class ReportModel(Base):
    __tablename__ = "reports"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    challenge_id = Column(String, ForeignKey("challenges.id"), nullable=False)
    title = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)

class KnowledgeEntryModel(Base):
    __tablename__ = "knowledge_entries"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    category = Column(String, nullable=False)
    technique = Column(String, nullable=False)
    technology = Column(String, default="")
    attack_pattern = Column(Text, default="")
    solution_summary = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

class ProviderUsageModel(Base):
    __tablename__ = "provider_usage"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    provider_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=utcnow)

class AuditLogModel(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    agent = Column(String, nullable=False)
    action = Column(String, nullable=False)
    target = Column(String, nullable=True)
    privilege_level = Column(String, default="SAFE")
    approved = Column(Boolean, default=True)
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=utcnow)


class ChatMessageModel(Base):
    """Stores persistent chat turns scoped to a challenge."""
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    challenge_id = Column(String, ForeignKey("challenges.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, default=utcnow, index=True)

    challenge = relationship("ChallengeModel", back_populates="messages")


# =============================================================================
# EXPERIENCE / MEMORY LAYER
# -----------------------------------------------------------------------------
# FORGE's experience-based memory. A completed run is distilled into a reusable,
# GENERALIZED ExperienceModel (challenge-specific secrets stripped) so future
# missions can retrieve "we have seen conditions like this before".
#
# Provenance note: source_run_id / source_challenge_id are PLAIN string columns
# (NOT ForeignKeys). Memory must OUTLIVE the challenge/run it came from — a hard
# FK with the challenges cascade ("all, delete-orphan") would wipe learned
# experience the moment an operator deletes the original challenge. The link is
# preserved for traceability (§13) without coupling lifecycle.
# =============================================================================

class ExperienceModel(Base):
    __tablename__ = "experiences"

    id = Column(String, primary_key=True, default=generate_uuid)

    # ── Provenance (§13) — plain strings, no FK cascade, so memory survives deletion ──
    source = Column(String, default="forge_run")          # forge_run | external
    source_run_id = Column(String, nullable=True)
    source_challenge_id = Column(String, nullable=True)
    challenge_name = Column(String, default="")

    # ── Classification ────────────────────────────────────────────────────────
    category = Column(String, default="web")
    difficulty = Column(String, default="MEDIUM")
    technique = Column(String, nullable=False)            # generalized technique label
    tags = Column(JSON, default=list)                     # searchable keywords

    # ── Observed conditions & environment (generalized) ───────────────────────
    target_characteristics = Column(JSON, default=dict)
    initial_observations = Column(Text, default="")
    observed_conditions = Column(Text, default="")        # what FORGE observed
    applicable_conditions = Column(Text, default="")      # when to consider this
    discovered_endpoints = Column(JSON, default=list)
    technologies = Column(JSON, default=list)
    vulnerabilities = Column(JSON, default=list)

    # ── What worked / what did not (§5) ───────────────────────────────────────
    successful_techniques = Column(JSON, default=list)
    failed_techniques = Column(JSON, default=list)        # [{approach, reason}]
    commands_used = Column(JSON, default=list)            # generalized commands
    important_tool_outputs = Column(JSON, default=list)   # generalized snippets
    successful_attack_chain = Column(JSON, default=list)  # ordered steps
    verification_evidence = Column(Text, default="")
    success_indicators = Column(JSON, default=list)
    prerequisites = Column(JSON, default=list)
    generalized_strategy = Column(Text, default="")

    # ── Blue-team knowledge derived from the attack (§11) ─────────────────────
    detection_indicators = Column(JSON, default=dict)

    # ── Environment requirements for environment-aware skills (Phase 2, Step 12) ──
    # FORGE runs on Windows but executes tooling on Linux; a skill/experience declares
    # what the EXECUTION environment must provide. "any" ⇒ OS-independent. These are
    # advisory: the ExecutionBackend decides where a command can actually run.
    required_os = Column(String, default="any")           # any | linux | windows | darwin
    required_tools = Column(JSON, default=list)            # ["nmap", "ffuf", ...]
    required_python_libs = Column(JSON, default=list)      # ["pwntools", "requests", ...]

    # ── Outcome + learning-flywheel statistics (§12) ──────────────────────────
    outcome = Column(String, default="success")           # success | failure
    confidence = Column(Float, default=0.6)
    times_retrieved = Column(Integer, default=0)
    times_used = Column(Integer, default=0)
    times_successful = Column(Integer, default=0)
    times_failed = Column(Integer, default=0)
    success_rate = Column(Float, default=1.0)
    last_used = Column(DateTime, nullable=True)

    # ── Playbook promotion link (§10) ─────────────────────────────────────────
    promoted_playbook_id = Column(String, nullable=True)

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    attempts = relationship("ExperienceAttemptModel", back_populates="experience", cascade="all, delete-orphan")
    usages = relationship("MemoryUsageModel", back_populates="experience", cascade="all, delete-orphan")


class ExperienceAttemptModel(Base):
    __tablename__ = "experience_attempts"

    id = Column(String, primary_key=True, default=generate_uuid)
    experience_id = Column(String, ForeignKey("experiences.id"), nullable=False)
    sequence = Column(Integer, default=0)
    approach = Column(String, default="")                 # generalized approach label
    technique = Column(String, default="")
    outcome = Column(String, default="failure")           # success | failure
    reason = Column(Text, default="")                     # why it failed / how verified
    evidence = Column(Text, default="")                   # generalized supporting output
    created_at = Column(DateTime, default=utcnow)

    experience = relationship("ExperienceModel", back_populates="attempts")


class MemoryUsageModel(Base):
    __tablename__ = "memory_usage"

    id = Column(String, primary_key=True, default=generate_uuid)
    experience_id = Column(String, ForeignKey("experiences.id"), nullable=False)
    run_id = Column(String, nullable=True)
    challenge_id = Column(String, nullable=True)
    event = Column(String, default="retrieved")           # retrieved | used | success | failure
    note = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

    experience = relationship("ExperienceModel", back_populates="usages")


# =============================================================================
# AGENT RUNTIME — CANONICAL SESSION & TRAJECTORY (HERMES-inspired, FORGE-native)
# -----------------------------------------------------------------------------
# Central principle: "THE MODEL IS NOT THE MEMORY." FORGE owns the complete
# session/trajectory state so any external LLM provider can be swapped at any
# time (Groq → Gemini → OpenRouter → a future local model) WITHOUT losing the
# mission. These two tables are the durable source of truth for
#
#     command → stdout/stderr → observation → state change → decision → next action
#
# for every agent turn, so a run can be resumed after a process crash with zero
# information loss.
#
# Provenance columns (run_id / challenge_id / agent_id) are PLAIN INDEXED strings,
# NOT ForeignKeys — the same deliberate choice made for the experience layer.
# A hard FK into `challenges` (cascade "all, delete-orphan") would wipe a
# session's trajectory the moment an operator deletes the challenge; the runtime
# must instead OUTLIVE the challenge/run for cross-session recall (§4 FTS5).
# ToolExecutionModel remains valid and useful, but is no longer the ONLY
# representation of agent history.
# =============================================================================

class AgentSessionModel(Base):
    """One durable mission session. Survives model/provider changes and restarts."""
    __tablename__ = "agent_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)

    # ── Provenance (plain indexed strings, no FK cascade) ──────────────────────
    run_id = Column(String, index=True, nullable=True)
    challenge_id = Column(String, index=True, nullable=True)
    agent_id = Column(String, index=True, default="orchestrator")

    # ── Which engine owns/created this session (facade compatibility) ──────────
    engine = Column(String, default="runtime")     # runtime | swarm | cli_agent | orchestrator

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    status = Column(String, default="CREATED", index=True)  # CREATED, RUNNING, PAUSED, COMPLETED, CANCELLED, FAILED
    phase = Column(String, default="recon")
    objective = Column(Text, default="")
    target_scope = Column(String, default="")

    # ── Serialized structured MissionState (the resumable brain) ───────────────
    state = Column(JSON, default=dict)
    last_sequence = Column(Integer, default=0)      # highest trajectory sequence persisted

    # ── Current provider/model (informational — a session is provider-agnostic) ─
    provider_name = Column(String, default="")
    model_name = Column(String, default="")

    # ── Outcome ─────────────────────────────────────────────────────────────────
    verified_flag = Column(String, nullable=True)
    outcome = Column(String, nullable=True)         # success | failure | None (in progress)
    total_prompt_tokens = Column(Integer, default=0)
    total_completion_tokens = Column(Integer, default=0)

    created_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    completed_at = Column(DateTime, nullable=True)


class TrajectoryEventModel(Base):
    """A single meaningful agent event. Never depend solely on in-memory history."""
    __tablename__ = "trajectory_events"

    id = Column(String, primary_key=True, default=generate_uuid)

    # ── Identity / provenance (all indexed for recall — §3) ────────────────────
    session_id = Column(String, index=True, nullable=False)
    run_id = Column(String, index=True, nullable=True)
    challenge_id = Column(String, index=True, nullable=True)
    agent_id = Column(String, index=True, default="orchestrator")
    sequence = Column(Integer, default=0, index=True)

    # ── Event classification (drives the observability stream — §13) ───────────
    # SESSION_START, PLAN, ACTION, COMMAND, OUTPUT, OBSERVATION, DECISION,
    # STATE_UPDATE, REPLAN, RECOVERY, FLAG_CANDIDATE, FLAG_VERIFIED, SESSION_COMPLETE
    event_type = Column(String, default="ACTION", index=True)

    # ── Action ──────────────────────────────────────────────────────────────────
    action_type = Column(String, default="")        # command | python_script | tool_call | complete
    command = Column(Text, default="")
    tool_name = Column(String, default="")

    # ── Complete captured result ─────────────────────────────────────────────────
    stdout = Column(Text, default="")
    stderr = Column(Text, default="")
    exit_code = Column(Integer, nullable=True)
    duration_ms = Column(Float, default=0.0)

    # ── Derived reasoning artefacts ───────────────────────────────────────────────
    observation = Column(JSON, default=dict)         # serialized Observation
    state_delta = Column(JSON, default=dict)         # what changed in MissionState
    decision_summary = Column(Text, default="")
    strategy = Column(String, default="")
    result = Column(String, default="")              # SUCCESS | FAILED | TIMEOUT | REJECTED | ...

    # ── Provider/model + token usage (if available) ───────────────────────────────
    provider = Column(String, default="")
    model = Column(String, default="")
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)

    created_at = Column(DateTime, default=utcnow, index=True)

    # Composite index for the hot read path: ordered replay of one session.
    __table_args__ = (
        Index("ix_trajectory_session_seq", "session_id", "sequence"),
        Index("ix_trajectory_challenge_type", "challenge_id", "event_type"),
    )


# ============================================================================ #
# Phase 4 — Multi-agent swarm coordination (durable, resumable)
#
# These three tables persist the coordination layer that sits ABOVE the Phase-1
# AgentRuntime (agent_sessions / trajectory_events). Following the same durable
# pattern as that layer, provenance columns (mission_id / run_id / challenge_id)
# are PLAIN indexed strings with NO ForeignKey cascade, so coordination state
# OUTLIVES challenge deletion and survives restarts (crash-resumable missions).
# Created by Base.metadata.create_all in init_db(); no manual migration needed.
# ============================================================================ #

class SwarmMissionModel(Base):
    """One coordinated multi-agent mission — the supervisor's resumable brain."""
    __tablename__ = "swarm_missions"

    id = Column(String, primary_key=True, default=generate_uuid)

    # ── Provenance (plain indexed strings, no FK cascade) ──────────────────────
    run_id = Column(String, index=True, nullable=True)
    challenge_id = Column(String, index=True, nullable=True)
    # The supervisor's own trajectory session (coordination events live there,
    # kept separate from each specialist agent's execution trajectory).
    coord_session_id = Column(String, index=True, nullable=True)

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    # PLANNING, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED
    status = Column(String, default="PLANNING", index=True)
    strategy = Column(Text, default="")
    progress = Column(Integer, default=0)

    # ── Serialized SharedMissionState (aggregated cross-agent knowledge) ───────
    shared_state = Column(JSON, default=dict)

    # ── Outcome ─────────────────────────────────────────────────────────────────
    verified_flag = Column(String, nullable=True)

    created_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    completed_at = Column(DateTime, nullable=True)


class SwarmTaskModel(Base):
    """A structured unit of coordinated work assigned to a specialist agent."""
    __tablename__ = "swarm_tasks"

    id = Column(String, primary_key=True, default=generate_uuid)

    # ── Provenance / ownership ─────────────────────────────────────────────────
    mission_id = Column(String, index=True, nullable=False)
    run_id = Column(String, index=True, nullable=True)
    challenge_id = Column(String, index=True, nullable=True)
    parent_task_id = Column(String, index=True, nullable=True)

    # ── Assignment ─────────────────────────────────────────────────────────────
    role = Column(String, default="recon", index=True)   # recon|web|forensics|crypto|pwn|rev
    assigned_agent = Column(String, default="", index=True)  # concrete agent id (role#n)
    objective = Column(Text, default="")
    priority = Column(Integer, default=50)                # higher = more urgent

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    # PENDING, READY, RUNNING, BLOCKED, COMPLETED, FAILED, CANCELLED, REASSIGNED
    status = Column(String, default="PENDING", index=True)
    dependencies = Column(JSON, default=list)            # [task_id, ...] must COMPLETE first
    evidence_ids = Column(JSON, default=list)            # evidence produced by this task

    # ── Bounds / bookkeeping ───────────────────────────────────────────────────
    retry_count = Column(Integer, default=0)
    timeout_seconds = Column(Integer, default=0)
    signature = Column(String, index=True, default="")   # normalized dedup signature

    # ── Result ─────────────────────────────────────────────────────────────────
    result = Column(JSON, default=dict)
    failure_reason = Column(Text, default="")
    agent_session_id = Column(String, index=True, nullable=True)  # the AgentSession that ran it

    # ── Phase 4.x coordination hints ───────────────────────────────────────────
    # Persisted so the pre-dispatch capability gate and target-mismatch gate still
    # fire for a task that is reloaded after a checkpoint/resume, instead of being
    # silently dropped because these were in-memory only (Phase 4.x hardening §5/§6).
    required_capabilities = Column(JSON, default=list)   # capability names the task needs
    target_type = Column(String, default="")             # required TargetType.value (or "")

    created_at = Column(DateTime, default=utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_swarm_task_mission_status", "mission_id", "status"),
    )


class SwarmEvidenceModel(Base):
    """Structured evidence published on the Evidence Bus (agent → agent comms)."""
    __tablename__ = "swarm_evidence"

    id = Column(String, primary_key=True, default=generate_uuid)

    # ── Provenance ─────────────────────────────────────────────────────────────
    mission_id = Column(String, index=True, nullable=False)
    run_id = Column(String, index=True, nullable=True)
    challenge_id = Column(String, index=True, nullable=True)
    agent_id = Column(String, index=True, default="")
    task_id = Column(String, index=True, nullable=True)

    # ── Content ────────────────────────────────────────────────────────────────
    # service|endpoint|technology|credential|vulnerability|artifact|flag|note|failure
    evidence_type = Column(String, default="note", index=True)
    title = Column(String, default="")
    description = Column(Text, default="")
    source = Column(String, default="")            # command | observation | agent | supervisor
    command = Column(Text, default="")
    output = Column(Text, default="")              # truncated raw output / reference
    artifact_id = Column(String, nullable=True)
    confidence = Column(Float, default=0.7)
    tags = Column(JSON, default=list)

    # ── Cross-references (let another specialist consume this lead) ────────────
    related_endpoint = Column(String, default="")
    related_technology = Column(String, default="")
    related_vulnerability = Column(String, default="")

    signature = Column(String, index=True, default="")  # dedup signature
    created_at = Column(DateTime, default=utcnow, index=True)

    __table_args__ = (
        Index("ix_swarm_evidence_mission_type", "mission_id", "evidence_type"),
    )

# ============================================================================ #
# Schema versioning
#
# `Base.metadata.create_all()` only creates tables that are missing; it never
# alters an existing one. Columns added to a model after a database was first
# created therefore need an explicit ALTER TABLE for that database — the ordered
# list in backend/database/session.py. This table records which of those have
# been applied, so `SELECT MAX(version) FROM schema_version` is the authoritative
# answer to "what schema is this database at?".
#
# Deliberately not Alembic: for a single-developer tool with one migration set, a
# versioned list plus this table is the honest size of the problem.
# ============================================================================ #

class SchemaVersionModel(Base):
    """One row per applied migration in session.py's MIGRATIONS list."""
    __tablename__ = "schema_version"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, nullable=False)
    applied_at = Column(DateTime, default=utcnow)


class FailedApproachModel(Base):
    """Cross-mission record of approaches that repeatedly failed (Workstream A4).

    Keyed by (category, signature) and SHARED by both swarm engines. This is the
    cross-mission half of the failed-method guarantee: it is ADVISORY recall injected
    into prompts ("these approaches dead-ended in prior missions"), while the HARD
    per-mission block stays in each engine's own state (blocked_failure_sigs /
    mission.failed_approaches). No ForeignKey: this memory must OUTLIVE the challenge
    it came from, exactly like the experience/technique-stats layer."""
    __tablename__ = "failed_approaches"

    id = Column(String, primary_key=True, default=generate_uuid)
    category = Column(String, default="", index=True)
    signature = Column(String, nullable=False)
    fail_count = Column(Integer, default=1)
    last_reason = Column(Text, default="")
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (Index("ix_failed_approach_cat_sig", "category", "signature", unique=True),)


class ProviderBreakerModel(Base):
    """Persisted circuit-breaker state (Workstream A2) so a tripped provider/model/
    capability stays circuit-broken across a process restart instead of resetting to
    'healthy' every launch. The key is QuotaManager.breaker_key(provider, model, capability);
    expiry_ts is a POSIX timestamp. Rows past expiry are ignored on load and overwritten."""
    __tablename__ = "provider_breakers"

    key = Column(String, primary_key=True)
    expiry_ts = Column(Float, nullable=False)
    reason = Column(Text, default="")
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
