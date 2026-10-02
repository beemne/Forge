"""Challenge lifecycle: CRUD, control (pause/resume), log, plan, candidates,
decisions, derived artifacts, writeup and report routes."""

import asyncio
import logging
import os
import re
import shutil
import uuid as _uuid_mod
from datetime import datetime, timezone
from backend.utils.time import utcnow
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.runner import workflow_runner
from backend.database.models import (
    ChallengeModel,
    ChatMessageModel,
    EvidenceModel,
    FindingModel,
    ReportModel,
    RunModel,
    SwarmEvidenceModel,
    SwarmMissionModel,
    TargetProfileModel,
    ToolExecutionModel,
    TrajectoryEventModel,
)
from backend.database.session import get_db
from backend.providers.router import model_router
from backend.reporting.generator import report_generator
from backend.utils.workspace import CTF_WORKSPACE_ROOT, is_deletable_working_dir
from backend.websocket.manager import ws_manager

router = APIRouter()
logger = logging.getLogger("forge.routes")

# ---------------------------------------------------------------------------
# CHAT-SESSION STATE (in-memory, per-process, keyed by session_id)
# Each entry lives only until the session commits or is garbage-collected.
# Structure: { session_id: { "name": str, "platform": str, "type": str,
#              "step": 1|2, "uploaded_paths": [str] } }
# ---------------------------------------------------------------------------
_CHAT_SESSIONS: Dict[str, dict] = {}


def _safe_delete_working_dir(working_dir: str):
    """Delete a challenge working directory ONLY if it is safely inside the CTF workspace root.

    Historic bug: challenges created before the ~/Documents/CTF hierarchy stored
    working_directory="." which resolved to the project root, so rmtree wiped the
    whole project. Path safety now lives in backend.utils.workspace and uses a
    strict allowlist instead of a blocklist.
    """
    if not working_dir or not isinstance(working_dir, str):
        return
    clean_path = os.path.abspath(working_dir.strip())
    if not is_deletable_working_dir(clean_path):
        logger.warning(
            f"Refused to delete working directory outside CTF workspace root: {clean_path} "
            f"(workspace root: {CTF_WORKSPACE_ROOT})"
        )
        return
    if os.path.exists(clean_path) and os.path.isdir(clean_path):
        try:
            shutil.rmtree(clean_path, ignore_errors=True)
            logger.info(f"Successfully deleted challenge working directory: {clean_path}")
        except Exception as e:
            logger.warning(f"Error removing working directory {clean_path}: {e}")


def _delete_challenge_log(challenge_id: str):
    """Delete the dedicated challenge log file(s) if they exist.

    Removes both the mirrored-structure log (logs/<Platform>/<Category>/
    <Difficulty>/<Name>/challenge_<id>.log) and any legacy flat log, then prunes
    now-empty parent directories up to (but not including) the logs base.
    """
    try:
        from backend.utils.challenge_paths import (
            iter_candidate_log_paths, forget_challenge_log_path, logs_base,
        )
        base = logs_base()
        removed_dirs: set = set()
        for log_file in iter_candidate_log_paths(challenge_id):
            try:
                if os.path.exists(log_file):
                    os.remove(log_file)
                removed_dirs.add(os.path.dirname(log_file))
            except Exception as e:
                logger.debug(f"Error removing log {log_file}: {e}")
        # Prune empty mirrored parent dirs (never the base itself).
        for start in removed_dirs:
            d = start
            while d and os.path.abspath(d) != os.path.abspath(base) and \
                    os.path.abspath(d).startswith(os.path.abspath(base) + os.sep):
                try:
                    if os.path.isdir(d) and not os.listdir(d):
                        os.rmdir(d)
                        d = os.path.dirname(d)
                    else:
                        break
                except Exception:
                    break
        forget_challenge_log_path(challenge_id)
    except Exception as e:
        logger.debug(f"Error removing log for challenge {challenge_id}: {e}")


def extract_target_from_text(text: str) -> str:
    """Intelligently extracts target network endpoint, URL, netcat connection, or artifact from description."""
    if not text:
        return ""
    # Look for http(s) URL
    url_match = re.search(r'https?://[^\s]+', text, re.IGNORECASE)
    if url_match:
        return url_match.group(0).rstrip(".,;)\"'>")
    # Look for netcat connection: nc <host> <port>
    nc_match = re.search(r'nc\s+([a-zA-Z0-9.\-_]+)\s+(\d+)', text, re.IGNORECASE)
    if nc_match:
        return f"{nc_match.group(1)}:{nc_match.group(2)}"
    # Look for IP:port or IP
    ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b', text)
    if ip_match:
        return ip_match.group(0)
    # Look for hostname:port
    host_match = re.search(r'\b([a-zA-Z0-9-]+\.[a-zA-Z0-9.\-]+:\d+)\b', text)
    if host_match:
        return host_match.group(0)
    # Look for artifact or file path
    file_match = re.search(r'(?:[a-zA-Z]:[\\/]|(?:\/|~\/|\.\/))[^\s]+?\.(?:pcap|zip|bin|elf|tar|gz|py|c|exe|txt|raw)', text, re.IGNORECASE)
    if file_match:
        return file_match.group(0)
    return ""


# Request Models
class CreateChallengeRequest(BaseModel):
    name: str
    category: str = "WEB"
    difficulty: str = "MEDIUM"
    description: str = ""
    target_address: Optional[str] = ""
    working_directory: Optional[str] = ""
    platform_name: Optional[str] = ""
    requires_root: bool = False
    # Per-run agent config (flexible-agent engine + HITL checkpoint).
    flag_pattern: Optional[str] = ""            # VALIDATION FILTER only, never a target
    max_iterations: Optional[int] = 0           # 0 -> config default AGENT_MAX_ITERATIONS
    max_minutes: Optional[int] = 0              # 0 -> config default AGENT_MAX_MINUTES
    instance_expiry_minutes: Optional[int] = 0  # minutes from now the instance dies; 0 -> none
    attached_file_paths: Optional[List[str]] = None   # server paths from /challenges/upload
    approval_mode: Optional[str] = None               # "auto", "manual", or None (inherit global)


class UpdateChallengeModeRequest(BaseModel):
    mode: Optional[str] = None                        # "auto", "manual", or None (inherit global)


class SaveWriteupRequest(BaseModel):
    # The operator-confirmed markdown to persist. When omitted, the backend
    # regenerates a deterministic writeup from the challenge's real telemetry.
    content: Optional[str] = None


# Chat-session request / response models
class ChatSessionStartRequest(BaseModel):
    """Body for POST /challenges/chat-session — starts a new two-turn session."""
    # No required fields: the first bot message is always the same fixed prompt.


class ChatSessionMessageRequest(BaseModel):
    """Body for POST /challenges/chat-session/{session_id}/message."""
    # Turn-1 fields (required on step == 1)
    challenge_name: Optional[str] = None
    platform_name: Optional[str] = None
    challenge_type: Optional[str] = None
    # Turn-2 fields (required on step == 2)
    target_address: Optional[str] = None
    description: Optional[str] = None
    # Paths returned by POST /challenges/upload — already staged on disk
    attached_file_paths: Optional[List[str]] = None


class PostChallengeChatMessageRequest(BaseModel):
    """Body for POST /challenges/{challenge_id}/messages."""
    content: str


# ----------------------------------------------------
# CHAT-DRIVEN CHALLENGE CREATION
# ----------------------------------------------------

@router.post("/challenges/chat-session")
async def start_chat_session(_req: ChatSessionStartRequest = None):
    """Open a new two-turn chat session.

    Returns a session_id the client must carry through the second turn, plus
    the first bot prompt the UI should display in the chat window.
    """
    session_id = _uuid_mod.uuid4().hex
    _CHAT_SESSIONS[session_id] = {
        "step": 1,
        "name": None,
        "platform": None,
        "type": None,
        "uploaded_paths": [],
    }
    return {
        "session_id": session_id,
        "step": 1,
        "bot_message": (
            "Let's set up your challenge.\n\n"
            "Please tell me:\n"
            "1. **Challenge name** — what is this challenge called?\n"
            "2. **Platform / event name** — e.g. PicoCTF, HackTheBox, DEF CON, …\n"
            "3. **Challenge type** — e.g. Web, Pwn, Crypto, Forensics, Rev, or anything you like.\n\n"
            "You can answer all three in one message."
        ),
    }


@router.post("/challenges/chat-session/{session_id}/message")
async def send_chat_message(
    session_id: str,
    req: ChatSessionMessageRequest,
    db: Session = Depends(get_db),
):
    """Advance a chat session by one turn.

    * **Turn 1** (step==1): expects challenge_name, platform_name, challenge_type.
      Stores them and returns the step-2 prompt.
    * **Turn 2** (step==2): expects description (required) plus optional
      target_address and attached_file_paths.  Commits by calling the existing
      create_challenge() logic and returns the new challenge row.
    """
    session = _CHAT_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found or already committed")

    step = session["step"]

    # ------------------------------------------------------------------
    # TURN 1: collect the three required fields
    # ------------------------------------------------------------------
    if step == 1:
        name = (req.challenge_name or "").strip()
        platform = (req.platform_name or "").strip()
        challenge_type = (req.challenge_type or "").strip()

        if not name or not platform or not challenge_type:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Please provide all three fields: challenge_name, "
                    "platform_name, and challenge_type."
                ),
            )

        session["name"] = name
        session["platform"] = platform
        session["type"] = challenge_type
        session["step"] = 2

        return {
            "session_id": session_id,
            "step": 2,
            "bot_message": (
                f"Got it — **{name}** on **{platform}** ({challenge_type}).\n\n"
                "Now, optionally:\n"
                "• Paste a **target address** (IP, URL, or `nc host port`) if you have one.\n"
                "• Attach a **challenge file** using the upload button.\n\n"
                "And in one sentence or more: **what is your goal for this challenge?** "
                "(This becomes the challenge description.)\n\n"
                "Send your message when ready — the challenge will be created immediately."
            ),
        }

    # ------------------------------------------------------------------
    # TURN 2: collect optional target / files / goal, then create
    # ------------------------------------------------------------------
    if step == 2:
        description = (req.description or "").strip()
        if not description:
            raise HTTPException(
                status_code=422,
                detail="Please include a description / goal for the challenge.",
            )

        # Merge any file paths uploaded before this turn (via /challenges/upload)
        extra_paths = [p for p in (req.attached_file_paths or []) if p and os.path.isfile(p)]
        all_paths = session.get("uploaded_paths", []) + extra_paths

        # Build the same request object the form path uses
        creation_req = CreateChallengeRequest(
            name=session["name"],
            category=session["type"],          # stored verbatim, no coercion
            difficulty="MEDIUM",               # sensible default; user can change via form later
            description=description,
            target_address=(req.target_address or "").strip(),
            working_directory="",
            platform_name=session["platform"],
            requires_root=False,
            flag_pattern="",
            max_iterations=0,
            max_minutes=0,
            instance_expiry_minutes=0,
            attached_file_paths=all_paths or None,
        )

        # Delegate entirely to the existing creation endpoint so logic is never forked.
        # Remove session before awaiting to avoid double-commit on retry.
        _CHAT_SESSIONS.pop(session_id, None)

        challenge_row = await create_challenge(creation_req, db)

        # Seed initial creation exchange into ChatMessageModel for durable history
        try:
            bot_turn1 = (
                "Let's set up your challenge.\n\n"
                "Please tell me:\n"
                "1. **Challenge name** — what is this challenge called?\n"
                "2. **Platform / event name** — e.g. PicoCTF, HackTheBox, DEF CON, …\n"
                "3. **Challenge type** — e.g. Web, Pwn, Crypto, Forensics, Rev, or anything you like."
            )
            user_turn1 = f"Challenge Name: {session['name']}\nPlatform: {session['platform']}\nType: {session['type']}"
            bot_turn2 = f"Got it — **{session['name']}** on **{session['platform']}** ({session['type']})."
            user_turn2 = f"Target: {req.target_address or 'None'}\nGoal/Description: {description}"
            bot_turn3 = f"Challenge **{challenge_row.name}** created successfully!"

            db.add_all([
                ChatMessageModel(challenge_id=challenge_row.id, role="assistant", content=bot_turn1),
                ChatMessageModel(challenge_id=challenge_row.id, role="user", content=user_turn1),
                ChatMessageModel(challenge_id=challenge_row.id, role="assistant", content=bot_turn2),
                ChatMessageModel(challenge_id=challenge_row.id, role="user", content=user_turn2),
                ChatMessageModel(challenge_id=challenge_row.id, role="assistant", content=bot_turn3),
            ])
            db.commit()
        except Exception as seed_err:
            logger.debug(f"Failed to seed creation chat messages: {seed_err}")

        return {
            "session_id": session_id,
            "step": "committed",
            "bot_message": (
                f"Challenge **{challenge_row.name}** created successfully! "
                f"Redirecting you to the challenges list…"
            ),
            "challenge": challenge_row,
        }

    # Should never reach here
    raise HTTPException(status_code=400, detail="Invalid session state")


# ----------------------------------------------------
# PERSISTENT CHALLENGE CHAT (STATE-AWARE ASSISTANT)
# ----------------------------------------------------

def _build_challenge_chat_context(challenge: ChallengeModel, db: Session) -> str:
    """Build a concise, rich operational snapshot of the challenge for LLM reasoning."""
    lines = []
    lines.append("=== CHALLENGE PROFILE ===")
    lines.append(f"Name: {challenge.name}")
    lines.append(f"Platform: {challenge.platform_name or 'Unknown'} | Category: {challenge.category} | Difficulty: {challenge.difficulty}")
    lines.append(f"Status: {challenge.status} | Progress: {challenge.progress}% | Flag Status: {challenge.flag_status}")
    if challenge.flag:
        lines.append(f"Discovered/Verified Flag: {challenge.flag}")
    if challenge.description:
        lines.append(f"Description / Objective: {challenge.description}")

    # Targets
    targets = challenge.targets or []
    if targets:
        target_lines = [f"{t.current_address} (status: {t.verification_status})" for t in targets]
        lines.append(f"Configured Targets: {', '.join(target_lines)}")

    # Active / Latest Run & Swarm Mission
    latest_run = (
        db.query(RunModel)
        .filter(RunModel.challenge_id == challenge.id)
        .order_by(RunModel.started_at.desc())
        .first()
    )
    if latest_run:
        lines.append("\n=== EXECUTION RUN STATE ===")
        lines.append(
            f"Run ID: {latest_run.id} | Status: {latest_run.status} | "
            f"Phase: {latest_run.current_phase} | Agent: {latest_run.current_agent}"
        )

    swarm_mission = (
        db.query(SwarmMissionModel)
        .filter(SwarmMissionModel.challenge_id == challenge.id)
        .order_by(SwarmMissionModel.created_at.desc())
        .first()
    )
    if swarm_mission:
        lines.append(f"Swarm Mission Status: {swarm_mission.status}")
        if swarm_mission.strategy:
            lines.append(f"Swarm Current Strategy: {swarm_mission.strategy}")

    # Findings
    findings = (
        db.query(FindingModel)
        .filter(FindingModel.challenge_id == challenge.id)
        .order_by(FindingModel.created_at.desc())
        .limit(6)
        .all()
    )
    if findings:
        lines.append(f"\n=== RECENT FINDINGS ({len(findings)}) ===")
        for f in reversed(findings):
            ep = f" (endpoint: {f.endpoint})" if f.endpoint else ""
            lines.append(f"- [{f.severity}] {f.title}{ep} - by agent '{f.agent}'")

    # Evidence (check SwarmEvidenceModel then EvidenceModel)
    swarm_evidence = (
        db.query(SwarmEvidenceModel)
        .filter(SwarmEvidenceModel.challenge_id == challenge.id)
        .order_by(SwarmEvidenceModel.created_at.desc())
        .limit(6)
        .all()
    )
    if swarm_evidence:
        lines.append(f"\n=== RECENT SWARM EVIDENCE ({len(swarm_evidence)}) ===")
        for ev in reversed(swarm_evidence):
            title = ev.title or ev.evidence_type or "evidence"
            desc = (ev.description or ev.output or "")[:180].strip().replace("\n", " ")
            lines.append(f"- [{ev.agent_id or 'agent'}] {title}: {desc}")
    else:
        classic_evidence = (
            db.query(EvidenceModel)
            .filter(EvidenceModel.challenge_id == challenge.id)
            .order_by(EvidenceModel.created_at.desc())
            .limit(6)
            .all()
        )
        if classic_evidence:
            lines.append(f"\n=== RECENT EVIDENCE ({len(classic_evidence)}) ===")
            for ev in reversed(classic_evidence):
                content = (ev.content or "")[:180].strip().replace("\n", " ")
                lines.append(f"- [{ev.agent}] {ev.evidence_type} ({ev.source}): {content}")

    # Recent Trajectory Events & Tool Executions
    traj_events = (
        db.query(TrajectoryEventModel)
        .filter(TrajectoryEventModel.challenge_id == challenge.id)
        .order_by(TrajectoryEventModel.created_at.desc())
        .limit(8)
        .all()
    )
    if traj_events:
        lines.append(f"\n=== RECENT ACTIONS & TELEMETRY ({len(traj_events)}) ===")
        for te in reversed(traj_events):
            cmd_snippet = (te.command or te.tool_name or te.action_type or te.event_type)[:120]
            status_snippet = f"status={te.result}" if te.result else ""
            if te.exit_code is not None:
                status_snippet += f", exit={te.exit_code}"
            lines.append(f"- [{te.agent_id}] {te.event_type}: `{cmd_snippet}` ({status_snippet})")
            if te.stderr:
                err_clean = te.stderr.strip().replace("\n", " ")[:140]
                lines.append(f"  Error: {err_clean}")
            elif te.stdout:
                out_clean = te.stdout.strip().replace("\n", " ")[:140]
                lines.append(f"  Output: {out_clean}")
    else:
        tool_execs = (
            db.query(ToolExecutionModel)
            .join(RunModel)
            .filter(RunModel.challenge_id == challenge.id)
            .order_by(ToolExecutionModel.created_at.desc())
            .limit(8)
            .all()
        )
        if tool_execs:
            lines.append(f"\n=== RECENT TOOL EXECUTIONS ({len(tool_execs)}) ===")
            for te in reversed(tool_execs):
                cmd_snippet = (te.command or te.tool_name)[:120]
                lines.append(f"- [{te.agent}] `{cmd_snippet}` -> {te.status} (exit {te.exit_code})")
                if te.stderr:
                    err_clean = te.stderr.strip().replace("\n", " ")[:140]
                    lines.append(f"  Stderr: {err_clean}")

    return "\n".join(lines)


@router.get("/challenges/{challenge_id}/messages")
def get_challenge_messages(challenge_id: str, db: Session = Depends(get_db)):
    """Retrieve full persistent chat history for a challenge in chronological order."""
    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    messages = (
        db.query(ChatMessageModel)
        .filter(ChatMessageModel.challenge_id == challenge_id)
        .order_by(ChatMessageModel.created_at.asc())
        .all()
    )
    return [
        {
            "id": m.id,
            "challenge_id": m.challenge_id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
    ]


def _extract_rebind_address(text: str) -> Optional[str]:
    """Detect if operator chat message is requesting a target address rebind."""
    text_clean = text.strip()
    slash_match = re.match(r"^/rebind(?:[_\-]target)?\s+(\S+)", text_clean, re.IGNORECASE)
    if slash_match:
        return slash_match.group(1).strip()

    natural_match = re.search(
        r"(?:rebind|update|change|switch)\s+(?:the\s+)?(?:target\s+)?(?:address|ip|domain|url|host)?\s*(?:to|as)\s+([^\s,;]+)",
        text_clean,
        re.IGNORECASE,
    )
    if natural_match:
        val = natural_match.group(1).strip().strip("'\"`")
        if val and not val.lower().startswith("what") and not val.lower().startswith("how"):
            return val

    return None


@router.post("/challenges/{challenge_id}/messages")
async def post_challenge_message(
    challenge_id: str,
    req: PostChallengeChatMessageRequest,
    db: Session = Depends(get_db),
):
    """Post a new chat message to an existing challenge and get an AI response aware of live state."""
    user_content = (req.content or "").strip()
    if not user_content:
        raise HTTPException(status_code=422, detail="Message content cannot be empty")

    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    # 1. Persist user message
    user_msg = ChatMessageModel(
        challenge_id=challenge_id,
        role="user",
        content=user_content,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # 1.1 Check if operator message is an address rebind instruction
    rebind_address = _extract_rebind_address(user_content)
    if rebind_address:
        from backend.api.routes.targets import perform_target_rebind

        target = (
            db.query(TargetProfileModel)
            .filter(TargetProfileModel.challenge_id == challenge_id)
            .order_by(TargetProfileModel.last_verified_at.desc())
            .first()
        )
        if not target:
            target = TargetProfileModel(
                challenge_id=challenge_id,
                current_address=rebind_address,
                hostname=rebind_address,
                verification_status="address_updated",
                address_history=[rebind_address],
            )
            db.add(target)
            db.commit()
            db.refresh(target)
            old_addr = "none"
            new_addr = rebind_address
            history_display = f"[{new_addr}]"
        else:
            old_addr = target.current_address
            target = await perform_target_rebind(target, rebind_address, db)
            new_addr = target.current_address
            history_display = " -> ".join(target.address_history or [new_addr])

        assistant_content = (
            f"🎯 **Target Address Re-bound Successfully**\n\n"
            f"- **Previous Address:** `{old_addr}`\n"
            f"- **Active Address:** `{new_addr}`\n"
            f"- **Address History Lineage:** `{history_display}`\n\n"
            f"All past evidence, captured telemetry, and execution runs remain fully preserved for challenge **{challenge.name}**."
        )

        assistant_msg = ChatMessageModel(
            challenge_id=challenge_id,
            role="assistant",
            content=assistant_content,
        )
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)

        return {
            "user_message": {
                "id": user_msg.id,
                "challenge_id": user_msg.challenge_id,
                "role": user_msg.role,
                "content": user_msg.content,
                "created_at": user_msg.created_at.isoformat() if user_msg.created_at else None,
            },
            "assistant_message": {
                "id": assistant_msg.id,
                "challenge_id": assistant_msg.challenge_id,
                "role": assistant_msg.role,
                "content": assistant_msg.content,
                "created_at": assistant_msg.created_at.isoformat() if assistant_msg.created_at else None,
            },
        }

    # 2. Query recent history for context window (last 16 messages)
    history_records = (
        db.query(ChatMessageModel)
        .filter(ChatMessageModel.challenge_id == challenge_id)
        .order_by(ChatMessageModel.created_at.desc())
        .limit(16)
        .all()
    )
    history_records.reverse()

    history_lines = []
    for msg in history_records:
        speaker = "Operator" if msg.role == "user" else "Assistant"
        history_lines.append(f"{speaker}: {msg.content}")
    history_text = "\n".join(history_lines)

    # 3. Build live challenge context
    live_context = _build_challenge_chat_context(challenge, db)

    system_instruction = (
        "You are FORGE CTF Assistant, an AI cybersecurity assistant embedded in the FORGE autonomous platform.\n"
        "You are conversing with the operator about a specific challenge.\n"
        "You have live access to the challenge's status, findings, reconnaissance, evidence, and recent agent actions.\n\n"
        "Guidelines:\n"
        "- Answer the operator's questions accurately based on the real telemetry provided below.\n"
        "- If asked 'what is happening' or 'what happened', summarize the current run status, recent actions, findings, or errors.\n"
        "- Be concise, direct, and technical.\n"
        "- Never hallucinate fake flags, open ports, or commands that do not appear in the telemetry or prompt.\n\n"
        f"--- LIVE CHALLENGE TELEMETRY ---\n"
        f"{live_context}\n"
        f"--------------------------------"
    )

    prompt = (
        f"Recent Conversation:\n{history_text}\n\n"
        f"Please provide your response to the operator's latest message."
    )

    # 4. Route request via existing model router
    response = await model_router.route_request(
        prompt=prompt,
        capability="general_reasoning",
        system_instruction=system_instruction,
    )

    assistant_content = response.content if response and not response.is_refusal else (
        response.refusal_reason or response.content or "Unable to generate response from live model providers."
    )

    # 5. Persist assistant message
    assistant_msg = ChatMessageModel(
        challenge_id=challenge_id,
        role="assistant",
        content=assistant_content,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return {
        "user_message": {
            "id": user_msg.id,
            "challenge_id": user_msg.challenge_id,
            "role": user_msg.role,
            "content": user_msg.content,
            "created_at": user_msg.created_at.isoformat() if user_msg.created_at else None,
        },
        "assistant_message": {
            "id": assistant_msg.id,
            "challenge_id": assistant_msg.challenge_id,
            "role": assistant_msg.role,
            "content": assistant_msg.content,
            "created_at": assistant_msg.created_at.isoformat() if assistant_msg.created_at else None,
        },
    }


# ----------------------------------------------------
# CHALLENGES CRUD & CONTROL
# ----------------------------------------------------

@router.get("/challenges")
def list_challenges(db: Session = Depends(get_db)):
    return db.query(ChallengeModel).all()

@router.get("/challenges/{challenge_id}")
def get_challenge(challenge_id: str, db: Session = Depends(get_db)):
    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenge

@router.patch("/challenges/{challenge_id}/mode")
@router.put("/challenges/{challenge_id}/mode")
async def update_challenge_mode(challenge_id: str, req: UpdateChallengeModeRequest, db: Session = Depends(get_db)):
    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    clean_mode = req.mode.strip().lower() if req.mode else None
    if clean_mode is not None and clean_mode not in ("auto", "manual"):
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'auto', 'manual', or null.")
    
    challenge.approval_mode = clean_mode
    db.commit()
    db.refresh(challenge)
    
    await ws_manager.broadcast({
        "event": "CHALLENGE_MODE_UPDATED",
        "challenge_id": challenge.id,
        "approval_mode": challenge.approval_mode
    })
    
    return {
        "status": "SUCCESS",
        "challenge_id": challenge.id,
        "approval_mode": challenge.approval_mode
    }


@router.get("/challenges/{challenge_id}/plan")
def get_challenge_plan(challenge_id: str, db: Session = Depends(get_db)):
    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenge.mission_plan or {"tasks": [], "status": "PENDING"}

@router.get("/challenges/{challenge_id}/log")
def get_challenge_log(challenge_id: str, db: Session = Depends(get_db)):
    """Retrieve dedicated challenge log path and file content."""
    from backend.utils.challenge_paths import resolve_challenge_log_path
    log_path = resolve_challenge_log_path(challenge_id)
    content = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            logger.warning(f"Error reading challenge log {log_path}: {e}")
    return {
        "status": "SUCCESS",
        "challenge_id": challenge_id,
        "log_file": log_path,
        "content": content,
    }


@router.get("/challenges/{challenge_id}/candidates")
def get_challenge_candidates(challenge_id: str, db: Session = Depends(get_db)):
    """Retrieve flag candidates (live or persisted snapshot) for a challenge."""
    try:
        from backend.agents.swarm_orchestrator import swarm_orchestrator
        if challenge_id in swarm_orchestrator.active_swarms:
            board = swarm_orchestrator.active_swarms[challenge_id]
            return {"challenge_id": challenge_id, "candidates": list(board.flag_candidates)}
    except Exception:
        pass

    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    plan = challenge.mission_plan or {}
    bb_state = plan.get("blackboard_state", {}) if isinstance(plan, dict) else {}
    candidates = bb_state.get("flag_candidates", [])
    return {"challenge_id": challenge_id, "candidates": candidates}


@router.get("/challenges/{challenge_id}/derived-artifacts")
def get_challenge_derived_artifacts(challenge_id: str, db: Session = Depends(get_db)):
    """Retrieve reconstructed derived artifacts (live or persisted snapshot) for a challenge."""
    try:
        from backend.agents.swarm_orchestrator import swarm_orchestrator
        if challenge_id in swarm_orchestrator.active_swarms:
            board = swarm_orchestrator.active_swarms[challenge_id]
            return {"challenge_id": challenge_id, "artifacts": list(board.derived_artifacts)}
    except Exception:
        pass

    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    plan = challenge.mission_plan or {}
    bb_state = plan.get("blackboard_state", {}) if isinstance(plan, dict) else {}
    artifacts = bb_state.get("derived_artifacts", [])
    return {"challenge_id": challenge_id, "artifacts": artifacts}


@router.get("/challenges/{challenge_id}/decisions")
def get_challenge_decisions(challenge_id: str, db: Session = Depends(get_db)):
    """Retrieve AI decision history from TrajectoryEventModel and live/persisted execution history."""
    decisions = []
    # Query database trajectory events for this challenge
    events = (db.query(TrajectoryEventModel)
              .filter(TrajectoryEventModel.challenge_id == challenge_id)
              .filter(TrajectoryEventModel.event_type.in_(["DECISION", "AI_DECISION", "PLAN", "REPLAN"]))
              .order_by(TrajectoryEventModel.created_at.desc())
              .all())

    for ev in events:
        decisions.append({
            "id": ev.id,
            "timestamp": ev.created_at.isoformat() if ev.created_at else None,
            "agent": ev.agent_id or "ORCHESTRATOR",
            "goal": ev.decision_summary or ev.strategy or "Strategic Action",
            "capability": ev.tool_name or ev.action_type or "reasoning",
            "selectedTool": ev.tool_name or ev.command or "reasoning",
            "reason": ev.decision_summary,
            "result": ev.result or ev.stdout[:200] or "SUCCESS",
            "confidence": 90,
            "model": ev.model or "FORGE Router",
            "challengeId": challenge_id
        })

    # Check live blackboard execution history if active
    try:
        from backend.agents.swarm_orchestrator import swarm_orchestrator
        if challenge_id in swarm_orchestrator.active_swarms:
            board = swarm_orchestrator.active_swarms[challenge_id]
            for idx, item in enumerate(board.execution_history):
                decisions.append({
                    "id": f"live-dec-{idx}-{item.get('ts', '')}",
                    "timestamp": item.get("ts") or "Just now",
                    "agent": item.get("agent", "SWARM_WORKER"),
                    "goal": item.get("note") or f"Executed {item.get('command', '')[:50]}",
                    "capability": "execution",
                    "selectedTool": item.get("command", "")[:40],
                    "result": (item.get("output") or "")[:200],
                    "confidence": 95,
                    "model": "Swarm Worker",
                    "challengeId": challenge_id
                })
    except Exception:
        pass

    return {"challenge_id": challenge_id, "decisions": decisions}


@router.delete("/challenges/{challenge_id}")
async def delete_challenge(challenge_id: str, db: Session = Depends(get_db)):
    """Deletes a challenge from the database, cascading to runs/findings, and deletes its working directory on disk."""
    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    working_dir = challenge.working_directory

    # 1. Delete associated reports
    db.query(ReportModel).filter(ReportModel.challenge_id == challenge_id).delete()

    # 2. Delete challenge (SQLAlchemy relationship cascade deletes runs, targets, checkpoints, tool_executions, evidence, findings)
    db.delete(challenge)
    db.commit()

    # 3. Clean up challenge working directory on disk safely
    _safe_delete_working_dir(working_dir)

    # 4. Clean up challenge dedicated log file
    _delete_challenge_log(challenge_id)

    # 5. Broadcast real-time WebSocket event
    try:
        await ws_manager.broadcast({
            "event": "CHALLENGE_DELETED",
            "challenge_id": challenge_id
        })
    except Exception:
        pass

    return {
        "status": "SUCCESS",
        "message": f"Challenge '{challenge_id}' and associated working directory deleted successfully.",
        "challenge_id": challenge_id
    }

@router.delete("/challenges")
async def delete_all_challenges(db: Session = Depends(get_db)):
    """Deletes all challenges from the database and deletes all associated working directories on disk."""
    challenges = db.query(ChallengeModel).all()
    deleted_count = 0
    for ch in challenges:
        working_dir = ch.working_directory
        ch_id = ch.id
        db.query(ReportModel).filter(ReportModel.challenge_id == ch_id).delete()
        db.delete(ch)
        _safe_delete_working_dir(working_dir)
        _delete_challenge_log(ch_id)
        deleted_count += 1

    db.commit()

    try:
        await ws_manager.broadcast({
            "event": "ALL_CHALLENGES_DELETED",
            "count": deleted_count
        })
    except Exception:
        pass

    return {
        "status": "SUCCESS",
        "message": f"All {deleted_count} challenges and associated working directories deleted successfully.",
        "deleted_count": deleted_count
    }

@router.post("/challenges")
async def create_challenge(req: CreateChallengeRequest, db: Session = Depends(get_db)):
    platform = req.platform_name.strip() if (req.platform_name and req.platform_name.strip()) else "PicoCTF"
    category = req.category.strip().upper() if req.category else "WEB"
    difficulty = req.difficulty.strip().upper() if req.difficulty else "MEDIUM"
    name = req.name.strip() if (req.name and req.name.strip()) else "Challenge_Target"

    # Enforce structured hierarchy: ~/Documents/CTF/<Platform>/<Category>/<Difficulty>/<Name>
    ctf_root_dir = os.path.expanduser(os.path.join("~", "Documents", "CTF"))
    working_dir = os.path.abspath(os.path.join(ctf_root_dir, platform, category, difficulty, name))
    os.makedirs(working_dir, exist_ok=True)

    # Move any operator-uploaded artifacts (staged by POST /challenges/upload) into
    # the challenge workspace byte-for-byte, and collect their final paths.
    attached_final: List[str] = []
    for _src in (req.attached_file_paths or []):
        try:
            if _src and os.path.isfile(_src):
                _dest = os.path.join(working_dir, os.path.basename(_src))
                if os.path.abspath(_src) != os.path.abspath(_dest):
                    shutil.move(_src, _dest)
                attached_final.append(_dest)
        except Exception as _move_err:
            logger.warning(f"Could not stage uploaded artifact '{_src}': {_move_err}")

    approval_mode = req.approval_mode.strip().lower() if (req.approval_mode and req.approval_mode.strip()) else None
    if approval_mode not in ("auto", "manual"):
        approval_mode = None

    challenge = ChallengeModel(
        name=name,
        category=category,
        difficulty=difficulty,
        description=req.description,
        working_directory=working_dir,
        platform_name=platform,
        requires_root=req.requires_root,
        approval_mode=approval_mode,
        status="RUNNING"
    )
    db.add(challenge)
    db.commit()
    db.refresh(challenge)

    resolved_target = (req.target_address or "").strip()
    if not resolved_target:
        resolved_target = extract_target_from_text(req.description)
    if not resolved_target:
        resolved_target = f"{name.lower().replace(' ', '_')}.ctf"

    multi_targets = [t.strip() for t in resolved_target.replace("+", ",").split(",") if t.strip()]
    first_target = multi_targets[0] if multi_targets else resolved_target
    is_file = os.path.exists(first_target) or len(multi_targets) > 1
    target = TargetProfileModel(
        challenge_id=challenge.id,
        current_address=resolved_target,
        hostname=os.path.basename(first_target) if (is_file and os.path.exists(first_target)) else f"{name.lower()}.ctf",
        verification_status="verified_file" if is_file else "verified_network"
    )
    db.add(target)
    db.commit()

    run = RunModel(
        challenge_id=challenge.id,
        status="RUNNING",
        current_phase="ingest",
        current_agent="orchestrator"
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    # Pre-flight Mission Plan Generation
    from backend.agents.strategic_planner import strategic_planner
    try:
        initial_plan = await strategic_planner.generate_initial_plan(
            challenge_id=challenge.id,
            challenge_name=challenge.name,
            category=challenge.category,
            difficulty=challenge.difficulty,
            target=resolved_target,
            description=challenge.description
        )
        challenge.mission_plan = initial_plan
        challenge.started_at = utcnow()
        db.commit()
        db.refresh(challenge)
    except Exception as plan_err:
        logger.warning(f"Initial plan creation fallback: {plan_err}")

    # Persist per-run agent config into mission_plan.run_config so WorkflowRunner can
    # thread budget / flag pattern / uploaded artifacts / instance timer into run_swarm.
    expiry_ts = None
    if req.instance_expiry_minutes and req.instance_expiry_minutes > 0:
        expiry_ts = datetime.now(timezone.utc).timestamp() + (req.instance_expiry_minutes * 60)
    run_config = {
        "flag_pattern": (req.flag_pattern or "").strip(),
        "max_iterations": int(req.max_iterations or 0),
        "max_minutes": int(req.max_minutes or 0),
        "attached_file_paths": attached_final,
        "instance_expiry_ts": expiry_ts,
    }
    try:
        mp = dict(challenge.mission_plan or {})
        mp["run_config"] = run_config
        challenge.mission_plan = mp
        db.commit()
        db.refresh(challenge)
    except Exception as rc_err:
        logger.warning(f"Could not persist run_config: {rc_err}")

    # Initialize Challenge Dedicated Log File FIRST — stored in a subtree mirroring the
    # challenge's own structure (logs/<Platform>/<Category>/<Difficulty>/<Name>/).
    # Must happen BEFORE background recon or runner execution starts writing log appends.
    from backend.utils.challenge_paths import register_challenge_log_path
    ch_log_path = register_challenge_log_path(
        challenge.id, platform, category, difficulty, challenge.name)
    with open(ch_log_path, "w", encoding="utf-8") as f:
        f.write(f"=== FORGE CTF CHALLENGE LOG STARTED ===\n")
        f.write(f"Timestamp: {utcnow().isoformat()} UTC\n")
        f.write(f"Challenge ID: {challenge.id}\n")
        f.write(f"Challenge Name: {challenge.name}\n")
        f.write(f"Platform: {platform} | Category: {category} | Difficulty: {difficulty}\n")
        f.write(f"Target Scope: {req.target_address}\n")
        f.write(f"Working Directory: {working_dir}\n")
        f.write(f"Run ID: {run.id}\n")
        f.write(f"=======================================\n\n")

    # Phase 3 Turbo Recon: Pre-warm recon in background immediately
    from backend.recon.turbo_recon import turbo_recon
    if resolved_target:
        asyncio.create_task(turbo_recon.start_turbo_recon(challenge.id, resolved_target, category.lower(), working_directory=working_dir))

    workflow_runner.start_run(run.id, challenge.id, resolved_target)

    await ws_manager.broadcast({
        "event": "CHALLENGE_CREATED",
        "challenge_id": challenge.id,
        "name": challenge.name,
        "target": req.target_address,
        "working_directory": working_dir,
        "log_file": ch_log_path
    })

    if challenge.mission_plan:
        await ws_manager.broadcast({
            "event": "PLAN_GENERATED",
            "challenge_id": challenge.id,
            "run_id": run.id,
            "plan": challenge.mission_plan
        })

    return challenge


@router.post("/challenges/upload")
async def upload_artifact(file: UploadFile = File(...)):
    """Byte-safe upload of a challenge artifact. Streams to a staging dir under the
    CTF workspace and returns its absolute path; create_challenge then moves it into
    the challenge workspace. Bytes never pass through any text-decoding layer."""
    import uuid as _uuid
    safe_name = os.path.basename(file.filename or "artifact.bin").replace("\\", "_").replace("/", "_")
    safe_name = "".join(c for c in safe_name if c not in '<>:"|?*').strip() or "artifact.bin"
    staging_dir = os.path.join(CTF_WORKSPACE_ROOT, "_uploads", _uuid.uuid4().hex[:12])
    os.makedirs(staging_dir, exist_ok=True)
    dest = os.path.join(staging_dir, safe_name)
    try:
        with open(dest, "wb") as fh:                     # wb — byte-for-byte, never decoded
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    finally:
        await file.close()
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    logger.info(f"[upload] staged artifact {dest} ({size} bytes)")
    return {"path": dest, "filename": safe_name, "size": size}


@router.post("/challenges/{challenge_id}/pause")
async def pause_challenge(challenge_id: str, db: Session = Depends(get_db)):
    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    # Gracefully suspend any live swarm for this challenge and persist a resume
    # snapshot. run_swarm finalizes the run as PAUSED; a later /start resumes it.
    from backend.agents.swarm_orchestrator import swarm_orchestrator
    paused_live = await swarm_orchestrator.request_pause(challenge_id)

    challenge.status = "PAUSED"
    db.commit()
    await ws_manager.broadcast({"event": "CHALLENGE_PAUSED", "challenge_id": challenge_id, "paused_live_swarm": paused_live})
    return {"status": "PAUSED", "id": challenge_id, "paused_live_swarm": paused_live}

@router.post("/challenges/{challenge_id}/resume")
async def resume_challenge(challenge_id: str, db: Session = Depends(get_db)):
    challenge = db.query(ChallengeModel).filter(ChallengeModel.id == challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    challenge.status = "RUNNING"
    db.commit()
    await ws_manager.broadcast({"event": "CHALLENGE_RESUMED", "challenge_id": challenge_id})
    return {"status": "RUNNING", "id": challenge_id}

@router.post("/challenges/{challenge_id}/report")
async def generate_report(challenge_id: str, db: Session = Depends(get_db)):
    """Author a detailed technical writeup (Gemini-first AI chain, deterministic
    fallback) from the real run telemetry and persist it under reports/."""
    content, generated_by = await report_generator.craft_writeup(db, challenge_id)
    if not content:
        raise HTTPException(status_code=404, detail="Could not generate report for challenge")
    report_path = report_generator.save_writeup(db, challenge_id, content, output_dir="reports")
    return {"status": "GENERATED", "file_path": report_path,
            "content": content, "generated_by": generated_by}


@router.get("/challenges/{challenge_id}/writeup")
async def get_writeup(challenge_id: str, refresh: bool = False, db: Session = Depends(get_db)):
    """Return the challenge writeup.

    Task #2: once the operator has SAVED a writeup, this returns THAT saved artifact
    (``saved=true``) instead of authoring a brand-new one on every open. Pass
    ``?refresh=1`` to force a fresh AI-crafted draft (Gemini-first for the
    report_generation capability; provider-chain fallback, then a deterministic
    writeup from the same real telemetry — never fabricated or empty filler).
    A saved writeup is NOT overwritten until the operator explicitly saves again.
    """
    if not refresh:
        saved = report_generator.load_saved_writeup(db, challenge_id)
        if saved is not None:
            content, file_path = saved
            return {"content": content, "generated_by": "saved",
                    "saved": True, "file_path": file_path}

    content, generated_by = await report_generator.craft_writeup(db, challenge_id)
    if not content:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return {"content": content, "generated_by": generated_by, "saved": False}


@router.post("/challenges/{challenge_id}/writeup/save")
def save_writeup_endpoint(challenge_id: str, req: SaveWriteupRequest,
                          db: Session = Depends(get_db)):
    """Persist the operator-confirmed writeup markdown into the challenge working
    folder (path-safety enforced by backend.utils.workspace)."""
    content = req.content
    if content is None:
        ctx = report_generator.gather_context(db, challenge_id)
        if ctx is None:
            raise HTTPException(status_code=404, detail="Challenge not found")
        content = report_generator.render_deterministic(ctx)
    report_path = report_generator.save_writeup(db, challenge_id, content)
    if not report_path:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return {"status": "SAVED", "file_path": report_path,
            "content": content, "generated_by": "saved", "saved": True}
