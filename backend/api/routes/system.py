"""System, environment, observability and operator-tooling routes: health and
environment detection, agent fleet state, coordinated-swarm snapshots, capability
discovery, evidence/findings/reports, settings, package-install approvals, and the
native directory browser."""

import asyncio
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.models import (
    EvidenceModel,
    FindingModel,
    ReportModel,
    SwarmEvidenceModel,
    SwarmMissionModel,
    SwarmTaskModel,
)
from backend.database.session import get_db
from backend.environment.detector import environment_detector
from backend.providers.router import model_router
from backend.websocket.manager import ws_manager

router = APIRouter()
logger = logging.getLogger("forge.routes")


# ----------------------------------------------------
# SYSTEM & ENVIRONMENT
# ----------------------------------------------------

@router.get("/health")
def health_check():
    env = environment_detector.detect_environment()
    return {
        "status": "healthy",
        "system": env["os"],
        "distro": env["distro"],
        "installed_tools_count": sum(1 for t in env["installed_tools"].values() if t["installed"]),
        "paid_models_allowed": model_router.paid_allowed,
        "daily_budget_usd": model_router.daily_budget_usd,
        "current_spent_usd": model_router.current_spent_usd
    }

@router.get("/environment")
def get_environment():
    return environment_detector.detect_environment()

@router.get("/agents")
def get_agents():
    """Live swarm worker fleet state. Empty when no swarms are active."""
    from backend.agents.swarm_orchestrator import swarm_orchestrator
    agents = []
    for _run_id, board in swarm_orchestrator.active_swarms.items():
        agents.extend(board._build_agent_states())
    return agents


# ============================================================================ #
# Phase 4 — coordinated swarm observability (§13). Read-only views of the
# supervisor + specialist agents, task queue, evidence bus, and mission state.
# Serves the LIVE coordinator when a mission is running, else the durable rows.
# ============================================================================ #

def _swarm_task_dict(t) -> dict:
    return {
        "id": t.id, "mission_id": t.mission_id, "role": t.role, "assigned_agent": t.assigned_agent,
        "objective": t.objective, "priority": t.priority, "status": t.status,
        "dependencies": t.dependencies or [], "evidence_ids": t.evidence_ids or [],
        "retry_count": t.retry_count, "parent_task_id": t.parent_task_id,
        "failure_reason": t.failure_reason, "agent_session_id": t.agent_session_id,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
    }


def _swarm_evidence_dict(e) -> dict:
    return {
        "id": e.id, "mission_id": e.mission_id, "agent_id": e.agent_id, "task_id": e.task_id,
        "type": e.evidence_type, "title": e.title, "description": e.description,
        "source": e.source, "confidence": e.confidence, "tags": e.tags or [],
        "related_endpoint": e.related_endpoint, "related_technology": e.related_technology,
        "related_vulnerability": e.related_vulnerability,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


def _assemble_swarm_snapshot(mission_row, db) -> dict:
    tasks = (db.query(SwarmTaskModel)
             .filter(SwarmTaskModel.mission_id == mission_row.id)
             .order_by(SwarmTaskModel.created_at.asc()).all())
    evidence = (db.query(SwarmEvidenceModel)
                .filter(SwarmEvidenceModel.mission_id == mission_row.id)
                .order_by(SwarmEvidenceModel.created_at.desc()).all())
    state = mission_row.shared_state or {}
    counts: dict = {"total": len(tasks)}
    for t in tasks:
        counts[t.status] = counts.get(t.status, 0) + 1
    return {
        "mission_id": mission_row.id, "run_id": mission_row.run_id,
        "challenge_id": mission_row.challenge_id, "status": mission_row.status,
        "progress": mission_row.progress, "strategy": mission_row.strategy,
        "verified_flag": mission_row.verified_flag, "shared_state": state,
        "agents": [{"agent_id": aid, **(info or {})}
                   for aid, info in (state.get("agent_statuses") or {}).items()],
        "tasks": [_swarm_task_dict(t) for t in tasks],
        "task_counts": counts,
        "evidence_count": len(evidence),
        "evidence": [_swarm_evidence_dict(e) for e in evidence[:100]],
        "live": False,
    }


def _live_swarm_snapshot(coord) -> dict:
    snap = coord.snapshot()
    try:
        snap["evidence"] = [e.to_dict() for e in coord.bus.all()[:100]]
    except Exception:
        snap["evidence"] = []
    snap["live"] = True
    return snap


@router.get("/swarm/missions")
def list_swarm_missions(db: Session = Depends(get_db)):
    """All coordinated missions (most recent first)."""
    rows = db.query(SwarmMissionModel).order_by(SwarmMissionModel.created_at.desc()).all()
    return [{
        "mission_id": r.id, "run_id": r.run_id, "challenge_id": r.challenge_id,
        "status": r.status, "progress": r.progress, "verified_flag": r.verified_flag,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in rows]


@router.get("/swarm/missions/{mission_id}")
def get_swarm_mission(mission_id: str, db: Session = Depends(get_db)):
    """Full mission snapshot — live coordinator if running, else durable rows."""
    try:
        from backend.swarm.coordinator import active_missions
        coord = active_missions.get(mission_id)
        if coord is not None:
            return _live_swarm_snapshot(coord)
    except Exception:
        pass
    row = db.query(SwarmMissionModel).filter(SwarmMissionModel.id == mission_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Mission not found")
    return _assemble_swarm_snapshot(row, db)


@router.get("/swarm/challenges/{challenge_id}")
def get_swarm_for_challenge(challenge_id: str, db: Session = Depends(get_db)):
    """Latest coordinated mission for a challenge (live if running)."""
    try:
        from backend.swarm.coordinator import active_missions
        coord = active_missions.get(challenge_id)
        if coord is not None:
            return _live_swarm_snapshot(coord)
    except Exception:
        pass
    row = (db.query(SwarmMissionModel)
           .filter(SwarmMissionModel.challenge_id == challenge_id)
           .order_by(SwarmMissionModel.created_at.desc()).first())
    if not row:
        return {"mission_id": None, "challenge_id": challenge_id, "status": "NONE",
                "tasks": [], "evidence": [], "agents": [], "evidence_count": 0,
                "task_counts": {"total": 0}, "live": False}
    return _assemble_swarm_snapshot(row, db)

@router.get("/capabilities")
def list_capabilities():
    """Phase 4.x — discovered capabilities in this environment (read-only).

    Each entry reports availability, the chosen/alternative providers, and the
    recommended action (execute / request_acquisition / replan). Nothing is spawned
    or installed; discovery only inspects PATH and importable libraries.
    """
    from backend.execution.capabilities import capability_service
    return {"capabilities": capability_service.summary()}


@router.get("/capabilities/{name}")
def get_capability(name: str):
    """Discover a single capability by name (read-only)."""
    from backend.execution.capabilities import capability_service
    cap = capability_service.discover(name)
    return cap.to_dict()


# ----------------------------------------------------
# EVIDENCE, FINDINGS, REPORTS & KNOWLEDGE
# ----------------------------------------------------

class EvidenceCreateRequest(BaseModel):
    challenge_id: str
    agent: str
    evidence_type: str
    source: str
    content: str
    confidence: float = 1.0

class FindingCreateRequest(BaseModel):
    challenge_id: str
    agent: str
    title: str
    description: str = ""
    vulnerability_class: str = "web"
    severity: str = "HIGH"


@router.get("/evidence")
def list_evidence(db: Session = Depends(get_db)):
    return db.query(EvidenceModel).order_by(EvidenceModel.created_at.desc()).all()

@router.post("/evidence")
def create_evidence(req: EvidenceCreateRequest, db: Session = Depends(get_db)):
    ev = EvidenceModel(
        challenge_id=req.challenge_id,
        agent=req.agent,
        evidence_type=req.evidence_type,
        source=req.source,
        content=req.content,
        confidence=req.confidence
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev

@router.get("/findings")
def list_findings(db: Session = Depends(get_db)):
    return db.query(FindingModel).order_by(FindingModel.created_at.desc()).all()

@router.post("/findings")
def create_finding(req: FindingCreateRequest, db: Session = Depends(get_db)):
    finding = FindingModel(
        challenge_id=req.challenge_id,
        agent=req.agent,
        title=req.title,
        description=req.description,
        vulnerability_class=req.vulnerability_class,
        verified=True,
        confidence=0.9
    )
    db.add(finding)
    db.commit()
    db.refresh(finding)
    return finding

@router.get("/reports")
def list_reports(db: Session = Depends(get_db)):
    return db.query(ReportModel).order_by(ReportModel.created_at.desc()).all()


# ----------------------------------------------------
# SYSTEM SETTINGS
# ----------------------------------------------------

class UpdateSystemSettingsRequest(BaseModel):
    execution_mode: str = "CTF_OFFENSIVE_CONTROLLED"
    auto_approve_privileged: bool = False
    command_timeout_seconds: int = 300
    daily_budget_usd: float = 5.00
    session_budget_usd: float = 2.00
    paid_model_allowed: bool = True
    default_strategy: str = "EXPLOIT_FIRST"


@router.get("/system/requirements")
def check_system_requirements():
    return environment_detector.perform_requirements_audit()

@router.get("/system/settings")
def get_system_settings():
    from backend.config import settings
    return {
        "execution_mode": getattr(settings, "EXECUTION_MODE", "CTF_OFFENSIVE_CONTROLLED"),
        "auto_approve_privileged": getattr(settings, "AUTO_APPROVE_PRIVILEGED", False),
        "command_timeout_seconds": getattr(settings, "COMMAND_TIMEOUT_SECONDS", 300),
        "daily_budget_usd": settings.DAILY_BUDGET_USD,
        "session_budget_usd": settings.SESSION_BUDGET_USD,
        "paid_model_allowed": settings.PAID_MODEL_ALLOWED,
        "default_strategy": getattr(settings, "DEFAULT_STRATEGY", "EXPLOIT_FIRST")
    }

@router.post("/system/settings")
def update_system_settings(req: UpdateSystemSettingsRequest):
    from backend.config import settings
    settings.PAID_MODEL_ALLOWED = req.paid_model_allowed
    settings.DAILY_BUDGET_USD = req.daily_budget_usd
    settings.SESSION_BUDGET_USD = req.session_budget_usd
    setattr(settings, "EXECUTION_MODE", req.execution_mode)
    setattr(settings, "AUTO_APPROVE_PRIVILEGED", req.auto_approve_privileged)
    setattr(settings, "COMMAND_TIMEOUT_SECONDS", req.command_timeout_seconds)
    setattr(settings, "DEFAULT_STRATEGY", req.default_strategy)
    return {
        "status": "SUCCESS",
        "message": "System Settings updated successfully across backend framework.",
        "settings": get_system_settings()
    }


# ----------------------------------------------------
# PACKAGE INSTALL APPROVAL SYSTEM
# ----------------------------------------------------

class PackageInstallRequest(BaseModel):
    request_id: str
    package_name: str
    challenge_id: Optional[str] = None

# In-memory registry of pending install requests from the orchestrator
pending_install_requests: dict = {}

@router.post("/package/install")
async def install_package(req: PackageInstallRequest):
    """User-approved pip install for a missing solver dependency."""
    import sys
    import logging
    logger = logging.getLogger("forge.package_installer")

    package_name = req.package_name.strip()
    # Basic safety check: only allow simple package names
    if not all(c.isalnum() or c in '-_.[]=<>!' for c in package_name):
        raise HTTPException(status_code=400, detail=f"Invalid package name: {package_name}")

    logger.info(f"User approved pip install: {package_name}")
    await ws_manager.broadcast({
        "event": "PACKAGE_INSTALL_STARTED",
        "request_id": req.request_id,
        "package_name": package_name,
        "challenge_id": req.challenge_id
    })

    try:
        result = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", package_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(result.communicate(), timeout=120)
        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace")
        success = result.returncode == 0

        await ws_manager.broadcast({
            "event": "PACKAGE_INSTALL_RESULT",
            "request_id": req.request_id,
            "package_name": package_name,
            "challenge_id": req.challenge_id,
            "success": success,
            "output": stdout_text[:2000] if success else stderr_text[:2000]
        })

        # Remove from pending queue
        pending_install_requests.pop(req.request_id, None)

        # Signal orchestrator to resume if it was waiting
        from backend.agents.orchestrator_loop import orchestrator_loop
        orchestrator_loop.resolve_install_request(req.request_id, success)

        return {
            "status": "SUCCESS" if success else "FAILED",
            "package_name": package_name,
            "output": stdout_text[:2000] if success else stderr_text[:2000]
        }
    except asyncio.TimeoutError:
        await ws_manager.broadcast({
            "event": "PACKAGE_INSTALL_RESULT",
            "request_id": req.request_id,
            "package_name": package_name,
            "success": False,
            "output": "Installation timed out after 120 seconds."
        })
        return {"status": "TIMEOUT", "package_name": package_name}
    except Exception as e:
        return {"status": "ERROR", "package_name": package_name, "error": str(e)}

class PackageSkipRequest(BaseModel):
    request_id: str
    challenge_id: Optional[str] = None

@router.post("/package/skip")
async def skip_package_install(req: PackageSkipRequest):
    """User skipped or rejected package installation."""
    from backend.agents.orchestrator_loop import orchestrator_loop
    pending_install_requests.pop(req.request_id, None)
    orchestrator_loop.resolve_install_request(req.request_id, False)
    return {"status": "SKIPPED", "request_id": req.request_id}


# ----------------------------------------------------
# DIRECTORY BROWSER & CREATOR API
# ----------------------------------------------------

class CreateDirRequest(BaseModel):
    parent_path: str
    dir_name: str

@router.get("/system/browse-dir")
def browse_directory(path: Optional[str] = None):
    if not path or not path.strip():
        current_path = os.getcwd()
    else:
        current_path = os.path.abspath(path.strip())

    if not os.path.exists(current_path):
        current_path = os.getcwd()

    parent_path = os.path.dirname(current_path)

    drives = []
    if os.name == 'nt':
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append(drive)
    else:
        drives = ["/"]

    directories = []
    try:
        with os.scandir(current_path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False) and not entry.name.startswith('.'):
                        directories.append(entry.name)
                except OSError:
                    continue
    except OSError:
        pass

    directories.sort()

    return {
        "current_path": current_path,
        "parent_path": parent_path,
        "drives": drives,
        "directories": directories
    }

@router.post("/system/create-dir")
def create_directory(req: CreateDirRequest):
    target_path = os.path.abspath(os.path.join(req.parent_path, req.dir_name.strip()))
    try:
        os.makedirs(target_path, exist_ok=True)
        return {"status": "SUCCESS", "created_path": target_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create directory: {str(e)}")

@router.post("/system/select-folder-dialog")
async def open_native_folder_dialog():
    selected_path = ""
    def _open_tkinter():
        nonlocal selected_path
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            selected = filedialog.askdirectory(title="FORGE CTF — Select Challenge Working Directory")
            root.destroy()
            if selected:
                selected_path = os.path.abspath(selected)
        except Exception:
            pass

    await asyncio.to_thread(_open_tkinter)
    if selected_path:
        return {"status": "SUCCESS", "selected_path": selected_path}
    return {"status": "CANCELLED", "selected_path": ""}
