"""Target identity management — target IP ≠ identity. A verified target can change
address without losing its accumulated profile."""

import os
from backend.utils.time import utcnow

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.models import TargetProfileModel
from backend.database.session import get_db
from backend.websocket.manager import ws_manager

router = APIRouter()


class UpdateTargetAddressRequest(BaseModel):
    new_address: str


# ----------------------------------------------------
# TARGET IDENTITY MANAGEMENT (TARGET IP ≠ IDENTITY)
# ----------------------------------------------------

@router.get("/targets")
def list_targets(db: Session = Depends(get_db)):
    targets = db.query(TargetProfileModel).all()
    out = []
    for t in targets:
        svcs = t.expected_services or []
        if not svcs:
            svcs = [{"port": 80, "proto": "tcp", "service": "HTTP", "version": "Target Server"}]
        techs = getattr(t, "technologies", None) or ["Linux", "HTTP"]
        addr_hist = getattr(t, "address_history", None) or [t.current_address]
        disc_method = getattr(t, "discovery_method", None) or "FORGE Auto Ingest"
        last_ver = t.last_verified_at.isoformat() if t.last_verified_at else None

        out.append({
            "id": t.id,
            "challenge_id": t.challenge_id,
            "current_address": t.current_address,
            "hostname": t.hostname or t.current_address,
            "expected_services": svcs,
            "technologies": techs,
            "address_history": addr_hist,
            "discovery_method": disc_method,
            "verification_status": t.verification_status,
            "last_verified_at": last_ver
        })
    return out


@router.get("/targets/{target_id}")
def get_target(target_id: str, db: Session = Depends(get_db)):
    target = db.query(TargetProfileModel).filter(TargetProfileModel.id == target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target identity not found")
    return target

@router.post("/targets/{target_id}/verify")
async def verify_target(target_id: str, db: Session = Depends(get_db)):
    target = db.query(TargetProfileModel).filter(TargetProfileModel.id == target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target identity not found")

    is_file = os.path.exists(target.current_address)
    target.verification_status = "verified_file" if is_file else "verified_network"
    target.last_verified_at = utcnow()
    db.commit()

    await ws_manager.broadcast({"event": "TARGET_VERIFIED", "target_id": target_id, "address": target.current_address, "status": target.verification_status})
    return target

@router.post("/targets/{target_id}/rediscover")
async def rediscover_target(target_id: str, db: Session = Depends(get_db)):
    target = db.query(TargetProfileModel).filter(TargetProfileModel.id == target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target identity not found")

    target.verification_status = "rediscovered"
    target.last_verified_at = utcnow()
    db.commit()

    await ws_manager.broadcast({"event": "TARGET_REDISCOVERED", "target_id": target_id, "address": target.current_address})
    return target

async def perform_target_rebind(target: TargetProfileModel, new_address: str, db: Session) -> TargetProfileModel:
    """Execute dynamic address re-binding on a target identity while preserving past evidence and lineage."""
    old_addr = target.current_address
    new_addr = new_address.strip()
    if not new_addr:
        raise HTTPException(status_code=400, detail="New address cannot be empty")

    existing_history = list(target.address_history or [])
    if old_addr and old_addr not in existing_history:
        existing_history.append(old_addr)
    if new_addr not in existing_history:
        existing_history.append(new_addr)
    elif existing_history and existing_history[-1] != new_addr:
        existing_history.remove(new_addr)
        existing_history.append(new_addr)

    target.current_address = new_addr
    target.address_history = list(existing_history)
    target.verification_status = "address_updated"
    target.last_verified_at = utcnow()
    db.commit()
    db.refresh(target)

    await ws_manager.broadcast({
        "event": "TARGET_ADDRESS_UPDATED",
        "target_id": target.id,
        "challenge_id": target.challenge_id,
        "old_address": old_addr,
        "new_address": new_addr,
        "address_history": target.address_history,
        "status": target.verification_status,
    })
    await ws_manager.broadcast({
        "event": "TARGET_CHANGED",
        "target_id": target.id,
        "challenge_id": target.challenge_id,
        "current_address": new_addr,
        "target": new_addr,
        "address_history": target.address_history,
    })
    return target


@router.put("/targets/{target_id}/address")
async def update_target_address(target_id: str, req: UpdateTargetAddressRequest, db: Session = Depends(get_db)):
    target = db.query(TargetProfileModel).filter(TargetProfileModel.id == target_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target identity not found")

    updated = await perform_target_rebind(target, req.new_address, db)
    svcs = updated.expected_services or [{"port": 80, "proto": "tcp", "service": "HTTP", "version": "Target Server"}]
    techs = updated.technologies or ["Linux", "HTTP"]
    return {
        "id": updated.id,
        "challenge_id": updated.challenge_id,
        "current_address": updated.current_address,
        "hostname": updated.hostname or updated.current_address,
        "expected_services": svcs,
        "technologies": techs,
        "address_history": updated.address_history,
        "discovery_method": updated.discovery_method or "FORGE Auto Ingest",
        "verification_status": updated.verification_status,
        "last_verified_at": updated.last_verified_at.isoformat() if updated.last_verified_at else None,
    }

