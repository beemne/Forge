import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.database.models import CheckpointModel, RunModel

logger = logging.getLogger("forge.checkpoints")

class CheckpointManager:
    """Manages snapshot persistence and state machine restoration for interrupted runs."""

    def create_checkpoint(
        self,
        db: Session,
        run_id: str,
        current_phase: str,
        current_agent: str,
        last_action: str,
        state_snapshot: Dict[str, Any]
    ) -> CheckpointModel:
        checkpoint = CheckpointModel(
            run_id=run_id,
            state_snapshot={
                "current_phase": current_phase,
                "current_agent": current_agent,
                "data": state_snapshot
            },
            last_successful_action=last_action,
            resumable=True
        )
        db.add(checkpoint)
        
        # Update current run status checkpoint pointer
        run = db.query(RunModel).filter(RunModel.id == run_id).first()
        if run:
            run.current_phase = current_phase
            run.current_agent = current_agent
        
        db.commit()
        db.refresh(checkpoint)
        logger.info(f"Created checkpoint {checkpoint.id} for run {run_id} at phase '{current_phase}'")
        return checkpoint

    def get_latest_checkpoint(self, db: Session, run_id: str) -> Optional[CheckpointModel]:
        return (
            db.query(CheckpointModel)
            .filter(CheckpointModel.run_id == run_id, CheckpointModel.resumable == True)
            .order_by(CheckpointModel.created_at.desc())
            .first()
        )

checkpoint_manager = CheckpointManager()
