import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.config import settings
from backend.auth import require_api_key
from backend.database.guard import authorize_production_database
from backend.database.session import init_db
from backend.api.routes import router as api_router
from backend.websocket.manager import ws_manager

log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "forge.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, encoding="utf-8")
    ]
)
logger = logging.getLogger("forge.main")

def _mark_stale_runs_interrupted():
    """On boot, mark runs/challenges left RUNNING by a previous session as INTERRUPTED.

    Swarms are in-memory only, so any RUNNING row at startup is a zombie from a
    crashed/stopped server. Without this sweep the Command Center keeps showing
    dead challenges as live operations (fake uptime, "RUNNING" forever).
    """
    from backend.database.session import SessionLocal
    from backend.database.models import RunModel, ChallengeModel
    db = SessionLocal()
    try:
        runs = db.query(RunModel).filter(RunModel.status == "RUNNING").all()
        challenges = db.query(ChallengeModel).filter(ChallengeModel.status == "RUNNING").all()
        for r in runs:
            r.status = "INTERRUPTED"
        for c in challenges:
            c.status = "INTERRUPTED"
        if runs or challenges:
            db.commit()
            logger.info(f"[StartupRecovery] Marked {len(runs)} stale run(s) and {len(challenges)} challenge(s) as INTERRUPTED.")
    except Exception as e:
        logger.warning(f"[StartupRecovery] Stale-run sweep failed: {e}")
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan (replaces the deprecated on_event startup/shutdown hooks).

    Startup: this is the production entrypoint. Both launchers -- this module's
    __main__ and launch_forge.py -- serve backend.main:app, so authorizing here
    covers them, and this is the only place the production database is authorized.
    A test process never enters this lifespan, so tests cannot open forge.db
    however they are launched. See backend/database/guard.py.

    Shutdown: release every FORGE-tracked interactive process so none survives a
    controlled shutdown (Phase 4.x hardening §7). Interactive sessions own
    long-lived OS processes; without this they would outlive the server and hold
    file locks on forge.db / logs. close_all() kills each tracked process tree,
    reaps it, and unregisters its PID -- idempotent and guarded so a cleanup error
    can never block shutdown.
    """
    authorize_production_database()
    logger.info("Initializing database tables...")
    init_db()
    _mark_stale_runs_interrupted()
    logger.info(f"{settings.PROJECT_NAME} initialized and ready.")

    yield

    try:
        from backend.execution.interactive import interactive_manager
        closed = await interactive_manager.close_all(reason="app_shutdown")
        if closed:
            logger.info(f"[Shutdown] Closed {closed} tracked interactive session(s).")
    except Exception as e:
        logger.warning(f"[Shutdown] Interactive session cleanup failed: {e}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous CTF Intelligence & Exploitation Framework",
    lifespan=lifespan,
)

# CORS is restricted to an explicit origin allowlist (FORGE_ALLOWED_ORIGINS).
# A wildcard here is unsafe: with allow_credentials=True Starlette reflects the
# caller's Origin header back, so any site the operator visits could drive this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.FORGE_ALLOWED_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single router-level gate: covers every current and future /api handler without
# touching routes.py. No-op while FORGE_API_KEY is unset (dev mode).
app.include_router(api_router, prefix="/api", dependencies=[Depends(require_api_key)])

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming client ping/messages
            await websocket.send_json({"status": "received", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# Single System Mount: Serve compiled React Web Portal frontend directly from FastAPI
frontend_dist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.exists(frontend_dist_dir):
    assets_dir = os.path.join(frontend_dist_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static_assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(frontend_dist_dir, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_frontend_spa(full_path: str):
        if full_path.startswith("api") or full_path.startswith("ws"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="API Route Not Found")
        file_path = os.path.join(frontend_dist_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
