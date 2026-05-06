"""
BioSpark-Light FastAPI app — local desktop variant.

Trimmed surface area vs. the web version:
  * No auth (single local user, sentinel uid=0)
  * No inference / streaming / pre-trained model registry
  * Mounted routers: prep + training + figures + model_history
  * Serves the frontend (index.html + /css + /js + /assets) under /

The launcher (../launcher.py) imports `app` and runs it via uvicorn on
loopback only (127.0.0.1:8765).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from backend.config import FRONTEND_DIR, USER_DATA_DIR
from backend.database import async_session, init_db
from backend.routers import figures as figures_router
from backend.routers import model_history as model_history_router
from backend.routers import prep as prep_router
from backend.routers import training as training_router
from backend.services.license import license_service

VERSION = "0.2.0-light"

_log = logging.getLogger("biospark.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Self-check: count rows in training_runs so users see "warm-start chain has N entries"
    try:
        async with async_session() as session:
            row = await session.execute(text("SELECT COUNT(*) FROM training_runs"))
            run_count = row.scalar_one()
            _log.warning(
                "BioSpark-Light ready · data dir=%s · training_runs=%s",
                USER_DATA_DIR, run_count,
            )
    except Exception as exc:  # noqa: BLE001
        _log.warning("DB self-check failed (tables may be fresh): %s", exc)
    yield


app = FastAPI(
    title="BioSpark-Light",
    description="Local desktop biosignal lab — Data Prep · Training · My Models. No cloud, no auth.",
    version=VERSION,
    lifespan=lifespan,
)

# CORS is permissive because the only client is the local browser tab the
# launcher opens; nothing remote can reach 127.0.0.1:8765 anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": VERSION, "data_dir": str(USER_DATA_DIR)}


# ─── License endpoints ──────────────────────────────────────────────────────

@app.get("/api/license/status")
def license_status():
    return license_service.get_status()


class _ActivateRequest(BaseModel):
    key: str


@app.post("/api/license/activate")
def license_activate(req: _ActivateRequest):
    result = license_service.activate(req.key.strip())
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


app.include_router(prep_router.router, prefix="/api", tags=["DataPrep"])
app.include_router(training_router.router, prefix="/api", tags=["Training"])
app.include_router(figures_router.router, prefix="/api", tags=["Figures"])
app.include_router(model_history_router.router, prefix="/api", tags=["ModelHistory"])


# --- Frontend static assets ---
if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    async def serve_index():
        index = FRONTEND_DIR / "index.html"
        if not index.exists():
            return {"error": "frontend/index.html missing"}
        return FileResponse(str(index))
