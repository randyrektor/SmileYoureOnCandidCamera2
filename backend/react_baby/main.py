"""FastAPI server for the React Baby UI."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import paths
from .events import manager
from .face_analyzer import ensure_model
from .jobs import JobManager
from .models import DetectRoiRequest, JobRequest
from .roi import detect_roi
from .video_io import encode_jpeg, find_video_files, read_frame_at, resize_frame, video_info

load_dotenv()

# Quiet MediaPipe's native logging in this process too (workers set their own).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MAX_PREVIEW_DIMENSION = 1280

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class WebSocketLogHandler(logging.Handler):
    """Mirror this package's log lines to the UI. Safe to call from any thread."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.loop: asyncio.AbstractEventLoop | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.loop is None or self.loop.is_closed():
            return
        message = {
            "type": "log",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "message": record.getMessage(),
        }
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(manager.broadcast(message)))


ws_log_handler = WebSocketLogHandler()
logging.getLogger("react_baby").addHandler(ws_log_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    ws_log_handler.loop = loop
    video_dir, output_dir = paths.ensure_directories()
    logger.info("Video directory: %s", video_dir)
    logger.info("Output directory: %s", output_dir)

    try:
        await loop.run_in_executor(None, ensure_model)
    except Exception as exc:
        logger.error("Failed to prefetch face landmarker model: %s", exc)

    app.state.jobs = JobManager(manager.broadcast, paths.get_worker_count())
    logger.info("Ready (%d parallel workers)", app.state.jobs.max_workers)
    yield
    await app.state.jobs.shutdown()


app = FastAPI(title="React Baby", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _resolve(raw: str):
    try:
        return paths.resolve_video_path(raw)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Video file not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# ---------------------------------------------------------------- read-only


@app.get("/api/health")
async def health(request: Request):
    return {"status": "ok", "job_running": request.app.state.jobs.is_running}


@app.get("/api/config")
async def get_config(request: Request):
    video_dir, output_dir = paths.ensure_directories()
    return {
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "workers": request.app.state.jobs.max_workers,
    }


# Plain `def`: FastAPI runs these in a threadpool so OpenCV never blocks the loop.
@app.get("/api/videos")
def list_videos():
    video_dir, _ = paths.ensure_directories()
    videos = [info for p in find_video_files(video_dir) if (info := video_info(p))]
    return {"videos": videos}


@app.get("/api/preview-frame")
def preview_frame(video_path: str):
    path = _resolve(video_path)
    frame = read_frame_at(path, 0.5)
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not read frame from video")
    frame = resize_frame(frame, MAX_PREVIEW_DIMENSION)
    return Response(
        content=encode_jpeg(frame),
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post("/api/detect-roi")
async def detect_roi_endpoint(params: DetectRoiRequest):
    path = _resolve(params.video_path)
    loop = asyncio.get_running_loop()
    try:
        roi = await loop.run_in_executor(None, detect_roi, path, params.num_samples)
    except Exception as exc:
        logger.error("ROI detection failed for %s: %s", path.name, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return {"status": "success", "roi": roi}


# --------------------------------------------------------------------- jobs


@app.post("/api/jobs")
async def start_job(params: JobRequest, request: Request):
    jobs: JobManager = request.app.state.jobs
    if jobs.is_running:
        raise HTTPException(status_code=409, detail="A job is already running")
    output_dir = paths.get_output_dir()
    specs = []
    for raw in params.videoPaths:
        path = _resolve(raw)
        specs.append(
            {
                "video_path": str(path),
                "video_name": path.name,
                "output_dir": str(output_dir),
                "sensitivity": params.emotionSensitivity,
                "target_emotions": params.targetEmotions,
                "skip_frames": params.skipInterval,
                "roi": params.roiPosition.model_dump() if params.roiPosition else None,
            }
        )
    job = await jobs.start(specs)
    return {"status": "started", **job}


@app.post("/api/jobs/stop")
async def stop_job(request: Request):
    jobs: JobManager = request.app.state.jobs
    if not jobs.is_running:
        return {"status": "idle"}
    await jobs.stop()
    return {"status": "stopped"}


@app.get("/api/jobs/current")
async def current_job(request: Request):
    return request.app.state.jobs.status()


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive pings from the client
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


def run() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("REACT_BABY_HOST", "127.0.0.1"),
        port=int(os.environ.get("REACT_BABY_PORT", "8000")),
    )


if __name__ == "__main__":
    run()
