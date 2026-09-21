"""Per-video job that runs inside a worker process.

Kept free of FastAPI/asyncio so it pickles cleanly into a ProcessPoolExecutor.
Progress flows back to the server through a multiprocessing Manager queue.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


def init_worker() -> None:
    """ProcessPoolExecutor initializer: runs before the task module is imported.

    Quiets MediaPipe's native logging (glog + TF Lite XNNPACK chatter and the
    harmless clearcut telemetry error spam). Must be set before the native
    library loads, hence here rather than at import time.
    """
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [worker %(process)d] %(levelname)s %(message)s",
    )


def run_video_job(spec: dict, queue, stop_event) -> dict:
    """Process one video. `spec` keys: video_path, output_dir, sensitivity,
    target_emotions, skip_frames, roi (dict or None -> auto-detect)."""
    from .emotion_detector import EmotionDetector, ProcessingConfig
    from .models import RoiPosition
    from .roi import detect_roi

    video_path = Path(spec["video_path"])
    name = video_path.name

    def emit(msg: dict) -> None:
        msg.setdefault("video", name)
        queue.put(msg)

    if stop_event.is_set():
        return {"video": name, "saved": 0, "stopped": True}

    try:
        roi = spec.get("roi")
        if roi is None:
            emit({"type": "log", "message": f"Detecting face area in {name}"})
            roi = detect_roi(video_path, 8)
            emit({"type": "roi", "roi": roi})

        config = ProcessingConfig(
            emotion_sensitivity=spec["sensitivity"],
            roi_position=RoiPosition(**roi),
            target_emotions=spec.get("target_emotions"),
            skip_frames=spec.get("skip_frames", 3),
        )
        output_dir = Path(spec["output_dir"]) / video_path.stem
        saved = EmotionDetector(config).extract_emotions_best_frames(
            video_path, output_dir, should_stop=stop_event.is_set, progress_callback=emit
        )
        return {"video": name, "saved": saved, "stopped": stop_event.is_set()}
    except Exception as exc:  # report, don't crash the pool
        logging.getLogger(__name__).exception("Job failed for %s", name)
        return {"video": name, "saved": 0, "error": str(exc)}
