"""Auto-detect a tight region of interest by sampling faces across a video."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from .face_analyzer import FaceAnalyzer
from .video_io import open_video

logger = logging.getLogger(__name__)

DEFAULT_ROI = {"top": 20.0, "bottom": 65.0, "left": 25.0, "right": 75.0}


def detect_roi(video_path: Path, num_samples: int = 8) -> dict:
    """Sample frames across the middle 80% of the video and box the faces.

    Returns percentages: {"top", "bottom", "left", "right"}.
    """
    cap = open_video(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames < 1 or frame_width < 1 or frame_height < 1:
            raise ValueError("Video has no frames")

        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)
        sample_interval = max(1, (end_frame - start_frame) // num_samples)
        positions = [start_frame + i * sample_interval for i in range(num_samples)]
        full_frame = (0, 0, frame_width, frame_height)

        face_boxes = []
        with FaceAnalyzer(running_mode="image") as analyzer:
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if not ok:
                    continue
                analysis = analyzer.analyze(frame, full_frame)
                if analysis is not None:
                    face_boxes.append(analysis.bbox)
    finally:
        cap.release()

    if not face_boxes:
        logger.warning("No faces in sampled frames of %s; using default ROI", video_path.name)
        return dict(DEFAULT_ROI)

    min_x = min(b[0] for b in face_boxes)
    min_y = min(b[1] for b in face_boxes)
    max_x = max(b[0] + b[2] for b in face_boxes)
    max_y = max(b[1] + b[3] for b in face_boxes)

    # 30% padding on each side for context.
    pad_x = (max_x - min_x) * 0.3
    pad_y = (max_y - min_y) * 0.3
    left = max(0.0, min_x - pad_x) / frame_width * 100
    right = min(frame_width, max_x + pad_x) / frame_width * 100
    top = max(0.0, min_y - pad_y) / frame_height * 100
    bottom = min(frame_height, max_y + pad_y) / frame_height * 100

    # Never smaller than 20% of the frame in either axis.
    if right - left < 20:
        cx = (left + right) / 2
        left, right = max(0.0, cx - 10), min(100.0, cx + 10)
    if bottom - top < 20:
        cy = (top + bottom) / 2
        top, bottom = max(0.0, cy - 10), min(100.0, cy + 10)

    roi = {
        "top": round(top, 1),
        "bottom": round(bottom, 1),
        "left": round(left, 1),
        "right": round(right, 1),
    }
    logger.info("ROI for %s from %d samples: %s", video_path.name, len(face_boxes), roi)
    return roi
