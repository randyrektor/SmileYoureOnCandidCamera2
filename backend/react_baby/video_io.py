"""Thin OpenCV helpers: discovery, hardware-accelerated decode, JPEG encoding."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def find_video_files(directory: Path) -> list[Path]:
    """All supported videos in `directory`, case-insensitive, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def open_video(path: Path) -> cv2.VideoCapture:
    """Open a video through FFmpeg and ask for hardware decoding.

    On macOS this lands on VideoToolbox. OpenCV silently falls back to
    software decoding when the codec isn't supported, so this is always safe.
    """
    cap = cv2.VideoCapture(
        str(path),
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
    )
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    return cap


def video_info(path: Path) -> dict | None:
    cap = open_video(path)
    try:
        if not cap.isOpened():
            logger.warning("Failed to open video: %s", path)
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(frame_count / fps) if fps > 0 else 0
        return {
            "name": path.name,
            "path": str(path),
            "duration": duration,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def read_frame_at(path: Path, fraction: float = 0.5) -> np.ndarray | None:
    """Decode a single frame at `fraction` of the way through the video."""
    cap = open_video(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * fraction))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def resize_frame(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(width, height)
    if longest <= max_dimension:
        return frame
    scale = max_dimension / longest
    return cv2.resize(
        frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
