"""
Face analysis built on MediaPipe Face Landmarker (Tasks API).

A single model gives us:
  - Face detection (bounding box derived from landmarks)
  - 478 3D facial landmarks
  - 52 blendshapes (eyeBlink, mouthSmile, jawOpen, browInnerUp, ...)
  - Head pose (from the facial transformation matrix)

Blendshapes are the centerpiece. Each one is an independent, physically
meaningful signal (e.g. "left eye is 30% closed", "mouth is 80% smiling")
rather than the entangled softmax of a 7-class FER classifier. Combining
them gives us reliable expression detection plus an honest "is this frame
attractive" signal in one pass.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

# MediaPipe ships the model bundle on a public CDN. Float16 is fine for our
# use case and shaves a few MB. We cache to ~/.cache/react-baby.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_CACHE_DIR = Path.home() / ".cache" / "react-baby"
_MODEL_PATH = _MODEL_CACHE_DIR / "face_landmarker.task"


def _ensure_model() -> Path:
    """Download the Face Landmarker model bundle on first use."""
    if _MODEL_PATH.exists() and _MODEL_PATH.stat().st_size > 0:
        return _MODEL_PATH

    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading face landmarker model to {_MODEL_PATH} ...")
    tmp = _MODEL_PATH.with_suffix(".task.partial")
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        os.replace(tmp, _MODEL_PATH)
        logger.info("Face landmarker model downloaded.")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return _MODEL_PATH


@dataclass
class FaceAnalysis:
    """Everything we know about one face in one frame.

    Coordinates are in *original frame* pixel space (not ROI-local).
    """

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    blendshapes: Dict[str, float] = field(default_factory=dict)
    # Quality signals, all in 0..1 (higher is better unless noted)
    sharpness: float = 0.0          # Laplacian variance, normalized
    sharpness_raw: float = 0.0      # Raw Laplacian variance (for absolute floor)
    eyes_open: float = 0.0          # 1 - max(eyeBlinkL, eyeBlinkR)
    head_pose_centered: float = 0.0 # 1 - normalized(yaw + pitch)
    face_size: float = 0.0          # face area / frame area, normalized to ~1.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    @property
    def is_blinking(self) -> bool:
        """True if either eye is meaningfully closed."""
        return self.eyes_open < 0.5  # i.e. max blink > 0.5

    @property
    def is_sharp(self) -> bool:
        """Absolute sharpness floor. Below this, reject regardless of score."""
        return self.sharpness_raw >= 80.0


class FaceAnalyzer:
    """Stateful wrapper around MediaPipe Face Landmarker for a single face per frame."""

    def __init__(self, min_detection_confidence: float = 0.3):
        model_path = _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,                      # one person per video
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe Face Landmarker initialized")

    def close(self):
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------ public

    def analyze(
        self,
        frame_bgr: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> Optional[FaceAnalysis]:
        """Analyze the first face found inside the ROI.

        Args:
            frame_bgr: full-frame BGR image (OpenCV native).
            roi: (x1, y1, x2, y2) absolute pixel bounds. Set to whole frame to
                 analyze everywhere.

        Returns:
            FaceAnalysis with coords in *original frame space*, or None.
        """
        x1, y1, x2, y2 = roi
        # Clamp ROI to frame bounds defensively.
        h_full, w_full = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w_full - 1))
        y1 = max(0, min(y1, h_full - 1))
        x2 = max(x1 + 1, min(x2, w_full))
        y2 = max(y1 + 1, min(y2, h_full))

        roi_bgr = frame_bgr[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return None

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)

        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        roi_h, roi_w = roi_bgr.shape[:2]

        # Compute a tight bounding box from landmarks.
        xs = [lm.x * roi_w for lm in landmarks]
        ys = [lm.y * roi_h for lm in landmarks]
        bx1 = max(0, int(min(xs)))
        by1 = max(0, int(min(ys)))
        bx2 = min(roi_w, int(max(xs)))
        by2 = min(roi_h, int(max(ys)))
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        # Shift into original frame coordinates.
        bbox_frame = (bx1 + x1, by1 + y1, bw, bh)

        # Blendshapes -> dict for easy access.
        blendshapes: Dict[str, float] = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                blendshapes[cat.category_name] = float(cat.score)

        # Sharpness from a face-only crop (in original frame for accuracy).
        face_crop = frame_bgr[bbox_frame[1]:bbox_frame[1] + bbox_frame[3],
                              bbox_frame[0]:bbox_frame[0] + bbox_frame[2]]
        sharpness_raw, sharpness = self._compute_sharpness(face_crop)

        # Head pose from the 4x4 transformation matrix (if available).
        yaw_deg = pitch_deg = roll_deg = 0.0
        if result.facial_transformation_matrixes:
            mat = result.facial_transformation_matrixes[0]
            yaw_deg, pitch_deg, roll_deg = _matrix_to_euler(mat)

        # Composite quality signals.
        eye_blink_l = blendshapes.get("eyeBlinkLeft", 0.0)
        eye_blink_r = blendshapes.get("eyeBlinkRight", 0.0)
        eyes_open = 1.0 - max(eye_blink_l, eye_blink_r)

        # 30deg off-axis is "very off". Linearly map below that.
        yaw_pitch = math.sqrt(yaw_deg * yaw_deg + pitch_deg * pitch_deg)
        head_pose_centered = max(0.0, 1.0 - yaw_pitch / 30.0)

        face_area = bw * bh
        frame_area = w_full * h_full
        # Normalize: 5% of frame area -> 1.0 (typical talking-head reaction shot).
        face_size = min(1.0, (face_area / max(1, frame_area)) / 0.05)

        return FaceAnalysis(
            bbox=bbox_frame,
            blendshapes=blendshapes,
            sharpness=sharpness,
            sharpness_raw=sharpness_raw,
            eyes_open=eyes_open,
            head_pose_centered=head_pose_centered,
            face_size=face_size,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )

    # ---------------------------------------------------------------- internal

    @staticmethod
    def _compute_sharpness(face_crop: np.ndarray) -> Tuple[float, float]:
        """Return (raw_variance, normalized_score).

        Laplacian variance is the standard motion-blur indicator. We compute
        it at the *native* face resolution (not downsampled) because the whole
        point is to distinguish in-focus from blurry frames.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0, 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # 400 is a reasonable "very sharp" target for studio video. Saturate.
        normalized = min(1.0, variance / 400.0)
        return variance, normalized


def _matrix_to_euler(mat) -> Tuple[float, float, float]:
    """Convert a 4x4 MediaPipe facial transformation matrix to yaw/pitch/roll (deg).

    MediaPipe returns matrices where the rotation block follows a right-handed
    camera coordinate system. We extract intrinsic ZYX Euler angles, which is
    the most natural for head pose: roll about Z, yaw about Y, pitch about X.
    """
    m = np.asarray(mat).reshape(4, 4)
    r = m[:3, :3]

    # Standard rotation matrix decomposition.
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
