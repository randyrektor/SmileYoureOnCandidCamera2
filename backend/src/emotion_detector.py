"""
Thumbnail-quality emotion frame extraction.

Pipeline (replaces the old MediaPipe-detector + HuggingFace-ViT stack):

  1. For each sampled frame, run FaceAnalyzer once -> bbox, blendshapes, quality.
  2. Map blendshapes -> emotion scores via deterministic recipes
     (no ML softmax noise; each blendshape is its own physical signal).
  3. Group contiguous high-emotion frames into REACTION EVENTS, so we save
     one or two best frames per real reaction instead of a peak-confidence
     fluke from each ~25-frame window.
  4. Within each event, pick the top-K frames by a COMPOSITE quality score
     (emotion * eyes-open * sharpness * pose * size), and hard-reject any
     frame with closed/blinking eyes or absolute blur.
  5. Save those frames to disk, named <emotion>_<score>_<frame>.<ext>.

Class and method names (EmotionDetector, ProcessingConfig, RoiPosition,
extract_emotions_best_frames) are kept stable for API compatibility with
main.py and the frontend.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel

try:
    from .face_analyzer import FaceAnalysis, FaceAnalyzer
except ImportError:
    from face_analyzer import FaceAnalysis, FaceAnalyzer

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- types


class RoiPosition(BaseModel):
    top: float
    bottom: float
    left: float
    right: float


@dataclass
class ProcessingConfig:
    """Configuration for a processing run.

    Older fields (focus_threshold, glasses_*, emotion_input_size, etc.) are
    intentionally absent: their behavior is now implicit in the blendshape
    pipeline or unnecessary (e.g. no model input size to configure).
    """

    emotion_sensitivity: int
    roi_position: RoiPosition
    debug: bool = False
    target_emotions: Optional[List[str]] = None
    image_format: str = "png"
    image_quality: int = 6  # PNG compression level, or JPG quality
    # Frame stepping during the analysis pass.
    analysis_skip_frames: int = 5
    # Reaction events: contiguous run of frames where emotion >= threshold.
    min_event_frames: int = 2          # at least this many frames to count as an event
    max_event_gap_frames: int = 2      # allow brief dips this large within an event
    max_frames_per_event: int = 2      # save up to N best frames from each event
    # Min real-time gap (seconds) between two saved picks within one event.
    # Stops us from saving the same expression twice 100ms apart.
    min_pick_spread_seconds: float = 0.5
    # Quality gates / weights for the composite score.
    min_absolute_sharpness: float = 80.0   # Laplacian variance floor (motion blur reject)
    max_blink: float = 0.5                  # reject if either eye blink > this
    # Extreme head poses break MediaPipe blendshapes (e.g. head thrown back
    # while laughing reads as "sad" because mouth corners pull weird). Reject
    # frames where the face isn't reasonably camera-facing.
    max_yaw_deg: float = 30.0
    max_pitch_deg: float = 25.0
    composite_weights: Dict[str, float] = field(default_factory=lambda: {
        "emotion": 0.40,
        "eyes_open": 0.25,
        "sharpness": 0.20,
        "pose": 0.10,
        "size": 0.05,
    })


# -------------------------------------------------------- emotion blendshape map


# Each emotion is (positive_features, negative_features), where each feature
# is a (blendshape_name, weight) pair. Score:
#     score = clip(sum(w * blendshape, positives)
#                  - sum(w * blendshape, negatives),
#                  0, 1)
#
# Weights are calibrated so the *primary* feature alone can drive the score
# high (e.g. mouthSmile=1.0 alone produces happy~=1.0). Secondary features add
# on top and naturally saturate via the clip. This puts realistic reactions
# in the 0.7-1.0 range, matching the existing sensitivity->threshold mapping.
#
# For pairs (Left/Right), use the *average* of the two sides in the recipe so
# the score doesn't spike on asymmetric expressions (mid-syllable mouth twitch).
EMOTION_RECIPES: Dict[str, Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]] = {
    # Open, eyes-still-open smile or laugh. We weight cheek squint to favor
    # genuine ("Duchenne") smiles vs. polite mouth-only ones.
    "happy": (
        [
            ("_mouthSmileAvg", 1.0),
            ("_cheekSquintAvg", 0.4),
            ("jawOpen", 0.3),
        ],
        [
            ("_eyeBlinkAvg", 0.8),     # Eyes shut kills the shot
            ("mouthFrownLeft", 0.5),
            ("mouthFrownRight", 0.5),
        ],
    ),
    # Eyes wide, brows up, jaw drop. The classic reaction-shot recipe.
    "surprise": (
        [
            ("browInnerUp", 0.8),
            ("_browOuterUpAvg", 0.6),
            ("_eyeWideAvg", 0.8),
            ("jawOpen", 0.6),
        ],
        [
            ("_mouthSmileAvg", 0.3),    # Smiling = happy, not surprised
            ("_eyeBlinkAvg", 0.8),
        ],
    ),
    # Anger requires *both* a furrowed brow and lower-face tension. A heavy
    # resting brow on its own (common!) should NOT read as angry. This is
    # enforced multiplicatively by `_angry_gate` below, separate from the
    # additive recipe. We also penalize jawOpen (talking/laughing) and
    # browInnerUp (which signals surprise/sad, not anger).
    "angry": (
        [
            ("browDownLeft", 0.5),
            ("browDownRight", 0.5),
            ("mouthFrownLeft", 0.4),
            ("mouthFrownRight", 0.4),
            ("mouthPressLeft", 0.2),
            ("mouthPressRight", 0.2),
        ],
        [
            ("_mouthSmileAvg", 1.0),
            ("jawOpen", 0.4),
            ("browInnerUp", 0.5),
        ],
    ),
    "sad": (
        [
            ("mouthFrownLeft", 0.7),
            ("mouthFrownRight", 0.7),
            ("browInnerUp", 0.3),
            ("mouthLowerDownLeft", 0.2),
            ("mouthLowerDownRight", 0.2),
        ],
        [("_mouthSmileAvg", 0.8), ("jawOpen", 0.3)],
    ),
    "fear": (
        [
            ("browInnerUp", 0.6),
            ("_browOuterUpAvg", 0.5),
            ("_eyeWideAvg", 0.7),
            ("mouthStretchLeft", 0.3),
            ("mouthStretchRight", 0.3),
        ],
        [("_mouthSmileAvg", 0.4)],
    ),
    "disgust": (
        [
            ("noseSneerLeft", 0.7),
            ("noseSneerRight", 0.7),
            ("mouthUpperUpLeft", 0.3),
            ("mouthUpperUpRight", 0.3),
        ],
        [("_mouthSmileAvg", 0.5)],
    ),
    # "Neutral" is best expressed as the absence of everything else.
    "neutral": (
        [],
        [
            ("_mouthSmileAvg", 0.9),
            ("mouthFrownLeft", 0.6),
            ("mouthFrownRight", 0.6),
            ("jawOpen", 0.4),
            ("browInnerUp", 0.4),
            ("_browOuterUpAvg", 0.4),
            ("browDownLeft", 0.4),
            ("browDownRight", 0.4),
            ("_eyeWideAvg", 0.4),
            ("noseSneerLeft", 0.5),
            ("noseSneerRight", 0.5),
        ],
    ),
}


def _expand_aliases(blendshapes: Dict[str, float]) -> Dict[str, float]:
    """Add convenient L/R averages under '_xxxAvg' keys."""

    def avg(a: str, b: str) -> float:
        return 0.5 * (blendshapes.get(a, 0.0) + blendshapes.get(b, 0.0))

    extras = {
        "_mouthSmileAvg": avg("mouthSmileLeft", "mouthSmileRight"),
        "_eyeBlinkAvg": avg("eyeBlinkLeft", "eyeBlinkRight"),
        "_eyeWideAvg": avg("eyeWideLeft", "eyeWideRight"),
        "_browOuterUpAvg": avg("browOuterUpLeft", "browOuterUpRight"),
        "_cheekSquintAvg": avg("cheekSquintLeft", "cheekSquintRight"),
    }
    return {**blendshapes, **extras}


# Emotions that must not co-occur with a real smile. A smiling person is not
# angry, sad, scared, or disgusted — regardless of what their brow is doing.
_SMILE_INCOMPATIBLE = {"angry", "sad", "disgust", "fear"}
_SMILE_VETO_THRESHOLD = 0.4


def _angry_gate(expanded: Dict[str, float]) -> float:
    """Multiplicative gate that requires BOTH a brow-down AND lower-face
    tension to call something 'angry'. People with a heavy resting brow
    would otherwise score angry constantly; people who never tense their
    mouth aren't actually angry no matter what their brows do.
    """
    brow = 0.5 * (
        expanded.get("browDownLeft", 0.0) + expanded.get("browDownRight", 0.0)
    )
    mouth_tension = max(
        0.5 * (expanded.get("mouthFrownLeft", 0.0)
               + expanded.get("mouthFrownRight", 0.0)),
        0.5 * (expanded.get("mouthPressLeft", 0.0)
               + expanded.get("mouthPressRight", 0.0)),
    )
    # Soft floors: brow needs >=0.6 and mouth >=0.4 for the gate to be fully
    # open; below that, the score is scaled down linearly. Either being zero
    # → score is zero.
    return min(brow / 0.6, 1.0) * min(mouth_tension / 0.4, 1.0)


def score_emotions(blendshapes: Dict[str, float]) -> Dict[str, float]:
    """Convert blendshapes to per-emotion scores in [0, 1].

    Calibration notes:
      - Positive features are summed (not normalized by sum-of-weights).
        Weights are chosen so the primary feature alone can already reach
        a strong score (e.g. a 1.0 mouthSmile hits 1.0 happy on its own);
        secondary features push past saturation, which is then clipped.
      - Penalty features subtract directly.
      - Neutral uses *max* penalty (any single strong expression should
        tank it) rather than averaging dozens of weak penalties to nothing.
      - Negative emotions (angry/sad/disgust/fear) are hard-vetoed when a
        real smile is present.
      - 'angry' has an extra multiplicative gate requiring brow AND mouth
        tension together, since the additive recipe alone over-fires on
        people with a strong resting-brow.
    """
    expanded = _expand_aliases(blendshapes)
    out: Dict[str, float] = {}
    if not expanded:
        return {name: 0.0 for name in EMOTION_RECIPES}

    smile_strength = expanded.get("_mouthSmileAvg", 0.0)
    smile_veto = smile_strength > _SMILE_VETO_THRESHOLD

    if "neutral" in EMOTION_RECIPES:
        _, neg = EMOTION_RECIPES["neutral"]
        penalty = max(
            (w * expanded.get(name, 0.0) for name, w in neg),
            default=0.0,
        )
        out["neutral"] = max(0.0, min(1.0, 1.0 - penalty))

    for name, (pos, neg) in EMOTION_RECIPES.items():
        if name == "neutral":
            continue
        if smile_veto and name in _SMILE_INCOMPATIBLE:
            out[name] = 0.0
            continue
        pos_total = sum(w * expanded.get(bname, 0.0) for bname, w in pos)
        neg_total = sum(w * expanded.get(bname, 0.0) for bname, w in neg)
        score = pos_total - neg_total
        out[name] = max(0.0, min(1.0, score))

    if out.get("angry", 0.0) > 0.0:
        out["angry"] *= _angry_gate(expanded)

    return out


# -------------------------------------------------------------- data structures


@dataclass
class FrameCandidate:
    """One analyzed frame's worth of information."""
    frame_number: int
    analysis: FaceAnalysis
    emotion_scores: Dict[str, float]

    def composite(self, emotion: str, weights: Dict[str, float]) -> float:
        emo = self.emotion_scores.get(emotion, 0.0)
        a = self.analysis
        return (
            weights["emotion"] * emo
            + weights["eyes_open"] * a.eyes_open
            + weights["sharpness"] * a.sharpness
            + weights["pose"] * a.head_pose_centered
            + weights["size"] * a.face_size
        )


# ------------------------------------------------------------- progress helper


class _ProgressTracker:
    """Lightweight FPS / ETA tracker shared with the frontend over websocket."""

    def __init__(self, total_frames: int, update_interval: float = 1.0):
        self.total_frames = max(1, total_frames)
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_frame = 0
        self.smoothed_fps = 0.0
        self.alpha = 0.15
        self.last_eta = None

    def update(self, frame_count: int) -> Optional[dict]:
        now = time.time()
        if now - self.last_time < self.update_interval:
            return None

        dt = now - self.last_time
        df = frame_count - self.last_frame
        current_fps = df / dt if dt > 0 else 0.0
        self.smoothed_fps = (
            (self.alpha * current_fps) + ((1 - self.alpha) * self.smoothed_fps)
            if self.smoothed_fps > 0
            else current_fps
        )
        progress = min(100.0, (frame_count / self.total_frames) * 100.0)
        elapsed = now - self.start_time

        if self.smoothed_fps > 0:
            eta = (self.total_frames - frame_count) / self.smoothed_fps
            if self.last_eta is not None:
                eta = 0.3 * eta + 0.7 * self.last_eta
            self.last_eta = eta
        else:
            eta = 0.0

        self.last_time = now
        self.last_frame = frame_count
        return {
            "type": "progress",
            "progress": round(progress, 1),
            "fps": round(self.smoothed_fps, 1),
            "elapsed": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
            "eta": time.strftime("%H:%M:%S", time.gmtime(max(0, eta))),
        }


# --------------------------------------------------------------- main detector


class EmotionDetector:
    """Public class consumed by main.py. Method names kept for compatibility."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.analyzer = FaceAnalyzer(min_detection_confidence=0.3)
        self._cached_roi: Optional[Tuple[int, int, int, int]] = None
        self._cached_roi_dims = None

        if self.config.target_emotions is None:
            self.config.target_emotions = list(EMOTION_RECIPES.keys())

    # ---------- helpers

    def calculate_roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        rp = self.config.roi_position
        cache_key = (width, height, rp.top, rp.bottom, rp.left, rp.right)
        if self._cached_roi_dims == cache_key and self._cached_roi is not None:
            return self._cached_roi
        roi = (
            int(width * rp.left / 100),
            int(height * rp.top / 100),
            int(width * rp.right / 100),
            int(height * rp.bottom / 100),
        )
        self._cached_roi = roi
        self._cached_roi_dims = cache_key
        return roi

    def _sensitivity_to_threshold(self) -> float:
        # Keep the existing sensitivity contract (1..5 -> 0.5..0.9).
        # Blendshape-derived emotion scores are calibrated to feel comparable.
        return min(0.95, max(0.30, 0.4 + 0.1 * self.config.emotion_sensitivity))

    def _write_frame(self, frame: np.ndarray, filepath: Path):
        fmt = self.config.image_format.lower()
        if fmt == "png":
            cv2.imwrite(str(filepath), frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, self.config.image_quality])
        elif fmt == "jpg":
            cv2.imwrite(str(filepath), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
        elif fmt == "tiff":
            cv2.imwrite(str(filepath), frame,
                        [cv2.IMWRITE_TIFF_COMPRESSION, self.config.image_quality])
        elif fmt == "webp":
            cv2.imwrite(str(filepath), frame,
                        [cv2.IMWRITE_WEBP_QUALITY, self.config.image_quality])
        else:
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # ---------- one-frame analysis (used by ROI auto-detect and tests)

    def analyze_frame(self, frame: np.ndarray) -> Optional[FrameCandidate]:
        h, w = frame.shape[:2]
        roi = self.calculate_roi(w, h)
        analysis = self.analyzer.analyze(frame, roi)
        if analysis is None:
            return None
        scores = score_emotions(analysis.blendshapes)
        return FrameCandidate(frame_number=-1, analysis=analysis, emotion_scores=scores)

    # ---------- the main entry point (signature preserved for main.py)

    def extract_emotions_best_frames(
        self,
        video_path: Path,
        output_dir: Path,
        should_stop: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
        skip_interval: int = 3,
        emotion_threshold: float = 0.5,
    ) -> int:
        """Extract best emotion thumbnails from a video."""
        self._current_video_path = video_path

        # Sensitivity overrides the explicit threshold if both are present and
        # the explicit one looks like a default. We treat the caller-supplied
        # threshold as a *floor*; sensitivity tightens it further.
        sensitivity_threshold = self._sensitivity_to_threshold()
        effective_threshold = max(emotion_threshold, sensitivity_threshold)

        # Honor the caller-supplied skip if non-default; otherwise prefer the
        # tighter sampling needed by reaction-event grouping.
        analysis_step = max(1, min(skip_interval, 10))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_dir.mkdir(parents=True, exist_ok=True)

        targets = set(self.config.target_emotions or list(EMOTION_RECIPES.keys()))
        logger.info(
            f"Processing {video_path.name}: {total_frames} frames @ {fps:.1f} fps, "
            f"step={analysis_step}, threshold={effective_threshold:.2f}, "
            f"targets={sorted(targets)}"
        )

        progress = _ProgressTracker(total_frames=total_frames, update_interval=1.0)
        # Per-emotion running events: list of FrameCandidate, last-seen-frame.
        active_events: Dict[str, Dict] = {}
        saved_count = 0

        def _finalize_event(emotion: str):
            nonlocal saved_count
            ev = active_events.pop(emotion, None)
            if not ev or len(ev["frames"]) < self.config.min_event_frames:
                return
            saved_count += self._save_best_from_event(
                ev["frames"], emotion, output_dir, progress_callback, fps
            )

        try:
            frame_index = 0
            while True:
                if should_stop and should_stop():
                    logger.info("Processing stopped by user")
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % analysis_step == 0:
                    candidate = self._candidate_for_frame(frame, frame_index)
                    if candidate is not None:
                        a = candidate.analysis
                        camera_facing = (
                            abs(a.yaw_deg) <= self.config.max_yaw_deg
                            and abs(a.pitch_deg) <= self.config.max_pitch_deg
                        )
                        for emo, score in candidate.emotion_scores.items():
                            if emo not in targets:
                                continue
                            if (
                                score >= effective_threshold
                                and not a.is_blinking
                                and a.is_sharp
                                and camera_facing
                            ):
                                ev = active_events.get(emo)
                                if ev is None:
                                    active_events[emo] = {
                                        "frames": [candidate],
                                        "last_seen": frame_index,
                                    }
                                else:
                                    ev["frames"].append(candidate)
                                    ev["last_seen"] = frame_index

                    # Close out any events that have gone quiet.
                    gap_frames = self.config.max_event_gap_frames * analysis_step
                    for emo in list(active_events.keys()):
                        if frame_index - active_events[emo]["last_seen"] > gap_frames:
                            _finalize_event(emo)

                # Progress update
                if progress_callback and frame_index % (analysis_step * 6) == 0:
                    msg = progress.update(frame_index)
                    if msg:
                        progress_callback(msg)

                frame_index += 1

            # Final flush.
            for emo in list(active_events.keys()):
                _finalize_event(emo)
        finally:
            cap.release()
            self.analyzer.close()

        logger.info(f"Done. Saved {saved_count} best emotion frame(s) from {video_path.name}")
        if progress_callback:
            progress_callback({
                "type": "log",
                "timestamp": time.strftime("%H:%M:%S"),
                "message": (f"Summary: saved {saved_count} best emotion frame(s) "
                            f"from '{video_path.name}'."),
            })
            progress_callback({"type": "complete"})
        return saved_count

    # ---------- per-event best-frame selection

    def _candidate_for_frame(
        self, frame: np.ndarray, frame_index: int
    ) -> Optional[FrameCandidate]:
        h, w = frame.shape[:2]
        roi = self.calculate_roi(w, h)
        analysis = self.analyzer.analyze(frame, roi)
        if analysis is None:
            return None
        scores = score_emotions(analysis.blendshapes)
        return FrameCandidate(frame_number=frame_index, analysis=analysis,
                              emotion_scores=scores)

    def _save_best_from_event(
        self,
        frames: List[FrameCandidate],
        emotion: str,
        output_dir: Path,
        progress_callback: Optional[Callable[[dict], None]],
        fps: float,
    ) -> int:
        """Pick top-K frames from an event by composite score, then save them.

        Critically: we re-read each chosen frame from the video file so we
        save the *full original* frame (not a crop), since the user wants
        the whole shot for thumbnail use.
        """
        if not frames:
            return 0

        weights = self.config.composite_weights
        ranked = sorted(
            frames,
            key=lambda c: c.composite(emotion, weights),
            reverse=True,
        )
        # Enforce a minimum real-time gap between picks. Two frames 100ms
        # apart are visually identical for thumbnail use; only pick distinct
        # moments inside the same reaction event.
        min_gap_frames = max(1, int(self.config.min_pick_spread_seconds * fps))
        picks: List[FrameCandidate] = []
        for cand in ranked:
            if len(picks) >= self.config.max_frames_per_event:
                break
            if all(abs(cand.frame_number - p.frame_number) >= min_gap_frames
                   for p in picks):
                picks.append(cand)

        if not picks:
            return 0

        # Re-open the video fresh to grab each picked frame at its native
        # quality. `_current_video_path` is set by extract_emotions_best_frames.
        cap = cv2.VideoCapture(str(self._current_video_path))
        saved = 0
        try:
            for cand in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cand.frame_number)
                ret, frame = cap.read()
                if not ret:
                    continue
                score = cand.composite(emotion, weights)
                emo_score = cand.emotion_scores.get(emotion, 0.0)
                # Filename: <emotion>_<emotion_score>_<frame>.ext
                # (Composite score is logged in the message; on-disk score
                # stays as the emotion score for continuity with prior naming.)
                stem = f"{emotion}_{emo_score:.2f}_{cand.frame_number:06d}"
                filepath = output_dir / f"{stem}.{self.config.image_format}"
                self._write_frame(frame, filepath)
                saved += 1
                msg = (
                    f"Saved {emotion}={emo_score:.2f} (composite={score:.2f}, "
                    f"sharp={cand.analysis.sharpness:.2f}, "
                    f"eyes_open={cand.analysis.eyes_open:.2f}) "
                    f"at frame {cand.frame_number}"
                )
                logger.info(msg)
                if progress_callback:
                    progress_callback({
                        "type": "log",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "message": msg,
                    })
        finally:
            cap.release()
        return saved

