"""
Thumbnail-quality emotion frame extraction.

Pipeline:

  1. Decode the video once (hardware accelerated where possible). Skipped
     frames are only *grabbed*, never decoded to pixels.
  2. Run FaceAnalyzer in VIDEO mode on every Nth frame -> bbox, blendshapes,
     quality signals.
  3. Map blendshapes -> emotion scores via deterministic recipes.
  4. Group contiguous high-emotion frames into REACTION EVENTS, so we save a
     couple of best frames per real reaction instead of a peak-confidence
     fluke from each fixed window.
  5. Within each event keep a small reservoir of the best frames (pixels and
     all) so the picked frames are the exact frames we scored. No second
     decode pass, no seeking.
  6. Save the top-K per event, at least `min_pick_spread_seconds` apart, to
     disk as <emotion>_<score>_<frame>.<ext>.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .face_analyzer import FaceAnalysis, FaceAnalyzer
from .models import RoiPosition
from .video_io import open_video

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], None]

# ------------------------------------------------------------------- config


@dataclass
class ProcessingConfig:
    emotion_sensitivity: int  # 1 (loose) .. 5 (strict)
    roi_position: RoiPosition
    target_emotions: list[str] | None = None
    skip_frames: int = 3  # analyze every Nth frame
    image_format: str = "png"
    png_compression: int = 6
    # Reaction events: contiguous run of analyzed frames where emotion >= threshold.
    min_event_frames: int = 2  # at least this many frames to count as an event
    max_event_gap_frames: int = 2  # allow brief dips this large (in analyzed frames)
    max_frames_per_event: int = 2  # save up to N best frames from each event
    min_pick_spread_seconds: float = 0.5  # picks within an event must be this far apart
    # Keep this many best candidates (with pixels) per open event. Bounded so a
    # long laugh at 4K doesn't eat memory; 3x the picks is plenty of slack for
    # the spread rule.
    reservoir_size: int = 6
    # Blur gate. Absolute Laplacian variance depends on the camera, codec and
    # face size, so the gate is *relative* to the video's own rolling median:
    # a frame must be at least `min_sharpness_ratio` x that median. The small
    # absolute floor only catches truly smeared frames.
    min_sharpness_ratio: float = 0.6
    min_absolute_sharpness: float = 15.0
    # Extreme head poses break MediaPipe blendshapes (head thrown back while
    # laughing reads as "sad"). Reject frames that aren't camera-facing.
    max_yaw_deg: float = 30.0
    max_pitch_deg: float = 25.0
    composite_weights: dict[str, float] = field(
        default_factory=lambda: {
            "emotion": 0.40,
            "eyes_open": 0.25,
            "sharpness": 0.20,
            "pose": 0.10,
            "size": 0.05,
        }
    )

    def __post_init__(self) -> None:
        self.skip_frames = max(1, min(int(self.skip_frames), 10))
        if not self.target_emotions:
            self.target_emotions = [e for e in EMOTION_RECIPES if e != "neutral"]

    @property
    def emotion_threshold(self) -> float:
        """Sensitivity 1..5 -> threshold 0.5..0.9."""
        return min(0.95, max(0.30, 0.4 + 0.1 * self.emotion_sensitivity))


# -------------------------------------------------------- emotion blendshape map

# Each emotion is (positive_features, negative_features), each a list of
# (blendshape_name, weight). score = clip(sum(pos) - sum(neg), 0, 1).
#
# Weights are calibrated so the *primary* feature alone can drive the score
# high (mouthSmile=1.0 alone produces happy~=1.0). Secondary features add on
# top and saturate via the clip, putting realistic reactions in 0.7-1.0.
#
# Left/Right pairs are averaged under '_xxxAvg' keys so asymmetric mouth
# twitches mid-syllable don't spike a score.
EMOTION_RECIPES: dict[str, tuple[list[tuple[str, float]], list[tuple[str, float]]]] = {
    # Open smile or laugh. Cheek squint favors genuine (Duchenne) smiles.
    "happy": (
        [("_mouthSmileAvg", 1.0), ("_cheekSquintAvg", 0.4), ("jawOpen", 0.3)],
        [("_eyeBlinkAvg", 0.8), ("mouthFrownLeft", 0.5), ("mouthFrownRight", 0.5)],
    ),
    # Eyes wide, brows up, jaw drop. The classic reaction-shot recipe.
    "surprise": (
        [
            ("browInnerUp", 0.8),
            ("_browOuterUpAvg", 0.6),
            ("_eyeWideAvg", 0.8),
            ("jawOpen", 0.6),
        ],
        [("_mouthSmileAvg", 0.3), ("_eyeBlinkAvg", 0.8)],
    ),
    # Anger needs *both* a furrowed brow and lower-face tension; enforced
    # multiplicatively by `_angry_gate`. A heavy resting brow alone is not anger.
    "angry": (
        [
            ("browDownLeft", 0.5),
            ("browDownRight", 0.5),
            ("mouthFrownLeft", 0.4),
            ("mouthFrownRight", 0.4),
            ("mouthPressLeft", 0.2),
            ("mouthPressRight", 0.2),
        ],
        [("_mouthSmileAvg", 1.0), ("jawOpen", 0.4), ("browInnerUp", 0.5)],
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
    # Neutral is the absence of everything else (max penalty, see score_emotions).
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

EMOTIONS: tuple[str, ...] = tuple(EMOTION_RECIPES)

# Emotions that must not co-occur with a real smile.
_SMILE_INCOMPATIBLE = frozenset({"angry", "sad", "disgust", "fear"})
_SMILE_VETO_THRESHOLD = 0.4


def _expand_aliases(blendshapes: dict[str, float]) -> dict[str, float]:
    """Add L/R averages under '_xxxAvg' keys."""

    def avg(a: str, b: str) -> float:
        return 0.5 * (blendshapes.get(a, 0.0) + blendshapes.get(b, 0.0))

    return {
        **blendshapes,
        "_mouthSmileAvg": avg("mouthSmileLeft", "mouthSmileRight"),
        "_eyeBlinkAvg": avg("eyeBlinkLeft", "eyeBlinkRight"),
        "_eyeWideAvg": avg("eyeWideLeft", "eyeWideRight"),
        "_browOuterUpAvg": avg("browOuterUpLeft", "browOuterUpRight"),
        "_cheekSquintAvg": avg("cheekSquintLeft", "cheekSquintRight"),
    }


def _angry_gate(expanded: dict[str, float]) -> float:
    """Require BOTH brow-down AND lower-face tension to call something angry.

    Brow needs >=0.6 and mouth >=0.4 for the gate to be fully open; below
    that the score scales down linearly. Either being zero -> zero.
    """
    brow = 0.5 * (expanded.get("browDownLeft", 0.0) + expanded.get("browDownRight", 0.0))
    mouth_tension = max(
        0.5 * (expanded.get("mouthFrownLeft", 0.0) + expanded.get("mouthFrownRight", 0.0)),
        0.5 * (expanded.get("mouthPressLeft", 0.0) + expanded.get("mouthPressRight", 0.0)),
    )
    return min(brow / 0.6, 1.0) * min(mouth_tension / 0.4, 1.0)


def score_emotions(blendshapes: dict[str, float]) -> dict[str, float]:
    """Convert blendshapes to per-emotion scores in [0, 1]."""
    if not blendshapes:
        return dict.fromkeys(EMOTION_RECIPES, 0.0)
    expanded = _expand_aliases(blendshapes)
    out: dict[str, float] = {}

    smile_veto = expanded["_mouthSmileAvg"] > _SMILE_VETO_THRESHOLD

    _, neutral_neg = EMOTION_RECIPES["neutral"]
    penalty = max((w * expanded.get(name, 0.0) for name, w in neutral_neg), default=0.0)
    out["neutral"] = max(0.0, min(1.0, 1.0 - penalty))

    for name, (pos, neg) in EMOTION_RECIPES.items():
        if name == "neutral":
            continue
        if smile_veto and name in _SMILE_INCOMPATIBLE:
            out[name] = 0.0
            continue
        score = sum(w * expanded.get(b, 0.0) for b, w in pos) - sum(
            w * expanded.get(b, 0.0) for b, w in neg
        )
        out[name] = max(0.0, min(1.0, score))

    if out["angry"] > 0.0:
        out["angry"] *= _angry_gate(expanded)
    return out


# -------------------------------------------------------------- data structures


class SharpnessBaseline:
    """Rolling per-video sharpness statistics so the blur gate adapts to the footage."""

    def __init__(self, window: int = 300, warmup: int = 20):
        self._values: deque[float] = deque(maxlen=window)
        self._warmup = warmup

    def add(self, raw: float) -> None:
        self._values.append(raw)

    @property
    def median(self) -> float:
        return statistics.median(self._values) if self._values else 0.0

    def is_sharp(self, raw: float, ratio: float, absolute_floor: float) -> bool:
        if raw < absolute_floor:
            return False
        if len(self._values) < self._warmup:
            return True
        return raw >= ratio * self.median

    def normalized(self, raw: float) -> float:
        """0..1 where 1.0 means 'as sharp as the sharpest recent frames'."""
        if len(self._values) < 2:
            return 1.0
        ordered = sorted(self._values)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        return min(1.0, raw / p95) if p95 > 0 else 0.0


@dataclass
class FrameCandidate:
    """One analyzed frame: its pixels, face analysis and emotion scores."""

    frame_number: int
    analysis: FaceAnalysis
    emotion_scores: dict[str, float]
    frame: np.ndarray | None = None

    def composite(self, emotion: str, weights: dict[str, float]) -> float:
        a = self.analysis
        return (
            weights["emotion"] * self.emotion_scores.get(emotion, 0.0)
            + weights["eyes_open"] * a.eyes_open
            + weights["sharpness"] * a.sharpness
            + weights["pose"] * a.head_pose_centered
            + weights["size"] * a.face_size
        )


class ReactionEvent:
    """A run of frames where one emotion stays above threshold.

    Keeps only the `reservoir_size` best candidates (by composite score), so
    memory stays bounded no matter how long the reaction lasts.
    """

    def __init__(self, emotion: str, weights: dict[str, float], reservoir_size: int):
        self.emotion = emotion
        self._weights = weights
        self._size = max(1, reservoir_size)
        self.count = 0  # total frames that joined this event
        self.last_seen = -1  # frame_number of the most recent member
        self._reservoir: list[tuple[float, FrameCandidate]] = []

    def add(self, cand: FrameCandidate) -> None:
        self.count += 1
        self.last_seen = cand.frame_number
        score = cand.composite(self.emotion, self._weights)
        if len(self._reservoir) < self._size:
            self._reservoir.append((score, cand))
            return
        worst_i = min(range(self._size), key=lambda i: self._reservoir[i][0])
        if score > self._reservoir[worst_i][0]:
            self._reservoir[worst_i] = (score, cand)

    def picks(self, max_picks: int, min_gap_frames: int) -> list[tuple[float, FrameCandidate]]:
        """Best candidates, greedily enforcing a minimum frame gap between them."""
        ranked = sorted(self._reservoir, key=lambda sc: sc[0], reverse=True)
        chosen: list[tuple[float, FrameCandidate]] = []
        for score, cand in ranked:
            if len(chosen) >= max_picks:
                break
            if all(abs(cand.frame_number - c.frame_number) >= min_gap_frames for _, c in chosen):
                chosen.append((score, cand))
        return chosen


# ------------------------------------------------------------- progress helper


class ProgressTracker:
    """Smoothed FPS / ETA, rate-limited to one message per `update_interval`."""

    def __init__(self, total_frames: int, update_interval: float = 1.0):
        self.total_frames = max(1, total_frames)
        self.update_interval = update_interval
        self.start_time = time.monotonic()
        self.last_time = self.start_time
        self.last_frame = 0
        self.smoothed_fps = 0.0
        self.alpha = 0.15
        self.last_eta: float | None = None

    def update(self, frame_count: int) -> dict | None:
        now = time.monotonic()
        dt = now - self.last_time
        if dt < self.update_interval:
            return None
        current_fps = (frame_count - self.last_frame) / dt
        self.smoothed_fps = (
            self.alpha * current_fps + (1 - self.alpha) * self.smoothed_fps
            if self.smoothed_fps > 0
            else current_fps
        )
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
            "progress": round(min(100.0, frame_count / self.total_frames * 100.0), 1),
            "fps": round(self.smoothed_fps, 1),
            "elapsed": round(elapsed),
            "eta": round(max(0.0, eta)),
        }


# --------------------------------------------------------------- main detector


class EmotionDetector:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def roi_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        rp = self.config.roi_position
        return (
            int(width * rp.left / 100),
            int(height * rp.top / 100),
            int(width * rp.right / 100),
            int(height * rp.bottom / 100),
        )

    def _write_frame(self, frame: np.ndarray, filepath: Path) -> None:
        fmt = self.config.image_format.lower()
        if fmt == "png":
            cv2.imwrite(
                str(filepath), frame, [cv2.IMWRITE_PNG_COMPRESSION, self.config.png_compression]
            )
        else:
            cv2.imwrite(str(filepath), frame)

    def extract_emotions_best_frames(
        self,
        video_path: Path,
        output_dir: Path,
        should_stop: Callable[[], bool] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> int:
        """Extract best emotion thumbnails from a video. Returns frames saved."""
        cfg = self.config
        emit = progress_callback or (lambda _msg: None)
        threshold = cfg.emotion_threshold
        step = cfg.skip_frames
        targets = set(cfg.target_emotions or ())

        cap = open_video(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Processing %s: %d frames @ %.1f fps, step=%d, threshold=%.2f, targets=%s",
            video_path.name,
            total_frames,
            fps,
            step,
            threshold,
            sorted(targets),
        )

        progress = ProgressTracker(total_frames)
        sharpness = SharpnessBaseline()
        active: dict[str, ReactionEvent] = {}
        gap_frames = cfg.max_event_gap_frames * step
        min_gap_frames = max(1, int(cfg.min_pick_spread_seconds * fps))
        saved = 0

        def finalize(emotion: str) -> None:
            nonlocal saved
            ev = active.pop(emotion)
            if ev.count < cfg.min_event_frames:
                return
            for score, cand in ev.picks(cfg.max_frames_per_event, min_gap_frames):
                saved += self._save(cand, emotion, score, output_dir, emit)

        roi: tuple[int, int, int, int] | None = None
        frame_index = 0
        try:
            with FaceAnalyzer(running_mode="video") as analyzer:
                while True:
                    if should_stop and should_stop():
                        logger.info("Processing stopped by user")
                        break
                    if not cap.grab():
                        break
                    if frame_index % step == 0:
                        ok, frame = cap.retrieve()
                        if ok:
                            if roi is None:
                                h, w = frame.shape[:2]
                                roi = self.roi_pixels(w, h)
                            cand = self._analyze(analyzer, frame, frame_index, roi, fps)
                            if cand is not None:
                                sharpness.add(cand.analysis.sharpness_raw)
                                cand.analysis.sharpness = sharpness.normalized(
                                    cand.analysis.sharpness_raw
                                )
                                self._feed_events(cand, targets, threshold, active, sharpness)
                        # Close out events that have gone quiet.
                        for emo in [
                            e for e, ev in active.items() if frame_index - ev.last_seen > gap_frames
                        ]:
                            finalize(emo)
                    if frame_index % (step * 6) == 0:
                        msg = progress.update(frame_index)
                        if msg:
                            emit(msg)
                    frame_index += 1
                for emo in list(active):
                    finalize(emo)
        finally:
            cap.release()

        logger.info("Done. Saved %d frame(s) from %s", saved, video_path.name)
        emit(
            {
                "type": "progress",
                "progress": 100.0,
                "fps": progress.smoothed_fps,
                "elapsed": round(time.monotonic() - progress.start_time),
                "eta": 0,
            }
        )
        return saved

    # ---------- helpers

    def _analyze(
        self,
        analyzer: FaceAnalyzer,
        frame: np.ndarray,
        frame_index: int,
        roi: tuple[int, int, int, int],
        fps: float,
    ) -> FrameCandidate | None:
        timestamp_ms = int(frame_index * 1000 / fps)
        analysis = analyzer.analyze(frame, roi, timestamp_ms)
        if analysis is None:
            return None
        return FrameCandidate(
            frame_number=frame_index,
            analysis=analysis,
            emotion_scores=score_emotions(analysis.blendshapes),
            frame=frame,
        )

    def _feed_events(
        self,
        cand: FrameCandidate,
        targets: set[str],
        threshold: float,
        active: dict[str, ReactionEvent],
        sharpness: SharpnessBaseline,
    ) -> None:
        cfg = self.config
        a = cand.analysis
        usable = (
            not a.is_blinking
            and sharpness.is_sharp(
                a.sharpness_raw, cfg.min_sharpness_ratio, cfg.min_absolute_sharpness
            )
            and abs(a.yaw_deg) <= cfg.max_yaw_deg
            and abs(a.pitch_deg) <= cfg.max_pitch_deg
        )
        if not usable:
            return
        for emo, score in cand.emotion_scores.items():
            if emo in targets and score >= threshold:
                ev = active.get(emo)
                if ev is None:
                    ev = active[emo] = ReactionEvent(emo, cfg.composite_weights, cfg.reservoir_size)
                ev.add(cand)

    def _save(
        self,
        cand: FrameCandidate,
        emotion: str,
        composite: float,
        output_dir: Path,
        emit: ProgressCallback,
    ) -> int:
        if cand.frame is None:
            return 0
        emo_score = cand.emotion_scores.get(emotion, 0.0)
        filepath = (
            output_dir
            / f"{emotion}_{emo_score:.2f}_{cand.frame_number:06d}.{self.config.image_format}"
        )
        self._write_frame(cand.frame, filepath)
        msg = (
            f"Saved {emotion}={emo_score:.2f} (composite={composite:.2f}, "
            f"sharp={cand.analysis.sharpness:.2f}, eyes_open={cand.analysis.eyes_open:.2f}) "
            f"at frame {cand.frame_number}"
        )
        logger.info(msg)
        emit({"type": "log", "message": msg})
        return 1
