import numpy as np

from react_baby.emotion_detector import (
    EMOTIONS,
    FrameCandidate,
    ProcessingConfig,
    ReactionEvent,
    SharpnessBaseline,
    score_emotions,
)
from react_baby.face_analyzer import FaceAnalysis, matrix_to_euler
from react_baby.models import RoiPosition


def test_all_zero_blendshapes_is_neutral():
    scores = score_emotions({"mouthSmileLeft": 0.0})
    assert set(scores) == set(EMOTIONS)
    assert scores["neutral"] == 1.0
    assert all(scores[e] == 0.0 for e in EMOTIONS if e != "neutral")


def test_empty_blendshapes_scores_zero_everywhere():
    assert all(v == 0.0 for v in score_emotions({}).values())


def test_big_smile_is_happy_and_vetoes_negatives():
    scores = score_emotions(
        {
            "mouthSmileLeft": 0.9,
            "mouthSmileRight": 0.9,
            "cheekSquintLeft": 0.5,
            "cheekSquintRight": 0.5,
            "browDownLeft": 0.9,
            "browDownRight": 0.9,
            "noseSneerLeft": 0.6,
            "noseSneerRight": 0.6,
        }
    )
    assert scores["happy"] >= 0.9
    for negative in ("angry", "sad", "disgust", "fear"):
        assert scores[negative] == 0.0
    assert scores["neutral"] < 0.3


def test_surprise_recipe():
    scores = score_emotions(
        {
            "browInnerUp": 0.8,
            "browOuterUpLeft": 0.7,
            "browOuterUpRight": 0.7,
            "eyeWideLeft": 0.8,
            "eyeWideRight": 0.8,
            "jawOpen": 0.6,
        }
    )
    assert scores["surprise"] == 1.0
    assert scores["happy"] < 0.3


def test_resting_brow_alone_is_not_angry():
    scores = score_emotions({"browDownLeft": 0.9, "browDownRight": 0.9})
    assert scores["angry"] == 0.0


def test_brow_plus_mouth_tension_is_angry():
    scores = score_emotions(
        {
            "browDownLeft": 0.9,
            "browDownRight": 0.9,
            "mouthPressLeft": 0.6,
            "mouthPressRight": 0.6,
            "mouthFrownLeft": 0.5,
            "mouthFrownRight": 0.5,
        }
    )
    assert scores["angry"] > 0.5


def test_sensitivity_maps_to_threshold():
    roi = RoiPosition(top=0, bottom=100, left=0, right=100)
    assert ProcessingConfig(1, roi).emotion_threshold == 0.5
    assert ProcessingConfig(5, roi).emotion_threshold == 0.9
    assert ProcessingConfig(3, roi, skip_frames=99).skip_frames == 10
    assert "neutral" not in ProcessingConfig(3, roi).target_emotions


def _candidate(frame_number: int, happy: float) -> FrameCandidate:
    analysis = FaceAnalysis(
        bbox=(0, 0, 10, 10), eyes_open=1.0, sharpness=1.0, head_pose_centered=1.0, face_size=1.0
    )
    return FrameCandidate(frame_number, analysis, {"happy": happy}, frame=np.zeros((2, 2, 3)))


def test_reservoir_keeps_best_and_enforces_spread():
    weights = {"emotion": 1.0, "eyes_open": 0, "sharpness": 0, "pose": 0, "size": 0}
    ev = ReactionEvent("happy", weights, reservoir_size=3)
    for n, happy in [(0, 0.5), (3, 0.9), (6, 0.95), (9, 0.6), (12, 0.7), (60, 0.85)]:
        ev.add(_candidate(n, happy))
    assert ev.count == 6
    assert ev.last_seen == 60
    picks = ev.picks(max_picks=2, min_gap_frames=15)
    picked_frames = [c.frame_number for _, c in picks]
    # Best is frame 6 (0.95). Frame 3 (0.9) is too close, so frame 60 wins slot two.
    assert picked_frames == [6, 60]


def test_identity_matrix_has_zero_pose():
    yaw, pitch, roll = matrix_to_euler(np.eye(4))
    assert (round(yaw), round(pitch), round(roll)) == (0, 0, 0)


def test_sharpness_baseline_is_relative_to_video():
    base = SharpnessBaseline(window=50, warmup=5)
    assert base.is_sharp(40, ratio=0.6, absolute_floor=15)  # warmup: accept
    assert not base.is_sharp(5, ratio=0.6, absolute_floor=15)  # but never smeared frames
    for v in [50, 60, 55, 58, 62, 57, 59]:
        base.add(v)
    assert base.median == 58
    assert base.is_sharp(40, ratio=0.6, absolute_floor=15)  # 40 >= 0.6*58
    assert not base.is_sharp(30, ratio=0.6, absolute_floor=15)  # motion blur dip
    assert base.normalized(62) == 1.0
    assert 0.6 < base.normalized(40) < 0.7
