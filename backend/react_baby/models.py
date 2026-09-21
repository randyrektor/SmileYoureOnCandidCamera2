"""Pydantic request/response models shared by the API and the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RoiPosition(BaseModel):
    """Region of interest as percentages of the frame."""

    top: float = Field(ge=0, le=100)
    bottom: float = Field(ge=0, le=100)
    left: float = Field(ge=0, le=100)
    right: float = Field(ge=0, le=100)


class JobRequest(BaseModel):
    videoPaths: list[str] = Field(min_length=1)
    emotionSensitivity: int = Field(4, ge=1, le=5)
    targetEmotions: list[str] | None = None
    skipInterval: int = Field(3, ge=1, le=10)
    # Applied to every video in the job. Omit to auto-detect per video.
    roiPosition: RoiPosition | None = None


class DetectRoiRequest(BaseModel):
    video_path: str
    num_samples: int = Field(8, ge=1, le=32)
