"""Where videos come from and where exports go.

Everything is overridable through environment variables (see .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_BASE = Path.home() / "Desktop" / "Smile Youre On Candid Camera"


def get_base_dir() -> Path:
    return Path(os.environ.get("REACT_BABY_BASE_DIR", _DEFAULT_BASE)).expanduser()


def get_video_dir() -> Path:
    env = os.environ.get("REACT_BABY_VIDEO_DIR")
    return Path(env).expanduser() if env else get_base_dir() / "1 VIDEO"


def get_output_dir() -> Path:
    env = os.environ.get("REACT_BABY_OUTPUT_DIR")
    return Path(env).expanduser() if env else get_base_dir() / "2 EMOTIONS"


def get_worker_count() -> int:
    """How many videos to process at once in a batch job."""
    env = os.environ.get("REACT_BABY_WORKERS")
    if env:
        return max(1, int(env))
    # MediaPipe already multithreads inside each worker, so leave headroom.
    return max(1, min(6, (os.cpu_count() or 4) // 4))


def ensure_directories() -> tuple[Path, Path]:
    video_dir, output_dir = get_video_dir(), get_output_dir()
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, output_dir


def resolve_video_path(raw: str) -> Path:
    """Resolve a client-supplied path and require it to live inside the video dir.

    The API accepts absolute paths from the browser. Without this check any
    process that can reach the port could read frames from arbitrary files.
    """
    video_dir = get_video_dir().resolve()
    path = Path(raw).expanduser().resolve()
    if not path.is_relative_to(video_dir):
        raise ValueError(f"{raw!r} is outside the video directory {video_dir}")
    if not path.is_file():
        raise FileNotFoundError(raw)
    return path
