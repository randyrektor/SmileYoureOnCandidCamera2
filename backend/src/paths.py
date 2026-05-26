import os
from pathlib import Path

_DEFAULT_BASE = Path.home() / "Desktop" / "Smile Youre On Candid Camera"


def get_base_dir() -> Path:
    return Path(os.environ.get("REACT_BABY_BASE_DIR", _DEFAULT_BASE))


def get_video_dir() -> Path:
    env = os.environ.get("REACT_BABY_VIDEO_DIR")
    if env:
        return Path(env)
    return get_base_dir() / "1 VIDEO"


def get_output_dir() -> Path:
    env = os.environ.get("REACT_BABY_OUTPUT_DIR")
    if env:
        return Path(env)
    return get_base_dir() / "2 EMOTIONS"
