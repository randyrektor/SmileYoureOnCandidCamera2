#!/usr/bin/env python3
"""Clear the cached MediaPipe face landmarker model (forces a fresh download)."""

import shutil
import sys
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "react-baby"


def main() -> int:
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared {CACHE_DIR}")
    else:
        print(f"Nothing to clear at {CACHE_DIR}")
    for pycache in Path(__file__).parent.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    print("Done. Next run will re-download the face landmarker model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
