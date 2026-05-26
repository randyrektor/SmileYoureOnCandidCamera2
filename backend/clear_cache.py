#!/usr/bin/env python3
"""Clear the local face landmarker model cache.

Run this if the bundled MediaPipe model gets corrupted or you want to force a
fresh download from Google's CDN.
"""

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

    for pycache in [Path("__pycache__"), Path("src/__pycache__"), Path(".pytest_cache")]:
        if pycache.exists():
            shutil.rmtree(pycache)
            print(f"Cleared {pycache}")

    print("Done. Next run will re-download the face landmarker model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
