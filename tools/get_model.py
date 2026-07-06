"""Download the MediaPipe hand landmark model (run once after cloning)."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hand_teleop import ASSETS_DIR

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/latest/hand_landmarker.task")
DEST = ASSETS_DIR / "models" / "hand_landmarker.task"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)")
        return
    print(f"downloading {URL}\n  -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"done, {DEST.stat().st_size} bytes")


if __name__ == "__main__":
    main()
