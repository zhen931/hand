"""Download the MediaPipe models (run once after cloning).

Hand model is required; the pose model is only needed for the tracker's --arm
arm overlay.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hand_teleop import ASSETS_DIR

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
}


def main():
    (ASSETS_DIR / "models").mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        dest = ASSETS_DIR / "models" / name
        if dest.exists():
            print(f"already present: {name} ({dest.stat().st_size} bytes)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  done, {dest.stat().st_size} bytes")


if __name__ == "__main__":
    main()
