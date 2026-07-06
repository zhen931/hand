"""Markerless vision-based teleoperation of a simulated LEAP hand.

Pipeline: webcam -> MediaPipe Hands (21 keypoints) -> UDP -> retargeting -> MuJoCo.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
DEFAULT_SCENE = ASSETS_DIR / "leap_hand" / "scene_right.xml"
RECORDINGS_DIR = REPO_ROOT / "recordings"
