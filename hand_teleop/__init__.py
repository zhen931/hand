"""Markerless vision-based teleoperation of a simulated dexterous hand.

Pipeline: webcam -> MediaPipe Hands (21 keypoints) -> UDP -> retargeting -> MuJoCo.
The hand is selectable (see hands.py); ORCA (5-finger) is the default.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
RECORDINGS_DIR = REPO_ROOT / "recordings"
