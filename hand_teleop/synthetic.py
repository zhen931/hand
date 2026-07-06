"""Synthetic 21-point hand poses for testing the pipeline without a camera.

Produces MediaPipe-style world landmarks (metres) for a right hand whose four
fingers and thumb curl by a controllable amount. Used by the offscreen smoke
test to confirm that an opening/closing human hand drives an opening/closing
robot hand.
"""

from __future__ import annotations

import numpy as np

# Open right hand, wrist at origin, fingers along +y, across (index->pinky) +x,
# palm in the z=0 plane. Units: metres.
_MCP = {
    "index": np.array([-0.020, 0.090, 0.0]),
    "middle": np.array([0.000, 0.095, 0.0]),
    "ring": np.array([0.020, 0.090, 0.0]),
    "pinky": np.array([0.038, 0.083, 0.0]),
}
_SEG = {  # proximal, middle, distal segment lengths per finger
    "index": (0.040, 0.026, 0.022),
    "middle": (0.045, 0.030, 0.024),
    "ring": (0.040, 0.026, 0.022),
    "pinky": (0.032, 0.022, 0.018),
}
_FINGER_ORDER = ("index", "middle", "ring", "pinky")
_LM_SLOTS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
             "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _finger_points(mcp: np.ndarray, segs, curl: float) -> np.ndarray:
    """Three joint points + tip along an arc that bends into the palm (-z)."""
    forward = np.array([0.0, 1.0, 0.0])
    max_bend = np.deg2rad(85.0) * curl
    pts, p, ang = [], mcp.copy(), 0.0
    for i, L in enumerate(segs):
        ang += max_bend if i == 0 else max_bend * 0.9
        direction = _rot_x(ang) @ forward
        p = p + L * direction
        pts.append(p.copy())
    return np.array(pts)  # PIP, DIP, TIP


def hand_pose(curls) -> np.ndarray:
    """Return (21, 3) world landmarks.

    curls: scalar in [0,1] applied to all fingers, or a dict per finger name
    (keys: index, middle, ring, pinky, thumb).
    """
    if np.isscalar(curls):
        curls = {name: float(curls) for name in ("index", "middle", "ring", "pinky", "thumb")}
    lm = np.zeros((21, 3))

    lm[0] = np.array([0.0, 0.0, 0.0])  # wrist

    for name in _FINGER_ORDER:
        mcp = _MCP[name]
        i_mcp, i_pip, i_dip, i_tip = _LM_SLOTS[name]
        lm[i_mcp] = mcp
        pip, dip, tip = _finger_points(mcp, _SEG[name], curls[name])
        lm[i_pip], lm[i_dip], lm[i_tip] = pip, dip, tip

    # Thumb: sits off the index side, curls across the palm toward the fingers.
    t = curls["thumb"]
    lm[1] = np.array([-0.030, 0.020, 0.005])                       # CMC
    lm[2] = np.array([-0.045, 0.045, 0.010]) + t * np.array([0.030, 0.005, -0.010])  # MCP
    lm[3] = np.array([-0.052, 0.070, 0.014]) + t * np.array([0.055, 0.010, -0.020])  # IP
    lm[4] = np.array([-0.058, 0.090, 0.018]) + t * np.array([0.080, 0.015, -0.030])  # tip
    return lm


def _rot(axis: str, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def apply_transform(world: np.ndarray, R: np.ndarray | None = None,
                    t: np.ndarray | None = None) -> np.ndarray:
    """Rigidly transform 21 landmarks. Rotation is about the wrist landmark."""
    out = world.copy()
    wrist = world[0].copy()  # landmark 0 is the wrist
    if R is not None:
        out = (R @ (out - wrist).T).T + wrist
    if t is not None:
        out = out + t
    return out


def wrist_sweep(n: int = 120, curl: float = 0.2) -> np.ndarray:
    """Static finger pose while the whole hand tilts (x), rolls (y), yaws (z)."""
    base = hand_pose(curl)
    frames = []
    for k in range(n):
        p = k / n
        R = (_rot("x", 0.6 * np.sin(p * 2 * np.pi))
             @ _rot("y", 0.6 * np.sin(p * 4 * np.pi))
             @ _rot("z", 0.4 * np.sin(p * 2 * np.pi)))
        frames.append(apply_transform(base, R=R))
    return np.array(frames)


def sweep(n: int = 120) -> np.ndarray:
    """A test trajectory: open -> fist -> open, plus an isolated index wiggle."""
    frames = []
    for k in range(n):
        phase = k / n
        if phase < 0.5:
            c = np.sin(phase * 2 * np.pi) ** 2  # 0 -> 1 -> 0 over first half
            frames.append(hand_pose(c))
        else:
            # second half: only the index finger curls
            idx = np.sin((phase - 0.5) * 2 * np.pi) ** 2
            frames.append(hand_pose({"index": idx, "middle": 0, "ring": 0,
                                     "pinky": 0, "thumb": 0}))
    return np.array(frames)
