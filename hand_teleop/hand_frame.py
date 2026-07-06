"""Build a stable local coordinate frame from 21 MediaPipe landmarks and
extract wrist-relative fingertip vectors.

The frame is rebuilt every frame from the palm geometry so it is invariant to
where the hand sits in the camera image and how the wrist is globally rotated
(the spec's "wrist joint as the origin" requirement). Fingertip vectors are
expressed in this frame and normalized by a reference bone length, which gives
the hand-scale normalization the spec calls for.

MediaPipe landmark indices used here:
    0  wrist
    5  index MCP     9  middle MCP    13 ring MCP    17 pinky MCP
    4  thumb tip     8  index tip     12 middle tip  16 ring tip   20 pinky tip
"""

from __future__ import annotations

import numpy as np

WRIST = 0
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP = 4, 8, 12, 16

# Robot fingertip order used everywhere downstream: index, middle, ring, thumb.
# The LEAP hand has no pinky, so human pinky (20) is intentionally dropped.
FINGERTIP_LANDMARKS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, THUMB_TIP)
FINGER_NAMES = ("index", "middle", "ring", "thumb")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def hand_local_frame(world: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation whose columns are the local hand axes in world space.

    x: across the palm, from the index side toward the pinky side
    y: along the fingers, from wrist toward the knuckles
    z: palm normal (x cross y)
    """
    wrist = world[WRIST]
    across = _normalize(world[PINKY_MCP] - world[INDEX_MCP])
    knuckles = 0.5 * (world[INDEX_MCP] + world[PINKY_MCP])
    forward = _normalize(knuckles - wrist)
    normal = _normalize(np.cross(across, forward))
    # Re-orthogonalize so the frame is exactly orthonormal.
    across = _normalize(np.cross(forward, normal))
    return np.column_stack([across, forward, normal])


def hand_scale(world: np.ndarray) -> float:
    """Reference length for size normalization: wrist to middle knuckle."""
    return float(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST]))


def fingertip_vectors(world: np.ndarray, landmarks=FINGERTIP_LANDMARKS) -> np.ndarray:
    """(N, 3) scale-normalized fingertip vectors in the local hand frame.

    landmarks is the ordered list of MediaPipe fingertip indices to use, one per
    robot finger. Defaults to index, middle, ring, thumb (the LEAP layout);
    five-finger hands pass their own list including the pinky.
    """
    wrist = world[WRIST]
    R = hand_local_frame(world)
    scale = hand_scale(world)
    if scale < 1e-6:
        scale = 1.0
    out = np.zeros((len(landmarks), 3), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        out[i] = (R.T @ (world[lm] - wrist)) / scale
    return out
