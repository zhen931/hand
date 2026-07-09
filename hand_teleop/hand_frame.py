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


def orient_normal(across, forward, normal, thumb_vec):
    """Flip the normal so the thumb sits on a consistent (palmar) side.

    The four fingers are nearly coplanar, so the sign of the palm normal is
    otherwise ambiguous and can point either way depending on pose. The thumb is
    off that plane, so we use it to pin the normal direction. Flipping the normal
    also flips across to keep the frame right-handed; that across flip cancels out
    in the retargeting mapping, but the normal fix is what makes curl consistent
    between the human hand and any robot hand.
    """
    if np.dot(thumb_vec, normal) > 0:
        normal = -normal
        across = _normalize(np.cross(forward, normal))
    return across, forward, normal


def hand_local_frame(world: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation whose columns are the local hand axes in world space.

    x: across the palm (index toward pinky), y: along the fingers, z: palm normal.
    Built only from the wrist and the index/pinky knuckles, all stable landmarks,
    so it does NOT depend on the thumb: moving the thumb must not move the wrist.
    The normal sign is deterministic (across x forward) and consistent frame to
    frame; any absolute palm-facing offset is handled once by the wrist flip.
    """
    wrist = world[WRIST]
    across = _normalize(world[PINKY_MCP] - world[INDEX_MCP])
    knuckles = 0.5 * (world[INDEX_MCP] + world[PINKY_MCP])
    forward = _normalize(knuckles - wrist)
    normal = _normalize(np.cross(across, forward))
    across = _normalize(np.cross(forward, normal))
    return np.column_stack([across, forward, normal])


def hand_scale(world: np.ndarray) -> float:
    """Reference length for size normalization: wrist to middle knuckle."""
    return float(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST]))


def finger_bend(world: np.ndarray, chain, skip_base: bool = False) -> float:
    """Total flexion of a finger: the sum of bend angles (radians) along its
    joints, wrist included as the metacarpal reference.

    chain is the finger's landmark indices from base to tip, e.g. index is
    (5, 6, 7, 8). A straight finger returns ~0; a curled one returns a few
    radians. Intrinsic to the finger, so no calibration and no dependence on hand
    size or wrist orientation.

    skip_base drops the first joint's angle. For the thumb that first angle is the
    carpometacarpal position (how the thumb is splayed), not flexion, and it would
    otherwise make the thumb curl as it changes position.
    """
    pts = [world[WRIST]] + [world[i] for i in chain]
    total = 0.0
    start = 2 if skip_base else 1
    for k in range(start, len(pts) - 1):
        a = pts[k] - pts[k - 1]
        b = pts[k + 1] - pts[k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-6 and nb > 1e-6:
            total += float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
    return total


def finger_lateral(world: np.ndarray, chain, R: np.ndarray) -> float:
    """Signed sideways angle (radians) of a finger within the palm plane.

    Uses the proximal segment direction (base to next joint) expressed in the hand
    frame R (columns across, forward, normal): the angle from 'forward' toward
    'across'. A finger pointing straight down the hand gives ~0; spreading toward
    the thumb or pinky gives a signed deviation. Drives the abduction joints.
    """
    base = world[chain[0]]
    nxt = world[chain[1]]
    d = nxt - base
    n = np.linalg.norm(d)
    if n < 1e-6:
        return 0.0
    d = d / n
    return float(np.arctan2(float(d @ R[:, 0]), float(d @ R[:, 1])))


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
