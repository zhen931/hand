"""Wrist pose mapping: human hand orientation (and optional position) -> the
floating base of the simulated hand.

Three modes:
  off    base stays at its rest pose (finger articulation only).
  orient base rotation only, driven by the palm-plane frame we already build in
         hand_frame.py. No new tracking, reuses rotation we currently discard.
  full   orientation plus translation. Translation comes from the 2D image
         landmarks: wrist pixel position drives sideways/vertical motion and the
         apparent hand size drives depth. Depth is the noisy axis, as expected
         from a single RGB camera.

All motion is relative to a one-shot calibration snapshot (hold an open, neutral
hand). Output pose is exponentially smoothed to damp tracking jitter.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .hand_frame import MIDDLE_MCP, WRIST, hand_local_frame


def mat2quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


def quat2mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.ascontiguousarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def _nlerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Normalized linear interpolation between quaternions (cheap slerp)."""
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = (1.0 - a) * q0 + a * q1
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else q0


class WristMapper:
    def __init__(self, rest_pos, rest_quat, mode="orient",
                 xy_gain=0.18, z_gain=0.30, pos_alpha=0.4, rot_alpha=0.4):
        self.rest_pos = np.asarray(rest_pos, dtype=np.float64).copy()
        self.rest_quat = np.asarray(rest_quat, dtype=np.float64).copy()
        self.rest_R = quat2mat(self.rest_quat)
        self.mode = mode
        self.xy_gain = xy_gain
        self.z_gain = z_gain
        self.pos_alpha = pos_alpha
        self.rot_alpha = rot_alpha

        self.R_human_cal = None
        self.img_wrist_cal = None
        self.img_size_cal = None

        self._pos = self.rest_pos.copy()
        self._quat = self.rest_quat.copy()

    def calibrate(self, world: np.ndarray, image: np.ndarray | None) -> None:
        self.R_human_cal = hand_local_frame(world)
        if image is not None and np.any(image):
            self.img_wrist_cal = image[WRIST].copy()
            self.img_size_cal = max(
                float(np.linalg.norm(image[MIDDLE_MCP] - image[WRIST])), 1e-3)

    def reset(self) -> None:
        self._pos = self.rest_pos.copy()
        self._quat = self.rest_quat.copy()

    def pose(self, world: np.ndarray, image: np.ndarray | None):
        """Return (pos[3], quat[4]) for the floating base, smoothed."""
        if self.mode == "off" or self.R_human_cal is None:
            return self.rest_pos.copy(), self.rest_quat.copy()

        # Orientation: rotate the robot palm from rest by the same rotation the
        # human hand made relative to its calibration frame.
        R_human = hand_local_frame(world)
        R_rel = self.R_human_cal.T @ R_human
        R_robot = self.rest_R @ R_rel
        target_quat = mat2quat(R_robot)

        # Translation.
        target_pos = self.rest_pos.copy()
        if self.mode == "full" and image is not None and np.any(image) \
                and self.img_wrist_cal is not None:
            iw = image[WRIST]
            dy = self.xy_gain * (self.img_wrist_cal[0] - iw[0])   # image left/right
            dz = self.xy_gain * (self.img_wrist_cal[1] - iw[1])   # image up/down
            size = max(float(np.linalg.norm(image[MIDDLE_MCP] - iw)), 1e-3)
            dx = self.z_gain * (size / self.img_size_cal - 1.0)   # apparent depth
            target_pos = self.rest_pos + np.array([dx, dy, dz])

        self._pos = self.pos_alpha * target_pos + (1 - self.pos_alpha) * self._pos
        self._quat = _nlerp(self._quat, target_quat, self.rot_alpha)
        return self._pos.copy(), self._quat.copy()
