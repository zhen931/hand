"""Wrist pose mapping: human hand orientation (and optional position) -> the
floating base of the simulated hand.

Three modes:
  off    base stays at its rest pose (finger articulation only).
  orient absolute orientation: the robot palm points where the human palm points,
         so the robot mirrors the actual angle of the hand rather than moving
         relative to some preset rest pose.
  full   orientation plus translation. Translation comes from the 2D image
         landmarks: wrist pixel position drives sideways/vertical motion and the
         apparent hand size drives depth. Depth is the noisy axis, as expected
         from a single RGB camera.

Orientation is absolute and needs no calibration. Translation is relative to a
one-shot calibration snapshot. Output pose is exponentially smoothed.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .hand_frame import MIDDLE_MCP, WRIST, hand_local_frame

# Maps the MediaPipe world axes (x right, y down, z toward camera) to the MuJoCo
# world axes for a natural front view (x right, z up, y into the screen). Chosen
# so fingers-up maps to fingers-up and a hand tilt reads as the same tilt.
MEDIAPIPE_TO_MUJOCO = np.array([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, -1.0, 0.0]])


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
    def __init__(self, rest_pos, rest_quat, robot_frame, mode="orient",
                 mp_to_world=MEDIAPIPE_TO_MUJOCO,
                 xy_gain=0.18, z_gain=0.30, pos_alpha=0.4, rot_alpha=0.5):
        self.rest_pos = np.asarray(rest_pos, dtype=np.float64).copy()
        self.rest_quat = np.asarray(rest_quat, dtype=np.float64).copy()
        self.rest_R = quat2mat(self.rest_quat)
        # robot_frame: the robot hand anatomical frame (across, forward, normal)
        # in world at base rest. Used to relate the base orientation to the
        # anatomical orientation we are matching.
        self.robot_frame = np.asarray(robot_frame, dtype=np.float64).copy()
        self.mode = mode
        self.M = np.asarray(mp_to_world, dtype=np.float64).copy()
        self.xy_gain = xy_gain
        self.z_gain = z_gain
        self.pos_alpha = pos_alpha
        self.rot_alpha = rot_alpha

        self.img_wrist_cal = None
        self.img_size_cal = None

        self._pos = self.rest_pos.copy()
        self._quat = self.rest_quat.copy()

    def calibrate(self, world: np.ndarray, image: np.ndarray | None) -> None:
        """Only the translation reference needs calibrating; orientation is absolute."""
        if image is not None and np.any(image):
            self.img_wrist_cal = image[WRIST].copy()
            self.img_size_cal = max(
                float(np.linalg.norm(image[MIDDLE_MCP] - image[WRIST])), 1e-3)

    def reset(self) -> None:
        self._pos = self.rest_pos.copy()
        self._quat = self.rest_quat.copy()

    def pose(self, world: np.ndarray, image: np.ndarray | None):
        """Return (pos[3], quat[4]) for the floating base, smoothed."""
        if self.mode == "off":
            return self.rest_pos.copy(), self.rest_quat.copy()

        # Absolute orientation. We want the robot anatomical frame in world to
        # equal M @ H, where H is the human hand frame. Since the anatomical frame
        # rotates rigidly with the base, the base orientation that achieves this is
        #   B = (M H) robot_frame^T rest_R.
        H = hand_local_frame(world)
        R_base = self.M @ H @ self.robot_frame.T @ self.rest_R
        target_quat = mat2quat(R_base)

        target_pos = self.rest_pos.copy()
        if self.mode == "full" and image is not None and np.any(image) \
                and self.img_wrist_cal is not None:
            iw = image[WRIST]
            dy = self.xy_gain * (self.img_wrist_cal[0] - iw[0])
            dz = self.xy_gain * (self.img_wrist_cal[1] - iw[1])
            size = max(float(np.linalg.norm(image[MIDDLE_MCP] - iw)), 1e-3)
            dx = self.z_gain * (size / self.img_size_cal - 1.0)
            target_pos = self.rest_pos + np.array([dx, dy, dz])

        self._pos = self.pos_alpha * target_pos + (1 - self.pos_alpha) * self._pos
        self._quat = _nlerp(self._quat, target_quat, self.rot_alpha)
        return self._pos.copy(), self._quat.copy()
