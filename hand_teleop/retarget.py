"""Cross-embodiment retargeting: human fingertip vectors -> robot joint angles.

Hand-agnostic. The hand geometry comes from a HandConfig (see hands.py); this
module only knows about fingertip sites, a palm site, and a set of driven joints.

Native MuJoCo implementation. We do NOT use dex-retargeting / Pinocchio here:
on Windows that stack fails to build (Pinocchio pulls Boost and compiles from
source). Instead we solve the same optimization directly with MuJoCo's own
forward kinematics and site Jacobians, one dependency (`mujoco`).

Objective solved per frame (matches the spec's cost function):

    L(q) = sum_i w_i || p_i(q) - target_i ||^2  +  lambda || q - q_prev ||^2

where p_i(q) is robot fingertip i from forward kinematics and target_i is the
human fingertip mapped into the robot palm frame. Warm-starting from q_prev plus
the lambda term gives temporal smoothing. Solved with damped Gauss-Newton
(Levenberg-Marquardt). Angles are palm-relative, so the wrist/base pose is
handled entirely separately (see wrist.py); this solver keeps the base fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .hand_frame import fingertip_vectors
from .hands import DEFAULT_HAND, HANDS, build_model


@dataclass
class RetargetConfig:
    lam: float = 0.02          # temporal smoothing weight (lambda)
    damping: float = 0.05      # Levenberg-Marquardt damping
    iters: int = 8             # Gauss-Newton iterations per frame


class Retargeter:
    def __init__(self, hand=None, config: RetargetConfig | None = None):
        if hand is None:
            hand = HANDS[DEFAULT_HAND]
        elif isinstance(hand, str):
            hand = HANDS[hand]
        self.hand = hand
        self.model, self.data, self.info = build_model(hand, floating=False)
        self.cfg = config or RetargetConfig()

        self.qadr = self.info.finger_qadr
        self.vadr = self.info.finger_vadr
        self.lo, self.hi = self.info.lo, self.info.hi
        self.tip_ids = self.info.tip_site_ids
        self.palm_id = self.info.palm_site_id
        self.landmarks = self.info.landmarks
        self.weights = self.info.weights
        self.n = len(self.qadr)

        self.q = np.clip(np.zeros(self.n), self.lo, self.hi)
        self._scratch_jac = np.zeros((3, self.model.nv))

        # Robot reference geometry at the open pose (driven joints = 0).
        self._palm0, self._robot_open_vecs = self._robot_geometry(self.q)
        self.scale = float(np.mean(np.linalg.norm(self._robot_open_vecs, axis=1)))
        # Default mapping rotation; refined by calibrate(). Identity until then.
        self.align = np.eye(3)

    # -- forward kinematics helpers -------------------------------------------------

    def _set_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = np.clip(q, self.lo, self.hi)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def _robot_geometry(self, q: np.ndarray):
        self._set_qpos(q)
        palm = self.data.site_xpos[self.palm_id].copy()
        vecs = np.array([self.data.site_xpos[t] - palm for t in self.tip_ids])
        return palm, vecs

    def _tip_targets(self, world_landmarks: np.ndarray) -> np.ndarray:
        u = fingertip_vectors(world_landmarks, self.landmarks)   # (N, 3)
        mapped = (self.align @ u.T).T * self.scale
        return self._palm0 + mapped

    # -- calibration ----------------------------------------------------------------

    def calibrate(self, world_landmarks: np.ndarray) -> None:
        """Snap the align rotation + scale to a captured open-hand pose (Kabsch)."""
        u = fingertip_vectors(world_landmarks, self.landmarks)
        r = self._robot_open_vecs
        H = u.T @ r
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        self.align = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        self.scale = float(np.mean(np.linalg.norm(r, axis=1)))

    # -- the solve ------------------------------------------------------------------

    def solve(self, world_landmarks: np.ndarray) -> np.ndarray:
        targets = self._tip_targets(world_landmarks)
        q = self.q.copy()
        q_prev = self.q.copy()
        w = np.sqrt(self.weights)
        lam = np.sqrt(self.cfg.lam)

        for _ in range(self.cfg.iters):
            self._set_qpos(q)
            rows_J, rows_r = [], []
            for i, tid in enumerate(self.tip_ids):
                mujoco.mj_jacSite(self.model, self.data, self._scratch_jac, None, tid)
                res = self.data.site_xpos[tid] - targets[i]
                rows_J.append(w[i] * self._scratch_jac[:, self.vadr])
                rows_r.append(w[i] * res)
            rows_J.append(lam * np.eye(self.n))
            rows_r.append(lam * (q - q_prev))

            J = np.vstack(rows_J)
            r = np.concatenate(rows_r)
            JTJ = J.T @ J + self.cfg.damping * np.eye(self.n)
            dq = np.linalg.solve(JTJ, -J.T @ r)
            q = np.clip(q + dq, self.lo, self.hi)

        self.q = q
        return q.copy()

    def reset(self) -> None:
        self.q = np.clip(np.zeros(self.n), self.lo, self.hi)


def fingertip_error_mm(rt: Retargeter, world_landmarks: np.ndarray) -> np.ndarray:
    """Diagnostic: per-finger distance (mm) between robot tip and its target."""
    targets = rt._tip_targets(world_landmarks)
    rt._set_qpos(rt.q)
    return np.array([
        np.linalg.norm(rt.data.site_xpos[tid] - targets[i]) * 1000.0
        for i, tid in enumerate(rt.tip_ids)])
