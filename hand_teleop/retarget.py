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

from .hand_frame import fingertip_vectors, orient_normal
from .hands import DEFAULT_HAND, HANDS, build_model


@dataclass
class RetargetConfig:
    lam: float = 0.004         # temporal smoothing weight (lambda); low = snappy
    damping: float = 0.05      # Levenberg-Marquardt damping
    iters: int = 12            # Gauss-Newton iterations per frame


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
        self.reg = self.info.reg          # per-joint pull toward neutral
        self.n = len(self.qadr)

        self.q = np.clip(np.zeros(self.n), self.lo, self.hi)
        self._scratch_jac = np.zeros((3, self.model.nv))

        # Robot reference geometry at the open pose (driven joints = 0).
        self._palm0, self._robot_open_vecs = self._robot_geometry(self.q)
        # The mapping rotation is the robot's own hand frame (across, forward,
        # normal). Building it from the orthonormal frame keeps the palm-normal
        # (curl) axis well defined, unlike a Kabsch fit on the near-coplanar open
        # fingertips. Human fingertip vectors arrive already in the human frame
        # (see hand_frame.fingertip_vectors), so this rotation carries them into
        # the robot frame directly and is invariant to wrist orientation.
        self.align = self._robot_frame()
        self.scale = float(np.mean(np.linalg.norm(self._robot_open_vecs, axis=1)))

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

    def _robot_frame(self) -> np.ndarray:
        """Robot hand frame (across, forward, normal) as world-space columns.

        Same construction as the human hand_local_frame, so the two frames are
        directly comparable: across from the outer non-thumb fingertips, forward
        from palm to the fingertip centroid, normal from their cross product.
        """
        r = self._robot_open_vecs
        names = [f.name for f in self.hand.fingers]
        nz = [i for i, n in enumerate(names) if n != "thumb"]

        def unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-9 else v

        across = unit(r[nz[-1]] - r[nz[0]])
        forward = unit(r.mean(axis=0))
        normal = unit(np.cross(across, forward))
        across = unit(np.cross(forward, normal))
        thumb_idx = names.index("thumb")
        across, forward, normal = orient_normal(across, forward, normal, r[thumb_idx])
        return np.column_stack([across, forward, normal])

    def _tip_targets(self, world_landmarks: np.ndarray) -> np.ndarray:
        u = fingertip_vectors(world_landmarks, self.landmarks)   # (N, 3)
        mapped = (self.align @ u.T).T * self.scale
        return self._palm0 + mapped

    # -- calibration ----------------------------------------------------------------

    def calibrate(self, world_landmarks: np.ndarray) -> None:
        """Fit the mapping scale from an open-hand pose.

        The rotation is fixed (the robot frame, see __init__); calibration only
        sets the scale that maps the operator's open hand onto the robot's open
        hand. This is essential: without it the human wrist-to-tip vectors (long)
        overshoot the robot's palm-to-tip reach (short), pinning the fingers
        extended so they barely curl. Optimal scale for fixed rotation is
        <r, R u> / <R u, R u>.
        """
        u = fingertip_vectors(world_landmarks, self.landmarks)
        r = self._robot_open_vecs
        Ru = (self.align @ u.T).T
        denom = float(np.sum(Ru * Ru))
        if denom > 1e-9:
            self.scale = float(np.sum(r * Ru) / denom)

    # -- the solve ------------------------------------------------------------------

    def solve(self, world_landmarks: np.ndarray) -> np.ndarray:
        targets = self._tip_targets(world_landmarks)
        q = self.q.copy()
        q_prev = self.q.copy()
        w = np.sqrt(self.weights)
        lam = np.sqrt(self.cfg.lam)
        reg = np.sqrt(self.reg)

        for _ in range(self.cfg.iters):
            self._set_qpos(q)
            rows_J, rows_r = [], []
            for i, tid in enumerate(self.tip_ids):
                mujoco.mj_jacSite(self.model, self.data, self._scratch_jac, None, tid)
                res = self.data.site_xpos[tid] - targets[i]
                rows_J.append(w[i] * self._scratch_jac[:, self.vadr])
                rows_r.append(w[i] * res)
            # Temporal smoothing toward the previous pose.
            rows_J.append(lam * np.eye(self.n))
            rows_r.append(lam * (q - q_prev))
            # Neutral pull on the spread joints so fingers do not splay/cross.
            rows_J.append(np.diag(reg))
            rows_r.append(reg * q)

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
