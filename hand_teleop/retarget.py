"""Cross-embodiment retargeting: human fingertip vectors -> LEAP joint angles.

Native MuJoCo implementation. We do NOT use dex-retargeting / Pinocchio here:
on Windows that stack fails to build (Pinocchio pulls Boost and compiles from
source). Instead we solve the same optimization directly with MuJoCo's own
forward kinematics and site Jacobians, which keeps the dependency surface to a
single `pip install mujoco`.

Objective solved per frame (matches the spec's cost function):

    L(q) = sum_i w_i || p_i(q) - target_i ||^2  +  lambda || q - q_prev ||^2

where p_i(q) is the robot fingertip site position from forward kinematics and
target_i is the human fingertip position mapped into the robot palm frame.
Warm-starting from q_prev plus the lambda term gives the temporal smoothing
that suppresses servo jitter. Solved with damped Gauss-Newton (Levenberg-
Marquardt), which converges in a handful of iterations for 16 DoF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import DEFAULT_SCENE
from .hand_frame import FINGER_NAMES, fingertip_vectors

TIP_SITES = ("if_tip_site", "mf_tip_site", "rf_tip_site", "th_tip_site")
PALM_SITE = "palm_site"


@dataclass
class RetargetConfig:
    lam: float = 0.02          # temporal smoothing weight (lambda)
    damping: float = 0.05      # Levenberg-Marquardt damping
    iters: int = 8             # Gauss-Newton iterations per frame
    # Per-finger position weights (thumb matters most for grasps).
    weights: tuple = (1.0, 1.0, 1.0, 1.2)
    # Default rotation mapping local-hand vectors into the robot palm frame.
    # Human frame (x=across index->pinky, y=along fingers, z=palm normal) ->
    # robot palm frame (x=along fingers, y=across, z=normal): a 90 deg turn
    # about the shared normal. Refined by calibrate() when available.
    align: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
    )


class LeapRetargeter:
    def __init__(self, scene_path=DEFAULT_SCENE, config: RetargetConfig | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.cfg = config or RetargetConfig()

        self.tip_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s) for s in TIP_SITES
        ]
        self.palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE)

        # Joint ranges for clipping the solution.
        self.lo = self.model.jnt_range[:, 0].copy()
        self.hi = self.model.jnt_range[:, 1].copy()
        self.nq = self.model.nq

        self.q = np.clip(np.zeros(self.nq), self.lo, self.hi)
        self._scratch_jac = np.zeros((3, self.model.nv))

        # Robot reference geometry at the open pose, used to set the mapping scale.
        self._palm0, self._robot_open_vecs = self._robot_geometry(np.zeros(self.nq))
        self.scale = float(np.mean(np.linalg.norm(self._robot_open_vecs, axis=1)))

    # -- forward kinematics helpers -------------------------------------------------

    def _robot_geometry(self, q: np.ndarray):
        """Return (palm_pos, tip_vectors) for a given joint configuration."""
        self.data.qpos[:] = np.clip(q, self.lo, self.hi)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        palm = self.data.site_xpos[self.palm_id].copy()
        vecs = np.array([self.data.site_xpos[t] - palm for t in self.tip_ids])
        return palm, vecs

    def _tip_targets(self, world_landmarks: np.ndarray) -> np.ndarray:
        """Map human landmarks to robot fingertip target positions (world frame)."""
        u = fingertip_vectors(world_landmarks)          # (4,3) normalized, local frame
        mapped = (self.cfg.align @ u.T).T * self.scale  # into robot palm frame
        return self._palm0 + mapped

    # -- calibration ----------------------------------------------------------------

    def calibrate(self, world_landmarks: np.ndarray) -> None:
        """Snap the align rotation + scale to a captured open-hand pose.

        Aligns the human open-hand fingertip vectors to the robot's open-pose
        fingertip vectors with a scaled orthogonal Procrustes (Kabsch) fit.
        """
        u = fingertip_vectors(world_landmarks)
        r = self._robot_open_vecs
        H = u.T @ r
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        self.cfg.align = Vt.T @ D @ U.T
        self.scale = float(np.mean(np.linalg.norm(r, axis=1)))

    # -- the solve ------------------------------------------------------------------

    def solve(self, world_landmarks: np.ndarray) -> np.ndarray:
        """Retarget one frame. Returns the 16-vector of LEAP joint targets."""
        targets = self._tip_targets(world_landmarks)
        q = self.q.copy()
        w = np.sqrt(np.asarray(self.cfg.weights))
        lam = np.sqrt(self.cfg.lam)
        q_prev = self.q.copy()

        for _ in range(self.cfg.iters):
            self.data.qpos[:] = np.clip(q, self.lo, self.hi)
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            rows_J, rows_r = [], []
            for i, tid in enumerate(self.tip_ids):
                mujoco.mj_jacSite(self.model, self.data, self._scratch_jac, None, tid)
                res = self.data.site_xpos[tid] - targets[i]
                rows_J.append(w[i] * self._scratch_jac.copy())
                rows_r.append(w[i] * res)
            # Temporal smoothing block: pull q toward q_prev.
            rows_J.append(lam * np.eye(self.nq))
            rows_r.append(lam * (q - q_prev))

            J = np.vstack(rows_J)                 # (12+nq, nq)
            r = np.concatenate(rows_r)            # (12+nq,)
            JTJ = J.T @ J + self.cfg.damping * np.eye(self.nq)
            dq = np.linalg.solve(JTJ, -J.T @ r)
            q = np.clip(q + dq, self.lo, self.hi)

        self.q = q
        return q.copy()

    def reset(self) -> None:
        self.q = np.clip(np.zeros(self.nq), self.lo, self.hi)


def fingertip_error_mm(retargeter: LeapRetargeter, world_landmarks: np.ndarray) -> np.ndarray:
    """Diagnostic: per-finger distance (mm) between robot tip and its target."""
    targets = retargeter._tip_targets(world_landmarks)
    retargeter.data.qpos[:] = retargeter.q
    mujoco.mj_kinematics(retargeter.model, retargeter.data)
    errs = []
    for i, tid in enumerate(retargeter.tip_ids):
        errs.append(np.linalg.norm(retargeter.data.site_xpos[tid] - targets[i]) * 1000.0)
    return np.array(errs)
