"""Cross-embodiment retargeting: human hand pose -> robot finger joint angles.

Hand-agnostic. The hand geometry comes from a HandConfig (see hands.py); this
module only needs each finger's landmark chain and which joints flex it.

Fingers use direct flexion mapping: we measure how curled each human finger is
(the sum of its bend angles, see hand_frame.finger_bend) and drive the robot
finger's flexion joints across their range by that amount. This is monotonic and
robust across embodiments. Fingertip-position matching (mapping the human tip to
a robot target and solving IK) was tried first and abandoned: a single hand
rotation plus scale cannot align each finger's direction, so it produced splayed,
crossing, non-monotonic poses (open hands that would not fully open, fists that
hyperextended). Abduction and thumb-opposition joints are left neutral, which
keeps fingers parallel (no crossing) and the thumb from folding.

Native MuJoCo, one dependency. We do NOT use dex-retargeting / Pinocchio: on
Windows that stack fails to build (Pinocchio compiles Boost from source).

Finger angles are palm-relative, so the wrist/base pose is handled entirely
separately (see wrist.py); the robot hand frame built here (self.align) is what
that module uses. The unused position-IK helpers (_tip_targets, scale) are kept
only for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .hand_frame import (finger_bend, finger_lateral, fingertip_vectors,
                        hand_local_frame)
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

        # Per-finger flexion and abduction joints, for the direct mapping in solve.
        self._names = self._joint_names()
        self._flex = self._flex_groups()
        self._lat = self._lateral_groups()
        self._beta = 0.6                  # joint-space temporal smoothing
        self.lat_sign = 1.0               # flip if fingers spread the wrong way
        nfing = len(self.hand.fingers)
        # Bend/spread at a relaxed open hand are not zero (fingers curve and fan
        # slightly, plus tracking noise), so subtract per-finger open baselines;
        # press 'c' with a flat hand to set them. Gains amplify to full range.
        self._bend_open = np.array([0.35] * nfing)
        self._spread_open = np.zeros(nfing)
        self._bend_gain = np.array([f.bend_gain for f in self.hand.fingers])
        self._lat_gain = np.array([f.lat_gain for f in self.hand.fingers])
        self._lat_cap = np.array([f.lat_cap for f in self.hand.fingers])
        self._lat_bias = np.array([f.lat_bias for f in self.hand.fingers])

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
        # Per-finger scale (one per fingertip). A single shared scale makes long
        # fingers fall short of full extension and short fingers overshoot; per
        # finger, each one's open maps to its own full extension and curl tracks.
        self.scale = np.linalg.norm(self._robot_open_vecs, axis=1) / 1.8

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
        # No thumb-based normal flip here, to match hand_local_frame (which drops
        # it so the thumb cannot move the wrist). Both use the same raw rule, so
        # the mapping stays consistent.
        return np.column_stack([across, forward, normal])

    def _joint_names(self):
        names = []
        for qa in self.qadr:
            nm = ""
            for j in range(self.model.njnt):
                if self.model.jnt_qposadr[j] == qa:
                    nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    break
            names.append(nm or "")
        return names

    def _flex_groups(self):
        """For each finger, the indices of its flexion joints and their upper
        limits (fully curled). Abduction, thumb opposition (cmc), etc. are not
        flexion and are left at neutral by the solve."""
        groups = []
        for finger in self.hand.fingers:
            idxs, his = [], []
            for i, nm in enumerate(self._names):
                if finger.token in nm and any(t in nm for t in ("mcp", "pip", "dip")):
                    idxs.append(i)
                    his.append(self.hi[i])
            groups.append((np.array(idxs, dtype=int), np.array(his)))
        return groups

    def _lateral_groups(self):
        """For each finger, the indices of its sideways (abduction / thumb
        opposition) joints, driven from the human finger's spread."""
        groups = []
        for finger in self.hand.fingers:
            idxs = []
            for i, nm in enumerate(self._names):
                if finger.token in nm and any(t in nm for t in ("abd", "rot", "cmc")):
                    idxs.append(i)
            groups.append(np.array(idxs, dtype=int))
        return groups

    def _tip_targets(self, world_landmarks: np.ndarray) -> np.ndarray:
        u = fingertip_vectors(world_landmarks, self.landmarks)   # (N, 3)
        mapped = (self.align @ u.T).T * self.scale[:, None]
        return self._palm0 + mapped

    # -- calibration ----------------------------------------------------------------

    def calibrate(self, world_landmarks: np.ndarray) -> None:
        """Record each finger's bend and spread at this (open, neutral) pose as the
        baselines, so a fully open hand drives the robot fingers fully open and a
        neutral spread reads as zero abduction. Hold a flat hand and press 'c'."""
        R = hand_local_frame(world_landmarks)
        self._bend_open = np.array([
            finger_bend(world_landmarks, f.chain) for f in self.hand.fingers])
        self._spread_open = np.array([
            finger_lateral(world_landmarks, f.chain, R) for f in self.hand.fingers])

    # -- the solve ------------------------------------------------------------------

    def solve(self, world_landmarks: np.ndarray) -> np.ndarray:
        """Direct flexion mapping: drive each robot finger's flexion joints from
        how curled the corresponding human finger is.

        Cross-embodiment fingertip-position matching (a single rotation plus
        scale) cannot align each finger's direction, giving splayed, non-monotonic
        poses. Mapping the human finger bend straight onto the robot flexion range
        is monotonic and robust: an open hand opens, a fist closes. Spread and
        thumb-opposition joints are left neutral (no crossing, no fold).
        """
        q = np.zeros(self.n)
        R = hand_local_frame(world_landmarks)
        for fi, finger in enumerate(self.hand.fingers):
            # Flexion: distribute the finger's curl across its flexion joints.
            idxs, his = self._flex[fi]
            sum_hi = float(np.sum(his))
            if len(idxs) and sum_hi > 1e-6:
                bend = finger_bend(world_landmarks, finger.chain)
                curl = (bend - self._bend_open[fi]) * self._bend_gain[fi]
                q[idxs] = float(np.clip(curl, 0.0, sum_hi)) * (his / sum_hi)
            # Abduction: sideways spread relative to the neutral baseline, capped
            # well inside the joint limits so fingers fan but never cross.
            lat = self._lat[fi]
            if len(lat):
                spread = finger_lateral(world_landmarks, finger.chain, R)
                a = (spread - self._spread_open[fi]) * self._lat_gain[fi] * self.lat_sign
                a = self._lat_bias[fi] + np.clip(a, -self._lat_cap[fi], self._lat_cap[fi])
                q[lat] = np.clip(a, self.lo[lat], self.hi[lat])
        q = np.clip(q, self.lo, self.hi)
        self.q = self._beta * q + (1 - self._beta) * self.q
        return self.q.copy()

    def reset(self) -> None:
        self.q = np.clip(np.zeros(self.n), self.lo, self.hi)


def fingertip_error_mm(rt: Retargeter, world_landmarks: np.ndarray) -> np.ndarray:
    """Diagnostic: per-finger distance (mm) between robot tip and its target."""
    targets = rt._tip_targets(world_landmarks)
    rt._set_qpos(rt.q)
    return np.array([
        np.linalg.norm(rt.data.site_xpos[tid] - targets[i]) * 1000.0
        for i, tid in enumerate(rt.tip_ids)])
