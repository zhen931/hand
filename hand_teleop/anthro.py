"""Parametric anthropomorphic hand: a MuJoCo hand built to human kinematics.

This is our own design, generated from parameters (finger lengths, joint ranges,
palm size), so it doubles as:
  - a canonical, near-human hand where retargeting is almost a direct angle copy
    (each finger has MCP flexion + abduction, PIP, DIP, like a real hand),
  - the design prototype for the physical hand we intend to build (the parameters
    are the design; change them and regenerate),
  - a per-user fit (scale the finger lengths to the operator).

Frame: palm in the x-y plane, fingers along +y, across +x, palm normal +z.
Built with mjSpec so it is generated fresh from the parameters, no XML to hand
edit. Kinematic control (we set qpos), so no actuators are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

HINGE = mujoco.mjtJoint.mjJNT_HINGE
CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
BOX = mujoco.mjtGeom.mjGEOM_BOX


@dataclass
class FingerSpec:
    name: str
    x: float                 # base offset across the palm (m)
    lengths: tuple           # (proximal, middle, distal) phalanx lengths (m)
    radius: float = 0.008


@dataclass
class AnthroParams:
    palm_w: float = 0.075
    palm_l: float = 0.085
    palm_t: float = 0.022
    palm_z: float = 0.15
    fingers: list = field(default_factory=lambda: [
        FingerSpec("index", -0.027, (0.040, 0.024, 0.020)),
        FingerSpec("middle", -0.009, (0.045, 0.028, 0.022)),
        FingerSpec("ring", 0.009, (0.040, 0.026, 0.020)),
        FingerSpec("pinky", 0.027, (0.033, 0.020, 0.018), radius=0.007),
    ])
    # Thumb: base on the radial side, resting pointing direction (world), and
    # phalanx lengths (metacarpal, proximal, distal). The direction points the
    # thumb forward, across toward the fingers, and up onto the palmar side, so it
    # sits opposed rather than sticking out.
    thumb_pos: tuple = (-0.040, 0.020, 0.006)
    thumb_dir: tuple = (-0.3, 0.92, 0.0)
    thumb_lengths: tuple = (0.034, 0.030, 0.024)
    thumb_radius: float = 0.010

    # Joint ranges (radians).
    mcp_flex: tuple = (-0.3, 1.6)
    abd: tuple = (-0.35, 0.35)
    pip: tuple = (-0.1, 1.7)
    dip: tuple = (-0.1, 1.5)
    thumb_cmc: tuple = (-0.6, 1.0)
    thumb_abd: tuple = (-0.5, 0.9)
    thumb_mcp_flex: tuple = (-0.2, 1.2)
    thumb_ip: tuple = (-0.1, 1.4)


def _finger_bodies(palm, f: FingerSpec, p: AnthroParams):
    """Add a 3-phalanx finger (MCP flex+abd, PIP, DIP) to the palm body."""
    L1, L2, L3 = f.lengths
    r = f.radius
    prox = palm.add_body(name=f"{f.name}_prox", pos=[f.x, p.palm_l, 0])
    prox.add_joint(name=f"{f.name}_abd", type=HINGE, axis=[0, 0, 1], range=list(p.abd))
    prox.add_joint(name=f"{f.name}_mcp", type=HINGE, axis=[1, 0, 0], range=list(p.mcp_flex))
    prox.add_geom(type=CAPSULE, fromto=[0, 0, 0, 0, L1, 0], size=[r, 0, 0])

    mid = prox.add_body(name=f"{f.name}_mid", pos=[0, L1, 0])
    mid.add_joint(name=f"{f.name}_pip", type=HINGE, axis=[1, 0, 0], range=list(p.pip))
    mid.add_geom(type=CAPSULE, fromto=[0, 0, 0, 0, L2, 0], size=[r, 0, 0])

    dist = mid.add_body(name=f"{f.name}_dist", pos=[0, L2, 0])
    dist.add_joint(name=f"{f.name}_dip", type=HINGE, axis=[1, 0, 0], range=list(p.dip))
    dist.add_geom(type=CAPSULE, fromto=[0, 0, 0, 0, L3, 0], size=[r * 0.9, 0, 0])


def _quat_from_y(d):
    """Shortest-arc quaternion rotating local +y onto world direction d."""
    a = np.array([0.0, 1.0, 0.0])
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    axis = np.cross(a, d)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    axis = axis / s
    ang = np.arccos(np.clip(np.dot(a, d), -1.0, 1.0))
    return [np.cos(ang / 2), *(axis * np.sin(ang / 2))]


def _thumb_bodies(palm, p: AnthroParams):
    L1, L2, L3 = p.thumb_lengths
    r = p.thumb_radius
    meta = palm.add_body(name="thumb_meta", pos=list(p.thumb_pos),
                         quat=_quat_from_y(p.thumb_dir))
    # Carpometacarpal saddle joint: two DOF (flexion across the palm + opposition).
    meta.add_joint(name="thumb_cmc", type=HINGE, axis=[1, 0, 0], range=list(p.thumb_cmc))
    meta.add_joint(name="thumb_abd", type=HINGE, axis=[0, 0, 1], range=list(p.thumb_abd))
    meta.add_geom(type=CAPSULE, fromto=[0, 0, 0, 0, L1, 0], size=[r, 0, 0])

    prox = meta.add_body(name="thumb_prox", pos=[0, L1, 0])
    prox.add_joint(name="thumb_mcp", type=HINGE, axis=[1, 0, 0], range=list(p.thumb_mcp_flex))
    prox.add_geom(type=CAPSULE, fromto=[0, 0, 0, 0, L2, 0], size=[r * 0.9, 0, 0])

    dist = prox.add_body(name="thumb_dist", pos=[0, L2, 0])
    dist.add_joint(name="thumb_pip", type=HINGE, axis=[1, 0, 0], range=list(p.thumb_ip))
    dist.add_geom(type=CAPSULE, fromto=[0, 0, 0, 0, L3, 0], size=[r * 0.85, 0, 0])


def build_anthro_spec(p: AnthroParams | None = None) -> mujoco.MjSpec:
    p = p or AnthroParams()
    spec = mujoco.MjSpec()
    spec.compiler.degree = False  # radians

    spec.add_material(name="skin", rgba=[0.85, 0.72, 0.62, 1])
    # A simple scene so it renders on its own (skybox, ground, a light).
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=256, height=256)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], pos=[0, 0, 0], rgba=[0.3, 0.35, 0.4, 1])
    spec.worldbody.add_light(pos=[0.2, 0.4, 1.0], dir=[-0.1, -0.3, -1.0],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    palm = spec.worldbody.add_body(name="palm", pos=[0, 0, p.palm_z], quat=[1, 0, 0, 0])
    palm.add_geom(type=BOX, pos=[0, p.palm_l / 2, 0],
                  size=[p.palm_w / 2, p.palm_l / 2, p.palm_t / 2])
    for f in p.fingers:
        _finger_bodies(palm, f, p)
    _thumb_bodies(palm, p)

    for g in spec.geoms:
        g.material = "skin"
    return spec


def tip_offsets(p: AnthroParams | None = None) -> dict:
    """Fingertip site offsets (distal length along +y), for the model factory."""
    p = p or AnthroParams()
    out = {f.name: (0.0, f.lengths[2], 0.0) for f in p.fingers}
    out["thumb"] = (0.0, p.thumb_lengths[2], 0.0)
    return out
