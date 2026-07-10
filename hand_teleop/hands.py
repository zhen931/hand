"""Hand configurations and a single model factory.

Adding a new robot hand means writing one HandConfig: which body is the palm,
which bodies are the fingertips, which human landmark each maps to, and which
joints to drive. The retargeter and the mirror both build their models through
build_model, so nothing downstream is hand-specific.

Fingertip and palm sites are injected at load time with mjSpec, so we do not
hand-edit the vendored model XML. The mirror also injects a free joint on the
base body for wrist control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import ASSETS_DIR

# MediaPipe fingertip landmark indices.
LM_THUMB, LM_INDEX, LM_MIDDLE, LM_RING, LM_PINKY = 4, 8, 12, 16, 20


@dataclass
class Finger:
    name: str
    tip_body: str
    landmark: int
    chain: tuple            # landmark indices base->tip, e.g. index (5,6,7,8)
    token: str              # substring identifying this finger's joints
    weight: float = 1.0
    bend_gain: float = 1.6  # amplify bend->curl (thumb needs more)
    bend_skip_base: bool = False  # thumb: ignore the CMC-position angle
    lat_gain: float = 0.6   # sideways spread -> abduction (kept modest to not cross)
    lat_cap: float = 0.3    # max abduction magnitude (radians), so fingers cannot cross
    lat_bias: float = 0.0   # resting offset on the sideways joints
    tip_offset: tuple = (0.0, 0.0, 0.0)


@dataclass
class HandConfig:
    name: str
    scene: str                      # path relative to assets/ (or "" if builder set)
    palm_body: str
    base_body: str                  # body to attach the free joint to
    fingers: list                   # list[Finger], canonical order
    builder: object = None          # callable -> MjSpec, instead of loading a scene
    exclude_joints: tuple = ()      # actuated joints not driven by retargeting
    palm_offset: tuple = (0.0, 0.0, 0.0)
    # Per-joint neutral regularization: joint-name substring -> weight. Keeps the
    # solver from splaying/crossing fingers (abduction joints) or folding the
    # thumb to its limits, without stopping flexion tracking. A joint takes the
    # max weight of the tokens it matches.
    reg_tokens: dict = field(default_factory=lambda: {"abd": 0.08, "rot": 0.08})

    @property
    def scene_path(self):
        return ASSETS_DIR / self.scene


# ORCA v2 right hand: 5 fingers, fully actuated. Body names carry mjSpec hashes.
# Middle vs ring resolved by rest lateral position (both use the M mesh).
ORCA = HandConfig(
    name="orca",
    scene="orca_hand/scene_right.xml",
    palm_body="right_R-Carpals_8d1f1041",
    base_body="right_mount",
    # tip_offset pushes the site from the fingertip-link origin (at the knuckle)
    # out to the actual tip, along the link's local +z. Without this the site sits
    # on the joint pivot and the fingers cannot be driven to curl.
    fingers=[
        Finger("index", "right_I-FingerTipAssembly_ec49c16c", LM_INDEX,
               (5, 6, 7, 8), "_i-", tip_offset=(0, 0, 0.028)),
        Finger("middle", "right_M-FingerTipAssembly_34afb748", LM_MIDDLE,
               (9, 10, 11, 12), "_m-", tip_offset=(0, 0, 0.028)),
        Finger("ring", "right_M-FingerTipAssembly_424a8e75", LM_RING,
               (13, 14, 15, 16), "_r-", tip_offset=(0, 0, 0.028)),
        Finger("pinky", "right_P-FingerTipAssembly_cd219176", LM_PINKY,
               (17, 18, 19, 20), "_p-", tip_offset=(0, 0, 0.028)),
        Finger("thumb", "right_T-DP_b7429e50", LM_THUMB,
               (1, 2, 3, 4), "_t-", weight=0.5, bend_gain=2.5, bend_skip_base=True,
               lat_gain=1.0, lat_cap=0.9, lat_bias=0.35,
               tip_offset=(0, 0, 0.028)),
    ],
    exclude_joints=("right_wrist",),
    # abd/rot: keep fingers from splaying. t-cmc: strong, or the thumb swings to
    # its limit and folds across the palm. t-mcp/t-pip: light, so the thumb rests
    # mostly extended but can still curl a little to track.
    reg_tokens={"abd": 0.08, "rot": 0.08, "t-cmc": 0.15, "t-mcp": 0.04, "t-pip": 0.04},
)

def _anthro_config():
    """Our own parametric anthropomorphic hand (human kinematics, sim-only). Built
    from parameters, so it doubles as the design prototype for a physical hand and
    the canonical near-human representation. See anthro.py."""
    from .anthro import build_anthro_spec, tip_offsets
    offs = tip_offsets()
    return HandConfig(
        name="anthro", scene="", palm_body="palm", base_body="palm",
        builder=build_anthro_spec,
        fingers=[
            Finger("index", "index_dist", LM_INDEX, (5, 6, 7, 8), "index_",
                   tip_offset=offs["index"]),
            Finger("middle", "middle_dist", LM_MIDDLE, (9, 10, 11, 12), "middle_",
                   tip_offset=offs["middle"]),
            Finger("ring", "ring_dist", LM_RING, (13, 14, 15, 16), "ring_",
                   tip_offset=offs["ring"]),
            Finger("pinky", "pinky_dist", LM_PINKY, (17, 18, 19, 20), "pinky_",
                   tip_offset=offs["pinky"]),
            Finger("thumb", "thumb_dist", LM_THUMB, (1, 2, 3, 4), "thumb_",
                   bend_gain=2.4, bend_skip_base=True, lat_gain=1.0, lat_cap=0.9,
                   tip_offset=offs["thumb"]),
        ],
    )


ANTHRO = _anthro_config()
HANDS = {"orca": ORCA, "anthro": ANTHRO}
DEFAULT_HAND = "anthro"


@dataclass
class ModelInfo:
    cfg: HandConfig
    tip_site_ids: np.ndarray
    palm_site_id: int
    finger_qadr: np.ndarray         # qpos indices of the driven joints
    finger_vadr: np.ndarray         # dof indices of the driven joints
    lo: np.ndarray
    hi: np.ndarray
    landmarks: list
    weights: np.ndarray
    reg: np.ndarray                 # per-joint pull toward neutral (spread joints)
    base_qadr: int | None = None
    base_vadr: int | None = None


def build_model(cfg: HandConfig, floating: bool = False):
    """Compile the hand with fingertip/palm sites (and an optional free joint).

    Returns (model, data, ModelInfo).
    """
    spec = cfg.builder() if cfg.builder is not None \
        else mujoco.MjSpec.from_file(str(cfg.scene_path))

    spec.body(cfg.palm_body).add_site(
        name="palm_site", pos=list(cfg.palm_offset),
        size=[0.008, 0.008, 0.008], rgba=[0, 1, 0, 0.5])
    for f in cfg.fingers:
        spec.body(f.tip_body).add_site(
            name=f"{f.name}_tip", pos=list(f.tip_offset),
            size=[0.006, 0.006, 0.006], rgba=[1, 0, 0, 0.7])

    if floating:
        spec.body(cfg.base_body).add_freejoint()
        spec.option.gravity = [0.0, 0.0, 0.0]

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tip_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{f.name}_tip")
        for f in cfg.fingers])
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm_site")

    # Driven joints: every hinge joint except the excluded ones.
    qadr, vadr, lo, hi, reg = [], [], [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in cfg.exclude_joints:
            continue
        qadr.append(model.jnt_qposadr[j])
        vadr.append(model.jnt_dofadr[j])
        lo.append(model.jnt_range[j, 0])
        hi.append(model.jnt_range[j, 1])
        matched = [w for tok, w in cfg.reg_tokens.items() if tok in name]
        reg.append(max(matched) if matched else 0.0)

    base_q = base_v = None
    if floating:
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                base_q, base_v = model.jnt_qposadr[j], model.jnt_dofadr[j]
                break

    info = ModelInfo(
        cfg=cfg,
        tip_site_ids=tip_ids,
        palm_site_id=palm_id,
        finger_qadr=np.array(qadr),
        finger_vadr=np.array(vadr),
        lo=np.array(lo),
        hi=np.array(hi),
        landmarks=[f.landmark for f in cfg.fingers],
        weights=np.array([f.weight for f in cfg.fingers]),
        reg=np.array(reg),
        base_qadr=base_q,
        base_vadr=base_v,
    )
    return model, data, info
