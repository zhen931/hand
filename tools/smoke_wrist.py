"""Headless check for wrist control: a static finger pose while the whole hand
rotates should rotate the robot palm; a synthetic image shift should translate
it. Renders PNGs to out/wrist and prints base pose so the motion is verifiable
without a display.

    python tools/smoke_wrist.py [hand]
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_teleop import synthetic
from hand_teleop.hands import DEFAULT_HAND, HANDS, build_model
from hand_teleop.render import OffscreenRenderer
from hand_teleop.retarget import Retargeter
from hand_teleop.wrist import WristMapper

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

OUT = Path(__file__).resolve().parent.parent / "out" / "wrist"
OUT.mkdir(parents=True, exist_ok=True)


def save(img, name):
    if HAVE_PIL:
        Image.fromarray(img).save(OUT / name)


def apply(rt, model, data, info, wrist, world, image):
    rt.solve(world)
    pos, quat = wrist.pose(world, image)
    data.qpos[info.finger_qadr] = rt.q
    data.qpos[info.base_qadr:info.base_qadr + 3] = pos
    data.qpos[info.base_qadr + 3:info.base_qadr + 7] = quat
    mujoco.mj_forward(model, data)
    return pos, quat


def main():
    hand = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HAND
    cfg = HANDS[hand]
    rt = Retargeter(cfg)
    model, data, info = build_model(cfg, floating=True)
    rest_pos = model.qpos0[info.base_qadr:info.base_qadr + 3].copy()
    rest_quat = model.qpos0[info.base_qadr + 3:info.base_qadr + 7].copy()
    renderer = OffscreenRenderer(model)
    pid = info.palm_site_id

    print(f"== {cfg.name} orientation ==")
    wrist = WristMapper(rest_pos, rest_quat, mode="orient", rot_alpha=1.0)
    frames = synthetic.wrist_sweep(120, curl=0.2)
    wrist.calibrate(frames[0], None)
    quats = []
    for label, k in [("rest", 0), ("tilt", 15), ("roll", 30), ("mix", 60), ("mix2", 90)]:
        pos, quat = apply(rt, model, data, info, wrist, frames[k], None)
        quats.append(quat)
        save(renderer.frame(data), f"wrist_{cfg.name}_orient_{label}.png")
        palm_R = data.site_xmat[pid].reshape(3, 3)
        print(f"  {label:5s} base quat {np.round(quat,3)}  palm z {np.round(palm_R[:,2],2)}")
    spread = max(np.linalg.norm(q - quats[0]) for q in quats)
    print(f"  max quat deviation from rest: {spread:.3f} "
          f"({'MOVES' if spread > 0.05 else 'STATIC - PROBLEM'})")

    print(f"== {cfg.name} translation (full) ==")
    wrist2 = WristMapper(rest_pos, rest_quat, mode="full", pos_alpha=1.0, rot_alpha=1.0)
    base_world = synthetic.hand_pose(0.2)
    img_cal = np.zeros((21, 2), np.float32)
    img_cal[0] = [0.5, 0.5]
    img_cal[9] = [0.5, 0.35]
    wrist2.calibrate(base_world, img_cal)
    for label, shift, scale in [("center", [0, 0], 1.0), ("left", [-0.2, 0], 1.0),
                                ("up", [0, -0.2], 1.0), ("closer", [0, 0], 1.6)]:
        img = np.zeros((21, 2), np.float32)
        img[0] = [0.5 + shift[0], 0.5 + shift[1]]
        img[9] = [0.5 + shift[0], 0.5 + shift[1] - 0.15 * scale]
        pos, _ = apply(rt, model, data, info, wrist2, base_world, img)
        save(renderer.frame(data), f"wrist_{cfg.name}_full_{label}.png")
        print(f"  {label:7s} base pos delta {np.round(pos - rest_pos, 3)}")


if __name__ == "__main__":
    main()
