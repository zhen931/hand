"""Headless check for wrist control: a static finger pose while the whole hand
rotates should rotate the robot palm; a synthetic image shift should translate
it. Renders PNGs to out/wrist and prints base pose so the motion is verifiable
without a display.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_teleop import synthetic
from hand_teleop.mirror import build_floating_model
from hand_teleop.render import OffscreenRenderer
from hand_teleop.retarget import LeapRetargeter
from hand_teleop.wrist import WristMapper

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

OUT = Path(__file__).resolve().parent.parent / "out" / "wrist"
OUT.mkdir(parents=True, exist_ok=True)


def step_and_render(rt, model, data, base_q, base_v, wrist, world, image, renderer, n=4):
    for _ in range(n):
        rt.solve(world)
        pos, quat = wrist.pose(world, image)
        data.ctrl[:] = rt.q
        data.qpos[base_q:base_q + 3] = pos
        data.qpos[base_q + 3:base_q + 7] = quat
        data.qvel[base_v:base_v + 6] = 0.0
        mujoco.mj_step(model, data)
    return renderer.frame(data), pos, quat


def save(img, name):
    if HAVE_PIL:
        Image.fromarray(img).save(OUT / name)


def main():
    rt = LeapRetargeter()
    model, data, base_q, base_v = build_floating_model()
    rest_pos = model.qpos0[base_q:base_q + 3].copy()
    rest_quat = model.qpos0[base_q + 3:base_q + 7].copy()
    renderer = OffscreenRenderer(model)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "palm_site")

    # --- orientation ---
    print("== orientation ==")
    wrist = WristMapper(rest_pos, rest_quat, mode="orient", rot_alpha=1.0)
    frames = synthetic.wrist_sweep(120, curl=0.2)
    wrist.calibrate(frames[0], None)
    quats = []
    for label, k in [("rest", 0), ("tilt", 15), ("roll", 30), ("mix", 60), ("mix2", 90)]:
        img, pos, quat = step_and_render(rt, model, data, base_q, base_v, wrist,
                                         frames[k], None, renderer)
        quats.append(quat)
        save(img, f"wrist_orient_{label}.png")
        palm_R = data.site_xmat[pid].reshape(3, 3)
        print(f"  {label:5s} frame {k:3d}  base quat {np.round(quat,3)}  "
              f"palm z-axis {np.round(palm_R[:,2],2)}")
    spread = np.max([np.linalg.norm(quats[i] - quats[0]) for i in range(len(quats))])
    print(f"  max quat deviation from rest: {spread:.3f}  "
          f"({'MOVES' if spread > 0.05 else 'STATIC - PROBLEM'})")

    # --- translation (full mode) with a crafted image shift ---
    print("== translation (full) ==")
    wrist2 = WristMapper(rest_pos, rest_quat, mode="full", pos_alpha=1.0, rot_alpha=1.0)
    base_world = synthetic.hand_pose(0.2)
    # calibration image: wrist centered, nominal hand size
    img_cal = np.zeros((21, 2), np.float32)
    img_cal[0] = [0.5, 0.5]                 # wrist center
    img_cal[9] = [0.5, 0.35]                # middle mcp above -> size 0.15
    wrist2.calibrate(base_world, img_cal)
    for label, shift, scale in [("center", [0, 0], 1.0), ("left", [-0.2, 0], 1.0),
                                ("up", [0, -0.2], 1.0), ("closer", [0, 0], 1.6)]:
        img = np.zeros((21, 2), np.float32)
        img[0] = [0.5 + shift[0], 0.5 + shift[1]]
        img[9] = [0.5 + shift[0], 0.5 + shift[1] - 0.15 * scale]
        _, pos, quat = step_and_render(rt, model, data, base_q, base_v, wrist2,
                                       base_world, img, renderer, n=2)
        save(_, f"wrist_full_{label}.png")
        print(f"  {label:7s} base pos {np.round(pos,3)}  (delta from rest "
              f"{np.round(pos-rest_pos,3)})")


if __name__ == "__main__":
    main()
