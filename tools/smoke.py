"""Headless smoke test: synthetic open/fist -> retarget -> offscreen PNGs.

Confirms the geometry pipeline works end to end without a camera, for whichever
hand is selected (default: the configured DEFAULT_HAND). Writes images to out/
and prints per-pose joint travel and fingertip tracking error.

    python tools/smoke.py            # default hand
    python tools/smoke.py leap       # or a named hand
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_teleop import synthetic
from hand_teleop.hands import DEFAULT_HAND, HANDS
from hand_teleop.render import OffscreenRenderer
from hand_teleop.retarget import Retargeter, fingertip_error_mm

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)


def joint_names(rt):
    names = []
    for qadr in rt.qadr:
        for j in range(rt.model.njnt):
            if rt.model.jnt_qposadr[j] == qadr:
                names.append(mujoco.mj_id2name(rt.model, mujoco.mjtObj.mjOBJ_JOINT, j))
                break
    return names


def render(rt, renderer):
    rt.data.qpos[rt.qadr] = rt.q
    mujoco.mj_forward(rt.model, rt.data)
    return renderer.frame(rt.data)


def save(img, name):
    if HAVE_PIL:
        Image.fromarray(img).save(OUT / name)


def main():
    hand = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HAND
    rt = Retargeter(hand)
    rt.calibrate(synthetic.hand_pose(0.0))
    rt.reset()
    renderer = OffscreenRenderer(rt.model)
    fingers = [f.name for f in rt.hand.fingers]

    print(f"hand={rt.hand.name}  fingers={fingers}")
    print(f"{'pose':6s} {'mean|q|(rad)':>13s}   tip err mm per finger")
    for name, curl in {"open": 0.0, "half": 0.5, "fist": 1.0}.items():
        lm = synthetic.hand_pose(curl)
        for _ in range(6):
            rt.solve(lm)
        errs = fingertip_error_mm(rt, lm)
        save(render(rt, renderer), f"smoke_{rt.hand.name}_{name}.png")
        errstr = " ".join(f"{e:4.0f}" for e in errs)
        print(f"{name:6s} {np.mean(np.abs(rt.q)):13.3f}   {errstr}")
        rt.reset()

    # Isolated index curl: only index joints should move.
    rt.reset()
    for _ in range(6):
        rt.solve(synthetic.hand_pose(0.0))
    q_open = rt.q.copy()
    idx_pose = synthetic.hand_pose({"index": 1.0, "middle": 0, "ring": 0,
                                    "pinky": 0, "thumb": 0})
    for _ in range(6):
        rt.solve(idx_pose)
    dq = np.abs(rt.q - q_open)
    print("\nisolated index curl - joint deltas (rad):")
    for n, d in zip(joint_names(rt), dq):
        flag = "  <-- moved" if d > 0.1 else ""
        print(f"  {n:16s} {d:5.2f}{flag}")


if __name__ == "__main__":
    main()
