"""Headless smoke test: synthetic open/fist -> retarget -> offscreen PNGs.

Confirms the geometry pipeline works end to end without a camera. Writes images
to out/ and prints per-pose joint travel and fingertip tracking error.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_teleop import synthetic
from hand_teleop.render import OffscreenRenderer
from hand_teleop.retarget import LeapRetargeter, fingertip_error_mm

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)


def save(img, name):
    path = OUT / name
    if HAVE_PIL:
        Image.fromarray(img).save(path)
    else:
        # Minimal PPM fallback if Pillow is absent.
        path = path.with_suffix(".ppm")
        h, w, _ = img.shape
        with open(path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(img.tobytes())
    return path


def main():
    rt = LeapRetargeter()
    # Calibrate on the open hand so the mapping is snug.
    rt.calibrate(synthetic.hand_pose(0.0))
    rt.reset()
    renderer = OffscreenRenderer(rt.model)

    poses = {"open": 0.0, "half": 0.5, "fist": 1.0}
    print(f"{'pose':6s} {'mean|q|(rad)':>13s} {'tip err (mm) idx/mid/rng/thb':>34s}")
    for name, curl in poses.items():
        lm = synthetic.hand_pose(curl)
        # Iterate a few frames so the warm-started solver settles.
        for _ in range(6):
            q = rt.solve(lm)
        errs = fingertip_error_mm(rt, lm)
        rt.data.qpos[:] = q
        mujoco.mj_forward(rt.model, rt.data)
        img = renderer.frame(rt.data)
        p = save(img, f"smoke_{name}.png")
        print(f"{name:6s} {np.mean(np.abs(q)):13.3f}   "
              f"{errs[0]:6.1f}/{errs[1]:5.1f}/{errs[2]:5.1f}/{errs[3]:5.1f}   -> {p.name}")
        rt.reset()

    # Isolated index-finger curl: only index joints should move.
    rt.reset()
    lm_open = synthetic.hand_pose(0.0)
    for _ in range(6):
        rt.solve(lm_open)
    q_open = rt.q.copy()
    lm_idx = synthetic.hand_pose({"index": 1.0, "middle": 0, "ring": 0, "pinky": 0, "thumb": 0})
    for _ in range(6):
        rt.solve(lm_idx)
    dq = np.abs(rt.q - q_open)
    names = ["if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
             "rf_mcp", "rf_rot", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl"]
    print("\nIsolated index curl - joint deltas (rad):")
    for n, d in zip(names, dq):
        flag = "  <-- moved" if d > 0.1 else ""
        print(f"  {n:8s} {d:5.2f}{flag}")


if __name__ == "__main__":
    main()
