"""Simulation layer: keypoints -> smoothing -> retarget -> MuJoCo LEAP hand.

Run (two terminals):
    python -m hand_teleop.tracker --show        # terminal 1
    python -m hand_teleop.mirror                # terminal 2

Or self-contained without a camera / display (used for headless verification):
    python -m hand_teleop.mirror --source synthetic --headless 90

The consumer holds its last pose when tracking confidence drops below a
threshold (the spec's occlusion freeze) and exponentially smooths incoming
landmarks to damp jitter.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import mujoco
import numpy as np

from . import DEFAULT_SCENE
from .protocol import DEFAULT_PORT, KeypointFrame, KeypointReceiver
from .retarget import LeapRetargeter


class LandmarkSmoother:
    """Confidence-gated exponential smoothing with an occlusion freeze."""

    def __init__(self, alpha: float = 0.5, conf_freeze: float = 0.4):
        self.alpha = alpha
        self.conf_freeze = conf_freeze
        self.state: np.ndarray | None = None

    def update(self, frame: KeypointFrame) -> tuple[np.ndarray | None, bool]:
        """Return (smoothed_world_or_None, frozen). None means 'hold pose'."""
        if not frame.tracked or frame.confidence < self.conf_freeze:
            return None, True
        if self.state is None:
            self.state = frame.world.astype(np.float64).copy()
        else:
            self.state = self.alpha * frame.world + (1 - self.alpha) * self.state
        return self.state, False


def _synthetic_stream():
    from . import synthetic

    frames = synthetic.sweep(150)
    i = 0
    while True:
        world = frames[i % len(frames)]
        i += 1
        yield KeypointFrame(frame_id=i, t_capture=time.time(), handedness=0,
                            tracked=True, confidence=1.0, world=world,
                            image=np.zeros((21, 2), np.float32))


def run(source="udp", port=DEFAULT_PORT, headless=0, calibrate_first=True,
        alpha=0.5, out_dir=None):
    rt = LeapRetargeter(DEFAULT_SCENE)
    smoother = LandmarkSmoother(alpha=alpha)
    model, data = rt.model, rt.data

    receiver = gen = None
    if source == "udp":
        receiver = KeypointReceiver(port=port)
    else:
        gen = _synthetic_stream()

    def next_frame():
        if receiver is not None:
            return receiver.latest()
        return next(gen)

    # Optional one-shot calibration on the first well-tracked frame.
    calibrated = not calibrate_first

    lat = deque(maxlen=120)
    step_ns = model.opt.timestep
    substeps = max(1, int(round((1 / 60) / step_ns)))  # ~>=200 Hz actuation

    renderer = frames_written = None
    if headless:
        from .render import OffscreenRenderer
        renderer = OffscreenRenderer(model)
        frames_written = 0
        if out_dir is None:
            from . import REPO_ROOT
            out_dir = REPO_ROOT / "out" / "mirror"
        out_dir.mkdir(parents=True, exist_ok=True)

    viewer = None
    if not headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    print(f"[mirror] source={source} calibrate_first={calibrate_first} "
          f"substeps/frame={substeps}")

    try:
        loop = range(headless) if headless else iter(int, 1)  # finite vs infinite
        for _ in loop:
            frame = next_frame()
            if frame is not None:
                if not calibrated and frame.tracked and frame.confidence >= 0.5:
                    rt.calibrate(frame.world)
                    calibrated = True
                    print("[mirror] calibrated on first tracked frame")
                world, frozen = smoother.update(frame)
                if world is not None:
                    rt.solve(world)
                    lat.append((time.time() - frame.t_capture) * 1000.0)

            data.ctrl[:] = rt.q
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            if headless:
                img = renderer.frame(data)
                try:
                    from PIL import Image
                    Image.fromarray(img).save(out_dir / f"frame_{frames_written:04d}.png")
                except ImportError:
                    pass
                frames_written += 1
            else:
                viewer.sync()
                if lat and frame is not None and frame.frame_id % 30 == 0:
                    print(f"[mirror] end-to-end latency ~{np.mean(lat):5.1f} ms "
                          f"(p95 {np.percentile(lat,95):5.1f})  dropped={receiver.dropped}")
                time.sleep(max(0.0, 1/60 - 0.001))
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
        if receiver is not None:
            receiver.close()
        if lat:
            print(f"[mirror] latency mean={np.mean(lat):.1f} ms "
                  f"p95={np.percentile(lat,95):.1f} ms over {len(lat)} frames")
        if headless:
            print(f"[mirror] wrote {frames_written} frames to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Retargeting + MuJoCo mirror")
    ap.add_argument("--source", choices=["udp", "synthetic"], default="udp")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--headless", type=int, default=0,
                    help="render N frames offscreen instead of opening a viewer")
    ap.add_argument("--alpha", type=float, default=0.5, help="smoothing factor")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    args = ap.parse_args()
    run(source=args.source, port=args.port, headless=args.headless,
        calibrate_first=args.calibrate, alpha=args.alpha)


if __name__ == "__main__":
    main()
