"""Simulation layer: keypoints -> smoothing -> retarget -> MuJoCo hand.

Run (two terminals):
    python -m hand_teleop.tracker --show        # terminal 1
    python -m hand_teleop.mirror                # terminal 2

Or self-contained without a camera / display (used for headless verification):
    python -m hand_teleop.mirror --source synthetic --headless 90

The displayed hand has a free-jointed floating base (for the wrist) and gravity
off, so we set its pose kinematically each frame: finger joint angles from the
retargeter, base pose from the wrist mapper. The consumer holds its last pose
when tracking confidence drops below a threshold (the occlusion freeze) and
exponentially smooths incoming landmarks to damp jitter.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import mujoco
import numpy as np

from .hands import DEFAULT_HAND, HANDS, build_model
from .protocol import DEFAULT_PORT, KeypointFrame, KeypointReceiver
from .retarget import Retargeter
from .wrist import WristMapper


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
        alpha=0.5, wrist_mode="orient", hand=DEFAULT_HAND, out_dir=None):
    cfg = HANDS[hand] if isinstance(hand, str) else hand
    rt = Retargeter(cfg)
    smoother = LandmarkSmoother(alpha=alpha)
    model, data, info = build_model(cfg, floating=True)
    base_q, base_v = info.base_qadr, info.base_vadr
    fq = info.finger_qadr
    rest_pos = model.qpos0[base_q:base_q + 3].copy()
    rest_quat = model.qpos0[base_q + 3:base_q + 7].copy()
    wrist = WristMapper(rest_pos, rest_quat, rt.align, mode=wrist_mode)
    base_pos, base_quat = rest_pos.copy(), rest_quat.copy()

    receiver = gen = None
    if source == "udp":
        receiver = KeypointReceiver(port=port)
    else:
        gen = _synthetic_stream()

    def next_frame():
        if receiver is not None:
            return receiver.latest()
        return next(gen)

    # Optional one-shot calibration on the first well-tracked frame. Pressing 'c'
    # in the viewer re-arms it (re-snap the open-hand scale and translation zero).
    calibrated = not calibrate_first
    recal = {"pending": False}

    lat = deque(maxlen=120)

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

        def key_callback(keycode):
            if keycode in (ord("C"), ord("c")):
                recal["pending"] = True

        viewer = mj_viewer.launch_passive(model, data, key_callback=key_callback)
        # Front view matching the MediaPipe-to-world mapping, so the robot reads
        # as a mirror of the hand. The user can still orbit freely.
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = 1.4 * model.stat.extent
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -12

    print(f"[mirror] hand={cfg.name} source={source} "
          f"calibrate_first={calibrate_first} wrist={wrist_mode}")

    try:
        loop = range(headless) if headless else iter(int, 1)  # finite vs infinite
        for _ in loop:
            frame = next_frame()
            if frame is not None:
                if recal["pending"]:
                    calibrated, recal["pending"] = False, False
                if not calibrated and frame.tracked and frame.confidence >= 0.5:
                    rt.calibrate(frame.world)
                    wrist.calibrate(frame.world, frame.image)
                    calibrated = True
                    print("[mirror] calibrated (hold an open, neutral hand; press c to redo)")
                world, frozen = smoother.update(frame)
                if world is not None:
                    rt.solve(world)
                    base_pos, base_quat = wrist.pose(world, frame.image)
                    lat.append((time.time() - frame.t_capture) * 1000.0)

            # Kinematic mirror: place finger joints and the floating base directly.
            data.qpos[fq] = rt.q
            data.qpos[base_q:base_q + 3] = base_pos
            data.qpos[base_q + 3:base_q + 7] = base_quat
            mujoco.mj_forward(model, data)

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
                    dropped = receiver.dropped if receiver is not None else 0
                    print(f"[mirror] end-to-end latency ~{np.mean(lat):5.1f} ms "
                          f"(p95 {np.percentile(lat,95):5.1f})  dropped={dropped}")
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
    ap.add_argument("--wrist", choices=["off", "orient", "full"], default="orient",
                    help="base motion: off, orientation only, or 6-DoF")
    ap.add_argument("--hand", choices=list(HANDS), default=DEFAULT_HAND)
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    args = ap.parse_args()
    run(source=args.source, port=args.port, headless=args.headless,
        calibrate_first=args.calibrate, alpha=args.alpha, wrist_mode=args.wrist,
        hand=args.hand)


if __name__ == "__main__":
    main()
