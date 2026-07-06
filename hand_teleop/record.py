"""Record a UDP keypoint stream to disk, and replay it back onto UDP.

The record/replay harness lets the simulation side be developed and regression
tested from a fixed keypoint stream, with no live camera and no second person
in the loop.

    python -m hand_teleop.record capture recordings/wave.npz   # Ctrl-C to stop
    python -m hand_teleop.record replay  recordings/wave.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import RECORDINGS_DIR
from .protocol import DEFAULT_PORT, KeypointFrame, KeypointReceiver, KeypointSender


def capture(path, port=DEFAULT_PORT):
    receiver = KeypointReceiver(port=port)
    rows_meta, rows_world, rows_image = [], [], []
    print(f"[record] capturing on udp:{port} -> {path} (Ctrl-C to stop)")
    try:
        while True:
            f = receiver.latest()
            if f is None:
                time.sleep(0.002)
                continue
            rows_meta.append([f.frame_id, f.t_capture, f.handedness,
                              int(f.tracked), f.confidence])
            rows_world.append(f.world)
            rows_image.append(f.image)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
    meta = np.array(rows_meta, dtype=np.float64)
    world = np.array(rows_world, dtype=np.float32)
    image = np.array(rows_image, dtype=np.float32)
    np.savez_compressed(path, meta=meta, world=world, image=image)
    print(f"[record] saved {len(meta)} frames to {path}")


def replay(path, port=DEFAULT_PORT, realtime=True, loop=False):
    data = np.load(path)
    meta, world, image = data["meta"], data["world"], data["image"]
    sender = KeypointSender(port=port)
    t0_wall = time.time()
    t0_cap = meta[0, 1] if len(meta) else 0.0
    print(f"[record] replaying {len(meta)} frames onto udp:{port} "
          f"(realtime={realtime}, loop={loop})")
    try:
        while True:
            for i in range(len(meta)):
                if realtime and i > 0:
                    target = t0_wall + (meta[i, 1] - t0_cap)
                    dt = target - time.time()
                    if dt > 0:
                        time.sleep(dt)
                sender.send(KeypointFrame(
                    frame_id=int(meta[i, 0]), t_capture=time.time(),
                    handedness=int(meta[i, 2]), tracked=bool(meta[i, 3]),
                    confidence=float(meta[i, 4]), world=world[i], image=image[i],
                ))
            if not loop:
                break
            t0_wall = time.time()
            t0_cap = meta[0, 1]
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
    print("[record] replay done")


def main():
    ap = argparse.ArgumentParser(description="Record/replay keypoint streams")
    ap.add_argument("mode", choices=["capture", "replay"])
    ap.add_argument("path", nargs="?", default=str(RECORDINGS_DIR / "session.npz"))
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--fast", action="store_true", help="replay as fast as possible")
    args = ap.parse_args()
    if args.mode == "capture":
        capture(args.path, port=args.port)
    else:
        replay(args.path, port=args.port, realtime=not args.fast, loop=args.loop)


if __name__ == "__main__":
    main()
