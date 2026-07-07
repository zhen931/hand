"""Perception layer: webcam -> MediaPipe Hands -> 21 keypoints -> UDP.

Run:
    python -m hand_teleop.tracker                 # live webcam
    python -m hand_teleop.tracker --show          # with an on-screen overlay
    python -m hand_teleop.tracker --synthetic     # no camera, streams a test sweep

t_capture is stamped immediately after the camera read so downstream latency
measurements start at the true photon-to-sample instant.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import ASSETS_DIR
from .protocol import DEFAULT_PORT, RIGHT, LEFT, KeypointFrame, KeypointSender

HAND_MODEL = ASSETS_DIR / "models" / "hand_landmarker.task"

# 21-point MediaPipe hand skeleton, for the manual overlay.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]


def _run_synthetic(sender: KeypointSender, fps: float) -> None:
    from . import synthetic

    frames = synthetic.sweep(150)
    print(f"[tracker] synthetic mode: streaming {len(frames)} frames on a loop")
    fid = 0
    dt = 1.0 / fps
    while True:
        for world in frames:
            image = np.zeros((21, 2), dtype=np.float32)  # no image plane in synthetic
            frame = KeypointFrame(
                frame_id=fid, t_capture=time.time(), handedness=RIGHT,
                tracked=True, confidence=1.0, world=world, image=image,
            )
            sender.send(frame)
            fid += 1
            time.sleep(dt)


def _draw_overlay(cv2, bgr, image_lm, w, h):
    for a, b in HAND_CONNECTIONS:
        pa = (int(image_lm[a, 0] * w), int(image_lm[a, 1] * h))
        pb = (int(image_lm[b, 0] * w), int(image_lm[b, 1] * h))
        cv2.line(bgr, pa, pb, (0, 200, 0), 2)
    for i in range(21):
        p = (int(image_lm[i, 0] * w), int(image_lm[i, 1] * h))
        cv2.circle(bgr, p, 3, (0, 0, 255), -1)


def _run_live(sender: KeypointSender, camera: int, width: int, height: int,
              want_fps: int, show: bool, mirror: bool) -> None:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not HAND_MODEL.exists():
        raise FileNotFoundError(
            f"hand model missing: {HAND_MODEL}\n"
            "download it once with tools/get_model.py")

    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, want_fps)
    # Keep only the newest frame so read() never returns a buffered stale one.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera}")

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        num_hands=1, running_mode=mp_vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    actual = cap.get(cv2.CAP_PROP_FPS)
    print(f"[tracker] camera {camera} @ {int(cap.get(3))}x{int(cap.get(4))} "
          f"reporting {actual:.0f} fps; press q in the window to quit")

    fid, last_world, last_image = 0, np.zeros((21, 3), np.float32), np.zeros((21, 2), np.float32)
    ema_dt = None
    t_start = time.time()
    while True:
        ok, bgr = cap.read()
        t_cap = time.time()
        if not ok:
            continue
        if mirror:
            bgr = cv2.flip(bgr, 1)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((t_cap - t_start) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        tracked = bool(result.hand_world_landmarks)
        conf, handed = 0.0, RIGHT
        if tracked:
            wl = result.hand_world_landmarks[0]
            il = result.hand_landmarks[0]
            last_world = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)
            last_image = np.array([[p.x, p.y] for p in il], dtype=np.float32)
            cat = result.handedness[0][0]
            conf = float(cat.score)
            # Handedness label is from the camera's view; with a mirrored image
            # the user's right hand reads as "Left". Report the anatomical side.
            handed = RIGHT if cat.category_name == "Left" else LEFT

        sender.send(KeypointFrame(
            frame_id=fid, t_capture=t_cap, handedness=handed,
            tracked=tracked, confidence=conf, world=last_world, image=last_image,
        ))
        fid += 1

        dt = time.time() - t_cap
        ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt
        if show:
            if tracked:
                _draw_overlay(cv2, bgr, last_image, w, h)
            cv2.putText(bgr, f"proc {ema_dt*1000:4.0f} ms  conf {conf:.2f}  "
                             f"{'TRACK' if tracked else 'LOST '}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("tracker", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand keypoint tracker -> UDP")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--show", action="store_true", help="show camera window with overlay")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false",
                    help="do not horizontally flip the camera image")
    ap.add_argument("--synthetic", action="store_true", help="stream a synthetic sweep")
    args = ap.parse_args()

    sender = KeypointSender(args.host, args.port)
    try:
        if args.synthetic:
            _run_synthetic(sender, fps=args.fps)
        else:
            _run_live(sender, args.camera, args.width, args.height,
                      args.fps, args.show, args.mirror)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


if __name__ == "__main__":
    main()
