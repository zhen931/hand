# hand

Markerless, vision-based teleoperation of a simulated dexterous hand. A webcam
tracks a human hand and a simulated [LEAP hand](https://leaphand.com/) mirrors it
in real time. This is stage 1 of the platform: a zero-cost, all-software loop.

## What works today

- Webcam -> MediaPipe HandLandmarker -> 21-point 3D hand skeleton.
- Cross-embodiment retargeting from the 21 human keypoints to the LEAP hand's
  16 joints, solved natively in MuJoCo (no Pinocchio).
- LEAP hand mirrors the human hand in a MuJoCo viewer, with occlusion freeze and
  landmark smoothing.
- Record / replay of keypoint streams so the sim side can be developed without a
  camera.

Finger articulation only in this stage: keypoints are wrist-relative, so global
hand position and wrist 6-DoF are intentionally out of scope until stage 2.

## Architecture

```
[tracker.py]                         [mirror.py]
 webcam --> MediaPipe --> 21 kpts --UDP--> smooth+freeze --> retarget --> MuJoCo LEAP
            (perception)  protocol.py       mirror.py        retarget.py   (sim)
```

The two halves run as separate processes and talk over a local UDP socket, so
perception (Founder 2) and simulation (Founder 1) can be developed and profiled
independently. The packet format is the interface contract; see
[hand_teleop/protocol.py](hand_teleop/protocol.py).

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python tools/get_model.py         # downloads the MediaPipe hand model (~8 MB)
```

## Run the mirror

Two terminals:

```
python -m hand_teleop.tracker --show     # terminal 1: camera + overlay
python -m hand_teleop.mirror             # terminal 2: MuJoCo LEAP hand
```

Hold an open hand toward the camera for the first second so the retargeter
calibrates, then move your fingers.

### No camera? Everything runs synthetically.

```
python -m hand_teleop.mirror --source synthetic          # self-contained loop
python -m hand_teleop.tracker --synthetic                # stream a test sweep over UDP
python tools/smoke.py                                    # render open/half/fist PNGs
python tools/test_pipeline.py                            # protocol + smoother checks
```

## Record / replay

```
python -m hand_teleop.record capture recordings/wave.npz    # Ctrl-C to stop
python -m hand_teleop.record replay  recordings/wave.npz --loop
python -m hand_teleop.mirror                                 # mirror the replay
```

## Measured on the dev laptop (Windows 11, CPU only)

| Stage | Number |
|---|---|
| MediaPipe inference | ~10 ms mean, 12 ms p95 |
| Retarget (16-DoF Gauss-Newton, 8 iters) | < 2 ms |
| Built-in webcam frame rate | ~30 fps (reports 60, delivers 30) |

The 30 fps cap is a built-in-webcam limitation, not a software one; a ~$40
external UVC camera reaches 60+ fps. The retargeting math and its divergence
from the spec (native MuJoCo instead of dex-retargeting/Pinocchio) is documented
in the [hand_teleop/retarget.py](hand_teleop/retarget.py) module docstring.

## Layout

```
hand_teleop/       package: protocol, hand_frame, retarget, tracker, mirror, record
assets/leap_hand/  vendored MuJoCo Menagerie LEAP hand (+ fingertip/palm sites)
assets/models/     MediaPipe hand_landmarker.task (downloaded, gitignored)
tools/             smoke test, pipeline test, model downloader
```
