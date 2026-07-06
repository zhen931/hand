# hand

Markerless, vision-based teleoperation of a simulated dexterous hand. A webcam
tracks a human hand and a simulated robot hand mirrors it in real time. The
default hand is the 5-finger anthropomorphic [ORCA hand](https://www.orcahand.com/);
the 4-finger [LEAP hand](https://leaphand.com/) is also supported (`--hand leap`).

## So far

- Webcam -> MediaPipe HandLandmarker -> 21-point 3D hand skeleton.
- Cross-embodiment retargeting from the 21 human keypoints to the robot hand's
  joints, solved natively in MuJoCo (no Pinocchio). Hand-agnostic: adding a hand
  is one config entry (see [hand_teleop/hands.py](hand_teleop/hands.py)).
- The hand mirrors the human hand in a MuJoCo viewer, with occlusion freeze and
  landmark smoothing.
- Wrist orientation: the palm tilts and rolls with the hand (reusing the
  palm-plane rotation the retargeting otherwise discards). Optional 6-DoF adds
  translation from the 2D image landmarks.
- Record / replay of keypoint streams so the sim side can be developed without a
  camera.

Finger articulation plus wrist orientation for now. Wrist translation (6-DoF) is
available but noisy on the depth axis from a single RGB camera; a full arm is
not built yet.

## Architecture

```
[tracker.py]                         [mirror.py]
 webcam --> MediaPipe --> 21 kpts --UDP--> smooth+freeze --> retarget --> MuJoCo hand
            (perception)  protocol.py       mirror.py        retarget.py   (sim)
```

The two halves run as separate processes and talk over a local UDP socket, so
perception and simulation can be developed and profiled
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
python -m hand_teleop.mirror             # terminal 2: MuJoCo hand (ORCA by default)
```

Hold an open, neutral hand toward the camera for the first second so the
retargeter and wrist calibrate, then move your fingers and tilt your hand.

Pick the hand with `--hand`, wrist motion with `--wrist`:

```
python -m hand_teleop.mirror --hand orca      # 5-finger ORCA (default)
python -m hand_teleop.mirror --hand leap      # 4-finger LEAP
python -m hand_teleop.mirror --wrist orient   # rotation only (default)
python -m hand_teleop.mirror --wrist full     # add translation (noisy depth)
python -m hand_teleop.mirror --wrist off      # fingers only, palm fixed
```


## Record / replay

```
python -m hand_teleop.record capture recordings/wave.npz    # Ctrl-C to stop
python -m hand_teleop.record replay  recordings/wave.npz --loop
python -m hand_teleop.mirror                                 # mirror the replay
```

## Measurements

| Step | Number |
|---|---|
| MediaPipe inference | ~10 ms mean, 12 ms p95 |
| Retarget (Gauss-Newton, 8 iters) | < 2 ms |
| Built-in webcam frame rate | ~30 fps (reports 60, delivers 30) |


## Layout

```
hand_teleop/       package: hands, protocol, hand_frame, retarget, wrist, tracker, mirror, record
assets/orca_hand/  vendored ORCA v2 right hand (MIT), default target
assets/leap_hand/  vendored MuJoCo Menagerie LEAP hand
assets/models/     MediaPipe hand_landmarker.task (downloaded, gitignored)
tools/             smoke tests, pipeline test, model downloader
```
