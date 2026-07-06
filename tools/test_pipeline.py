"""Fast checks that don't need a camera or display:
  1. UDP protocol pack/unpack round-trip is exact.
  2. Live UDP loopback: sender -> receiver delivers the newest frame.
  3. Smoother freezes on low confidence and passes through on high.
"""

import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_teleop import synthetic
from hand_teleop.mirror import LandmarkSmoother
from hand_teleop.protocol import (PACKET_SIZE, KeypointFrame, KeypointReceiver,
                                  KeypointSender)


def test_roundtrip():
    world = synthetic.hand_pose(0.3).astype(np.float32)
    image = np.random.rand(21, 2).astype(np.float32)
    f = KeypointFrame(frame_id=42, t_capture=1234.5, handedness=1, tracked=True,
                      confidence=0.87, world=world, image=image)
    buf = f.pack()
    assert len(buf) == PACKET_SIZE, f"packet size {len(buf)} != {PACKET_SIZE}"
    g = KeypointFrame.unpack(buf)
    assert g.frame_id == 42 and g.handedness == 1 and g.tracked
    assert abs(g.confidence - 0.87) < 1e-6
    assert np.allclose(g.world, world, atol=1e-5)
    assert np.allclose(g.image, image, atol=1e-5)
    print(f"OK  roundtrip: {PACKET_SIZE} B packet, world/image exact")


def test_udp_loopback():
    port = 6199
    rx = KeypointReceiver(port=port)
    tx = KeypointSender(port=port)
    for k in range(5):
        tx.send(KeypointFrame(frame_id=k, t_capture=time.time(), handedness=0,
                              tracked=True, confidence=1.0,
                              world=synthetic.hand_pose(0.0).astype(np.float32),
                              image=np.zeros((21, 2), np.float32)))
    time.sleep(0.05)
    latest = rx.latest()
    assert latest is not None and latest.frame_id == 4, "should keep newest frame"
    assert rx.dropped == 4, f"expected 4 dropped, got {rx.dropped}"
    tx.close(); rx.close()
    print(f"OK  udp loopback: newest frame delivered, {rx.dropped} stale drained")


def test_smoother():
    sm = LandmarkSmoother(alpha=0.5, conf_freeze=0.4)
    hi = KeypointFrame(0, time.time(), 0, True, 0.9,
                       synthetic.hand_pose(0.0).astype(np.float32), np.zeros((21, 2), np.float32))
    lo = KeypointFrame(1, time.time(), 0, True, 0.1,
                       synthetic.hand_pose(1.0).astype(np.float32), np.zeros((21, 2), np.float32))
    out, frozen = sm.update(hi)
    assert out is not None and not frozen
    out2, frozen2 = sm.update(lo)
    assert out2 is None and frozen2, "low confidence must freeze"
    print("OK  smoother: passes on high confidence, freezes on low")


if __name__ == "__main__":
    test_roundtrip()
    test_udp_loopback()
    test_smoother()
    print("\nall pipeline checks passed")
