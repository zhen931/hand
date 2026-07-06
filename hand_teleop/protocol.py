"""Keypoint frame definition and UDP transport.

This is the interface contract between the perception process (tracker) and
the simulation process (mirror). One packet per camera frame, fire and forget.

Packet layout, little endian, 442 bytes total:

    u32   magic           0x48545031 ("HTP1")
    u32   frame_id        monotonically increasing per tracker session
    f64   t_capture       UNIX time, stamped immediately after camera read
    u8    handedness      0 = right, 1 = left
    u8    tracked         1 if a hand was detected in this frame
    f32   confidence      detector confidence in [0, 1], 0 when not tracked
    f32[63]  world        21 x 3 metric landmarks in metres (MediaPipe world
                          coordinates, origin near the hand centroid)
    f32[42]  image        21 x 2 normalized image coordinates in [0, 1]

When tracked is 0 the landmark payload repeats the last known values and the
consumer is expected to hold its current pose (occlusion freeze).
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x48545031
_FMT = "<IId2Bf63f42f"
PACKET_SIZE = struct.calcsize(_FMT)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6060

RIGHT = 0
LEFT = 1


@dataclass
class KeypointFrame:
    frame_id: int
    t_capture: float
    handedness: int
    tracked: bool
    confidence: float
    world: np.ndarray  # (21, 3) float32, metres
    image: np.ndarray  # (21, 2) float32, normalized

    def pack(self) -> bytes:
        return struct.pack(
            _FMT,
            MAGIC,
            self.frame_id,
            self.t_capture,
            self.handedness,
            1 if self.tracked else 0,
            self.confidence,
            *self.world.astype(np.float32).ravel(),
            *self.image.astype(np.float32).ravel(),
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "KeypointFrame":
        vals = struct.unpack(_FMT, buf)
        if vals[0] != MAGIC:
            raise ValueError(f"bad magic {vals[0]:#x}")
        world = np.array(vals[6:69], dtype=np.float32).reshape(21, 3)
        image = np.array(vals[69:111], dtype=np.float32).reshape(21, 2)
        return cls(
            frame_id=vals[1],
            t_capture=vals[2],
            handedness=vals[3],
            tracked=bool(vals[4]),
            confidence=vals[5],
            world=world,
            image=image,
        )


class KeypointSender:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame: KeypointFrame) -> None:
        self.sock.sendto(frame.pack(), self.addr)

    def close(self) -> None:
        self.sock.close()


class KeypointReceiver:
    """Non-blocking receiver that drains the socket and keeps only the newest frame."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.dropped = 0

    def latest(self) -> KeypointFrame | None:
        newest = None
        while True:
            try:
                buf, _ = self.sock.recvfrom(PACKET_SIZE)
            except BlockingIOError:
                break
            if len(buf) != PACKET_SIZE:
                continue
            if newest is not None:
                self.dropped += 1
            newest = buf
        return KeypointFrame.unpack(newest) if newest else None

    def close(self) -> None:
        self.sock.close()
