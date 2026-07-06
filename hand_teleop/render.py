"""Offscreen rendering helper, used for headless smoke tests and screenshots."""

from __future__ import annotations

import mujoco
import numpy as np


class OffscreenRenderer:
    def __init__(self, model, width=640, height=480, azimuth=135, elevation=-25, distance=0.45):
        self.model = model
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.cam.distance = distance
        # Look at the palm region.
        self.cam.lookat[:] = np.array([0.0, 0.0, 0.10])

    def frame(self, data) -> np.ndarray:
        self.renderer.update_scene(data, self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
