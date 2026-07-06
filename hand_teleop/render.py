"""Offscreen rendering helper, used for headless smoke tests and screenshots."""

from __future__ import annotations

import mujoco
import numpy as np


class OffscreenRenderer:
    def __init__(self, model, width=640, height=480, azimuth=140, elevation=-20,
                 distance=None):
        self.model = model
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        # Auto-frame from the model's own statistics so any hand fits.
        self.cam.lookat[:] = model.stat.center
        self.cam.distance = distance if distance is not None else 1.4 * model.stat.extent

    def frame(self, data) -> np.ndarray:
        self.renderer.update_scene(data, self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
