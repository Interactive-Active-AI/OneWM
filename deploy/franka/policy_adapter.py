"""Offline dual-Franka target prediction from any compatible checkpoint."""

import numpy as np
import torch

from onewm.data.franka_transforms import CAMERAS, compose
from onewm.eval.policy import OneWMPolicy


class FrankaPolicy:
    def __init__(self, config, checkpoint, device="cuda:0", overrides=None):
        self.policy = OneWMPolicy(
            config, checkpoint, device=device, overrides=overrides
        )
        cfg = self.policy.config
        if cfg.data.dataset_type != "franka_v3" or cfg.framework.action_dim != 16:
            raise ValueError("expected a Franka absolute-joint OneWM configuration")

    def predict(self, images, state, instruction, seed=42):
        if set(images) != set(CAMERAS):
            raise ValueError(f"images require {CAMERAS}")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("expected finite 16D absolute state")
        views = []
        for camera in CAMERAS:
            array = np.asarray(images[camera])
            if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                raise ValueError("each camera must be HWC uint8 RGB")
            views.append(torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0))
        video, _ = compose(views)
        self.policy.seed = seed
        return self.policy.predict_chunk(
            video[:, 0].unsqueeze(0), state[None, :], instruction
        )
