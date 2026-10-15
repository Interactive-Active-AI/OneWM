"""Base world-action model interface."""

from __future__ import annotations

import torch.nn as nn


class WAMModel(nn.Module):
    """Base class for OneWM methods."""

    def training_step(self, sample: dict):
        raise NotImplementedError

    def infer_action(self, *args, **kwargs):
        raise NotImplementedError

    def infer_joint(self, *args, **kwargs):
        raise NotImplementedError

    def infer_world(self, *args, **kwargs):
        raise NotImplementedError
