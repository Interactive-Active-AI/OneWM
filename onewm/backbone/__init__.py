"""Backbone builders for OneWM."""

from __future__ import annotations

from pathlib import Path

import torch

from onewm.backbone.base import BaseBackbone, BackboneInfo
from onewm.config import BackboneConfig


WAN_BACKBONES = {"wan22_5b", "wan22_14b"}


def build_backbone(
    config: BackboneConfig,
    device: str = "cpu",
    dtype=None,
    *,
    load_dit: bool = True,
) -> BaseBackbone:
    """Build a OneWM backbone."""

    if dtype is None:
        dtype = torch.bfloat16
    if config.type in WAN_BACKBONES:
        from onewm.backbone.wan22 import Wan22Backbone

        return Wan22Backbone(
            config,
            device=device,
            dtype=dtype,
            load_dit=load_dit,
            load_text_encoder=getattr(config, "load_text_encoder", False),
        )
    raise ValueError(
        f"Unknown OneWM backbone type: {config.type}. "
        f"Available: {sorted(WAN_BACKBONES)}"
    )


def build_text_encoder(
    config: BackboneConfig,
    *,
    text_len: int,
    device: str = "cpu",
    dtype=None,
    model_dir=None,
):
    """Build the standalone T5 text encoder matching ``config.type``.

    Returns an object exposing ``encode(list[str]) -> (context, mask)``. Shared by
    eval (``onewm.eval.policy``), training cache warm-up (``builder``), and the
    precompute CLI.
    """
    if dtype is None:
        dtype = torch.bfloat16
    if model_dir is None:
        model_dir = config.pretrained_model_id
    model_dir = Path(model_dir)

    if config.type in WAN_BACKBONES:
        from onewm.backbone.wan22 import Wan22TextEncoder

        return Wan22TextEncoder(
            ckpt_path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(model_dir / "google" / "umt5-xxl"),
            text_len=text_len,
            device=device,
            dtype=dtype,
        )
    raise ValueError(
        f"Unknown OneWM backbone type: {config.type}. "
        f"Available: {sorted(WAN_BACKBONES)}"
    )


__all__ = [
    "BaseBackbone",
    "BackboneInfo",
    "Wan22Backbone",
    "build_backbone",
    "build_text_encoder",
]


def __getattr__(name):
    if name == "Wan22Backbone":
        from onewm.backbone.wan22 import Wan22Backbone

        return Wan22Backbone
    raise AttributeError(name)
