"""Model-side image preparation for the LIBERO policy server."""

from __future__ import annotations

import numpy as np
import torch

from onewm.data.lerobot import compose_camera_views


def _rgb_clip(image: np.ndarray, name: str) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be HWC RGB, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)


def compose_policy_image(
    primary: np.ndarray,
    wrist: np.ndarray,
    *,
    view_size: tuple[int, int],
    concat_mode: str,
) -> torch.Tensor:
    """Compose one policy image with exactly the offline dataset transform."""

    image = compose_camera_views(
        [_rgb_clip(primary, "primary"), _rgb_clip(wrist, "wrist")],
        concat_mode=concat_mode,
        view_size=view_size,
    )
    return image.float().div(255.0).mul(2.0).sub(1.0).contiguous()
