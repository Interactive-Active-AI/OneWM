"""Shared, invertible Franka numeric transforms and aspect-preserving RGB geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F

CAMERAS = ("middle_zed", "left_camera", "right_camera")
VIDEO_OFFSETS = tuple(range(0, 33, 4))
JOINT_DIMS = tuple(range(7)) + tuple(range(8, 15))
GRIPPER_DIMS = (7, 15)
CONTRACT = "franka_filtered_joint16_p25f15_v1"


def normalize(values, stats):
    values = torch.as_tensor(values).float()
    low = torch.as_tensor(stats["q01"], device=values.device, dtype=values.dtype)
    high = torch.as_tensor(stats["q99"], device=values.device, dtype=values.dtype)
    return 2 * (values - low) / (high - low).clamp_min(1e-6) - 1


def denormalize(values, stats):
    values = torch.as_tensor(values).float()
    low = torch.as_tensor(stats["q01"], device=values.device, dtype=values.dtype)
    high = torch.as_tensor(stats["q99"], device=values.device, dtype=values.dtype)
    return (values + 1) * 0.5 * (high - low).clamp_min(1e-6) + low


def letterbox(view, size=224):
    """[T,C,H,W] -> float RGB [T,C,size,size], plus valid y0,y1,x0,x1."""
    if view.ndim != 4 or view.shape[1] != 3:
        raise ValueError(f"expected TCHW RGB, got {view.shape}")
    h, w = view.shape[-2:]
    scale = min(size / h, size / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    image = F.interpolate(
        view.float(),
        size=(rh, rw),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    y, x = (size - rh) // 2, (size - rw) // 2
    return F.pad(image, (x, size - rw - x, y, size - rh - y), value=128), (
        y,
        y + rh,
        x,
        x + rw,
    )


def compose(views):
    if len(views) != 3:
        raise ValueError("Franka requires middle/left/right RGB")
    boxed = [letterbox(view) for view in views]
    # Model contract is C,T,H,W in [-1,1].
    video = torch.cat([view for view, _ in boxed], dim=-1).permute(1, 0, 2, 3)
    return video.div(127.5).sub(1), torch.tensor(
        [roi for _, roi in boxed], dtype=torch.int64
    )
