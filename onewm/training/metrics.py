"""Evaluation metrics for OneWM frameworks.

Core validation metrics:

- ``action_mse``: per-dim mean squared error between predicted action chunk
  and ground-truth action chunk. Returns a ``float`` (mean over batch /
  time / dim) and an optional ``[T, D]`` per-step breakdown.
- ``video_psnr``: PSNR (dB) between predicted video pixels and ground-truth
  video pixels. Inputs are assumed to be in ``[-1, 1]`` (Wan2.2 convention)
  or ``[0, 1]`` — pass ``data_range`` accordingly.
- ``video_ssim``: windowed structural similarity over decoded RGB frames.

Both metrics tolerate ``is_pad`` masks (broadcastable to the metric tensor
shape; ``True`` means *padded position, exclude from metric*).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _apply_pad_mask(diff_sq: Tensor, is_pad: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """Multiply ``diff_sq`` by ``(1 - is_pad)`` and return (sum, count)."""
    if is_pad is None:
        return diff_sq.sum(), torch.tensor(
            float(diff_sq.numel()), device=diff_sq.device, dtype=diff_sq.dtype
        )
    keep = (~is_pad).to(diff_sq.dtype)
    while keep.dim() < diff_sq.dim():
        keep = keep.unsqueeze(-1)
    keep = keep.expand_as(diff_sq)
    return (diff_sq * keep).sum(), keep.sum().clamp_min(1.0)


@torch.no_grad()
def action_mse(
    pred: Tensor,
    target: Tensor,
    is_pad: Optional[Tensor] = None,
    per_step: bool = False,
) -> dict[str, Tensor]:
    """Compute action MSE.

    Args:
        pred:   ``[B, T, D]`` predicted actions.
        target: ``[B, T, D]`` ground-truth actions.
        is_pad: optional ``[B, T]`` boolean mask (True = padded).
        per_step: if True also returns the per-step ``[T]`` MSE.

    Returns:
        dict with keys: ``mse`` (scalar) and optionally ``mse_per_step``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"action_mse: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    diff_sq = (pred.float() - target.float()) ** 2
    total, count = _apply_pad_mask(diff_sq, is_pad)
    out = {"mse": total / count}
    if per_step:
        # Per-step MSE: average over batch and dim, masking out padded steps.
        if is_pad is None:
            out["mse_per_step"] = diff_sq.mean(dim=(0, 2))
        else:
            keep = (~is_pad).to(diff_sq.dtype).unsqueeze(-1).expand_as(diff_sq)
            num = (diff_sq * keep).sum(dim=(0, 2))
            den = keep.sum(dim=(0, 2)).clamp_min(1.0)
            out["mse_per_step"] = num / den
    return out


def action_dim_mse(
    pred: Tensor,
    target: Tensor,
    is_pad: Optional[Tensor] = None,
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"action_dim_mse: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    diff_sq = (pred.float() - target.float()) ** 2
    if is_pad is None:
        return diff_sq.mean(dim=(0, 1))
    keep = (~is_pad).to(diff_sq.dtype).unsqueeze(-1).expand_as(diff_sq)
    num = (diff_sq * keep).sum(dim=(0, 1))
    den = keep.sum(dim=(0, 1)).clamp_min(1.0)
    return num / den


def resolve_action_gripper_dims(
    action_dim: int,
    *,
    gripper_dim: int = -1,
    gripper_dims: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    """Resolve possibly-negative gripper dimensions to unique indices."""
    raw_dims = [gripper_dim] if gripper_dims is None else list(gripper_dims)
    if not raw_dims:
        raise ValueError("gripper_dims must contain at least one dimension")

    resolved = tuple(dim if dim >= 0 else action_dim + dim for dim in raw_dims)
    if any(dim < 0 or dim >= action_dim for dim in resolved):
        raise ValueError(
            f"gripper dims {raw_dims} out of range for action dim={action_dim}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"gripper dims resolve to duplicate indices: {resolved}")
    return resolved


def action_monitor_metrics(
    pred: Tensor,
    target: Tensor,
    action: Tensor,
    is_pad: Optional[Tensor] = None,
    gripper_dim: int = -1,
    gripper_dims: Optional[Sequence[int]] = None,
) -> dict[str, float]:
    dim_mse = action_dim_mse(pred, target, is_pad=is_pad)
    gripper_indices = resolve_action_gripper_dims(
        dim_mse.shape[0],
        gripper_dim=gripper_dim,
        gripper_dims=gripper_dims,
    )

    eef_mask = torch.ones(dim_mse.shape[0], dtype=torch.bool, device=dim_mse.device)
    eef_mask[list(gripper_indices)] = False
    action_f = action.float()
    gripper_values = []
    for gripper_idx in gripper_indices:
        if is_pad is None:
            values = action_f[..., gripper_idx].reshape(-1)
        else:
            values = action_f[..., gripper_idx][~is_pad]
            if values.numel() == 0:
                values = action_f[..., gripper_idx].reshape(-1)
        gripper_values.append(values)

    combined_values = torch.cat(gripper_values)
    result = {
        "loss_action_eef": float(dim_mse[eef_mask].mean().detach().item()),
        "loss_action_gripper": float(
            dim_mse[list(gripper_indices)].mean().detach().item()
        ),
        "action_target_gripper_mean": float(combined_values.mean().detach().item()),
        "action_target_gripper_open_rate": float(
            (combined_values > 0).float().mean().detach().item()
        ),
    }
    labels = (
        ("left", "right")
        if len(gripper_indices) == 2
        else tuple(f"dim_{idx}" for idx in gripper_indices)
    )
    for label, gripper_idx, values in zip(labels, gripper_indices, gripper_values):
        result[f"loss_action_gripper_{label}"] = float(
            dim_mse[gripper_idx].detach().item()
        )
        result[f"action_target_gripper_{label}_mean"] = float(
            values.mean().detach().item()
        )
        result[f"action_target_gripper_{label}_open_rate"] = float(
            (values > 0).float().mean().detach().item()
        )
    return result


@torch.no_grad()
def video_psnr(
    pred: Tensor,
    target: Tensor,
    data_range: float = 2.0,
    reduce: str = "mean",
) -> Tensor:
    """Compute PSNR (dB) between predicted and target video tensors.

    Args:
        pred / target: ``[B, C, T, H, W]`` (or any matching shape). Both are
            expected in the same range; defaults assume ``[-1, 1]`` so
            ``data_range=2.0``. Use ``data_range=1.0`` for ``[0, 1]`` data.
        reduce: ``mean`` returns a scalar; ``per_sample`` returns ``[B]``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"video_psnr: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    diff_sq = (pred.float() - target.float()) ** 2
    if reduce == "per_sample":
        # Reduce over all non-batch dims.
        dims = tuple(range(1, diff_sq.dim()))
        mse = diff_sq.mean(dim=dims).clamp_min(1e-12)
    else:
        mse = diff_sq.mean().clamp_min(1e-12)
    psnr = 10.0 * torch.log10((data_range**2) / mse)
    return psnr


@torch.no_grad()
def video_ssim(
    pred: Tensor,
    target: Tensor,
    data_range: float = 2.0,
    reduce: str = "mean",
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Compute Gaussian-window SSIM over decoded ``[B,C,T,H,W]`` videos."""
    if pred.shape != target.shape:
        raise ValueError(
            f"video_ssim: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    if pred.ndim != 5:
        raise ValueError(f"video_ssim expects [B,C,T,H,W], got {tuple(pred.shape)}")
    if reduce not in {"mean", "per_sample"}:
        raise ValueError(f"unsupported video_ssim reduction: {reduce!r}")

    batch, channels, frames, height, width = pred.shape
    size = min(int(window_size), int(height), int(width))
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)
    coordinates = torch.arange(size, device=pred.device, dtype=torch.float32)
    coordinates = coordinates - (size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(channels, 1, size, size).contiguous()

    pred_frames = (
        pred.float().permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
    )
    target_frames = (
        target.float().permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
    )
    mu_pred = F.conv2d(pred_frames, kernel, groups=channels)
    mu_target = F.conv2d(target_frames, kernel, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target
    var_pred = F.conv2d(pred_frames.square(), kernel, groups=channels) - mu_pred_sq
    var_target = (
        F.conv2d(target_frames.square(), kernel, groups=channels) - mu_target_sq
    )
    covariance = (
        F.conv2d(pred_frames * target_frames, kernel, groups=channels) - mu_cross
    )

    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    numerator = (2 * mu_cross + c1) * (2 * covariance + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (var_pred + var_target + c2)
    frame_scores = (numerator / denominator.clamp_min(1e-12)).mean(dim=(1, 2, 3))
    sample_scores = frame_scores.reshape(batch, frames).mean(dim=1)
    return sample_scores if reduce == "per_sample" else sample_scores.mean()
