"""Loss utilities for OneWM training."""

import torch
from torch import Tensor
from onewm.modules.scheduler import FlowMatchScheduler


def flow_matching_loss(
    pred: Tensor,
    target: Tensor,
    timesteps: Tensor,
    scheduler: FlowMatchScheduler,
    is_pad_mask: Tensor | None = None,
    average_only_valid_samples: bool = False,
) -> Tensor:
    """Compute weighted MSE loss with flow-matching importance sampling.

    Args:
        pred: model prediction [B, ...]
        target: training target [B, ...]
        timesteps: timestep indices [B]
        scheduler: FlowMatchScheduler for computing weights
        is_pad_mask: optional bool mask [B, ...] where True = padded (ignore)
        average_only_valid_samples: exclude samples with no valid elements
            from the batch mean instead of treating them as zero-loss samples
    Returns:
        scalar loss
    """
    # Per-element MSE
    mse = (pred - target) ** 2

    # Average over non-batch dims
    reduce_dims = list(range(1, mse.dim()))
    has_valid = torch.ones(mse.shape[0], dtype=torch.bool, device=mse.device)
    if is_pad_mask is not None:
        # Mask out padded positions. Broadcast the mask to the tensor layout.
        # Action tensors use [B, T, D], while video latents use [B, C, T, H, W].
        valid_mask = ~is_pad_mask
        if valid_mask.dim() == 2 and mse.dim() == 5:
            valid_mask = valid_mask[:, None, :, None, None]
        while valid_mask.dim() < mse.dim():
            valid_mask = valid_mask.unsqueeze(-1)
        valid_f = valid_mask.to(mse.dtype).expand_as(mse)
        mse = mse * valid_f
        valid_count = valid_f.sum(dim=reduce_dims)
        has_valid = valid_count > 0
        denom = valid_count.clamp(min=1.0)
        mse = mse.sum(dim=reduce_dims) / denom
    else:
        mse = mse.mean(dim=reduce_dims)

    # Apply per-sample importance weight
    weight = scheduler.training_weight(timesteps).to(mse.device, mse.dtype)
    weighted_mse = mse * weight

    if average_only_valid_samples:
        valid_sample_count = has_valid.to(weighted_mse.dtype).sum().clamp_min(1.0)
        return weighted_mse[has_valid].sum() / valid_sample_count
    return weighted_mse.mean()


def masked_mse_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    average_only_valid_samples: bool = True,
) -> Tensor:
    """MSE over a boolean valid-token mask, with safe empty-batch handling."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if valid_mask.shape != pred.shape[: valid_mask.ndim]:
        raise ValueError(
            "valid_mask must match a leading pred-shape prefix, got "
            f"mask={tuple(valid_mask.shape)}, pred={tuple(pred.shape)}"
        )
    while valid_mask.ndim < pred.ndim:
        valid_mask = valid_mask.unsqueeze(-1)
    valid_f = valid_mask.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
    reduce_dims = tuple(range(1, pred.ndim))
    valid_count = valid_f.sum(dim=reduce_dims)
    per_sample = (((pred - target) ** 2) * valid_f).sum(
        dim=reduce_dims
    ) / valid_count.clamp_min(1.0)
    has_valid = valid_count > 0
    if average_only_valid_samples:
        valid_sample_count = has_valid.to(per_sample.dtype).sum().clamp_min(1.0)
        return per_sample[has_valid].sum() / valid_sample_count
    return per_sample.mean()
