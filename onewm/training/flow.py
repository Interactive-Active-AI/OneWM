"""Flow-matching helpers for OneWM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from onewm.modules.scheduler import FlowMatchScheduler


@dataclass(frozen=True)
class PartialCleanActionNoise:
    noisy: Tensor
    target: Tensor
    timesteps: Tensor
    token_timesteps: Tensor
    known_mask: Tensor
    clean_prefix_lengths: Tensor


def aligned_action_boundaries(valid_length: int, num_future_frames: int) -> list[int]:
    """Return proper causal action-prefix boundaries for a runtime geometry."""
    if valid_length < 0:
        raise ValueError(f"valid_length must be non-negative, got {valid_length}")
    if num_future_frames <= 0 or valid_length <= 1:
        return []
    return sorted(
        {
            (frame_idx * valid_length + num_future_frames - 1) // num_future_frames
            for frame_idx in range(1, num_future_frames)
            if 0
            < (frame_idx * valid_length + num_future_frames - 1) // num_future_frames
            < valid_length
        }
    )


def sample_clean_prefix_lengths(
    valid_lengths: Tensor,
    *,
    num_future_frames: int,
    prob_none: float,
    prob_partial: float,
    prob_full: float,
    boundary_ratio: float,
    jitter: int,
    jitter_weights: list[float] | tuple[float, ...],
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample one causal clean-prefix length per batch item.

    Partial samples mix aligned-boundary-plus-jitter draws with uniform draws
    over every proper prefix.  Padding is represented through ``valid_lengths``
    so sampled prefixes never include padded action tokens.
    """
    if valid_lengths.ndim != 1:
        raise ValueError(f"valid_lengths must be [B], got {tuple(valid_lengths.shape)}")
    if num_future_frames < 0:
        raise ValueError(
            f"num_future_frames must be non-negative, got {num_future_frames}"
        )
    probs = (float(prob_none), float(prob_partial), float(prob_full))
    if any(prob < 0.0 or prob > 1.0 for prob in probs) or abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(
            f"clean-prefix probabilities must be in [0, 1] and sum to 1, got {probs}"
        )
    if not 0.0 <= boundary_ratio <= 1.0:
        raise ValueError(f"boundary_ratio must be in [0, 1], got {boundary_ratio}")
    if jitter < 0:
        raise ValueError(f"jitter must be non-negative, got {jitter}")
    if len(jitter_weights) != 2 * jitter + 1:
        raise ValueError(
            f"jitter_weights must have {2 * jitter + 1} entries, got {len(jitter_weights)}"
        )

    device = valid_lengths.device
    if (
        any(float(weight) < 0.0 for weight in jitter_weights)
        or sum(jitter_weights) <= 0.0
    ):
        raise ValueError(
            "jitter_weights must be non-negative with positive total weight"
        )
    weights = torch.as_tensor(jitter_weights, device=device, dtype=torch.float32)

    mode_draw = torch.rand(valid_lengths.shape, device=device, generator=generator)
    partial_cutoff = probs[0] + probs[1]
    prefix_lengths = torch.zeros_like(valid_lengths, dtype=torch.long)
    full_mask = mode_draw >= partial_cutoff
    prefix_lengths[full_mask] = valid_lengths[full_mask].to(torch.long)

    partial_mask = (mode_draw >= probs[0]) & (mode_draw < partial_cutoff)
    partial_possible = partial_mask & (valid_lengths > 1)
    # A proper partial prefix does not exist for length 0/1. Treat these rare
    # samples as full-clean without introducing host/device synchronization.
    degenerate_partial = partial_mask & ~partial_possible
    prefix_lengths[degenerate_partial] = valid_lengths[degenerate_partial]

    batch_size = valid_lengths.shape[0]
    safe_span = (valid_lengths - 1).clamp_min(1)
    uniform_draw = torch.rand((batch_size,), device=device, generator=generator)
    uniform_prefix = (uniform_draw * safe_span.to(uniform_draw.dtype)).floor().to(
        torch.long
    ) + 1

    has_boundary = torch.zeros(batch_size, dtype=torch.bool, device=device)
    selected_boundary = torch.zeros(batch_size, dtype=torch.long, device=device)
    if num_future_frames > 1:
        stages = torch.arange(1, num_future_frames, device=device, dtype=torch.long)
        boundaries = (
            valid_lengths[:, None] * stages[None, :] + num_future_frames - 1
        ) // num_future_frames
        valid_boundaries = (boundaries > 0) & (boundaries < valid_lengths[:, None])
        has_boundary = valid_boundaries.any(dim=1)
        # Argmax of IID uniform scores is a uniform choice among valid runtime
        # boundaries, and works for different valid lengths in the same batch.
        boundary_scores = torch.rand(
            boundaries.shape, device=device, generator=generator
        ).masked_fill(~valid_boundaries, -1.0)
        boundary_index = boundary_scores.argmax(dim=1)
        selected_boundary = boundaries.gather(1, boundary_index[:, None]).squeeze(1)

    boundary_draw = torch.rand((batch_size,), device=device, generator=generator)
    use_boundary = partial_possible & has_boundary & (boundary_draw < boundary_ratio)
    jitter_index = torch.multinomial(
        weights, batch_size, replacement=True, generator=generator
    )
    boundary_prefix = selected_boundary + jitter_index - jitter
    boundary_prefix = torch.maximum(boundary_prefix, torch.ones_like(boundary_prefix))
    boundary_prefix = torch.minimum(boundary_prefix, safe_span)
    selected_prefix = torch.where(use_boundary, boundary_prefix, uniform_prefix)
    prefix_lengths[partial_possible] = selected_prefix[partial_possible]

    return prefix_lengths


def add_partial_clean_action_noise(
    scheduler: FlowMatchScheduler,
    action: Tensor,
    *,
    action_is_pad: Optional[Tensor],
    num_future_frames: int,
    prob_none: float,
    prob_partial: float,
    prob_full: float,
    boundary_ratio: float,
    jitter: int,
    jitter_weights: list[float] | tuple[float, ...],
) -> PartialCleanActionNoise:
    """Apply sample-wise clean-prefix conditioning and suffix flow noise."""
    if action.ndim != 3:
        raise ValueError(f"action must be [B, T, D], got {tuple(action.shape)}")
    batch, horizon, _ = action.shape
    if action_is_pad is None:
        valid_lengths = torch.full(
            (batch,), horizon, device=action.device, dtype=torch.long
        )
    else:
        if action_is_pad.shape != (batch, horizon):
            raise ValueError(
                f"action_is_pad must be {(batch, horizon)}, got {tuple(action_is_pad.shape)}"
            )
        action_is_pad = action_is_pad.to(device=action.device, dtype=torch.bool)
        valid_lengths = (~action_is_pad).sum(dim=1)

    prefix_lengths = sample_clean_prefix_lengths(
        valid_lengths,
        num_future_frames=num_future_frames,
        prob_none=prob_none,
        prob_partial=prob_partial,
        prob_full=prob_full,
        boundary_ratio=boundary_ratio,
        jitter=jitter,
        jitter_weights=jitter_weights,
    )
    positions = torch.arange(horizon, device=action.device).unsqueeze(0)
    known_mask = positions < prefix_lengths.unsqueeze(1)
    known_mask &= positions < valid_lengths.unsqueeze(1)
    if action_is_pad is not None:
        known_mask &= ~action_is_pad

    noise = torch.randn_like(action)
    timesteps = scheduler.sample_timesteps(batch, action.device, action.dtype)
    token_timesteps = timesteps[:, None].expand(batch, horizon).clone()
    token_timesteps.masked_fill_(known_mask, 0.0)
    sigma_tokens = scheduler.timestep_to_sigma(token_timesteps).to(
        device=action.device, dtype=action.dtype
    )
    noisy = (1.0 - sigma_tokens[..., None]) * action + sigma_tokens[..., None] * noise
    target = scheduler.training_target(action, noise, timesteps)
    return PartialCleanActionNoise(
        noisy=noisy,
        target=target,
        timesteps=timesteps,
        token_timesteps=token_timesteps,
        known_mask=known_mask,
        clean_prefix_lengths=prefix_lengths,
    )


def add_flow_noise(
    scheduler: FlowMatchScheduler,
    sample: Tensor,
    pin_first_latent_step: bool = False,
    pin_latent_steps: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    noise = torch.randn_like(sample)
    timesteps = scheduler.sample_timesteps(sample.shape[0], sample.device, sample.dtype)
    noisy = scheduler.add_noise(sample, noise, timesteps)
    steps = (
        int(pin_latent_steps)
        if pin_latent_steps is not None
        else int(bool(pin_first_latent_step) and sample.shape[2] > 1)
    )
    if steps < 0 or steps > sample.shape[2]:
        raise ValueError(
            f"pin_latent_steps must be in [0,{sample.shape[2]}], got {steps}"
        )
    if steps:
        noisy[:, :, :steps] = sample[:, :, :steps]
    target = scheduler.training_target(sample, noise, timesteps)
    return noisy, target, timesteps


def build_inference_schedule(
    scheduler_config, num_inference_steps: int, device, dtype
) -> tuple[Tensor, Tensor]:
    scheduler = FlowMatchScheduler(
        num_train_timesteps=scheduler_config.num_train_timesteps,
        shift=scheduler_config.infer_shift,
    )
    return scheduler.build_inference_schedule(num_inference_steps, device, dtype)


def video_latent_pad_mask(
    image_is_pad: Optional[Tensor],
    latent_steps: int,
    include_initial_video_step: bool,
    temporal_factor: int = 4,
    excluded_prefix_latent_steps: Optional[int] = None,
) -> Optional[Tensor]:
    if image_is_pad is None:
        return None
    if image_is_pad.shape[1] < 1:
        raise ValueError("image_is_pad must contain at least one frame")
    if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
        raise ValueError(
            "Cannot align image_is_pad with video latent steps: "
            f"num_frames={image_is_pad.shape[1]}, temporal_factor={temporal_factor}"
        )
    tail_is_pad = image_is_pad[:, 1:]
    latent_tail_is_pad = tail_is_pad.view(
        image_is_pad.shape[0], -1, temporal_factor
    ).all(dim=2)
    full_video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
    excluded = (
        int(excluded_prefix_latent_steps)
        if excluded_prefix_latent_steps is not None
        else (0 if include_initial_video_step else 1)
    )
    if excluded < 0 or excluded > full_video_is_pad.shape[1]:
        raise ValueError(
            f"excluded_prefix_latent_steps={excluded} is invalid for "
            f"{full_video_is_pad.shape[1]} latent steps"
        )
    video_is_pad = full_video_is_pad[:, excluded:]
    if video_is_pad.shape[1] != latent_steps:
        raise ValueError(
            "Video-loss mask shape mismatch: "
            f"mask steps={video_is_pad.shape[1]}, latent steps={latent_steps}"
        )
    return video_is_pad
