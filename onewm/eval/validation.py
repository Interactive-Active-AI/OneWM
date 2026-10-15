"""Deterministic action and world-model denoising validation."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from onewm.training.metrics import video_psnr, video_ssim


LOGGER = logging.getLogger(__name__)


def sample_rng_seed(sample_seed: int, dataset_index: int) -> int:
    return (int(sample_seed) * 1_000_003 + int(dataset_index)) % (2**63 - 1)


def _move_batch(
    raw_batch: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in raw_batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.dtype.is_floating_point and value.dtype != dtype:
                value = value.to(dtype)
        batch[key] = value
    return batch


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _validation_loader(dataset, sample_indices: list[int]) -> DataLoader:
    if not sample_indices:
        raise ValueError("validation sample_indices cannot be empty")
    return DataLoader(
        Subset(dataset, sample_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _clean_prefix_raw_frames(model: torch.nn.Module) -> int:
    """Return the raw-frame span represented by the configured clean latents."""
    clean_latents = int(getattr(model, "video_clean_prefix_latent_steps", 1))
    if clean_latents <= 0:
        raise ValueError("video_clean_prefix_latent_steps must be positive")
    temporal_compress = 4
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        vae = backbone.get_vae()
        temporal_compress = int(getattr(vae, "temporal_compress", temporal_compress))
    if temporal_compress <= 0:
        raise ValueError("VAE temporal_compress must be positive")
    return 1 + (clean_latents - 1) * temporal_compress


def _action_state(batch: dict[str, Any]) -> torch.Tensor:
    state_history = batch.get("proprio_history")
    if state_history is not None:
        if state_history.ndim != 3:
            raise ValueError(
                "action validation proprio_history must be [B,N,D], got "
                f"{tuple(state_history.shape)}"
            )
        return state_history

    proprio = batch.get("proprio")
    if proprio is None or proprio.ndim not in (2, 3):
        raise ValueError(
            "action validation requires proprio_history or proprio [B,D]/[B,T,D]"
        )
    return proprio[:, 0, :] if proprio.ndim == 3 else proprio


def _clean_video_input(video: torch.Tensor, clean_raw_frames: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(
            f"validation video must be [B,C,T,H,W], got {tuple(video.shape)}"
        )
    if video.shape[2] < clean_raw_frames:
        raise ValueError(
            "validation video is shorter than its clean prefix: "
            f"{video.shape[2]} < {clean_raw_frames} raw frames"
        )
    if clean_raw_frames == 1:
        return video[:, :, 0]
    return video[:, :, :clean_raw_frames]


@torch.no_grad()
def evaluate_action_denoising(
    model: torch.nn.Module,
    val_dataset,
    *,
    sample_indices: list[int],
    sample_seed: int,
    num_inference_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Denoise actions from pure noise using the real visual/state prefix."""
    model.eval()
    squared_error_sum = 0.0
    element_count = 0
    loader = _validation_loader(val_dataset, sample_indices)
    clean_raw_frames = _clean_prefix_raw_frames(model)

    for offset, (dataset_index, raw_batch) in enumerate(
        zip(sample_indices, loader), start=1
    ):
        batch = _move_batch(raw_batch, device=device, dtype=dtype)
        video = batch["video"]
        target_action = batch["action"]
        context = batch["context"]
        context_mask = batch.get("context_mask")
        action_is_pad = batch.get("action_is_pad")
        proprio = _action_state(batch)
        clean_video = _clean_video_input(video, clean_raw_frames)

        rng_seed = sample_rng_seed(sample_seed, dataset_index)
        with _autocast(device, dtype):
            predicted_action = model.infer_action(
                clean_video,
                context,
                context_mask,
                action_horizon=target_action.shape[1],
                num_inference_steps=num_inference_steps,
                seed=rng_seed,
                num_video_frames=video.shape[2],
                proprio=proprio,
                known_action=None,
                known_action_mask=None,
            )

        diff_sq = (predicted_action.float() - target_action.float()).square()
        if action_is_pad is None:
            squared_error_sum += float(diff_sq.sum().item())
            element_count += int(diff_sq.numel())
        else:
            keep = (~action_is_pad).unsqueeze(-1).expand_as(diff_sq)
            squared_error_sum += float(diff_sq[keep].sum().item())
            element_count += int(keep.sum().item())
        if offset % 8 == 0 or offset == len(sample_indices):
            LOGGER.info(
                "action denoising %d/%d | running_mse=%.6f",
                offset,
                len(sample_indices),
                squared_error_sum / max(element_count, 1),
            )

    if element_count <= 0:
        raise RuntimeError("action validation produced no non-padded action elements")
    return {
        "action_n_samples": len(sample_indices),
        "action_num_inference_steps": int(num_inference_steps),
        "action_initial_condition": "all_noise",
        "action_clean_prefix_tokens": 0,
        "action_clean_prefix_raw_frames": clean_raw_frames,
        "action_state_conditioned": True,
        "action_mse": squared_error_sum / element_count,
    }


@torch.no_grad()
def evaluate_world_denoising(
    model: torch.nn.Module,
    val_dataset,
    *,
    sample_indices: list[int],
    sample_seed: int,
    num_inference_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Denoise future video while pinning every action token to clean GT."""
    model.eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    loader = _validation_loader(val_dataset, sample_indices)
    clean_raw_frames = _clean_prefix_raw_frames(model)

    for offset, (dataset_index, raw_batch) in enumerate(
        zip(sample_indices, loader), start=1
    ):
        batch = _move_batch(raw_batch, device=device, dtype=dtype)
        target_video = batch["video"]
        clean_action = batch["action"]
        context = batch["context"]
        context_mask = batch.get("context_mask")
        image_is_pad = batch.get("image_is_pad")
        clean_video = _clean_video_input(target_video, clean_raw_frames)
        all_clean_mask = torch.ones(
            clean_action.shape[:2], dtype=torch.bool, device=clean_action.device
        )
        rng_seed = sample_rng_seed(sample_seed, dataset_index)
        with _autocast(device, dtype):
            prediction = model.infer_world(
                clean_video,
                context,
                context_mask,
                num_video_frames=target_video.shape[2],
                action_horizon=clean_action.shape[1],
                num_inference_steps=num_inference_steps,
                seed=rng_seed,
                known_action=clean_action,
                known_action_mask=all_clean_mask,
            )
        predicted_video = prediction["video"]
        predicted_action = prediction.get("action")
        if predicted_action is None or not torch.allclose(
            predicted_action.float(), clean_action.float(), atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError("world validation did not keep every action token clean")

        common_frames = min(predicted_video.shape[2], target_video.shape[2])
        if common_frames <= clean_raw_frames:
            continue
        valid_frames = torch.ones(common_frames, dtype=torch.bool, device=device)
        valid_frames[:clean_raw_frames] = False
        if image_is_pad is not None:
            valid_frames &= ~image_is_pad[0, :common_frames].bool()
        if not bool(valid_frames.any().item()):
            continue
        predicted_future = predicted_video[:, :, :common_frames][:, :, valid_frames]
        target_future = target_video[:, :, :common_frames][:, :, valid_frames]
        psnr_values.append(float(video_psnr(predicted_future, target_future).item()))
        ssim_values.append(float(video_ssim(predicted_future, target_future).item()))

        if offset % 2 == 0 or offset == len(sample_indices):
            LOGGER.info(
                "world denoising %d/%d | psnr=%.3f | ssim=%.4f",
                offset,
                len(sample_indices),
                sum(psnr_values) / max(len(psnr_values), 1),
                sum(ssim_values) / max(len(ssim_values), 1),
            )

    if not psnr_values:
        raise RuntimeError("world validation produced no valid future video frames")
    return {
        "video_n_samples": len(psnr_values),
        "video_num_inference_steps": int(num_inference_steps),
        "video_action_condition": "all_clean_ground_truth",
        "video_excludes_input_frame": True,
        "video_excludes_clean_prefix": True,
        "video_clean_prefix_raw_frames": clean_raw_frames,
        "video_psnr": sum(psnr_values) / len(psnr_values),
        "video_ssim": sum(ssim_values) / len(ssim_values),
    }
