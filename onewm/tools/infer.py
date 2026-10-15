"""Run either deployment path with a model-ready NPZ request."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch


def video_to_uint8(video):
    value = video.detach().float().cpu()
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError("expected one video")
        value = value[0]
    if value.ndim != 4 or value.shape[0] != 3 or not torch.isfinite(value).all():
        raise ValueError("video must be finite [3,T,H,W]")
    return (
        value.clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 3, 0).numpy()
    )


def uint8_to_image(frame, device, dtype):
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("image must be HWC uint8 RGB")
    return (
        torch.from_numpy(array.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(127.5)
        .sub(1)
        .to(device, dtype)
    )


def read_request(path, policy):
    """Read explicit tensors; never consult an episode or a reference video."""
    with np.load(path, allow_pickle=False) as request:
        if "input_video" in request:
            video = np.asarray(request["input_video"])
            if (
                video.ndim != 4
                or video.shape[0] != 3
                or not np.isfinite(video).all()
                or np.abs(video).max() > 1.001
            ):
                raise ValueError(
                    "input_video must be finite normalized [3,T,H,W] in [-1,1]"
                )
            image = (
                torch.from_numpy(video.copy())
                .unsqueeze(0)
                .to(policy.device, policy.dtype)
            )
        else:
            image = uint8_to_image(request["image"], policy.device, policy.dtype)
        if "context" in request:
            context = torch.as_tensor(
                request["context"].copy(), device=policy.device, dtype=policy.dtype
            ).unsqueeze(0)
            mask = torch.as_tensor(
                request["context_mask"].copy(), device=policy.device, dtype=torch.bool
            ).unsqueeze(0)
            if (
                context.ndim != 3
                or mask.shape != context.shape[:2]
                or not torch.isfinite(context).all()
            ):
                raise ValueError("context/mask must be finite [L,D] and [L]")
        else:
            context, mask = policy._encode_context(str(request["instruction"].item()))
        proprio = None
        if "proprio" in request:
            proprio = torch.as_tensor(
                request["proprio"].copy(), device=policy.device, dtype=policy.dtype
            ).unsqueeze(0)
            if not torch.isfinite(proprio).all():
                raise ValueError(
                    "proprio must be finite, already normalized model input"
                )
        actions = (
            np.asarray(request["actions_normalized"]).copy()
            if "actions_normalized" in request
            else None
        )
    return image, context, mask, proprio, actions


def load_policy(args):
    from onewm.eval.policy import OneWMPolicy

    return OneWMPolicy(
        args.config,
        args.checkpoint,
        device=args.device,
        num_inference_steps=args.steps,
        seed=args.seed,
        overrides=args.override,
    )


def common_arguments(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input", required=True, help="model-ready NPZ; see docs/deployment.md"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override", nargs="*", default=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--mode", required=True, choices=("wm", "wam"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    policy = load_policy(args)
    image, context, mask, proprio, actions = read_request(args.input, policy)
    horizon = policy.action_horizon
    common = dict(
        action_horizon=horizon,
        num_inference_steps=args.steps,
        seed=args.seed,
        num_video_frames=policy._sampled_video_frame_count(),
    )
    with torch.inference_mode():
        if args.mode == "wam":
            action = policy.model.infer_action(
                image, context, mask, proprio=proprio, **common
            )
            output = {"actions_normalized": action.float().cpu().numpy()[0]}
        else:
            if (
                actions is None
                or actions.shape != (horizon, policy.config.framework.action_dim)
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    "WM requires finite actions_normalized [horizon,action_dim]"
                )
            known = torch.as_tensor(
                actions, device=policy.device, dtype=policy.dtype
            ).unsqueeze(0)
            prediction = policy.model.infer_world(
                image, context, mask, known_action=known, **common
            )
            if not torch.equal(prediction["action"], known):
                raise RuntimeError("world inference changed its known actions")
            output = {
                "frames": video_to_uint8(prediction["video"]),
                "actions_normalized": actions,
            }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **output)
    print(f"OneWM {args.mode}: saved {path}")


if __name__ == "__main__":
    main()
