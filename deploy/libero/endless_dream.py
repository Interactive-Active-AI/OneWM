"""An Endless Dream: action prediction followed by clean-action world prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from onewm.tools.infer import (
    common_arguments,
    load_policy,
    read_request,
    uint8_to_image,
    video_to_uint8,
)


@torch.inference_mode()
def imagine(
    model,
    image,
    context,
    context_mask,
    *,
    rounds=16,
    action_horizon=32,
    num_video_frames=9,
    steps=10,
    seed=42,
    proprio=None,
):
    """Work with any compatible model exposing infer_action and infer_world.

    Geometry and weights are supplied by the caller. Only the initial image,
    language and optional fixed normalized state enter this bounded rollout.
    """
    if rounds <= 0 or steps <= 0 or action_horizon <= 0 or num_video_frames < 2:
        raise ValueError(
            "positive rounds, steps, horizon and at least two video frames are required"
        )
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("imagination requires a single initial image [1,3,H,W]")
    frames = [video_to_uint8(image.unsqueeze(2))[0]]
    actions, chunks = [], []
    current = image
    for index in range(rounds):
        common = dict(
            action_horizon=action_horizon,
            num_inference_steps=steps,
            seed=seed,
            num_video_frames=num_video_frames,
        )
        action = model.infer_action(
            current, context, context_mask, proprio=proprio, **common
        )
        if action.shape[:2] != (1, action_horizon) or not torch.isfinite(action).all():
            raise ValueError("invalid predicted action chunk")
        known = action.clone()
        prediction = model.infer_world(
            current,
            context,
            context_mask,
            known_action=known,
            known_action_mask=torch.ones(
                (1, action_horizon), device=action.device, dtype=torch.bool
            ),
            **common,
        )
        if not torch.equal(prediction["action"], known):
            raise RuntimeError("world inference changed the action condition")
        generated = video_to_uint8(prediction["video"])
        if generated.shape != (num_video_frames, image.shape[-2], image.shape[-1], 3):
            raise ValueError("generated video geometry differs from the model request")
        chunks.append(
            {
                "input_frame": frames[-1].copy(),
                "frames": generated,
                "actions_normalized": known.float().cpu().numpy()[0],
            }
        )
        actions.append(chunks[-1]["actions_normalized"])
        frames.extend(generated[1:])
        # Feedback uses lossless RGB, never encoded MP4 or later observations.
        current = uint8_to_image(generated[-1], image.device, image.dtype)
    return {
        "frames": np.stack(frames),
        "actions_normalized": np.concatenate(actions),
        "chunks": chunks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()
    policy = load_policy(args)
    image, context, mask, proprio, _ = read_request(args.input, policy)
    if image.ndim != 4:
        parser.error("the LIBERO imagination loop requires one initial image")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "rollout.npz").exists():
        raise FileExistsError(f"rollout already exists: {output}")
    result = imagine(
        policy.model,
        image,
        context,
        mask,
        rounds=args.rounds,
        action_horizon=policy.action_horizon,
        num_video_frames=policy._sampled_video_frame_count(),
        steps=args.steps,
        seed=args.seed,
        proprio=proprio,
    )
    chunks = result.pop("chunks")
    np.savez_compressed(output / "rollout.npz", **result)
    for index, chunk in enumerate(chunks):
        np.savez_compressed(output / f"chunk_{index:03d}.npz", **chunk)
    from onewm.tools.video import write_mp4

    write_mp4(output / "rollout.mp4", result["frames"], args.fps)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "rounds": args.rounds,
                "predicted_actions": len(result["actions_normalized"]),
                "frames": len(result["frames"]),
                "observations_after_initial": 0,
                "state_feedback": "fixed initial state if supplied",
                "simulator_execution": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"OneWM An Endless Dream: saved {output}")


if __name__ == "__main__":
    main()
