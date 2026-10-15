"""Offline Franka policy/world metrics with exact sample provenance."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from deploy.franka.common import atomic
from onewm.data.franka_transforms import CAMERAS, JOINT_DIMS, denormalize, normalize
from onewm.training.metrics import video_psnr, video_ssim


def sample_seed(seed, sample_id):
    return int.from_bytes(
        hashlib.sha256(f"{seed}/{sample_id}".encode()).digest()[:8], "little"
    ) % (2**63 - 1)


def action_metrics(prediction, sample, stats):
    pred = prediction.detach().float().cpu()
    target = sample["action"].float()
    physical = denormalize(pred, stats["action"])
    raw = sample["raw_action"].float()
    result = {}
    for tag, sl in [
        ("first", slice(0, 1)),
        ("first8", slice(0, 8)),
        ("chunk32", slice(0, 32)),
        ("last", slice(31, 32)),
    ]:
        diff = pred[sl] - target[sl]
        error = physical[sl] - raw[sl]
        result[f"{tag}/normalized_mae"] = diff.abs().mean().item()
        result[f"{tag}/normalized_mse"] = diff.square().mean().item()
        result[f"{tag}/joint_mae_rad"] = error[:, JOINT_DIMS].abs().mean().item()
        result[f"{tag}/joint_mse_rad2"] = error[:, JOINT_DIMS].square().mean().item()
        for arm, dim in [("left", 7), ("right", 15)]:
            result[f"{tag}/{arm}_gripper_mae_m"] = error[:, dim].abs().mean().item()
    result["per_dim_mae"] = (physical - raw).abs().mean(0).tolist()
    hold = sample["raw_state"].float().expand_as(raw)
    hold_norm = normalize(hold, stats["action"])
    result["hold/normalized_mae"] = (hold_norm - target).abs().mean().item()
    result["motion/joint_mean_rad"] = (
        (raw[:, JOINT_DIMS] - hold[:, JOINT_DIMS]).abs().mean().item()
    )
    result["motion_bin"] = (
        "low"
        if result["motion/joint_mean_rad"] < 0.01
        else ("medium" if result["motion/joint_mean_rad"] < 0.05 else "high")
    )
    if not all(
        math.isfinite(v) for v in result.values() if isinstance(v, (int, float))
    ):
        raise FloatingPointError("nonfinite action metrics")
    return result, physical


def summarize_action(records):
    if not records:
        raise ValueError("no action records")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate action sample IDs")
    keys = [k for k, v in records[0]["metrics"].items() if isinstance(v, (int, float))]

    def average(items):
        out = {k: float(np.mean([r["metrics"][k] for r in items])) for k in keys}
        out["n_windows"] = len(items)
        out["per_dim_mae"] = np.mean(
            [r["metrics"]["per_dim_mae"] for r in items], axis=0
        ).tolist()
        for key in list(out):
            if key.endswith("normalized_mse"):
                out[key.replace("mse", "rmse")] = math.sqrt(out[key])
            if key.endswith("joint_mse_rad2"):
                out[key.replace("mse_rad2", "rmse_rad")] = math.sqrt(out[key])
        return out

    tasks = defaultdict(list)
    prompts = defaultdict(list)
    motion = defaultdict(list)
    for r in records:
        tasks[r["task"]].append(r)
        prompts[r["instruction"]].append(r)
        motion[r["metrics"]["motion_bin"]].append(r)
    by_task = {k: average(v) for k, v in sorted(tasks.items())}
    # Deterministic, disjoint raw-frame support within each source episode.
    selected = []
    last = {}
    for r in sorted(records, key=lambda r: r["sample_id"]):
        episode, start = r["sample_id"].rsplit("/", 1)
        if int(start) > last.get(episode, -1):
            selected.append(r)
            last[episode] = int(start) + 32
    return dict(
        n_windows=len(records),
        by_task=by_task,
        by_instruction={k: average(v) for k, v in prompts.items()},
        by_motion={k: average(v) for k, v in motion.items()},
        micro=average(records),
        macro_normalized_mae=float(
            np.mean([r["chunk32/normalized_mae"] for r in by_task.values()])
        ),
        nonoverlapping=average(selected),
        source_episodes=len({r["sample_id"].rsplit("/", 1)[0] for r in records}),
    )


@torch.no_grad()
def evaluate_samples(
    model,
    dataset,
    indices,
    seed=42,
    steps=10,
    world_indices=(),
    output=None,
    lpips_model=None,
    progress=None,
):
    device = next(model.action_expert.parameters()).device
    dtype = next(model.action_expert.parameters()).dtype
    actions = []
    worlds = []
    world_set = set(world_indices)
    for ordinal, index in enumerate(indices):
        sample = dataset[index]
        sid = sample["sample_id"]
        rng = sample_seed(seed, sid)
        video = sample["video"].unsqueeze(0).to(device=device, dtype=dtype)
        context = sample["context"].unsqueeze(0).to(device=device, dtype=dtype)
        mask = sample["context_mask"].unsqueeze(0).to(device)
        proprio = sample["proprio"].unsqueeze(0).to(device=device, dtype=dtype)
        pred = model.infer_action(
            video[:, :, 0],
            context,
            mask,
            action_horizon=32,
            num_inference_steps=steps,
            seed=rng,
            proprio=proprio,
            num_video_frames=9,
            known_action=None,
            known_action_mask=None,
        )[0]
        metrics, physical = action_metrics(pred, sample, dataset.stats)
        row = dict(
            sample_id=sid,
            task=sample["task_name"],
            instruction=sample["instruction"],
            seed=seed,
            metrics=metrics,
        )
        actions.append(row)
        stem = sid.replace("/", "_")
        if output:
            out = Path(output)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out / f"{stem}_action.npz",
                predicted_normalized=pred.float().cpu().numpy(),
                predicted_physical=physical.numpy(),
                target=sample["raw_action"].numpy(),
                state=sample["raw_state"].numpy(),
            )
            atomic(out / f"{stem}_action.json", row)
        if index in world_set:
            action = sample["action"].unsqueeze(0).to(device=device, dtype=dtype)
            world = model.infer_world(
                video[:, :, 0],
                context,
                mask,
                num_video_frames=9,
                action_horizon=32,
                num_inference_steps=steps,
                seed=rng,
                known_action=action,
                known_action_mask=torch.ones((1, 32), device=device, dtype=torch.bool),
            )
            if world["video"].shape != video.shape or not torch.equal(
                world["action"], action
            ):
                raise ValueError("world geometry or clean-action clamp mismatch")
            predicted = world["video"].float().clamp(-1, 1)
            target = sample["video"].unsqueeze(0).to(device=device, dtype=torch.float32)
            camera_metrics = {}
            for camera, roi, offset in zip(
                CAMERAS, sample["valid_rois"], [0, 224, 448]
            ):
                y0, y1, x0, x1 = map(int, roi)
                p = predicted[:, :, 1:, y0:y1, offset + x0 : offset + x1]
                gt = target[:, :, 1:, y0:y1, offset + x0 : offset + x1]
                values = {
                    "psnr": video_psnr(p, gt).item(),
                    "ssim": video_ssim(p, gt).item(),
                }
                if lpips_model is not None:
                    values["lpips_alex"] = (
                        lpips_model(p[0].permute(1, 0, 2, 3), gt[0].permute(1, 0, 2, 3))
                        .mean()
                        .item()
                    )
                if not all(math.isfinite(v) for v in values.values()):
                    raise FloatingPointError(f"nonfinite video metric {sid}")
                camera_metrics[camera] = values
            wrow = dict(
                sample_id=sid,
                task=sample["task_name"],
                instruction=sample["instruction"],
                seed=seed,
                cameras=camera_metrics,
            )
            worlds.append(wrow)
            if output:
                np.savez_compressed(
                    out / f"{stem}_world.npz",
                    prediction=predicted[0].cpu().numpy(),
                    target=sample["video"].numpy(),
                    valid_rois=sample["valid_rois"].numpy(),
                )
                atomic(out / f"{stem}_world.json", wrow)
        if progress and ((ordinal + 1) % 8 == 0 or ordinal + 1 == len(indices)):
            progress(ordinal + 1, len(indices))
    return actions, worlds


def summarize_world(records):
    if not records:
        return {"n_windows": 0}

    def avg(items):
        return {
            camera: {
                key: float(np.mean([r["cameras"][camera][key] for r in items]))
                for key in items[0]["cameras"][camera]
            }
            for camera in CAMERAS
        }

    groups = defaultdict(list)
    for r in records:
        groups[r["task"]].append(r)
    unique_gt = set()
    for row in records:
        episode, start = row["sample_id"].rsplit("/", 1)
        unique_gt.update((episode, int(start) + offset) for offset in range(4, 33, 4))
    return {
        "n_windows": len(records),
        "future_composite_frame_pairs": len(records) * 8,
        "single_view_pairs": len(records) * 24,
        "unique_gt_composite_frames": len(unique_gt),
        "unique_gt_single_view_frames": len(unique_gt) * 3,
        "cameras": avg(records),
        "by_task": {
            k: {"n_windows": len(v), "cameras": avg(v)} for k, v in groups.items()
        },
    }
