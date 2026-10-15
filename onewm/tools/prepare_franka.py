"""Audit immutable Franka v3 data and prepare an explicit leakage-free split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
import torch

from deploy.franka.common import ASSETS, atomic, status
from onewm.data.franka_transforms import CAMERAS, CONTRACT, compose
from onewm.data.franka_v3 import (
    DATA_ROOT,
    REVISION,
    TASKS,
    data_path,
    decode_frames,
    digest,
    episode_rows,
    read_records,
    video_location,
)


def apportion(count, weights):
    raw = np.asarray(weights, dtype=np.float64) * count / sum(weights)
    values = np.floor(raw).astype(int)
    order = sorted(range(len(values)), key=lambda i: (-(raw[i] - values[i]), i))
    for i in order[: count - int(values.sum())]:
        values[i] += 1
    return values.tolist()


def make_split(records, root, seed=42, fraction=0.005):
    rng = random.Random(seed)
    counts = [
        sum(
            max(0, int(r["length"]) - 32) for r in records.values() if r["task"] == task
        )
        for task in TASKS
    ]
    n_val = round(sum(counts) * fraction)
    quotas = apportion(n_val, counts)
    segments = []
    for task, quota in zip(TASKS, quotas):
        groups = defaultdict(list)
        for key, record in sorted(records.items()):
            if record["task"] == task:
                groups[record["tasks"][0]].append(key)
        # Equal instruction coverage within the drawer task.
        per_prompt = apportion(quota, [1] * len(groups))
        for (prompt, keys), n in zip(sorted(groups.items()), per_prompt):
            if n < 1:
                raise ValueError("validation fraction cannot cover all instructions")
            keys = [key for key in keys if int(records[key]["length"]) >= n + 97]
            if not keys:
                raise ValueError(
                    "no episode large enough for an interior holdout block"
                )
            key = rng.choice(keys)
            # At least 32 valid starts on either side for deterministic purge.
            first = rng.randint(32, int(records[key]["length"]) - n - 65)
            segments.append(
                dict(
                    episode=key,
                    first=first,
                    count=n,
                    prompt=prompt,
                    source_rows=[first, first + n + 32],
                )
            )
    segments_by_ep = {s["episode"]: s for s in segments}
    train, val, excluded = [], [], []
    for key, record in sorted(records.items()):
        s = segments_by_ep.get(key)
        for start in range(int(record["length"]) - 32):
            item = [key, start]
            if s and s["first"] <= start < s["first"] + s["count"]:
                val.append(item)
            elif s and start < s["source_rows"][1] and start + 33 > s["source_rows"][0]:
                excluded.append(item)
            else:
                train.append(item)
    if len(val) != n_val:
        raise ValueError("validation cardinality mismatch")
    return dict(
        contract=CONTRACT,
        revision=REVISION,
        root=str(root),
        seed=seed,
        val_fraction_requested=fraction,
        val_fraction_actual=len(val) / sum(counts),
        candidate_windows=sum(counts),
        task_quotas=dict(zip(TASKS, quotas)),
        segments=segments,
        train_windows=train,
        val_windows=val,
        excluded_windows=excluded,
    )


def source_masks(records, windows):
    masks = {key: np.zeros(int(r["length"]), dtype=bool) for key, r in records.items()}
    for key, start in windows:
        masks[key][start : start + 33] = True
    return masks


def quantile_stats(values):
    values = np.concatenate(values).astype(np.float64)
    return {
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def prepare(root, output, video_audit=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "prepared.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        for name, expected in previous["asset_sha256"].items():
            if digest(output / name) != expected:
                raise ValueError(f"frozen asset changed: {name}")
        print("Prepared assets already complete and fingerprints match", flush=True)
        return
    records = read_records(root)
    manifest = make_split(records, root)
    train_masks, val_masks = (
        source_masks(records, manifest["train_windows"]),
        source_masks(records, manifest["val_windows"]),
    )
    for key in records:
        if np.any(train_masks[key] & val_masks[key]):
            raise ValueError(f"train/val source overlap: {key}")
    # The same physical recording must never silently occur twice.
    provenance = {}
    for task in TASKS:
        items = json.loads((root / task / "meta/episode_provenance.json").read_text())[
            "episodes"
        ]
        for item in items:
            source = (item["source_session"], int(item["source_episode_index"]))
            key = f"{task}/{int(item['episode_index']):06d}"
            if source in provenance:
                raise ValueError(
                    f"duplicate original recording: {provenance[source]} and {key}"
                )
            provenance[source] = key
    atomic(output / "split_manifest.json", manifest)
    action_values, state_values = [], []
    files = {}
    total_frames = 0
    video_checks = 0
    for ordinal, (key, record) in enumerate(sorted(records.items()), 1):
        path = data_path(root, record)
        rows = episode_rows(root, record)
        state, action = rows["observation.state"], rows["action"]
        if state.shape != (int(record["length"]), 16) or action.shape != state.shape:
            raise ValueError(f"invalid 16D geometry: {key}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"nonfinite data: {key}")
        if not np.allclose(
            action[:-1], state[1:], atol=1e-7, rtol=0
        ) or not np.allclose(action[-1], state[-1], atol=1e-7, rtol=0):
            raise ValueError(f"filtered action-next-state mismatch: {key}")
        if not np.allclose(rows["timestamp"], np.arange(len(state)) / 30, atol=2e-5):
            raise ValueError(f"timestamps are not the filtered 30 FPS grid: {key}")
        tasks = {
            int(r["task_index"]): r["task"]
            for r in pq.read_table(
                root / record["task"] / "meta/tasks.parquet"
            ).to_pylist()
        }
        task_indices = np.unique(rows["task_index"])
        if len(task_indices) != 1 or tasks[int(task_indices[0])] != record["tasks"][0]:
            raise ValueError(f"instruction mismatch: {key}")
        mask = train_masks[key]
        state_values.append(state[mask])
        # Each selected action row references its next state; exclude rows whose next state is not training-owned.
        action_mask = mask & np.r_[mask[1:], False]
        action_values.append(action[action_mask])
        total_frames += len(state)
        files[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        if video_audit:
            views = []
            for camera in CAMERAS:
                path, base = video_location(root, record, camera)
                frames = decode_frames(
                    path, [base, base + len(state) // 2, base + len(state) - 1]
                )
                views.append(frames)
                video_checks += 3
                files[str(path.relative_to(root))] = {
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
            if ordinal in (1, 51, 101):
                from PIL import Image
                from onewm.data.video_augmentation import MultiViewVideoAugmentation

                original, _ = compose([v[1:2] for v in views])
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(42)
                    aug = MultiViewVideoAugmentation(
                        probability=1,
                        crop_ratio=0.98,
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.1,
                        hue=0,
                    )
                    augmented, _ = compose(aug([v[1:2] for v in views]))
                canvas = (
                    torch.cat([original[:, 0], augmented[:, 0]], dim=1)
                    .add(1)
                    .mul(127.5)
                    .clamp(0, 255)
                    .byte()
                    .permute(1, 2, 0)
                    .numpy()
                )
                Image.fromarray(canvas).save(
                    output / f"augmentation_{record['task']}.png"
                )
        if ordinal % 10 == 0 or ordinal == len(records):
            status(
                output / "preparation_progress.json",
                phase="data_audit",
                episodes=ordinal,
                total=len(records),
            )
            print(f"audit {ordinal}/{len(records)} frames={total_frames}", flush=True)
    stats = {
        "contract": CONTRACT,
        "manifest_sha256": digest(output / "split_manifest.json"),
        "action": quantile_stats(action_values),
        "state": quantile_stats(state_values),
    }
    atomic(output / "train_stats.json", stats)
    atomic(output / "source_inventory.json", files)
    audit = dict(
        episodes=len(records),
        frames=total_frames,
        train_windows=len(manifest["train_windows"]),
        val_windows=len(manifest["val_windows"]),
        excluded_windows=len(manifest["excluded_windows"]),
        overlap_rows=0,
        val_source_rows=sum(int(m.sum()) for m in val_masks.values()),
        checked_video_frames=video_checks,
        video_audit=video_audit,
        contract=CONTRACT,
    )
    atomic(output / "audit.json", audit)
    atomic(
        complete,
        dict(
            **audit,
            asset_sha256={
                name: digest(output / name)
                for name in (
                    "split_manifest.json",
                    "train_stats.json",
                    "source_inventory.json",
                    "audit.json",
                )
            },
        ),
    )
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DATA_ROOT))
    parser.add_argument("--output", default=str(ASSETS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    prepare(args.root, args.output)


if __name__ == "__main__":
    main()
