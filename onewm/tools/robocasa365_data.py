"""Audit Human300 and compute statistics in OneWM's transformed coordinates."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


from onewm.data.lerobot import _decode_video_frames
from onewm.data.robocasa365 import (
    CAMERA_KEYS,
    _validate_modality,
    canonical_task_from_root,
    compute_robocasa365_stats,
    discover_robocasa365_roots,
    robocasa365_episode_split,
    save_robocasa365_stats,
)


def official_pretrain300(registry: Path) -> list[str]:
    tree = ast.parse(registry.read_text(encoding="utf-8"), filename=str(registry))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "PRETRAINING_TASKS"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call):
            break
        for keyword in node.value.keywords:
            if keyword.arg == "pretrain300":
                tasks = ast.literal_eval(keyword.value)
                if len(tasks) != 300 or len(set(tasks)) != 300:
                    raise ValueError(
                        f"official pretrain300 registry is not 300 unique tasks: {registry}"
                    )
                return [str(task) for task in tasks]
    raise ValueError(
        f"could not statically read PRETRAINING_TASKS.pretrain300 from {registry}"
    )


def validate_official_task_set(roots: list[Path], registry: Path) -> None:
    expected = set(official_pretrain300(registry))
    actual = {canonical_task_from_root(path) for path in roots}
    if actual != expected:
        raise ValueError(
            "Human300 differs from the pinned official pretrain300 registry: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def audit(root: Path, expected_tasks: int, decode_tasks: int, registry: Path) -> dict:
    roots = discover_robocasa365_roots(root)
    if len(roots) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} Human task roots, found {len(roots)}"
        )
    if expected_tasks == 300:
        validate_official_task_set(roots, registry)
    categories: dict[str, int] = {"atomic": 0, "composite": 0}
    episodes = frames = 0
    for index, task_root in enumerate(roots):
        category = task_root.parents[2].name
        categories[category] += 1
        info = json.loads((task_root / "meta" / "info.json").read_text())
        episode_rows = [
            json.loads(line)
            for line in (task_root / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        episodes += len(episode_rows)
        task_frames = sum(int(row["length"]) for row in episode_rows)
        frames += task_frames
        if int(info.get("fps", -1)) != 20:
            raise ValueError(f"{task_root}: expected 20Hz")
        if int(info.get("total_episodes", -1)) != len(episode_rows):
            raise ValueError(f"{task_root}: episode metadata count mismatch")
        if int(info.get("total_frames", -1)) != task_frames:
            raise ValueError(f"{task_root}: frame metadata count mismatch")
        _validate_modality(task_root)
        for camera in CAMERA_KEYS:
            feature = info.get("features", {}).get(camera, {})
            if feature.get("dtype") != "video" or feature.get("shape") != [256, 256, 3]:
                raise ValueError(f"{task_root}: invalid {camera}: {feature}")
            video_count = sum(
                1 for _ in task_root.glob(f"videos/chunk-*/{camera}/episode_*.mp4")
            )
            if video_count != len(episode_rows):
                raise ValueError(
                    f"{task_root}: {camera} has {video_count} videos for {len(episode_rows)} episodes"
                )
        if info.get("features", {}).get("observation.state", {}).get("shape") != [16]:
            raise ValueError(f"{task_root}: state is not 16D")
        if info.get("features", {}).get("action", {}).get("shape") != [12]:
            raise ValueError(f"{task_root}: action is not 12D")
        parquet_count = sum(1 for _ in task_root.glob("data/chunk-*/episode_*.parquet"))
        if parquet_count != len(episode_rows):
            raise ValueError(
                f"{task_root}: has {parquet_count} parquet files for {len(episode_rows)} episodes"
            )
        if index < decode_tasks:
            episode_index = int(episode_rows[0]["episode_index"])
            chunk_size = int(info.get("chunks_size", 1000))
            chunk = episode_index // chunk_size
            for camera in CAMERA_KEYS:
                video = (
                    task_root
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / camera
                    / f"episode_{episode_index:06d}.mp4"
                )
                decoded = _decode_video_frames(video, [0])
                if tuple(decoded.shape) != (1, 3, 256, 256):
                    raise ValueError(
                        f"unexpected decoded shape for {video}: {tuple(decoded.shape)}"
                    )
    return {
        "root": str(root.resolve()),
        "tasks": len(roots),
        "categories": categories,
        "episodes": episodes,
        "frames": frames,
        "decoded_tasks": min(decode_tasks, len(roots)),
        "canonical_tasks": [canonical_task_from_root(path) for path in roots],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_registry = (
        Path(__file__).resolve().parents[2]
        / "third_party/RoboCasa365/robocasa/utils/dataset_registry.py"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--root", type=Path, required=True)
    audit_parser.add_argument("--expected-tasks", type=int, default=300)
    audit_parser.add_argument("--decode-tasks", type=int, default=3)
    audit_parser.add_argument("--output", type=Path)
    audit_parser.add_argument("--registry", type=Path, default=default_registry)
    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--root", type=Path, required=True)
    stats_parser.add_argument("--output", type=Path, required=True)
    stats_parser.add_argument("--expected-tasks", type=int, default=300)
    stats_parser.add_argument("--val-split", type=float, default=0.01)
    stats_parser.add_argument("--seed", type=int, default=42)
    stats_parser.add_argument("--registry", type=Path, default=default_registry)
    args = parser.parse_args()

    if args.command == "audit":
        report = audit(args.root, args.expected_tasks, args.decode_tasks, args.registry)
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text, end="")
        return

    roots = discover_robocasa365_roots(args.root)
    if len(roots) != args.expected_tasks:
        raise ValueError(
            f"expected {args.expected_tasks} task roots, found {len(roots)}"
        )
    if args.expected_tasks == 300:
        validate_official_task_set(roots, args.registry)
    train_ids, _ = robocasa365_episode_split(roots, args.val_split, args.seed)
    stats = compute_robocasa365_stats(roots, train_ids)
    save_robocasa365_stats(
        stats,
        args.output,
        task_names=[canonical_task_from_root(path) for path in roots],
    )
    print(f"wrote {args.output} from {len(roots)} task roots")


if __name__ == "__main__":
    main()
