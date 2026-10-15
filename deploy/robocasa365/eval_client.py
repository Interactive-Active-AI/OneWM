"""Official target50 closed-loop evaluator for the split RoboCasa environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deploy.robocasa365.protocol import encode_observation


def _action_dict(action: list[float]) -> dict[str, np.ndarray]:
    value = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if value.shape != (12,) or not np.isfinite(value).all():
        raise ValueError(f"policy action must be finite [12], got {value.shape}")
    value[6] = 1.0 if value[6] > 0 else -1.0
    value[11] = 1.0 if value[11] > 0 else -1.0
    return {
        "action.end_effector_position": value[0:3],
        "action.end_effector_rotation": value[3:6],
        "action.gripper_close": value[6:7],
        "action.base_motion": value[7:11],
        "action.control_mode": value[11:12],
    }


class PolicyClient:
    def __init__(self, uri: str) -> None:
        from websockets.sync.client import connect

        self.socket = connect(uri, max_size=8 * 1024 * 1024, open_timeout=60)

    def call(self, payload: dict) -> dict:
        self.socket.send(json.dumps(payload))
        response = json.loads(self.socket.recv())
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown policy-server error"))
        return response

    def close(self) -> None:
        self.socket.close()


def _task_sets() -> tuple[list[str], dict[str, str], dict[str, int]]:
    from robocasa.utils.dataset_registry import TARGET_TASKS
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    groups = {
        task: group
        for group in ("atomic_seen", "composite_seen", "composite_unseen")
        for task in TARGET_TASKS[group]
    }
    horizons = {task: int(get_task_horizon(task)) for task in groups}
    return list(groups), groups, horizons


def _completed(path: Path, split: str) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if str(record.get("split", "pretrain")) == split:
                keys.add((str(record["task"]), int(record["scenario"])))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://127.0.0.1:8765")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--rollouts-per-task", type=int, default=50)
    parser.add_argument("--split", default="pretrain", choices=("pretrain", "target"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    import os

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import gymnasium as gym
    import imageio.v2 as imageio
    import robocasa  # noqa: F401 -- registers gym environments

    target50, groups, official_horizons = _task_sets()
    tasks = list(dict.fromkeys(args.tasks or target50))
    unknown = sorted(set(tasks) - set(target50))
    if unknown:
        raise ValueError(f"--tasks contains non-target50 tasks: {unknown}")
    if args.rollouts_per_task <= 0:
        raise ValueError("--rollouts-per-task must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    already = _completed(episodes_path, args.split)
    client = PolicyClient(args.server)
    try:
        client.call({"op": "health"})
        for task in tasks:
            for scenario in range(args.rollouts_per_task):
                if (task, scenario) in already:
                    continue
                rollout_id = f"{task}:{scenario}"
                seed = args.seed_offset + scenario
                env = gym.make(
                    f"robocasa/{task}", split=args.split, seed=seed, enable_render=True
                )
                writer = None
                started = time.time()
                try:
                    observation, _ = env.reset(seed=seed)
                    if args.video:
                        video_dir = args.output_dir / "videos" / task
                        video_dir.mkdir(parents=True, exist_ok=True)
                        writer = imageio.get_writer(
                            video_dir / f"scenario_{scenario:03d}.mp4", fps=20
                        )
                    client.call({"op": "reset", "rollout_id": rollout_id})
                    success = False
                    steps = 0
                    horizon = int(
                        args.max_steps
                        if args.max_steps is not None
                        else official_horizons[task]
                    )
                    while steps < horizon and not success:
                        response = client.call(
                            {
                                "op": "predict",
                                "rollout_id": rollout_id,
                                "observation": encode_observation(observation),
                            }
                        )
                        actions = response["actions"]
                        if not actions or len(actions) > 8:
                            raise ValueError(
                                f"server returned invalid execute chunk: {len(actions)}"
                            )
                        for action_index, action in enumerate(actions):
                            observation, _, terminated, truncated, info = env.step(
                                _action_dict(action)
                            )
                            steps += 1
                            if writer is not None:
                                writer.append_data(
                                    observation["video.robot0_agentview_left"]
                                )
                            success = bool(info.get("success", False))
                            if success or terminated or truncated or steps >= horizon:
                                break
                            if action_index + 1 < len(actions):
                                client.call(
                                    {
                                        "op": "observe",
                                        "rollout_id": rollout_id,
                                        "observation": encode_observation(observation),
                                    }
                                )
                    record = {
                        "task": task,
                        "group": groups[task],
                        "scenario": scenario,
                        "seed": seed,
                        "split": args.split,
                        "success": success,
                        "steps": steps,
                        "elapsed_seconds": time.time() - started,
                    }
                    with episodes_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                finally:
                    if writer is not None:
                        writer.close()
                    env.close()
    finally:
        client.close()

    records = [
        json.loads(line)
        for line in episodes_path.read_text().splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in records
        if row["task"] in tasks
        and row.get("split") == args.split
        and int(row["scenario"]) < args.rollouts_per_task
    ]
    by_group = {}
    for group in ("atomic_seen", "composite_seen", "composite_unseen"):
        rows = [row for row in selected if row["group"] == group]
        by_group[group] = (
            sum(bool(row["success"]) for row in rows) / len(rows) if rows else None
        )
    per_task = {
        task: sum(bool(row["success"]) for row in selected if row["task"] == task)
        / max(1, sum(row["task"] == task for row in selected))
        for task in tasks
    }
    summary = {
        "tasks": len(tasks),
        "episodes": len(selected),
        "micro_success": sum(bool(row["success"]) for row in selected)
        / max(1, len(selected)),
        "macro_success": sum(per_task.values()) / max(1, len(per_task)),
        "groups": by_group,
        "per_task": per_task,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
