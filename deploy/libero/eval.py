"""Run standard LIBERO or LIBERO-Plus against a remote OneWM policy."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from deploy.libero.adapter import (
    LIBERO_DUMMY_ACTION,
    LIBERO_SUITES,
    action_to_libero,
    extract_camera_views,
    extract_state,
    max_steps_for_suite,
    select_shard,
)
from deploy.libero.protocol import ProtocolError, decode_message, encode_message


LOGGER = logging.getLogger("onewm.libero.eval")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class TaskRef:
    suite: str
    task_id: int


class RemotePolicyError(RuntimeError):
    """Raised when the policy server returns a structured failure."""


class PolicyClient:
    """Persistent synchronous client with bounded reconnect retries."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout: float = 300.0,
        retries: int = 2,
    ) -> None:
        self.uri = f"ws://{host}:{port}"
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self._connection = None
        self._request_id = 0
        self._metadata: Optional[dict[str, Any]] = None

    def __enter__(self) -> "PolicyClient":
        self._connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _connect(self) -> None:
        if self._connection is not None:
            return
        from websockets.sync.client import connect

        self._connection = connect(
            self.uri,
            open_timeout=min(self.timeout, 30.0),
            close_timeout=10.0,
            max_size=8 * 1024 * 1024,
        )

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    def _exchange_once(self, request: dict[str, Any]) -> dict[str, Any]:
        self._connect()
        assert self._connection is not None
        self._connection.send(encode_message(request))
        frame = self._connection.recv(timeout=self.timeout)
        if not isinstance(frame, (bytes, bytearray, memoryview)):
            raise ProtocolError("policy server returned a text WebSocket frame")
        response = decode_message(frame)
        if response.get("request_id") != request["request_id"]:
            raise ProtocolError(
                "policy response request_id mismatch: "
                f"{response.get('request_id')} != {request['request_id']}"
            )
        if response.get("ok") is not True:
            raise RemotePolicyError(
                f"{response.get('error_type', 'RemoteError')}: "
                f"{response.get('error', 'unknown policy error')}"
            )
        return response

    def request(self, op: str, **payload: Any) -> dict[str, Any]:
        self._request_id += 1
        request = {
            "op": op,
            "request_id": self._request_id,
            **payload,
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                return self._exchange_once(request)
            except RemotePolicyError:
                raise
            except Exception as exc:
                last_error = exc
                self.close()
                if attempt >= self.retries:
                    break
                LOGGER.warning(
                    "policy request failed (%s); reconnecting attempt %d/%d",
                    exc,
                    attempt + 1,
                    self.retries,
                )
        assert last_error is not None
        raise RuntimeError(
            f"policy request failed after {self.retries + 1} attempts"
        ) from last_error

    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            self._metadata = self.request("metadata")
        return dict(self._metadata)

    def predict(
        self,
        primary: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        response = self.request(
            "predict",
            primary=primary,
            wrist=wrist,
            state=np.asarray(state, dtype=np.float32),
            instruction=instruction,
        )
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        metadata = self.metadata()
        expected = (
            int(metadata["action_horizon"]),
            int(metadata["action_dim"]),
        )
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ProtocolError(
                f"invalid policy action chunk {actions.shape}; expected {expected}"
            )
        return actions


def _default_libero_root(benchmark_kind: str) -> Path:
    name = "LIBERO" if benchmark_kind == "standard" else "LIBERO-plus"
    return REPO_ROOT / "third_party" / name


def prepare_libero_runtime(
    *,
    benchmark_kind: str,
    libero_root: str | Path,
    config_dir: str | Path,
) -> Path:
    """Create an isolated LIBERO config before importing either libero package."""

    if "libero" in sys.modules:
        raise RuntimeError("prepare_libero_runtime must run before importing libero")
    root = Path(libero_root).expanduser().resolve()
    package_root = root / "libero" / "libero"
    required = {
        "package": package_root / "__init__.py",
        "bddl_files": package_root / "bddl_files",
        "init_states": package_root / "init_files",
        "assets": package_root / "assets",
    }
    if benchmark_kind == "plus":
        required["task_classification"] = (
            package_root / "benchmark" / "task_classification.json"
        )
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"incomplete {benchmark_kind} LIBERO runtime: " + ", ".join(missing)
        )

    config_root = Path(config_dir).expanduser().resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    paths = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(root / "libero" / "datasets"),
        "assets": str(package_root / "assets"),
    }
    temporary = config_path.with_suffix(".yaml.tmp")
    temporary.write_text(json.dumps(paths, indent=2), encoding="utf-8")
    temporary.replace(config_path)

    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return package_root


def _runtime_args(args: argparse.Namespace) -> tuple[Path, Path]:
    root = (
        Path(args.libero_root)
        if args.libero_root
        else _default_libero_root(args.benchmark)
    )
    config_dir = (
        Path(args.config_dir)
        if args.config_dir
        else REPO_ROOT
        / "data"
        / "onewm_libero_assets"
        / "eval_configs"
        / args.benchmark
    )
    return root, config_dir


def _suite_names(values: Optional[list[str]]) -> list[str]:
    return list(values) if values else list(LIBERO_SUITES)


def _build_task_refs(
    benchmark_module: Any,
    suites: list[str],
    task_ids: Optional[list[int]],
) -> tuple[list[TaskRef], dict[str, Any]]:
    benchmark_dict = benchmark_module.get_benchmark_dict()
    suite_objects: dict[str, Any] = {}
    refs: list[TaskRef] = []
    selected_ids = None if task_ids is None else sorted(set(task_ids))
    for suite_name in suites:
        suite = benchmark_dict[suite_name]()
        suite_objects[suite_name] = suite
        ids = range(int(suite.n_tasks)) if selected_ids is None else selected_ids
        for task_id in ids:
            if not 0 <= int(task_id) < int(suite.n_tasks):
                raise ValueError(
                    f"task_id={task_id} out of range for {suite_name} "
                    f"with {suite.n_tasks} tasks"
                )
            refs.append(TaskRef(suite_name, int(task_id)))
    return refs, suite_objects


def _load_plus_classification(
    package_root: Path,
    suites: list[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    path = package_root / "benchmark" / "task_classification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for suite in suites:
        for item in payload[suite]:
            result[(suite, int(item["id"]) - 1)] = {
                "category": str(item["category"]),
                "perturbation": str(item["name"]),
            }
    return result


def _make_env(task: Any, *, resolution: int, seed: int, get_libero_path: Any) -> Any:
    from libero.libero.envs import OffScreenRenderEnv  # pyright: ignore[reportMissingImports]

    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    is_virtual_plus_path = (
        "_view_" in bddl_path.name and "_initstate_" in bddl_path.name
    )
    if not bddl_path.is_file() and not is_virtual_plus_path:
        raise FileNotFoundError(f"missing task BDDL file: {bddl_path}")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=int(resolution),
        camera_widths=int(resolution),
    )
    env.seed(int(seed))
    return env


def _run_episode(
    *,
    env: Any,
    initial_state: np.ndarray,
    instruction: str,
    client: PolicyClient,
    replan_steps: int,
    max_steps: int,
    wait_steps: int,
    capture_video: bool,
) -> dict[str, Any]:
    env.reset()
    observation = env.set_init_state(initial_state)
    done = False
    for _ in range(wait_steps):
        observation, _, done, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        if done:
            break

    pending: deque[np.ndarray] = deque()
    frames: list[np.ndarray] = []
    steps = 0
    replans = 0
    while not done and steps < max_steps:
        if not pending:
            primary, wrist = extract_camera_views(observation)
            actions = client.predict(
                primary,
                wrist,
                extract_state(observation),
                instruction,
            )
            pending.extend(actions[:replan_steps])
            replans += 1
        if capture_video:
            frames.append(extract_camera_views(observation)[0])
        action = action_to_libero(pending.popleft())
        observation, _, done, _ = env.step(action.tolist())
        steps += 1

    if capture_video:
        frames.append(extract_camera_views(observation)[0])
    return {
        "success": bool(done),
        "steps": steps,
        "replans": replans,
        "frames": frames,
    }


def _safe_instruction_name(instruction: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", instruction.strip())
    return value.strip("_")[:80] or "task"


def _save_video(
    frames: list[np.ndarray],
    *,
    output_dir: Path,
    suite: str,
    task_id: int,
    episode_idx: int,
    instruction: str,
    success: bool,
    fps: int,
) -> Path:
    import imageio.v2 as imageio

    directory = output_dir / "videos" / suite
    directory.mkdir(parents=True, exist_ok=True)
    status = "success" if success else "failure"
    path = directory / (
        f"task-{task_id:05d}_episode-{episode_idx:02d}_{status}_"
        f"{_safe_instruction_name(instruction)}.mp4"
    )
    if not frames:
        raise ValueError("cannot save an empty rollout video")
    with imageio.get_writer(str(path), fps=int(fps)) as writer:
        for frame in frames:
            writer.append_data(frame)  # pyright: ignore[reportAttributeAccessIssue]
    return path


def _record_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record["benchmark"]),
        str(record["suite"]),
        int(record["task_id"]),
        int(record["episode_idx"]),
    )


def _read_records(paths: list[Path]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    records: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[_record_key(record)] = record
            except Exception as exc:
                raise ValueError(f"invalid record {path}:{line_number}") from exc
    return records


def _rate_bucket() -> dict[str, Any]:
    return {
        "episodes": 0,
        "successes": 0,
        "failures": 0,
        "errors": 0,
        "success_rate": None,
    }


def _finish_bucket(bucket: dict[str, Any]) -> None:
    completed = int(bucket["successes"]) + int(bucket["failures"])
    bucket["episodes"] = completed
    bucket["success_rate"] = (
        float(bucket["successes"]) / completed if completed else None
    )


def aggregate_records(paths: list[Path]) -> dict[str, Any]:
    records = list(_read_records(paths).values())
    total = _rate_bucket()
    suites: dict[str, dict[str, Any]] = {}
    categories: dict[str, dict[str, Any]] = {}
    for record in records:
        suite_bucket = suites.setdefault(str(record["suite"]), _rate_bucket())
        category = record.get("category")
        category_bucket = (
            categories.setdefault(str(category), _rate_bucket()) if category else None
        )
        buckets = [total, suite_bucket]
        if category_bucket is not None:
            buckets.append(category_bucket)
        status = str(record.get("status"))
        field = {
            "success": "successes",
            "failure": "failures",
            "error": "errors",
        }.get(status)
        if field is None:
            raise ValueError(f"unknown episode status: {status!r}")
        for bucket in buckets:
            bucket[field] += 1

    _finish_bucket(total)
    for bucket in suites.values():
        _finish_bucket(bucket)
    for bucket in categories.values():
        _finish_bucket(bucket)
    return {
        "record_files": [str(path) for path in paths],
        "unique_records": len(records),
        "total": total,
        "suites": dict(sorted(suites.items())),
        "categories": dict(sorted(categories.items())),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def preflight(args: argparse.Namespace) -> None:
    # Metadata-only preflight never creates a GL context. Robosuite 1.4 forces
    # EGL unless MUJOCO_GL is glx or osmesa, so select glx here to avoid loading
    # system EGL libraries merely to inspect the task registry.
    os.environ["MUJOCO_GL"] = args.mujoco_gl if args.instantiate_env else "glx"
    root, config_dir = _runtime_args(args)
    package_root = prepare_libero_runtime(
        benchmark_kind=args.benchmark,
        libero_root=root,
        config_dir=config_dir,
    )
    from libero.libero import benchmark, get_libero_path  # pyright: ignore[reportMissingImports]

    refs, suite_objects = _build_task_refs(
        benchmark,
        _suite_names(args.suite),
        args.task_id,
    )
    report: dict[str, Any] = {
        "benchmark": args.benchmark,
        "libero_root": str(root.resolve()),
        "package_root": str(package_root),
        "config_dir": str(config_dir.resolve()),
        "assets": str(get_libero_path("assets")),
        "tasks": len(refs),
        "suites": {name: int(suite.n_tasks) for name, suite in suite_objects.items()},
    }

    if args.instantiate_env:
        first = refs[0]
        suite = suite_objects[first.suite]
        task = suite.get_task(first.task_id)
        env = _make_env(
            task,
            resolution=args.resolution,
            seed=args.seed,
            get_libero_path=get_libero_path,
        )
        try:
            env.reset()
            initial_states = suite.get_task_init_states(first.task_id)
            observation = env.set_init_state(initial_states[0])
            env.step(LIBERO_DUMMY_ACTION.tolist())
            extract_camera_views(observation)
            extract_state(observation)
            report["environment_smoke"] = "passed"
        finally:
            env.close()
    print(json.dumps(report, indent=2, sort_keys=True))


def run_evaluation(args: argparse.Namespace) -> None:
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    if args.mujoco_gl == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    root, config_dir = _runtime_args(args)
    package_root = prepare_libero_runtime(
        benchmark_kind=args.benchmark,
        libero_root=root,
        config_dir=config_dir,
    )
    from libero.libero import benchmark, get_libero_path  # pyright: ignore[reportMissingImports]

    suite_names = _suite_names(args.suite)
    task_refs, suite_objects = _build_task_refs(
        benchmark,
        suite_names,
        args.task_id,
    )
    task_refs = select_shard(
        task_refs,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if not task_refs:
        raise ValueError("selected evaluation shard contains no tasks")

    classifications = (
        _load_plus_classification(package_root, suite_names)
        if args.benchmark == "plus"
        else {}
    )
    num_trials = (
        int(args.num_trials)
        if args.num_trials is not None
        else (1 if args.benchmark == "plus" else 50)
    )
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / (
        f"episodes.{args.benchmark}.shard-{args.shard_index:05d}-"
        f"of-{args.num_shards:05d}.jsonl"
    )
    if record_path.exists() and not args.resume:
        raise FileExistsError(
            f"{record_path} already exists; pass --resume or choose another output"
        )
    existing = _read_records([record_path]) if args.resume else {}
    LOGGER.info(
        "evaluation benchmark=%s tasks=%d trials=%d shard=%d/%d output=%s",
        args.benchmark,
        len(task_refs),
        num_trials,
        args.shard_index,
        args.num_shards,
        record_path,
    )

    with PolicyClient(
        host=args.host,
        port=args.port,
        timeout=args.request_timeout,
        retries=args.request_retries,
    ) as client:
        metadata = client.metadata()
        if int(metadata["action_dim"]) != 7 or int(metadata["proprio_dim"]) != 8:
            raise ValueError(f"server is not a LIBERO policy: {metadata}")
        replan_steps = (
            int(args.replan_steps)
            if args.replan_steps is not None
            else int(metadata["replan_steps"])
        )
        if not 1 <= replan_steps <= int(metadata["action_horizon"]):
            raise ValueError(f"invalid replan_steps={replan_steps}")

        for task_ref in task_refs:
            pending_episode_indices = []
            for episode_idx in range(num_trials):
                key = (
                    args.benchmark,
                    task_ref.suite,
                    task_ref.task_id,
                    episode_idx,
                )
                prior = existing.get(key)
                if prior is not None and (
                    prior.get("status") in {"success", "failure"}
                    or not args.retry_errors
                ):
                    continue
                pending_episode_indices.append(episode_idx)
            if not pending_episode_indices:
                continue

            suite = suite_objects[task_ref.suite]
            task = suite.get_task(task_ref.task_id)
            instruction = str(task.language)
            initial_states = suite.get_task_init_states(task_ref.task_id)
            if num_trials > len(initial_states):
                raise ValueError(
                    f"{task_ref} requests {num_trials} trials but only "
                    f"{len(initial_states)} initial states are available"
                )
            classification = classifications.get(
                (task_ref.suite, task_ref.task_id),
                {},
            )
            env = _make_env(
                task,
                resolution=args.resolution,
                seed=args.seed,
                get_libero_path=get_libero_path,
            )
            try:
                for episode_idx in pending_episode_indices:
                    key = (
                        args.benchmark,
                        task_ref.suite,
                        task_ref.task_id,
                        episode_idx,
                    )
                    started_at = time.perf_counter()
                    base_record = {
                        "benchmark": args.benchmark,
                        "suite": task_ref.suite,
                        "task_id": task_ref.task_id,
                        "episode_idx": episode_idx,
                        "instruction": instruction,
                        **classification,
                    }
                    try:
                        result = _run_episode(
                            env=env,
                            initial_state=initial_states[episode_idx],
                            instruction=instruction,
                            client=client,
                            replan_steps=replan_steps,
                            max_steps=(
                                int(args.max_steps)
                                if args.max_steps is not None
                                else max_steps_for_suite(task_ref.suite)
                            ),
                            wait_steps=args.wait_steps,
                            capture_video=args.save_video != "none",
                        )
                        success = bool(result["success"])
                        record = {
                            **base_record,
                            "status": "success" if success else "failure",
                            "success": success,
                            "steps": int(result["steps"]),
                            "replans": int(result["replans"]),
                            "elapsed_seconds": time.perf_counter() - started_at,
                        }
                        should_save = args.save_video == "all" or (
                            args.save_video == "failures" and not success
                        )
                        if should_save:
                            video_path = _save_video(
                                result["frames"],
                                output_dir=output_dir,
                                suite=task_ref.suite,
                                task_id=task_ref.task_id,
                                episode_idx=episode_idx,
                                instruction=instruction,
                                success=success,
                                fps=args.video_fps,
                            )
                            record["video"] = str(video_path)
                    except Exception as exc:
                        LOGGER.exception(
                            "episode failed suite=%s task=%d episode=%d",
                            task_ref.suite,
                            task_ref.task_id,
                            episode_idx,
                        )
                        record = {
                            **base_record,
                            "status": "error",
                            "success": None,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "elapsed_seconds": time.perf_counter() - started_at,
                        }
                    _append_record(record_path, record)
                    existing[key] = record
                    LOGGER.info(
                        "episode suite=%s task=%d episode=%d status=%s",
                        task_ref.suite,
                        task_ref.task_id,
                        episode_idx,
                        record["status"],
                    )
            finally:
                env.close()

    shard_summary = aggregate_records([record_path])
    _write_json(
        output_dir
        / (
            f"summary.{args.benchmark}.shard-{args.shard_index:05d}-"
            f"of-{args.num_shards:05d}.json"
        ),
        shard_summary,
    )


def aggregate_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = sorted(output_dir.glob(f"episodes.{args.benchmark}.shard-*-of-*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no {args.benchmark} episode shards found under {output_dir}"
        )
    summary = aggregate_records(paths)
    target = output_dir / f"summary.{args.benchmark}.json"
    _write_json(target, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", choices=("standard", "plus"), required=True)
    parser.add_argument("--libero-root", default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument(
        "--suite",
        action="append",
        choices=LIBERO_SUITES,
        default=None,
    )
    parser.add_argument("--task-id", action="append", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--mujoco-gl",
        choices=("egl",),
        default=os.environ.get("MUJOCO_GL", "egl"),
    )
    parser.add_argument("--seed", type=int, default=7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    _add_runtime_arguments(preflight_parser)
    preflight_parser.add_argument("--instantiate-env", action="store_true")
    preflight_parser.set_defaults(handler=preflight)

    run_parser = subparsers.add_parser("run")
    _add_runtime_arguments(run_parser)
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=10093)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--num-trials", type=int, default=None)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--replan-steps", type=int, default=None)
    run_parser.add_argument("--wait-steps", type=int, default=30)
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--request-timeout", type=float, default=300.0)
    run_parser.add_argument("--request-retries", type=int, default=2)
    run_parser.add_argument(
        "--save-video",
        choices=("none", "failures", "all"),
        default="failures",
    )
    run_parser.add_argument("--video-fps", type=int, default=20)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--retry-errors", action="store_true")
    run_parser.set_defaults(handler=run_evaluation)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--benchmark",
        choices=("standard", "plus"),
        required=True,
    )
    aggregate_parser.add_argument("--output-dir", required=True)
    aggregate_parser.set_defaults(handler=aggregate_command)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
