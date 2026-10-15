"""RoboCasa365 Human300 LeRobot-v2.1 data and online geometry contracts.

The official release stores one independent LeRobot dataset per canonical task.
This module presents those roots as one task-balanced dataset while keeping the
RoboCasa-specific state/action ordering, history sampling, and three-view strip
out of the generic LeRobot reader.
"""

from __future__ import annotations

import bisect
import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import Dataset

from onewm.data.lerobot import (
    DEFAULT_TEXT_CACHE_ENCODER_ID,
    DEFAULT_TEXT_PROMPT,
    _decode_video_frames,
    load_text_cache,
    text_cache_path,
)
from onewm.data.rng import isolated_cpu_rng


ROBOCASA365_CONTRACT_VERSION = "rc365_xiaomi14_action12_v1"
LEFT_CAMERA = "observation.images.robot0_agentview_left"
RIGHT_CAMERA = "observation.images.robot0_agentview_right"
WRIST_CAMERA = "observation.images.robot0_eye_in_hand"
CAMERA_KEYS = (LEFT_CAMERA, WRIST_CAMERA, RIGHT_CAMERA)
ACTION_BINARY_DIMS = (6, 11)


def _decode_robocasa365_video_frames(
    video_path: Path, frame_indices: list[int]
) -> Tensor:
    """Decode a local 20Hz window with timestamp seeking, with an exact fallback.

    Human300 episodes are independent MP4 files and training samples touch only
    a roughly 22-frame window. Seeking to the preceding H264 keyframe avoids
    decoding every frame from episode start for every random sample.
    """
    target = sorted(set(int(index) for index in frame_indices))
    if not target or target[0] < 32:
        return _decode_video_frames(video_path, frame_indices)
    try:
        import av

        decoded: dict[int, Tensor] = {}
        with av.open(str(video_path), timeout=30) as container:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            stream.codec_context.thread_count = 1
            time_base = float(stream.time_base)
            seek_pts = max(0, int((target[0] / 20.0) / time_base))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            target_set = set(target)
            try:
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise RuntimeError("decoded frame has no timestamp")
                    frame_index = int(round(float(frame.pts * stream.time_base) * 20.0))
                    if frame_index in target_set:
                        array = frame.to_ndarray(format="rgb24")
                        decoded[frame_index] = torch.from_numpy(array.copy()).permute(
                            2, 0, 1
                        )
                    if frame_index >= target[-1]:
                        break
            finally:
                stream.codec_context.close()
        if all(index in decoded for index in target):
            return torch.stack([decoded[index] for index in frame_indices], dim=0)
    except Exception:  # noqa: BLE001 -- exact sequential decoder is the safe fallback
        pass
    return _decode_video_frames(video_path, frame_indices)


def quaternion_xyzw_to_axis_angle(quaternion: Tensor) -> Tensor:
    """Convert ``[...,4]`` XYZW quaternions to shortest-path axis-angle."""
    quaternion = torch.as_tensor(quaternion)
    if quaternion.shape[-1] != 4:
        raise ValueError(
            f"quaternion must end in 4 values, got {tuple(quaternion.shape)}"
        )
    original_dtype = quaternion.dtype
    value = quaternion.to(torch.float64)
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    safe = value / norm.clamp_min(1e-12)
    identity = torch.zeros_like(safe)
    identity[..., 3] = 1.0
    safe = torch.where(norm > 1e-12, safe, identity)
    safe = torch.where(safe[..., 3:4] < 0.0, -safe, safe)
    xyz = safe[..., :3]
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, safe[..., 3:4].clamp(-1.0, 1.0))
    scale = torch.where(sin_half > 1e-8, angle / sin_half.clamp_min(1e-12), 2.0)
    return (xyz * scale).to(original_dtype)


def robocasa365_state_to_xiaomi14(state: Tensor) -> Tensor:
    """Convert official base-first quaternion state to Xiaomi's EE-first 14D state."""
    state = torch.as_tensor(state)
    if state.shape[-1] != 16:
        raise ValueError(f"raw RoboCasa365 state must be 16D, got {tuple(state.shape)}")
    return torch.cat(
        (
            state[..., 7:10],
            quaternion_xyzw_to_axis_angle(state[..., 10:14]),
            state[..., 14:16],
            state[..., 0:3],
            quaternion_xyzw_to_axis_angle(state[..., 3:7]),
        ),
        dim=-1,
    ).to(torch.float32)


def robocasa365_action_to_xiaomi12(action: Tensor) -> Tensor:
    """Reorder official LeRobot actions into the RoboCasa env/Xiaomi order."""
    action = torch.as_tensor(action)
    if action.shape[-1] != 12:
        raise ValueError(
            f"raw RoboCasa365 action must be 12D, got {tuple(action.shape)}"
        )
    output = torch.cat(
        (
            action[..., 5:8],
            action[..., 8:11],
            action[..., 11:12],
            action[..., 0:4],
            action[..., 4:5],
        ),
        dim=-1,
    ).to(torch.float32)
    # The official data already uses {-1,+1}; sign canonicalization also makes
    # the adapter robust to a future {0,1} export without z-scoring binaries.
    for dim in ACTION_BINARY_DIMS:
        output[..., dim] = torch.where(output[..., dim] > 0, 1.0, -1.0)
    return output


def robocasa365_action_to_env_dict(action: Tensor) -> dict[str, Tensor]:
    """Convert one Xiaomi-order action to the official Gym wrapper dictionary."""
    value = torch.as_tensor(action, dtype=torch.float32)
    if value.shape != (12,):
        raise ValueError(f"env action must be [12], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError("RoboCasa365 action contains NaN/Inf")
    value = value.clone()
    value[6] = 1.0 if value[6] > 0.0 else -1.0
    value[11] = 1.0 if value[11] > 0.0 else -1.0
    return {
        "action.end_effector_position": value[0:3],
        "action.end_effector_rotation": value[3:6],
        "action.gripper_close": value[6:7],
        "action.base_motion": value[7:11],
        "action.control_mode": value[11:12],
    }


def compose_robocasa365_views(
    left: Tensor,
    wrist: Tensor,
    right: Tensor,
    *,
    output_view_size: tuple[int, int] = (256, 256),
    separator_width: int = 16,
    separator_value: float = 128.0,
) -> Tensor:
    """Compose ``left | separator | wrist | separator | right``.

    Inputs have shape ``[...,3,H,W]`` and may contain arbitrary leading batch or
    time dimensions. Output preserves those dimensions and has
    ``[...,3,view_H,3*view_W+2*separator_width]``.
    """
    views = [torch.as_tensor(view) for view in (left, wrist, right)]
    for view in views:
        if view.ndim < 3 or view.shape[-3] != 3:
            raise ValueError(f"views must be [...,3,H,W], got {tuple(view.shape)}")
    if any(view.shape[:-3] != views[0].shape[:-3] for view in views[1:]):
        raise ValueError("all RoboCasa365 views must have identical leading dimensions")
    target_h, target_w = (int(output_view_size[0]), int(output_view_size[1]))
    resized: list[Tensor] = []
    for view in views:
        if tuple(view.shape[-2:]) == (target_h, target_w):
            resized.append(view)
            continue
        leading = view.shape[:-3]
        flat = view.reshape(-1, 3, *view.shape[-2:]).float()
        flat = F.interpolate(
            flat,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        resized.append(flat.reshape(*leading, 3, target_h, target_w).to(view.dtype))
    if separator_width < 0:
        raise ValueError("separator_width must be non-negative")
    if separator_width == 0:
        return torch.cat(resized, dim=-1)
    separator = torch.full(
        (*resized[0].shape[:-1], int(separator_width)),
        float(separator_value),
        dtype=resized[0].dtype,
        device=resized[0].device,
    )
    return torch.cat((resized[0], separator, resized[1], separator, resized[2]), dim=-1)


class RoboCasa365VideoTransform:
    """Multi-view, temporally and cross-view consistent augmentation.

    Every call performs a 0.95 crop. With ``random_probability`` it samples one
    crop position and shared color transform for every view/time; otherwise it
    uses the deterministic center crop with no color transform. Inputs and
    outputs are sequences of ``[T,3,H,W]`` tensors in the [0,255] range.
    """

    def __init__(
        self,
        *,
        random_probability: float = 0.40,
        crop_ratio: float = 0.95,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ) -> None:
        if not 0.0 <= random_probability <= 1.0:
            raise ValueError("random_probability must be in [0,1]")
        if not 0.0 < crop_ratio <= 1.0:
            raise ValueError("crop_ratio must be in (0,1]")
        if min(brightness, contrast, saturation, hue) < 0.0 or hue > 0.5:
            raise ValueError("invalid color-jitter magnitude")
        self.random_probability = float(random_probability)
        self.crop_ratio = float(crop_ratio)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    @staticmethod
    def _validate(views: Sequence[Tensor]) -> None:
        if len(views) != 3:
            raise ValueError(
                f"RoboCasa365 transform expects three views, got {len(views)}"
            )
        for view in views:
            if view.ndim != 4 or view.shape[1] != 3:
                raise ValueError(f"views must be [T,3,H,W], got {tuple(view.shape)}")
        if any(view.shape[0] != views[0].shape[0] for view in views[1:]):
            raise ValueError("all views must have the same temporal length")

    @staticmethod
    def _factor(magnitude: float) -> Optional[float]:
        if magnitude <= 0:
            return None
        return float(
            torch.empty(()).uniform_(max(0.0, 1.0 - magnitude), 1.0 + magnitude)
        )

    def __call__(self, views: Sequence[Tensor]) -> list[Tensor]:
        self._validate(views)
        random_branch = self.random_probability >= 1.0 or (
            self.random_probability > 0.0
            and float(torch.rand(())) < self.random_probability
        )
        y_fraction = float(torch.rand(())) if random_branch else 0.5
        x_fraction = float(torch.rand(())) if random_branch else 0.5
        cropped: list[Tensor] = []
        for view in views:
            height, width = int(view.shape[-2]), int(view.shape[-1])
            crop_h = max(1, int(height * self.crop_ratio))
            crop_w = max(1, int(width * self.crop_ratio))
            top = int(round(y_fraction * (height - crop_h)))
            left = int(round(x_fraction * (width - crop_w)))
            value = view[..., top : top + crop_h, left : left + crop_w].float()
            cropped.append(
                F.interpolate(
                    value,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
            )
        if not random_branch:
            return cropped

        factors = (
            self._factor(self.brightness),
            self._factor(self.contrast),
            self._factor(self.saturation),
            float(torch.empty(()).uniform_(-self.hue, self.hue)) if self.hue else None,
        )
        order = torch.randperm(4).tolist()
        output: list[Tensor] = []
        for view in cropped:
            value = (view / 255.0).clamp(0.0, 1.0)
            for operation in order:
                factor = factors[operation]
                if factor is None:
                    continue
                if operation == 0:
                    value = TF.adjust_brightness(value, factor)
                elif operation == 1:
                    value = TF.adjust_contrast(value, factor)
                elif operation == 2:
                    value = TF.adjust_saturation(value, factor)
                else:
                    value = TF.adjust_hue(value, factor)
            output.append(value.clamp(0.0, 1.0).mul(255.0))
        return output


def discover_robocasa365_roots(pretrain_root: str | Path) -> list[Path]:
    """Discover only canonical Human300 ``.../{date}/lerobot`` roots."""
    root = Path(pretrain_root)
    discovered: list[Path] = []
    for category in ("atomic", "composite"):
        category_root = root / category
        if not category_root.is_dir():
            continue
        for candidate in category_root.glob("*/*/lerobot"):
            if (candidate / "meta" / "info.json").is_file():
                discovered.append(candidate.resolve())
    discovered.sort(
        key=lambda path: (path.parents[2].name, path.parents[1].name, path.parent.name)
    )
    canonical = [path.parents[1].name for path in discovered]
    if len(canonical) != len(set(canonical)):
        raise ValueError("RoboCasa365 Human300 contains duplicate canonical task roots")
    return discovered


def canonical_task_from_root(root: str | Path) -> str:
    path = Path(root)
    if path.name != "lerobot" or path.parent.parent.parent.name not in {
        "atomic",
        "composite",
    }:
        raise ValueError(f"not a canonical RoboCasa365 Human root: {path}")
    return path.parent.parent.name


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _validate_modality(root: Path) -> None:
    modality_path = root / "meta" / "modality.json"
    if not modality_path.is_file():
        raise FileNotFoundError(
            f"missing RoboCasa365 modality metadata: {modality_path}"
        )
    modality = json.loads(modality_path.read_text(encoding="utf-8"))
    expected_state = {
        "base_position": (0, 3),
        "base_rotation": (3, 7),
        "end_effector_position_relative": (7, 10),
        "end_effector_rotation_relative": (10, 14),
        "gripper_qpos": (14, 16),
    }
    expected_action = {
        "base_motion": (0, 4),
        "control_mode": (4, 5),
        "end_effector_position": (5, 8),
        "end_effector_rotation": (8, 11),
        "gripper_close": (11, 12),
    }
    for section, expected in (("state", expected_state), ("action", expected_action)):
        actual = modality.get(section, {})
        for name, (start, end) in expected.items():
            field = actual.get(name, {})
            if (field.get("start"), field.get("end")) != (start, end):
                raise ValueError(
                    f"{root}: {section}.{name} slice changed; expected {(start, end)}, got {field}"
                )


def robocasa365_episode_split(
    roots: Sequence[str | Path], val_fraction: float, seed: int
) -> tuple[dict[Path, list[int]], dict[Path, list[int]]]:
    """Task-stratified split with at least one held-out episode per task."""
    train: dict[Path, list[int]] = {}
    val: dict[Path, list[int]] = {}
    generator = random.Random(int(seed))
    for root_value in roots:
        root = Path(root_value)
        episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
        ids = sorted(int(record["episode_index"]) for record in episodes)
        if val_fraction <= 0:
            train[root], val[root] = ids, []
            continue
        shuffled = list(ids)
        generator.shuffle(shuffled)
        val_count = min(len(ids) - 1, max(1, int(round(len(ids) * val_fraction))))
        val[root] = sorted(shuffled[:val_count])
        train[root] = sorted(shuffled[val_count:])
    return train, val


@dataclass
class _TaskRecord:
    root: Path
    canonical_task: str
    episodes: list[dict]
    cumulative_lengths: list[int]
    total_windows: int
    chunks_size: int
    tasks: dict[int, str]


class _RunningStats:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(dim, dtype=torch.float64)
        self.minimum = torch.full((dim,), float("inf"), dtype=torch.float32)
        self.maximum = torch.full((dim,), float("-inf"), dtype=torch.float32)

    def update(self, values: Tensor) -> None:
        value = values.reshape(-1, values.shape[-1]).float()
        if value.numel() == 0:
            return
        self.count += int(value.shape[0])
        value64 = value.double()
        self.sum += value64.sum(0)
        self.sum_sq += value64.square().sum(0)
        self.minimum = torch.minimum(self.minimum, value.amin(0))
        self.maximum = torch.maximum(self.maximum, value.amax(0))

    def result(self) -> dict[str, Tensor]:
        if self.count <= 0:
            raise RuntimeError("cannot finalize empty RoboCasa365 statistics")
        mean64 = self.sum / self.count
        variance = (self.sum_sq / self.count - mean64.square()).clamp_min(0)
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean64.float(),
            "std": variance.sqrt().float(),
            "count": torch.tensor([self.count], dtype=torch.float32),
        }


def compute_robocasa365_stats(
    roots: Sequence[str | Path], episode_ids: Optional[dict[Path, list[int]]] = None
) -> dict[str, dict[str, Tensor]]:
    """Compute stats after semantic conversion, using only selected train episodes."""
    action_stats, state_stats = _RunningStats(12), _RunningStats(14)
    for root_value in roots:
        root = Path(root_value)
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        chunks_size = int(info.get("chunks_size", 1000))
        selected = None
        if episode_ids is not None:
            selected = set(episode_ids.get(root, episode_ids.get(root.resolve(), [])))
        for episode in _read_jsonl(root / "meta" / "episodes.jsonl"):
            episode_index = int(episode["episode_index"])
            if selected is not None and episode_index not in selected:
                continue
            path = (
                root
                / "data"
                / f"chunk-{episode_index // chunks_size:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            table = pq.read_table(path, columns=["observation.state", "action"])
            raw_state = torch.tensor(table.column("observation.state").to_pylist())
            raw_action = torch.tensor(table.column("action").to_pylist())
            state_stats.update(robocasa365_state_to_xiaomi14(raw_state))
            action_stats.update(robocasa365_action_to_xiaomi12(raw_action))
    action = action_stats.result()
    # Binary channels already live in {-1,+1}; identity normalization preserves
    # their categorical meaning and lets generic policy denormalization work.
    for dim in ACTION_BINARY_DIMS:
        action["mean"][dim] = 0.0
        action["std"][dim] = 1.0
        action["min"][dim] = -1.0
        action["max"][dim] = 1.0
    return {"action": action, "state": state_stats.result()}


def save_robocasa365_stats(
    stats: dict[str, dict[str, Tensor]], path: str | Path, *, task_names: Sequence[str]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: {name: value.detach().cpu().tolist() for name, value in section.items()}
        for key, section in stats.items()
    }
    payload["metadata"] = {
        "contract_version": ROBOCASA365_CONTRACT_VERSION,
        "task_count": len(task_names),
        "task_names": sorted(str(name) for name in task_names),
        "action_order": [
            "ee_delta_position[3]",
            "ee_delta_rotation_axis_angle[3]",
            "gripper_close[1]",
            "base_motion[4]",
            "control_mode[1]",
        ],
        "action_binary_dims": list(ACTION_BINARY_DIMS),
        "state_order": [
            "ee_relative_position[3]",
            "ee_relative_axis_angle[3]",
            "gripper_qpos[2]",
            "base_position[3]",
            "base_axis_angle[3]",
        ],
    }
    temp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, target)


class RoboCasa365Dataset(Dataset):
    """Task-balanced Human300 reader with the exact OneWM training sample contract."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        history_offsets: Sequence[int] = (-6, -4, -2, 0),
        future_offsets: Sequence[int] = (2, 4, 6, 8, 10, 12, 14, 16),
        repeat_current: int = 1,
        chunk_size: int = 16,
        view_size: tuple[int, int] = (256, 256),
        separator_width: int = 16,
        separator_value: int = 128,
        text_len: int = 128,
        text_dim: int = 4096,
        text_embedding_cache_dir: Optional[str | Path] = None,
        text_prompt_template: str = DEFAULT_TEXT_PROMPT,
        text_cache_encoder_id: str = DEFAULT_TEXT_CACHE_ENCODER_ID,
        normalize_action_stats: Optional[dict[str, Tensor]] = None,
        normalize_state_stats: Optional[dict[str, Tensor]] = None,
        video_transform: Optional[RoboCasa365VideoTransform] = None,
        deterministic_transform_seed: Optional[int] = None,
        val_fraction: float = 0.0,
        split: str = "train",
        split_seed: int = 42,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.roots = [Path(root).resolve() for root in roots]
        if not self.roots:
            raise ValueError("RoboCasa365 dataset roots are empty")
        self.history_offsets = tuple(int(value) for value in history_offsets)
        self.future_offsets = tuple(int(value) for value in future_offsets)
        if (
            self.history_offsets != tuple(sorted(self.history_offsets))
            or not self.history_offsets
        ):
            raise ValueError("history_offsets must be a non-empty sorted sequence")
        if self.history_offsets[-1] != 0:
            raise ValueError("history_offsets must end at the current observation (0)")
        if any(value <= 0 for value in self.future_offsets):
            raise ValueError("future_offsets must be positive")
        self.repeat_current = int(repeat_current)
        self.chunk_size = int(chunk_size)
        self.view_size = tuple(int(value) for value in view_size)
        self.separator_width = int(separator_width)
        self.separator_value = int(separator_value)
        self.text_len, self.text_dim = int(text_len), int(text_dim)
        self.text_embedding_cache_dir = (
            Path(text_embedding_cache_dir) if text_embedding_cache_dir else None
        )
        self.text_prompt_template = text_prompt_template
        self.text_cache_encoder_id = text_cache_encoder_id
        self.action_stats = normalize_action_stats
        self.state_stats = normalize_state_stats
        self.video_transform = video_transform
        self.deterministic_transform_seed = deterministic_transform_seed

        raw_frames = (
            len(self.history_offsets) + self.repeat_current + len(self.future_offsets)
        )
        if (raw_frames - 1) % 4:
            raise ValueError(f"Wan VAE requires 1+4k raw frames, got {raw_frames}")

        train_ids, val_ids = robocasa365_episode_split(
            self.roots, val_fraction, split_seed
        )
        selected_ids = train_ids if split == "train" else val_ids
        if split == "val" and not any(selected_ids.values()):
            selected_ids = train_ids

        self.tasks: list[_TaskRecord] = []
        for root in self.roots:
            _validate_modality(root)
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            if int(info.get("fps", -1)) != 20:
                raise ValueError(f"{root}: expected fps=20")
            features = info.get("features", {})
            for camera in CAMERA_KEYS:
                feature = features.get(camera, {})
                if feature.get("dtype") != "video" or feature.get("shape") != [
                    256,
                    256,
                    3,
                ]:
                    raise ValueError(
                        f"{root}: incompatible camera feature {camera}: {feature}"
                    )
            if features.get("observation.state", {}).get("shape") != [16]:
                raise ValueError(f"{root}: expected 16D observation.state")
            if features.get("action", {}).get("shape") != [12]:
                raise ValueError(f"{root}: expected 12D action")
            allowed = set(int(value) for value in selected_ids[root])
            episodes = [
                record
                for record in _read_jsonl(root / "meta" / "episodes.jsonl")
                if int(record["episode_index"]) in allowed
            ]
            if not episodes:
                raise ValueError(f"{root}: selected {split} split has no episodes")
            cumulative: list[int] = []
            total = 0
            for record in episodes:
                total += int(record["length"])
                cumulative.append(total)
            task_map = {
                int(record["task_index"]): str(record["task"])
                for record in _read_jsonl(root / "meta" / "tasks.jsonl")
            }
            self.tasks.append(
                _TaskRecord(
                    root=root,
                    canonical_task=canonical_task_from_root(root),
                    episodes=episodes,
                    cumulative_lengths=cumulative,
                    total_windows=total,
                    chunks_size=int(info.get("chunks_size", 1000)),
                    tasks=task_map,
                )
            )
        self._length = sum(task.total_windows for task in self.tasks)
        self._episode_cache: OrderedDict[
            tuple[int, int], tuple[Tensor, Tensor, Tensor]
        ] = OrderedDict()
        self._episode_cache_max = 8
        self._text_cache: OrderedDict[Path, tuple[Tensor, Tensor]] = OrderedDict()
        self._text_cache_max = 256

    def __len__(self) -> int:
        return self._length

    def _balanced_coordinate(self, index: int) -> tuple[int, int, int]:
        task_index = int(index) % len(self.tasks)
        task = self.tasks[task_index]
        local = (int(index) // len(self.tasks)) % task.total_windows
        episode_position = bisect.bisect_right(task.cumulative_lengths, local)
        previous = (
            0
            if episode_position == 0
            else task.cumulative_lengths[episode_position - 1]
        )
        return task_index, episode_position, local - previous

    @staticmethod
    def _normalize(value: Tensor, stats: Optional[dict[str, Tensor]]) -> Tensor:
        if stats is None:
            return value
        return ((value - stats["mean"]) / stats["std"].clamp_min(1e-6)).clamp(-5, 5)

    def _episode_paths(
        self, task: _TaskRecord, episode_index: int
    ) -> tuple[Path, list[Path]]:
        chunk = episode_index // task.chunks_size
        parquet = (
            task.root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        videos = [
            task.root
            / "videos"
            / f"chunk-{chunk:03d}"
            / key
            / f"episode_{episode_index:06d}.mp4"
            for key in CAMERA_KEYS
        ]
        return parquet, videos

    def _load_episode(
        self, task_index: int, episode_position: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        task = self.tasks[task_index]
        episode_index = int(task.episodes[episode_position]["episode_index"])
        key = (task_index, episode_index)
        if key in self._episode_cache:
            self._episode_cache.move_to_end(key)
            return self._episode_cache[key]
        parquet, _ = self._episode_paths(task, episode_index)
        table = pq.read_table(
            parquet, columns=["observation.state", "action", "task_index"]
        )
        payload = (
            torch.tensor(table.column("observation.state").to_pylist()),
            torch.tensor(table.column("action").to_pylist()),
            torch.tensor(table.column("task_index").to_pylist(), dtype=torch.long),
        )
        self._episode_cache[key] = payload
        if len(self._episode_cache) > self._episode_cache_max:
            self._episode_cache.popitem(last=False)
        return payload

    def _load_context(self, instruction: str) -> tuple[Tensor, Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("RoboCasa365 requires data.text_embedding_cache_dir")
        path = text_cache_path(
            self.text_embedding_cache_dir,
            instruction,
            self.text_len,
            self.text_prompt_template,
            self.text_cache_encoder_id,
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"missing text cache for {instruction!r}: {path}; run precompute_text_cache"
            )
        if path not in self._text_cache:
            self._text_cache[path] = load_text_cache(path, self.text_len, self.text_dim)
            if len(self._text_cache) > self._text_cache_max:
                self._text_cache.popitem(last=False)
        else:
            self._text_cache.move_to_end(path)
        return self._text_cache[path]

    def __getitem__(self, index: int) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(20):
            sample_index = int(index) if attempt == 0 else random.randrange(len(self))
            try:
                return self._get_item_once(sample_index)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"failed to load RoboCasa365 sample after 20 attempts, index={index}: {last_error}"
        ) from last_error

    def _get_item_once(self, index: int) -> dict:
        task_index, episode_position, start = self._balanced_coordinate(index)
        task = self.tasks[task_index]
        episode = task.episodes[episode_position]
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        raw_state, raw_action, task_indices = self._load_episode(
            task_index, episode_position
        )
        length = min(length, raw_state.shape[0], raw_action.shape[0])
        start = min(start, length - 1)

        action_positions = torch.arange(self.chunk_size)
        action_is_pad = start + action_positions >= length
        action_indices = (start + action_positions).clamp_max(length - 1)
        action = robocasa365_action_to_xiaomi12(raw_action[action_indices])
        action = self._normalize(action, self.action_stats)

        history_indices = [max(0, start + offset) for offset in self.history_offsets]
        state_history = robocasa365_state_to_xiaomi14(raw_state[history_indices])
        state_history = self._normalize(state_history, self.state_stats)

        video_offsets = (
            list(self.history_offsets)
            + [0] * self.repeat_current
            + list(self.future_offsets)
        )
        video_indices = [
            min(max(0, start + offset), length - 1) for offset in video_offsets
        ]
        image_is_pad = torch.tensor(
            [False] * (len(self.history_offsets) + self.repeat_current)
            + [start + offset >= length for offset in self.future_offsets],
            dtype=torch.bool,
        )
        _, video_paths = self._episode_paths(task, episode_index)
        cameras = [
            _decode_robocasa365_video_frames(path, video_indices)
            for path in video_paths
        ]
        if self.video_transform is not None:
            if self.deterministic_transform_seed is None:
                cameras = self.video_transform(cameras)
            else:
                sample_seed = (
                    int(self.deterministic_transform_seed) * 1_000_003 + int(index)
                ) % (2**63 - 1)
                with isolated_cpu_rng(sample_seed):
                    cameras = self.video_transform(cameras)
        strip = compose_robocasa365_views(
            cameras[0],
            cameras[1],
            cameras[2],
            output_view_size=self.view_size,
            separator_width=self.separator_width,
            separator_value=self.separator_value,
        )
        video = (strip.float() / 255.0 * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous()

        frame_task_index = int(task_indices[start])
        instruction = task.tasks.get(frame_task_index)
        if not instruction:
            tasks = episode.get("tasks") or []
            instruction = str(tasks[0]) if tasks else task.canonical_task
        context, context_mask = self._load_context(instruction)
        return {
            "video": video,
            "action": action,
            "context": context,
            "context_mask": context_mask,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "proprio_history": state_history,
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "frame_index": torch.tensor(start, dtype=torch.int64),
            "canonical_task_index": torch.tensor(task_index, dtype=torch.int64),
        }
