"""Minimal LeRobot-format dataset loader.

Reads the canonical LeRobot v2 / v2.1 layout::

    <root>/
        meta/info.json                                # global feature metadata
        meta/episodes.jsonl                           # per-episode metadata (length, task, etc.)
        data/chunk-000/episode_000000.parquet         # per-timestep action / state rows
        videos/chunk-000/<video_key>/episode_000000.mp4
        text_embedding_cache_dir/*.pt                 # trusted precomputed T5 cache [L, D]

Produces samples matching the schema expected by
:class:`onewm.wam.mot_wam.MoTWAM.training_step`::

    {
        "video":         [3, T, H, W]        in [-1, 1],
        "action":        [chunk_size, A]     normalized actions,
        "context":       [L, text_dim]       (zero-padded T5 embeddings),
        "context_mask":  [L]                 bool,
        "action_is_pad": [chunk_size]        bool,
        "image_is_pad":  [T]                 bool,
    }

For tests / smoke runs without a real dataset, use
:class:`LeRobotSyntheticDataset` which keeps the original stub behaviour.
"""

from __future__ import annotations
import hashlib
import json
import os
import random
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from onewm.config import DataConfig

DEFAULT_TEXT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_TEXT_CACHE_ENCODER_ID = "wan22ti2v5b"


def format_text_prompt(task: str, template: str = DEFAULT_TEXT_PROMPT) -> str:
    return template.format(task=task)


def text_cache_filename(
    prompt: str, context_len: int, encoder_id: str = DEFAULT_TEXT_CACHE_ENCODER_ID
) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{digest}.t5_len{context_len}.{encoder_id}.pt"


def text_cache_path(
    cache_dir: str | Path,
    task: str,
    context_len: int,
    prompt_template: str = DEFAULT_TEXT_PROMPT,
    encoder_id: str = DEFAULT_TEXT_CACHE_ENCODER_ID,
) -> Path:
    prompt = format_text_prompt(task, prompt_template)
    return Path(cache_dir) / text_cache_filename(prompt, context_len, encoder_id)


def load_text_cache(
    path: str | Path, text_len: int, text_dim: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or "context" not in payload
        or "mask" not in payload
    ):
        raise ValueError(f"Text cache must be a dict with context/mask: {path}")
    context = payload["context"].float()
    mask = payload["mask"].bool()
    if context.dim() == 3:
        context = context[0]
    if mask.dim() == 2:
        mask = mask[0]
    out = torch.zeros(text_len, text_dim, dtype=torch.float32)
    out_mask = torch.zeros(text_len, dtype=torch.bool)
    n = min(text_len, context.shape[0], mask.shape[0])
    d = min(text_dim, context.shape[-1])
    if n > 0 and d > 0:
        out[:n, :d] = context[:n, :d]
        out_mask[:n] = mask[:n]
    return (out, out_mask)


def save_text_cache(
    path: str | Path, context: torch.Tensor, mask: torch.Tensor, prompt: str, task: str
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = context.detach().cpu()
    mask = mask.detach().cpu().bool()
    if mask.ndim == 1 and context.ndim == 2 and (mask.numel() == context.shape[0]):
        n = int(mask.sum().item())
        if 0 < n < context.shape[0]:
            context = context[:n]
            mask = mask[:n]
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    torch.save(
        {
            "context": context.to(torch.bfloat16).contiguous(),
            "mask": mask.contiguous(),
            "prompt": prompt,
            "task": task,
        },
        tmp,
    )
    tmp.replace(path)


def iter_task_records(dataset_dir: str | Path) -> list[dict]:
    path = Path(dataset_dir) / "meta" / "tasks.jsonl"
    if path.is_file():
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    parquet_path = Path(dataset_dir) / "meta" / "tasks.parquet"
    if not parquet_path.is_file():
        return []
    records = pq.read_table(parquet_path).to_pylist()
    normalized = []
    for record in records:
        task = record.get("task", record.get("__index_level_0__"))
        if task is not None:
            normalized.append(
                {
                    "task_index": int(record.get("task_index", len(normalized))),
                    "task": str(task),
                }
            )
    return normalized


def collect_lerobot_tasks(roots: list[str | Path]) -> list[str]:
    seen: set[str] = set()
    tasks: list[str] = []
    for root in roots:
        for record in iter_task_records(root):
            task = str(record["task"])
            if task not in seen:
                seen.add(task)
                tasks.append(task)
    return tasks


class LeRobotSyntheticDataset(Dataset):
    """Generates synthetic samples with the correct schema.

    Use this for unit tests and quick smoke runs that do not need a real
    LeRobot dataset on disk.
    """

    def __init__(
        self,
        config: DataConfig,
        action_dim: int = 7,
        chunk_size: int = 16,
        text_dim: int = 4096,
        is_training: bool = True,
        length: Optional[int] = None,
        proprio_dim: Optional[int] = None,
    ) -> None:
        self.config = config
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.text_dim = text_dim
        self.is_training = is_training
        self.num_frames = config.num_frames
        self.video_num_frames = len(
            range(0, self.num_frames, max(1, config.action_freq_ratio))
        )
        self.proprio_dim = (
            None if not proprio_dim or proprio_dim <= 0 else int(proprio_dim)
        )
        self.H, self.W = config.video_size
        self._length = length if length is not None else 1000 if is_training else 100

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        T = self.video_num_frames
        video = torch.rand(3, T, self.H, self.W) * 2 - 1
        action = torch.randn(self.chunk_size, self.action_dim)
        L = 77
        context = torch.randn(L, self.text_dim)
        context_mask = torch.ones(L, dtype=torch.bool)
        sample = {
            "video": video,
            "action": action,
            "context": context,
            "context_mask": context_mask,
            "action_is_pad": torch.zeros(self.chunk_size, dtype=torch.bool),
            "image_is_pad": torch.zeros(T, dtype=torch.bool),
        }
        if self.proprio_dim is not None:
            sample["proprio"] = torch.randn(self.chunk_size, self.proprio_dim)
            sample["proprio_is_pad"] = torch.zeros(self.chunk_size, dtype=torch.bool)
        return sample


def _resize_frames(frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a [T, C, H, W] uint8 / float tensor to [T, C, target_H, target_W]."""
    import torch.nn.functional as F

    t, c, h, w = frames.shape
    if (h, w) == size:
        return frames
    flat = frames.float().reshape(t * c, 1, h, w)
    out = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    return out.reshape(t, c, *size)


def compose_camera_views(
    camera_frames: Sequence[torch.Tensor],
    *,
    concat_mode: str,
    view_size: tuple[int, int],
) -> torch.Tensor:
    """Resize and concatenate camera clips using the LeRobot layout contract.

    Every input has shape [T, C, H, W] and is resized independently to
    view_size before concatenation. This function is shared by offline loading
    and online benchmark adapters so camera ordering and geometry cannot
    silently drift between training and evaluation.
    """
    if concat_mode not in {"horizontal", "vertical"}:
        raise ValueError(
            f"compose_camera_views supports horizontal or vertical concat, got {concat_mode!r}"
        )
    if not camera_frames:
        raise ValueError("camera_frames must contain at least one view")
    resized: list[torch.Tensor] = []
    expected_prefix: Optional[tuple[int, int]] = None
    for index, frames in enumerate(camera_frames):
        if frames.ndim != 4:
            raise ValueError(
                f"camera_frames[{index}] must be [T,C,H,W], got {tuple(frames.shape)}"
            )
        prefix = (int(frames.shape[0]), int(frames.shape[1]))
        if expected_prefix is None:
            expected_prefix = prefix
        elif prefix != expected_prefix:
            raise ValueError(
                f"all camera clips must share [T,C], got {expected_prefix} and {prefix}"
            )
        resized.append(_resize_frames(frames, view_size))
    if len(resized) == 1:
        return resized[0]
    concat_dim = -1 if concat_mode == "horizontal" else -2
    return torch.cat(resized, dim=concat_dim)


def _decode_video_frames(video_path: Path, frame_indices: list[int]) -> torch.Tensor:
    """Decode specific frames from an mp4 file.

    Returns ``[T, 3, H, W]`` uint8 tensor in RGB order.
    """
    try:
        import av

        target = sorted(set((int(i) for i in frame_indices)))
        target_set = set(target)
        max_idx = max(target)
        decoded: dict[int, torch.Tensor] = {}
        with av.open(str(video_path), timeout=30) as container:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            stream.codec_context.thread_count = 1
            try:
                for i, frame in enumerate(container.decode(stream)):
                    if i in target_set:
                        arr = frame.to_ndarray(format="rgb24")
                        decoded[i] = torch.from_numpy(arr.copy()).permute(2, 0, 1)
                    if i >= max_idx:
                        break
            finally:
                stream.codec_context.close()
    except Exception as e:
        raise RuntimeError(
            f"failed to decode video frames from {video_path}: {e}"
        ) from e
    if not decoded:
        raise RuntimeError(f"failed to decode any requested frames from {video_path}")
    H, W = next(iter(decoded.values())).shape[-2:]
    out = torch.empty(len(frame_indices), 3, H, W, dtype=torch.uint8)
    for j, i in enumerate(frame_indices):
        if i in decoded:
            out[j] = decoded[i]
        else:
            ref = max((k for k in decoded if k < i), default=None)
            if ref is None:
                ref = min(decoded) if decoded else None
            out[j] = (
                decoded[ref]
                if ref is not None
                else torch.zeros(3, H, W, dtype=torch.uint8)
            )
    return out


def _read_episodes_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _stats_from_json(raw: dict) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32) for key, value in raw.items()
    }


def _stats_to_json(stats: dict[str, torch.Tensor]) -> dict:
    return {key: value.detach().cpu().tolist() for key, value in stats.items()}


def load_lerobot_stats(path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "action" in raw or "state" in raw:
        return {
            key: _stats_from_json(value)
            for key, value in raw.items()
            if key in {"action", "state"} and isinstance(value, dict)
        }
    return {"action": _stats_from_json(raw)}


def save_lerobot_stats(
    stats: dict[str, dict[str, torch.Tensor]], path: str | Path
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: _stats_to_json(value) for key, value in stats.items()}
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, target)


def load_action_stats(path: str | Path) -> dict[str, torch.Tensor]:
    stats = load_lerobot_stats(path)
    if "action" not in stats:
        raise KeyError(f"No action stats found in {path}")
    return stats["action"]


def save_action_stats(stats: dict[str, torch.Tensor], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _stats_to_json(stats)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, target)


def _compute_column_stats(
    roots: list[str | Path], column_key: str, label: str
) -> dict[str, torch.Tensor]:
    count = 0
    total = None
    total_sq = None
    value_min = None
    value_max = None
    for root_str in roots:
        root = Path(root_str)
        meta_dir = root / "meta"
        ep_path = meta_dir / "episodes.jsonl"
        if ep_path.is_file():
            episodes = _read_episodes_jsonl(ep_path)
        else:
            episodes = []
            for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
                episode_index = int(path.stem.split("_")[-1])
                episodes.append({"episode_index": episode_index})
        chunks_size = int(
            json.loads((meta_dir / "info.json").read_text()).get("chunks_size", 1000)
        )
        for ep in episodes:
            episode_index = int(ep["episode_index"])
            chunk = episode_index // chunks_size
            parquet_path = (
                root
                / "data"
                / f"chunk-{chunk:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            if not parquet_path.is_file():
                continue
            table = pq.read_table(parquet_path, columns=[column_key])
            value = torch.tensor(
                table.column(column_key).to_pylist(), dtype=torch.float32
            )
            if value.numel() == 0:
                continue
            if total is None:
                total = torch.zeros(value.shape[-1], dtype=torch.float64)
                total_sq = torch.zeros(value.shape[-1], dtype=torch.float64)
                value_min = torch.full(
                    (value.shape[-1],), float("inf"), dtype=torch.float32
                )
                value_max = torch.full(
                    (value.shape[-1],), float("-inf"), dtype=torch.float32
                )
            count += int(value.shape[0])
            value_f64 = value.to(torch.float64)
            total += value_f64.sum(dim=0)
            total_sq += (value_f64 * value_f64).sum(dim=0)
            value_min = torch.minimum(value_min, value.amin(dim=0))
            value_max = torch.maximum(value_max, value.amax(dim=0))
    if (
        count == 0
        or total is None
        or total_sq is None
        or (value_min is None)
        or (value_max is None)
    ):
        raise RuntimeError(
            f"No {label} values found while computing stats for roots={roots}"
        )
    mean = (total / count).to(torch.float32)
    var = (total_sq / count - total / count * total / count).clamp_min(0.0)
    return {
        "min": value_min,
        "max": value_max,
        "mean": mean,
        "std": torch.sqrt(var).to(torch.float32),
        "count": torch.tensor([count], dtype=torch.float32),
    }


def compute_action_stats(
    roots: list[str | Path], action_key: str = "action"
) -> dict[str, torch.Tensor]:
    return _compute_column_stats(roots, action_key, "actions")


def compute_state_stats(
    roots: list[str | Path], state_key: str = "observation.state"
) -> dict[str, torch.Tensor]:
    return _compute_column_stats(roots, state_key, "states")


class LeRobotDataset(Dataset):
    """Minimal reader for LeRobot v2 / v2.1 datasets.

    Args:
        root: dataset root directory containing ``meta/``, ``data/``, ``videos/``.
        video_key: feature key for the video stream
            (e.g. ``observation.images.cam_high``).
        video_keys: optional multi-camera stream keys. Frames are resized per
            camera and concatenated.
        action_key: column name for the action vector inside the parquet.
        state_key: column name for the proprio/state vector inside the parquet.
        num_frames: action-time observation grid length. With ``action_freq_ratio > 1``,
            video frames are sampled as ``range(0, num_frames, action_freq_ratio)``.
        chunk_size: number of action steps per sample.
        video_size: (H, W) target size after resize.
        text_len: T5 sequence length to pad/truncate to.
        text_dim: T5 hidden dim (default 4096 for UMT5-XXL).
        episodes: optional explicit list of episode indices to use.
        action_freq_ratio: number of action steps per video frame
            (1 means action and video share the same timestep grid).
        normalize_action_stats: optional dict with action stats for min/max or
            z-score normalization (each stat shape ``[action_dim]``).
        action_norm_mode: ``minmax`` maps raw actions to [-1, 1];
            ``zscore`` uses mean/std.
        normalize_state_stats: optional dict with proprio/state stats for the
            same normalization modes.
        state_norm_mode: ``minmax`` maps raw proprio/state to [-1, 1].
    """

    def __init__(
        self,
        root: str | Path,
        video_key: str = "observation.images.cam_high",
        video_keys: Optional[list[str]] = None,
        concat_multi_camera: str = "horizontal",
        action_key: str = "action",
        state_key: str = "observation.state",
        num_frames: int = 8,
        chunk_size: int = 16,
        video_size: tuple[int, int] = (256, 256),
        text_len: int = 512,
        text_dim: int = 4096,
        episodes: Optional[list[int]] = None,
        action_freq_ratio: int = 1,
        normalize_action_stats: Optional[dict] = None,
        action_norm_mode: str = "minmax",
        normalize_state_stats: Optional[dict] = None,
        state_norm_mode: str = "minmax",
        delta_action_dim_mask: Optional[list[bool]] = None,
        proprio_dim: Optional[int] = None,
        text_embedding_cache_dir: Optional[str | Path] = None,
        text_prompt_template: str = DEFAULT_TEXT_PROMPT,
        text_cache_encoder_id: str = DEFAULT_TEXT_CACHE_ENCODER_ID,
    ) -> None:
        self.root = Path(root)
        self.video_key = video_key
        self.video_keys = list(video_keys) if video_keys else [video_key]
        self.concat_multi_camera = concat_multi_camera
        if self.concat_multi_camera not in {"horizontal", "vertical"}:
            raise ValueError("concat_multi_camera must be horizontal or vertical")
        self.action_key = action_key
        self.state_key = state_key
        self.num_frames = num_frames
        self.chunk_size = chunk_size
        self.video_size = video_size
        self.text_len = text_len
        self.text_dim = text_dim
        self.action_freq_ratio = action_freq_ratio
        self.normalize_action_stats = normalize_action_stats
        self.action_norm_mode = action_norm_mode
        self.normalize_state_stats = normalize_state_stats
        self.state_norm_mode = state_norm_mode
        self.proprio_dim = (
            None if not proprio_dim or proprio_dim <= 0 else int(proprio_dim)
        )
        self.text_embedding_cache_dir = (
            Path(text_embedding_cache_dir) if text_embedding_cache_dir else None
        )
        self.text_prompt_template = text_prompt_template
        self.text_cache_encoder_id = text_cache_encoder_id
        self._text_cache: "OrderedDict[Path, tuple[torch.Tensor, torch.Tensor]]" = (
            OrderedDict()
        )
        self._text_cache_max = 1024
        self.delta_action_dim_mask = (
            torch.as_tensor(delta_action_dim_mask, dtype=torch.bool)
            if delta_action_dim_mask is not None
            else None
        )
        meta_dir = self.root / "meta"
        if not meta_dir.is_dir():
            raise FileNotFoundError(f"`{meta_dir}` not found; not a LeRobot dataset.")
        self.info = json.loads((meta_dir / "info.json").read_text())
        ep_path = meta_dir / "episodes.jsonl"
        if ep_path.is_file():
            all_eps = _read_episodes_jsonl(ep_path)
        else:
            all_eps = self._scan_episodes_from_data()
        if episodes is not None:
            ep_set = set((int(i) for i in episodes))
            self.episodes = [
                e for e in all_eps if int(e.get("episode_index", -1)) in ep_set
            ]
        else:
            self.episodes = all_eps
        if not self.episodes:
            raise ValueError(f"No episodes found at {ep_path}")
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self._episode_to_task_text: dict[int, str] = {}
        self._load_task_metadata(meta_dir / "tasks.jsonl")
        self._task_index_to_text: dict[int, str] = {}
        for record in iter_task_records(self.root):
            try:
                self._task_index_to_text[int(record["task_index"])] = str(
                    record["task"]
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._episode_task_indices: dict[int, list[int]] = {}
        self._samples: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = int(ep.get("length", 0))
            if ep_len <= 0:
                episode_index = int(ep["episode_index"])
                parquet_path, _ = self._episode_paths(episode_index)
                try:
                    ep_len = pq.read_metadata(parquet_path).num_rows
                    ep["length"] = ep_len
                except Exception:
                    ep_len = 1
            for start in range(max(ep_len, 1)):
                self._samples.append((ep_idx, start))
        if not self._samples:
            raise ValueError(f"No training samples found at {self.root}")

    def _load_task_metadata(self, tasks_jsonl: Path) -> None:
        if not tasks_jsonl.is_file():
            return
        for ep in self.episodes:
            ep_idx = int(ep.get("episode_index", -1))
            tasks_field = ep.get("tasks") or []
            if not tasks_field:
                continue
            task = tasks_field[0] if isinstance(tasks_field, list) else tasks_field
            self._episode_to_task_text[ep_idx] = str(task)

    def _fastwam_cache_path(self, task: str) -> Path | None:
        if self.text_embedding_cache_dir is None:
            return None
        return text_cache_path(
            self.text_embedding_cache_dir,
            task,
            self.text_len,
            self.text_prompt_template,
            self.text_cache_encoder_id,
        )

    def _scan_episodes_from_data(self) -> list[dict]:
        """Discover episodes by scanning ``data/chunk-*/episode_*.parquet``.

        Returns minimal dicts ``{"episode_index": i, "length": N}`` so the
        rest of the pipeline does not care whether episodes.jsonl exists.
        """
        import re
        import pyarrow.parquet as pq

        data_dir = self.root / "data"
        if not data_dir.is_dir():
            return []
        eps: list[dict] = []
        seen: set[int] = set()
        ep_re = re.compile("episode_(\\d+)\\.parquet$")
        for parquet in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
            m = ep_re.search(parquet.name)
            if not m:
                continue
            idx = int(m.group(1))
            if idx in seen:
                continue
            seen.add(idx)
            try:
                length = pq.read_metadata(parquet).num_rows
            except Exception:
                length = 0
            eps.append({"episode_index": idx, "length": int(length)})
        eps.sort(key=lambda e: e["episode_index"])
        return eps

    def __len__(self) -> int:
        return len(self._samples)

    def _episode_paths(self, episode_index: int) -> tuple[Path, list[Path]]:
        chunk_id = episode_index // self.chunks_size
        parquet = (
            self.root
            / "data"
            / f"chunk-{chunk_id:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        videos = [
            self.root
            / "videos"
            / f"chunk-{chunk_id:03d}"
            / key
            / f"episode_{episode_index:06d}.mp4"
            for key in self.video_keys
        ]
        return (parquet, videos)

    def _load_actions(self, parquet_path: Path) -> torch.Tensor:
        table = pq.read_table(parquet_path, columns=[self.action_key])
        col = table.column(self.action_key).to_pylist()
        return torch.tensor(col, dtype=torch.float32)

    def _load_proprio(self, parquet_path: Path) -> torch.Tensor | None:
        try:
            table = pq.read_table(parquet_path, columns=[self.state_key])
        except Exception:
            return None
        col = table.column(self.state_key).to_pylist()
        return torch.tensor(col, dtype=torch.float32)

    def _resolve_task_text(
        self, parquet_path: Path, episode_index: int, frame: int
    ) -> str | None:
        """Instruction for the conditioning frame.

        Uses the sampled frame's ``task_index`` (Fast-WAM/starVLA-style
        frame-level instruction diversity). Falls back to the per-episode first
        instruction when the parquet has no ``task_index`` column.
        """
        if self._task_index_to_text:
            task_indices = self._episode_task_indices.get(episode_index)
            if task_indices is None:
                try:
                    table = pq.read_table(parquet_path, columns=["task_index"])
                    task_indices = [
                        int(x) for x in table.column("task_index").to_pylist()
                    ]
                except Exception:
                    task_indices = []
                self._episode_task_indices[episode_index] = task_indices
            if task_indices:
                f = min(max(int(frame), 0), len(task_indices) - 1)
                text = self._task_index_to_text.get(int(task_indices[f]))
                if text:
                    return text
        return self._episode_to_task_text.get(episode_index)

    def _load_t5(self, task: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not task:
            raise KeyError(
                f"No task text resolved for text conditioning in {self.root}"
            )
        cache = self._fastwam_cache_path(task)
        if cache is None:
            raise ValueError(
                "data.text_embedding_cache_dir must be set for LeRobot text conditioning"
            )
        if not cache.is_file():
            raise FileNotFoundError(
                f"Missing text cache for task {task!r}: {cache}. Run `python -m onewm.tools.precompute_text_cache` or let training precompute it before workers start."
            )
        if cache not in self._text_cache:
            self._text_cache[cache] = load_text_cache(
                cache, self.text_len, self.text_dim
            )
            if len(self._text_cache) > self._text_cache_max:
                self._text_cache.popitem(last=False)
        else:
            self._text_cache.move_to_end(cache)
        return self._text_cache[cache]

    @staticmethod
    def _stat_tensor(
        stats: dict[str, torch.Tensor],
        key: str,
        dim: int,
        dtype: torch.dtype,
        label: str,
    ) -> torch.Tensor:
        value = stats[key].to(dtype)
        if value.numel() < dim:
            raise ValueError(
                f"{label} stats {key} dim {value.numel()} is smaller than data dim {dim}"
            )
        return value[:dim]

    @classmethod
    def _normalize_values(
        cls,
        values: torch.Tensor,
        stats: dict[str, torch.Tensor] | None,
        mode: str,
        label: str,
    ) -> torch.Tensor:
        if stats is None:
            return values
        dim = int(values.shape[-1])
        if mode == "zscore":
            mean = cls._stat_tensor(stats, "mean", dim, values.dtype, label)
            std = cls._stat_tensor(stats, "std", dim, values.dtype, label).clamp_min(
                1e-06
            )
            return ((values - mean) / std).clamp(-5.0, 5.0)
        if mode != "minmax":
            raise ValueError(f"Unsupported {label}_norm_mode: {mode}")
        value_min = cls._stat_tensor(stats, "min", dim, values.dtype, label)
        value_max = cls._stat_tensor(stats, "max", dim, values.dtype, label)
        value_range = (value_max - value_min).clamp_min(1e-06)
        normalized = 2.0 * (values - value_min) / value_range - 1.0
        return normalized.clamp(-5.0, 5.0)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self._normalize_values(
            action, self.normalize_action_stats, self.action_norm_mode, "action"
        )

    def _normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return self._normalize_values(
            proprio, self.normalize_state_stats, self.state_norm_mode, "state"
        )

    def __getitem__(self, idx: int) -> dict:
        last_error: Exception | None = None
        for attempt in range(100):
            sample_idx = idx if attempt == 0 else random.randrange(len(self._samples))
            try:
                return self._get_item_once(sample_idx)
            except Exception as e:
                last_error = e
        raise RuntimeError(
            f"failed to load sample after 100 retries, original idx={idx}: {last_error}"
        ) from last_error

    def _get_item_once(self, idx: int) -> dict:
        ep_idx, start = self._samples[idx]
        ep = self.episodes[ep_idx]
        episode_index = int(ep["episode_index"])
        ep_len = int(ep.get("length", 0))
        parquet_path, video_paths = self._episode_paths(episode_index)
        actions = self._load_actions(parquet_path)
        if ep_len <= 0:
            ep_len = actions.shape[0]
        ep_len = min(ep_len, actions.shape[0])
        start = min(int(start), max(0, ep_len - 1))
        action_indices = [min(start + i, ep_len - 1) for i in range(self.chunk_size)]
        action = actions[action_indices]
        action_is_pad = torch.tensor(
            [start + i >= ep_len for i in range(self.chunk_size)], dtype=torch.bool
        )
        if self.delta_action_dim_mask is not None and bool(action_is_pad.any().item()):
            dim_mask = self.delta_action_dim_mask.to(action.device)
            if dim_mask.numel() != action.shape[-1]:
                raise ValueError(
                    f"delta_action_dim_mask length {dim_mask.numel()} does not match action dim {action.shape[-1]}"
                )
            action = action.clone()
            action[action_is_pad.unsqueeze(-1) & dim_mask.unsqueeze(0)] = 0.0
        action = self._normalize_action(action)
        step = max(1, self.action_freq_ratio)
        video_offsets = list(range(0, self.num_frames, step))
        video_indices = [min(start + offset, ep_len - 1) for offset in video_offsets]
        image_is_pad = torch.tensor(
            [start + offset >= ep_len for offset in video_offsets], dtype=torch.bool
        )
        camera_frames = []
        for video_path in video_paths:
            frames_uint8 = _decode_video_frames(video_path, video_indices)
            camera_frames.append(frames_uint8)
        frames = (
            compose_camera_views(
                camera_frames,
                concat_mode=self.concat_multi_camera,
                view_size=self.video_size,
            ).float()
            / 255.0
        )
        video = (frames * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous()
        task_text = self._resolve_task_text(parquet_path, episode_index, start)
        context, context_mask = self._load_t5(task_text)
        sample = {
            "video": video,
            "action": action,
            "context": context,
            "context_mask": context_mask,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }
        if self.proprio_dim is not None:
            proprio_all = self._load_proprio(parquet_path)
            if proprio_all is None:
                raise ValueError(
                    f"framework.proprio_dim={self.proprio_dim} requires state column {self.state_key!r} in {parquet_path}"
                )
            if proprio_all.shape[-1] < self.proprio_dim:
                raise ValueError(
                    f"{self.state_key} dim {proprio_all.shape[-1]} is smaller than configured proprio_dim={self.proprio_dim}"
                )
            proprio = proprio_all[action_indices, : self.proprio_dim]
            proprio = self._normalize_proprio(proprio)
            sample["proprio"] = proprio
            sample["proprio_is_pad"] = action_is_pad.clone()
        return sample
