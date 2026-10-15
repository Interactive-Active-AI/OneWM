"""Strict packed LeRobot v3 reader for the filtered dual-Franka recordings."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from onewm.data.franka_transforms import (
    CAMERAS,
    CONTRACT,
    VIDEO_OFFSETS,
    compose,
    normalize,
)
from onewm.data.lerobot import load_text_cache, text_cache_path
from onewm.data.video_augmentation import MultiViewVideoAugmentation

TASKS = ("prepare_breakfast", "put_block_in_drawer", "stack_blocks")
DATA_ROOT = Path("data/franka")
REVISION = "franka-v3-absolute-joint"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_records(root):
    records = {}
    for task in TASKS:
        task_root = Path(root) / task
        for path in sorted((task_root / "meta/episodes").glob("*/*.parquet")):
            for row in pq.read_table(path).to_pylist():
                record = {k: v for k, v in row.items() if not k.startswith("stats/")}
                record["task"] = task
                key = f"{task}/{int(row['episode_index']):06d}"
                if key in records:
                    raise ValueError(f"duplicate episode {key}")
                records[key] = record
    return records


def data_path(root, record):
    return (
        Path(root)
        / record["task"]
        / "data"
        / f"chunk-{record['data/chunk_index']:03d}"
        / f"file-{record['data/file_index']:03d}.parquet"
    )


def episode_rows(root, record, table=None):
    table = pq.read_table(data_path(root, record)) if table is None else table
    first, last = int(record["dataset_from_index"]), int(record["dataset_to_index"])
    rows = table.slice(first, last - first)
    if len(rows) != int(record["length"]):
        raise ValueError("episode file-local offsets/length mismatch")
    result = {
        key: np.asarray(rows[key].to_pylist())
        for key in (
            "observation.state",
            "action",
            "frame_index",
            "episode_index",
            "task_index",
            "timestamp",
        )
    }
    if not np.all(result["episode_index"] == record["episode_index"]):
        raise ValueError("packed episode identity mismatch")
    if not np.array_equal(result["frame_index"], np.arange(len(rows))):
        raise ValueError("noncontiguous local frame indices")
    return result


def video_location(root, record, camera):
    key = f"videos/observation.images.{camera}"
    path = (
        Path(root)
        / record["task"]
        / "videos"
        / f"observation.images.{camera}"
        / f"chunk-{record[key + '/chunk_index']:03d}"
        / f"file-{record[key + '/file_index']:03d}.mp4"
    )
    return path, round(float(record[key + "/from_timestamp"]) * 30)


def decode_frames(path, indices):
    """Exact PTS-based frame lookup. Never substitute a neighbouring frame."""
    import av

    targets = set(map(int, indices))
    found = {}
    with av.open(str(path), timeout=30) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate)
        if abs(fps - 30) > 1e-5:
            raise ValueError(f"expected 30 FPS: {path}: {fps}")
        container.seek(
            int(max(0, min(targets) - 2) / fps / float(stream.time_base)),
            stream=stream,
            backward=True,
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            index = round(float(frame.pts * stream.time_base) * fps)
            if index in targets:
                found[index] = torch.from_numpy(
                    frame.to_ndarray(format="rgb24").copy()
                ).permute(2, 0, 1)
            if len(found) == len(targets) or index > max(targets) + 1:
                break
    if targets - found.keys():
        raise ValueError(
            f"missing exact frames in {path}: {sorted(targets - found.keys())}"
        )
    return torch.stack([found[int(i)] for i in indices])


class FrankaDataset(Dataset):
    def __init__(self, config, is_training=True, text_dim=4096):
        self.config = config
        self.root = Path(config.root)
        self.training = is_training
        self.manifest = json.loads(Path(config.split_manifest).read_text())
        if self.manifest["contract"] != CONTRACT or self.manifest["root"] != str(
            self.root
        ):
            raise ValueError("Franka manifest contract/root mismatch")
        self.stats = json.loads(Path(config.action_stats_path).read_text())
        if self.stats["manifest_sha256"] != digest(config.split_manifest):
            raise ValueError("Franka stats belong to another split")
        self.records = read_records(self.root)
        self.windows = self.manifest["train_windows" if is_training else "val_windows"]
        self.text_dim = text_dim
        self._tables = OrderedDict()
        self._episodes = OrderedDict()
        self._text = {}
        self.augmentation = MultiViewVideoAugmentation(
            probability=config.video_augmentation_probability
            if is_training and config.video_augmentation_enabled
            else 0,
            crop_ratio=config.video_random_crop_ratio,
            brightness=config.video_brightness,
            contrast=config.video_contrast,
            saturation=config.video_saturation,
            hue=config.video_hue,
        )

    def __len__(self):
        return len(self.windows)

    def numeric(self, key):
        if key not in self._episodes:
            record = self.records[key]
            path = str(data_path(self.root, record))
            if path not in self._tables:
                self._tables[path] = pq.read_table(path)
                if len(self._tables) > 4:
                    self._tables.popitem(last=False)
            self._episodes[key] = episode_rows(self.root, record, self._tables[path])
            if len(self._episodes) > 16:
                self._episodes.popitem(last=False)
        return self._episodes[key]

    def __getitem__(self, index):
        key, start = self.windows[index]
        start = int(start)
        record = self.records[key]
        rows = self.numeric(key)
        if start < 0 or start + 32 >= int(record["length"]):
            raise ValueError("Franka window must contain 33 complete timesteps")
        views = []
        for camera in CAMERAS:
            path, base = video_location(self.root, record, camera)
            views.append(decode_frames(path, [base + start + i for i in VIDEO_OFFSETS]))
        video, roi = compose(self.augmentation(views))
        task = str(record["tasks"][0])
        if task not in self._text:
            path = text_cache_path(
                self.config.text_embedding_cache_dir,
                task,
                self.config.text_len,
                self.config.text_prompt_template,
                self.config.text_cache_encoder_id,
            )
            self._text[task] = load_text_cache(
                path, self.config.text_len, self.text_dim
            )
        context, mask = self._text[task]
        raw_action = torch.tensor(
            rows["action"][start : start + 32], dtype=torch.float32
        )
        raw_state = torch.tensor(rows["observation.state"][start], dtype=torch.float32)
        return {
            "video": video,
            "action": normalize(raw_action, self.stats["action"]),
            "proprio": normalize(raw_state, self.stats["state"]).unsqueeze(0),
            "context": context,
            "context_mask": mask,
            "action_is_pad": torch.zeros(32, dtype=torch.bool),
            "image_is_pad": torch.zeros(9, dtype=torch.bool),
            "raw_action": raw_action,
            "raw_state": raw_state,
            "valid_rois": roi,
            "sample_id": f"{key}/{start:06d}",
            "task_name": record["task"],
            "instruction": task,
            "dataset_index": index,
        }
