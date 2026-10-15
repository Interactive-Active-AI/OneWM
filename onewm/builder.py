"""OneWM builders."""

from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
import torch
from onewm.config import OneWMConfig
from onewm.data.lerobot import (
    DEFAULT_TEXT_CACHE_ENCODER_ID,
    DEFAULT_TEXT_PROMPT,
    collect_lerobot_tasks,
    format_text_prompt,
    load_lerobot_stats,
    save_text_cache,
    text_cache_path,
)

LOGGER = logging.getLogger("onewm.builder")


def build_framework(config: Any, device: str = "cpu", dtype=None):
    """Build the Wan-based MoT world-action model."""
    if dtype is None:
        dtype = torch.bfloat16
    if config.framework.type != "mot":
        raise ValueError("OneWM supports only framework.type='mot'")
    from onewm.backbone import build_backbone
    from onewm.wam import MoTWAM

    backbone = build_backbone(config.backbone, device=device, dtype=dtype)
    LOGGER.info("Building OneWM MoT WAM with backbone=%s", config.backbone.type)
    return MoTWAM(backbone, config.framework, device=device, dtype=dtype)


def _load_or_compute_lerobot_stats(
    stats_path: Path,
    roots: list[str],
    action_key: str,
    state_key: str,
    need_action: bool,
    need_state: bool,
):
    from onewm.data.lerobot import (
        compute_action_stats,
        compute_state_stats,
        load_lerobot_stats,
        save_lerobot_stats,
    )

    if stats_path.is_file():
        stats = load_lerobot_stats(stats_path)
        if (not need_action or "action" in stats) and (
            not need_state or "state" in stats
        ):
            return stats
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stats_path.with_suffix(stats_path.suffix + ".lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                stats = load_lerobot_stats(stats_path) if stats_path.is_file() else {}
                if need_action and "action" not in stats:
                    stats["action"] = compute_action_stats(roots, action_key=action_key)
                if need_state and "state" not in stats:
                    stats["state"] = compute_state_stats(roots, state_key=state_key)
                save_lerobot_stats(stats, stats_path)
            finally:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
            return load_lerobot_stats(stats_path)
        except FileExistsError:
            if stats_path.is_file():
                stats = load_lerobot_stats(stats_path)
                if (not need_action or "action" in stats) and (
                    not need_state or "state" in stats
                ):
                    return stats
            time.sleep(5)


def _ensure_text_caches(config: OneWMConfig, roots: list[str]) -> None:
    cache_dir = getattr(config.data, "text_embedding_cache_dir", None)
    if not cache_dir:
        raise ValueError(
            "data.text_embedding_cache_dir must be set for LeRobot training"
        )
    text_len = int(getattr(config.data, "text_len", 128))
    prompt_template = (
        getattr(config.data, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
    )
    encoder_id = (
        getattr(config.data, "text_cache_encoder_id", None)
        or DEFAULT_TEXT_CACHE_ENCODER_ID
    )
    tasks = collect_lerobot_tasks(roots)
    if not tasks:
        raise ValueError(
            f"No tasks found in meta/tasks.jsonl under dataset roots: {roots}"
        )
    missing = [
        task
        for task in tasks
        if not text_cache_path(
            cache_dir, task, text_len, prompt_template, encoder_id
        ).is_file()
    ]
    if not missing:
        return
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".onewm_text_cache.lock"
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                missing = [
                    task
                    for task in tasks
                    if not text_cache_path(
                        cache_dir, task, text_len, prompt_template, encoder_id
                    ).is_file()
                ]
                if not missing:
                    return
                model_dir = Path(config.backbone.pretrained_model_id)
                device = (
                    f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                encoder_dtype = (
                    torch.bfloat16 if device.startswith("cuda") else torch.float32
                )
                from onewm.backbone import build_text_encoder

                encoder = build_text_encoder(
                    config.backbone,
                    text_len=text_len,
                    device=device,
                    dtype=encoder_dtype,
                    model_dir=model_dir,
                )
                LOGGER.info(
                    "Generating %d missing OneWM text caches in %s",
                    len(missing),
                    cache_dir,
                )
                for start in range(0, len(missing), 4):
                    batch = missing[start : start + 4]
                    prompts = [
                        format_text_prompt(task, prompt_template) for task in batch
                    ]
                    with torch.no_grad():
                        context, mask = encoder.encode(prompts)
                    for i, task in enumerate(batch):
                        save_text_cache(
                            text_cache_path(
                                cache_dir, task, text_len, prompt_template, encoder_id
                            ),
                            context[i],
                            mask[i],
                            prompts[i],
                            task,
                        )
                del encoder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            finally:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
        except FileExistsError:
            if all(
                (
                    text_cache_path(
                        cache_dir, task, text_len, prompt_template, encoder_id
                    ).is_file()
                    for task in tasks
                )
            ):
                return
            time.sleep(5)


def build_dataset(config: OneWMConfig, is_training: bool = True, text_dim: int = 4096):
    """Build the OneWM dataset."""
    from torch.utils.data import ConcatDataset, Subset
    from onewm.data.lerobot import (
        LeRobotDataset,
        LeRobotSyntheticDataset,
        compute_action_stats,
        compute_state_stats,
    )

    data_cfg = config.data
    if data_cfg.dataset_type == "franka_v3":
        from onewm.data.franka_v3 import FrankaDataset

        return FrankaDataset(data_cfg, is_training=is_training, text_dim=text_dim)
    if data_cfg.dataset_type == "synthetic":
        return LeRobotSyntheticDataset(
            config=data_cfg,
            action_dim=config.framework.action_dim,
            chunk_size=config.framework.chunk_size,
            text_dim=text_dim,
            is_training=is_training,
            proprio_dim=getattr(config.framework, "proprio_dim", None),
        )
    if data_cfg.dataset_type == "robocasa365":
        from onewm.data.robocasa365 import (
            ACTION_BINARY_DIMS,
            CAMERA_KEYS,
            ROBOCASA365_CONTRACT_VERSION,
            RoboCasa365Dataset,
            RoboCasa365VideoTransform,
            discover_robocasa365_roots,
        )

        if not data_cfg.root:
            raise ValueError(
                "robocasa365 requires data.root pointing to the Human300 pretrain root"
            )
        roots = discover_robocasa365_roots(data_cfg.root)
        expected_tasks = int(data_cfg.expected_num_tasks or 300)
        if bool(data_cfg.require_complete_dataset) and len(roots) != expected_tasks:
            raise ValueError(
                f"RoboCasa365 dataset is incomplete: discovered {len(roots)} task roots, expected {expected_tasks}. Wait for/download and audit Human300 first."
            )
        if not roots:
            raise ValueError(
                f"no canonical RoboCasa365 LeRobot roots found under {data_cfg.root}"
            )
        if list(data_cfg.video_keys) != list(CAMERA_KEYS):
            raise ValueError(
                "robocasa365 video_keys must be [left, wrist, right] in the canonical order"
            )
        if tuple(data_cfg.video_size) != (256, 800):
            raise ValueError("robocasa365 final video_size must be [256, 800]")
        if data_cfg.concat_multi_camera != "robocasa365_strip":
            raise ValueError(
                "robocasa365 requires concat_multi_camera='robocasa365_strip'"
            )
        if (
            int(data_cfg.view_separator_width) != 16
            or int(data_cfg.view_separator_value) != 128
        ):
            raise ValueError("robocasa365 requires a 16px RGB-128 view separator")
        if list(data_cfg.observation_history_offsets) != [-6, -4, -2, 0]:
            raise ValueError("robocasa365 history offsets must be [-6,-4,-2,0]")
        if list(data_cfg.video_future_offsets) != [2, 4, 6, 8, 10, 12, 14, 16]:
            raise ValueError("robocasa365 future offsets must be [2,4,...,16]")
        raw_frames = (
            len(data_cfg.observation_history_offsets)
            + int(data_cfg.repeat_current_for_vae_alignment)
            + len(data_cfg.video_future_offsets)
        )
        if int(data_cfg.num_frames) != raw_frames or raw_frames != 13:
            raise ValueError("robocasa365 requires exactly 13 raw video frames")
        if (
            int(config.framework.action_dim) != 12
            or int(config.framework.chunk_size) != 16
        ):
            raise ValueError("robocasa365 requires action_dim=12 and chunk_size=16")
        if str(config.framework.state_conditioning) != "action_post_mot":
            raise ValueError("robocasa365 requires action_post_mot state conditioning")
        if (
            int(config.framework.state_token_dim) != 14
            or int(config.framework.state_num_tokens or 0) != 4
        ):
            raise ValueError("robocasa365 requires four 14D history state tokens")
        if str(config.framework.state_token_semantics) != "history":
            raise ValueError("robocasa365 requires state_token_semantics='history'")
        if int(config.framework.video_clean_prefix_latent_steps) != 2:
            raise ValueError("robocasa365 requires two clean video latent steps")
        if str(config.framework.action_video_conditioning) != "clean_prefix":
            raise ValueError(
                "robocasa365 requires action_video_conditioning='clean_prefix'"
            )
        if list(data_cfg.action_binary_dims) != list(ACTION_BINARY_DIMS):
            raise ValueError("robocasa365 action_binary_dims must be [6,11]")
        if list(config.framework.action_gripper_dims or []) != list(ACTION_BINARY_DIMS):
            raise ValueError("robocasa365 monitoring binary dims must be [6,11]")
        if data_cfg.action_representation != "xiaomi_ee_base_12d":
            raise ValueError("robocasa365 action representation contract changed")
        if data_cfg.state_representation != "xiaomi_ee_base_14d_history4":
            raise ValueError("robocasa365 state representation contract changed")
        if not data_cfg.normalize_actions or data_cfg.action_norm_mode != "zscore":
            raise ValueError(
                "robocasa365 requires transformed action z-score normalization"
            )
        if not data_cfg.normalize_states or data_cfg.state_norm_mode != "zscore":
            raise ValueError(
                "robocasa365 requires transformed state z-score normalization"
            )
        stats_path_value = data_cfg.action_stats_path
        if not stats_path_value:
            raise ValueError(
                "robocasa365 requires a benchmark-specific action stats file"
            )
        state_stats_path = data_cfg.state_stats_path or stats_path_value
        if Path(state_stats_path) != Path(stats_path_value):
            raise ValueError(
                "robocasa365 action/state stats must use one benchmark-specific file"
            )
        stats_path = Path(stats_path_value)
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"RoboCasa365 transformed stats are missing: {stats_path}. Run deploy/robocasa365/scripts/compute_stats.sh first."
            )
        raw_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        metadata = raw_stats.get("metadata", {})
        if metadata.get("contract_version") != ROBOCASA365_CONTRACT_VERSION:
            raise ValueError(
                f"RoboCasa365 stats contract mismatch in {stats_path}: {metadata}"
            )
        if list(metadata.get("action_binary_dims", [])) != list(ACTION_BINARY_DIMS):
            raise ValueError("RoboCasa365 stats binary-channel metadata mismatch")
        if bool(data_cfg.require_complete_dataset):
            actual_names = sorted((path.parents[1].name for path in roots))
            if int(metadata.get("task_count", -1)) != expected_tasks:
                raise ValueError(
                    "RoboCasa365 stats were not computed over exact Human300"
                )
            if sorted(metadata.get("task_names", [])) != actual_names:
                raise ValueError(
                    "RoboCasa365 stats task set differs from the dataset task set"
                )
        stats = load_lerobot_stats(stats_path)
        random_probability = (
            float(data_cfg.video_augmentation_probability)
            if is_training and bool(data_cfg.video_augmentation_enabled)
            else 0.0
        )
        transform = RoboCasa365VideoTransform(
            random_probability=random_probability,
            crop_ratio=float(data_cfg.video_random_crop_ratio),
            brightness=float(data_cfg.video_brightness),
            contrast=float(data_cfg.video_contrast),
            saturation=float(data_cfg.video_saturation),
            hue=float(data_cfg.video_hue),
        )
        return RoboCasa365Dataset(
            roots=roots,
            history_offsets=data_cfg.observation_history_offsets,
            future_offsets=data_cfg.video_future_offsets,
            repeat_current=data_cfg.repeat_current_for_vae_alignment,
            chunk_size=config.framework.chunk_size,
            view_size=(256, 256),
            separator_width=data_cfg.view_separator_width,
            separator_value=data_cfg.view_separator_value,
            text_len=data_cfg.text_len,
            text_dim=text_dim,
            text_embedding_cache_dir=data_cfg.text_embedding_cache_dir,
            text_prompt_template=data_cfg.text_prompt_template or DEFAULT_TEXT_PROMPT,
            text_cache_encoder_id=data_cfg.text_cache_encoder_id
            or DEFAULT_TEXT_CACHE_ENCODER_ID,
            normalize_action_stats=stats["action"]
            if data_cfg.normalize_actions
            else None,
            normalize_state_stats=stats["state"] if data_cfg.normalize_states else None,
            video_transform=transform,
            deterministic_transform_seed=None
            if is_training
            else int(config.training.eval_sample_seed),
            val_fraction=float(data_cfg.val_split),
            split="train" if is_training else "val",
            split_seed=int(data_cfg.val_split_seed),
        )
    if data_cfg.dataset_type != "lerobot":
        raise ValueError(f"Unknown dataset_type: {data_cfg.dataset_type}")
    roots = (
        list(data_cfg.dataset_dirs)
        if data_cfg.dataset_dirs
        else [data_cfg.root]
        if data_cfg.root
        else []
    )
    roots = [str(root) for root in roots if root]
    if not roots:
        raise ValueError(
            "data.root or data.dataset_dirs must be set for dataset_type='lerobot'"
        )
    action_stats = None
    state_stats = None
    normalize_actions = bool(getattr(data_cfg, "normalize_actions", False))
    normalize_states = bool(getattr(data_cfg, "normalize_states", False))
    state_key = getattr(data_cfg, "state_key", "observation.state")
    if normalize_actions or normalize_states:
        shared_stats_path = getattr(data_cfg, "action_stats_path", None)
        state_stats_path = (
            getattr(data_cfg, "state_stats_path", None) or shared_stats_path
        )
        if shared_stats_path and state_stats_path == shared_stats_path:
            stats = _load_or_compute_lerobot_stats(
                Path(shared_stats_path),
                roots,
                data_cfg.action_key,
                state_key,
                normalize_actions,
                normalize_states,
            )
            action_stats = stats.get("action") if normalize_actions else None
            state_stats = stats.get("state") if normalize_states else None
        else:
            if normalize_actions:
                action_stats = (
                    _load_or_compute_lerobot_stats(
                        Path(shared_stats_path),
                        roots,
                        data_cfg.action_key,
                        state_key,
                        True,
                        False,
                    )["action"]
                    if shared_stats_path
                    else compute_action_stats(roots, action_key=data_cfg.action_key)
                )
            if normalize_states:
                state_stats = (
                    _load_or_compute_lerobot_stats(
                        Path(state_stats_path),
                        roots,
                        data_cfg.action_key,
                        state_key,
                        False,
                        True,
                    )["state"]
                    if state_stats_path
                    else compute_state_stats(roots, state_key=state_key)
                )
    subsets = []
    for root in roots:
        subsets.append(
            LeRobotDataset(
                root=root,
                video_key=data_cfg.video_key,
                video_keys=getattr(data_cfg, "video_keys", None),
                concat_multi_camera=data_cfg.concat_multi_camera,
                action_key=data_cfg.action_key,
                state_key=state_key,
                num_frames=data_cfg.num_frames,
                chunk_size=config.framework.chunk_size,
                video_size=tuple(data_cfg.video_size),
                text_len=data_cfg.text_len,
                text_dim=text_dim,
                action_freq_ratio=data_cfg.action_freq_ratio,
                normalize_action_stats=action_stats,
                action_norm_mode=data_cfg.action_norm_mode,
                normalize_state_stats=state_stats,
                state_norm_mode=getattr(data_cfg, "state_norm_mode", "minmax"),
                delta_action_dim_mask=getattr(data_cfg, "delta_action_dim_mask", None),
                proprio_dim=getattr(config.framework, "proprio_dim", None),
                text_embedding_cache_dir=getattr(
                    data_cfg, "text_embedding_cache_dir", None
                ),
                text_prompt_template=getattr(data_cfg, "text_prompt_template", None)
                or DEFAULT_TEXT_PROMPT,
                text_cache_encoder_id=getattr(data_cfg, "text_cache_encoder_id", None)
                or DEFAULT_TEXT_CACHE_ENCODER_ID,
            )
        )
    dataset = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)
    val_split = float(getattr(data_cfg, "val_split", 0.0) or 0.0)
    if val_split <= 0:
        return dataset
    generator = torch.Generator().manual_seed(
        int(getattr(data_cfg, "val_split_seed", 42))
    )
    perm = torch.randperm(len(dataset), generator=generator).tolist()
    val_count = max(1, int(len(dataset) * val_split))
    indices = perm[val_count:] if is_training else perm[:val_count]
    return Subset(dataset, indices)


def build_trainer(model: Any, config: OneWMConfig):
    """Build the OneWM trainer."""
    from onewm.training.trainer import OneWMTrainer

    text_dim = (
        getattr(getattr(model, "backbone", None), "info", None).text_dim
        if hasattr(model, "backbone")
        else 4096
    )
    train_dataset = build_dataset(config, is_training=True, text_dim=text_dim)
    val_dataset = None
    if float(getattr(config.data, "val_split", 0.0) or 0.0) > 0:
        val_dataset = build_dataset(config, is_training=False, text_dim=text_dim)
    return OneWMTrainer(model, train_dataset, val_dataset, config.training)
