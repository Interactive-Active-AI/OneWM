#!/usr/bin/env python
"""Precompute Wan text embedding caches for OneWM training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from onewm.data.lerobot import (
    DEFAULT_TEXT_CACHE_ENCODER_ID,
    DEFAULT_TEXT_PROMPT,
    collect_lerobot_tasks,
    format_text_prompt,
    save_text_cache,
    text_cache_path,
)
from onewm.config import load_config
from onewm.utils.config_cli import apply_overrides


def _resolve_cache_dir(config, override: str | None) -> Path:
    cache_dir = override or getattr(config.data, "text_embedding_cache_dir", None)
    if not cache_dir:
        raise ValueError(
            "Set data.text_embedding_cache_dir in the recipe or pass --output-dir."
        )
    return Path(cache_dir)


def _build_text_encoder(
    config, model_dir: Path, context_len: int, device: str, dtype: torch.dtype
):
    from onewm.backbone import build_text_encoder

    return build_text_encoder(
        config.backbone,
        text_len=context_len,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="OneWM recipe YAML")
    parser.add_argument(
        "--pretrained-model-id",
        default=None,
        help="Local backbone checkpoint directory. Overrides backbone.pretrained_model_id from the recipe.",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Override data.text_embedding_cache_dir"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the task list into this many shards (for multi-GPU precompute).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this process handles (0 <= shard-index < num-shards).",
    )
    parser.add_argument(
        "--override", nargs="*", default=[], help="Config overrides key=value"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    if args.pretrained_model_id:
        config.backbone.pretrained_model_id = args.pretrained_model_id

    if config.data.dataset_type == "robocasa365":
        from onewm.data.robocasa365 import discover_robocasa365_roots

        dataset_dirs = discover_robocasa365_roots(config.data.root)
        if bool(getattr(config.data, "require_complete_dataset", False)):
            expected = int(getattr(config.data, "expected_num_tasks", None) or 300)
            if len(dataset_dirs) != expected:
                raise ValueError(
                    f"RoboCasa365 Human300 is incomplete: {len(dataset_dirs)} != {expected}"
                )
    else:
        dataset_dirs = (
            list(config.data.dataset_dirs)
            if config.data.dataset_dirs
            else ([config.data.root] if config.data.root else [])
        )
    dataset_dirs = [str(path) for path in dataset_dirs if path]
    if not dataset_dirs:
        raise ValueError("No dataset dirs found in config.data.")

    tasks = collect_lerobot_tasks(dataset_dirs)
    if not tasks:
        raise ValueError("No tasks found from meta/tasks.jsonl in dataset dirs.")

    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise ValueError(
                f"shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
            )
        tasks = tasks[args.shard_index :: args.num_shards]
        print(
            f"[text-cache] shard {args.shard_index}/{args.num_shards}: {len(tasks)} tasks"
        )

    cache_dir = _resolve_cache_dir(config, args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = (
        getattr(config.data, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
    )
    encoder_id = (
        getattr(config.data, "text_cache_encoder_id", None)
        or DEFAULT_TEXT_CACHE_ENCODER_ID
    )
    context_len = int(
        getattr(
            config.data, "text_len", getattr(config.backbone, "tokenizer_max_len", 128)
        )
    )
    prompts = [format_text_prompt(task, prompt_template) for task in tasks]
    pending: list[tuple[str, str, Path]] = []
    for task, prompt in zip(tasks, prompts):
        path = text_cache_path(
            cache_dir, task, context_len, prompt_template, encoder_id
        )
        if args.overwrite or not path.exists():
            pending.append((task, prompt, path))

    print(
        f"[text-cache] tasks={len(tasks)} pending={len(pending)} cache_dir={cache_dir}"
    )
    if not pending:
        return

    model_dir = Path(config.backbone.pretrained_model_id)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    encoder = _build_text_encoder(config, model_dir, context_len, args.device, dtype)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_prompts = [item[1] for item in batch]
        with torch.no_grad():
            context, mask = encoder.encode(batch_prompts)
        for i, (task, prompt, path) in enumerate(batch):
            save_text_cache(path, context[i], mask[i], prompt, task)
            print(f"[text-cache] wrote {path.name} task={task}")


if __name__ == "__main__":
    main()
