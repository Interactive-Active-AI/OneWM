"""Generic training entry point for OneWM recipes."""

import argparse
import logging
from typing import Any

import torch

from onewm.utils.config_cli import apply_overrides
from onewm.utils.output_paths import enforce_training_output_dir


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("onewm.train")


def _configure_rank_logging():
    import os

    if int(os.environ.get("RANK", "0")) != 0:
        logging.getLogger("onewm").setLevel(logging.WARNING)
        logging.getLogger("onewm.train").setLevel(logging.WARNING)


def _maybe_prepare_onewm_text_cache(cfg: Any) -> None:
    data = getattr(cfg, "data", None)
    dataset_type = getattr(data, "dataset_type", None)
    if dataset_type not in {"lerobot", "robocasa365"}:
        return
    if dataset_type == "robocasa365":
        from onewm.data.robocasa365 import discover_robocasa365_roots

        roots = discover_robocasa365_roots(data.root)
        if bool(getattr(data, "require_complete_dataset", False)):
            expected = int(getattr(data, "expected_num_tasks", None) or 300)
            if len(roots) != expected:
                raise ValueError(
                    f"RoboCasa365 Human300 is incomplete: found {len(roots)} tasks, "
                    f"expected {expected}; refusing text-cache/training startup"
                )
    else:
        roots = list(
            getattr(data, "dataset_dirs", None)
            or ([] if getattr(data, "root", None) is None else [data.root])
        )
    roots = [str(root) for root in roots if root]
    if roots:
        from onewm.builder import _ensure_text_caches

        _ensure_text_caches(cfg, roots)


def main():
    _configure_rank_logging()
    parser = argparse.ArgumentParser(description="OneWM training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides as dot.notation=value (e.g. training.batch_size=2)",
    )
    args = parser.parse_args()

    from onewm import build_framework, build_trainer
    from onewm.config import config_to_dict, load_config

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    output_dir = enforce_training_output_dir(config)

    logger.info(
        f"Config: framework={config.framework.type}, backbone={config.backbone.type}"
    )
    logger.info(
        f"action_dim={config.framework.action_dim}, "
        f"chunk_size={config.framework.chunk_size}, "
        f"strategy={config.training.strategy}"
    )
    logger.info(f"Training outputs use persistent checkpoint path: {output_dir}")

    _maybe_prepare_onewm_text_cache(config)

    logger.info("Building model...")
    if torch.cuda.is_available():
        import os as _os

        local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    mp = (config.training.mixed_precision or "no").lower()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp, torch.float32)
    logger.info(f"Build device={device}, dtype={dtype}")
    model = build_framework(config, device=device, dtype=dtype).to(device)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built: {total / 1e6:.1f}M params total")

    logger.info("Building trainer...")
    trainer = build_trainer(model, config)

    wandb_run = getattr(trainer, "_wandb_run", None) or getattr(
        trainer, "wandb_run", None
    )
    if wandb_run is not None:
        try:
            wandb_run.config.update(config_to_dict(config), allow_val_change=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to log config to wandb: {e}")

    logger.info("Starting training...")
    try:
        trainer.train()
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to finish wandb run cleanly: {e}")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
