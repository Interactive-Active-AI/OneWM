"""OneWM trainer: unified training loop with HuggingFace Accelerate + DeepSpeed."""

import json
import os
import re
import time
import logging
from math import ceil, cos, pi
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset
from onewm.config import TrainingConfig
from onewm.wam.base import WAMModel

logger = logging.getLogger(__name__)


def _configured_learning_rates(config: TrainingConfig) -> tuple[float, float]:
    """Resolve video/action peak LRs with legacy single-LR compatibility."""
    fallback = float(config.learning_rate)
    video_value = getattr(config, "video_learning_rate", None)
    action_value = getattr(config, "action_learning_rate", None)
    video_lr = fallback if video_value is None else float(video_value)
    action_lr = fallback if action_value is None else float(action_value)
    if video_lr <= 0.0 or action_lr <= 0.0:
        raise ValueError(
            f"video_learning_rate and action_learning_rate must be positive, got {video_lr} and {action_lr}"
        )
    return (video_lr, action_lr)


def _optimizer_parameter_groups(model: WAMModel, config: TrainingConfig) -> list[dict]:
    """Partition every optimized parameter into video or action ownership.

    MoT owns no shared learned attention parameters: its registered parameters
    belong to either the video expert or the action expert. Post-MoT state
    adapters affect only action tokens and therefore use the action LR.
    """
    if not hasattr(model, "backbone") or not hasattr(model, "action_expert"):
        raise TypeError(
            "split learning rates require a MoTWAM model with backbone and action_expert modules"
        )
    video_expert = model.backbone.get_dit()
    action_expert = model.action_expert
    video_ids = {id(parameter) for parameter in video_expert.parameters()}
    action_ids = {id(parameter) for parameter in action_expert.parameters()}
    for name in ("state_encoder", "state_norm", "state_cross_attention"):
        module = getattr(model, name, None)
        if module is not None:
            action_ids.update((id(parameter) for parameter in module.parameters()))
    for name in ("learned_null_state", "state_gate"):
        parameter = getattr(model, name, None)
        if parameter is not None:
            action_ids.add(id(parameter))
    overlap = video_ids & action_ids
    if overlap:
        raise ValueError("video and action optimizer ownership unexpectedly overlaps")
    include_frozen_experts = getattr(config, "strategy", "full") == "staged"
    expert_ids = video_ids | action_ids
    video_params = []
    action_params = []
    unassigned = []
    for parameter_name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        should_optimize = parameter.requires_grad or (
            include_frozen_experts and parameter_id in expert_ids
        )
        if not should_optimize:
            continue
        if parameter_id in video_ids:
            video_params.append(parameter)
        elif parameter_id in action_ids:
            action_params.append(parameter)
        else:
            unassigned.append(parameter_name)
    if unassigned:
        raise ValueError(
            "trainable parameters have no video/action LR ownership: "
            + ", ".join(unassigned[:20])
        )
    if not video_params and (not action_params):
        raise ValueError("model has no parameters eligible for optimization")
    video_lr, action_lr = _configured_learning_rates(config)
    video_min_lr = float(config.min_lr)
    action_min_lr = video_min_lr * action_lr / video_lr
    groups = []
    if video_params:
        groups.append(
            {
                "params": video_params,
                "name": "video",
                "lr": video_lr,
                "base_lr": video_lr,
                "min_lr": video_min_lr,
            }
        )
    if action_params:
        groups.append(
            {
                "params": action_params,
                "name": "action",
                "lr": action_lr,
                "base_lr": action_lr,
                "min_lr": action_min_lr,
            }
        )
    return groups


def _optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    """Return current video/action LRs, including legacy unnamed groups."""
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    fallback = float(optimizer.param_groups[0]["lr"])
    by_name = {
        str(group.get("name")): float(group["lr"])
        for group in optimizer.param_groups
        if group.get("name") is not None
    }
    return (by_name.get("video", fallback), by_name.get("action", fallback))


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_start_factor: float,
):
    """Build a fresh scheduler for a complete or resumed training segment."""
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(max(int(warmup_steps), 0), max(total_steps - 1, 0))
    scheduler_type = str(scheduler_type).strip().lower()
    if scheduler_type not in {"cosine", "cosine_with_min_lr", "constant"}:
        raise ValueError(f"unsupported lr_scheduler_type: {scheduler_type!r}")
    if not 0.0 < float(warmup_start_factor) <= 1.0:
        raise ValueError("warmup_start_factor must satisfy 0 < factor <= 1")
    group_base_lrs = []
    group_min_lrs = []
    for group in optimizer.param_groups:
        group_base_lr = float(group.get("base_lr", base_lr))
        group["lr"] = group_base_lr
        group["initial_lr"] = group_base_lr
        group_base_lrs.append(group_base_lr)
        if scheduler_type == "constant":
            group_min_lr = 0.0
        elif scheduler_type == "cosine":
            group_min_lr = group_base_lr * 0.01
        else:
            group_min_lr = float(group.get("min_lr", min_lr))
        if scheduler_type != "constant" and (
            group_min_lr < 0.0 or group_min_lr >= group_base_lr
        ):
            raise ValueError(
                f"min_lr must satisfy 0 <= min_lr < group base LR, got {group_min_lr} >= {group_base_lr} for {group.get('name', '<unnamed>')}"
            )
        group_min_lrs.append(group_min_lr)
    if scheduler_type == "constant":
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        cosine_steps = max(total_steps - warmup_steps, 1)
        min_factors = [
            group_min_lr / group_base_lr
            for group_min_lr, group_base_lr in zip(group_min_lrs, group_base_lrs)
        ]

        def _cosine_lambda(min_factor: float):

            def schedule(step: int) -> float:
                progress = min(max(float(step) / cosine_steps, 0.0), 1.0)
                return (
                    min_factor + (1.0 - min_factor) * (1.0 + cos(pi * progress)) / 2.0
                )

            return schedule

        main_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[_cosine_lambda(value) for value in min_factors]
        )
    if warmup_steps <= 0:
        return main_scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(warmup_start_factor),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )


def _fixed_random_eval_indices(
    dataset_size: int, max_samples: int, seed: int
) -> list[int]:
    """Select a deterministic random validation subset without replacement."""
    if dataset_size <= 0:
        return []
    sample_count = min(max(int(max_samples), 1), int(dataset_size))
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(dataset_size), generator=generator)[
        :sample_count
    ].tolist()


class _RunningMetricMean:
    """Accumulate scalar micro-batch metrics without retaining tensors."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict) -> None:
        for key, value in metrics.items():
            scalar = float(value)
            self._sums[key] = self._sums.get(key, 0.0) + scalar
            self._counts[key] = self._counts.get(key, 0) + 1

    def mean(self) -> dict[str, float]:
        return {key: total / self._counts[key] for key, total in self._sums.items()}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch("checkpoint-(\\d+)", path.name)
    if match:
        return int(match.group(1))
    meta_path = path / "trainer_state.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            return int(json.load(f)["global_step"])
    raise ValueError(f"Cannot infer global step from checkpoint path: {path}")


class OneWMTrainer:
    """Unified trainer for OneWM models.

    Features:
    - HuggingFace Accelerate for distributed training (DeepSpeed ZeRO-2)
    - Freezes VAE + text encoder, trains only MoT (video + action experts)
    - Gradient accumulation, mixed precision, gradient clipping
    - Periodic eval, checkpointing, logging
    """

    def __init__(
        self,
        model: WAMModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        try:
            from accelerate import Accelerator

            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                log_with="wandb" if self.config.wandb_enabled else None,
                step_scheduler_with_optimizer=False,
            )
        except ImportError:
            self.accelerator = None
            logger.warning(
                "accelerate not installed, running without distributed support"
            )
        if self.accelerator is not None:
            zero_stage = "none"
            deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
            if deepspeed_plugin is not None:
                zero_stage = deepspeed_plugin.deepspeed_config.get(
                    "zero_optimization", {}
                ).get("stage", "unknown")
            logger.info(
                "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d mixed_precision=%s grad_accum=%d",
                self.accelerator.distributed_type,
                zero_stage,
                self.accelerator.num_processes,
                self.accelerator.mixed_precision,
                self.config.gradient_accumulation_steps,
            )
        self.global_step = 0
        self._wandb_run = None
        self._resume_step: Optional[int] = None
        self._load_model_only_if_requested()
        self._setup()
        self._resume_if_requested()
        self._apply_resume_lr_override_if_requested()
        self._init_wandb()

    def _init_wandb(self):
        """Initialise a W&B run on the main process when enabled."""
        if not getattr(self.config, "wandb_enabled", False):
            return
        if not self._is_main_process():
            return
        required = bool(getattr(self.config, "wandb_required", False))
        try:
            import wandb
        except ImportError as exc:
            message = "wandb_enabled=True but wandb is not installed"
            if required:
                raise RuntimeError(message) from exc
            logger.warning("%s; skipping.", message)
            return
        try:
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=getattr(self.config, "wandb_run_name", None),
                config={
                    "learning_rate": self.config.learning_rate,
                    "video_learning_rate": _configured_learning_rates(self.config)[0],
                    "action_learning_rate": _configured_learning_rates(self.config)[1],
                    "batch_size": self.config.batch_size,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "max_steps": self.max_steps,
                    "mixed_precision": self.config.mixed_precision,
                    "strategy": self.config.strategy,
                },
                reinit=True,
            )
            self._configure_wandb_metrics(self._wandb_run)
        except Exception as e:
            self._wandb_run = None
            if required:
                raise RuntimeError(f"required wandb.init failed: {e}") from e
            logger.warning(f"wandb.init failed ({e}); continuing without wandb.")

    @staticmethod
    def _configure_wandb_metrics(wandb_run) -> None:
        """Use the optimizer step as the x-axis for train and eval metrics.

        W&B's internal history step advances after every committed ``log``
        call. Training and validation are logged separately at the same
        optimizer step, so passing that optimizer step through ``step=`` makes
        the later validation row look out of order and W&B drops it. A custom
        step metric lets both rows retain the same semantic optimizer step
        while W&B continues managing its own monotonically increasing history
        index.
        """
        wandb_run.define_metric("trainer/global_step")
        wandb_run.define_metric("train/*", step_metric="trainer/global_step")
        wandb_run.define_metric("eval/*", step_metric="trainer/global_step")

    def _wandb_log(self, payload: dict):
        if self._wandb_run is None:
            return
        try:
            history_payload = {"trainer/global_step": self.global_step}
            history_payload.update(payload)
            self._wandb_run.log(history_payload, commit=True)
        except Exception as e:
            logger.warning(f"wandb.log failed ({e})")

    @staticmethod
    def _train_wandb_payload(
        loss_dict: dict,
        lr: float,
        steps_per_sec: float,
        action_lr: Optional[float] = None,
        optimizer_metrics: Optional[dict] = None,
        samples_per_sec: Optional[float] = None,
    ) -> dict:
        action_lr = lr if action_lr is None else action_lr
        payload = {
            "train/loss_total": loss_dict["loss_total"],
            "train/loss_video": loss_dict["loss_video"],
            "train/loss_action": loss_dict["loss_action"],
            "train/lr": lr,
            "train/lr_video": lr,
            "train/lr_action": action_lr,
            "train/steps_per_sec": steps_per_sec,
        }
        if samples_per_sec is not None:
            payload["train/samples_per_sec"] = samples_per_sec
        optional_keys = (
            "action_clean_prefix_length_mean",
            "action_clean_token_fraction",
            "action_clean_sample_none_fraction",
            "action_clean_sample_partial_fraction",
            "action_clean_sample_full_fraction",
            "action_clean_prob_none",
            "action_clean_prob_partial",
            "action_clean_prob_full",
            "state_dropout_sample_fraction",
            "loss_action_eef",
            "loss_action_gripper",
            "loss_action_gripper_left",
            "loss_action_gripper_right",
            "action_target_gripper_mean",
            "action_target_gripper_open_rate",
            "action_target_gripper_left_mean",
            "action_target_gripper_left_open_rate",
            "action_target_gripper_right_mean",
            "action_target_gripper_right_open_rate",
        )
        for key in optional_keys:
            if key in loss_dict:
                payload[f"train/{key}"] = loss_dict[key]
        if optimizer_metrics:
            for key, value in optimizer_metrics.items():
                payload[f"train/{key}"] = value
        return payload

    def _distributed_metric_mean(
        self, accumulator: _RunningMetricMean
    ) -> dict[str, float]:
        """Average accumulated micro-batch metrics over every process."""
        local_means = accumulator.mean()
        if not local_means or self.accelerator is None:
            return local_means
        keys = sorted(local_means)
        values = torch.tensor(
            [local_means[key] for key in keys],
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        values = self.accelerator.reduce(values, reduction="mean")
        return {
            key: float(value)
            for key, value in zip(keys, values.detach().cpu().tolist())
        }

    def _distributed_grad_norm_is_finite(self, grad_norm) -> bool:
        norm = (
            torch.as_tensor(
                grad_norm, device=self.accelerator.device, dtype=torch.float32
            )
            .detach()
            .reshape(1)
        )
        local_finite = torch.isfinite(norm).to(dtype=torch.int32)
        all_finite = self.accelerator.gather(local_finite)
        return bool(all_finite.all().item())

    @staticmethod
    def _format_action_monitor(loss_dict: dict) -> str:
        if "loss_action_gripper" not in loss_dict:
            return ""
        result = f" | action_eef={loss_dict['loss_action_eef']:.4f} | gripper={loss_dict['loss_action_gripper']:.4f} | grip_open={loss_dict['action_target_gripper_open_rate']:.2f}"
        if "loss_action_gripper_left" in loss_dict:
            result += f" | grip_l={loss_dict['loss_action_gripper_left']:.4f} | grip_r={loss_dict['loss_action_gripper_right']:.4f}"
        return result

    def _setup(self):
        """Setup optimizer, scheduler, dataloader, freeze strategy."""
        self._apply_freeze_strategy()
        parameter_groups = _optimizer_parameter_groups(self.model, self.config)
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
        dataloader_kwargs = {}
        if self.config.num_workers > 0:
            dataloader_kwargs.update(
                {
                    "timeout": float(self.config.dataloader_timeout_seconds),
                    "persistent_workers": True,
                    "prefetch_factor": 1,
                }
            )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            **dataloader_kwargs,
        )
        world_size = (
            self.accelerator.num_processes if self.accelerator is not None else 1
        )
        if self.config.max_steps is not None:
            max_steps = max(int(self.config.max_steps), 1)
        else:
            global_batch_size = max(
                int(self.config.batch_size) * max(int(world_size), 1), 1
            )
            micro_steps_per_epoch = max(
                ceil(len(self.train_dataset) / global_batch_size), 1
            )
            max_steps = max(
                ceil(
                    micro_steps_per_epoch
                    / max(int(self.config.gradient_accumulation_steps), 1)
                )
                * int(self.config.num_epochs),
                1,
            )
        if self.config.warmup_steps is not None:
            warmup_steps = int(self.config.warmup_steps)
        else:
            warmup_steps = int(max_steps * self.config.warmup_ratio)
        video_lr, action_lr = _configured_learning_rates(self.config)
        self.lr_scheduler = _build_lr_scheduler(
            self.optimizer,
            scheduler_type=self.config.lr_scheduler_type,
            total_steps=max_steps,
            warmup_steps=warmup_steps,
            base_lr=video_lr,
            min_lr=self.config.min_lr,
            warmup_start_factor=self.config.warmup_start_factor,
        )
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.world_size = world_size
        self.effective_global_batch_size = (
            int(self.config.batch_size)
            * max(int(world_size), 1)
            * max(int(self.config.gradient_accumulation_steps), 1)
        )
        expected_global_batch = getattr(self.config, "expected_global_batch_size", None)
        if (
            expected_global_batch is not None
            and self.effective_global_batch_size != int(expected_global_batch)
        ):
            raise ValueError(
                f"effective global batch mismatch: {self.effective_global_batch_size} != {int(expected_global_batch)} (world_size={world_size}, per_device={self.config.batch_size}, accumulation={self.config.gradient_accumulation_steps})"
            )
        logger.info(
            "Training schedule: dataset_size=%d world_size=%d per_device_batch=%d grad_accum=%d effective_global_batch=%d max_steps=%d warmup_steps=%d scheduler=%s video_lr=%.2e action_lr=%.2e",
            len(self.train_dataset),
            world_size,
            self.config.batch_size,
            self.config.gradient_accumulation_steps,
            self.effective_global_batch_size,
            self.max_steps,
            self.warmup_steps,
            self.config.lr_scheduler_type,
            video_lr,
            action_lr,
        )
        if self.accelerator:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
                )
            )

    def _apply_freeze_strategy(self):
        """Apply the configured training strategy.

        Supported strategies:
        - ``full``: train the MoT video/action experts end-to-end. The VAE and
          text encoder remain frozen.
        - ``lora``: freeze the entire base model and inject LoRA adapters
          (via ``peft``) into all DiT linear layers matching
          ``config.lora_target_modules``. Action-specific heads are kept
          trainable as full-rank parameters.
        - ``staged``: phase 1 -- train only the small action heads
          (action_encoder/decoder/embedder, state_encoder, action_proj_out)
          for ``config.staged_warmup_steps`` steps; phase 2 -- unfreeze
          MoT video/action experts and continue full fine-tuning.
          The phase-2 transition is triggered from the training loop via
          :meth:`_maybe_unfreeze_staged`.
        """
        strategy = getattr(self.config, "strategy", "full")
        self._staged_unfrozen = False
        if strategy == "lora":
            self._apply_lora_strategy()
            return
        if strategy == "staged":
            self._apply_staged_phase1()
            return
        self.model.requires_grad_(False)
        if not hasattr(self.model, "mot"):
            raise TypeError("OneWMTrainer requires a MoTWAM model")
        self.model.mot.requires_grad_(True)
        self._set_state_adapter_trainable()
        total = sum((p.numel() for p in self.model.parameters()))
        trainable = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        logger.info(
            f"Strategy=full | params: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.1f}%)"
        )

    def _apply_lora_strategy(self):
        """Inject LoRA adapters into the MoT experts."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise RuntimeError(
                "training.strategy='lora' requires the `peft` package. Install with: pip install peft"
            ) from e
        self.model.requires_grad_(False)
        target_modules = list(self.config.lora_target_modules)
        if not hasattr(self.model, "mot"):
            raise TypeError("OneWMTrainer requires a MoTWAM model")
        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model.mot = get_peft_model(self.model.mot, lora_cfg)
        self._set_state_adapter_trainable()
        total = sum((p.numel() for p in self.model.parameters()))
        trainable = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        logger.info(
            f"Strategy=lora (r={self.config.lora_r}, alpha={self.config.lora_alpha}) | params: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)"
        )

    def _apply_staged_phase1(self):
        """Phase 1 of the staged strategy: only action heads are trainable."""
        self.model.requires_grad_(False)
        unfrozen = []
        containers = [
            ("model", self.model),
            ("action_expert", self.model.action_expert),
        ]
        for prefix, container in containers:
            for name in (
                "action_encoder",
                "head",
                "state_encoder",
                "state_norm",
                "state_cross_attention",
            ):
                mod = getattr(container, name, None)
                if mod is not None:
                    mod.requires_grad_(True)
                    unfrozen.append(f"{prefix}.{name}")
        self._set_state_adapter_trainable()
        for name in ("learned_null_state", "state_gate"):
            if getattr(self.model, name, None) is not None:
                unfrozen.append(f"model.{name}")
        total = sum((p.numel() for p in self.model.parameters()))
        trainable = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        warmup = getattr(self.config, "staged_warmup_steps", 1000)
        logger.info(
            f"Strategy=staged | phase=1/2 (action heads only) | warmup_steps={warmup} | unfrozen={unfrozen} | params: {total:,} total, {trainable:,} trainable ({100 * trainable / max(total, 1):.2f}%)"
        )

    def _set_state_adapter_trainable(self) -> None:
        """Keep every post-MoT state adapter parameter in the optimizer."""
        for name in ("state_encoder", "state_norm", "state_cross_attention"):
            module = getattr(self.model, name, None)
            if module is not None:
                module.requires_grad_(True)
        for name in ("learned_null_state", "state_gate"):
            parameter = getattr(self.model, name, None)
            if parameter is not None:
                parameter.requires_grad_(True)

    def _maybe_unfreeze_staged(self):
        """If running under ``staged`` and warmup is complete, unfreeze the
        MoT video/action experts in-place. Idempotent and safe to call
        every step. Both expert groups are installed before training starts;
        frozen phase-2 parameters allocate optimizer state only after they
        first receive gradients.
        """
        if getattr(self, "_staged_unfrozen", True):
            return
        if getattr(self.config, "strategy", "full") != "staged":
            return
        warmup = getattr(self.config, "staged_warmup_steps", 1000)
        if self.global_step < warmup:
            return
        added_params = []
        for p in self.model.mot.parameters():
            if not p.requires_grad:
                p.requires_grad_(True)
                added_params.append(p)
        if added_params:
            optimizer_parameter_ids = {
                id(parameter)
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            }
            missing = [
                parameter
                for parameter in added_params
                if id(parameter) not in optimizer_parameter_ids
            ]
            if missing:
                raise RuntimeError(
                    "staged phase-2 parameters were not registered in the split-LR optimizer"
                )
        self._staged_unfrozen = True
        total = sum((p.numel() for p in self.model.parameters()))
        trainable = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        logger.info(
            f"Strategy=staged | phase=2/2 unfrozen at step={self.global_step} | params: {total:,} total, {trainable:,} trainable ({100 * trainable / max(total, 1):.2f}%)"
        )

    def _unwrap(self):
        """Return the underlying model, unwrapping DDP/Accelerate wrappers."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def _model_compute_param(self):
        for param in self.model.parameters():
            if param.requires_grad:
                return param
        return next(self.model.parameters())

    def _load_model_only_if_requested(self):
        resume = getattr(self.config, "resume", None)
        if not resume or not bool(getattr(self.config, "resume_model_only", False)):
            return
        resume_dir = Path(resume)
        self._validate_checkpoint_metadata(resume_dir)
        if not resume_dir.is_dir():
            raise FileNotFoundError(
                f"training.resume must point to a checkpoint directory: {resume}"
            )
        if self.accelerator is None:
            model_path = resume_dir / "model.pt"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"single-GPU checkpoint missing model.pt: {model_path}"
                )
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload)
        else:
            model_path = resume_dir / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"DeepSpeed checkpoint missing model state: {model_path}"
                )
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            state = payload.get("module")
            if not isinstance(state, dict):
                raise KeyError(f"No module state dict found in {model_path}")
        result = self.model.load_state_dict(state, strict=False)
        self.global_step = int(_checkpoint_step(resume_dir))
        self._resume_step = self.global_step
        if result.missing_keys:
            logger.warning(
                "Missing model-only resume keys, first 20: %s", result.missing_keys[:20]
            )
        if result.unexpected_keys:
            logger.warning(
                "Unexpected model-only resume keys, first 20: %s",
                result.unexpected_keys[:20],
            )
        if self._is_main_process():
            logger.info(
                "Loaded model weights from %s at step=%d; restarting optimizer/scheduler with video_lr=%.2e action_lr=%.2e",
                resume_dir,
                self.global_step,
                *_configured_learning_rates(self.config),
            )

    def _resume_if_requested(self):
        resume = getattr(self.config, "resume", None)
        if not resume or bool(getattr(self.config, "resume_model_only", False)):
            return
        resume_dir = Path(resume)
        self._validate_checkpoint_metadata(resume_dir)
        if not resume_dir.is_dir():
            raise FileNotFoundError(
                f"training.resume must point to a checkpoint directory: {resume}"
            )
        if self.accelerator is None:
            model_path = resume_dir / "model.pt"
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"single-GPU checkpoint missing model.pt: {model_path}"
                )
            payload = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(payload["model_state_dict"])
            self.global_step = int(payload.get("step", _checkpoint_step(resume_dir)))
        else:
            self.accelerator.load_state(str(resume_dir))
            self.global_step = int(_checkpoint_step(resume_dir))
        self._resume_step = self.global_step
        if self._is_main_process():
            logger.info(
                f"Resumed training state from {resume_dir} at step={self.global_step}"
            )

    def _apply_resume_lr_override_if_requested(self):
        resume_lr = getattr(self.config, "resume_lr", None)
        resume_video_lr = getattr(self.config, "resume_video_learning_rate", None)
        resume_action_lr = getattr(self.config, "resume_action_learning_rate", None)
        resume_scheduler = getattr(self.config, "resume_lr_scheduler_type", None)
        if (
            resume_lr is None
            and resume_video_lr is None
            and (resume_action_lr is None)
            and (not resume_scheduler)
        ):
            return
        if not getattr(self.config, "resume", None):
            raise ValueError(
                "resume LR overrides and resume_lr_scheduler_type require training.resume"
            )
        current_video_lr, current_action_lr = _optimizer_learning_rates(self.optimizer)
        if resume_lr is not None:
            current_video_lr = float(resume_lr)
            current_action_lr = float(resume_lr)
        if resume_video_lr is not None:
            current_video_lr = float(resume_video_lr)
        if resume_action_lr is not None:
            current_action_lr = float(resume_action_lr)
        if current_video_lr <= 0.0 or current_action_lr <= 0.0:
            raise ValueError("resume video/action learning rates must be positive")
        group_targets = {"video": current_video_lr, "action": current_action_lr}
        for group in self.optimizer.param_groups:
            name = str(group.get("name", "video"))
            target_lr = group_targets.get(name, current_video_lr)
            group["lr"] = target_lr
            group["initial_lr"] = target_lr
            group["base_lr"] = target_lr
            group["min_lr"] = float(self.config.min_lr) * target_lr / current_video_lr
        sched_type = str(resume_scheduler or "").strip().lower()
        if sched_type:
            remaining_steps = max(int(self.max_steps) - int(self.global_step), 1)
            self.lr_scheduler = _build_lr_scheduler(
                self.optimizer,
                scheduler_type=sched_type,
                total_steps=remaining_steps,
                warmup_steps=int(getattr(self.config, "resume_warmup_steps", 0)),
                base_lr=current_video_lr,
                min_lr=float(self.config.min_lr),
                warmup_start_factor=float(
                    getattr(self.config, "resume_warmup_start_factor", 0.2)
                ),
            )
        if self.accelerator is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        if self._is_main_process():
            current_video_lr, current_action_lr = _optimizer_learning_rates(
                self.optimizer
            )
            logger.info(
                "Applied resume LR override: video_lr=%.2e action_lr=%.2e scheduler=%s remaining_steps=%d",
                current_video_lr,
                current_action_lr,
                sched_type or "restored",
                max(int(self.max_steps) - int(self.global_step), 0),
            )

    def train(self):
        """Main training loop."""
        logger.info(
            f"Starting training: max_steps={self.max_steps}, batch_size={self.config.batch_size}"
        )
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective global batch size: {self.effective_global_batch_size}")
        self.model.train()

        def _inf_loader(dl):
            while True:
                yield from dl

        data_iter = _inf_loader(self.train_dataloader)
        start_time = time.time()
        metric_accumulator = _RunningMetricMean()
        nonfinite_updates_skipped = 0
        consecutive_nonfinite_updates = 0
        local_micro_step = 0
        stop_step = self._target_stop_step()
        while self.global_step < stop_step:
            batch = next(data_iter)
            if not self.accelerator:
                model_param = self._model_compute_param()
                device = model_param.device
                model_dtype = model_param.dtype
                new_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(device)
                        if v.dtype.is_floating_point and v.dtype != model_dtype:
                            v = v.to(model_dtype)
                    new_batch[k] = v
                batch = new_batch
            training_model = self._unwrap()
            set_training_progress = getattr(
                training_model, "set_training_progress", None
            )
            if set_training_progress is not None:
                set_training_progress(self.global_step, self.max_steps)
            if self.accelerator:
                stepped = False
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        loss, loss_dict = training_model.training_step(batch)
                    metric_accumulator.update(loss_dict)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        grad_is_finite = self._distributed_grad_norm_is_finite(
                            grad_norm
                        )
                        if grad_is_finite or not self.config.skip_nonfinite_updates:
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            self.global_step += 1
                            self._maybe_unfreeze_staged()
                            consecutive_nonfinite_updates = 0
                            stepped = True
                        else:
                            self.optimizer.zero_grad(set_to_none=True)
                            metric_accumulator.reset()
                            nonfinite_updates_skipped += 1
                            consecutive_nonfinite_updates += 1
                            if self._is_main_process():
                                logger.warning(
                                    "Skipped optimizer update because the global gradient norm is non-finite (consecutive=%d, total=%d)",
                                    consecutive_nonfinite_updates,
                                    nonfinite_updates_skipped,
                                )
                            if (
                                consecutive_nonfinite_updates
                                >= self.config.max_consecutive_nonfinite_updates
                            ):
                                raise FloatingPointError(
                                    f"Aborting after {consecutive_nonfinite_updates} consecutive non-finite optimizer updates"
                                )
                if stepped:
                    should_log = self.global_step % self.config.log_every == 0
                    averaged_loss_dict = (
                        self._distributed_metric_mean(metric_accumulator)
                        if should_log
                        else None
                    )
                    metric_accumulator.reset()
                    if should_log:
                        elapsed = time.time() - start_time
                        steps_per_sec = max(
                            0, self.global_step - (self._resume_step or 0)
                        ) / max(elapsed, 1e-09)
                        samples_per_sec = (
                            steps_per_sec * self.effective_global_batch_size
                        )
                        video_lr, action_lr = _optimizer_learning_rates(self.optimizer)
                        grad_norm_value = float(
                            torch.as_tensor(grad_norm).detach().float().cpu().item()
                        )
                        max_grad_norm = float(self.config.max_grad_norm)
                        clip_ratio = (
                            min(1.0, max_grad_norm / max(grad_norm_value, 1e-12))
                            if max_grad_norm > 0.0
                            else 1.0
                        )
                        optimizer_metrics = {
                            "grad_norm": grad_norm_value,
                            "grad_clip_ratio": clip_ratio,
                            "grad_was_clipped": float(
                                max_grad_norm > 0.0 and grad_norm_value > max_grad_norm
                            ),
                            "nonfinite_updates_skipped": float(
                                nonfinite_updates_skipped
                            ),
                        }
                        if self._is_main_process():
                            logger.info(
                                f"step={self.global_step} | loss={averaged_loss_dict['loss_total']:.4f} | video={averaged_loss_dict['loss_video']:.4f} | action={averaged_loss_dict['loss_action']:.4f}{self._format_action_monitor(averaged_loss_dict)} | grad_norm={grad_norm_value:.4f} | clip_ratio={clip_ratio:.3f} | video_lr={video_lr:.2e} | action_lr={action_lr:.2e} | steps/s={steps_per_sec:.2f} | samples/s={samples_per_sec:.2f}"
                            )
                            self._wandb_log(
                                self._train_wandb_payload(
                                    averaged_loss_dict,
                                    video_lr,
                                    steps_per_sec,
                                    action_lr=action_lr,
                                    optimizer_metrics=optimizer_metrics,
                                    samples_per_sec=samples_per_sec,
                                )
                            )
                    if self.global_step % self.config.save_every == 0:
                        self._save_checkpoint()
                    if (
                        self.val_dataset is not None
                        and self.global_step % self.config.eval_every == 0
                    ):
                        self._run_eval()
            else:
                loss, loss_dict = training_model.training_step(batch)
                metric_accumulator.update(loss_dict)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                local_micro_step += 1
                if local_micro_step % self.config.gradient_accumulation_steps != 0:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                grad_norm_value = float(grad_norm.detach().float().cpu().item())
                grad_is_finite = bool(torch.isfinite(grad_norm).item())
                if grad_is_finite or not self.config.skip_nonfinite_updates:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self._maybe_unfreeze_staged()
                    consecutive_nonfinite_updates = 0
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    metric_accumulator.reset()
                    nonfinite_updates_skipped += 1
                    consecutive_nonfinite_updates += 1
                    logger.warning(
                        "Skipped optimizer update because the gradient norm is non-finite (consecutive=%d, total=%d)",
                        consecutive_nonfinite_updates,
                        nonfinite_updates_skipped,
                    )
                    if (
                        consecutive_nonfinite_updates
                        >= self.config.max_consecutive_nonfinite_updates
                    ):
                        raise FloatingPointError(
                            f"Aborting after {consecutive_nonfinite_updates} consecutive non-finite optimizer updates"
                        )
                    continue
                averaged_loss_dict = metric_accumulator.mean()
                metric_accumulator.reset()
                if self.global_step % self.config.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = max(
                        0, self.global_step - (self._resume_step or 0)
                    ) / max(elapsed, 1e-09)
                    samples_per_sec = steps_per_sec * self.effective_global_batch_size
                    video_lr, action_lr = _optimizer_learning_rates(self.optimizer)
                    max_grad_norm = float(self.config.max_grad_norm)
                    clip_ratio = (
                        min(1.0, max_grad_norm / max(grad_norm_value, 1e-12))
                        if max_grad_norm > 0.0
                        else 1.0
                    )
                    optimizer_metrics = {
                        "grad_norm": grad_norm_value,
                        "grad_clip_ratio": clip_ratio,
                        "grad_was_clipped": float(
                            max_grad_norm > 0.0 and grad_norm_value > max_grad_norm
                        ),
                        "nonfinite_updates_skipped": float(nonfinite_updates_skipped),
                    }
                    logger.info(
                        f"step={self.global_step} | loss={averaged_loss_dict['loss_total']:.4f} | video={averaged_loss_dict['loss_video']:.4f} | action={averaged_loss_dict['loss_action']:.4f}{self._format_action_monitor(averaged_loss_dict)} | grad_norm={grad_norm_value:.4f} | clip_ratio={clip_ratio:.3f} | video_lr={video_lr:.2e} | action_lr={action_lr:.2e} | steps/s={steps_per_sec:.2f} | samples/s={samples_per_sec:.2f}"
                    )
                    self._wandb_log(
                        self._train_wandb_payload(
                            averaged_loss_dict,
                            video_lr,
                            steps_per_sec,
                            action_lr=action_lr,
                            optimizer_metrics=optimizer_metrics,
                            samples_per_sec=samples_per_sec,
                        )
                    )
                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint()
                if (
                    self.val_dataset is not None
                    and self.global_step % self.config.eval_every == 0
                ):
                    self._run_eval()
        if self.global_step >= self.max_steps:
            logger.info(f"Training complete. Total steps: {self.global_step}")
            if self.global_step % self.config.save_every != 0:
                self._save_checkpoint()
            if (
                self.val_dataset is not None
                and self.global_step % self.config.eval_every != 0
            ):
                self._run_eval()
        else:
            logger.info(
                f"Debug stop reached at step={self.global_step} (scheduler max_steps={self.max_steps})"
            )

    def _target_stop_step(self) -> int:
        debug_stop = getattr(self.config, "debug_stop_after_steps", None)
        if debug_stop is None:
            return self.max_steps
        if debug_stop <= 0:
            raise ValueError(
                "training.debug_stop_after_steps must be positive when set"
            )
        return min(
            self.max_steps, (self._resume_step or self.global_step) + int(debug_stop)
        )

    def _validate_checkpoint_metadata(self, checkpoint_dir: Path) -> None:
        expected = dict(getattr(self.config, "checkpoint_metadata", {}) or {})
        if not expected:
            return
        state_path = checkpoint_dir / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"checkpoint metadata is required but missing: {state_path}"
            )
        actual = json.loads(state_path.read_text(encoding="utf-8")).get(
            "checkpoint_metadata", {}
        )
        mismatches = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint benchmark contract mismatch: {mismatches}")

    def _is_main_process(self) -> bool:
        """Return True if this is the rank-0 process (or no accelerator)."""
        if self.accelerator is None:
            return True
        return getattr(self.accelerator, "is_main_process", True)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        save_dir = os.path.join(
            self.config.output_dir, f"checkpoint-{self.global_step}"
        )
        os.makedirs(save_dir, exist_ok=True)
        if self.accelerator:
            self.accelerator.save_state(save_dir)
        else:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "step": self.global_step,
                    "checkpoint_metadata": dict(
                        getattr(self.config, "checkpoint_metadata", {}) or {}
                    ),
                },
                os.path.join(save_dir, "model.pt"),
            )
        if self._is_main_process():
            with open(
                os.path.join(save_dir, "trainer_state.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "global_step": self.global_step,
                        "checkpoint_metadata": dict(
                            getattr(self.config, "checkpoint_metadata", {}) or {}
                        ),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved checkpoint to {save_dir}")
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Keep only the last ``save_total_limit`` checkpoints (by step)."""
        limit = getattr(self.config, "save_total_limit", None)
        if not limit or limit <= 0:
            return
        try:
            import re
            import shutil

            out_dir = self.config.output_dir
            if not os.path.isdir(out_dir):
                return
            ckpts = []
            for name in os.listdir(out_dir):
                m = re.fullmatch("checkpoint-(\\d+)", name)
                if m:
                    ckpts.append((int(m.group(1)), os.path.join(out_dir, name)))
            ckpts.sort(key=lambda x: x[0])
            to_delete = ckpts[: max(0, len(ckpts) - int(limit))]
            for _, p in to_delete:
                shutil.rmtree(p, ignore_errors=True)
                logger.info(f"Removed old checkpoint: {p}")
        except Exception as e:
            logger.warning(f"checkpoint cleanup failed: {e}")

    def _run_eval(self):
        """Run separate all-noise action and clean-action world validation.

        ALL ranks run eval simultaneously (no barriers, no rank-0-only path).
        This is critical: any barrier asymmetry between rank 0 and other ranks
        desynchronises the NCCL sequence counter, causing the next epoch-start
        ``synchronize_rng_states`` broadcast (NumelIn=5056) to hang.
        Only rank 0 logs the results.
        """
        from onewm.eval.validation import (
            evaluate_action_denoising,
            evaluate_world_denoising,
        )

        if self.val_dataset is None:
            return
        was_training = self.model.training
        self.model.eval()
        try:
            run_action_eval = bool(getattr(self.config, "eval_action_denoising", True))
            video_max_samples = max(
                1, int(getattr(self.config, "eval_video_max_samples", 16))
            )
            sample_seed = int(
                getattr(self.config, "eval_sample_seed", self.config.seed)
            )
            video_indices = _fixed_random_eval_indices(
                len(self.val_dataset), video_max_samples, sample_seed
            )
            model_param = self._model_compute_param()
            device = model_param.device
            model_dtype = model_param.dtype
            action_result = None
            if run_action_eval:
                action_max_samples = max(1, int(self.config.eval_max_samples))
                action_indices = _fixed_random_eval_indices(
                    len(self.val_dataset), action_max_samples, sample_seed
                )
                action_result = evaluate_action_denoising(
                    self._unwrap(),
                    self.val_dataset,
                    sample_indices=action_indices,
                    sample_seed=sample_seed,
                    num_inference_steps=int(self.config.eval_num_inference_steps),
                    device=device,
                    dtype=model_dtype,
                )
            video_result = evaluate_world_denoising(
                self._unwrap(),
                self.val_dataset,
                sample_indices=video_indices,
                sample_seed=sample_seed,
                num_inference_steps=int(self.config.eval_num_inference_steps),
                device=device,
                dtype=model_dtype,
            )
            if self._is_main_process():
                payload = {
                    "eval/video_psnr": video_result["video_psnr"],
                    "eval/video_ssim": video_result["video_ssim"],
                    "eval/video_n_samples": video_result["video_n_samples"],
                }
                if action_result is None:
                    logger.info(
                        "[eval] step=%d | video_psnr=%.3f | video_ssim=%.4f (n=%d, all-clean-action; action eval disabled)",
                        self.global_step,
                        video_result["video_psnr"],
                        video_result["video_ssim"],
                        video_result["video_n_samples"],
                    )
                else:
                    logger.info(
                        "[eval] step=%d | action_mse=%.6f (n=%d, all-noise) | video_psnr=%.3f | video_ssim=%.4f (n=%d, all-clean-action)",
                        self.global_step,
                        action_result["action_mse"],
                        action_result["action_n_samples"],
                        video_result["video_psnr"],
                        video_result["video_ssim"],
                        video_result["video_n_samples"],
                    )
                    payload.update(
                        {
                            "eval/action_mse": action_result["action_mse"],
                            "eval/action_n_samples": action_result["action_n_samples"],
                        }
                    )
                self._wandb_log(payload)
        except Exception as e:
            if self._is_main_process():
                logger.warning(f"[eval] skipped due to error: {e}")
        finally:
            if was_training:
                self.model.train()
