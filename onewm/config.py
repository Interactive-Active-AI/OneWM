"""OneWM configuration system. Dataclass-based, no Hydra."""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class BackboneConfig:
    """Backbone config -- only user-facing choices. Internal dims auto-inferred."""

    type: str = "wan22_5b"
    pretrained_model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    load_text_encoder: bool = True
    tokenizer_max_len: int = 512


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    train_shift: float = 5.0
    infer_shift: float = 5.0
    stratified_timestep_sampling: bool = False


@dataclass
class FrameworkConfig:
    type: str = "mot"
    action_dim: int = 7
    action_gripper_dim: int = -1
    action_gripper_dims: Optional[list[int]] = None
    chunk_size: int = 16
    action_expert_type: str = "action_dit"
    action_expert_hidden_dim: int = 1536
    action_expert_num_layers: Optional[int] = None
    video_scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    action_scheduler: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(train_shift=5.0, infer_shift=5.0)
    )
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0
    mot_checkpoint_mixed_attn: bool = True
    action_expert_use_gradient_checkpointing: bool = False
    proprio_dim: Optional[int] = None
    state_conditioning: str = "none"
    state_token_dim: int = 14
    state_num_tokens: Optional[int] = None
    state_token_semantics: str = "legacy"
    state_dropout_prob: float = 0.0
    state_null_mode: str = "learned"
    state_fusion_num_heads: int = 8
    action_video_conditioning: str = "first_frame"
    video_clean_prefix_latent_steps: int = 1
    action_cached_aligned_mask: bool = False
    action_clean_prefix_enabled: bool = True
    action_clean_prefix_prob_none: float = 0.60
    action_clean_prefix_prob_partial: float = 0.25
    action_clean_prefix_prob_full: float = 0.15
    action_clean_prefix_boundary_ratio: float = 0.6
    action_clean_prefix_jitter: int = 2
    action_clean_prefix_jitter_weights: list = field(
        default_factory=lambda: [0.1, 0.15, 0.5, 0.15, 0.1]
    )
    action_clean_prefix_curriculum_start_fraction: float = 0.0
    action_clean_prefix_warmup_fraction: float = 0.0
    action_clean_prefix_initial_prob_none: float = 0.60
    action_clean_prefix_initial_prob_partial: float = 0.25
    action_clean_prefix_initial_prob_full: float = 0.15
    action_expert_init_from: Optional[str] = None
    action_expert_head_init: Literal["random", "zero", "payload"] = "random"

    def __post_init__(self) -> None:
        if self.state_conditioning not in {"none", "action_post_mot"}:
            raise ValueError(
                f"state_conditioning must be 'none' or 'action_post_mot', got {self.state_conditioning!r}"
            )
        if not 0.0 <= self.state_dropout_prob <= 1.0:
            raise ValueError("state_dropout_prob must be in [0, 1]")
        if self.state_null_mode != "learned":
            raise ValueError("only state_null_mode='learned' is supported")
        if self.state_token_dim <= 0:
            raise ValueError("state_token_dim must be positive")
        if self.state_num_tokens is not None and self.state_num_tokens <= 0:
            raise ValueError("state_num_tokens must be positive when set")
        proprio_dim = self.proprio_dim
        if proprio_dim is not None and proprio_dim <= 0:
            raise ValueError("proprio_dim must be positive when set")
        if self.state_conditioning != "none":
            if self.state_num_tokens is None and proprio_dim is None:
                raise ValueError(
                    "action_post_mot state conditioning requires either state_num_tokens or legacy proprio_dim"
                )
            if (
                self.state_num_tokens is None
                and proprio_dim is not None
                and (proprio_dim % self.state_token_dim != 0)
            ):
                raise ValueError(
                    f"proprio_dim must be divisible by state_token_dim for action_post_mot conditioning, got {proprio_dim} and {self.state_token_dim}"
                )
            if proprio_dim is not None and self.state_num_tokens is not None:
                expected = self.state_token_dim * self.state_num_tokens
                if proprio_dim != expected:
                    raise ValueError(
                        f"proprio_dim={proprio_dim} does not match state_token_dim*state_num_tokens={expected}"
                    )
        if self.state_fusion_num_heads <= 0:
            raise ValueError("state_fusion_num_heads must be positive")
        if self.action_video_conditioning not in {
            "first_frame",
            "clean_prefix",
            "full_video",
        }:
            raise ValueError(
                "action_video_conditioning must be first_frame, clean_prefix, or full_video"
            )
        if self.video_clean_prefix_latent_steps <= 0:
            raise ValueError("video_clean_prefix_latent_steps must be positive")
        if not self.action_clean_prefix_enabled:
            return
        final_probs = (
            self.action_clean_prefix_prob_none,
            self.action_clean_prefix_prob_partial,
            self.action_clean_prefix_prob_full,
        )
        initial_probs = (
            self.action_clean_prefix_initial_prob_none,
            self.action_clean_prefix_initial_prob_partial,
            self.action_clean_prefix_initial_prob_full,
        )
        for name, probs in (("final", final_probs), ("initial", initial_probs)):
            if any((prob < 0.0 or prob > 1.0 for prob in probs)):
                raise ValueError(
                    f"action clean-prefix {name} probabilities must be in [0, 1]"
                )
            if abs(sum(probs) - 1.0) > 1e-06:
                raise ValueError(
                    f"action clean-prefix {name} probabilities must sum to 1, got {sum(probs):.8f}"
                )
        if not 0.0 <= self.action_clean_prefix_boundary_ratio <= 1.0:
            raise ValueError("action_clean_prefix_boundary_ratio must be in [0, 1]")
        if self.action_clean_prefix_jitter < 0:
            raise ValueError("action_clean_prefix_jitter must be non-negative")
        expected_weights = 2 * self.action_clean_prefix_jitter + 1
        if len(self.action_clean_prefix_jitter_weights) != expected_weights:
            raise ValueError(
                f"action_clean_prefix_jitter_weights must contain {expected_weights} entries for jitter={self.action_clean_prefix_jitter}"
            )
        if any((weight < 0.0 for weight in self.action_clean_prefix_jitter_weights)):
            raise ValueError("action_clean_prefix_jitter_weights must be non-negative")
        if sum(self.action_clean_prefix_jitter_weights) <= 0.0:
            raise ValueError(
                "action_clean_prefix_jitter_weights must have positive total weight"
            )
        start = self.action_clean_prefix_curriculum_start_fraction
        end = self.action_clean_prefix_warmup_fraction
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError(
                "clean-prefix curriculum fractions must satisfy 0 <= start <= warmup <= 1"
            )


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    learning_rate: float = 1e-05
    video_learning_rate: Optional[float] = None
    action_learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    expected_global_batch_size: Optional[int] = None
    num_epochs: int = 100
    max_steps: Optional[int] = None
    debug_stop_after_steps: Optional[int] = None
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    skip_nonfinite_updates: bool = True
    max_consecutive_nonfinite_updates: int = 3
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 500
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    min_lr: float = 0.0
    warmup_steps: Optional[int] = None
    warmup_start_factor: float = 0.01
    save_total_limit: Optional[int] = None
    resume: Optional[str] = None
    resume_model_only: bool = False
    resume_lr: Optional[float] = None
    resume_video_learning_rate: Optional[float] = None
    resume_action_learning_rate: Optional[float] = None
    resume_lr_scheduler_type: Optional[str] = None
    resume_warmup_steps: int = 0
    resume_warmup_start_factor: float = 0.2
    strategy: str = "full"
    staged_warmup_steps: int = 1000
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list = field(
        default_factory=lambda: [
            "self_attn.q",
            "self_attn.k",
            "self_attn.v",
            "self_attn.o",
            "cross_attn.q",
            "cross_attn.k",
            "cross_attn.v",
            "cross_attn.o",
            "ffn.0",
            "ffn.2",
        ]
    )
    num_workers: int = 4
    dataloader_timeout_seconds: int = 120
    eval_num_inference_steps: int = 20
    eval_action_denoising: bool = True
    warmup_ratio: float = 0.05
    wandb_enabled: bool = False
    wandb_required: bool = False
    wandb_project: str = "onewm"
    wandb_run_name: Optional[str] = None
    eval_max_samples: int = 128
    eval_video_max_samples: int = 16
    eval_sample_seed: int = 42
    checkpoint_metadata: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    dataset_type: str = "synthetic"
    root: Optional[str] = None
    split_manifest: Optional[str] = None
    dataset_dirs: list = field(default_factory=list)
    num_frames: int = 17
    video_size: list = field(default_factory=lambda: [256, 256])
    video_key: str = "observation.images.cam_high"
    video_keys: list = field(default_factory=list)
    concat_multi_camera: str = "horizontal"
    video_augmentation_enabled: bool = False
    eval_video_augmentation_enabled: bool = False
    video_augmentation_probability: float = 0.25
    video_random_crop_ratio: float = 0.95
    video_brightness: float = 0.2
    video_contrast: float = 0.2
    video_saturation: float = 0.2
    video_hue: float = 0.05
    action_key: str = "action"
    state_key: str = "observation.state"
    action_freq_ratio: int = 1
    text_len: int = 512
    text_embedding_cache_dir: Optional[str] = None
    text_prompt_template: str = "A video recorded from a robot's point of view executing the following instruction: {task}"
    text_cache_encoder_id: str = "wan22ti2v5b"
    normalize_actions: bool = False
    action_norm_mode: str = "minmax"
    action_stats_path: Optional[str] = None
    normalize_states: bool = False
    state_norm_mode: str = "minmax"
    state_stats_path: Optional[str] = None
    delta_action_dim_mask: Optional[list] = None
    action_representation: str = "raw"
    state_representation: str = "raw"
    val_split: float = 0.0
    val_split_seed: int = 42
    observation_history_offsets: list = field(default_factory=list)
    video_future_offsets: list = field(default_factory=list)
    repeat_current_for_vae_alignment: int = 0
    view_separator_width: int = 0
    view_separator_value: int = 128
    expected_num_tasks: Optional[int] = None
    require_complete_dataset: bool = False
    action_binary_dims: list = field(default_factory=list)


@dataclass
class InferenceConfig:
    num_inference_steps: int = 20
    seed: int = 42


@dataclass
class OneWMConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


import dataclasses as _dc
from pathlib import Path as _Path
from typing import Any as _Any


def _from_dict(cls, raw: _Any):
    """Recursively materialize a dataclass instance from a (possibly nested)
    dict. Unknown keys are rejected to catch typos early."""
    if raw is None:
        return cls()
    if not _dc.is_dataclass(cls):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(raw)}")
    field_map = {f.name: f for f in _dc.fields(cls)}
    unknown = set(raw.keys()) - set(field_map.keys())
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}. Allowed: {sorted(field_map.keys())}"
        )
    kwargs: dict = {}
    for name, fld in field_map.items():
        if name not in raw:
            continue
        val = raw[name]
        ftype = fld.type
        if isinstance(ftype, str):
            ftype = eval(ftype, globals(), locals())
        if _dc.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[name] = _from_dict(ftype, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_config(path: str | _Path) -> OneWMConfig:
    """Load a YAML recipe and return a fully-populated :class:`OneWMConfig`.

    - Unknown keys raise immediately (no silent typos).
    - Nested sub-configs (`backbone`, `framework`, `training`, `data`,
      `inference`, and the two `*_scheduler`s under framework) are resolved
      recursively.
    - Missing top-level sections fall back to dataclass defaults.

    Example:
        cfg = load_config("configs/libero.yaml")
        model = build_framework(cfg)
    """
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "load_config requires PyYAML. Install with: pip install pyyaml"
        ) from e
    p = _Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw)}")
    return _from_dict(OneWMConfig, raw)


def config_to_dict(cfg: OneWMConfig) -> dict:
    """Serialize a config back to a plain dict (useful for wandb/logging)."""
    return _dc.asdict(cfg)
