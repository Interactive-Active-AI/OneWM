"""MoT WAM implementation.

MoT WAM uses separate video and action experts, with per-layer joint attention
between their Q/K/V streams. The implementation keeps Fast-WAM-aligned details:
configurable clean-prefix pinning and action attention,
velocity target ``noise - sample``, weighted flow matching, and action inference.
"""

from __future__ import annotations
from typing import Any, Optional
import torch
import torch.nn as nn
from torch import Tensor
from onewm.action_model import build_action_expert, load_action_dit_init
from onewm.training.flow import (
    add_flow_noise,
    add_partial_clean_action_noise,
    build_inference_schedule,
    video_latent_pad_mask,
)
from onewm.wam.base import WAMModel
from onewm.backbone.base import BaseBackbone
from onewm.config import FrameworkConfig
from onewm.modules.mot import MoT
from onewm.modules.scheduler import FlowMatchScheduler
from onewm.training.loss import flow_matching_loss
from onewm.training.metrics import action_monitor_metrics


class MoTWAM(WAMModel):
    """Fast-WAM/Motus-style multi-stream WAM."""

    def __init__(
        self,
        backbone: BaseBackbone,
        config: FrameworkConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self._device = device
        self._dtype = dtype
        self.backbone = backbone
        self.action_expert = build_action_expert(backbone.info, config)
        load_action_dit_init(
            self.action_expert,
            getattr(config, "action_expert_init_from", None),
            head_init=getattr(config, "action_expert_head_init", "random"),
        )
        self.mot = MoT(
            experts={"video": backbone.get_dit(), "action": self.action_expert},
            checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        )
        self.video_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.video_scheduler.num_train_timesteps,
            shift=config.video_scheduler.train_shift,
            stratified_timestep_sampling=config.video_scheduler.stratified_timestep_sampling,
        )
        self.action_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.action_scheduler.num_train_timesteps,
            shift=config.action_scheduler.train_shift,
            stratified_timestep_sampling=config.action_scheduler.stratified_timestep_sampling,
        )
        self.loss_lambda_video = config.loss_lambda_video
        self.loss_lambda_action = config.loss_lambda_action
        self.action_video_conditioning = getattr(
            config, "action_video_conditioning", "first_frame"
        )
        self.video_clean_prefix_latent_steps = int(
            getattr(config, "video_clean_prefix_latent_steps", 1)
        )
        self._training_progress = 1.0
        if self.action_video_conditioning not in {
            "first_frame",
            "clean_prefix",
            "full_video",
        }:
            raise ValueError(
                f"framework.action_video_conditioning must be first_frame, clean_prefix, or full_video, got {self.action_video_conditioning!r}"
            )
        self.proprio_dim: Optional[int] = None
        self.state_token_dim: Optional[int] = None
        self.state_num_tokens: Optional[int] = None
        self.state_conditioning = str(getattr(config, "state_conditioning", "none"))
        self.state_dropout_prob = float(getattr(config, "state_dropout_prob", 0.0))
        self.state_encoder: Optional[nn.Sequential] = None
        self.state_norm: Optional[nn.LayerNorm] = None
        self.state_cross_attention: Optional[nn.MultiheadAttention] = None
        self.learned_null_state: Optional[nn.Parameter] = None
        self.state_gate: Optional[nn.Parameter] = None
        if self.state_conditioning == "action_post_mot":
            raw_proprio_dim = getattr(config, "proprio_dim", None)
            state_token_dim = int(getattr(config, "state_token_dim", 14))
            configured_tokens = getattr(config, "state_num_tokens", None)
            if configured_tokens is not None:
                state_num_tokens = int(configured_tokens)
            elif raw_proprio_dim is not None:
                state_num_tokens = int(raw_proprio_dim) // state_token_dim
            else:
                raise ValueError(
                    "action_post_mot requires state_num_tokens or proprio_dim"
                )
            self.proprio_dim = None if raw_proprio_dim is None else int(raw_proprio_dim)
            self.state_token_dim = state_token_dim
            self.state_num_tokens = state_num_tokens
            hidden_dim = int(self.action_expert.hidden_dim)
            num_heads = int(getattr(config, "state_fusion_num_heads", 8))
            if hidden_dim % num_heads:
                raise ValueError(
                    f"action hidden dim {hidden_dim} must be divisible by state_fusion_num_heads={num_heads}"
                )
            self.state_encoder = nn.Sequential(
                nn.Linear(state_token_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ).to(device=device, dtype=dtype)
            self.state_norm = nn.LayerNorm(hidden_dim).to(device=device, dtype=dtype)
            self.state_cross_attention = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True
            ).to(device=device, dtype=dtype)
            self.learned_null_state = nn.Parameter(
                torch.zeros(1, state_num_tokens, hidden_dim, device=device, dtype=dtype)
            )
            self.state_gate = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.backbone.get_dit().to(device=device, dtype=dtype)
        self.action_expert.to(device=device, dtype=dtype)
        self.mot.to(device=device, dtype=dtype)

    def set_training_progress(self, global_step: int, max_steps: int) -> None:
        """Set optimizer-step progress used by the clean-prefix curriculum."""
        if max_steps <= 0:
            self._training_progress = 1.0
        else:
            self._training_progress = min(
                max(float(global_step) / float(max_steps), 0.0), 1.0
            )

    def _clean_prefix_probabilities(self) -> tuple[float, float, float]:
        final = (
            float(self.config.action_clean_prefix_prob_none),
            float(self.config.action_clean_prefix_prob_partial),
            float(self.config.action_clean_prefix_prob_full),
        )
        start = float(self.config.action_clean_prefix_curriculum_start_fraction)
        end = float(self.config.action_clean_prefix_warmup_fraction)
        if self._training_progress >= end:
            return final
        initial = (
            float(self.config.action_clean_prefix_initial_prob_none),
            float(self.config.action_clean_prefix_initial_prob_partial),
            float(self.config.action_clean_prefix_initial_prob_full),
        )
        if self._training_progress <= start or end <= start:
            return initial
        alpha = (self._training_progress - start) / (end - start)
        interpolated = tuple(
            (
                initial_prob + alpha * (final_prob - initial_prob)
                for initial_prob, final_prob in zip(initial, final)
            )
        )
        total = sum(interpolated)
        return tuple((prob / total for prob in interpolated))

    def training_step(self, sample: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        video = sample["video"]
        action = sample["action"]
        context = sample["context"]
        context_mask = sample.get("context_mask")
        action_is_pad = sample.get("action_is_pad")
        image_is_pad = sample.get("image_is_pad")
        action_pad_mask = (
            None
            if action_is_pad is None
            else action_is_pad.to(device=action.device, dtype=torch.bool)
        )
        proprio = self._proprio_from_sample(sample)
        with torch.no_grad():
            video_latents = self.backbone.encode_video(video)
        training_clean_steps = (
            self.video_clean_prefix_latent_steps
            if video_latents.shape[2] > self.video_clean_prefix_latent_steps
            else 0
        )
        noisy_video, target_video, t_video = add_flow_noise(
            self.video_scheduler, video_latents, pin_latent_steps=training_clean_steps
        )
        if self.config.action_clean_prefix_enabled:
            prob_none, prob_partial, prob_full = self._clean_prefix_probabilities()
            action_noise = add_partial_clean_action_noise(
                self.action_scheduler,
                action,
                action_is_pad=action_pad_mask,
                num_future_frames=max(
                    video_latents.shape[2] - self.video_clean_prefix_latent_steps, 0
                ),
                prob_none=prob_none,
                prob_partial=prob_partial,
                prob_full=prob_full,
                boundary_ratio=float(self.config.action_clean_prefix_boundary_ratio),
                jitter=int(self.config.action_clean_prefix_jitter),
                jitter_weights=self.config.action_clean_prefix_jitter_weights,
            )
            noisy_action = action_noise.noisy
            target_action = action_noise.target
            t_action = action_noise.timesteps
            action_timestep_input = action_noise.token_timesteps
            known_action_mask = action_noise.known_mask
            clean_prefix_lengths = action_noise.clean_prefix_lengths
        else:
            noisy_action, target_action, t_action = add_flow_noise(
                self.action_scheduler, action
            )
            action_timestep_input = t_action
            known_action_mask = torch.zeros(
                action.shape[:2], dtype=torch.bool, device=action.device
            )
            clean_prefix_lengths = torch.zeros(
                action.shape[0], dtype=torch.long, device=action.device
            )
            prob_none, prob_partial, prob_full = (1.0, 0.0, 0.0)
        video_dit = self.backbone.get_dit()
        video_state = self._video_pre_dit(
            video_dit, noisy_video, t_video, context, context_mask
        )
        action_state = self.action_expert.pre_dit(
            noisy_action, action_timestep_input, context, context_mask
        )
        attention_mask = self.build_attention_mask(video_state, action_state)
        output_tokens = self.mot(
            {"video": video_state, "action": action_state}, attention_mask
        )
        pred_video_v = video_dit.post_dit(
            output_tokens["video"], video_state["meta"], video_state["t"]
        )
        pred_action_v: Optional[Tensor] = None
        state_drop_fraction = 0.0
        action_tokens, state_drop_fraction = self._fuse_action_state(
            output_tokens["action"], proprio, apply_dropout=True
        )
        pred_action_v = self.action_expert.post_dit(action_tokens)
        clean_steps = training_clean_steps
        if clean_steps:
            pred_video_v = pred_video_v[:, :, clean_steps:]
            target_video = target_video[:, :, clean_steps:]
        video_is_pad = video_latent_pad_mask(
            image_is_pad,
            latent_steps=pred_video_v.shape[2],
            include_initial_video_step=False,
            excluded_prefix_latent_steps=clean_steps,
        )
        loss_video = flow_matching_loss(
            pred_video_v,
            target_video,
            t_video,
            self.video_scheduler,
            is_pad_mask=video_is_pad,
        )
        action_flow_ignore_mask = known_action_mask
        if action_pad_mask is not None:
            action_flow_ignore_mask = action_flow_ignore_mask | action_pad_mask
        assert pred_action_v is not None
        loss_action = flow_matching_loss(
            pred_action_v,
            target_action,
            t_action,
            self.action_scheduler,
            is_pad_mask=action_flow_ignore_mask,
            average_only_valid_samples=True,
        )
        loss = (
            self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        )
        if action_pad_mask is None:
            valid_action_count = torch.full_like(clean_prefix_lengths, action.shape[1])
        else:
            valid_action_count = (~action_pad_mask).sum(dim=1)
        is_full = (valid_action_count > 0) & (
            clean_prefix_lengths == valid_action_count
        )
        is_none = (clean_prefix_lengths == 0) & ~is_full
        is_partial = ~(is_none | is_full)
        loss_dict = {
            "loss_video": loss_video.item(),
            "loss_action": loss_action.item(),
            "loss_total": loss.item(),
            "action_clean_prefix_length_mean": float(
                clean_prefix_lengths.float().mean().item()
            ),
            "action_clean_token_fraction": float(
                known_action_mask.sum()
                .float()
                .div(valid_action_count.sum().clamp_min(1))
                .item()
            ),
            "action_clean_sample_none_fraction": float(is_none.float().mean().item()),
            "action_clean_sample_partial_fraction": float(
                is_partial.float().mean().item()
            ),
            "action_clean_sample_full_fraction": float(is_full.float().mean().item()),
            "action_clean_prob_none": prob_none,
            "action_clean_prob_partial": prob_partial,
            "action_clean_prob_full": prob_full,
            "state_dropout_sample_fraction": state_drop_fraction,
        }
        if pred_action_v is not None:
            loss_dict.update(
                action_monitor_metrics(
                    pred_action_v.detach(),
                    target_action.detach(),
                    action.detach(),
                    is_pad=action_flow_ignore_mask,
                    gripper_dim=getattr(self.config, "action_gripper_dim", -1),
                    gripper_dims=getattr(self.config, "action_gripper_dims", None),
                )
            )
        return (loss, loss_dict)

    @torch.no_grad()
    def infer_action(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Tensor,
        action_horizon: int,
        num_inference_steps: int = 10,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        proprio = kwargs.get("proprio")
        num_video_frames = kwargs.get("num_video_frames")
        device = input_image.device
        dtype = input_image.dtype
        if input_image.ndim not in {4, 5}:
            raise ValueError(
                f"input_image must be [B,C,H,W] or clean video [B,C,T,H,W], got {tuple(input_image.shape)}"
            )
        batch_size = int(input_image.shape[0])
        if context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
            raise ValueError(
                f"image/context batch mismatch: {batch_size}, {context.shape[0]}, {context_mask.shape[0]}"
            )
        force_null_state = bool(kwargs.get("force_null_state", False))
        generator = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else None
        )
        known_action, known_action_mask = self._prepare_known_action_prefix(
            kwargs.get("known_action"),
            kwargs.get("known_action_mask"),
            action_horizon=action_horizon,
            action_dim=self.config.action_dim,
            device=device,
            dtype=dtype,
        )
        if self.action_video_conditioning == "full_video":
            if num_video_frames is None:
                raise ValueError(
                    "num_video_frames is required for full-video action inference"
                )
            clean_video = (
                input_image.unsqueeze(2) if input_image.ndim == 4 else input_image
            )
            clean_latents = self.backbone.encode_video(clean_video)
            if clean_latents.shape[2] != self.video_clean_prefix_latent_steps:
                raise ValueError(
                    f"encoded clean video prefix has the wrong latent length: {clean_latents.shape[2]} != {self.video_clean_prefix_latent_steps}"
                )
            _, channels, _, height_latent, width_latent = clean_latents.shape
            vae = self.backbone.get_vae()
            latent_steps = max(
                1, (int(num_video_frames) - 1) // vae.temporal_compress + 1
            )
            video_latents = torch.randn(
                batch_size,
                channels,
                latent_steps,
                height_latent,
                width_latent,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            video_latents[:, :, : self.video_clean_prefix_latent_steps] = clean_latents
            action_latents = torch.randn(
                batch_size,
                action_horizon,
                self.config.action_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            if known_action_mask is not None:
                action_latents = torch.where(
                    known_action_mask[..., None], known_action, action_latents
                )
            video_timesteps, video_deltas = build_inference_schedule(
                self.config.video_scheduler, num_inference_steps, device, dtype
            )
            action_timesteps, action_deltas = build_inference_schedule(
                self.config.action_scheduler, num_inference_steps, device, dtype
            )
            video_dit = self.backbone.get_dit()
            for i in range(num_inference_steps):
                t_video = video_timesteps[i].reshape(1).expand(batch_size)
                t_action = action_timesteps[i].reshape(1).expand(batch_size)
                action_timestep_input = self._action_timestep_with_known_prefix(
                    t_action, known_action_mask, action_horizon
                )
                video_state = self._video_pre_dit(
                    video_dit, video_latents, t_video, context, context_mask
                )
                action_state = self.action_expert.pre_dit(
                    action_latents, action_timestep_input, context, context_mask
                )
                attention_mask = self._build_mot_attention_mask(
                    video_seq_len=video_state["tokens"].shape[1],
                    action_seq_len=action_state["tokens"].shape[1],
                    video_tokens_per_frame=self._compute_video_tokens_per_frame(
                        video_state
                    ),
                    device=device,
                    action_video_conditioning="full_video",
                    clean_prefix_latent_steps=self.video_clean_prefix_latent_steps,
                )
                output_tokens = self.mot(
                    {"video": video_state, "action": action_state}, attention_mask
                )
                video_velocity = video_dit.post_dit(
                    output_tokens["video"], video_state["meta"], video_state["t"]
                )
                action_tokens, _ = self._fuse_action_state(
                    output_tokens["action"],
                    proprio,
                    apply_dropout=False,
                    force_null_state=force_null_state,
                )
                action_velocity = self.action_expert.post_dit(action_tokens)
                video_latents = FlowMatchScheduler.step(
                    video_velocity, video_deltas[i], video_latents
                )
                action_latents = FlowMatchScheduler.step(
                    action_velocity, action_deltas[i], action_latents
                )
                if known_action_mask is not None:
                    action_latents = torch.where(
                        known_action_mask[..., None], known_action, action_latents
                    )
                video_latents[:, :, : self.video_clean_prefix_latent_steps] = (
                    clean_latents
                )
            return action_latents
        clean_video = input_image.unsqueeze(2) if input_image.ndim == 4 else input_image
        video_latents = self.backbone.encode_video(clean_video)
        if video_latents.shape[2] != self.video_clean_prefix_latent_steps:
            raise ValueError(
                f"encoded clean video prefix has the wrong latent length: {video_latents.shape[2]} != {self.video_clean_prefix_latent_steps}"
            )
        video_dit = self.backbone.get_dit()
        t_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
        video_state = self._video_pre_dit(
            video_dit, video_latents, t_zero, context, context_mask
        )
        video_tokens_per_frame = self._compute_video_tokens_per_frame(video_state)
        video_only_mask = self._build_mot_attention_mask(
            video_seq_len=video_state["tokens"].shape[1],
            action_seq_len=0,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            action_video_conditioning=self.action_video_conditioning,
            clean_prefix_latent_steps=self.video_clean_prefix_latent_steps,
        )
        video_kv_cache = self.mot.prefill_video_cache(video_state, video_only_mask)
        cached_action_attention_mask = None
        if self.video_clean_prefix_latent_steps > 1 or bool(
            getattr(self.config, "action_cached_aligned_mask", False)
        ):
            if num_video_frames is None:
                raise ValueError(
                    "num_video_frames is required for multi-latent clean-prefix action inference"
                )
            temporal_compress = int(
                getattr(self.backbone.get_vae(), "temporal_compress", 4)
            )
            if temporal_compress <= 0:
                raise ValueError("VAE temporal_compress must be positive")
            full_video_latent_steps = (
                1 + (int(num_video_frames) - 1) // temporal_compress
            )
            cached_action_attention_mask = self._build_cached_action_attention_mask(
                full_video_seq_len=full_video_latent_steps * video_tokens_per_frame,
                clean_video_seq_len=video_state["tokens"].shape[1],
                action_seq_len=action_horizon,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
                action_video_conditioning=self.action_video_conditioning,
                clean_prefix_latent_steps=self.video_clean_prefix_latent_steps,
            )
        action_latents = torch.randn(
            batch_size,
            action_horizon,
            self.config.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        if known_action_mask is not None:
            action_latents = torch.where(
                known_action_mask[..., None], known_action, action_latents
            )
        timesteps, deltas = build_inference_schedule(
            self.config.action_scheduler, num_inference_steps, device, dtype
        )
        for i in range(num_inference_steps):
            t_action = timesteps[i].reshape(1).expand(batch_size)
            action_timestep_input = self._action_timestep_with_known_prefix(
                t_action, known_action_mask, action_horizon
            )
            action_state = self.action_expert.pre_dit(
                action_latents, action_timestep_input, context, context_mask
            )
            if cached_action_attention_mask is None:
                action_out = self.mot.forward_action_with_video_cache(
                    action_state, video_kv_cache
                )
            else:
                action_out = self.mot.forward_action_with_video_cache(
                    action_state,
                    video_kv_cache,
                    attention_mask=cached_action_attention_mask,
                )
            action_out, _ = self._fuse_action_state(
                action_out,
                proprio,
                apply_dropout=False,
                force_null_state=force_null_state,
            )
            velocity = self.action_expert.post_dit(action_out)
            action_latents = FlowMatchScheduler.step(
                velocity, deltas[i], action_latents
            )
            if known_action_mask is not None:
                action_latents = torch.where(
                    known_action_mask[..., None], known_action, action_latents
                )
        return action_latents

    @torch.no_grad()
    def infer_world(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Tensor,
        num_video_frames: int,
        action_horizon: int,
        num_inference_steps: int = 20,
        **kwargs,
    ) -> dict[str, Any]:
        device = input_image.device
        dtype = input_image.dtype
        seed = kwargs.get("seed")
        if kwargs.get("proprio") is not None:
            raise ValueError("infer_world is state-free and does not accept proprio")
        generator = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else None
        )
        known_action, known_action_mask = self._prepare_known_action_prefix(
            kwargs.get("known_action"),
            kwargs.get("known_action_mask"),
            action_horizon=action_horizon,
            action_dim=self.config.action_dim,
            device=device,
            dtype=dtype,
        )
        if input_image.ndim not in {4, 5}:
            raise ValueError(
                f"input_image must be [B,C,H,W] or clean video [B,C,T,H,W], got {tuple(input_image.shape)}"
            )
        if input_image.shape[0] != 1:
            raise ValueError("infer_world currently supports batch size 1")
        clean_video = input_image.unsqueeze(2) if input_image.ndim == 4 else input_image
        clean_latents = self.backbone.encode_video(clean_video)
        if clean_latents.shape[2] != self.video_clean_prefix_latent_steps:
            raise ValueError(
                f"encoded clean video prefix has the wrong latent length: {clean_latents.shape[2]} != {self.video_clean_prefix_latent_steps}"
            )
        _, channels, _, height_latent, width_latent = clean_latents.shape
        vae = self.backbone.get_vae()
        latent_steps = max(1, (int(num_video_frames) - 1) // vae.temporal_compress + 1)
        video_latents = torch.randn(
            1,
            channels,
            latent_steps,
            height_latent,
            width_latent,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        video_latents[:, :, : self.video_clean_prefix_latent_steps] = clean_latents
        action_latents = torch.randn(
            1,
            action_horizon,
            self.config.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        if known_action_mask is not None:
            action_latents = torch.where(
                known_action_mask[..., None], known_action, action_latents
            )
        video_timesteps, video_deltas = build_inference_schedule(
            self.config.video_scheduler, num_inference_steps, device, dtype
        )
        action_timesteps, action_deltas = build_inference_schedule(
            self.config.action_scheduler, num_inference_steps, device, dtype
        )
        video_dit = self.backbone.get_dit()
        for i in range(num_inference_steps):
            t_video = video_timesteps[i].reshape(1)
            t_action = action_timesteps[i].reshape(1)
            action_timestep_input = self._action_timestep_with_known_prefix(
                t_action, known_action_mask, action_horizon
            )
            video_state = self._video_pre_dit(
                video_dit, video_latents, t_video, context, context_mask
            )
            action_state = self.action_expert.pre_dit(
                action_latents, action_timestep_input, context, context_mask
            )
            attention_mask = self.build_attention_mask(video_state, action_state)
            output_tokens = self.mot(
                {"video": video_state, "action": action_state}, attention_mask
            )
            video_velocity = video_dit.post_dit(
                output_tokens["video"], video_state["meta"], video_state["t"]
            )
            video_latents = FlowMatchScheduler.step(
                video_velocity, video_deltas[i], video_latents
            )
            action_velocity = self.action_expert.post_dit(output_tokens["action"])
            action_latents = FlowMatchScheduler.step(
                action_velocity, action_deltas[i], action_latents
            )
            if known_action_mask is not None:
                action_latents = torch.where(
                    known_action_mask[..., None], known_action, action_latents
                )
            video_latents[:, :, : self.video_clean_prefix_latent_steps] = clean_latents
        decoded_video = self.backbone.decode_latents(video_latents)
        return {"video": decoded_video, "action": action_latents}

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs) -> dict[str, Any]:
        """Backward-compatible state-free alias for video/action diagnostics."""
        if kwargs.get("proprio") is not None:
            raise ValueError(
                "infer_joint diagnostics are state-free; use infer_action for WAM state"
            )
        return self.infer_world(*args, **kwargs)

    @staticmethod
    def _prepare_known_action_prefix(
        known_action: Optional[Tensor],
        known_action_mask: Optional[Tensor],
        *,
        action_horizon: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if known_action is None:
            if known_action_mask is not None:
                raise ValueError("known_action_mask requires known_action")
            return (None, None)
        if known_action.ndim == 2:
            known_action = known_action.unsqueeze(0)
        if known_action.ndim != 3 or known_action.shape[0] != 1:
            raise ValueError(
                f"known_action must be [K,D] or [1,K,D], got {tuple(known_action.shape)}"
            )
        if (
            known_action.shape[2] != action_dim
            or known_action.shape[1] > action_horizon
        ):
            raise ValueError(
                f"known_action must fit [1,{action_horizon},{action_dim}], got {tuple(known_action.shape)}"
            )
        known_length = known_action.shape[1]
        full_action = torch.zeros(
            (1, action_horizon, action_dim), device=device, dtype=dtype
        )
        full_action[:, :known_length] = known_action.to(device=device, dtype=dtype)
        if known_action_mask is None:
            full_mask = (
                torch.arange(action_horizon, device=device).unsqueeze(0) < known_length
            )
        else:
            if known_length != action_horizon:
                raise ValueError(
                    "known_action_mask is only accepted when known_action spans action_horizon"
                )
            if known_action_mask.ndim == 1:
                known_action_mask = known_action_mask.unsqueeze(0)
            if tuple(known_action_mask.shape) != (1, action_horizon):
                raise ValueError(
                    f"known_action_mask must be [1,{action_horizon}], got {tuple(known_action_mask.shape)}"
                )
            full_mask = known_action_mask.to(device=device, dtype=torch.bool)
        prefix_length = full_mask.sum(dim=1)
        expected_prefix = (
            torch.arange(action_horizon, device=device).unsqueeze(0)
            < prefix_length[:, None]
        )
        if not torch.equal(full_mask, expected_prefix):
            raise ValueError(
                "known_action_mask must describe one contiguous causal prefix"
            )
        return (full_action, full_mask)

    @staticmethod
    def _action_timestep_with_known_prefix(
        timestep: Tensor, known_action_mask: Optional[Tensor], action_horizon: int
    ) -> Tensor:
        if known_action_mask is None:
            return timestep
        token_timestep = (
            timestep[:, None].expand(timestep.shape[0], action_horizon).clone()
        )
        token_timestep.masked_fill_(known_action_mask, 0.0)
        return token_timestep

    def _proprio_from_sample(self, sample: dict[str, Any]) -> Optional[Tensor]:
        if self.state_conditioning == "none":
            return None
        state_token_dim = int(getattr(self, "state_token_dim", 14))
        state_num_tokens = int(
            getattr(
                self,
                "state_num_tokens",
                int(getattr(self, "proprio_dim", state_token_dim)) // state_token_dim,
            )
        )
        proprio_history = sample.get("proprio_history")
        if proprio_history is not None:
            if proprio_history.ndim != 3:
                raise ValueError(
                    f"sample['proprio_history'] must be [B,N,D], got {tuple(proprio_history.shape)}"
                )
            expected = (state_num_tokens, state_token_dim)
            if tuple(proprio_history.shape[1:]) != expected:
                raise ValueError(
                    f"sample['proprio_history'] must end in {expected}, got {tuple(proprio_history.shape[1:])}"
                )
            return proprio_history
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError(
                "sample['proprio'] is required for action_post_mot conditioning"
            )
        if proprio.ndim != 3:
            raise ValueError(
                f"sample['proprio'] must be [B, T, D], got {tuple(proprio.shape)}"
            )
        if self.proprio_dim is None or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"sample['proprio'] last dim must be {self.proprio_dim}, got {proprio.shape[-1]}"
            )
        return proprio[:, 0, :].reshape(
            proprio.shape[0], state_num_tokens, state_token_dim
        )

    def _fuse_action_state(
        self,
        action_tokens: Tensor,
        proprio: Optional[Tensor],
        *,
        apply_dropout: bool,
        force_null_state: bool = False,
    ) -> tuple[Tensor, float]:
        """Fuse state into the action endpoint only, after the shared MoT."""
        if self.state_conditioning == "none":
            return (action_tokens, 0.0)
        state_encoder = self.state_encoder
        state_norm = self.state_norm
        state_cross_attention = self.state_cross_attention
        learned_null_state = self.learned_null_state
        state_gate = self.state_gate
        state_num_tokens = self.state_num_tokens
        state_token_dim = self.state_token_dim
        if (
            state_encoder is None
            or state_norm is None
            or state_cross_attention is None
            or (learned_null_state is None)
            or (state_gate is None)
            or (state_num_tokens is None)
            or (state_token_dim is None)
        ):
            raise RuntimeError("action_post_mot state modules are not initialized")
        if proprio is None:
            raise ValueError("proprio is required for action_post_mot WAM inference")
        state_token_dim = int(getattr(self, "state_token_dim", 14))
        state_num_tokens = int(
            getattr(
                self,
                "state_num_tokens",
                int(getattr(self, "proprio_dim", state_token_dim)) // state_token_dim,
            )
        )
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim == 2:
            expected_flat = state_num_tokens * state_token_dim
            if proprio.shape[-1] != expected_flat:
                raise ValueError(
                    f"flat proprio must be [B,{expected_flat}], got {tuple(proprio.shape)}"
                )
            proprio = proprio.reshape(
                proprio.shape[0], state_num_tokens, state_token_dim
            )
        if proprio.ndim != 3 or tuple(proprio.shape[1:]) != (
            state_num_tokens,
            state_token_dim,
        ):
            raise ValueError(
                f"proprio must be tokenized [B,N,D] with N={state_num_tokens}, D={state_token_dim}; got {tuple(proprio.shape)}"
            )
        if proprio.shape[0] != action_tokens.shape[0]:
            raise ValueError(
                f"state/action batch mismatch: {proprio.shape[0]} != {action_tokens.shape[0]}"
            )
        encoder_param = next(state_encoder.parameters())
        state = proprio.to(device=encoder_param.device, dtype=encoder_param.dtype)
        state_tokens = state_encoder(state)
        if force_null_state:
            drop_mask = torch.ones(
                state.shape[0], dtype=torch.bool, device=state.device
            )
        elif apply_dropout and self.training and (self.state_dropout_prob > 0):
            drop_mask = (
                torch.rand(state.shape[0], device=state.device)
                < self.state_dropout_prob
            )
        else:
            drop_mask = torch.zeros(
                state.shape[0], dtype=torch.bool, device=state.device
            )
        null_tokens = learned_null_state.expand(state.shape[0], -1, -1)
        state_tokens = torch.where(drop_mask[:, None, None], null_tokens, state_tokens)
        query = state_norm(action_tokens)
        residual, _ = state_cross_attention(
            query=query, key=state_tokens, value=state_tokens, need_weights=False
        )
        fused = action_tokens + state_gate * residual
        return (fused, float(drop_mask.float().mean().item()))

    def _compute_video_tokens_per_frame(self, video_state: dict) -> int:
        meta = video_state.get("meta", {})
        height = int(meta.get("H", 0))
        width = int(meta.get("W", 0))
        _, patch_h, patch_w = self.backbone.info.patch_size
        return max(1, height // patch_h) * max(1, width // patch_w)

    def _video_pre_dit(
        self,
        video_dit: nn.Module,
        latents: Tensor,
        timestep: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor],
    ) -> dict:
        """Prepare video tokens without changing single-latent benchmarks.

        LIBERO keep the historical call signature exactly. The
        explicit keyword is used only by multi-latent clean-prefix recipes,
        currently RoboCasa365, so every pinned latent also receives t=0.
        """
        if self.video_clean_prefix_latent_steps == 1:
            return video_dit.pre_dit(latents, timestep, context, context_mask)
        return video_dit.pre_dit(
            latents,
            timestep,
            context,
            context_mask,
            clean_prefix_latent_steps=self.video_clean_prefix_latent_steps,
        )

    @staticmethod
    def _build_cached_action_attention_mask(
        *,
        full_video_seq_len: int,
        clean_video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_video_conditioning: str,
        clean_prefix_latent_steps: int,
    ) -> Tensor:
        """Project training action-query rows onto cached inference keys.

        Training contains clean and future video tokens, while cached policy
        inference stores only the clean video prefix. Action queries cannot see
        future video in ``clean_prefix`` mode, so removing those all-false key
        columns preserves the training attention graph exactly.
        """
        if clean_video_seq_len != clean_prefix_latent_steps * video_tokens_per_frame:
            raise ValueError(
                f"clean video cache does not match the configured latent prefix: {clean_video_seq_len} vs {clean_prefix_latent_steps * video_tokens_per_frame}"
            )
        if full_video_seq_len < clean_video_seq_len:
            raise ValueError(
                f"full video sequence cannot be shorter than its clean cache: {full_video_seq_len} < {clean_video_seq_len}"
            )
        training_mask = MoTWAM._build_mot_attention_mask(
            video_seq_len=full_video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            action_video_conditioning=action_video_conditioning,
            clean_prefix_latent_steps=clean_prefix_latent_steps,
        )
        action_rows = training_mask[full_video_seq_len:, :]
        return torch.cat(
            [action_rows[:, :clean_video_seq_len], action_rows[:, full_video_seq_len:]],
            dim=1,
        )

    def build_attention_mask(self, video_state: dict, action_state: dict) -> Tensor:
        """One mask policy for training, native inference, and batched evaluation."""
        return self._build_mot_attention_mask(
            video_seq_len=video_state["tokens"].shape[1],
            action_seq_len=action_state["tokens"].shape[1],
            video_tokens_per_frame=self._compute_video_tokens_per_frame(video_state),
            device=video_state["tokens"].device,
            action_video_conditioning=self.action_video_conditioning,
            clean_prefix_latent_steps=self.video_clean_prefix_latent_steps,
        )

    @staticmethod
    def _build_mot_attention_mask(
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_video_conditioning: str = "first_frame",
        clean_prefix_latent_steps: int = 1,
    ) -> Tensor:
        """Build the joint video/action mask used by every MoT layer.

        Video tokens are block-causal in latent time. Clean latent frames are
        observation-only; future latent frame ``i`` can attend to video
        frames ``0..i`` and to the action prefix aligned with that frame. The
        alignment is derived from runtime sequence lengths rather than a
        dataset-specific action/frame ratio, so uneven action chunks are
        distributed monotonically and the final future frame sees every action.

        Action queries read their configured clean observation prefix and
        the temporally aligned action block. Every future video query can
        attend only to its aligned action prefix.
        """
        if video_seq_len <= 0:
            raise ValueError(f"video_seq_len must be positive, got {video_seq_len}")
        if action_seq_len < 0:
            raise ValueError(
                f"action_seq_len must be non-negative, got {action_seq_len}"
            )
        if video_tokens_per_frame <= 0:
            raise ValueError(
                f"video_tokens_per_frame must be positive, got {video_tokens_per_frame}"
            )
        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                f"video_seq_len must contain complete latent frames: video_seq_len={video_seq_len}, video_tokens_per_frame={video_tokens_per_frame}"
            )
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )
        first_frame = video_tokens_per_frame
        num_video_frames = video_seq_len // video_tokens_per_frame
        clean_prefix_latent_steps = int(clean_prefix_latent_steps)
        if not 1 <= clean_prefix_latent_steps <= num_video_frames:
            raise ValueError(
                f"clean_prefix_latent_steps must be within the runtime video geometry: {clean_prefix_latent_steps} vs {num_video_frames}"
            )
        clean_prefix_tokens = clean_prefix_latent_steps * video_tokens_per_frame
        num_future_frames = num_video_frames - clean_prefix_latent_steps
        action_start = video_seq_len
        if action_video_conditioning not in {
            "first_frame",
            "clean_prefix",
            "full_video",
        }:
            raise ValueError(
                f"action_video_conditioning must be first_frame, clean_prefix, or full_video, got {action_video_conditioning!r}"
            )
        action_visible_video_tokens = (
            first_frame
            if action_video_conditioning == "first_frame"
            else clean_prefix_tokens
        )
        for frame_idx in range(clean_prefix_latent_steps):
            query_start = frame_idx * video_tokens_per_frame
            query_end = query_start + video_tokens_per_frame
            mask[query_start:query_end, :query_end] = True
        if num_future_frames == 0:
            mask[action_start:, action_start:] = True
            mask[action_start:, :action_visible_video_tokens] = True
            return mask
        previous_action_prefix = 0
        for future_index in range(1, num_future_frames + 1):
            frame_idx = clean_prefix_latent_steps + future_index - 1
            query_start = frame_idx * video_tokens_per_frame
            query_end = query_start + video_tokens_per_frame
            mask[query_start:query_end, :query_end] = True
            action_prefix = (
                future_index * action_seq_len + num_future_frames - 1
            ) // num_future_frames
            visible_action_tokens = action_prefix
            mask[
                query_start:query_end,
                action_start : action_start + visible_action_tokens,
            ] = True
            action_query = slice(
                action_start + previous_action_prefix, action_start + action_prefix
            )
            mask[action_query, action_start : action_start + action_prefix] = True
            if action_video_conditioning == "first_frame":
                mask[action_query, :first_frame] = True
            elif action_video_conditioning == "clean_prefix":
                mask[action_query, :clean_prefix_tokens] = True
            else:
                mask[action_query, :query_end] = True
            previous_action_prefix = action_prefix
        return mask
