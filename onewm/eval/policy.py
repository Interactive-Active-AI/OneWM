"""Benchmark-agnostic OneWM policy wrapper for closed-loop evaluation.

This module is the shared core used by benchmark deployment adapters:

    BenchmarkAdapter (per-benchmark)  ->  OneWMPolicy (this file)  ->  model

An adapter is responsible for turning raw environment observations into
``(image_tensor, state_vector, instruction)`` and for turning predicted action
chunks back into environment actions. ``OneWMPolicy`` owns everything generic:
recipe/checkpoint loading, text conditioning, (de)normalization, flow-matching
inference, and the replan action queue.
"""

from __future__ import annotations

import json
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from onewm.config import load_config
from onewm.data.lerobot import (
    DEFAULT_TEXT_CACHE_ENCODER_ID,
    DEFAULT_TEXT_PROMPT,
    format_text_prompt,
    load_lerobot_stats,
    load_text_cache,
    save_text_cache,
    text_cache_path,
)


# --------------------------------------------------------------------------- #
# Generic checkpoint loading used by rollout and offline WM inference.
# --------------------------------------------------------------------------- #
def _strip_known_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.", "_orig_mod.")
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out


def _extract_checkpoint_state(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state_dict", "module", "state_dict"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    return payload


def load_checkpoint_into(
    model: torch.nn.Module, checkpoint: str | Path, *, assign: bool = False
) -> None:
    """Strictly load a model file or training checkpoint directory."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        for name in (
            "model.pt",
            "pytorch_model.bin",
            "model.bin",
            "mp_rank_00_model_states.pt",
            "pytorch_model/mp_rank_00_model_states.pt",
        ):
            path = checkpoint / name
            if path.is_file():
                checkpoint = path
                break
        else:
            safetensors_files = sorted(checkpoint.glob("*.safetensors"))
            if not safetensors_files:
                raise FileNotFoundError(
                    f"Unsupported checkpoint directory {checkpoint}: expected model.pt, "
                    "pytorch_model.bin, mp_rank_00_model_states.pt, or *.safetensors."
                )
            from safetensors.torch import load_file

            state: dict[str, Any] = {}
            location = str(next(model.parameters()).device) if assign else "cpu"
            for path in safetensors_files:
                state.update(load_file(str(path), device=location))
            model.load_state_dict(
                _strip_known_prefixes(state), strict=True, assign=assign
            )
            return

    location = next(model.parameters()).device if assign else "cpu"
    payload = torch.load(
        checkpoint, map_location=location, mmap=True, weights_only=False
    )
    state = _strip_known_prefixes(_extract_checkpoint_state(payload))
    model.load_state_dict(state, strict=True, assign=assign)


def validate_checkpoint_metadata(
    checkpoint: str | Path, expected: dict[str, Any]
) -> None:
    """Reject cross-benchmark checkpoints before non-strict weight loading."""
    if not expected:
        return
    path = Path(checkpoint)
    actual: dict[str, Any] = {}
    if path.is_dir() and (path / "trainer_state.json").is_file():
        payload = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        actual = dict(payload.get("checkpoint_metadata", {}) or {})
    elif path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            actual = dict(payload.get("checkpoint_metadata", {}) or {})
    if not actual:
        raise ValueError(f"checkpoint is missing required benchmark metadata: {path}")
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint benchmark contract mismatch: {mismatches}")


# --------------------------------------------------------------------------- #
# Normalization (generic; no benchmark-specific gripper post-processing).
# --------------------------------------------------------------------------- #
def _normalize_vector(
    vec: torch.Tensor, mode: str, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    dim = int(vec.shape[-1])
    if mode == "q01q99":
        low, high = stats["q01"][:dim].to(vec), stats["q99"][:dim].to(vec)
        return 2.0 * (vec - low) / (high - low).clamp_min(1e-6) - 1.0
    if mode == "zscore":
        mean = stats["mean"][:dim].to(vec.dtype)
        std = stats["std"][:dim].to(vec.dtype).clamp_min(1e-6)
        return ((vec - mean) / std).clamp(-5.0, 5.0)
    if mode == "minmax":
        v_min = stats["min"][:dim].to(vec.dtype)
        v_max = stats["max"][:dim].to(vec.dtype)
        return (2.0 * (vec - v_min) / (v_max - v_min).clamp_min(1e-6) - 1.0).clamp(
            -5.0, 5.0
        )
    raise ValueError(f"Unsupported norm mode: {mode!r}")


def _denormalize_action(
    action: torch.Tensor, mode: str, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    stats = {key: value.to(action) for key, value in stats.items()}
    if mode == "q01q99":
        low, high = stats["q01"], stats["q99"]
        return (action + 1.0) * 0.5 * (high - low).clamp_min(1e-6) + low
    if mode == "zscore":
        return action * stats["std"].clamp_min(1e-6) + stats["mean"]
    if mode == "minmax":
        a_min = stats["min"]
        a_max = stats["max"]
        return (action.clamp(-1.0, 1.0) + 1.0) * 0.5 * (a_max - a_min).clamp_min(
            1e-6
        ) + a_min
    raise ValueError(f"Unsupported norm mode: {mode!r}")


class OneWMPolicy:
    """Loads a OneWM recipe/checkpoint and runs closed-loop action inference.

    The policy is benchmark-neutral: callers pass a preprocessed image tensor
    (already in the model's expected layout and ``[-1, 1]`` range), a raw state
    vector, and a language instruction. It returns denormalized action chunks
    and manages an internal replan queue.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint: str,
        *,
        overrides: Optional[list[str]] = None,
        device: str = "cuda:0",
        num_inference_steps: Optional[int] = None,
        replan_steps: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        from onewm import build_framework
        from onewm.utils.config_cli import apply_overrides

        config = load_config(config_path)
        if overrides:
            config = apply_overrides(config, overrides)
        config.framework.action_expert_init_from = None
        self.config = config

        self.device = torch.device(
            device
            if torch.cuda.is_available() or not device.startswith("cuda")
            else "cpu"
        )
        mp = (config.training.mixed_precision or "no").lower()
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            mp, torch.float32
        )
        if self.device.type == "cpu":
            self.dtype = torch.float32

        self.model = build_framework(
            config, device=str(self.device), dtype=self.dtype
        ).to(self.device)
        validate_checkpoint_metadata(
            checkpoint,
            dict(getattr(config.training, "checkpoint_metadata", {}) or {}),
        )
        load_checkpoint_into(
            self.model, checkpoint, assign=config.data.dataset_type == "franka_v3"
        )
        self.model.eval()

        self.action_horizon = int(config.framework.chunk_size)
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else getattr(config.inference, "num_inference_steps", 8)
        )
        self.replan_steps = int(max(1, min(replan_steps, self.action_horizon)))
        self.seed = seed

        self._action_stats = self._load_stats("action")
        self._state_stats = self._load_stats("state")
        self._text_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )
        self._text_cache_max = 256
        self._text_encoder = None
        self.pending_actions: deque[np.ndarray] = deque()

    # -- stats -------------------------------------------------------------- #
    def _load_stats(self, key: str) -> Optional[dict[str, torch.Tensor]]:
        flag = "normalize_actions" if key == "action" else "normalize_states"
        if not getattr(self.config.data, flag, False):
            return None
        path_attr = "action_stats_path" if key == "action" else "state_stats_path"
        stats_path = getattr(self.config.data, path_attr, None) or getattr(
            self.config.data, "action_stats_path", None
        )
        if not stats_path:
            raise ValueError(f"{flag}=true requires data.{path_attr}")
        stats = load_lerobot_stats(stats_path)
        if key not in stats:
            raise KeyError(f"No {key} stats found in {stats_path}")
        return stats[key]

    # -- text conditioning -------------------------------------------------- #
    def _encode_context(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        if instruction in self._text_cache:
            ctx, mask = self._text_cache[instruction]
            self._text_cache.move_to_end(instruction)
            return ctx.to(self.device, self.dtype), mask.to(self.device)

        text_len = int(getattr(self.config.data, "text_len", 128))
        prompt_template = (
            getattr(self.config.data, "text_prompt_template", None)
            or DEFAULT_TEXT_PROMPT
        )
        encoder_id = (
            getattr(self.config.data, "text_cache_encoder_id", None)
            or DEFAULT_TEXT_CACHE_ENCODER_ID
        )
        cache_dir = getattr(self.config.data, "text_embedding_cache_dir", None)
        text_dim = int(
            getattr(
                getattr(getattr(self.model, "backbone", None), "info", None),
                "text_dim",
                4096,
            )
        )

        cache = (
            text_cache_path(
                cache_dir, instruction, text_len, prompt_template, encoder_id
            )
            if cache_dir
            else None
        )
        if cache is not None and cache.is_file():
            ctx, mask = load_text_cache(cache, text_len, text_dim)
            ctx = ctx.unsqueeze(0)
            mask = mask.unsqueeze(0)
        else:
            if self._text_encoder is None:
                from onewm.backbone import build_text_encoder

                self._text_encoder = build_text_encoder(
                    self.config.backbone,
                    text_len=text_len,
                    device=str(self.device),
                    dtype=torch.bfloat16
                    if self.device.type == "cuda"
                    else torch.float32,
                )
            prompt = format_text_prompt(instruction, prompt_template)
            with torch.no_grad():
                ctx, mask = self._text_encoder.encode([prompt])
            if cache is not None:
                save_text_cache(cache, ctx[0], mask[0], prompt, instruction)

        self._text_cache[instruction] = (ctx.cpu(), mask.cpu())
        if len(self._text_cache) > int(getattr(self, "_text_cache_max", 256)):
            self._text_cache.popitem(last=False)
        return ctx.to(self.device, self.dtype), mask.to(self.device)

    def _encode_contexts(
        self, instructions: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [
            self._encode_context(str(instruction)) for instruction in instructions
        ]
        if not encoded:
            raise ValueError("instructions must contain at least one item")
        return (
            torch.cat([context for context, _ in encoded], dim=0),
            torch.cat([mask for _, mask in encoded], dim=0),
        )

    # -- inference ---------------------------------------------------------- #
    def _sampled_video_frame_count(self) -> int:
        num = int(self.config.data.num_frames)
        step = max(1, int(getattr(self.config.data, "action_freq_ratio", 1)))
        return len(range(0, num, step))

    @torch.no_grad()
    def predict_chunk(
        self, image: torch.Tensor, state: Optional[np.ndarray], instruction: str
    ) -> np.ndarray:
        """Return a denormalized action chunk ``[T, action_dim]`` (numpy)."""
        return self.predict_chunks(
            image,
            [state],
            [instruction],
        )[0]

    @torch.no_grad()
    def predict_chunks(
        self,
        images: torch.Tensor,
        states: Sequence[Optional[np.ndarray]],
        instructions: Sequence[str],
    ) -> np.ndarray:
        """Return denormalized action chunks ``[B,T,action_dim]``.

        Batched clients can advance several simulation environments together. Keeping
        that leading batch dimension through the VAE, video cache, and action
        denoiser avoids one independent policy forward per environment.
        """
        if images.ndim not in {4, 5}:
            raise ValueError(
                "images must be [B,C,H,W] or clean video [B,C,T,H,W], got "
                f"{tuple(images.shape)}"
            )
        batch_size = int(images.shape[0])
        if batch_size < 1:
            raise ValueError("images batch must not be empty")
        if len(states) != batch_size or len(instructions) != batch_size:
            raise ValueError(
                "images/states/instructions batch mismatch: "
                f"{batch_size}, {len(states)}, {len(instructions)}"
            )

        image = images.to(device=self.device, dtype=self.dtype)
        context, context_mask = self._encode_contexts(instructions)

        proprio = None
        state_conditioning = getattr(self.config.framework, "state_conditioning", None)
        requires_state = (
            state_conditioning != "none"
            if state_conditioning is not None
            else bool(getattr(self.config.framework, "proprio_dim", None))
        )
        if requires_state:
            if any(state is None for state in states):
                raise ValueError(
                    "all states are required when state conditioning is enabled"
                )
            proprio_np = np.stack(
                [np.asarray(state, dtype=np.float32) for state in states],
                axis=0,
            )
            configured_tokens = getattr(self.config.framework, "state_num_tokens", None)
            token_dim = int(getattr(self.config.framework, "state_token_dim", 14))
            if configured_tokens is not None:
                expected = (int(configured_tokens), token_dim)
                if tuple(proprio_np.shape[1:]) != expected:
                    raise ValueError(
                        f"history state must have per-sample shape {expected}, got "
                        f"{tuple(proprio_np.shape[1:])}"
                    )
            else:
                proprio_dim = int(self.config.framework.proprio_dim)
                if tuple(proprio_np.shape[1:]) != (proprio_dim,):
                    raise ValueError(
                        f"legacy state must be [{proprio_dim}], got {tuple(proprio_np.shape[1:])}"
                    )
            proprio_t = torch.as_tensor(proprio_np)
            if self._state_stats is not None:
                proprio_t = _normalize_vector(
                    proprio_t,
                    self.config.data.state_norm_mode,
                    self._state_stats,
                )
            proprio = proprio_t.to(device=self.device, dtype=self.dtype)

        pred = self.model.infer_action(
            input_image=image,
            context=context,
            context_mask=context_mask,
            action_horizon=self.action_horizon,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            proprio=proprio,
            num_video_frames=self._sampled_video_frame_count(),
        )

        action = pred.detach().float().cpu()
        if self._action_stats is not None:
            action = _denormalize_action(
                action,
                self.config.data.action_norm_mode,
                self._action_stats,
            )
        out = action.numpy()
        if out.ndim != 3 or out.shape[0] != batch_size:
            raise ValueError(
                "model must return [B,T,action_dim], got "
                f"{tuple(out.shape)} for batch_size={batch_size}"
            )
        return out.astype(np.float32)

    @torch.no_grad()
    def predict_world(
        self,
        image: torch.Tensor,
        instruction: str,
        known_action: np.ndarray,
        *,
        num_video_frames: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """State-free action-conditioned future prediction."""
        image = image.to(device=self.device, dtype=self.dtype)
        context, context_mask = self._encode_context(instruction)
        action = torch.as_tensor(np.asarray(known_action), dtype=torch.float32)
        if action.ndim != 2 or action.shape[-1] != int(
            self.config.framework.action_dim
        ):
            raise ValueError(
                "known_action must be [T, action_dim], got "
                f"{tuple(action.shape)} for action_dim={self.config.framework.action_dim}"
            )
        if self._action_stats is not None:
            action = _normalize_vector(
                action, self.config.data.action_norm_mode, self._action_stats
            )
        action = action.to(device=self.device, dtype=self.dtype)
        horizon = int(action.shape[0])
        result = self.model.infer_world(
            input_image=image,
            context=context,
            context_mask=context_mask,
            num_video_frames=(
                self._sampled_video_frame_count()
                if num_video_frames is None
                else int(num_video_frames)
            ),
            action_horizon=horizon,
            known_action=action,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed if seed is None else seed,
        )
        generated_action = result["action"].detach().float().cpu()
        if self._action_stats is not None:
            generated_action = _denormalize_action(
                generated_action,
                self.config.data.action_norm_mode,
                self._action_stats,
            )
        return {
            "video": result["video"].detach().float().cpu(),
            "action": generated_action,
        }

    # -- replan queue ------------------------------------------------------- #
    def needs_observation(self) -> bool:
        return not self.pending_actions

    def fill_queue(
        self, image: torch.Tensor, state: Optional[np.ndarray], instruction: str
    ) -> None:
        chunk = self.predict_chunk(image, state, instruction)
        for i in range(min(self.replan_steps, chunk.shape[0])):
            self.pending_actions.append(np.asarray(chunk[i], dtype=np.float32))

    def pop_action(self) -> np.ndarray:
        return self.pending_actions.popleft()

    def reset(self) -> None:
        self.pending_actions.clear()
