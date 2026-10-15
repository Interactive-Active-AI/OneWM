"""Preprocess OneWM ActionDiT initialization from Wan2.2 DiT weights.

The output payload is consumed by ``framework.action_expert_init_from``.
Wan-compatible weights are copied or interpolated into ActionDiT. Use
``--head-init zero`` or ``--head-init payload`` to make output-head
initialization explicit and pair it with ``framework.action_expert_head_init``.

Examples:
    python -m onewm.tools.preprocess_action_dit_init \
        --config configs/libero.yaml \
        --pretrained-model-id checkpoints/Wan2.2-TI2V-5B \
        --output checkpoints/action_dit_init.pt \
        --device cuda --dtype bfloat16

    python -m onewm.tools.preprocess_action_dit_init \
        --config configs/libero.yaml \
        --video-state-dict checkpoints/wan22_dit_state_dict.pt \
        --output checkpoints/action_dit_init.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == target_shape:
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce rank: src={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != target_shape:
        raise ValueError(
            f"Resize produced wrong shape: src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _convert_tensor(
    src: torch.Tensor,
    target: torch.Tensor,
    apply_alpha_scaling: bool,
) -> tuple[torch.Tensor, bool]:
    target_shape = tuple(target.shape)
    if tuple(src.shape) == target_shape:
        value = src
        resized = False
    else:
        value = _resize_to_shape(src, target_shape)
        if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target_shape[-1]:
            alpha = (float(src.shape[-1]) / float(target_shape[-1])) ** 0.5
            value = value.to(torch.float32) * alpha
        resized = True
    return value.detach().to(dtype=target.dtype, device="cpu").contiguous(), resized


def _load_video_state_from_file(path: str) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if (
        isinstance(raw, dict)
        and "state_dict" in raw
        and isinstance(raw["state_dict"], dict)
    ):
        return raw["state_dict"]
    if (
        isinstance(raw, dict)
        and "model_state_dict" in raw
        and isinstance(raw["model_state_dict"], dict)
    ):
        return raw["model_state_dict"]
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unsupported source format: {type(raw)}")


def _build_action_dit_from_config(
    config_path: str,
    device: str,
    dtype: torch.dtype,
    pretrained_model_id: str | None = None,
):
    from onewm.action_model import build_action_dit
    from onewm.config import load_config
    from onewm.utils.checkpoint import infer_backbone_info

    cfg = load_config(config_path)
    if pretrained_model_id:
        cfg.backbone.pretrained_model_id = pretrained_model_id
    backbone_info = infer_backbone_info(cfg.backbone.pretrained_model_id)
    action_dit = build_action_dit(backbone_info, cfg.framework).to(
        device=device, dtype=dtype
    )
    return cfg, backbone_info, action_dit


def _load_wan22_video_state(
    config_path: str,
    dtype: torch.dtype,
    pretrained_model_id: str | None = None,
) -> dict[str, torch.Tensor]:
    from onewm.config import load_config
    from onewm.backbone.wan22 import Wan22Dit
    from onewm.utils.checkpoint import infer_backbone_info

    cfg = load_config(config_path)
    if pretrained_model_id:
        cfg.backbone.pretrained_model_id = pretrained_model_id
    info = infer_backbone_info(cfg.backbone.pretrained_model_id)
    dit = Wan22Dit(info)
    dit.load_pretrained(cfg.backbone.pretrained_model_id, dtype=dtype)
    return dit.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="OneWM YAML recipe used to infer ActionDiT shape.",
    )
    parser.add_argument("--output", required=True, help="Output .pt payload path.")
    parser.add_argument(
        "--video-state-dict",
        default=None,
        help="Optional torch state_dict for the source video DiT. If omitted, load from recipe backbone.",
    )
    parser.add_argument(
        "--pretrained-model-id",
        default=None,
        help="Local source backbone checkpoint directory. Overrides backbone.pretrained_model_id from the recipe.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument(
        "--head-init", default="random", choices=["random", "zero", "payload"]
    )
    parser.add_argument(
        "--no-alpha-scaling", dest="apply_alpha_scaling", action="store_false"
    )
    parser.set_defaults(apply_alpha_scaling=True)
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)
    cfg, backbone_info, action_dit = _build_action_dit_from_config(
        args.config,
        args.device,
        dtype,
        pretrained_model_id=args.pretrained_model_id,
    )
    action_state = action_dit.state_dict()

    if args.video_state_dict:
        print(
            f"[INFO] Loading source video DiT state_dict from {args.video_state_dict}"
        )
        video_state = _load_video_state_from_file(args.video_state_dict)
    else:
        print(
            f"[INFO] Loading source Wan2.2 DiT from {cfg.backbone.pretrained_model_id}"
        )
        video_state = _load_wan22_video_state(
            args.config, dtype, pretrained_model_id=args.pretrained_model_id
        )

    backbone_keys = action_dit.backbone_key_set(action_state.keys())
    backbone_state = {}
    copied = 0
    interpolated = 0
    missing: list[str] = []
    for key in sorted(backbone_keys):
        if key not in video_state:
            missing.append(key)
            continue
        value, resized = _convert_tensor(
            video_state[key], action_state[key], args.apply_alpha_scaling
        )
        backbone_state[key] = value
        if resized:
            interpolated += 1
        else:
            copied += 1

    if missing:
        raise ValueError(
            f"{len(missing)} keys missing in source video DiT state_dict: {missing[:10]}"
        )

    payload: dict[str, Any] = {
        "policy": {
            "alpha_scaling": bool(args.apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "action_backbone_skip_prefixes": list(
                action_dit.ACTION_BACKBONE_SKIP_PREFIXES
            ),
            "head_init": args.head_init,
            "source": "video_state_dict"
            if args.video_state_dict
            else "onewm_recipe_backbone",
            "source_backbone": "wan22",
        },
        "backbone_state_dict": backbone_state,
        "meta": {
            "hidden_dim": int(cfg.framework.action_expert_hidden_dim),
            "ffn_dim": int(cfg.framework.action_expert_hidden_dim) * 4,
            "num_layers": int(
                cfg.framework.action_expert_num_layers or backbone_info.num_layers
            ),
            "num_heads": int(backbone_info.num_heads),
            "attn_head_dim": int(backbone_info.attn_head_dim),
            "text_dim": int(backbone_info.text_dim),
            "freq_dim": int(backbone_info.freq_dim),
            "eps": float(backbone_info.eps),
            "action_dim": int(cfg.framework.action_dim),
        },
    }

    if args.head_init == "zero":
        payload["head_state_dict"] = {
            "head.weight": torch.zeros_like(action_dit.head.weight, device="cpu"),
            "head.bias": torch.zeros_like(action_dit.head.bias, device="cpu"),
        }
    elif args.head_init == "payload":
        payload["head_state_dict"] = {
            "head.weight": action_dit.head.weight.detach()
            .to(device="cpu")
            .contiguous(),
            "head.bias": action_dit.head.bias.detach().to(device="cpu").contiguous(),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    print(
        f"[INFO] Saved OneWM ActionDiT init to {out_path} "
        f"(source_backbone=wan22, copied={copied}, interpolated={interpolated}, "
        f"total={len(backbone_state)}, head_init={args.head_init})."
    )
    print(f"[INFO] Set framework.action_expert_init_from: {out_path}")
    print(f"[INFO] Set framework.action_expert_head_init: {args.head_init}")


if __name__ == "__main__":
    main()
