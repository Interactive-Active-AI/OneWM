"""Token-action ActionDiT builders for OneWM."""

from __future__ import annotations

from typing import Any

from onewm.modules.action_dit import ActionDiT


_ACTION_EXPERT_BUILDERS = {}


def register_action_expert(name: str):
    def decorator(builder):
        _ACTION_EXPERT_BUILDERS[name] = builder
        return builder

    return decorator


def build_action_dit(backbone_info: Any, config: Any) -> ActionDiT:
    """Build the generic token-action DiT used by MoT backbones."""

    num_layers = config.action_expert_num_layers or backbone_info.num_layers
    return ActionDiT(
        hidden_dim=config.action_expert_hidden_dim,
        action_dim=config.action_dim,
        ffn_dim=config.action_expert_hidden_dim * 4,
        text_dim=backbone_info.text_dim,
        freq_dim=backbone_info.freq_dim,
        eps=backbone_info.eps,
        num_heads=backbone_info.num_heads,
        attn_head_dim=backbone_info.attn_head_dim,
        num_layers=num_layers,
        max_seq_len=config.chunk_size * 2,
        use_gradient_checkpointing=config.action_expert_use_gradient_checkpointing,
    )


register_action_expert("action_dit")(build_action_dit)
register_action_expert("token_action_dit")(build_action_dit)
register_action_expert("wan_action_dit")(build_action_dit)


def build_action_expert(backbone_info: Any, config: Any):
    action_expert_type = getattr(config, "action_expert_type", "wan_action_dit")
    try:
        builder = _ACTION_EXPERT_BUILDERS[action_expert_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown action_expert_type={action_expert_type!r}. "
            f"Available action experts: {sorted(_ACTION_EXPERT_BUILDERS)}"
        ) from exc
    return builder(backbone_info, config)


def load_action_dit_init(
    action_dit: ActionDiT,
    init_from: str | None,
    head_init: str = "random",
) -> None:
    """Load preprocessed Wan-to-ActionDiT initialization if configured."""

    if not init_from:
        return
    from onewm.utils.checkpoint import load_action_dit_backbone_init

    info = load_action_dit_backbone_init(action_dit, init_from, head_init=head_init)
    print(
        f"[OneWM] Loaded ActionDiT init from {init_from} "
        f"(num_loaded={info['num_loaded']}, "
        f"head_init={info['head_init']}, "
        f"missing={len(info['missing_keys'])}, "
        f"unexpected={len(info['unexpected_keys'])})"
    )
