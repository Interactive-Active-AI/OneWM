"""Mixture-of-Transformers (MoT) layer for OneWM.

MoT is backbone-agnostic at the algorithm level. It does not require Wan2.2
specifically; it requires each expert returned by a backbone/action module to
implement the MoT expert contract:

- ``expert.blocks``: ordered transformer blocks with the same layer count across
  experts.
- ``block.get_qkv(tokens, t_mod, freqs)``: run the block's pre-self-attention
  normalization/modulation and return Q/K/V for external joint attention.
- ``block.post_attention(tokens, attn_out, t_mod, context, context_mask)``:
  consume the mixed-attention output and finish the block's output projection,
  residual/gating, cross-attention, and FFN logic.

Backbone-specific code, such as Wan2.2, should adapt its
video DiT to this contract instead of changing the MoT orchestration below.

Based on Fast-WAM's MoT implementation. Orchestrates joint attention between
video expert and action expert by:
1. Building Q/K/V from each expert independently
2. Concatenating Q/K/V along sequence dimension
3. Running unified attention with structured mask
4. Splitting output back to each expert
5. Applying per-expert post-attention (gate + cross-attn + FFN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.utils.checkpoint import checkpoint


class MoT(nn.Module):
    """Mixture-of-Transformers: orchestrates joint attention across experts.

    Each expert provides DiT blocks with get_qkv() and post_attention() methods.
    The MoT concatenates Q/K/V from all experts per layer, runs joint attention,
    then routes the split output back to each expert's post-attention logic.
    """

    def __init__(
        self, experts: dict[str, nn.Module], checkpoint_mixed_attn: bool = True
    ):
        """
        Args:
            experts: {"video": WanVideoDiT, "action": ActionDiT} or similar.
                     Each expert must have .blocks (ModuleList of DiTBlock-like).
            checkpoint_mixed_attn: use gradient checkpointing on mixed attention.
        """
        super().__init__()
        self.experts = nn.ModuleDict(experts)
        self.expert_order = list(experts.keys())
        self.checkpoint_mixed_attn = checkpoint_mixed_attn

        # Validate all experts share num_layers
        num_layers_set = set()
        for name, expert in experts.items():
            num_layers_set.add(len(expert.blocks))
        assert len(num_layers_set) == 1, (
            f"All experts must have the same number of layers, got {num_layers_set}"
        )
        self.num_layers = num_layers_set.pop()

    def forward(
        self,
        expert_states: dict[str, dict],
        attention_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Joint forward pass: per-layer concat Q/K/V -> attention -> split -> post-block.

        Args:
            expert_states: {expert_name: {tokens, freqs, t_mod, context, context_mask}}
                - tokens: [B, S_i, D_i]
                - freqs: RoPE frequencies for this expert's sequence
                - t_mod: [B, 6*D_i] timestep modulation
                - context: [B, L, D_i] text context
                - context_mask: [B, L]
            attention_mask: Optional [S_total, S_total] joint attention mask
                where S_total = sum of all expert sequence lengths
        Returns:
            {expert_name: output_tokens [B, S_i, D_i]}
        """
        # Track current tokens per expert
        current_tokens = {
            name: state["tokens"] for name, state in expert_states.items()
        }

        for layer_idx in range(self.num_layers):
            # Build Q/K/V from each expert
            all_q, all_k, all_v = [], [], []
            seq_lens = []

            for name in self.expert_order:
                expert = self.experts[name]
                block = expert.blocks[layer_idx]
                tokens = current_tokens[name]
                t_mod = expert_states[name]["t_mod"]
                freqs = expert_states[name]["freqs"]

                q, k, v = block.get_qkv(tokens, t_mod, freqs)
                all_q.append(q)
                all_k.append(k)
                all_v.append(v)
                seq_lens.append(tokens.shape[1])

            # Concat along sequence dimension: [B, S_total, head_space_dim]
            q_cat = torch.cat(all_q, dim=1)
            k_cat = torch.cat(all_k, dim=1)
            v_cat = torch.cat(all_v, dim=1)

            # Mixed attention
            B = q_cat.shape[0]
            num_heads = self.experts[self.expert_order[0]].blocks[0].num_heads
            head_dim = self.experts[self.expert_order[0]].blocks[0].attn_head_dim
            S_total = q_cat.shape[1]

            q_cat = q_cat.view(B, S_total, num_heads, head_dim).transpose(1, 2)
            k_cat = k_cat.view(B, S_total, num_heads, head_dim).transpose(1, 2)
            v_cat = v_cat.view(B, S_total, num_heads, head_dim).transpose(1, 2)

            if self.checkpoint_mixed_attn and self.training:
                attn_out = checkpoint(
                    self._attention_fn,
                    q_cat,
                    k_cat,
                    v_cat,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                attn_out = self._attention_fn(q_cat, k_cat, v_cat, attention_mask)

            # attn_out: [B, H, S_total, D] -> [B, S_total, H*D]
            attn_out = attn_out.transpose(1, 2).reshape(
                B, S_total, num_heads * head_dim
            )

            # Split and apply post-attention per expert
            offset = 0
            for name, s_len in zip(self.expert_order, seq_lens):
                expert = self.experts[name]
                block = expert.blocks[layer_idx]
                expert_attn_out = attn_out[:, offset : offset + s_len, :]
                t_mod = expert_states[name]["t_mod"]
                ctx = expert_states[name]["context"]
                ctx_mask = expert_states[name]["context_mask"]

                current_tokens[name] = block.post_attention(
                    current_tokens[name], expert_attn_out, t_mod, ctx, ctx_mask
                )
                offset += s_len

        return current_tokens

    def prefill_video_cache(
        self,
        video_state: dict,
        attention_mask: Optional[Tensor] = None,
    ) -> list[dict[str, Tensor]]:
        """Process video tokens once, caching K/V per layer for action inference.

        Args:
            video_state: {tokens, freqs, t_mod, context, context_mask} for video expert
            attention_mask: optional [S_video, S_video] video self-attention mask
        Returns:
            kv_cache: list of dicts per layer, each with {'k': [B, S_v, H*D], 'v': [B, S_v, H*D]}
        """
        video_expert = self.experts["video"]
        tokens = video_state["tokens"]
        t_mod = video_state["t_mod"]
        freqs = video_state["freqs"]
        ctx = video_state["context"]
        ctx_mask = video_state["context_mask"]
        kv_cache = []

        for layer_idx in range(self.num_layers):
            block = video_expert.blocks[layer_idx]
            q, k, v = block.get_qkv(tokens, t_mod, freqs)
            kv_cache.append({"k": k, "v": v})

            # Self-attention for video (to update tokens for next layer)
            B, S, _ = q.shape
            num_heads = block.num_heads
            head_dim = block.attn_head_dim

            q_r = q.view(B, S, num_heads, head_dim).transpose(1, 2)
            k_r = k.view(B, S, num_heads, head_dim).transpose(1, 2)
            v_r = v.view(B, S, num_heads, head_dim).transpose(1, 2)

            attn_out = self._attention_fn(q_r, k_r, v_r, attention_mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, S, num_heads * head_dim)

            # Post-attention
            tokens = block.post_attention(tokens, attn_out, t_mod, ctx, ctx_mask)

        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_state: dict,
        video_kv_cache: list[dict[str, Tensor]],
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run action expert using cached video K/V.

        For each layer:
        1. Get action Q/K/V
        2. Concat video cached K/V with action K/V
        3. Action Q attends to [video_K | action_K], [video_V | action_V]
        4. Post-attention for action

        Args:
            action_state: {tokens, freqs, t_mod, context, context_mask} for action expert
            video_kv_cache: per-layer cached {k, v} from prefill_video_cache
            attention_mask: optional [S_action, S_video + S_action] mask
        Returns:
            output action tokens [B, S_action, D_action]
        """
        action_expert = self.experts["action"]
        tokens = action_state["tokens"]
        t_mod = action_state["t_mod"]
        freqs = action_state["freqs"]
        ctx = action_state["context"]
        ctx_mask = action_state["context_mask"]

        for layer_idx in range(self.num_layers):
            block = action_expert.blocks[layer_idx]
            q, k_action, v_action = block.get_qkv(tokens, t_mod, freqs)

            # Concat with cached video K/V
            k_video = video_kv_cache[layer_idx]["k"]
            v_video = video_kv_cache[layer_idx]["v"]
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)

            # Attention: action queries attend to [video | action] keys/values
            B, S_a, _ = q.shape
            S_kv = k_cat.shape[1]
            num_heads = block.num_heads
            head_dim = block.attn_head_dim

            q_r = q.view(B, S_a, num_heads, head_dim).transpose(1, 2)
            k_r = k_cat.view(B, S_kv, num_heads, head_dim).transpose(1, 2)
            v_r = v_cat.view(B, S_kv, num_heads, head_dim).transpose(1, 2)

            attn_out = self._attention_fn(q_r, k_r, v_r, attention_mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, S_a, num_heads * head_dim)

            # Post-attention
            tokens = block.post_attention(tokens, attn_out, t_mod, ctx, ctx_mask)

        return tokens

    @staticmethod
    def _attention_fn(
        q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Scaled dot-product attention. q/k/v shape: [B, H, S, D]."""
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
