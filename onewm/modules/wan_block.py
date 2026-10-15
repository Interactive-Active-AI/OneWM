"""Wan2.2-compatible DiT block, ported to expose `get_qkv` / `post_attention`
methods required by OneWM's `MoT`.

The state_dict keys produced here match upstream Wan2.2 exactly, so this block
can directly load weights from
`checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-*.safetensors`.

Layout (per block):
    blocks.{i}.self_attn.{q,k,v,o}.{weight,bias}
    blocks.{i}.self_attn.{norm_q,norm_k}.weight       # RMSNorm
    blocks.{i}.cross_attn.{q,k,v,o}.{weight,bias}
    blocks.{i}.cross_attn.{norm_q,norm_k}.weight
    blocks.{i}.norm1.{weight,bias}                    # LayerNorm (no affine in Wan2.2; weight only when affine)
    blocks.{i}.norm2.{weight,bias}                    # LayerNorm (no affine)
    blocks.{i}.norm3.{weight,bias}                    # LayerNorm (with affine)
    blocks.{i}.ffn.0.{weight,bias}                    # Linear hidden -> ffn
    blocks.{i}.ffn.2.{weight,bias}                    # Linear ffn -> hidden
    blocks.{i}.modulation                             # [1, 6, hidden_dim]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ----------------------------- helpers -----------------------------------


class RMSNorm(nn.Module):
    """RMSNorm matching Wan2.2's implementation (param key: `weight`)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x_f.to(dtype) * self.weight


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_1d(
    dim: int, end: int = 1024, theta: float = 10000.0
) -> torch.Tensor:
    """1D RoPE for action sequences. Returns complex64 tensor [end, dim//2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(
    dim: int, end: int = 1024, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3D RoPE for (frame, height, width) — matches Wan2.2's split."""
    f_dim = dim - 2 * (dim // 3)
    h_dim = dim // 3
    w_dim = dim // 3
    return (
        precompute_freqs_cis_1d(f_dim, end, theta),
        precompute_freqs_cis_1d(h_dim, end, theta),
        precompute_freqs_cis_1d(w_dim, end, theta),
    )


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply RoPE to x of shape [B, S, num_heads*head_dim] using complex `freqs`.

    `freqs` should already be aligned with the sequence (shape [S, ..., head_dim/2])
    and able to broadcast against [B, S, num_heads, head_dim/2, 2] -> complex.
    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_c * freqs).flatten(2)  # [B, S, num_heads*head_dim]
    return out.to(x.dtype)


# ----------------------------- attention modules ------------------------


class SelfAttention(nn.Module):
    """Self-attention with separate q/k/v Linears and RMSNorm on q/k.

    State_dict keys: q.{weight,bias}, k.{weight,bias}, v.{weight,bias},
    o.{weight,bias}, norm_q.weight, norm_k.weight.
    """

    def __init__(
        self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def project_qkv(
        self, x: torch.Tensor, freqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute q/k/v with norm + RoPE. Used by MoT joint attention."""
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        return q, k, v


class CrossAttention(nn.Module):
    """Cross-attention to text context. Separate q/k/v Linears + RMSNorm on q/k."""

    def __init__(
        self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        # SDPA path (no RoPE on cross-attention).
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        out = rearrange(out, "b n s d -> b s (n d)")
        return self.o(out)


class GateModule(nn.Module):
    def forward(
        self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return x + gate * residual


# ----------------------------- DiT block ---------------------------------


class DiTBlock(nn.Module):
    """Wan2.2-compatible DiT block with MoT-friendly `get_qkv` / `post_attention`.

    Forward order (matches Wan2.2):
        1. AdaLN(norm1) -> self_attn -> gate_msa -> residual
        2. + cross_attn(norm3(x), context)
        3. AdaLN(norm2) -> ffn -> gate_mlp -> residual
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        # Wan2.2 uses LayerNorm without affine for norm1/norm2 (AdaLN provides scale/shift)
        # and LayerNorm with affine for norm3 (between self-attn and cross-attn).
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    # ----- AdaLN modulation split (matches Wan2.2 `_split_modulation`) ----
    def _split_modulation(self, t_mod: torch.Tensor):
        """Combine per-block `modulation` with global `t_mod` and chunk into 6.

        t_mod: [B, 6, D] (block-wide modulation) OR [B, S, 6, D] (per-token).
        Returns 6 tensors each with shape [B, D] or [B, S, D].
        """
        has_seq = t_mod.dim() == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        out = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            out = tuple(t.squeeze(2) for t in out)
        return out

    # ----- MoT-facing API ------------------------------------------------
    def get_qkv(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-attention: returns Q/K/V (with norm + RoPE) for joint attention.

        Note: this stashes the AdaLN parameters and pre-attention residual on
        the block so `post_attention` can reuse them in a single forward pass.
        Caller MUST invoke `post_attention(x, attn_out, t_mod, ...)` with the
        SAME `x` immediately after.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(t_mod)
        )
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.self_attn.project_qkv(attn_input, freqs)
        # Cache for post_attention. Token-id keying ensures correctness if a
        # block is reused for multiple expert tokens in the same MoT layer.
        self._cache = {
            "x_id": id(x),
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return q, k, v

    def post_attention(
        self,
        x: torch.Tensor,
        attn_output: torch.Tensor,
        t_mod: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply o-projection, gates, cross-attention and FFN."""
        cache = getattr(self, "_cache", None)
        if cache is not None and cache.get("x_id") == id(x):
            gate_msa = cache["gate_msa"]
            shift_mlp = cache["shift_mlp"]
            scale_mlp = cache["scale_mlp"]
            gate_mlp = cache["gate_mlp"]
        else:
            # Recompute from scratch (standalone post_attention call).
            _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(
                t_mod
            )

        # Self-attention residual with output projection + gate.
        x = self.gate(x, gate_msa, self.self_attn.o(attn_output))
        # Cross-attention.
        if context_mask is not None and context_mask.dim() == 3:
            # Expand to broadcast over heads in SDPA path.
            context_mask = context_mask.unsqueeze(1)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        # FFN.
        x = self.gate(
            x, gate_mlp, self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        return x

    # ----- Optional standalone forward (used outside MoT, e.g. tests) ---
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.get_qkv(x, t_mod, freqs)
        # SDPA flow.
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self_attn_mask)
        out = rearrange(out, "b n s d -> b s (n d)")
        return self.post_attention(x, out, t_mod, context, context_mask)
