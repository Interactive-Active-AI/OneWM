"""ActionDiT: lightweight action diffusion transformer for the MoT framework.

Reuses Wan2.2-compatible :class:`onewm.modules.wan_block.DiTBlock`, so its
``blocks.*`` state_dict layout matches the video DiT exactly. This is what
allows ``python -m onewm.tools.preprocess_action_dit_init`` to copy or interpolate
the video DiT weights into the ActionDiT initialisation directly.

Top-level state_dict keys::

    action_encoder.{weight,bias}                # Linear(action_dim -> hidden)
    text_embedding.0.{weight,bias}              # text_dim -> hidden
    text_embedding.2.{weight,bias}              # hidden -> hidden
    time_embedding.0.{weight,bias}              # freq_dim -> hidden
    time_embedding.2.{weight,bias}              # hidden -> hidden
    time_projection.1.{weight,bias}             # hidden -> 6*hidden
    blocks.{i}.<see wan_block.DiTBlock>
    head.{weight,bias}                          # Linear(hidden -> action_dim)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from onewm.modules.wan_block import (
    DiTBlock,
    precompute_freqs_cis_1d,
    sinusoidal_embedding_1d,
)


class ActionDiT(nn.Module):
    """Lightweight action diffusion transformer.

    The MoT layer operates between :meth:`pre_dit` and :meth:`post_dit`. For
    standalone usage (e.g. IDM smoke tests) call :meth:`forward`.
    """

    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(
                key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES
            )
        }

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        max_seq_len: int = 256,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.eps = eps
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Action input projection (1D linear, not Conv3d like the video DiT).
        self.action_encoder = nn.Linear(action_dim, hidden_dim, bias=True)

        # Text context projection — same MLP as Wan22Dit.text_embedding.
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Timestep embedding & per-block modulation projection (matches Wan2.2).
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

        # Transformer blocks — Wan2.2-compatible layout.
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Output head: hidden -> action_dim. Simple Linear (no AdaLN).
        self.head = nn.Linear(hidden_dim, action_dim, bias=True)

        # 1D RoPE table is complex64; rebuilt lazily per-device because
        # ``nn.Module.to(real_dtype)`` would discard the imaginary part of
        # registered complex buffers.
        self._max_seq_len = max_seq_len
        self._freqs_cache: dict = {}

    def _get_freqs(self, device: torch.device) -> Tensor:
        key = str(device)
        cached = self._freqs_cache.get(key)
        if cached is None:
            f = precompute_freqs_cis_1d(self.attn_head_dim, end=self._max_seq_len)
            # [T, head_dim/2] -> [T, 1, head_dim/2] for broadcasting.
            cached = f.unsqueeze(1).to(device)
            self._freqs_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # MoT-facing API: pre_dit / post_dit
    # ------------------------------------------------------------------
    def pre_dit(
        self,
        action_tokens: Tensor,
        timestep: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
    ) -> dict:
        """Encode inputs for MoT composition.

        Args:
            action_tokens: ``[B, T, action_dim]`` noisy action sequence.
            timestep: ``[B]`` sequence-level or ``[B, T]`` token-level
                timestep indices.
            context: ``[B, L, text_dim]`` text embeddings.
            context_mask: ``[B, L]`` boolean mask.

        Returns:
            dict with ``tokens``, ``freqs``, ``t_mod``, ``context``,
            ``context_mask``.
        """
        action_param = self.action_encoder.weight
        action_tokens = action_tokens.to(
            device=action_param.device, dtype=action_param.dtype
        )
        text_param = self.text_embedding[0].weight
        context = context.to(device=text_param.device, dtype=text_param.dtype)

        B, T, _ = action_tokens.shape
        if context_mask is not None:
            context_mask = context_mask.to(device=action_tokens.device)

        # Encode actions to hidden dim.
        tokens = self.action_encoder(action_tokens)  # [B, T, D]

        if timestep.ndim == 1:
            if timestep.shape[0] != B:
                raise ValueError(
                    f"sequence timestep must be [B], got {tuple(timestep.shape)}"
                )
            timestep_shape = (B,)
        elif timestep.ndim == 2:
            if tuple(timestep.shape) != (B, T):
                raise ValueError(
                    f"token timestep must be [B, T]={(B, T)}, got {tuple(timestep.shape)}"
                )
            timestep_shape = (B, T)
        else:
            raise ValueError(
                f"timestep must be [B] or [B, T], got rank {timestep.ndim}"
            )

        # Timestep -> block-wide [B, 6, D] or per-token [B, T, 6, D]
        # modulation. DiTBlock supports both layouts.
        t_input = timestep.reshape(-1).to(
            action_tokens.dtype if action_tokens.is_floating_point() else torch.float32
        )
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t_input)
        t = self.time_embedding(t_emb).reshape(*timestep_shape, self.hidden_dim)
        t_mod = self.time_projection(t).reshape(*timestep_shape, 6, self.hidden_dim)

        # Text projection.
        ctx = self.text_embedding(context)
        if context_mask is not None and context_mask.dim() == 2:
            # Expand to [B, S, L] for SDPA cross-attention.
            context_mask = context_mask.unsqueeze(1).expand(B, T, -1)

        freqs = self._get_freqs(action_tokens.device)[:T]  # [T, 1, head_dim/2] complex

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "context": ctx,
            "context_mask": context_mask,
        }

    def post_dit(self, tokens: Tensor) -> Tensor:
        """Project hidden states back to action space."""
        return self.head(tokens)

    # ------------------------------------------------------------------
    # Standalone forward (used outside MoT, e.g. IDM tests)
    # ------------------------------------------------------------------
    def forward(
        self,
        action_tokens: Tensor,
        timestep: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        state = self.pre_dit(action_tokens, timestep, context, context_mask)
        tokens = state["tokens"]
        freqs = state["freqs"]
        t_mod = state["t_mod"]
        ctx = state["context"]
        ctx_mask = state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                tokens = checkpoint(
                    block,
                    tokens,
                    ctx,
                    t_mod,
                    freqs,
                    ctx_mask,
                    None,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, ctx, t_mod, freqs, ctx_mask)

        return self.post_dit(tokens)
