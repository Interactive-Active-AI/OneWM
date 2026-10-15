"""Wan2.2-TI2V-5B/14B backbone.

Loads real diffusers-format Wan2.2 weights (safetensors) directly. The DiT
architecture and state_dict key layout match upstream Wan2.2 exactly:

    patch_embedding.{weight,bias}                  # Conv3d
    text_embedding.0.{weight,bias}                 # Linear(text_dim -> hidden)
    text_embedding.2.{weight,bias}                 # Linear(hidden -> hidden)
    time_embedding.0.{weight,bias}                 # Linear(freq_dim -> hidden)
    time_embedding.2.{weight,bias}                 # Linear(hidden -> hidden)
    time_projection.1.{weight,bias}                # Linear(hidden -> 6*hidden)
    blocks.{i}.<see wan_block.DiTBlock>
    head.head.{weight,bias}                        # Linear(hidden -> p*p*p*out_dim)
    head.modulation                                # [1, 2, hidden_dim]

The VAE (`Wan2.2_VAE.pth`) and UMT5 text encoder
(`models_t5_umt5-xxl-enc-bf16.pth`) are loaded from the configured
Wan2.2 checkpoint directory.
"""

from __future__ import annotations

import html
import json
import logging
import math
import string
from pathlib import Path
from typing import Optional, List

import re
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from onewm.backbone.base import BaseBackbone, BackboneInfo
from onewm.config import BackboneConfig
from onewm.utils.checkpoint import infer_backbone_info
from onewm.modules.wan_block import (
    DiTBlock,
    sinusoidal_embedding_1d,
    precompute_freqs_cis_3d,
)


# ---- Self-contained Wan2.2 VAE/T5 support ----
def basic_clean(text):
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)

        # arguments
        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update(
                {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": self.seq_len,
                }
            )
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn) ** -0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn) ** -0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads) ** -0.5
        )


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        )

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)
        )

    def forward(
        self, x, mask=None, encoder_states=None, encoder_mask=None, pos_bias=None
    ):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(
            x
            + self.cross_attn(self.norm2(x), context=encoder_states, mask=encoder_mask)
        )
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(
            lq, device=device
        ).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact)
                / math.log(self.max_dist / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1)
        )
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):
    def __init__(
        self,
        vocab,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        num_layers,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = (
            vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        )
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
            if shared_pos
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Decoder(nn.Module):
    def __init__(
        self,
        vocab,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        num_layers,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = (
            vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        )
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=False)
            if shared_pos
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5CrossAttention(
                    dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        b, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        encoder_layers,
        decoder_layers,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            encoder_layers,
            num_buckets,
            shared_pos,
            dropout,
        )
        self.decoder = T5Decoder(
            self.token_embedding,
            dim,
            dim_attn,
            dim_ffn,
            num_heads,
            decoder_layers,
            num_buckets,
            shared_pos,
            dropout,
        )
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(
    name,
    encoder_only=False,
    decoder_only=False,
    return_tokenizer=False,
    tokenizer_kwargs={},
    dtype=torch.float32,
    device="cpu",
    **kwargs,
):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs["vocab"] = kwargs.pop("vocab_size")
        kwargs["num_layers"] = kwargs.pop("encoder_layers")
        _ = kwargs.pop("decoder_layers")
    elif decoder_only:
        model_cls = T5Decoder
        kwargs["vocab"] = kwargs.pop("vocab_size")
        kwargs["num_layers"] = kwargs.pop("decoder_layers")
        _ = kwargs.pop("encoder_layers")
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)

    # init tokenizer
    if return_tokenizer:
        tokenizer = HuggingfaceTokenizer(f"google/{name}", **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1,
    )
    cfg.update(**kwargs)
    return _t5("umt5-xxl", **cfg)


class T5EncoderModel:
    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device="cpu",
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = (
            umt5_xxl(
                encoder_only=True, return_tokenizer=False, dtype=dtype, device=device
            )
            .eval()
            .requires_grad_(False)
        )
        logging.info(f"loading {checkpoint_path}")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        )
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean="whitespace"
        )

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]


CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
                # nn.Conv2d(dim, dim//2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != "Rep"
                    ):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == "Rep"
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    return x


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False
    ):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i] if i < len(temperal_downsample) else False
            )
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def forward(self, x, scale=[0, 1]):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=16, dim=160, device="cpu", **kwargs):
    # params
    cfg = dict(
        dim=dim,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f"loading {pretrained_path}")
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device, weights_only=False),
        assign=True,
    )

    return model


class Wan2_2_VAE:
    def __init__(
        self,
        z_dim=48,
        c_dim=160,
        vae_pth=None,
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=torch.float,
        device="cuda",
    ):

        self.dtype = dtype
        self.device = device

        mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=dtype,
            device=device,
        )
        std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=dtype,
            device=device,
        )
        self.scale = [mean, 1.0 / std]

        # init model
        self.model = (
            _video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
                dim=c_dim,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    def encode(self, videos):
        with torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=str(self.device).startswith("cuda")
        ):
            return self.model.encode(videos, self.scale)

    def decode(self, zs):
        try:
            if not isinstance(zs, list):
                raise TypeError("zs should be a list")
            with amp.autocast(
                dtype=self.dtype, enabled=str(self.device).startswith("cuda")
            ):
                return [
                    self.model.decode(u.unsqueeze(0), self.scale)
                    .float()
                    .clamp_(-1, 1)
                    .squeeze(0)
                    for u in zs
                ]
        except TypeError as e:
            logging.info(e)
            return None


def _load_wan22_vae(
    vae_pth: str, device: str = "cpu", dtype: torch.dtype = torch.float32
):
    return Wan2_2_VAE(
        z_dim=48,
        c_dim=160,
        vae_pth=vae_pth,
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=dtype,
        device=device,
    )


def _load_wan22_t5(
    ckpt_path: str,
    tokenizer_path: str,
    text_len: int = 512,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    return T5EncoderModel(
        text_len=text_len,
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        shard_fn=None,
    )


class Wan22Head(nn.Module):
    """Final projection: AdaLN(norm) -> Linear -> (B, S, out_dim*p_t*p_h*p_w).

    State_dict keys: head.head.{weight,bias}, head.modulation.
    """

    def __init__(self, hidden_dim: int, out_dim: int, patch_size, eps: float = 1e-6):
        super().__init__()
        p_t, p_h, p_w = patch_size
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, out_dim * p_t * p_h * p_w)
        # Wan2.2 head modulation is [1, 2, D]: shift, scale.
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: [B, hidden_dim] (single timestep) OR [B, S, hidden_dim] (per-token).
        has_seq = t.dim() == 3
        chunk_dim = 2 if has_seq else 1
        if has_seq:
            t_mod = t.unsqueeze(2)  # [B, S, 1, D]
        else:
            t_mod = t.unsqueeze(1)  # [B, 1, D]
        base = self.modulation.to(dtype=t.dtype, device=t.device)
        if has_seq:
            base = base.unsqueeze(0)  # [1,1,2,D]
        shift, scale = (base + t_mod).chunk(2, dim=chunk_dim)
        if has_seq:
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift = shift.squeeze(1)
            scale = scale.squeeze(1)
        x = self.norm(x) * (1 + scale) + shift
        return self.head(x)


class Wan22Dit(nn.Module):
    """Wan2.2 video DiT — keys match upstream safetensors layout.

    NOTE: this implements the `seperated_timestep + fuse_vae_embedding_in_latents`
    mode used by Wan2.2-TI2V-5B (per-token timestep, with first-frame timestep=0
    for image conditioning).
    """

    def __init__(self, info: BackboneInfo):
        super().__init__()
        self.info = info
        self.hidden_dim = info.hidden_dim
        self.num_heads = info.num_heads
        self.attn_head_dim = info.attn_head_dim
        self.num_layers = info.num_layers
        self.freq_dim = info.freq_dim
        self.text_dim = info.text_dim
        self.patch_size = tuple(info.patch_size)
        self.in_channels = info.in_channels
        self.eps = info.eps

        p_t, p_h, p_w = self.patch_size
        self.patch_embedding = nn.Conv3d(
            self.in_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=info.ffn_dim,
                    eps=self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = Wan22Head(
            self.hidden_dim, self.in_channels, self.patch_size, eps=self.eps
        )

        # 3D RoPE freqs are complex64; we (re)build them per-device on demand
        # because `nn.Module.to(real_dtype)` would silently cast away the
        # imaginary part of registered complex buffers.
        self._freqs_cache: dict = {}

    def _get_3d_freqs(self, device: torch.device):
        key = str(device)
        cached = self._freqs_cache.get(key)
        if cached is None:
            f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(self.attn_head_dim)
            cached = (f_freqs.to(device), h_freqs.to(device), w_freqs.to(device))
            self._freqs_cache[key] = cached
        return cached

    # --------------------------- patchify -------------------------------
    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """[B, C, T, H, W] -> [B, hidden, f, h, w]."""
        return self.patch_embedding(latents)

    def unpatchify(self, tokens_flat: torch.Tensor, grid: tuple) -> torch.Tensor:
        """[B, S, out_dim*p_t*p_h*p_w] -> [B, out_dim, T, H, W]."""
        from einops import rearrange

        f, h, w = grid
        p_t, p_h, p_w = self.patch_size
        return rearrange(
            tokens_flat,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f,
            h=h,
            w=w,
            x=p_t,
            y=p_h,
            z=p_w,
        )

    # --------------------------- 3D RoPE freqs --------------------------
    def _build_3d_freqs(
        self, f: int, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        """Concatenate 3D RoPE freqs into [S, 1, head_dim/2] complex tensor."""
        freqs_f, freqs_h, freqs_w = self._get_3d_freqs(device)
        ff = freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1)
        fh = freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1)
        fw = freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        freqs = torch.cat([ff, fh, fw], dim=-1).reshape(f * h * w, 1, -1)
        return freqs

    # --------------------------- pre/post DiT ---------------------------
    def pre_dit(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        clean_prefix_latent_steps: int = 1,
    ) -> dict:
        """Prepare video tokens for MoT joint attention.

        Args:
            latents: [B, in_channels, T, H, W] noisy video latents
            timestep: [B] integer timesteps (will become per-token internally).
            context: [B, L, text_dim] text embeddings.
            context_mask: [B, L] boolean.
            clean_prefix_latent_steps: number of leading latent frames whose
                per-token timestep is fixed to zero. The default preserves the
                single-observation behavior used by LIBERO.
        Returns:
            dict with tokens, freqs, t_mod, context, context_mask, meta.
        """
        from einops import rearrange

        patch_param = self.patch_embedding.weight
        latents = latents.to(device=patch_param.device, dtype=patch_param.dtype)
        text_param = self.text_embedding[0].weight
        context = context.to(device=text_param.device, dtype=text_param.dtype)

        B = latents.shape[0]
        device = latents.device
        if context_mask is not None:
            context_mask = context_mask.to(device=device)
        x = self.patchify(latents)  # [B, D, f, h, w]
        _, _, f, h, w = x.shape
        tokens_per_frame = h * w
        seq_len = f * h * w

        clean_prefix_latent_steps = int(clean_prefix_latent_steps)
        if not 1 <= clean_prefix_latent_steps <= f:
            raise ValueError(
                "clean_prefix_latent_steps must be within the latent sequence: "
                f"{clean_prefix_latent_steps} vs {f}"
            )

        # Per-token timesteps: every configured clean latent frame uses t=0.
        ref_dtype = latents.dtype if latents.is_floating_point() else torch.float32
        token_t = torch.ones(
            (B, f, tokens_per_frame), dtype=ref_dtype, device=device
        ) * timestep.view(B, 1, 1).to(ref_dtype)
        token_t[:, :clean_prefix_latent_steps, :] = 0
        token_t = token_t.reshape(B, -1)  # [B, S]
        t_emb = sinusoidal_embedding_1d(self.freq_dim, token_t.reshape(-1))
        t = self.time_embedding(t_emb).reshape(B, seq_len, self.hidden_dim)
        # time_projection -> [B, S, 6*D] -> [B, S, 6, D]
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        # Context (text) projection.
        ctx = self.text_embedding(context)
        if context_mask is not None and context_mask.dim() == 2:
            # Expand to [B, S, L] for use by cross-attention SDPA path.
            context_mask = context_mask.unsqueeze(1).expand(B, seq_len, -1)

        # Token sequence + 3D RoPE.
        tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = self._build_3d_freqs(f, h, w, device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "t": t,  # used by head modulation
            "context": ctx,
            "context_mask": context_mask,
            "meta": {
                "T": latents.shape[2],
                "H": latents.shape[3],
                "W": latents.shape[4],
                "f": f,
                "h": h,
                "w": w,
                "tokens_per_frame": tokens_per_frame,
                "clean_prefix_latent_steps": clean_prefix_latent_steps,
            },
        }

    def post_dit(
        self, tokens: torch.Tensor, meta: dict, t: torch.Tensor
    ) -> torch.Tensor:
        """Project tokens back to latent space via head + unpatchify."""
        x = self.head(tokens, t)
        return self.unpatchify(x, (meta["f"], meta["h"], meta["w"]))

    # --------------------------- weight loading -------------------------
    @torch.no_grad()
    def load_pretrained(
        self, model_path: str, dtype: Optional[torch.dtype] = None
    ) -> dict:
        """Load real Wan2.2 DiT weights from a diffusers-style directory.

        Looks for `diffusion_pytorch_model.safetensors.index.json` and
        the sharded `diffusion_pytorch_model-XXXXX-of-YYYYY.safetensors`.
        """
        from safetensors.torch import load_file

        path = Path(model_path)
        index_file = path / "diffusion_pytorch_model.safetensors.index.json"
        if index_file.exists():
            index = json.loads(index_file.read_text())
            shard_files: List[str] = sorted(set(index["weight_map"].values()))
            state_dict: dict = {}
            for shard in shard_files:
                state_dict.update(load_file(str(path / shard)))
        else:
            single = path / "diffusion_pytorch_model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"No safetensors found under {path}. "
                    "Expected `diffusion_pytorch_model.safetensors[.index.json]`."
                )
            state_dict = load_file(str(single))

        if dtype is not None:
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

        result = self.load_state_dict(state_dict, strict=False)
        return {
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "num_loaded": len(state_dict),
        }


class Wan22VAE(nn.Module):
    """Real Wan2.2 Video VAE adapter.

    Wraps the local Wan2.2 VAE implementation so it exposes the encode/decode
    tensor API expected by :class:`MoTFramework`.

    - ``encode(video)``: ``[B, 3, T, H, W]`` (pixel space, [-1, 1]) ->
      ``[B, 48, T/4, H/8, W/8]`` (latent, normalized).
    - ``decode(latents)``: inverse, returns ``[B, 3, T, H, W]`` clamped to
      [-1, 1].

    Compression ratios are fixed at temporal=4, spatial=16 for Wan2.2-TI2V-5B
    (the encoder applies a patch-size-2 stem on top of three 2x spatial
    downsamples).
    """

    temporal_compress = 4
    spatial_compress = 16

    def __init__(
        self,
        vae_pth: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        in_channels: int = 48,
    ):
        super().__init__()
        self.in_channels = in_channels
        self._device = device
        self._dtype = dtype
        self._impl = None
        if vae_pth is not None and Path(vae_pth).exists():
            self._impl = _load_wan22_vae(vae_pth, device=device, dtype=dtype)
            print(f"[Wan22VAE] Loaded VAE from {vae_pth}")
        else:
            print(
                f"[Wan22VAE] VAE checkpoint not found at {vae_pth}; "
                "encode/decode will raise on use."
            )

    def _ensure_loaded(self) -> None:
        if self._impl is None:
            raise RuntimeError(
                "Wan22VAE checkpoint was not loaded; pass a valid `Wan2.2_VAE.pth` "
                "via `BackboneConfig.pretrained_model_id` containing the file."
            )

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode pixel video to latents.

        Args:
            video: ``[B, 3, T, H, W]`` in [-1, 1].
        Returns:
            ``[B, 48, T/4, H/8, W/8]`` latents (already mean/std normalized).
        """
        self._ensure_loaded()
        # The VAE keeps weights in its own dtype (typically float32); cast
        # input to match before the conv stack.
        x = video.to(device=self._device, dtype=self._dtype)
        return self._impl.model.encode(x, self._impl.scale)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to pixel video, clamped to [-1, 1]."""
        self._ensure_loaded()
        z = latents.to(device=self._device, dtype=self._dtype)
        out = self._impl.model.decode(z, self._impl.scale)
        return out.float().clamp_(-1, 1)


class Wan22TextEncoder:
    """UMT5-XXL encoder adapter for Wan2.2.

    Loads the local UMT5 encoder implementation with the
    ``models_t5_umt5-xxl-enc-bf16.pth`` checkpoint and the
    ``google/umt5-xxl`` tokenizer directory shipped with Wan2.2.
    """

    def __init__(
        self,
        ckpt_path: str,
        tokenizer_path: str,
        text_len: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.text_len = text_len
        self.device = device
        self.dtype = dtype
        self._impl = _load_wan22_t5(
            ckpt_path=ckpt_path,
            tokenizer_path=tokenizer_path,
            text_len=text_len,
            device=device,
            dtype=dtype,
        )
        print(f"[Wan22TextEncoder] Loaded UMT5-XXL from {ckpt_path}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of strings.

        Returns:
            (context, mask):
                - context: ``[B, text_len, 4096]`` (zero-padded).
                - mask: ``[B, text_len]`` boolean (True at valid positions).
        """
        # Upstream `T5EncoderModel.__call__(texts, device)` returns a list of
        # variable-length tensors; pad them and build a boolean mask.
        ctx_list = self._impl(texts, self.device)  # list of [L_i, 4096]
        B = len(ctx_list)
        L = self.text_len
        D = ctx_list[0].shape[-1]
        context = torch.zeros(B, L, D, device=self.device, dtype=ctx_list[0].dtype)
        mask = torch.zeros(B, L, device=self.device, dtype=torch.bool)
        for i, c in enumerate(ctx_list):
            n = min(c.shape[0], L)
            context[i, :n] = c[:n]
            mask[i, :n] = True
        return context, mask


class Wan22Backbone(BaseBackbone):
    """Wan2.2-TI2V-5B/14B backbone. Loads real DiT/VAE/T5 weights when available.

    The ``pretrained_model_id`` is expected to be a directory laid out like
    ``Wan2.2-TI2V-5B/`` with the following entries (any missing files cause
    the corresponding submodule to remain unloaded; consuming code should
    check via :meth:`has_vae` / :meth:`has_text_encoder`)::

        diffusion_pytorch_model.safetensors[.index.json]
        Wan2.2_VAE.pth
        models_t5_umt5-xxl-enc-bf16.pth
        google/umt5-xxl/                              # HF tokenizer dir
    """

    def __init__(
        self,
        config: BackboneConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        load_vae: bool = True,
        load_dit: bool = True,
        load_text_encoder: bool = False,
    ):
        super().__init__()
        self._config = config
        self._dtype = dtype
        self._device = device

        self._info = infer_backbone_info(config.pretrained_model_id)
        self.dit = Wan22Dit(self._info)

        model_dir = (
            Path(config.pretrained_model_id) if config.pretrained_model_id else None
        )

        # ----- DiT weights (eager) -----
        if load_dit and model_dir is not None and model_dir.exists():
            try:
                info_dict = self.dit.load_pretrained(str(model_dir), dtype=dtype)
                print(
                    f"[Wan22Backbone] Loaded DiT weights from {model_dir} "
                    f"(num_loaded={info_dict['num_loaded']}, "
                    f"missing={len(info_dict['missing_keys'])}, "
                    f"unexpected={len(info_dict['unexpected_keys'])})"
                )
            except Exception as e:
                print(f"[Wan22Backbone] DiT weight load skipped: {e}")

        # ----- VAE (optional eager) -----
        self.vae: Optional[Wan22VAE] = None
        if load_vae and model_dir is not None:
            vae_pth = model_dir / "Wan2.2_VAE.pth"
            if vae_pth.exists():
                self.vae = Wan22VAE(
                    vae_pth=str(vae_pth),
                    device=device,
                    dtype=torch.float32,
                    in_channels=self._info.in_channels,
                )
            else:
                print(
                    f"[Wan22Backbone] VAE checkpoint not found at {vae_pth}; "
                    "encode/decode disabled."
                )

        # ----- Text encoder (optional eager; large, ~10GB) -----
        self.text_encoder: Optional[Wan22TextEncoder] = None
        if load_text_encoder and model_dir is not None:
            t5_pth = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
            tok_dir = model_dir / "google" / "umt5-xxl"
            if t5_pth.exists() and tok_dir.exists():
                self.text_encoder = Wan22TextEncoder(
                    ckpt_path=str(t5_pth),
                    tokenizer_path=str(tok_dir),
                    text_len=int(getattr(config, "tokenizer_max_len", 512)),
                    device=device,
                    dtype=torch.bfloat16,
                )
            else:
                print(
                    f"[Wan22Backbone] T5 ckpt or tokenizer missing under "
                    f"{model_dir}; text encoder disabled."
                )

    @property
    def info(self) -> BackboneInfo:
        return self._info

    def has_vae(self) -> bool:
        return self.vae is not None

    def has_text_encoder(self) -> bool:
        return self.text_encoder is not None

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("Wan22Backbone: VAE not loaded.")
        latents = self.vae.encode(video)
        # VAE runs in float32 on its own internal device (set at construction);
        # cast back to the backbone's compute device + dtype so the downstream
        # DiT sees a matching tensor.
        return latents.to(device=video.device, dtype=self._dtype)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("Wan22Backbone: VAE not loaded.")
        return self.vae.decode(latents)

    def encode_text(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            # Fallback to random embeddings to keep tensor flow alive in tests
            # that don't care about real text conditioning.
            B = len(text)
            L = 77
            device = torch.device(self._device)
            context = torch.randn(
                B, L, self._info.text_dim, device=device, dtype=self._dtype
            )
            mask = torch.ones(B, L, device=device, dtype=torch.bool)
            return context, mask
        return self.text_encoder.encode(text)

    def get_dit(self) -> nn.Module:
        return self.dit

    def get_vae(self) -> Optional[nn.Module]:
        return self.vae
