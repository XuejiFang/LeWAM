# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Transformer primitives for LeWAM models."""

"""Transformer blocks and embeddings used by the LeWAM predictor."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN-zero modulation: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """Feed-forward block for the released LeWAM architecture."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention with pre-normalization."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.norm_q = nn.RMSNorm(inner_dim, eps=1e-6, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(inner_dim, eps=1e-6, elementwise_affine=True)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        rope_position_ids: Optional[torch.Tensor] = None,
        rope_module=None,
    ) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)
        B, T, _ = x.shape
        q, k, v = (t.view(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3) for t in (q, k, v))
        if rope_position_ids is not None:
            if rope_module is None:
                raise ValueError("rope_module is required when rope_position_ids is provided")
            q, k = rope_module(q, k, rope_position_ids)
        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop,
            is_causal=False,
        )
        return self.to_out(out.permute(0, 2, 1, 3).reshape(B, T, -1))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def split_modulation(self, modulation: torch.Tensor):
        return self.adaLN_modulation(modulation).chunk(6, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        modulation: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope_position_ids: Optional[torch.Tensor] = None,
        rope_module=None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.split_modulation(modulation)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            attn_mask=attn_mask,
            rope_position_ids=rope_position_ids,
            rope_module=rope_module,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MLP(nn.Module):
    """Two-layer MLP with optional normalization and configurable activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None,
                 norm_fn=None, act_fn=nn.GELU):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), norm, act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 2:
            return self.net(x)
        shape = x.shape[:-1]
        return self.net(x.reshape(-1, x.shape[-1])).reshape(*shape, -1)


class TimestepEmbedder(nn.Module):
    """DiT-style sinusoidal timestep embedder followed by a two-layer MLP."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        expected_dtype = self.net[0].weight.dtype
        if t_freq.dtype != expected_dtype:
            t_freq = t_freq.to(expected_dtype)
        return self.net(t_freq)


class Transformer(nn.Module):
    """Single-stream Transformer with per-layer AdaLN and optional gradient checkpointing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int,
                 heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            ConditionalBlock(
                hidden_dim, heads, dim_head, mlp_dim, dropout,
            )
            for _ in range(depth)
        ])
        if input_dim != hidden_dim or output_dim != hidden_dim:
            raise ValueError("LeWAM Transformer input, hidden and output dimensions must match")

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_position_ids: Optional[torch.Tensor] = None,
        rope_module=None,
    ) -> torch.Tensor:
        if c is None:
            raise ValueError("c is required for conditional transformer blocks")
        modulation = c
        for block in self.layers:
            if self.gradient_checkpointing and self.training:
                def _block_forward(x_, modulation_, block_=block):
                    return block_(
                        x_,
                        modulation_,
                        attn_mask=attn_mask,
                        rope_position_ids=rope_position_ids,
                        rope_module=rope_module,
                    )

                x = torch_checkpoint(_block_forward, x, modulation, use_reentrant=False)
            else:
                x = block(
                    x,
                    modulation,
                    attn_mask=attn_mask,
                    rope_position_ids=rope_position_ids,
                    rope_module=rope_module,
                )
        return self.norm(x)
