"""3D rotary position embeddings for the LeWAM Predictor."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from .sparse_offsets import normalize_sparse_offsets

ACTION_SPATIAL_H = -1
ACTION_SPATIAL_W = -1


def split_head_dim_for_3d_rope(dim_head: int) -> Tuple[int, int, int]:
    """Split head dim into three even chunks for independent 1D RoPE axes."""
    if dim_head < 6:
        raise ValueError(f"dim_head must be >= 6 for 3D RoPE, got {dim_head}")

    def _largest_even_at_most(value: int) -> int:
        return value if value % 2 == 0 else value - 1

    dim_t = _largest_even_at_most(max(2, dim_head // 3))
    remaining = dim_head - dim_t
    dim_h = _largest_even_at_most(max(2, remaining // 2))
    dim_w = dim_head - dim_t - dim_h
    if dim_w < 2:
        raise ValueError(f"Unable to split dim_head={dim_head} into three RoPE chunks")
    if dim_w % 2 != 0:
        if dim_h > 2:
            dim_h -= 2
            dim_w += 2
        elif dim_t > 2:
            dim_t -= 2
            dim_w += 2
        else:
            raise ValueError(f"Unable to make all RoPE chunks even for dim_head={dim_head}")
    return dim_t, dim_h, dim_w


def _patch_position_ids(*, num_patches: int, grid_w: int, t: int, device: Optional[torch.device]) -> torch.Tensor:
    patch_ids = torch.arange(num_patches, dtype=torch.long, device=device)
    patch_t = torch.full((num_patches,), int(t), dtype=torch.long, device=device)
    patch_h = patch_ids // grid_w
    patch_w = patch_ids % grid_w
    return torch.stack([patch_t, patch_h, patch_w], dim=-1)


def _action_position_ids(*, start: int, end: int, device: Optional[torch.device]) -> torch.Tensor:
    action_t = torch.arange(int(start), int(end), dtype=torch.long, device=device)
    action_h = torch.full_like(action_t, ACTION_SPATIAL_H)
    action_w = torch.full_like(action_t, ACTION_SPATIAL_W)
    return torch.stack([action_t, action_h, action_w], dim=-1)


def build_rope_position_ids(
    *,
    num_patches: int,
    action_horizon: int,
    action_block_size: int,
    num_transition: int,
    grid_h: int,
    grid_w: int,
    sparse_offsets: Optional[torch.Tensor] = None,
    include_query_tokens: bool = True,
    action_segments: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build ``(T, 3)`` coordinates for a Predictor sequence layout."""
    if num_patches != grid_h * grid_w:
        raise ValueError(
            f"Expected num_patches={num_patches} to equal grid_h*grid_w={grid_h * grid_w}"
        )
    if action_segments < 1:
        raise ValueError(f"action_segments must be >= 1, got {action_segments}")

    sparse_offsets = normalize_sparse_offsets(
        sparse_offsets,
        action_block_size=action_block_size,
        num_transition=num_transition,
        device=device,
    )

    segments = [
        _patch_position_ids(
            num_patches=num_patches,
            grid_w=grid_w,
            t=0,
            device=device,
        )
    ]
    segments.extend(
        _action_position_ids(
            start=0,
            end=action_horizon,
            device=device,
        )
        for _ in range(action_segments)
    )

    if include_query_tokens:
        for k in range(num_transition):
            segments.append(
                _patch_position_ids(
                    num_patches=num_patches,
                    grid_w=grid_w,
                    t=int(sparse_offsets[k + 1].item()),
                    device=device,
                )
            )

    return torch.cat(segments, dim=0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_1d_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """Apply 1D RoPE to the last dimension of ``x``.

    Parameters
    ----------
    x:
        Tensor of shape ``(..., seq_len, dim)`` where ``dim`` is even.
    position_ids:
        Tensor of shape ``(seq_len,)``.
    inv_freq:
        Tensor of shape ``(dim // 2,)``.
    """
    seq_len = x.size(-2)
    dim = x.size(-1)
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")

    t = position_ids.to(device=x.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding3D(nn.Module):
    """Factored 3D RoPE over temporal and spatial axes."""

    def __init__(self, dim_head: int, theta: float = 10000.0):
        super().__init__()
        self.dim_head = int(dim_head)
        self.theta = float(theta)
        self.dim_t, self.dim_h, self.dim_w = split_head_dim_for_3d_rope(self.dim_head)

        def _inv_freq(dim: int) -> torch.Tensor:
            return 1.0 / (
                self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )

        self.register_buffer("inv_freq_t", _inv_freq(self.dim_t), persistent=False)
        self.register_buffer("inv_freq_h", _inv_freq(self.dim_h), persistent=False)
        self.register_buffer("inv_freq_w", _inv_freq(self.dim_w), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate merged Q/K tensors.

        Parameters
        ----------
        q, k:
            Tensors of shape ``(batch, heads, seq_len, dim_head)``.
        position_ids:
            Tensor of shape ``(seq_len, 3)`` with columns ``(t, h, w)``.
        """
        if position_ids.ndim != 2 or position_ids.size(-1) != 3:
            raise ValueError(f"Expected position_ids shape (T, 3), got {tuple(position_ids.shape)}")
        if q.size(-1) != self.dim_head or k.size(-1) != self.dim_head:
            raise ValueError(
                f"Expected dim_head={self.dim_head}, got q={q.size(-1)}, k={k.size(-1)}"
            )

        t_ids = position_ids[:, 0]
        h_ids = position_ids[:, 1]
        w_ids = position_ids[:, 2]

        q_t, q_h, q_w = q.split([self.dim_t, self.dim_h, self.dim_w], dim=-1)
        k_t, k_h, k_w = k.split([self.dim_t, self.dim_h, self.dim_w], dim=-1)

        q_t = _apply_1d_rope(q_t, t_ids, inv_freq=self.inv_freq_t)
        k_t = _apply_1d_rope(k_t, t_ids, inv_freq=self.inv_freq_t)
        q_h = _apply_1d_rope(q_h, h_ids, inv_freq=self.inv_freq_h)
        k_h = _apply_1d_rope(k_h, h_ids, inv_freq=self.inv_freq_h)
        q_w = _apply_1d_rope(q_w, w_ids, inv_freq=self.inv_freq_w)
        k_w = _apply_1d_rope(k_w, w_ids, inv_freq=self.inv_freq_w)

        return torch.cat([q_t, q_h, q_w], dim=-1), torch.cat([k_t, k_h, k_w], dim=-1)
