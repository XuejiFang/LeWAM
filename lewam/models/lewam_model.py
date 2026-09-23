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

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput

from .module import (
    MLP,
    TimestepEmbedder,
    Transformer,
)
from .rope import (
    RotaryEmbedding3D,
    build_rope_position_ids,
)
from .vision_encoder import StaticChannelwiseLayerFusion
from .sparse_offsets import normalize_sparse_offsets

@dataclass
class LeWAMTransformerOutput(BaseOutput):
    """
    Output class for [`LeWAMModel`].

    Args:
        sample (`torch.Tensor`):
            The predicted action hidden states of shape `(batch_size, action_horizon, action_hidden_dim)`.
        vision_hidden_states (`torch.Tensor` or `None`):
            Hidden states from next-feature query positions of shape
            `(batch_size, num_transition, num_patches, vision_hidden_dim)`, or `None` if not requested.
    """

    sample: torch.Tensor
    vision_hidden_states: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SequenceLayout:
    sparse_offsets: torch.Tensor
    z0: slice
    clean_actions: slice
    clean_positions: torch.Tensor
    query_slices: Tuple[slice, ...]
    seq_len: int
    noisy_actions: Optional[slice] = None


class LeWAMModel(ModelMixin, ConfigMixin):
    """Configurable conditional Transformer for LeWAM action flow matching."""

    _supports_gradient_checkpointing = True
    config_name = "config.json"

    VISION_BRANCH = 0
    ACTION_BRANCH = 1

    @register_to_config
    def __init__(
        self,
        action_block_size: int = 1,
        num_transition: int = 1,
        depth: int = 12,
        heads: int = 8,
        input_dim: int = 256,
        token_dim: Optional[int] = None,
        vision_hidden_dim: int = 256,
        action_hidden_dim: int = 256,
        vision_mlp_dim: int = 1024,
        action_mlp_dim: int = 1024,
        dim_head: int = 64,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        num_tasks: int = 1,
        context_length: int = 196,
        grid_h: int = 14,
        grid_w: int = 14,
        rope_theta: float = 10000.0,
        action_dim: int = 256,
        vision_feature_fusion_enabled: bool = False,
        vision_feature_fusion_num_layers: int = 0,
        vision_feature_fusion_initial_final_weight: float = 0.8,
    ):
        super().__init__()
        self.action_block_size = int(action_block_size)
        self.num_transition = int(num_transition)
        if self.action_block_size < 1:
            raise ValueError(f"action_block_size must be >= 1, got {self.action_block_size}")
        if self.num_transition < 1:
            raise ValueError(f"num_transition must be >= 1, got {self.num_transition}")
        self.action_horizon = self.num_transition * self.action_block_size
        self.context_length = int(context_length)
        if self.context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {self.context_length}")
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        if self.grid_h < 1 or self.grid_w < 1:
            raise ValueError(f"grid_h and grid_w must be >= 1, got {self.grid_h}x{self.grid_w}")
        if self.grid_h * self.grid_w != self.context_length:
            raise ValueError(
                f"Expected grid_h*grid_w == context_length "
                f"({self.grid_h}*{self.grid_w} != {self.context_length})"
            )
        self.rope_theta = float(rope_theta)
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        input_dim = int(input_dim)
        action_dim = int(action_dim)
        token_dim = input_dim if token_dim is None else int(token_dim)
        if token_dim < 1:
            raise ValueError(f"token_dim must be >= 1, got {token_dim}")
        vision_hidden_dim = int(vision_hidden_dim)
        action_hidden_dim = int(action_hidden_dim)
        vision_mlp_dim = int(vision_mlp_dim)
        action_mlp_dim = int(action_mlp_dim)
        self.vision_feature_fusion_enabled = bool(vision_feature_fusion_enabled)
        if self.vision_feature_fusion_enabled:
            self.vision_fuser = StaticChannelwiseLayerFusion(
                hidden_dim=input_dim,
                num_layers=int(vision_feature_fusion_num_layers),
                initial_final_weight=float(vision_feature_fusion_initial_final_weight),
            )
        else:
            self.vision_fuser = None
        if vision_hidden_dim != action_hidden_dim:
            raise ValueError(
                "single transformer mode requires vision_hidden_dim == action_hidden_dim, "
                f"got vision_hidden_dim={vision_hidden_dim}, action_hidden_dim={action_hidden_dim}"
            )
        if vision_mlp_dim != action_mlp_dim:
            raise ValueError(
                "single transformer mode requires vision_mlp_dim == action_mlp_dim, "
                f"got vision_mlp_dim={vision_mlp_dim}, action_mlp_dim={action_mlp_dim}"
            )


        # Training sequence: z0, noisy actions, clean actions, query rollout.
        self.task_embed = nn.Embedding(int(num_tasks), token_dim)
        self.time_embedder = TimestepEmbedder(token_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.query_pos_embed = nn.Parameter(torch.randn(1, context_length, token_dim) * 0.02)
        self.rope = RotaryEmbedding3D(dim_head, theta=self.rope_theta)
        self.transformer = Transformer(
            token_dim,
            vision_hidden_dim,
            vision_hidden_dim,
            self.depth,
            heads,
            dim_head,
            vision_mlp_dim,
            dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.action_proj = MLP(input_dim=action_dim, output_dim=token_dim, hidden_dim=4 * token_dim)
        self.vision_proj = MLP(
            input_dim=input_dim,
            output_dim=token_dim,
            hidden_dim=4 * token_dim,
        )
        self.action_head = MLP(input_dim=action_hidden_dim, output_dim=action_dim, hidden_dim=4 * action_hidden_dim)
        self.next_feature_head = MLP(input_dim=vision_hidden_dim, output_dim=input_dim, hidden_dim=4 * vision_hidden_dim)

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func=None,
    ):
        del gradient_checkpointing_func
        for module in self.modules():
            if isinstance(module, Transformer):
                module.gradient_checkpointing = bool(enable)

    def _run_transformer(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        branch_type_ids: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        rope_position_ids: torch.Tensor,
    ):
        hidden = self.transformer(
            tokens,
            cond,
            attn_mask=attn_mask,
            rope_position_ids=rope_position_ids,
            rope_module=self.rope,
        )
        vision_rows = hidden[:, branch_type_ids == self.VISION_BRANCH]
        action_rows = hidden[:, branch_type_ids == self.ACTION_BRANCH]
        return vision_rows, action_rows

    def _forward_action_only(
        self,
        vision_tokens: torch.Tensor,
        noisy_action_tokens: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> LeWAMTransformerOutput:
        """Action-only forward for ``Z0 | noisy A'`` without future-state queries."""
        if vision_tokens.ndim != 3:
            raise ValueError(
                "Expected vision_tokens with shape (B, P, D), "
                f"got {tuple(vision_tokens.shape)}"
            )
        if noisy_action_tokens.ndim != 3:
            raise ValueError(
                "Expected noisy_action_tokens with shape (B, H, D), "
                f"got {tuple(noisy_action_tokens.shape)}"
            )

        batch_size = vision_tokens.size(0)
        total_actions = self.action_horizon
        num_patches = vision_tokens.size(1)
        if num_patches != self.context_length:
            raise ValueError(
                f"Expected {self.context_length} patch tokens, got {num_patches}"
            )
        if noisy_action_tokens.shape[:2] != (batch_size, total_actions):
            raise ValueError(
                f"Expected noisy_action_tokens shape prefix {(batch_size, total_actions)}, "
                f"got {tuple(noisy_action_tokens.shape[:2])}"
            )

        tokens = torch.cat([vision_tokens, noisy_action_tokens], dim=1)
        branch_type_ids = torch.cat([
            torch.full((num_patches,), self.VISION_BRANCH, dtype=torch.long, device=tokens.device),
            torch.full((total_actions,), self.ACTION_BRANCH, dtype=torch.long, device=tokens.device),
        ])

        t, task_id = self._validate_time_and_task(
            t,
            task_id,
            batch_size=batch_size,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        t_emb = self.time_embedder(t)
        zero_t_emb = self.time_embedder(torch.zeros_like(t))
        cond = torch.cat(
            [
                zero_t_emb.expand(-1, num_patches, -1),
                t_emb.expand(-1, total_actions, -1),
            ],
            dim=1,
        )
        cond = cond + self.task_embed(task_id).unsqueeze(1)

        attn_mask = self.build_inference_attention_mask(
            num_patches=num_patches,
            device=tokens.device,
        )
        rope_position_ids = build_rope_position_ids(
            num_patches=num_patches,
            action_horizon=total_actions,
            action_block_size=self.action_block_size,
            num_transition=self.num_transition,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            include_query_tokens=False,
            device=tokens.device,
        )
        if branch_type_ids.size(0) != tokens.size(1):
            raise RuntimeError(f"Action-only branch length mismatch: {branch_type_ids.size(0)} != {tokens.size(1)}")
        if rope_position_ids.size(0) != tokens.size(1):
            raise RuntimeError(f"Action-only RoPE length mismatch: {rope_position_ids.size(0)} != {tokens.size(1)}")

        _, action_rows = self._run_transformer(
            tokens,
            cond,
            branch_type_ids,
            attn_mask=attn_mask,
            rope_position_ids=rope_position_ids,
        )
        return LeWAMTransformerOutput(sample=action_rows, vision_hidden_states=None)

    def _build_query_tokens(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build query tokens from shared mask token + spatial positional embedding."""
        pos_embed = self.query_pos_embed.to(device=device, dtype=dtype)
        query = self.mask_token.to(device=device, dtype=dtype) + pos_embed  # (1, num_patches, D)
        query = query.unsqueeze(1).expand(-1, self.num_transition, -1, -1)  # (1, T, P, D)
        return query.expand(batch_size, -1, -1, -1).reshape(batch_size, self.num_transition * self.context_length, -1)

    def _build_sequence_layout(
        self,
        sparse_offsets: torch.Tensor,
        *,
        num_patches: int,
        device: torch.device,
        include_noisy_actions: bool,
    ) -> SequenceLayout:
        sparse_offsets = normalize_sparse_offsets(
            sparse_offsets,
            action_block_size=self.action_block_size,
            num_transition=self.num_transition,
            device=device,
        )
        z0 = slice(0, num_patches)
        cursor = z0.stop
        noisy_actions = None
        if include_noisy_actions:
            noisy_actions = slice(cursor, cursor + self.action_horizon)
            cursor = noisy_actions.stop
        clean_actions = slice(cursor, cursor + self.action_horizon)
        query_slices = []
        cursor = clean_actions.stop
        for _ in range(self.num_transition):
            query_slice = slice(cursor, cursor + num_patches)
            cursor = query_slice.stop
            query_slices.append(query_slice)
        clean_positions = torch.arange(clean_actions.start, clean_actions.stop, device=device, dtype=torch.long)
        return SequenceLayout(
            sparse_offsets=sparse_offsets,
            z0=z0,
            noisy_actions=noisy_actions,
            clean_actions=clean_actions,
            clean_positions=clean_positions,
            query_slices=tuple(query_slices),
            seq_len=cursor,
        )

    def build_training_attention_mask(
        self,
        sparse_offsets: torch.Tensor,
        *,
        num_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Training mask for ``Z0 | noisy A' | clean A | Q1..QN`` without leakage."""
        layout = self._build_sequence_layout(
            sparse_offsets,
            num_patches=num_patches,
            device=device,
            include_noisy_actions=True,
        )
        return self._build_clean_query_attention_mask(layout=layout, device=device)

    def _build_clean_query_attention_mask(
        self,
        *,
        layout: SequenceLayout,
        device: torch.device,
    ) -> torch.Tensor:
        sparse_offsets = layout.sparse_offsets
        z0 = layout.z0
        clean_actions = layout.clean_actions
        clean_positions = layout.clean_positions
        query_slices = layout.query_slices

        mask = torch.zeros((layout.seq_len, layout.seq_len), device=device, dtype=torch.bool)

        def allow(rows, cols):
            mask[rows, cols] = True

        allow(z0, z0)
        if layout.noisy_actions is not None:
            noisy_actions = layout.noisy_actions
            allow(noisy_actions, z0)
            mask[noisy_actions, noisy_actions] = torch.tril(
                torch.ones((self.action_horizon, self.action_horizon), device=device, dtype=torch.bool)
            )

        allow(clean_actions, z0)
        mask[clean_actions, clean_actions] = torch.tril(
            torch.ones((self.action_horizon, self.action_horizon), device=device, dtype=torch.bool)
        )

        for query_idx, query_slice in enumerate(query_slices):
            clean_end = int(sparse_offsets[query_idx + 1].item())
            allow(query_slice, z0)
            if clean_end > 0:
                allow(query_slice, clean_positions[:clean_end])
            allow(query_slice, query_slice)

        return mask

    def build_inference_attention_mask(
        self,
        *,
        num_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Inference mask for ``Z0 + noisy action horizon`` with causal actions."""
        seq_len = num_patches + self.action_horizon
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)
        z0 = slice(0, num_patches)
        noisy_actions = slice(num_patches, seq_len)
        mask[z0, z0] = True
        mask[noisy_actions, z0] = True
        mask[noisy_actions, noisy_actions] = torch.tril(
            torch.ones((self.action_horizon, self.action_horizon), device=device, dtype=torch.bool)
        )
        return mask

    def build_future_state_attention_mask(
        self,
        sparse_offsets: torch.Tensor,
        *,
        num_patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Inference mask for ``Z0 | clean A | Q1..QN`` future-state prediction."""
        layout = self._build_sequence_layout(
            sparse_offsets,
            num_patches=num_patches,
            device=device,
            include_noisy_actions=False,
        )
        return self._build_clean_query_attention_mask(layout=layout, device=device)

    def _validate_time_and_task(self, t, task_id, *, batch_size, device, dtype):
        if t is None:
            raise ValueError("t is required")
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2 or t.size(0) != batch_size or t.size(1) != 1:
            raise ValueError(f"Expected t shape ({batch_size}, 1), got {tuple(t.shape)}")
        t = t.to(device=device, dtype=dtype)

        if task_id is None:
            raise ValueError("task_id is required for task-conditioned prediction")
        if task_id.ndim == 2 and task_id.size(1) == 1:
            task_id = task_id.squeeze(1)
        if task_id.ndim != 1 or task_id.size(0) != batch_size:
            raise ValueError(f"Expected task_id with shape (B,), got {tuple(task_id.shape)}")
        return t, task_id.to(device=device)

    def _forward_with_future_queries(
        self,
        vision_tokens: torch.Tensor,
        action_token_segments: Tuple[torch.Tensor, ...],
        action_time_embeddings: Tuple[torch.Tensor, ...],
        zero_time_embedding: torch.Tensor,
        task_id: torch.Tensor,
        sparse_offsets: Optional[torch.Tensor],
    ) -> LeWAMTransformerOutput:
        """Run a clean-action query layout, optionally prefixed by noisy actions."""
        if len(action_token_segments) not in {1, 2}:
            raise ValueError(
                "Expected one clean-action segment or noisy and clean-action segments"
            )
        if len(action_time_embeddings) != len(action_token_segments):
            raise ValueError("Action token and timestep segment counts must match")

        batch_size, num_patches = vision_tokens.shape[:2]
        total_actions = self.action_horizon
        include_noisy_actions = len(action_token_segments) == 2
        layout = self._build_sequence_layout(
            sparse_offsets,
            num_patches=num_patches,
            device=vision_tokens.device,
            include_noisy_actions=include_noisy_actions,
        )
        query_tokens = self._build_query_tokens(
            batch_size,
            device=vision_tokens.device,
            dtype=vision_tokens.dtype,
        ).reshape(batch_size, self.num_transition, num_patches, -1)

        token_segments = [vision_tokens, *action_token_segments]
        branch_segments = [
            torch.full(
                (num_patches,),
                self.VISION_BRANCH,
                dtype=torch.long,
                device=vision_tokens.device,
            ),
            *[
                torch.full(
                    (total_actions,),
                    self.ACTION_BRANCH,
                    dtype=torch.long,
                    device=vision_tokens.device,
                )
                for _ in action_token_segments
            ],
        ]
        for transition_index in range(self.num_transition):
            token_segments.append(query_tokens[:, transition_index])
            branch_segments.append(
                torch.full(
                    (num_patches,),
                    self.VISION_BRANCH,
                    dtype=torch.long,
                    device=vision_tokens.device,
                )
            )

        tokens = torch.cat(token_segments, dim=1)
        branch_type_ids = torch.cat(branch_segments)
        if tokens.size(1) != layout.seq_len:
            raise RuntimeError(
                f"Future-query layout length mismatch: "
                f"{tokens.size(1)} != {layout.seq_len}"
            )

        cond_segments = [
            zero_time_embedding.expand(-1, num_patches, -1),
            *[
                embedding.expand(-1, total_actions, -1)
                for embedding in action_time_embeddings
            ],
            *[
                zero_time_embedding.expand(-1, num_patches, -1)
                for _ in range(self.num_transition)
            ],
        ]
        cond = torch.cat(cond_segments, dim=1)
        cond = cond + self.task_embed(task_id).unsqueeze(1)
        if cond.size(1) != layout.seq_len:
            raise RuntimeError(
                f"Future-query cond length mismatch: "
                f"{cond.size(1)} != {layout.seq_len}"
            )

        attn_mask = self._build_clean_query_attention_mask(
            layout=layout,
            device=tokens.device,
        )
        rope_position_ids = build_rope_position_ids(
            num_patches=num_patches,
            action_horizon=total_actions,
            action_block_size=self.action_block_size,
            num_transition=self.num_transition,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            sparse_offsets=layout.sparse_offsets,
            include_query_tokens=True,
            action_segments=len(action_token_segments),
            device=tokens.device,
        )
        if branch_type_ids.size(0) != layout.seq_len:
            raise RuntimeError(
                f"Future-query branch length mismatch: "
                f"{branch_type_ids.size(0)} != {layout.seq_len}"
            )
        if rope_position_ids.size(0) != layout.seq_len:
            raise RuntimeError(
                f"Future-query RoPE length mismatch: "
                f"{rope_position_ids.size(0)} != {layout.seq_len}"
            )

        vision_rows, action_rows = self._run_transformer(
            tokens,
            cond,
            branch_type_ids,
            attn_mask=attn_mask,
            rope_position_ids=rope_position_ids,
        )
        query_hidden = vision_rows[:, num_patches:]
        vision_hidden = query_hidden.reshape(
            batch_size,
            self.num_transition,
            num_patches,
            -1,
        )
        return LeWAMTransformerOutput(
            sample=action_rows,
            vision_hidden_states=vision_hidden,
        )

    def forward(
        self,
        vision_tokens: torch.Tensor,
        noisy_action_tokens: torch.Tensor,
        clean_action_tokens: Optional[torch.Tensor],
        t: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        sparse_offsets: Optional[torch.Tensor] = None,
        *,
        predict_next_feature: bool = True,
        return_dict: bool = True,
    ) -> LeWAMTransformerOutput | Tuple[torch.Tensor, ...]:
        """Return hidden states as a model output, or a tuple when ``return_dict=False``."""
        if not predict_next_feature:
            output = self._forward_action_only(
                vision_tokens,
                noisy_action_tokens,
                t=t,
                task_id=task_id,
            )
            return output if return_dict else output.to_tuple()
        if sparse_offsets is None:
            raise ValueError("sparse_offsets is required for training forward")
        if clean_action_tokens is None:
            raise ValueError("clean_action_tokens is required when predict_next_feature=True")
        if vision_tokens.ndim != 3:
            raise ValueError(
                "Expected vision_tokens with shape (B, P, D), "
                f"got {tuple(vision_tokens.shape)}"
            )
        if noisy_action_tokens.ndim != 3:
            raise ValueError(
                "Expected noisy_action_tokens with shape (B, H, D), "
                f"got {tuple(noisy_action_tokens.shape)}"
            )
        if clean_action_tokens.ndim != 3:
            raise ValueError(
                "Expected clean_action_tokens with shape (B, H, D), "
                f"got {tuple(clean_action_tokens.shape)}"
            )

        batch_size = vision_tokens.size(0)
        total_actions = self.action_horizon
        z0_tokens = vision_tokens
        num_patches = z0_tokens.size(1)
        if num_patches != self.context_length:
            raise ValueError(
                f"Expected {self.context_length} patch tokens, got {num_patches}"
            )
        expected_action_shape = (batch_size, total_actions)
        if noisy_action_tokens.shape[:2] != expected_action_shape:
            raise ValueError(
                f"Expected noisy_action_tokens shape prefix {expected_action_shape}, "
                f"got {tuple(noisy_action_tokens.shape[:2])}"
            )
        if clean_action_tokens.shape[:2] != expected_action_shape:
            raise ValueError(
                f"Expected clean_action_tokens shape prefix {expected_action_shape}, "
                f"got {tuple(clean_action_tokens.shape[:2])}"
            )

        t, task_id = self._validate_time_and_task(
            t,
            task_id,
            batch_size=batch_size,
            device=z0_tokens.device,
            dtype=z0_tokens.dtype,
        )
        t_emb = self.time_embedder(t)
        zero_t_emb = self.time_embedder(torch.zeros_like(t))
        output = self._forward_with_future_queries(
            z0_tokens,
            (noisy_action_tokens, clean_action_tokens),
            (t_emb, zero_t_emb),
            zero_t_emb,
            task_id,
            sparse_offsets,
        )
        output = LeWAMTransformerOutput(
            sample=output.sample[:, :total_actions],
            vision_hidden_states=output.vision_hidden_states,
        )
        return output if return_dict else output.to_tuple()

    def predict_action_hidden(
        self,
        state_token: torch.Tensor,
        noisy_action_tokens: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> LeWAMTransformerOutput:
        """Predict one flow-matching action velocity hidden state."""
        return self._forward_action_only(
            state_token,
            noisy_action_tokens,
            t=t,
            task_id=task_id,
        )

    def predict_future_state_hidden(
        self,
        state_token: torch.Tensor,
        clean_action_tokens: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        sparse_offsets: Optional[torch.Tensor] = None,
    ) -> LeWAMTransformerOutput:
        """Predict future-state query hidden states from clean actions."""
        if state_token.ndim != 3:
            raise ValueError(
                f"Expected state_token with shape (B, P, D), got {tuple(state_token.shape)}"
            )

        batch_size = state_token.size(0)
        total_actions = self.action_horizon
        num_patches = state_token.size(1)
        if num_patches != self.context_length:
            raise ValueError(
                f"Expected {self.context_length} patch tokens, got {num_patches}"
            )

        if clean_action_tokens.ndim != 3:
            raise ValueError(
                "Expected clean_action_tokens with shape (B, H, D), "
                f"got {tuple(clean_action_tokens.shape)}"
            )
        if clean_action_tokens.shape[:2] != (batch_size, total_actions):
            raise ValueError(
                "Expected clean_action_tokens shape prefix "
                f"{(batch_size, total_actions)}, "
                f"got {tuple(clean_action_tokens.shape[:2])}"
            )
        t = torch.zeros(
            (batch_size, 1),
            device=state_token.device,
            dtype=state_token.dtype,
        )
        t, task_id = self._validate_time_and_task(
            t,
            task_id,
            batch_size=batch_size,
            device=state_token.device,
            dtype=state_token.dtype,
        )
        zero_t_emb = self.time_embedder(torch.zeros_like(t))
        return self._forward_with_future_queries(
            state_token,
            (clean_action_tokens,),
            (zero_t_emb,),
            zero_t_emb,
            task_id,
            sparse_offsets,
        )
