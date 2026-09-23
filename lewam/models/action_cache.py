"""Per-request visual keys/values for action-only LeWAM inference."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .module import modulate
from .rope import build_rope_position_ids


@dataclass
class VisionCache:
    layers: list
    task_embedding: torch.Tensor


class CachedActionPredictor:
    """Reuse visual K/V across denoising steps, never across observations."""

    def __init__(self, predictor):
        self.model = predictor
        p = next(predictor.parameters())
        self.positions = build_rope_position_ids(
            num_patches=predictor.context_length,
            action_horizon=predictor.action_horizon,
            action_block_size=predictor.action_block_size,
            num_transition=predictor.num_transition,
            grid_h=predictor.grid_h,
            grid_w=predictor.grid_w,
            include_query_tokens=False,
            device=p.device,
        )
        mask = predictor.build_inference_attention_mask(
            num_patches=predictor.context_length, device=p.device
        )
        n = predictor.context_length
        assert not mask[:n, n:].any(), "Visual tokens must not attend to noisy actions"
        self.vision_mask = mask[:n, :n][None, None]
        self.action_mask = mask[n:][None, None]
        self.vision_rope = self._rope_constants(self.positions[:n], p.dtype)
        self.action_rope = self._rope_constants(self.positions[n:], p.dtype)

    def _rope_constants(self, positions, dtype):
        rope = self.model.rope
        cos, sin = [], []
        for axis, inv_freq in enumerate(
            (rope.inv_freq_t, rope.inv_freq_h, rope.inv_freq_w)
        ):
            freqs = torch.outer(positions[:, axis].to(inv_freq.dtype), inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos.append(emb.cos().to(dtype))
            sin.append(emb.sin().to(dtype))
        return torch.cat(cos, dim=-1)[None, None], torch.cat(sin, dim=-1)[None, None]

    def _rotate(self, x, constants):
        rope = self.model.rope
        parts = x.split((rope.dim_t, rope.dim_h, rope.dim_w), dim=-1)
        rotated = []
        for part in parts:
            a, b = part.chunk(2, dim=-1)
            rotated.append(torch.cat((-b, a), dim=-1))
        cos, sin = constants
        return x * cos + torch.cat(rotated, dim=-1) * sin

    def _qkv(self, block, x, shift, scale, positions):
        attn = block.attn
        x = attn.norm(modulate(block.norm1(x), shift, scale))
        q, k, v = attn.to_qkv(x).chunk(3, dim=-1)
        q, k = attn.norm_q(q), attn.norm_k(k)
        b, n, _ = x.shape
        q, k, v = (
            z.view(b, n, attn.heads, attn.dim_head).permute(0, 2, 1, 3)
            for z in (q, k, v)
        )
        return self._rotate(q, positions), self._rotate(k, positions), v

    @staticmethod
    def _attention(block, q, k, v, mask):
        length = q.shape[2]
        if length != k.shape[2]:
            # Preserve the full-query SDPA/GEMM numerical path on PyTorch 2.6.
            # Dummy query rows cannot affect real action rows and are discarded.
            padding = k.shape[2] - length
            q = F.pad(q, (0, 0, padding, 0))
            mask = F.pad(mask, (0, 0, padding, 0), value=True)
        z = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        z = block.attn.to_out(z.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1))
        return z[:, -length:]

    @staticmethod
    def _action_mlp(block, x):
        # On this SM90/BF16 released architecture, small-M FFN down-projection
        # selects a different reduction kernel. Pad only this linear's rows to
        # retain the full-sequence numerical path; discard the dummy rows.
        for index, layer in enumerate(block.mlp.net):
            if index == 3 and x.dtype == torch.bfloat16 and x.shape[1] < 96:
                padding = 96 - x.shape[1]
                x = layer(F.pad(x, (0, 0, padding, 0)))[:, padding:]
            else:
                x = layer(x)
        return x

    @torch.no_grad()
    def prefill(self, vision_tokens, task_id):
        model = self.model
        if model.training:
            raise RuntimeError("KV cache is inference-only; call eval()")
        b = vision_tokens.shape[0]
        task_embedding = model.task_embed(task_id).unsqueeze(1)
        cond = (
            model.time_embedder(
                torch.zeros(
                    b, 1, device=vision_tokens.device, dtype=vision_tokens.dtype
                )
            )
            + task_embedding
        )
        cond = cond.expand(-1, vision_tokens.shape[1], -1)
        x = vision_tokens
        layers = []
        for block in model.transformer.layers:
            shift, scale, gate, shift_ff, scale_ff, gate_ff = block.split_modulation(
                cond
            )
            q, k, v = self._qkv(block, x, shift, scale, self.vision_rope)
            layers.append((k, v))
            x = x + gate * self._attention(block, q, k, v, self.vision_mask)
            x = x + gate_ff * block.mlp(modulate(block.norm2(x), shift_ff, scale_ff))
        return VisionCache(layers, task_embedding)

    @torch.no_grad()
    def predict(self, noisy_action_tokens, timestep, cache):
        model = self.model
        cond = model.time_embedder(timestep.reshape(-1, 1)) + cache.task_embedding
        cond = cond.expand(-1, noisy_action_tokens.shape[1], -1)
        x = noisy_action_tokens
        for block, (vision_k, vision_v) in zip(model.transformer.layers, cache.layers):
            shift, scale, gate, shift_ff, scale_ff, gate_ff = block.split_modulation(
                cond
            )
            q, k, v = self._qkv(block, x, shift, scale, self.action_rope)
            k = torch.cat((vision_k, k), dim=2)
            v = torch.cat((vision_v, v), dim=2)
            x = x + gate * self._attention(block, q, k, v, self.action_mask)
            x = x + gate_ff * self._action_mlp(
                block, modulate(block.norm2(x), shift_ff, scale_ff)
            )
        return model.transformer.norm(x)
