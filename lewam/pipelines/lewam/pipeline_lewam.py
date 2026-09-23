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

"""LeWAM pipeline for teacher-forced training and dense action sampling."""

from dataclasses import dataclass
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from ...models.vision_encoder import VisionEncoderAdapter
from ...models.action_cache import CachedActionPredictor
from ...models.sparse_offsets import default_sparse_offsets, sample_random_sparse_offsets
from .cuda_graph import InferenceGraph, model_storage_signature

@dataclass
class LeWAMPipelineOutput(BaseOutput):
    """Output class for the LeWAM pipeline.

    Parameters
    ----------
    action : torch.Tensor
        The predicted dense action horizon of shape
        ``(B, action_horizon, action_dim)``.
    pred_next_feature : torch.Tensor, optional
        Predicted future encoder features of shape
        ``(B, num_transition, num_patches, encoder_dim)`` when requested.
    """

    action: torch.Tensor
    pred_next_feature: Optional[torch.Tensor] = None


class LeWAMPipeline(DiffusionPipeline):
    """Pipeline for action prediction over an adapted pretrained vision encoder.

    This pipeline encodes visual observations into patch-like tokens, then
    denoises dense action horizons using a Transformer-based predictor with flow-matching
    scheduling. It supports both teacher-forced training (via ``flow_match_actions`` and
    ``predict``) and stateless dense action horizon inference (via ``__call__``).

    Parameters
    ----------
    vision_encoder : nn.Module
        Frozen official Transformers I-JEPA encoder.
    predictor : nn.Module
        Transformer predictor that denoises action tokens conditioned on vision state.
    scheduler : SchedulerMixin
        Diffusers-compatible scheduler for flow-matching inference steps.
    action_dim : int
        Dimensionality of the action space.
    action_horizon : int
        Total number of dense action steps predicted per denoising pass.
    num_train_timesteps : int, optional
        Legacy checkpoint value. Must match the scheduler when provided.
    infer_steps : int
        Default number of denoising steps at inference time.
    infer_shift : float, optional
        Legacy checkpoint value. Must match the scheduler when provided.
    """

    model_cpu_offload_seq = "vision_encoder->predictor"

    def __init__(
        self,
        vision_encoder,
        predictor,
        scheduler,
        *,
        action_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
        infer_steps: int = 10,
        infer_shift: Optional[float] = None,
        encoder_type: Optional[str] = None,
    ):
        super().__init__()

        vision_encoder.eval()
        vision_encoder.requires_grad_(False)

        if action_dim is None:
            action_dim = getattr(predictor.config, "action_dim", None)
            if action_dim is None:
                action_dim = getattr(predictor, "action_dim", 0)
            action_dim = int(action_dim)
        if action_horizon is None:
            action_horizon = int(getattr(predictor, "action_horizon", 0))

        scheduler_timesteps = int(scheduler.config.num_train_timesteps)
        scheduler_shift = float(scheduler.config.shift)
        if num_train_timesteps is not None and int(num_train_timesteps) != scheduler_timesteps:
            raise ValueError("num_train_timesteps must match the scheduler configuration")
        if infer_shift is not None and float(infer_shift) != scheduler_shift:
            raise ValueError("infer_shift must match the scheduler shift")

        self.register_modules(
            vision_encoder=vision_encoder,
            predictor=predictor,
            scheduler=scheduler,
        )
        self._vision_adapter = VisionEncoderAdapter(
            vision_encoder,
            capture_layer_features=self._feature_fusion_enabled(),
        )
        spec = self._vision_adapter.spec
        if encoder_type is not None and encoder_type != spec.encoder_type:
            raise ValueError(
                f"Checkpoint expects encoder_type={encoder_type!r}, but the supplied "
                f"encoder is {spec.encoder_type!r}"
            )
        self.encoder_type = spec.encoder_type
        self.image_size = spec.image_size
        self.patch_size = spec.patch_size
        self.num_patches = spec.num_patches
        self.register_to_config(
            action_dim=action_dim,
            action_horizon=action_horizon,
            infer_steps=infer_steps,
            encoder_type=self.encoder_type,
        )

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_train_timesteps = scheduler_timesteps
        self.infer_steps = infer_steps
        self.infer_shift = scheduler_shift

        self._inference_scheduler_cache = {}
        self._inference_graph = None
        self._inference_graph_key = None
        self._inference_lock = threading.Lock()

    def _inference_sparse_offsets(self, *, random_sparse_offsets: bool, device: torch.device) -> torch.Tensor:
        sampler = sample_random_sparse_offsets if random_sparse_offsets else default_sparse_offsets
        return sampler(
            getattr(self.predictor, "action_block_size", self.action_horizon),
            getattr(self.predictor, "num_transition", 1),
            device=device,
        )

    def eval(self):
        self.vision_encoder.eval()
        self.predictor.eval()
        return self

    def train(self, mode: bool = True):
        self._clear_inference_graph()
        self.vision_encoder.eval()
        self.predictor.train(mode)
        return self

    def _clear_inference_graph(self):
        if self._inference_graph is not None:
            self._inference_graph.close()
            self._inference_graph = None
            self._inference_graph_key = None

    def to(self, *args, **kwargs):
        self._clear_inference_graph()
        return super().to(*args, **kwargs)

    def parameters(self):
        return self.predictor.parameters()

    def requires_grad_(self, requires_grad: bool = True):
        self.predictor.requires_grad_(requires_grad)
        self.vision_encoder.requires_grad_(False)
        return self

    def _feature_fusion_enabled(self) -> bool:
        return bool(
            getattr(self.predictor, "vision_feature_fusion_enabled", False)
            and getattr(self.predictor, "vision_fuser", None) is not None
        )

    def _run_vision_encoder(
        self,
        pixels: torch.Tensor,
        *,
        capture_layer_features: bool = False,
    ):
        with torch.no_grad():
            hidden_states, layer_features = self._vision_adapter.encode_patch_tokens(
                pixels,
                capture_layer_features=capture_layer_features,
            )

        if capture_layer_features and len(layer_features) != int(
            self.predictor.vision_fuser.num_layers
        ):
            raise RuntimeError(
                "Vision encoder and feature fuser layer counts do not match: "
                f"{len(layer_features)} != {self.predictor.vision_fuser.num_layers}"
            )

        return hidden_states, layer_features

    def encode(
        self,
        info: Dict[str, torch.Tensor],
        *,
        return_current_layer_features: bool = False,
    ) -> Union[
        Dict[str, torch.Tensor],
        Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        """Encode pixel observations through the configured vision adapter.

        Parameters
        ----------
        info : Dict[str, torch.Tensor]
            Dictionary containing ``"pixels"`` with shape ``(B, T, C, H, W)`` or
            ``(B, C, H, W)``. Results are stored under ``info["emb"]``.

        return_current_layer_features : bool, optional
            Return the exact current-frame encoder-layer tensors consumed by the live
            fuser. This is an opt-in training API used to apply frozen reference
            fusers without running I-JEPA a second time. It is only valid when feature
            fusion is enabled.

        Returns
        -------
        Dict[str, torch.Tensor] or tuple
            By default, the input dictionary augmented with ``"emb"`` of shape
            ``(B, T, N, D)``. With feature fusion enabled, every frame contains a
            frozen final-layer feature except the current frame, which uses the
            trainable fuser. When ``return_current_layer_features=True``,
            also returns a tuple containing the current-frame feature from every
            encoder layer. The tuple only copies tensor references and does not
            stack or clone the layer features.
        """
        feature_fusion_enabled = self._feature_fusion_enabled()
        if return_current_layer_features and not feature_fusion_enabled:
            raise ValueError(
                "return_current_layer_features=True requires feature fusion to be enabled"
            )

        pixels = info["pixels"].float()
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        if pixels.ndim != 5:
            raise ValueError(f"Expected pixels with shape (B, T, C, H, W), got {tuple(pixels.shape)}")

        batch_size, num_frames = pixels.shape[:2]
        # Flatten batch and time: (B, T, C, H, W) -> (B*T, C, H, W)
        flat_pixels = pixels.reshape(batch_size * num_frames, *pixels.shape[2:])
        flat_pixels = self._vision_adapter.preprocess(flat_pixels)

        if feature_fusion_enabled:
            frame_pixels = flat_pixels.view(batch_size, num_frames, *flat_pixels.shape[1:])
            _, layer_features = self._run_vision_encoder(
                frame_pixels[:, 0],
                capture_layer_features=True,
            )
            current_features = self.predictor.vision_fuser(layer_features)
            current_layer_features = tuple(layer_features)
            token_count = current_features.size(1)
            hidden_states = current_features.unsqueeze(1)
            if num_frames > 1:
                future_pixels = frame_pixels[:, 1:].reshape(-1, *flat_pixels.shape[1:])
                final_patch_tokens, _ = self._run_vision_encoder(future_pixels)
                future_features = F.layer_norm(
                    final_patch_tokens, (final_patch_tokens.size(-1),),
                ).view(batch_size, num_frames - 1, token_count, -1)
                hidden_states = torch.cat((hidden_states, future_features), dim=1)
        else:
            final_patch_tokens, _ = self._run_vision_encoder(flat_pixels)
            hidden_states = F.layer_norm(
                final_patch_tokens,
                (final_patch_tokens.size(-1),),
            )
            token_count = hidden_states.size(1)
            hidden_states = hidden_states.view(batch_size, num_frames, token_count, -1)

        if token_count != self.num_patches:
            raise RuntimeError(
                f"Expected {self.num_patches} {self.encoder_type} patch tokens, "
                f"got {token_count}"
            )
        info["emb"] = hidden_states
        if return_current_layer_features:
            return info, tuple(current_layer_features)
        return info

    def flow_match_actions(
        self,
        action: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise_Q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct action-only flow-matching noisy inputs and velocity targets.

        Parameters
        ----------
        action : torch.Tensor
            Ground-truth actions of shape ``(B, K, A)``.
        t : torch.Tensor, optional
            Timestep values of shape ``(B, 1)``. Sampled uniformly if not provided.
        noise_Q : torch.Tensor, optional
            Noise tensor of same shape as ``action``. Sampled from N(0,1) if not provided.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(noisy_Q0, target_Q, t)`` -- the noisy actions, velocity targets, and timesteps.
        """
        action = torch.nan_to_num(action, 0.0)
        if action.ndim != 3:
            raise ValueError(f"Expected action with shape (B, K, A), got {tuple(action.shape)}")

        batch_size = action.size(0)
        total_actions = action.size(1)
        if total_actions != self.action_horizon:
            raise ValueError(f"Expected action horizon length {self.action_horizon}, got {total_actions}")
        device = action.device
        dtype = action.dtype
        if noise_Q is None:
            noise_Q = torch.randn_like(action)

        if t is None:
            t = torch.rand((batch_size, 1), device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.ndim != 2 or t.size(0) != batch_size or t.size(1) != 1:
            raise ValueError(f"Expected t shape ({batch_size}, 1), got {tuple(t.shape)}")
        if bool((t > 1).any().item()):
            t = t / float(self.num_train_timesteps)
        t = t.clamp(0.0, 1.0)

        t_expand = t.expand(-1, total_actions).unsqueeze(-1)
        noisy_Q0 = t_expand * noise_Q + (1 - t_expand) * action
        target_Q = noise_Q - action
        return noisy_Q0, target_Q, t

    def predict(
        self,
        projected_state: torch.Tensor,
        noisy_actions: torch.Tensor,
        clean_actions: Optional[torch.Tensor],
        t: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        sparse_offsets: Optional[torch.Tensor] = None,
        *,
        predict_next_feature: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced training prediction for action flow matching.

        Parameters
        ----------
        projected_state : torch.Tensor
            Projected encoder state of shape ``(B, T, N, D)`` or ``(B, N, D)``.
        noisy_actions : torch.Tensor
            Noisy action inputs of shape ``(B, K, A)``.
        clean_actions : torch.Tensor, optional
            Clean teacher-forced action inputs of shape ``(B, K, A)``. Required when
            ``predict_next_feature=True`` and ignored by the action-only path.
        t : torch.Tensor
            Normalized timesteps of shape ``(B, 1)``.
        task_id : torch.Tensor, optional
            Task identifier tensor for multi-task conditioning.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys ``"tokens"``, ``"attn_mask"``, ``"action_hidden"``,
            ``"vision_hidden"``, ``"pred_diff_Q"``, and ``"pred_next_feature"``.
        """
        if projected_state.ndim == 3:
            projected_state = projected_state.unsqueeze(1)
        if projected_state.ndim != 4:
            raise ValueError(
                "Expected projected_state with shape (B, T, N, D) or (B, N, D), "
                f"got {tuple(projected_state.shape)}"
            )
        t = t * float(self.num_train_timesteps)

        z0 = projected_state[:, 0]
        noisy_action_tokens = self.predictor.action_proj(noisy_actions)
        clean_action_tokens = None
        if predict_next_feature:
            if clean_actions is None:
                raise ValueError("clean_actions is required when predict_next_feature=True")
            clean_action_tokens = self.predictor.action_proj(clean_actions)

        output = self.predictor(
            z0,
            noisy_action_tokens,
            clean_action_tokens,
            t,
            task_id=task_id,
            sparse_offsets=sparse_offsets,
            predict_next_feature=predict_next_feature,
        )
        pred_diff_Q = self.predictor.action_head(output.sample)
        pred_next_feature = None
        if predict_next_feature:
            pred_next_feature = self.predictor.next_feature_head(output.vision_hidden_states)

        return {
            "action_hidden": output.sample,
            "vision_hidden": output.vision_hidden_states,
            "pred_diff_Q": pred_diff_Q,
            "pred_next_feature": pred_next_feature,
        }

    def predict_future_states_from_actions(
        self,
        projected_state: torch.Tensor,
        clean_actions: torch.Tensor,
        *,
        task_id: Optional[torch.Tensor] = None,
        sparse_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future encoder features from current state and a clean action horizon."""
        if projected_state.ndim == 4:
            projected_state = projected_state[:, 0]
        if projected_state.ndim != 3:
            raise ValueError(
                "Expected projected_state with shape (B, P, D) or (B, T, P, D), "
                f"got {tuple(projected_state.shape)}"
            )
        if clean_actions.ndim != 3:
            raise ValueError(f"Expected clean_actions with shape (B, H, A), got {tuple(clean_actions.shape)}")
        clean_action_tokens = self.predictor.action_proj(clean_actions)
        output = self.predictor.predict_future_state_hidden(
            projected_state,
            task_id=task_id,
            clean_action_tokens=clean_action_tokens,
            sparse_offsets=sparse_offsets,
        )
        return self.predictor.next_feature_head(output.vision_hidden_states)

    def prepare_latents(
        self,
        batch_size: int,
        horizon: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare initial action latents for diffusion inference."""
        shape = (batch_size, horizon, self.action_dim)
        if latents is not None:
            if tuple(latents.shape) != shape:
                raise ValueError(f"Unexpected latents shape, got {tuple(latents.shape)}, expected {shape}.")
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def _task_ids(self, task_id, batch_size, device):
        if task_id is None:
            raise ValueError("task_id is required for task-conditioned prediction")
        task_id = torch.as_tensor(task_id, device=device, dtype=torch.long)
        if task_id.ndim == 0 or (task_id.ndim == 1 and task_id.numel() == 1):
            task_id = task_id.reshape(1).expand(batch_size)
        if task_id.shape != (batch_size,):
            raise ValueError(f"Expected task_id shape {(batch_size,)}, got {tuple(task_id.shape)}")
        return task_id

    @torch.no_grad()
    def _run_cached_inference(
        self, observation, task_id, action, *, infer_steps, shift,
        action_predictor, encode_image=False, return_state=False,
    ):
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent or reentrant inference on one pipeline is not supported")
        try:
            inputs = (observation, task_id, action)
            device = observation.device
            infer_steps = int(infer_steps or self.infer_steps)
            infer_shift = float(self.infer_shift if shift is None else shift)
            if infer_steps < 1:
                raise ValueError("infer_steps must be positive")
            signature = (
                encode_image, tuple((tuple(t.shape), t.dtype, t.device) for t in inputs),
                infer_steps, infer_shift, self.num_train_timesteps,
                model_storage_signature(action_predictor),
                model_storage_signature(self.vision_encoder) if encode_image else None,
                repr(dict(self.scheduler.config)),
                torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type),
            )
            if device.type == "cuda" and self._inference_graph_key == signature:
                return self._inference_graph.replay(inputs, return_state=return_state)
            # Keep one graph, releasing its pool when shape/schedule/model changes.
            self._clear_inference_graph()
            cached = CachedActionPredictor(action_predictor)
            scheduler_config = dict(self.scheduler.config)
            scheduler_config["shift"] = infer_shift
            scheduler = self.scheduler.__class__.from_config(scheduler_config)
            scheduler.set_timesteps(num_inference_steps=infer_steps, device=device)
            timesteps = tuple(scheduler.timesteps.unbind())

            def forward(observation, task_id, action):
                if encode_image:
                    emb = self.encode({"pixels": observation})["emb"]
                    state = action_predictor.vision_proj(emb[:, -1])
                else:
                    state = observation
                cache = cached.prefill(state, task_id)
                # The fixed schedule begins at zero; avoid GPU-to-host index lookup.
                scheduler._step_index = 0
                for timestep in timesteps:
                    normalized = timestep.to(dtype=state.dtype) / float(self.num_train_timesteps)
                    time = normalized.expand(state.shape[0]) * float(self.num_train_timesteps)
                    hidden = cached.predict(action_predictor.action_proj(action), time, cache)
                    velocity = action_predictor.action_head(hidden)
                    action = scheduler.step(velocity, timestep, action, return_dict=True).prev_sample
                return action, state

            if device.type == "cuda":
                self._inference_graph = InferenceGraph(forward, inputs, (cached, scheduler))
                self._inference_graph_key = signature
                return self._inference_graph.replay(inputs, return_state=return_state)
            action, state = forward(*inputs)
            return action, state if return_state else None
        finally:
            self._inference_lock.release()

    @torch.no_grad()
    def sample_action_horizon(
        self,
        state_token: torch.Tensor,
        infer_steps: Optional[int] = None,
        task_id: Optional[torch.Tensor] = None,
        shift: Optional[float] = None,
        random_sparse_offsets: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        predict_future_states: bool = False,
        return_output: bool = False,
        predictor: Optional[torch.nn.Module] = None,
    ) -> Union[torch.Tensor, LeWAMPipelineOutput]:
        """Denoise a full dense action horizon conditioned on current fused state.

        Parameters
        ----------
        state_token : torch.Tensor
            Fused state tokens of shape ``(B, N, D)`` or ``(B, D)``.
        infer_steps : int, optional
            Number of denoising steps. Defaults to pipeline config value.
        task_id : torch.Tensor, optional
            Task identifier for multi-task conditioning.
        shift : float, optional
            Scheduler shift override. Defaults to pipeline config value.
        random_sparse_offsets : bool
            If true, sample inference sparse offsets from the same distribution used by training.
            If false, use the default fixed offsets.
        generator : torch.Generator or List[torch.Generator], optional
            Random generator(s) used to sample initial action latents when ``latents`` is not provided.
        latents : torch.Tensor, optional
            Pre-sampled initial action latents of shape ``(B, action_horizon, action_dim)``.
        predictor : torch.nn.Module, optional
            Predictor used for action denoising. Defaults to the pipeline predictor.

        Returns
        -------
        torch.Tensor or LeWAMPipelineOutput
            By default returns the denoised action horizon tensor for backward compatibility.
            When ``return_output`` or ``predict_future_states`` is true, returns an output
            object with ``action`` and optional ``pred_next_feature``.
        """
        if state_token.ndim == 2:
            state_token = state_token.unsqueeze(1)
        if state_token.ndim != 3:
            raise ValueError(f"Expected state_token with shape (B, N, D), got {tuple(state_token.shape)}")
        action_predictor = self.predictor if predictor is None else predictor
        if predict_future_states and action_predictor is not self.predictor:
            raise ValueError("A predictor override is only supported for action-only sampling")
        batch_size = state_token.size(0)
        horizon = self.action_horizon
        device = state_token.device
        dtype = state_token.dtype
        if state_token.shape[1] != action_predictor.context_length:
            raise ValueError(f"Expected {action_predictor.context_length} state tokens")
        task_id = self._task_ids(task_id, batch_size, device)
        action = self.prepare_latents(
            batch_size=batch_size,
            horizon=horizon,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        infer_steps = int(infer_steps or self.infer_steps)
        infer_shift = float(self.infer_shift if shift is None else shift)
        cache_key = (infer_steps, infer_shift)
        scheduler = self._inference_scheduler_cache.get(cache_key)
        if scheduler is None:
            scheduler_config = dict(self.scheduler.config)
            scheduler_config["shift"] = infer_shift
            scheduler = self.scheduler.__class__.from_config(scheduler_config)
            self._inference_scheduler_cache[cache_key] = scheduler
        scheduler.set_timesteps(num_inference_steps=infer_steps, device=device)

        if not action_predictor.training:
            action, _ = self._run_cached_inference(
                state_token, task_id, action, infer_steps=infer_steps, shift=infer_shift,
                action_predictor=action_predictor,
            )

        # Training-mode sampling preserves dropout semantics; deployment is eval-mode.
        for timestep in scheduler.timesteps if action_predictor.training else ():
            normalized_timestep = timestep.to(device=device, dtype=dtype) / float(self.num_train_timesteps)
            curr_t = normalized_timestep.expand(batch_size)
            noisy_action_tokens = action_predictor.action_proj(action)
            output = action_predictor.predict_action_hidden(
                state_token,
                noisy_action_tokens,
                curr_t * float(self.num_train_timesteps),
                task_id=task_id,
            )
            pred_diff_Q = action_predictor.action_head(output.sample)
            action = scheduler.step(
                pred_diff_Q,
                timestep,
                action,
                return_dict=True,
            ).prev_sample

        pred_next_feature = None
        if predict_future_states:
            sparse_offsets = self._inference_sparse_offsets(
                random_sparse_offsets=random_sparse_offsets,
                device=state_token.device,
            )
            pred_next_feature = self.predict_future_states_from_actions(
                state_token,
                action,
                task_id=task_id,
                sparse_offsets=sparse_offsets,
            )
        if return_output or predict_future_states:
            return LeWAMPipelineOutput(action=action, pred_next_feature=pred_next_feature)
        return action

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        *,
        task_id: Optional[Any] = None,
        infer_steps: Optional[int] = None,
        shift: Optional[float] = None,
        random_sparse_offsets: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        predict_future_states: bool = False,
        return_dict: bool = True,
    ) -> Union[LeWAMPipelineOutput, tuple]:
        """Run stateless inference and return a full dense action horizon.

        Parameters
        ----------
        image : torch.Tensor
            Current RGB observations with shape ``(B, 3, H, W)`` and values
            in ``[0, 1]``, prepared by the RobotWin policy adapter.
        task_id : int, array-like, or torch.Tensor
            Task identifier for task-conditioned action denoising.
        infer_steps : int, optional
            Number of denoising steps. Defaults to pipeline config value.
        shift : float, optional
            Scheduler shift override. Defaults to pipeline config value.
        random_sparse_offsets : bool
            If true, sample inference sparse offsets from the training distribution.
        generator : torch.Generator or List[torch.Generator], optional
            Random generator(s) used to sample initial action latents when ``latents`` is not provided.
        latents : torch.Tensor, optional
            Pre-sampled initial action latents of shape ``(B, action_horizon, action_dim)``.
        return_dict : bool
            Return ``LeWAMPipelineOutput`` when true, otherwise a tuple of its non-null fields.

        Returns
        -------
        LeWAMPipelineOutput
            A dataclass containing ``action`` with shape
            ``(B, action_horizon, action_dim)``.
        """
        if task_id is None:
            raise ValueError("task_id is required for task-conditioned prediction")

        model_param = next(self.parameters())
        device = model_param.device
        model_dtype = model_param.dtype

        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Expected floating-point RGB image with shape (B, 3, H, W) in [0, 1]")
        pixels = image.to(device=device, dtype=model_dtype)

        batch_size = pixels.size(0)

        task_id = self._task_ids(task_id, batch_size, device)

        if not self.predictor.training:
            action = self.prepare_latents(
                batch_size, self.action_horizon, model_dtype, device,
                generator=generator, latents=latents,
            )
            action, projected_state = self._run_cached_inference(
                pixels, task_id, action, infer_steps=infer_steps, shift=shift,
                action_predictor=self.predictor, encode_image=True,
                return_state=predict_future_states,
            )
            future = None
            if predict_future_states:
                offsets = self._inference_sparse_offsets(
                    random_sparse_offsets=random_sparse_offsets, device=device,
                )
                future = self.predict_future_states_from_actions(
                    projected_state, action, task_id=task_id, sparse_offsets=offsets,
                )
            output = LeWAMPipelineOutput(action=action, pred_next_feature=future)
            return output if return_dict else output.to_tuple()

        state_emb = self.encode({"pixels": pixels})["emb"]
        projected_state = self.predictor.vision_proj(state_emb[:, -1])
        output = self.sample_action_horizon(
            projected_state,
            infer_steps=infer_steps,
            task_id=task_id,
            shift=shift,
            random_sparse_offsets=random_sparse_offsets,
            generator=generator,
            latents=latents,
            predict_future_states=predict_future_states,
            return_output=True,
        )
        return output if return_dict else output.to_tuple()
