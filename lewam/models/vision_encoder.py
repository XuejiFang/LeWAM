import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VisionEncoderSpec:
    """Model-specific metadata needed by the shared vision adapter."""

    encoder_type: str
    image_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    grid_size: Tuple[int, int]
    hidden_size: int
    num_hidden_layers: int
    image_mean: Tuple[float, float, float]
    image_std: Tuple[float, float, float]

    @property
    def num_patches(self) -> int:
        return self.grid_size[0] * self.grid_size[1]


def _normalize_pair(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"Expected an int or pair, got {value!r}")
    return int(value[0]), int(value[1])


def get_vision_encoder_spec(encoder: nn.Module) -> VisionEncoderSpec:
    """Read the patch-token contract from the frozen I-JEPA encoder."""
    config = encoder.config
    if config.model_type != "ijepa":
        raise ValueError(f"Expected I-JEPA, got {config.model_type!r}")
    image_size = _normalize_pair(config.image_size)
    patch_size = _normalize_pair(config.patch_size)
    return VisionEncoderSpec(
        encoder_type="ijepa",
        image_size=image_size,
        patch_size=patch_size,
        grid_size=(image_size[0] // patch_size[0], image_size[1] // patch_size[1]),
        hidden_size=int(config.hidden_size),
        num_hidden_layers=int(config.num_hidden_layers),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )


def load_vision_encoder(
    model_name_or_path,
    *,
    local_files_only: bool = True,
) -> Tuple[nn.Module, VisionEncoderSpec]:
    """Load the pretrained I-JEPA encoder through its official model class."""
    from transformers import IJepaModel

    encoder = IJepaModel.from_pretrained(
        str(model_name_or_path), local_files_only=local_files_only,
    )
    return encoder, get_vision_encoder_spec(encoder)


class StaticChannelwiseLayerFusion(nn.Module):
    """Fuse encoder layers with input-independent per-channel weights."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        initial_final_weight: float = 0.8,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {self.num_layers}")
        if not 0.0 < initial_final_weight < 1.0:
            raise ValueError(
                "initial_final_weight must be between 0 and 1, "
                f"got {initial_final_weight}"
            )

        self.pre_fusion_norm = nn.LayerNorm(
            self.hidden_dim,
            eps=float(norm_eps),
            elementwise_affine=False,
        )
        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers, self.hidden_dim))
        final_logit = math.log(
            float(initial_final_weight)
            * (self.num_layers - 1)
            / (1.0 - float(initial_final_weight))
        )
        with torch.no_grad():
            self.layer_logits[-1].fill_(final_logit)
        self.post_fusion_norm = nn.LayerNorm(
            self.hidden_dim,
            eps=float(norm_eps),
        )

    def forward(self, layer_features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(layer_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer features, got {len(layer_features)}"
            )

        reference_shape = layer_features[0].shape
        weights = F.softmax(self.layer_logits.float(), dim=0).to(
            dtype=layer_features[0].dtype
        )
        weight_shape = (*([1] * (layer_features[0].ndim - 1)), self.hidden_dim)
        fused = torch.zeros_like(layer_features[0])

        for index, feature in enumerate(layer_features):
            if feature.shape != reference_shape:
                raise ValueError(
                    "All layer features must have the same shape; "
                    f"layer 0 has {tuple(reference_shape)}, layer {index} has {tuple(feature.shape)}"
                )
            if feature.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"Expected hidden dimension {self.hidden_dim}, got {feature.size(-1)} "
                    f"for layer {index}"
                )
            normalized_feature = self.pre_fusion_norm(feature)
            fused = fused + normalized_feature * weights[index].view(weight_shape)

        return self.post_fusion_norm(fused)

    @torch.no_grad()
    def average_layer_weights(self) -> torch.Tensor:
        """Return a ``[num_layers]`` summary for logging and inspection."""
        return F.softmax(self.layer_logits.float(), dim=0).mean(dim=-1)


class IJepaFeatureEncoder(nn.Module):
    """Wrap I-JEPA and optionally expose every encoder block output."""

    def __init__(self, encoder: nn.Module, capture_layer_features: bool = False):
        super().__init__()
        self.encoder = encoder
        self.num_layers = int(encoder.config.num_hidden_layers)
        self._captured_features: Optional[List[Optional[torch.Tensor]]] = None
        self._hook_handles = []

        if capture_layer_features:
            self._register_layer_hooks()

    @property
    def captures_layer_features(self) -> bool:
        return bool(self._hook_handles)

    def _register_layer_hooks(self) -> None:
        if self._hook_handles:
            return

        encoder_layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if encoder_layers is None:
            raise RuntimeError(
                "Vision feature fusion requires vision_encoder.encoder.layer"
            )
        if len(encoder_layers) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} encoder layers, got {len(encoder_layers)}"
            )

        for layer_index, layer in enumerate(encoder_layers):
            def capture_output(_module, _inputs, output, index=layer_index):
                if self._captured_features is not None:
                    self._captured_features[index] = (
                        output[0] if isinstance(output, tuple) else output
                    )

            self._hook_handles.append(layer.register_forward_hook(capture_output))

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        capture_layer_features: bool = False,
        interpolate_pos_encoding: bool = True,
    ) -> Tuple[object, List[torch.Tensor]]:
        if capture_layer_features and not self.captures_layer_features:
            raise RuntimeError(
                "Layer feature capture was not enabled when IJepaFeatureEncoder was created"
            )
        if self._captured_features is not None:
            raise RuntimeError("Concurrent or reentrant I-JEPA feature capture is not supported")

        self._captured_features = (
            [None] * self.num_layers if capture_layer_features else None
        )
        try:
            output = self.encoder(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
            if not capture_layer_features:
                return output, []

            captured_features = self._captured_features
            if captured_features is None or any(
                feature is None for feature in captured_features
            ):
                captured_count = 0 if captured_features is None else sum(
                    feature is not None for feature in captured_features
                )
                raise RuntimeError(
                    f"Expected {self.num_layers} encoder layer features, got {captured_count}"
                )

            layer_features = list(captured_features)
            # Match the normal I-JEPA output path for the final layer.
            layer_features[-1] = output.last_hidden_state
            return output, layer_features
        finally:
            self._captured_features = None


class VisionEncoderAdapter:
    """Normalize RGB tensors and extract I-JEPA patch tokens."""

    def __init__(self, encoder: nn.Module, capture_layer_features: bool = False):
        self.encoder = encoder
        self.spec = get_vision_encoder_spec(encoder)
        self._ijepa_encoder = IJepaFeatureEncoder(
            encoder, capture_layer_features=capture_layer_features,
        )
        self._normalization = None

    def preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float().clamp(0.0, 1.0)
        expected_dtype = next(self.encoder.parameters()).dtype
        # Preserve the pretrained numerical path: normalize and cast before resize.
        # Prepare constants once, before CUDA Graph capture; keep them in FP32.
        if self._normalization is None or self._normalization[0].device != pixels.device:
            self._normalization = (
                pixels.new_tensor(self.spec.image_mean).view(1, 3, 1, 1),
                pixels.new_tensor(self.spec.image_std).view(1, 3, 1, 1),
            )
        mean, std = self._normalization
        pixels = ((pixels - mean) / std).to(dtype=expected_dtype)
        if pixels.shape[-2:] != self.spec.image_size:
            pixels = F.interpolate(
                pixels, size=self.spec.image_size, mode="bilinear",
                align_corners=False, antialias=True,
            )
        return pixels

    def encode_patch_tokens(
        self,
        pixels: torch.Tensor,
        *,
        capture_layer_features: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output, layer_features = self._ijepa_encoder(
            pixels, capture_layer_features=capture_layer_features,
        )
        return output.last_hidden_state, layer_features
