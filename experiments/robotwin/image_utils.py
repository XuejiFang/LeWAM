"""Image and action-statistics helpers used by RoboTwin inference."""

import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF


def load_action_stats(stats_path: Path):
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing RoboTwin stats file: {stats_path}")
    if stats_path.suffix == ".npz":
        stats = np.load(stats_path)
        return stats["action_mean"].astype(np.float32), stats["action_std"].astype(np.float32)
    if stats_path.suffix == ".json":
        stats = json.loads(stats_path.read_text())["action"]["default"]
        return (
            np.asarray(stats["global_mean"], dtype=np.float32),
            np.asarray(stats["global_std"], dtype=np.float32),
        )
    raise ValueError(f"Unsupported RoboTwin stats format: {stats_path}")


def compose_robotwin_triplet(top: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Arrange three 4:3 camera images in the 384 x 320 model input."""
    cameras = (top, left, right)
    if any(image.ndim != 4 or image.shape[-1] * 3 != image.shape[-2] * 4 for image in cameras):
        raise ValueError("Expected three camera tensors shaped (T, C, H, W) with 4:3 aspect ratio.")
    top = TF.resize(top, [256, 320], antialias=True)
    left = TF.resize(left, [128, 160], antialias=True)
    right = TF.resize(right, [128, 160], antialias=True)
    return torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
