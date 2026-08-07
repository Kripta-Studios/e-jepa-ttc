"""Oracle box-pseudoflow target audit helpers for Object Event TTC v4.21."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ObjectEventV421AuditConfig:
    map_size: int = 16
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.map_size < 4:
            raise ValueError("map_size must be at least 4")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


def box_scale_proxies(boxes_xyxy: torch.Tensor, *, first_index: int = 1, second_index: int = 2, epsilon: float = 1.0e-6) -> dict[str, np.ndarray]:
    """Return train/eval-only geometric proxies; boxes are never forward features."""
    if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must be [B,T,4]")
    first = boxes_xyxy[:, first_index].float()
    second = boxes_xyxy[:, second_index].float()
    w1 = (first[:, 2] - first[:, 0]).clamp_min(epsilon)
    h1 = (first[:, 3] - first[:, 1]).clamp_min(epsilon)
    w2 = (second[:, 2] - second[:, 0]).clamp_min(epsilon)
    h2 = (second[:, 3] - second[:, 1]).clamp_min(epsilon)
    log_w = torch.log(w2 / w1)
    log_h = torch.log(h2 / h1)
    cx1 = 0.5 * (first[:, 0] + first[:, 2])
    cy1 = 0.5 * (first[:, 1] + first[:, 3])
    cx2 = 0.5 * (second[:, 0] + second[:, 2])
    cy2 = 0.5 * (second[:, 1] + second[:, 3])
    diagonal = torch.sqrt(w1.square() + h1.square()).clamp_min(epsilon)
    translation = torch.sqrt((cx2 - cx1).square() + (cy2 - cy1).square()) / diagonal
    values = {
        "box_log_width_ratio": log_w,
        "box_log_height_ratio": log_h,
        "box_geometric_log_scale": 0.5 * (log_w + log_h),
        "box_log_scale_anisotropy": (log_w - log_h).abs(),
        "box_normalized_translation": translation,
    }
    return {name: value.detach().cpu().numpy().astype(np.float64) for name, value in values.items()}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def train_orientation(score: np.ndarray, target: np.ndarray) -> float:
    return 1.0 if pearson(score, target) >= 0.0 else -1.0


__all__ = ["ObjectEventV421AuditConfig", "box_scale_proxies", "pearson", "train_orientation"]
