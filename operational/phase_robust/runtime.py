"""Load frozen fusion weights and expose tensor and causal-history inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read
from operational.phase_robust.geometry import expansion_features
from operational.phase_robust.model import QualityFusion


class FusionEnsemble(nn.Module):
    """Fixed arithmetic mean of all three seeds; no best-seed selection."""

    def __init__(self, models: list[QualityFusion]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        values = [model(native, geometry, quality) for model in self.models]
        combined = torch.stack(values).mean(0)
        base = native[:, 0]
        usable = (
            torch.isfinite(geometry)
            & torch.isfinite(quality).all(1)
            & (torch.sign(geometry) == torch.sign(base))
        )
        return torch.where(torch.isfinite(base) & ~usable, base, combined)


def load_fusion(root: Path) -> FusionEnsemble:
    """Strictly bind all selected checkpoints to the immutable final freeze."""
    freeze = read(root / "FINAL_FREEZE.json")
    if file_sha(root / "SELECTION.json") != freeze["selection_sha256"]:
        raise ValueError("Changed fusion selection")
    if file_sha(root / "CONFIG.json") != freeze["config_sha256"]:
        raise ValueError("Changed fusion configuration")
    models = []
    for seed in (7, 13, 21):
        path = root / "final" / str(seed) / "checkpoint.pt"
        if file_sha(path) != freeze["checkpoints"][str(seed)]:
            raise ValueError("Changed fusion checkpoint")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["seed"] != seed or payload["mode"] != freeze["selected"]:
            raise ValueError("Fusion identity mismatch")
        model = QualityFusion(payload["mode"])
        model.load_state_dict(payload["state_dict"], strict=True)
        models.append(model.eval())
    return FusionEnsemble(models).eval()


def predict_history(
    model: FusionEnsemble, native: np.ndarray, boxes: np.ndarray, clocks: np.ndarray, query_us: int
) -> tuple[float, dict]:
    """Fuse [F0, RGB context, event context] with only <=query bbox observations."""
    if np.asarray(native).shape != (3,):
        raise ValueError("Expected exactly three native TTC estimates")
    physical = expansion_features(boxes, clocks, query_us)
    geometry = float(physical["ttc"]) if physical["valid"] else float("nan")
    with torch.inference_mode():
        result = model(
            torch.tensor(native, dtype=torch.float32)[None],
            torch.tensor([geometry], dtype=torch.float32),
            torch.tensor([physical["quality"]], dtype=torch.float32),
        )
    return float(result[0]), physical
