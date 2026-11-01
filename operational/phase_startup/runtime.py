"""Strictly bound startup routers and causal sensor-only inference."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read
from operational.phase_startup.geometry import startup_geometry
from operational.phase_startup.model import ExpertRouter


class RouterEnsemble(nn.Module):
    """All three seeds, with bitwise original-model fallback without history."""

    def __init__(self, models: list[ExpertRouter]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        mean = torch.stack([model(native, geometry, quality) for model in self.models]).mean(0)
        history = (
            torch.isfinite(quality[:, 9])
            & (quality[:, 9] > 0)
            & torch.isfinite(quality[:, 11])
            & (quality[:, 11] <= 0.15)
        )
        return torch.where(history, mean, native[:, 0])


def load_router(root: Path) -> RouterEnsemble:
    """Reject changed config, selection, seed, coordinate or checkpoint identity."""
    freeze = read(root / "FINAL_FREEZE.json")
    if file_sha(root / "CONFIG.json") != freeze["config_sha256"]:
        raise ValueError("Changed router configuration")
    if file_sha(root / "SELECTION.json") != freeze["selection_sha256"]:
        raise ValueError("Changed router selection")
    models = []
    for seed in (7, 13, 21):
        path = root / "final" / str(seed) / "checkpoint.pt"
        if file_sha(path) != freeze["checkpoints"][str(seed)]:
            raise ValueError("Changed router checkpoint")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        coordinate = payload.get("coordinate", "seconds")
        if (
            payload["seed"] != seed
            or payload["mode"] != freeze["selected"]
            or coordinate != freeze.get("coordinate", "seconds")
        ):
            raise ValueError("Router identity mismatch")
        model = ExpertRouter("native" if payload["mode"] == "NAcap" else "quality", coordinate)
        model.load_state_dict(payload["state_dict"], strict=True)
        models.append(model.eval())
    return RouterEnsemble(models).eval()


def predict_history(
    model: RouterEnsemble, native: np.ndarray, boxes: np.ndarray, clocks: np.ndarray, query_us: int
) -> tuple[float, dict]:
    """Use native F0/RGB/event seconds and only causal inference-provided boxes."""
    if np.asarray(native).shape != (3,):
        raise ValueError("Expected three native estimates")
    physical = startup_geometry(boxes, clocks, query_us)
    geo = float(physical["ttc"]) if physical["valid"] else float("nan")
    with torch.inference_mode():
        value = model(
            torch.tensor(native, dtype=torch.float32)[None],
            torch.tensor([geo], dtype=torch.float32),
            torch.tensor([physical["quality"]], dtype=torch.float32),
        )
    return float(value[0]), physical


def load_bundle_router(root: Path, variant: str) -> RouterEnsemble:
    """Bind the evaluated/all-data identity, exports and calibration independently."""
    if variant not in ("evaluated", "all_data_refit"):
        raise ValueError("Unsupported router variant")
    binding = read(root / "ROUTER_BINDING.json")
    if binding.get("schema") != "phase_startup_bundle_v1":
        raise ValueError("Unsupported router binding")
    if set(binding["implementation"]) != {"model.py", "geometry.py", "runtime.py"}:
        raise ValueError("Changed router implementation inventory")
    for name, digest in binding["implementation"].items():
        if file_sha(Path(__file__).parent / name) != digest:
            raise ValueError("Changed router implementation")
    if file_sha(root / "upstream/BUNDLE.json") != binding["upstream_sha256"]:
        raise ValueError("Changed upstream identity")
    if file_sha(root / variant / "FINAL_FREEZE.json") != binding["freezes"][variant]:
        raise ValueError("Changed router variant identity")
    if set(binding["artifacts"]) != {
        "evaluated/router.ts",
        "all_data_refit/router.ts",
        "evaluated/CONFIRMATION.json",
    }:
        raise ValueError("Changed router artifact inventory")
    for relative, digest in binding["artifacts"].items():
        if file_sha(root / relative) != digest:
            raise ValueError("Changed router artifact")
    return load_router(root / variant)


def describe_prediction(
    prediction: float, native: np.ndarray, physical: dict, calibration: dict | None
) -> dict:
    """Attach inference-only flags and optional historically exposed calibration."""
    flags = []
    if physical["count"] < 3:
        flags.append("short_bbox_history")
    if not physical["valid"]:
        flags.append(physical["reason"])
    # Match the float32, +/-60s geometry convention used for calibration features.
    geometry = np.float32(np.clip(physical["ttc"], -60, 60)) if physical["valid"] else np.nan
    experts = np.r_[np.asarray(native, dtype=np.float32), geometry].astype(np.float64)
    finite = experts[np.isfinite(experts)]
    if len(finite) and np.any(finite < 0) and np.any(finite > 0):
        flags.append("expert_sign_disagreement")
    if not np.isfinite(prediction):
        flags.append("prediction_unavailable")
    interval = None
    if calibration is None:
        flags.append("uncertainty_unvalidated_after_all_data_refit")
    elif len(finite) and np.isfinite(prediction):
        scale = max(0.1, float(np.median(np.abs(finite - np.median(finite)))))
        radius = max(calibration["constant_radius"], calibration["scaled_factor"] * scale)
        interval = [prediction - radius, prediction + radius]
        flags.append("interval_is_descriptive_not_certified")
    return {
        "ttc_seconds": prediction if np.isfinite(prediction) else None,
        "interval_80_seconds": interval,
        "quality_flags": flags,
        "geometry_count": physical["count"],
    }
