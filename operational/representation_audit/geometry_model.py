"""P-only height calibration; original tokens remain the inputs to PAIR."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCOutput
from e_jepa_ttc.range_readout.readout import VectorReadout
from operational.range_transfer.common import freeze, read, sha

from .run import OLD, OWNER, RUN

CALIBRATION = RUN / "geometry/P_HEIGHT_CALIBRATION.json"


def calibrate() -> dict:
    """Fit ridge from sufficient statistics, admitting only labelled P observations."""
    started = time.perf_counter()
    folder = OLD / "vector_cache/P"
    complete = read(folder / "COMPLETE.json")
    if complete["binding"]["manifest_sha256"] != sha(OWNER / "P_MANIFEST.json"):
        raise ValueError("P identity changed")
    if (
        complete["binding"]["parent_sha256"]
        != read(OLD / "ADMISSION.json")["parents"]["E_A5_MATCHED"]
    ):
        raise ValueError("Wrong A5 parent")
    sx = np.zeros(128, np.float64)
    sxx = np.zeros((128, 128), np.float64)
    sxy = np.zeros(128, np.float64)
    sy = syy = 0.0
    count = 0
    hashes = {}
    for index in range(math.ceil(complete["rows"] / 128)):
        path = folder / f"part_{index:06d}.npz"
        hashes[path.name] = sha(path)
        if hashes[path.name] != read(path.with_suffix(".json"))["sha256"]:
            raise ValueError("P shard changed")
        with np.load(path, allow_pickle=False) as arc:
            valid = arc["target_valid"].astype(bool)
            target = arc["target_height_normalized"][valid].astype(np.float64)
            original = arc["visible_height"][valid].astype(np.float64)
            x = arc["geometry_tokens"][valid].astype(np.float64)
        if not ((target > 0).all() and (original > 0).all() and np.isfinite(x).all()):
            raise ValueError("Invalid admitted height")
        y = np.log(target) - np.log(original)
        sx += x.sum(0)
        sxx += x.T @ x
        sxy += x.T @ y
        sy += float(y.sum())
        syy += float(y @ y)
        count += len(y)
    if count != 107780:
        raise ValueError("Unexpected P observation count")
    mean = sx / count
    centered = sxx - np.outer(sx, sx) / count
    scale = np.sqrt(np.maximum(np.diag(centered) / count, 1e-12))
    zz = centered / np.outer(scale, scale)
    zy = (sxy - sx * sy / count) / scale
    coefficient = np.linalg.solve(zz + np.eye(128), zy)
    intercept = sy / count
    residual_ss = syy - sy * sy / count - 2 * coefficient @ zy + coefficient @ zz @ coefficient
    result = {
        "role": "P",
        "observations": count,
        "queries": complete["rows"],
        "closed_form_fits": 1,
        "optimizer_updates": 0,
        "ridge_lambda": 1.0,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": intercept,
        "delta_clip": [-2.0, 2.0],
        "target": "log_bbox_height_minus_log_original_height",
        "unclipped_training_rmse_log_height": float(np.sqrt(max(0.0, residual_ss / count))),
        "intercept_only_rmse_log_height": float(np.sqrt((syy - sy * sy / count) / count)),
        "sources": {
            "code": sha(__import__("pathlib").Path(__file__)),
            "complete": sha(folder / "COMPLETE.json"),
            "shards": hashes,
        },
    }
    # Elapsed time is separate so exact re-execution can verify the fitted coefficients.
    freeze(CALIBRATION, result)
    timing = CALIBRATION.with_name("FIT_COST.json")
    if not timing.exists():
        freeze(timing, {"cpu_wall_s": time.perf_counter() - started, "scope": "P ridge fit"})
    return result


class GeometryReadout(VectorReadout):
    """Calibrate the analytic branch without feeding changed geometry into PAIR."""

    height_mean: torch.Tensor
    height_scale: torch.Tensor
    height_coefficient: torch.Tensor
    height_intercept: torch.Tensor
    height_active: torch.Tensor

    def __init__(self, native: CausalScaleTTC, calibration: dict, arm: str) -> None:
        super().__init__(native)
        if arm not in ("Q0", "Q1") or calibration["role"] != "P":
            raise ValueError("Unregistered geometry calibration")
        for name in ("mean", "scale", "coefficient", "intercept"):
            self.register_buffer(
                "height_" + name, torch.tensor(calibration[name], dtype=torch.float32)
            )
        self.register_buffer("height_active", torch.tensor(float(arm == "Q1")))

    def corrected_inputs(self, cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Keep raw token, support and transport tensors unchanged by identity."""
        if any(key.startswith(("target_", "dino_")) for key in cache):
            raise ValueError("Training targets are forbidden in inference inputs")
        z = (cache["geometry_tokens"] - self.height_mean) / self.height_scale
        delta = (self.height_intercept + self.height_active * (z @ self.height_coefficient)).clamp(
            -2, 2
        )
        changed = dict(cache)
        changed["analytic"] = cache["analytic"] + (delta[:, 1:] - delta[:, :-1])
        changed["visible_height"] = cache["visible_height"] * delta.exp()
        changed["log_height_correction"] = cache["log_height_correction"] + delta
        return changed

    def forward(self, cache: dict[str, torch.Tensor]) -> CausalScaleTTCOutput:
        return super().forward(self.corrected_inputs(cache))
