"""Height readouts from frozen global/vertical maps with original PAIR inputs."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.range_transfer.common import freeze, read, sha

from .geometry_model import GeometryReadout
from .precision import Vectors
from .run import OLD, OWNER, RUN
from .spatial_cache import source_binding


def calibration_path(arm: str) -> Path:
    if arm not in ("SP0", "SP1"):
        raise ValueError("Unknown spatial arm")
    return RUN / f"spatial_geometry/{arm}_CALIBRATION.json"


def spatial_inputs(values: np.ndarray, active: bool) -> np.ndarray:
    if values.ndim != 3 or values.shape[1:] != (3, 512):
        raise ValueError("Expected three 512-dimensional observations")
    if active:
        return values
    result = np.zeros_like(values)
    result[:, :, :64] = values.reshape(-1, 3, 64, 8).mean(-1)
    return result


class SpatialVectors(Vectors):
    """Join immutable vector and spatial caches by exact sample identity/order."""

    def __init__(self, role: str, *, synthetic: bool = False) -> None:
        super().__init__(role, synthetic=synthetic)
        if synthetic:
            self.values["height_features"] = (
                np.random.default_rng(91).normal(0, 0.1, (64, 3, 512)).astype(np.float32)
            )
            return
        folder = RUN / "spatial_cache" / role
        complete = read(folder / "COMPLETE.json")
        binding = complete["binding"]
        if (
            binding["sources"] != source_binding()
            or binding["parent_sha256"] != read(OLD / "ADMISSION.json")["parents"]["E_A5_MATCHED"]
            or binding["manifest_sha256"] != sha(OWNER / f"{role}_MANIFEST.json")
        ):
            raise ValueError("Spatial provenance changed")
        if (
            complete["status"] != "COMPLETE"
            or complete["rows"] != self.n
            or complete["binding"]["role"] != role
        ):
            raise ValueError("Incomplete spatial role")
        parts = []
        hashes = {}
        for index in range(math.ceil(self.n / 128)):
            path = folder / f"part_{index:06d}.npz"
            hashes[path.name] = sha(path)
            if hashes[path.name] != read(path.with_suffix(".json"))["sha256"]:
                raise ValueError("Spatial shard changed")
            with np.load(path, allow_pickle=False) as arc:
                if not np.array_equal(
                    arc["sample_token"],
                    self.values["sample_token"][index * 128 : (index + 1) * 128],
                ):
                    raise ValueError("Spatial/vector sample order mismatch")
                parts.append(arc["height_features"])
        self.values["height_features"] = np.concatenate(parts)
        self.identity["spatial"] = {"complete": sha(folder / "COMPLETE.json"), "shards": hashes}


def calibrate() -> None:
    data = SpatialVectors("P")
    v = data.values
    valid = v["target_valid"].astype(bool)
    height = v["target_height_normalized"].astype(np.float64)
    original = v["visible_height"].astype(np.float64)
    y = np.log(height[valid]) - np.log(original[valid])
    pairs = valid[:, -1] & valid[:, -2]
    target = np.log(height[pairs, -1] / height[pairs, -2]) - v["analytic"][pairs, -1]
    if len(y) != 107780 or int(pairs.sum()) != 53890:
        raise ValueError("Wrong P target support")
    variance_y, variance_d = float(y.var()), float(target.var())
    intercept = float(y.mean())
    for arm in ("SP0", "SP1"):
        started = time.perf_counter()
        raw = spatial_inputs(v["height_features"].astype(np.float64), arm == "SP1")
        mean = raw[valid].mean(0)
        scale = np.maximum(raw[valid].std(0), 1e-6)
        z = (raw - mean) / scale
        x = z[valid]
        d = z[pairs, -1] - z[pairs, -2]
        factor = len(y) / len(d)
        matrix = x.T @ x / variance_y + factor * (d.T @ d) / variance_d + np.eye(512)
        rhs = x.T @ (y - intercept) / variance_y + factor * (d.T @ target) / variance_d
        beta = np.linalg.solve(matrix, rhs)
        result = {
            "role": "P",
            "arm": arm,
            "observations": len(y),
            "pairs": len(d),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "intercept": intercept,
            "coefficient": beta.tolist(),
            "ridge_lambda": 1.0,
            "delta_clip": [-2.0, 2.0],
            "variance_height_target": variance_y,
            "variance_temporal_target": variance_d,
            "closed_form_fits_new": 1,
            "optimizer_updates": 0,
            "unclipped_training_height_rmse": float(
                np.sqrt(np.mean((x @ beta + intercept - y) ** 2))
            ),
            "unclipped_training_temporal_rmse": float(np.sqrt(np.mean((d @ beta - target) ** 2))),
            "sources": {"cache": data.identity, "code": sha(Path(__file__))},
        }
        freeze(calibration_path(arm), result)
        cost = calibration_path(arm).with_name(f"{arm}_FIT_COST.json")
        if not cost.exists():
            freeze(
                cost,
                {
                    "cpu_wall_s": time.perf_counter() - started,
                    "scope": "P ridge fit; input loading excluded",
                },
            )
        print(arm, {k: r for k, r in result.items() if "rmse" in k}, flush=True)


class SpatialReadout(GeometryReadout):
    """The encoder map is measured without altering the original token route."""

    spatial_active: torch.Tensor

    def __init__(self, native: CausalScaleTTC, calibration: dict, arm: str) -> None:
        if arm not in ("SP0", "SP1"):
            raise ValueError("Unknown spatial operator")
        super().__init__(native, calibration, "Q1")
        self.register_buffer("spatial_active", torch.tensor(float(arm == "SP1")))

    def corrected_inputs(self, cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if any(key.startswith(("target_", "dino_")) for key in cache):
            raise ValueError("Training targets are forbidden in inference inputs")
        raw = cache["height_features"]
        if raw.shape != (len(raw), 3, 512):
            raise ValueError("Wrong spatial features")
        global_values = torch.nn.functional.pad(raw.reshape(-1, 3, 64, 8).mean(-1), (0, 448))
        inputs = self.spatial_active * raw + (1 - self.spatial_active) * global_values
        z = (inputs - self.height_mean) / self.height_scale
        delta = (self.height_intercept + z @ self.height_coefficient).clamp(-2, 2)
        changed = {k: v for k, v in cache.items() if k != "height_features"}
        changed["analytic"] = cache["analytic"] + (delta[:, 1:] - delta[:, :-1])
        changed["visible_height"] = cache["visible_height"] * delta.exp()
        changed["log_height_correction"] = cache["log_height_correction"] + delta
        return changed


if __name__ == "__main__":
    calibrate()
