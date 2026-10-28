"""Strict loaders for the three released Garl-TTC LHR comparators."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import yaml
from torch import nn

from .prepare import VARIANTS, Variant

NATIVE_DELTA_T_S = 0.1


class _Incompatible(Protocol):
    missing_keys: list[str]
    unexpected_keys: list[str]


class _NativeModel(Protocol):
    dT: float  # noqa: N815 - upstream public API

    def float(self) -> _NativeModel: ...

    def eval(self) -> _NativeModel: ...

    def requires_grad_(self, requires_grad: bool = True) -> _NativeModel: ...

    def to(self, device: torch.device) -> _NativeModel: ...

    def load_state_dict(
        self, state: Mapping[str, torch.Tensor], strict: bool = True
    ) -> _Incompatible: ...

    def forward_test(self, sensor: torch.Tensor) -> tuple[torch.Tensor, object]: ...


@dataclass(frozen=True)
class GarlPrediction:
    variant: Variant
    status: str
    model_ttc: float | None
    heights: list[float | str]
    reason: str
    native_delta_t_s: float
    conversion_delta_t_s: float
    diagnostics: dict[str, float | str]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_inventory(value: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Path):
        loaded = json.loads(value.read_text(encoding="utf-8"))
    else:
        loaded = dict(value)
    if loaded.get("schema") != "garl_official_local_inventory_v1":
        raise ValueError("unknown Garl official inventory schema")
    return loaded


def _serialize(value: float) -> float | str:
    if np.isfinite(value):
        return value
    if np.isnan(value):
        return "nan"
    return "+inf" if value > 0 else "-inf"


class OfficialGarlRuntime:
    """Strict-load three independent public checkpoints without checkpoint mixing."""

    def __init__(
        self,
        code_root: Path,
        inventory: Path | Mapping[str, Any],
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Garl runtime supports CPU or CUDA")
        self.inventory = _read_inventory(inventory)
        self.code_root = code_root.resolve(strict=True)
        upstream = self.inventory["upstream"]
        if self.code_root != Path(upstream["path"]).resolve(strict=True):
            raise ValueError("native code root differs from frozen inventory")
        for entry in upstream["native_import_graph"]:
            path = Path(entry["path"]).resolve(strict=True)
            if not path.is_relative_to(self.code_root) or _sha256(path) != entry["sha256"]:
                raise ValueError(f"native Garl source changed: {path}")

        root = str(self.code_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        module = importlib.import_module("garl_ttc.models.ttc_network")
        source = Path(module.__file__ or "").resolve(strict=True)
        if not source.is_relative_to(self.code_root):
            raise ValueError("Garl constructor imported from an unpinned root")
        constructor = module.TTCNetwork

        self.models: dict[Variant, _NativeModel] = {}
        self.bindings: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            record = self.inventory["variants"][variant]
            checkpoint = Path(record["checkpoint"]["path"]).resolve(strict=True)
            config_path = Path(record["config"]["path"]).resolve(strict=True)
            for path, expected in (
                (checkpoint, record["checkpoint"]),
                (config_path, record["config"]),
            ):
                if (
                    path.stat().st_size != int(expected["bytes"])
                    or _sha256(path) != expected["sha256"]
                ):
                    raise ValueError(f"frozen Garl asset changed: {variant}/{path.name}")
            configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            expected_mode = {"full": "image_event", "event": "event_only", "visual": "image_only"}[
                variant
            ]
            if (
                configuration["dataset"]["mode"] != expected_mode
                or configuration["dataset"]["sync"] != "front"
                or configuration["dataset"]["window_interval"] != 1
                or configuration["model"]["mode"] != "height_ratio"
                or configuration["model"]["frame_num"] != 2
            ):
                raise ValueError(f"Garl {variant} configuration contract changed")
            removed = {
                key: configuration["model"].pop(key, None)
                for key in ("pretrained_ckpt_rgb", "pretrained_ckpt_event")
            }
            model = cast(_NativeModel, constructor(configuration, is_train=False))
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            incompatible = model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(f"strict Garl state load failed: {variant}")
            model = model.float().eval().requires_grad_(False).to(self.device)
            if float(model.dT) != NATIVE_DELTA_T_S:
                raise ValueError(f"native Garl dT changed: {variant}")
            self.models[variant] = model
            parameter_count = sum(value.numel() for value in cast(nn.Module, model).parameters())
            self.bindings[variant] = {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": record["checkpoint"]["sha256"],
                "config_path": str(config_path),
                "config_sha256": record["config"]["sha256"],
                "strict_state_load": True,
                "removed_initializer_paths": removed,
                "parameter_count": parameter_count,
                "native_delta_t_s": NATIVE_DELTA_T_S,
                "dataset_mode": expected_mode,
            }

    def predict(self, variant: Variant, sensor: np.ndarray, delta_t_s: float) -> GarlPrediction:
        """Run raw heights and convert with measured cadence without mutating native dT."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown Garl variant: {variant}")
        channels = {"full": 46, "event": 40, "visual": 6}[variant]
        value = np.asarray(sensor)
        if value.shape != (channels, 128, 128) or value.dtype != np.float32:
            raise ValueError(f"{variant} sensor must be float32 [{channels},128,128]")
        if not np.isfinite(value).all():
            raise ValueError("Garl sensor must be finite")
        if not np.isfinite(delta_t_s) or delta_t_s <= 0:
            raise ValueError("conversion delta must be finite and positive")
        tensor = torch.from_numpy(np.ascontiguousarray(value)[None]).to(self.device)
        with torch.inference_mode():
            raw, _ = self.models[variant].forward_test(tensor)
        heights = raw.detach().float().cpu().numpy()
        if heights.shape != (1, 2):
            raise ValueError("Garl height output must have shape [1,2]")
        values = [float(item) for item in heights[0]]
        serialized = [_serialize(item) for item in values]
        diagnostics: dict[str, float | str] = {
            "height_t0": serialized[0],
            "height_t1": serialized[1],
        }
        if not np.isfinite(heights).all() or values[1] == 0.0:
            return GarlPrediction(
                variant,
                "unavailable",
                None,
                serialized,
                "nonfinite_or_zero_height_output",
                NATIVE_DELTA_T_S,
                float(delta_t_s),
                diagnostics,
            )
        # Preserve the released converter's FP32 NumPy operation order.  Only
        # dT is replaced by the measured endpoint spacing; model.dT stays 0.1.
        ratio = heights[:, 0] / heights[:, 1]
        denominator = np.float32(1.0) - ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            converted = np.asarray([delta_t_s], dtype=np.float32) / denominator
        ttc32 = converted[0]
        diagnostics["height_ratio_denominator"] = float(denominator[0])
        if not np.isfinite(ttc32):
            diagnostics["raw_ttc"] = _serialize(float(ttc32))
            return GarlPrediction(
                variant,
                "unavailable",
                None,
                serialized,
                "nonfinite_height_ratio_ttc",
                NATIVE_DELTA_T_S,
                float(delta_t_s),
                diagnostics,
            )
        return GarlPrediction(
            variant,
            "ok",
            float(ttc32),
            serialized,
            "available",
            NATIVE_DELTA_T_S,
            float(delta_t_s),
            diagnostics,
        )


__all__ = ["GarlPrediction", "OfficialGarlRuntime"]
