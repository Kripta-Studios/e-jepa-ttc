"""Exact packed expert extraction for the fixed TRAIN40 H8 B16 inference path."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.simplex_t.expert_phase import expert_phase_from_ttc
from operational.simplex_t_post_campaign.route_policy import ALLOWED


class H8Extractor:
    """Run canonical FP32 producers and combine their outputs in one D2H transfer."""

    def __init__(self, models: dict[str, Any], mode: str = "packed") -> None:
        if mode not in {"packed", "graph"}:
            raise ValueError("H8 extractor mode must be packed or graph")
        if set(models) != {"A5", "C2F", "PAIR"}:
            raise ValueError("H8 extraction requires exactly A5, C2F and PAIR")
        if any(model.training for model in models.values()):
            raise ValueError("H8 extraction requires frozen eval models")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.models = models
        self.mode = mode
        self._forwards: dict[str, Any] = {
            "A5": models["A5"],
            "C2F": models["C2F"],
        }
        if mode == "graph":
            self._forwards = {
                name: torch.compile(model, backend="cudagraphs", dynamic=False, fullgraph=False)
                for name, model in self._forwards.items()
            }
        self.calls = 0
        self.total_seconds = 0.0
        self.last_seconds = 0.0
        self.device_to_host_transfers = 0
        self.graph_enabled = mode == "graph"

    def _validate(
        self, model: str, models: dict[str, Any], events: Tensor, delta: Tensor
    ) -> frozenset[str]:
        if model not in ALLOWED:
            raise ValueError("unregistered frozen profiling model")
        if models is not self.models or any(
            models[name] is not self.models[name] for name in models
        ):
            raise ValueError("H8 extractor model identity changed")
        allowed = ALLOWED[model]
        if allowed != frozenset({"A5", "C2F", "PAIR"}):
            raise ValueError("fast H8 extraction is admitted only for the full expert family")
        if events.dtype != torch.float32 or delta.dtype != torch.float32:
            raise ValueError("frozen FP32 inputs required")
        if tuple(events.shape) != (16, 3, 12, 128, 128) or tuple(delta.shape) != (16, 2):
            raise ValueError("fast H8 extraction requires fixed B16,T3,C12,H128,W128")
        if not bool((delta > 0).all()) or any(models[key].training for key in allowed):
            raise ValueError("positive producer deltas and frozen eval models required")
        return allowed

    @staticmethod
    def _producer_fields(
        output: Any,  # noqa: ANN401 - frozen model output is a dynamic dataclass
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ttc = output.ttc_mean_seconds.float()
        flow = output.diagnostics["transport_flow_magnitude"][:, -1].float()
        quality = torch.minimum(
            output.log_height_ratio[:, -1].abs() / 0.002,
            output.sensor_support[:, -1] / 0.0001,
        ).float()
        log_variance = output.ttc_log_variance.float()
        return flow, quality, log_variance, ttc

    @torch.inference_mode()
    def __call__(
        self, model: str, models: dict[str, Any], events: Tensor, delta: Tensor
    ) -> np.ndarray:
        """Return the canonical PHASE17 matrix with exactly one packed D2H copy."""
        self._validate(model, models, events, delta)
        started = time.perf_counter()

        sensor_positive = events[:, -1, -2].mean((-2, -1))
        sensor_negative = events[:, -1, -1].mean((-2, -1))
        a5 = self._forwards["A5"](events, delta, return_dense_features=True)
        # CUDA-graph outputs have static lifetime and a later captured forward may
        # reuse their storage. Materialize every A5 primitive before invoking C2F.
        a5_fields = tuple(value.clone() for value in self._producer_fields(a5))

        elapsed = delta[:, 1]
        support = a5.sensor_support
        pair_features = torch.cat(
            (
                a5.pair_tokens[:, -1].float(),
                torch.stack((elapsed, torch.log(elapsed + 1e-8), elapsed.reciprocal()), -1),
                support[:, -1:],
                torch.minimum(support[:, -2], support[:, -1]).unsqueeze(-1),
            ),
            -1,
        ).float()
        pair_ttc = self.models["PAIR"].predict_ttc(PairFeatureBatch(pair_features)).float()

        c2f = self._forwards["C2F"](events, delta, return_dense_features=True)
        c2f_fields = self._producer_fields(c2f)
        packed = torch.stack(
            (sensor_positive, sensor_negative, *a5_fields, *c2f_fields, pair_ttc), dim=1
        ).float()
        host = packed.cpu().numpy()
        self.device_to_host_transfers += 1

        if not np.isfinite(host[:, :10]).all():
            raise ValueError("nonfinite permitted A5/C2F output")
        if np.isnan(host[:, 10]).any():
            raise ValueError("NaN permitted PAIR prediction")
        values = np.zeros((len(events), 17), np.float64)
        values[:, :5] = host[:, :5]
        values[:, 5:8] = host[:, 6:9]
        values[:, 8] = expert_phase_from_ttc(host[:, 5])
        values[:, 9] = expert_phase_from_ttc(host[:, 9])
        values[:, 10] = expert_phase_from_ttc(host[:, 10])
        values[:, 11] = values[:, 10] - values[:, 8]
        values[:, 12] = values[:, 10] - values[:, 9]
        values[:, 13] = values[:, 9] - values[:, 8]
        values[:, 14] = np.abs(values[:, 11])
        values[:, 15] = np.abs(values[:, 12])
        values[:, 16] = np.abs(values[:, 13])
        if not np.isfinite(values).all():
            raise ValueError("nonfinite permitted features")
        result = values.astype(np.float32)
        self.calls += 1
        self.last_seconds = time.perf_counter() - started
        self.total_seconds += self.last_seconds
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return bounded JSON-safe runtime counters."""
        return {
            "schema": "train40_h8_fast_extract_runtime_v1",
            "mode": self.mode,
            "graph_enabled": self.graph_enabled,
            "calls": self.calls,
            "total_seconds": self.total_seconds,
            "last_seconds": self.last_seconds,
            "device_to_host_transfers": self.device_to_host_transfers,
            "fixed_shape": [16, 3, 12, 128, 128],
            "precision": "float32",
            "enabled_models": ["A5", "C2F", "PAIR"],
            "positive_delta_required": True,
            "tf32_enabled": False,
            "amp_enabled": False,
            "inductor_fusion_enabled": False,
        }


__all__ = ["H8Extractor"]
