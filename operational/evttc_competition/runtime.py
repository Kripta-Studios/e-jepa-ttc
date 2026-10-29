"""Prepared-observation runtime for the sealed TRAIN40 competition model.

This module adapts causal sensor observations to the unchanged nine-component
inference graph.  Sensor cadence is measured from observation timestamps; the
model's final phase-to-TTC conversion remains the frozen 100 ms mapping.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCOutput
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.final_refit.test12 import FinalV16Train40Predictor
from operational.rgb_port import transfer as engine
from operational.rgb_port_streaming.contracts import NativeObservation
from scripts.final_train40_transfer import load_models, verify_candidate

Modality = Literal["event", "rgb"]
MODEL_PHASE_DELTA_SECONDS = 0.1
MAX_HISTORY = 8
MAX_INFORMATION_SPAN_US = 650_000


@dataclass(frozen=True)
class PreparedObservation:
    """A real causal T2/T3 sensor observation prepared at 128 x 128.

    ``timestamps_us`` are the endpoints of the actual frames/event windows.
    They, rather than a nominal cadence, determine every producer delta.
    """

    modality: Modality
    pixels: np.ndarray
    timestamps_us: tuple[int, ...]
    first_sensor_ts_us: int
    anchor_us: int
    available_us: int

    def __post_init__(self) -> None:
        if self.modality not in {"event", "rgb"}:
            raise ValueError("modality must be 'event' or 'rgb'")
        pixels = np.asarray(self.pixels)
        channels = 12 if self.modality == "event" else 3
        if pixels.dtype != np.float32:
            raise ValueError("prepared pixels must be float32")
        if pixels.ndim != 4 or pixels.shape[0] not in {2, 3}:
            raise ValueError("prepared observations must contain a real T2 or T3 sequence")
        if pixels.shape[1:] != (channels, 128, 128):
            raise ValueError(f"{self.modality} pixels must have shape [T,{channels},128,128]")
        if not np.isfinite(pixels).all():
            raise ValueError("prepared pixels must be finite")
        if self.modality == "rgb" and ((pixels < 0).any() or (pixels > 1).any()):
            raise ValueError("prepared RGB pixels must be in [0,1]")
        clocks = self.timestamps_us
        if len(clocks) != pixels.shape[0] or any(isinstance(item, bool) for item in clocks):
            raise ValueError("timestamps_us must contain one integer endpoint per sensor item")
        if any(not isinstance(item, (int, np.integer)) for item in clocks):
            raise ValueError("timestamps_us endpoints must be integers")
        if any(right <= left for left, right in zip(clocks, clocks[1:], strict=False)):
            raise ValueError("timestamps_us must be strictly increasing")
        integer_fields = (self.first_sensor_ts_us, self.anchor_us, self.available_us)
        if any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer))
            for item in integer_fields
        ):
            raise ValueError("sensor clocks must be integer microseconds")
        if self.modality == "rgb" and int(self.first_sensor_ts_us) != int(clocks[0]):
            raise ValueError("RGB first_sensor_ts_us must equal the first frame timestamp")
        if self.modality == "event" and int(self.first_sensor_ts_us) > int(clocks[0]):
            raise ValueError("event first_sensor_ts_us must not follow the first window endpoint")
        if int(self.anchor_us) != int(clocks[-1]):
            raise ValueError("anchor_us must equal the current sensor endpoint")
        if self.available_us < self.anchor_us:
            raise ValueError("an observation cannot be available before its anchor")
        if self.anchor_us - self.first_sensor_ts_us > MAX_INFORMATION_SPAN_US:
            raise ValueError("one observation exceeds the 650 ms information budget")
        owned = np.ascontiguousarray(pixels).copy()
        owned.setflags(write=False)
        object.__setattr__(self, "pixels", owned)
        object.__setattr__(self, "timestamps_us", tuple(int(item) for item in clocks))

    @property
    def delta_t_seconds(self) -> np.ndarray:
        """Return measured positive pair intervals, never a nominal replacement."""
        return np.diff(np.asarray(self.timestamps_us, dtype=np.int64)).astype(np.float32) / 1e6


@dataclass(frozen=True)
class RuntimePrediction:
    status: Literal["ok", "unavailable"]
    model_ttc: float | None
    event_ttc: float | None
    rgb_ttc: float | None
    model_reason: str
    event_reason: str
    rgb_reason: str
    event_history_count: int
    rgb_history_count: int
    selection_seconds: float
    producer_seconds: float
    context_seconds: float
    total_seconds: float
    producer_diagnostics: dict[str, dict[str, object]]
    sensor_cadence: str = "measured_preprocessing_adaptation"
    model_phase_delta_seconds: float = MODEL_PHASE_DELTA_SECONDS


class CompetitionRuntime:
    """Run prepared observations through the unchanged sealed TRAIN40 graph."""

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        normalizers: Mapping[str, tuple[Tensor, Tensor]],
        *,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.models = dict(models)
        self.normalizers = dict(normalizers)

    @classmethod
    def from_candidate(cls, candidate_root: Path, *, device: str = "cpu") -> CompetitionRuntime:
        """Verify and strict-load the portable nine-component candidate."""
        paths, _ = verify_candidate(candidate_root)
        models, normalizers = load_models(paths, device)
        return cls(models, normalizers, device=device)

    @staticmethod
    def _select(
        observations: Sequence[PreparedObservation], modality: Modality, query_us: int
    ) -> list[PreparedObservation]:
        if isinstance(query_us, bool) or not isinstance(query_us, (int, np.integer)):
            raise ValueError("query_us must be integer microseconds")
        if any(item.modality != modality for item in observations):
            raise ValueError(f"{modality} history contains another modality")
        causal = [
            item
            for item in observations
            if item.anchor_us <= query_us
            and item.available_us <= query_us
            and query_us - item.first_sensor_ts_us <= MAX_INFORMATION_SPAN_US
        ]
        causal.sort(key=lambda item: (item.anchor_us, item.available_us))
        anchors = [item.anchor_us for item in causal]
        if len(set(anchors)) != len(anchors):
            raise ValueError(f"duplicate {modality} anchors are ambiguous")
        return causal[-MAX_HISTORY:]

    def _prepare_values(
        self, observations: Sequence[PreparedObservation], prefix: Literal["E", "R"]
    ) -> tuple[
        list[PreparedObservation],
        list[tuple[np.ndarray, np.ndarray]],
        dict[str, object],
    ]:
        accepted: list[PreparedObservation] = []
        values: list[tuple[np.ndarray, np.ndarray]] = []
        known_true = 0
        known_total = 0
        support_count = 0
        support_min = float("inf")
        support_max = float("-inf")
        rejected_compute_seconds = 0.0
        rejected: list[dict[str, object]] = []
        observation_outputs: list[dict[str, object]] = []
        accepted_indices: list[int] = []

        def diagnostic_hook(
            producer_id: str,
        ) -> Callable[[nn.Module, tuple[Any, ...], CausalScaleTTCOutput], None]:
            def hook(
                module: nn.Module,
                arguments: tuple[Any, ...],
                output: CausalScaleTTCOutput,
            ) -> None:
                nonlocal known_true, known_total, support_count, support_min, support_max
                del module, arguments
                known = output.known_mask.detach()
                support = output.sensor_support.detach().float()
                ttc = output.ttc_mean_seconds.detach().float()
                ttc_values = [float(value) for value in ttc.cpu()]
                classifications = [
                    (
                        "unrepresentable_nan"
                        if np.isnan(value)
                        else "unrepresentable_zero"
                        if value == 0.0
                        else "unrepresentable_positive_le_model_delta"
                        if 0.0 < value <= MODEL_PHASE_DELTA_SECONDS
                        else "phase_limit_but_native_features_nonfinite"
                        if np.isinf(value)
                        else "representable_finite"
                    )
                    for value in ttc_values
                ]
                serialized_ttc: list[float | str] = [
                    value
                    if np.isfinite(value)
                    else "nan"
                    if np.isnan(value)
                    else "+inf"
                    if value > 0
                    else "-inf"
                    for value in ttc_values
                ]
                observation_outputs.append(
                    {
                        "producer_id": producer_id,
                        "ttc_seconds": serialized_ttc,
                        "domain_classification": classifications,
                        "unrepresentable": any(
                            value.startswith("unrepresentable") for value in classifications
                        ),
                        "known_mask": [bool(value) for value in known.cpu()],
                        "sensor_support": support.cpu().tolist(),
                    }
                )
                known_true += int(known.count_nonzero().cpu())
                known_total += int(known.numel())
                support_count += int(support.numel())
                support_min = min(support_min, float(support.min().cpu()))
                support_max = max(support_max, float(support.max().cpu()))

            return hook

        producer_names = ("E_A5_MATCHED", "E_C2F_MATCHED") if prefix == "E" else ("R_A5", "R_C2F")
        for observation_index, item in enumerate(observations):
            observation_outputs.clear()
            sensor = torch.from_numpy(np.array(item.pixels, copy=True))[None].to(self.device)
            delta = torch.from_numpy(item.delta_t_seconds.copy())[None].to(self.device)
            hooks = [
                self.models[name].register_forward_hook(diagnostic_hook(name))
                for name in producer_names
                if name in self.models
            ]
            observation_started = time.perf_counter()
            try:
                with (
                    torch.inference_mode(),
                    patch.object(engine, "ProducerObservation", NativeObservation),
                ):
                    raw, experts = engine._expert_block(self.models, prefix, sensor, delta)
            except ValueError as error:
                invalid = [
                    output for output in observation_outputs if output["unrepresentable"] is True
                ]
                if str(error) == "TTC lies outside the signed benchmark-phase domain" and invalid:
                    elapsed = time.perf_counter() - observation_started
                    rejected_compute_seconds += elapsed
                    rejected.append(
                        {
                            "observation_index": observation_index,
                            "anchor_us": item.anchor_us,
                            "reason": "unrepresentable_expert_ttc",
                            "compute_seconds": elapsed,
                            "invalid_producers": [dict(value) for value in invalid],
                            "all_producer_outputs": [dict(value) for value in observation_outputs],
                        }
                    )
                    continue
                raise
            finally:
                for hook in hooks:
                    hook.remove()
            accepted.append(item)
            accepted_indices.append(observation_index)
            values.append((raw[0].cpu().numpy(), experts[0].cpu().numpy()))
        diagnostics: dict[str, object] = {
            "known_true": known_true,
            "known_total": known_total,
            "support_count": support_count,
            "observations_requested": len(observations),
            "observations_accepted": len(accepted),
            "accepted_observation_indices": accepted_indices,
            "accepted_anchor_us": [item.anchor_us for item in accepted],
            "observations_rejected_unrepresentable": len(rejected),
            "rejected_compute_seconds": rejected_compute_seconds,
            "rejected": rejected,
        }
        if support_count:
            diagnostics.update(support_min=support_min, support_max=support_max)
        return accepted, values, diagnostics

    def _native_predictor(
        self,
        event_obs: Sequence[PreparedObservation],
        rgb_obs: Sequence[PreparedObservation],
        event_values: Sequence[tuple[np.ndarray, np.ndarray]],
        rgb_values: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[FinalV16Train40Predictor, dict[str, object], dict[str, object]]:
        event_ids = [f"event_{index:03d}" for index in range(len(event_obs))]
        rgb_ids = [f"rgb_{index:03d}" for index in range(len(rgb_obs))]
        inputs: dict[str, object] = {
            "event_observations": {
                identity: {"anchor_us": item.anchor_us, "available_us": item.available_us}
                for identity, item in zip(event_ids, event_obs, strict=True)
            },
            "rgb_observations": {
                identity: {"anchor_us": item.anchor_us, "available_us": item.available_us}
                for identity, item in zip(rgb_ids, rgb_obs, strict=True)
            },
        }
        native = FinalV16Train40Predictor.__new__(FinalV16Train40Predictor)
        native.device = self.device
        native.models = self.models
        native.normalizers = self.normalizers
        native.inputs = inputs
        native.event_values = dict(zip(event_ids, event_values, strict=True))
        native.rgb_values = dict(zip(rgb_ids, rgb_values, strict=True))
        query: dict[str, object] = {"event_history": event_ids, "rgb_history": rgb_ids}
        return native, inputs, query

    def predict(
        self,
        events: Sequence[PreparedObservation],
        rgb: Sequence[PreparedObservation],
        query_us: int,
    ) -> RuntimePrediction:
        """Predict from real causal support, or return an explicit unavailable result."""
        total_start = time.perf_counter()
        selected_events = self._select(events, "event", query_us)
        selected_rgb = self._select(rgb, "rgb", query_us)
        selection_seconds = time.perf_counter() - total_start
        if not selected_events:
            elapsed = time.perf_counter() - total_start
            return RuntimePrediction(
                status="unavailable",
                model_ttc=None,
                event_ttc=None,
                rgb_ttc=None,
                model_reason="no_causally_available_t2_event_observation",
                event_reason="no_causally_available_t2_event_observation",
                rgb_reason=(
                    "available_but_event_support_required"
                    if selected_rgb
                    else "no_causally_available_t2_rgb_observation"
                ),
                event_history_count=0,
                rgb_history_count=len(selected_rgb),
                selection_seconds=selection_seconds,
                producer_seconds=0.0,
                context_seconds=0.0,
                total_seconds=elapsed,
                producer_diagnostics={},
            )

        producer_start = time.perf_counter()
        selected_events, event_values, event_diagnostics = self._prepare_values(
            selected_events, "E"
        )
        selected_rgb, rgb_values, rgb_diagnostics = self._prepare_values(selected_rgb, "R")
        producer_seconds = time.perf_counter() - producer_start
        rgb_unavailable_reason = (
            "rgb_observations_unrepresentable"
            if not selected_rgb
            and cast(int, rgb_diagnostics["observations_rejected_unrepresentable"]) > 0
            else "no_causally_available_t2_rgb_observation"
        )
        if not selected_events:
            elapsed = time.perf_counter() - total_start
            return RuntimePrediction(
                status="unavailable",
                model_ttc=None,
                event_ttc=None,
                rgb_ttc=None,
                model_reason="event_observations_unrepresentable",
                event_reason="event_observations_unrepresentable",
                rgb_reason=(
                    "available_but_event_support_required"
                    if selected_rgb
                    else rgb_unavailable_reason
                ),
                event_history_count=0,
                rgb_history_count=len(selected_rgb),
                selection_seconds=selection_seconds,
                producer_seconds=producer_seconds,
                context_seconds=0.0,
                total_seconds=elapsed,
                producer_diagnostics={
                    "event": event_diagnostics,
                    "rgb": rgb_diagnostics,
                },
            )
        native, inputs, query = self._native_predictor(
            selected_events, selected_rgb, event_values, rgb_values
        )
        query["query_time_us"] = int(query_us)

        captured: dict[str, dict[str, Tensor]] = {}

        def capture(
            name: str,
        ) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Tensor]], None]:
            def hook(
                module: nn.Module, arguments: tuple[Any, ...], output: dict[str, Tensor]
            ) -> None:
                del module, arguments
                captured[name] = output

            return hook

        hooks = [self.models["E_CTX_MATCHED"].register_forward_hook(capture("event"))]
        if selected_rgb:
            hooks.append(self.models["R_CTX"].register_forward_hook(capture("rgb")))
        context_start = time.perf_counter()
        try:
            with torch.inference_mode():
                model_ttc = float(native.predict(query, inputs))
        finally:
            for hook in hooks:
                hook.remove()
        context_seconds = time.perf_counter() - context_start
        if "event" not in captured or (selected_rgb and "rgb" not in captured):
            raise RuntimeError("native context outputs were not captured")
        event_ttc = float(phase_to_ttc(captured["event"]["point_phase"])[0].cpu())
        rgb_ttc = (
            float(phase_to_ttc(captured["rgb"]["point_phase"])[0].cpu()) if selected_rgb else None
        )
        total_seconds = time.perf_counter() - total_start
        return RuntimePrediction(
            status="ok",
            model_ttc=model_ttc,
            event_ttc=event_ttc,
            rgb_ttc=rgb_ttc,
            model_reason=("fused_event_rgb" if selected_rgb else "event_only_fallback"),
            event_reason="available",
            rgb_reason=("available" if selected_rgb else rgb_unavailable_reason),
            event_history_count=len(selected_events),
            rgb_history_count=len(selected_rgb),
            selection_seconds=selection_seconds,
            producer_seconds=producer_seconds,
            context_seconds=context_seconds,
            total_seconds=total_seconds,
            producer_diagnostics={
                "event": event_diagnostics,
                "rgb": rgb_diagnostics,
            },
        )
