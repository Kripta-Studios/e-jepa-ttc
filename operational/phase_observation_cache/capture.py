"""Read-only instrumentation of the existing road and competition runtime seams."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTCOutput
from operational.evttc_competition.runtime import (
    CompetitionRuntime,
    PreparedObservation,
    RuntimePrediction,
)
from operational.final_refit.test12 import FinalV16Train40Predictor
from operational.rgb_port import transfer as engine
from scripts import final_train40_transfer as road_module


@contextmanager
def instrument_producers(
    models: dict, device: str
) -> Iterator[tuple[list[dict], dict[str, np.ndarray]]]:
    """Time the unchanged expert calls and preserve available support diagnostics.

    The engine patch is confined to the dedicated single-owner pilot process.
    It is restored on exit; no concurrent inference is allowed in this process.
    """
    original = engine._expert_block
    records: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    hooks = []
    active: dict[str, Any] = {}

    def hook(name: str) -> Callable:
        def capture(_module: torch.nn.Module, _args: tuple, output: CausalScaleTTCOutput) -> None:
            number = int(active["number"])
            for key in ("known_mask", "sensor_support", "ttc_mean_seconds"):
                value = getattr(output, key, None)
                if isinstance(value, torch.Tensor):
                    arrays[f"call{number}.{name}.{key}"] = value.detach().cpu().numpy().copy()

        return capture

    for name in ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"):
        hooks.append(models[name].register_forward_hook(hook(name)))

    def expert(
        model_map: dict, prefix: str, sensor: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active["number"] = len(records)
        if device == "cuda":
            torch.cuda.synchronize()
            start_event, end_event = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start_event.record()
        start = time.perf_counter()
        record = {
            "index": len(records),
            "prefix": prefix,
            "sensor_shape": list(sensor.shape),
            "delta_t_s": delta.detach().cpu().tolist(),
        }
        records.append(record)
        try:
            raw, experts = original(model_map, prefix, sensor, delta)
            arrays[f"call{record['index']}.raw17"] = raw.detach().cpu().numpy().copy()
            arrays[f"call{record['index']}.expert3"] = experts.detach().cpu().numpy().copy()
            record["status"] = "accepted"
            return raw, experts
        except Exception as error:
            record.update(status="rejected", error=repr(error))
            raise
        finally:
            if device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                record["cuda_elapsed_ms"] = start_event.elapsed_time(end_event)
            record["wall_seconds"] = time.perf_counter() - start

    try:
        with patch.object(engine, "_expert_block", expert):
            yield records, arrays
    finally:
        for handle in hooks:
            handle.remove()


def capture_road(runtime: road_module.FinalTransferPredictor, row: dict) -> tuple[dict, dict, dict]:
    """Capture the native object populated by the unchanged road predict method."""
    captured: dict[str, Any] = {}
    original_rgb, original_event = engine.raw_rgb_observations, engine.prepare_event

    class CapturedNative(FinalV16Train40Predictor):
        def predict(self, query: Mapping[str, Any], inputs: Mapping[str, Any]) -> float:
            captured["native"], captured["query"] = self, query
            return super().predict(query, inputs)

    def rgb(source: Mapping[str, Any]) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], str]:
        start = time.perf_counter()
        values, reason = original_rgb(source)
        captured["rgb_preparation_seconds"] = time.perf_counter() - start
        captured["rgb_prepared"] = [metadata for _, metadata in values]
        return values, reason

    def event(source: Mapping[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        values = original_event(source)
        captured["event_preparation_seconds"] = time.perf_counter() - start
        return values

    with (
        patch.object(road_module, "FinalV16Train40Predictor", CapturedNative),
        patch.object(engine, "raw_rgb_observations", rgb),
        patch.object(engine, "prepare_event", event),
    ):
        result = runtime.predict(row)
    native, query = captured["native"], captured["query"]
    observations: dict[str, list[dict]] = {"event": [], "rgb": []}
    windows = np.array(row["windows_us"], np.int64)
    for identity, lag in zip(query["event_history"], engine.EVENT_LAGS_US, strict=True):
        shifted = windows - int(lag)
        raw, experts = native.event_values[identity]
        observations["event"].append(
            {
                **native.inputs["event_observations"][identity],
                "first_sensor_ts_us": int(shifted[0, 0]),
                "timestamps_us": shifted[:, 1].tolist(),
                "windows_us": shifted.tolist(),
                "raw17": raw,
                "expert3": experts,
            }
        )
    rgb_metadata = {item["anchor_us"]: item for item in captured["rgb_prepared"]}
    for identity in query["rgb_history"]:
        metadata = native.inputs["rgb_observations"][identity]
        raw, experts = native.rgb_values[identity]
        observations["rgb"].append(
            {**rgb_metadata[metadata["anchor_us"]], **metadata, "raw17": raw, "expert3": experts}
        )
    accepted = {item["anchor_us"] for item in observations["rgb"]}
    details = {
        "event_preparation_seconds": captured["event_preparation_seconds"],
        "rgb_preparation_seconds": captured["rgb_preparation_seconds"],
        "rgb_prepared": [
            {**item, "accepted": item["anchor_us"] in accepted} for item in captured["rgb_prepared"]
        ],
        "precision_scope": "same freshly instrumented extraction; no historical bitwise claim",
    }
    return result, observations, details


class CapturingCompetitionRuntime(CompetitionRuntime):
    """Keep returned accepted raw17/expert3 values without changing runtime selection."""

    def __init__(self, models: dict, normalizers: dict, *, device: str) -> None:
        super().__init__(models, normalizers, device=device)
        self.observations: dict[str, list[dict]] = {"event": [], "rgb": []}
        self.requested: dict[str, list[dict]] = {}

    def predict(
        self,
        events: Sequence[PreparedObservation],
        rgb: Sequence[PreparedObservation],
        query_us: int,
    ) -> RuntimePrediction:
        """Reset query-owned captures before any native early return or failure."""
        self.observations = {"event": [], "rgb": []}
        self.requested = {}
        return super().predict(events, rgb, query_us)

    def _prepare_values(
        self, observations: Sequence[PreparedObservation], prefix: Literal["E", "R"]
    ) -> tuple[list[PreparedObservation], list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
        accepted, values, diagnostics = super()._prepare_values(observations, prefix)
        modality = "event" if prefix == "E" else "rgb"

        def metadata(item: PreparedObservation) -> dict:
            return {
                "first_sensor_ts_us": item.first_sensor_ts_us,
                "timestamps_us": list(item.timestamps_us),
                "anchor_us": item.anchor_us,
                "available_us": item.available_us,
            }

        self.observations[modality] = [
            {**metadata(item), "raw17": value[0], "expert3": value[1]}
            for item, value in zip(accepted, values, strict=True)
        ]
        self.requested[modality] = [
            {**metadata(item), "accepted": any(item is a for a in accepted)}
            for item in observations
        ]
        return accepted, values, diagnostics
