from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.evttc_competition.runtime import (
    CompetitionRuntime,
    PreparedObservation,
)
from operational.final_refit.test12 import FinalV16Train40Predictor
from operational.final_refit.training import PhaseOnlyFinal
from operational.rgb_port import transfer as engine
from operational.rgb_port_streaming.contracts import NativeObservation


def _models() -> dict[str, torch.nn.Module]:
    torch.manual_seed(7)
    config = TemporalConfig(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
    return {
        "E_CTX_MATCHED": TemporalRefiner(config).eval(),
        "R_CTX": TemporalRefiner(config).eval(),
        "PHASE_ONLY_EVENT_S7_TRAIN40": PhaseOnlyFinal().eval(),
    }


def _runtime() -> CompetitionRuntime:
    normalizer = (torch.zeros(17), torch.ones(17))
    return CompetitionRuntime(_models(), {"event": normalizer, "rgb": normalizer}, device="cpu")


def _observation(
    modality: str,
    timestamps_us: tuple[int, ...],
    *,
    available_us: int | None = None,
    first_sensor_ts_us: int | None = None,
    value: float = 0.25,
) -> PreparedObservation:
    channels = 12 if modality == "event" else 3
    pixels = np.full((len(timestamps_us), channels, 128, 128), value, np.float32)
    return PreparedObservation(
        modality=modality,  # type: ignore[arg-type]
        pixels=pixels,
        timestamps_us=timestamps_us,
        first_sensor_ts_us=(timestamps_us[0] if first_sensor_ts_us is None else first_sensor_ts_us),
        anchor_us=timestamps_us[-1],
        available_us=timestamps_us[-1] if available_us is None else available_us,
    )


def _fake_expert_block(
    models: object, prefix: str, sensor: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    del models
    batch = sensor.shape[0]
    base = sensor.mean(dim=(1, 2, 3, 4))
    cadence = delta.mean(dim=1)
    offset = 0.04 if prefix == "R" else -0.02
    features = torch.stack(
        [base + cadence * (index + 1) * 0.01 + offset for index in range(17)], dim=1
    )
    experts = torch.stack((base + 0.2, base + cadence + 0.3, base + 0.4), dim=1)
    assert features.shape == (batch, 17)
    return features.float(), experts.float()


class _DiagnosticProducer(torch.nn.Module):
    def __init__(self, *, invalid_when_high: bool, invalid_value: float = 0.05) -> None:
        super().__init__()
        self.invalid_when_high = invalid_when_high
        self.invalid_value = invalid_value

    def forward(self, sensor: torch.Tensor) -> SimpleNamespace:
        high = sensor.mean(dim=(1, 2, 3, 4)) > 0.5
        ttc = torch.where(
            high & self.invalid_when_high,
            torch.full_like(high, self.invalid_value, dtype=torch.float32),
            torch.full_like(high, 0.5, dtype=torch.float32),
        )
        support = torch.ones((len(sensor), sensor.shape[1]), dtype=torch.float32)
        # Deliberately true: domain rejection must not depend on confidence.
        known = torch.ones((len(sensor),), dtype=torch.bool)
        return SimpleNamespace(
            ttc_mean_seconds=ttc,
            known_mask=known,
            sensor_support=support,
        )


def _domain_runtime() -> CompetitionRuntime:
    runtime = _runtime()
    runtime.models.update(
        {
            "E_A5_MATCHED": _DiagnosticProducer(invalid_when_high=False),
            "E_C2F_MATCHED": _DiagnosticProducer(invalid_when_high=True),
            "R_A5": _DiagnosticProducer(invalid_when_high=False),
            "R_C2F": _DiagnosticProducer(invalid_when_high=True),
        }
    )
    return runtime


def _domain_expert_block(
    models: object, prefix: str, sensor: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(models, dict)
    names = ("E_A5_MATCHED", "E_C2F_MATCHED") if prefix == "E" else ("R_A5", "R_C2F")
    outputs = [models[name](sensor) for name in names]
    if any(
        bool(((output.ttc_mean_seconds > 0) & (output.ttc_mean_seconds <= 0.1)).any())
        for output in outputs
    ):
        raise ValueError("TTC lies outside the signed benchmark-phase domain")
    return _fake_expert_block(models, prefix, sensor, delta)


def test_prepared_observation_requires_real_t2_and_measured_positive_time() -> None:
    with pytest.raises(ValueError, match="T2 or T3"):
        _observation("event", (20_000,))
    with pytest.raises(ValueError, match="strictly increasing"):
        _observation("event", (20_000, 20_000))
    observation = _observation("event", (20_000, 57_000, 143_000))
    np.testing.assert_allclose(observation.delta_t_seconds, [0.037, 0.086], rtol=0, atol=1e-7)
    event_window = _observation("event", (100_000, 200_000, 300_000), first_sensor_ts_us=0)
    assert event_window.first_sensor_ts_us == 0
    with pytest.raises(ValueError, match="RGB first_sensor"):
        _observation("rgb", (100_000, 200_000), first_sensor_ts_us=0)


def test_no_real_event_t2_returns_explicit_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_expert_block", _fake_expert_block)
    result = _runtime().predict([], [_observation("rgb", (10_000, 20_000))], 20_000)
    assert result.status == "unavailable"
    assert result.model_ttc is None and result.event_ttc is None and result.rgb_ttc is None
    assert result.model_reason == "no_causally_available_t2_event_observation"
    assert result.event_history_count == 0


def test_cold_start_uses_one_real_t2_and_event_only_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[np.ndarray] = []

    def recording_block(models: object, prefix: str, sensor: torch.Tensor, delta: torch.Tensor):
        seen.append(delta.detach().cpu().numpy().copy())
        return _fake_expert_block(models, prefix, sensor, delta)

    monkeypatch.setattr(engine, "_expert_block", recording_block)
    observation = _observation("event", (4_000, 20_000), value=0.1)
    result = _runtime().predict([observation], [], 20_000)
    assert result.status == "ok"
    assert result.event_history_count == 1 and result.rgb_history_count == 0
    assert result.model_ttc == pytest.approx(result.event_ttc)
    assert result.rgb_ttc is None
    assert result.model_reason == "event_only_fallback"
    np.testing.assert_allclose(seen[0], [[0.016]], rtol=0, atol=1e-8)


def test_full_history_matches_unchanged_final_predictor_and_preserves_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, np.ndarray]] = []

    def recording_block(models: object, prefix: str, sensor: torch.Tensor, delta: torch.Tensor):
        captured.append((prefix, delta.detach().cpu().numpy().copy()))
        return _fake_expert_block(models, prefix, sensor, delta)

    monkeypatch.setattr(engine, "_expert_block", recording_block)
    runtime = _runtime()
    context_calls = {"event": 0, "rgb": 0}

    def count_event(*args: object) -> None:
        del args
        context_calls["event"] += 1

    def count_rgb(*args: object) -> None:
        del args
        context_calls["rgb"] += 1

    event_hook = runtime.models["E_CTX_MATCHED"].register_forward_hook(count_event)
    rgb_hook = runtime.models["R_CTX"].register_forward_hook(count_rgb)
    events = [
        _observation("event", (anchor - 190_000, anchor - 90_000, anchor), value=0.10 + i * 0.02)
        for i, anchor in enumerate((300_000, 400_000, 500_000, 600_000))
    ]
    rgbs = [
        _observation("rgb", (anchor - 82_000, anchor), value=0.30 + i * 0.03)
        for i, anchor in enumerate((440_000, 520_000, 600_000))
    ]
    try:
        result = runtime.predict(events, rgbs, 600_000)
    finally:
        event_hook.remove()
        rgb_hook.remove()

    # Rebuild the historical native cache contract independently, then invoke
    # the unchanged FinalV16Train40Predictor prediction method.
    event_values = []
    for item in events:
        raw, expert = _fake_expert_block(
            runtime.models,
            "E",
            torch.from_numpy(np.array(item.pixels, copy=True))[None],
            torch.from_numpy(item.delta_t_seconds.copy())[None],
        )
        event_values.append((raw[0].numpy(), expert[0].numpy()))
    rgb_values = []
    for item in rgbs:
        raw, expert = _fake_expert_block(
            runtime.models,
            "R",
            torch.from_numpy(np.array(item.pixels, copy=True))[None],
            torch.from_numpy(item.delta_t_seconds.copy())[None],
        )
        rgb_values.append((raw[0].numpy(), expert[0].numpy()))
    native, inputs, query = runtime._native_predictor(events, rgbs, event_values, rgb_values)
    query["query_time_us"] = 600_000
    expected = FinalV16Train40Predictor.predict(native, query, inputs)

    assert result.status == "ok"
    assert result.event_history_count == 4 and result.rgb_history_count == 3
    assert result.model_ttc == pytest.approx(expected, rel=1e-7, abs=1e-7)
    assert result.event_ttc is not None and result.rgb_ttc is not None
    assert result.model_reason == "fused_event_rgb"
    assert context_calls == {"event": 1, "rgb": 1}
    assert all(item[1].shape[1] in {1, 2} for item in captured)
    np.testing.assert_allclose(captured[0][1], [[0.1, 0.09]], rtol=0, atol=1e-7)
    np.testing.assert_allclose(captured[-1][1], [[0.082]], rtol=0, atol=1e-7)


def test_future_or_over_budget_observations_are_not_treated_as_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "_expert_block", _fake_expert_block)
    causal = _observation("event", (550_000, 600_000))
    future = _observation("event", (610_000, 620_000))
    stale = _observation("event", (-100_000, -90_000))
    result = _runtime().predict([future, stale, causal], [], 600_000)
    assert result.status == "ok"
    assert result.event_history_count == 1


def test_budget_counts_start_of_first_event_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "_expert_block", _fake_expert_block)
    valid = _observation("event", (500_000, 600_000), first_sensor_ts_us=400_000, value=0.2)
    with pytest.raises(ValueError, match="650 ms information budget"):
        _observation("event", (500_000, 600_000), first_sensor_ts_us=-100_000, value=0.3)
    result = _runtime().predict([valid], [], 600_000)
    assert result.status == "ok"
    assert result.event_history_count == 1


def test_unrepresentable_latest_event_retains_fixed_valid_past(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "_expert_block", _domain_expert_block)
    past = _observation("event", (100_000, 200_000), value=0.1)
    latest = _observation("event", (200_000, 300_000), value=0.9)
    result = _domain_runtime().predict([past, latest], [], 300_000)
    assert result.status == "ok"
    assert result.event_history_count == 1
    diagnostics = result.producer_diagnostics["event"]
    assert diagnostics["observations_accepted"] == 1
    assert diagnostics["accepted_observation_indices"] == [0]
    assert diagnostics["accepted_anchor_us"] == [200_000]
    assert diagnostics["observations_rejected_unrepresentable"] == 1
    rejected = diagnostics["rejected"]
    assert isinstance(rejected, list)
    invalid = rejected[0]["invalid_producers"]
    assert invalid[0]["producer_id"] == "E_C2F_MATCHED"
    assert invalid[0]["ttc_seconds"] == [pytest.approx(0.05)]
    assert invalid[0]["known_mask"] == [True]


def test_all_unrepresentable_events_return_null_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "_expert_block", _domain_expert_block)
    result = _domain_runtime().predict(
        [_observation("event", (200_000, 300_000), value=0.9)],
        [_observation("rgb", (200_000, 300_000), value=0.9)],
        300_000,
    )
    assert result.status == "unavailable"
    assert result.model_ttc is None
    assert result.model_reason == "event_observations_unrepresentable"
    assert result.rgb_reason == "rgb_observations_unrepresentable"
    assert result.producer_diagnostics["event"]["rejected_compute_seconds"] > 0


def test_unrepresentable_rgb_uses_native_event_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "_expert_block", _domain_expert_block)
    result = _domain_runtime().predict(
        [_observation("event", (200_000, 300_000), value=0.1)],
        [_observation("rgb", (200_000, 300_000), value=0.9)],
        300_000,
    )
    assert result.status == "ok"
    assert result.model_reason == "event_only_fallback"
    assert result.rgb_ttc is None and result.rgb_history_count == 0
    assert result.rgb_reason == "rgb_observations_unrepresentable"
    assert result.producer_diagnostics["rgb"]["observations_rejected_unrepresentable"] == 1


def test_unknown_producer_exception_still_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        del args
        raise ValueError("different invariant")

    monkeypatch.setattr(engine, "_expert_block", unexpected)
    with pytest.raises(ValueError, match="different invariant"):
        _domain_runtime().predict(
            [_observation("event", (200_000, 300_000), value=0.1)], [], 300_000
        )


def test_nan_rejection_diagnostic_is_strict_json(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _domain_runtime()
    runtime.models["E_C2F_MATCHED"] = _DiagnosticProducer(
        invalid_when_high=True, invalid_value=float("nan")
    )

    def nan_block(
        models: object, prefix: str, sensor: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(models, dict) and prefix == "E"
        del delta
        for name in ("E_A5_MATCHED", "E_C2F_MATCHED"):
            models[name](sensor)
        raise ValueError("TTC lies outside the signed benchmark-phase domain")

    monkeypatch.setattr(engine, "_expert_block", nan_block)
    result = runtime.predict([_observation("event", (200_000, 300_000), value=0.9)], [], 300_000)
    encoded = json.dumps(asdict(result), allow_nan=False)
    assert '"ttc_seconds": ["nan"]' in encoded
    assert '"domain_classification": ["unrepresentable_nan"]' in encoded


@pytest.mark.parametrize("infinite_ttc", [float("inf"), float("-inf")])
def test_infinite_ttc_native_feature_error_is_not_caught(
    monkeypatch: pytest.MonkeyPatch, infinite_ttc: float
) -> None:
    runtime = _domain_runtime()
    runtime.models["E_C2F_MATCHED"] = _DiagnosticProducer(
        invalid_when_high=True, invalid_value=infinite_ttc
    )

    def nonfinite_feature_error(
        models: object, prefix: str, sensor: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(models, dict) and prefix == "E"
        del delta
        for name in ("E_A5_MATCHED", "E_C2F_MATCHED"):
            models[name](sensor)
        raise ValueError("native producer features must be finite")

    monkeypatch.setattr(engine, "_expert_block", nonfinite_feature_error)
    with pytest.raises(ValueError, match="native producer features must be finite"):
        runtime.predict([_observation("event", (200_000, 300_000), value=0.9)], [], 300_000)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "EVTTC_CANDIDATE_ROOT" not in os.environ,
    reason="requires the sealed local candidate and CUDA",
)
def test_sealed_cuda_parity_with_native_transfer() -> None:
    """One controlled synthetic replay through the actual frozen weights."""
    root = Path(os.environ["EVTTC_CANDIDATE_ROOT"])
    runtime = CompetitionRuntime.from_candidate(root, device="cuda")
    events = [
        _observation(
            "event",
            (anchor - 200_000, anchor - 100_000, anchor),
            first_sensor_ts_us=anchor - 300_000,
            value=0.02 + index * 0.01,
        )
        for index, anchor in enumerate((300_000, 400_000, 500_000, 600_000))
    ]
    rgbs = [
        _observation(
            "rgb",
            (anchor - 80_000, anchor - 40_000, anchor),
            value=0.20 + index * 0.05,
        )
        for index, anchor in enumerate((480_000, 560_000, 600_000))
    ]

    def native_values(
        observations: list[PreparedObservation], prefix: str
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        values = []
        for item in observations:
            sensor = torch.from_numpy(np.array(item.pixels, copy=True))[None].cuda()
            delta = torch.from_numpy(item.delta_t_seconds.copy())[None].cuda()
            with (
                torch.inference_mode(),
                patch.object(engine, "ProducerObservation", NativeObservation),
            ):
                raw, experts = engine._expert_block(runtime.models, prefix, sensor, delta)
            values.append((raw[0].cpu().numpy(), experts[0].cpu().numpy()))
        return values

    event_values = native_values(events, "E")
    rgb_values = native_values(rgbs, "R")
    native, inputs, query = runtime._native_predictor(events, rgbs, event_values, rgb_values)
    query["query_time_us"] = 600_000
    with torch.inference_mode():
        expected = native.predict(query, inputs)
    actual = runtime.predict(events, rgbs, 600_000)
    assert actual.status == "ok"
    assert actual.model_ttc == pytest.approx(expected, rel=1e-7, abs=1e-7)
    assert actual.producer_diagnostics["event"]["known_total"] == 8
    assert actual.producer_diagnostics["event"]["support_count"] == 24
    assert actual.producer_diagnostics["rgb"]["known_total"] == 6
    assert actual.producer_diagnostics["rgb"]["support_count"] == 18
