"""Native PHASE transport without changing the historical TTC route."""

from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.detection_phase_service.phase_context import BatchedPhaseContextRuntime
from operational.final_refit.training import PhaseOnlyFinal
from operational.train40_runtime.batched_context import BatchedContextRuntime, ContextObservation
from operational.train40_runtime.context import HEAD


def bundle(*, zero: bool = False):
    torch.manual_seed(941)
    models = {
        "E_CTX_MATCHED": TemporalRefiner(TemporalConfig(hidden=160)),
        "R_CTX": TemporalRefiner(TemporalConfig(hidden=160)),
        HEAD: PhaseOnlyFinal(),
    }
    for model in models.values():
        if zero:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
        model.eval().requires_grad_(False)
    normalizers = {
        name: (torch.zeros(17, dtype=torch.float32), torch.ones(17, dtype=torch.float32))
        for name in ("event", "rgb")
    }
    return models, normalizers


def observation(identity: str, *, rgb: bool = True, zero: bool = False) -> ContextObservation:
    query = 2_000_000
    value = torch.zeros if zero else torch.randn
    rgb_count = 5 if rgb else 0
    return ContextObservation(
        identity,
        value(4, 17, dtype=torch.float32),
        value(4, 3, dtype=torch.float32) * (0 if zero else 0.01),
        [query - lag for lag in (300_000, 200_000, 100_000, 0)],
        value(rgb_count, 17, dtype=torch.float32),
        value(rgb_count, 3, dtype=torch.float32) * (0 if zero else 0.01),
        [
            {
                "first_sensor_ts_us": query - (6 - index) * 100_000,
                "anchor_us": query - (4 - index) * 100_000,
                "available_us": query,
            }
            for index in range(rgb_count)
        ],
        query,
    )


def test_native_context_preserves_historical_predictions_exactly() -> None:
    models, normalizers = bundle()
    rows = [observation("rgb"), observation("event", rgb=False)]
    historical = BatchedContextRuntime(models, normalizers, "cpu").predict_many(rows)
    native = BatchedPhaseContextRuntime(models, normalizers, "cpu").predict_many(rows)
    for old, new in zip(historical, native, strict=True):
        assert new["predictions"] == old["predictions"]
        assert new["history_lengths"] == old["history_lengths"]
        assert new["rgb_available"] == old["rgb_available"]
    assert native[0]["phase"] is not None
    assert native[1]["phase"] is None
    assert native[1]["phase_unavailable_reason"] == "rgb_history_unavailable"


def test_native_bounds_equal_canonical_phase_only_forward() -> None:
    models, normalizers = bundle()
    runtime = BatchedPhaseContextRuntime(models, normalizers, "cpu")
    row = observation("rgb")
    inputs, _ = runtime._prepare([row])
    event_block = runtime.phase_core.event(*inputs[:4])
    rgb_block, width_logits = runtime.phase_core.rgb(*inputs[4:8])
    expected = models[HEAD](torch.cat((rgb_block, event_block), -1), width_logits)
    output = runtime.predict_many([row])[0]
    assert output["phase"] == {
        "point_phase": float(expected["point_phase"][0]),
        "q10": float(expected["q10"][0]),
        "q90": float(expected["q90"][0]),
    }
    assert output["predictions"]["PHASE_ONLY_TRAIN40_S7"] == float(expected["ttc"][0])


def test_native_phase_bounds_may_cross_zero_without_ttc_conversion() -> None:
    models, normalizers = bundle(zero=True)
    output = BatchedPhaseContextRuntime(models, normalizers, "cpu").predict_many(
        [observation("zero", zero=True)]
    )[0]
    phase = output["phase"]
    assert phase is not None
    assert phase["q10"] < 0 < phase["q90"]
    assert output["predictions"]["PHASE_ONLY_TRAIN40_S7"] == pytest.approx(60.0)


def test_native_context_rejects_graph_mode_explicitly() -> None:
    models, normalizers = bundle()
    try:
        BatchedPhaseContextRuntime(models, normalizers, "cpu", graphs=True)
    except ValueError as error:
        assert "graphs=False" in str(error)
    else:
        raise AssertionError("graph mode must be rejected")
