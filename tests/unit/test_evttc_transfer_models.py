from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import operational.evttc_transfer.models as module
from e_jepa_ttc.simplex_t.phase import phase_to_ttc


class _Head(nn.Module):
    def __init__(self, phase: float) -> None:
        super().__init__()
        self.phase = phase

    def forward(self, *values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"point_phase": values[0].new_tensor([self.phase])}


class _Garl(nn.Module):
    dT = 0.1  # noqa: N815 - pinned upstream API

    def forward_test(self, sensor: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert sensor.shape == (1, 40, 128, 128)
        return sensor.new_tensor([[4.0, 5.0]]), None


def _runtime(monkeypatch: pytest.MonkeyPatch) -> module.FrozenModels:
    monkeypatch.setattr(
        module.FrozenModels,
        "_load_producers",
        lambda self: {"fixture": nn.Identity()},
    )
    monkeypatch.setattr(
        module.FrozenModels,
        "_load_h8_heads",
        lambda self: {7: _Head(0.01), 13: _Head(0.02), 23: _Head(0.03)},
    )
    monkeypatch.setattr(
        module.FrozenModels,
        "_load_normalizer",
        lambda self: (np.zeros(17, np.float64), np.ones(17, np.float64)),
    )
    monkeypatch.setattr(module.FrozenModels, "_load_public_garl", lambda self: _Garl())
    return module.FrozenModels(Path("fixture"))


def test_predict_preserves_canonical_batch_and_temporal_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(monkeypatch)
    observed: dict[str, object] = {}

    def fake_extract(
        name: str, models: dict[str, nn.Module], events: torch.Tensor, delta: torch.Tensor
    ) -> np.ndarray:
        observed["extract"] = (name, events.shape, delta.shape, float(delta[0, 0]))
        return np.arange(16 * 17, dtype=np.float32).reshape(16, 17)

    def fake_inputs(
        name: str,
        raw: np.ndarray,
        valid: np.ndarray,
        lag_us: np.ndarray,
        anchor_us: int,
        available_us: int,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observed["inputs"] = (
            name,
            raw.copy(),
            valid.copy(),
            lag_us.copy(),
            anchor_us,
            available_us,
        )
        return (
            torch.zeros(1, 8, 17),
            torch.zeros(1, 8, 4),
            torch.from_numpy(valid[None]),
            torch.zeros(1, 3),
        )

    monkeypatch.setattr(module, "canonical_extract", fake_extract)
    monkeypatch.setattr(module, "canonical_inputs", fake_inputs)
    valid = np.asarray([False, False, True, True, True, True, True, True])
    events = np.zeros((8, 3, 12, 128, 128), dtype=np.float32)
    detailed = runtime.predict_with_phase(
        events, delta_t_s=0.125, valid=valid, availability_lag_s=0.05
    )

    assert observed["extract"] == (
        "H8_SEED7",
        torch.Size([16, 3, 12, 128, 128]),
        torch.Size([16, 2]),
        0.125,
    )
    inputs = observed["inputs"]
    assert isinstance(inputs, tuple)
    assert inputs[0] == "H8_SEED7"
    np.testing.assert_array_equal(inputs[2], valid)
    np.testing.assert_array_equal(inputs[3], np.arange(350_000, -1, -50_000))
    assert inputs[4:] == (0, -50_000)
    raw = inputs[1]
    assert isinstance(raw, np.ndarray)
    assert not raw[:2].any()
    assert set(detailed) == {"H8_seed7", "H8_seed13", "H8_seed23"}
    expected_ttc = float(phase_to_ttc(torch.tensor([0.01], dtype=torch.float32))[0])
    assert detailed["H8_seed7"]["ttc"] == expected_ttc
    assert runtime.predict(events).keys() == detailed.keys()


def test_garl_predict_uses_native_unclipped_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(monkeypatch)
    result = runtime.garl_predict(np.zeros((40, 128, 128), dtype=np.float32))
    assert result["heights"] == [4.0, 5.0]
    assert result["ttc"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "events,match",
    [
        (np.zeros((8, 3, 12, 128, 128), dtype=np.float64), "float32"),
        (np.zeros((7, 3, 12, 128, 128), dtype=np.float32), "float32"),
    ],
)
def test_predict_rejects_noncanonical_events(
    monkeypatch: pytest.MonkeyPatch, events: np.ndarray, match: str
) -> None:
    runtime = _runtime(monkeypatch)
    with pytest.raises(ValueError, match=match):
        runtime.predict(events)


def test_predict_rejects_noncontiguous_validity(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(monkeypatch)
    events = np.zeros((8, 3, 12, 128, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="contiguous suffix"):
        runtime.predict(events, valid=np.asarray([True, False, True, True, True, True, True, True]))


def test_garl_rejects_wrong_shape_or_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(monkeypatch)
    with pytest.raises(ValueError, match="float32"):
        runtime.garl_predict(np.zeros((40, 128, 128), dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        runtime.garl_predict(np.zeros((39, 128, 128), dtype=np.float32))
