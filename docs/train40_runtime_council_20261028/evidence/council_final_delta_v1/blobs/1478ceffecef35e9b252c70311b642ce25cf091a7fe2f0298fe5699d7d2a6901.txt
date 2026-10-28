"""CPU doubles for the exact packed H8 extractor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from operational.simplex_t_shared_route.adapter import extract as canonical_extract
from operational.train40_system import h8_fast_admission
from operational.train40_system.h8_fast_extract import H8Extractor


class Producer(nn.Module):
    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = shift

    def forward(
        self, events: torch.Tensor, delta: torch.Tensor, *, return_dense_features: bool
    ) -> Any:
        assert return_dense_features
        base = events[:, -1, 0, 0, 0] + self.shift
        support = torch.stack((base + 0.2, base + 0.3, base + 0.4), 1)
        return SimpleNamespace(
            ttc_mean_seconds=base + 2.0,
            ttc_log_variance=base - 0.5,
            log_height_ratio=torch.stack((base, base + 0.001, base + 0.002), 1),
            sensor_support=support,
            pair_tokens=base[:, None, None].expand(-1, 3, 128),
            diagnostics={
                "transport_flow_magnitude": torch.stack((base + 3.0, base + 4.0, base + 5.0), 1)
            },
        )


class Pair(nn.Module):
    def predict_ttc(self, batch: Any) -> torch.Tensor:
        return batch.features[:, 0] + 4.0


class ConstantPair(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def predict_ttc(self, batch: Any) -> torch.Tensor:
        return torch.full(
            (len(batch.features),), self.value, dtype=torch.float32, device=batch.features.device
        )


class InfiniteProducer(Producer):
    def forward(
        self, events: torch.Tensor, delta: torch.Tensor, *, return_dense_features: bool
    ) -> Any:
        output = super().forward(events, delta, return_dense_features=return_dense_features)
        output.ttc_mean_seconds[:] = float("inf")
        return output


def inputs(value: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    events = torch.full((16, 3, 12, 128, 128), value, dtype=torch.float32)
    delta = torch.full((16, 2), 0.1, dtype=torch.float32)
    return events, delta


def models() -> dict[str, nn.Module]:
    return {"A5": Producer(0.1).eval(), "C2F": Producer(0.4).eval(), "PAIR": Pair().eval()}


def test_packed_matches_canonical_exactly_with_one_transfer() -> None:
    frozen = models()
    events, delta = inputs()
    expected = canonical_extract("H8_SEED7", frozen, events, delta)
    extractor = H8Extractor(frozen, "packed")
    actual = extractor("H8_SEED7", frozen, events, delta)
    assert np.array_equal(actual, expected)
    assert extractor.snapshot()["device_to_host_transfers"] == 1
    assert extractor.snapshot()["fixed_shape"] == [16, 3, 12, 128, 128]


def test_repeated_a_b_a_inputs_are_exact_and_do_not_alias() -> None:
    frozen = models()
    a_events, delta = inputs(0.2)
    b_events, _ = inputs(0.6)
    extractor = H8Extractor(frozen)
    first = extractor("H8_SEED7", frozen, a_events, delta)
    middle = extractor("H8_SEED7", frozen, b_events, delta)
    last = extractor("H8_SEED7", frozen, a_events, delta)
    assert np.array_equal(first, last)
    assert not np.array_equal(first, middle)
    assert not np.shares_memory(first, last)


def test_graph_mode_compiles_only_the_two_producer_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool, bool]] = []

    def compile_model(model: Any, *, backend: str, dynamic: bool, fullgraph: bool) -> Any:
        calls.append((backend, dynamic, fullgraph))
        return model

    monkeypatch.setattr(torch, "compile", compile_model)
    frozen = models()
    events, delta = inputs()
    extractor = H8Extractor(frozen, "graph")
    assert np.array_equal(
        extractor("H8_SEED7", frozen, events, delta),
        canonical_extract("H8_SEED7", frozen, events, delta),
    )
    assert calls == [("cudagraphs", False, False), ("cudagraphs", False, False)]
    assert extractor.snapshot()["graph_enabled"] is True


def test_graph_mode_clones_a5_primitives_before_c2f_overwrites_static_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a5_output: Any = None

    def compile_model(model: Any, **_: Any) -> Any:
        def captured(*args: Any, **kwargs: Any) -> Any:
            nonlocal a5_output
            output = model(*args, **kwargs)
            if model is frozen["A5"]:
                a5_output = output
            elif a5_output is not None:
                a5_output.ttc_mean_seconds.fill_(-999)
                a5_output.ttc_log_variance.fill_(-999)
                a5_output.log_height_ratio.fill_(-999)
                a5_output.sensor_support.fill_(-999)
                a5_output.diagnostics["transport_flow_magnitude"].fill_(-999)
            return output

        return captured

    monkeypatch.setattr(torch, "compile", compile_model)
    frozen = models()
    events, delta = inputs()
    expected = canonical_extract("H8_SEED7", frozen, events, delta)
    actual = H8Extractor(frozen, "graph")("H8_SEED7", frozen, events, delta)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda events, delta: (events.half(), delta), "FP32"),
        (lambda events, delta: (events[:8], delta[:8]), "fixed B16"),
        (lambda events, delta: (events, -delta), "positive producer deltas"),
    ],
)
def test_guards_reject_scientific_contract_changes(mutation: Any, message: str) -> None:
    frozen = models()
    events, delta = mutation(*inputs())
    with pytest.raises(ValueError, match=message):
        H8Extractor(frozen)("H8_SEED7", frozen, events, delta)


def test_model_identity_and_full_family_are_required() -> None:
    frozen = models()
    events, delta = inputs()
    extractor = H8Extractor(frozen)
    with pytest.raises(ValueError, match="identity"):
        extractor("H8_SEED7", dict(frozen), events, delta)
    with pytest.raises(ValueError, match="full expert family"):
        extractor("A5_ONLY_C0", frozen, events, delta)


def test_admission_row_selection_is_bounded_and_includes_existing_fragments(
    tmp_path: Path,
) -> None:
    sequences = np.asarray(["a"] * 40 + ["b"] * 40 + ["c"] * 1_920 + ["d"] * 6_172 + ["e"] * 1_000)
    fragments = tmp_path / "h8_feature_fragments"
    fragments.mkdir()
    (fragments / "query_00500.json").write_text("{}", encoding="utf-8")
    np.savez(fragments / "query_00500.npz", features=np.zeros((8, 17), np.float32))
    for row in (7188, 7189):
        (fragments / f"query_{row:05d}.json").write_text("{}", encoding="utf-8")
        np.savez(fragments / f"query_{row:05d}.npz", features=np.zeros((8, 17), np.float32))
    rows = h8_fast_admission._selected_rows(tmp_path, sequences)
    assert {0, 31, 40, 80, 1000, 2000, 7188, 7189, 8172} <= set(rows)
    assert len(rows) <= 12


def test_pair_infinity_is_allowed_with_exact_signed_zero_phase() -> None:
    frozen = models()
    frozen["PAIR"] = ConstantPair(float("inf")).eval()
    events, delta = inputs()
    expected = canonical_extract("H8_SEED7", frozen, events, delta)
    actual = H8Extractor(frozen)("H8_SEED7", frozen, events, delta)
    assert h8_fast_admission._exact(actual, expected)
    assert np.signbit(actual[:, 10]).tolist() == np.signbit(expected[:, 10]).tolist()


def test_pair_nan_and_producer_infinity_are_rejected() -> None:
    events, delta = inputs()
    pair_nan = models()
    pair_nan["PAIR"] = ConstantPair(float("nan")).eval()
    with pytest.raises(ValueError, match="NaN permitted PAIR"):
        H8Extractor(pair_nan)("H8_SEED7", pair_nan, events, delta)
    producer_inf = models()
    producer_inf["C2F"] = InfiniteProducer(0.4).eval()
    with pytest.raises(ValueError, match="nonfinite permitted A5/C2F"):
        H8Extractor(producer_inf)("H8_SEED7", producer_inf, events, delta)


def test_job_registers_hdf5_plugin_before_opening_raw_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import h5py

    from e_jepa_ttc.data import eap

    raw = tmp_path / "data/train/sequence-a/events.h5"
    raw.parent.mkdir(parents=True)
    with h5py.File(raw, "w") as handle:
        handle.create_dataset("events/t", data=np.asarray([10, 20], np.int64))
    registered = False
    original_file = h5py.File

    def register() -> None:
        nonlocal registered
        registered = True

    def checked_file(*args: Any, **kwargs: Any) -> Any:
        assert registered
        return original_file(*args, **kwargs)

    monkeypatch.setattr(eap, "_require_hdf5plugin", register)
    monkeypatch.setattr(h5py, "File", checked_file)
    index = {
        "sequences": np.asarray(["sequence-a"]),
        "windows_us": np.asarray([[[10, 11], [12, 13], [14, 15]]], np.int64),
        "square_xyxy": np.asarray([[0, 0, 4, 4]], np.float32),
    }
    job, valid = h8_fast_admission._job(tmp_path, 0, index)
    assert registered
    assert job["path"] == str(raw)
    assert valid.shape == (8,)
