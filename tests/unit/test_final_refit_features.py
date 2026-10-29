from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase
from operational.final_refit.features import (
    ROLE,
    build_context_cache,
    build_observation_cache,
    build_pair_cache,
    build_phase_cache,
    fit_train40_normalizer,
    load_sealed_cache,
    normalizer_population_binding,
    sha256_file,
    validate_cache_receipt,
)

PARENT = {"frozen_endpoint": "a" * 64}
NATIVE_ROWS_SHA256 = "e" * 64


def _pair_source(modality: str) -> dict[str, np.ndarray]:
    del modality
    return {
        "token128": np.arange(256, dtype=np.float32).reshape(2, 128),
        "delta_t_s": np.asarray([[0.2, 0.1], [0.3, 0.25]], np.float32),
        "support": np.asarray([[0.9, 0.7, 0.8], [0.3, 0.6, 0.5]], np.float32),
        "target_phase": np.asarray([0.1, -0.2], np.float32),
        "sequence_id": np.asarray(["s0", "s1"]),
        "sample_token": np.asarray(["q0", "q1"]),
        "group_id": np.asarray(["g0", "g1"]),
        "role": np.asarray([ROLE, ROLE]),
    }


def _observation_source(modality: str) -> dict[str, np.ndarray]:
    offset = 0.5 if modality == "rgb" else -0.5
    return {
        "observation_id": np.asarray(["o0", "o1", "o2"]),
        "base_statistics2": np.asarray([[1, 2], [3, 4], [5, 6]], np.float32) + offset,
        "a5_diagnostics3": np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5]], np.float32),
        "c2f_diagnostics3": np.asarray([[2, 3, 4], [3, 4, 5], [4, 5, 6]], np.float32),
        "expert_phase": np.asarray([[0.1, 0.2, 0.3], [0.2, 0.4, 0.5], [0.4, 0.1, 0.7]], np.float32),
        "anchor_us": np.asarray([10, 20, 30], np.int64),
        "available_us": np.asarray([10, 22, 35], np.int64),
        "input_span_us": np.asarray([5, 5, 5], np.int64),
        "group_id": np.asarray(["g0", "g0", "g0"]),
        "role": np.asarray([ROLE, ROLE, ROLE]),
    }


def _history() -> dict[str, np.ndarray]:
    return {
        "observation_id": np.asarray([["", "o0", "o1"], ["o0", "o1", "o2"]]),
        "timing": np.asarray(
            [
                [[0, 0, 0, 0], [20, 20, 0, 0], [10, 8, 10, 2]],
                [[30, 30, 0, 0], [20, 18, 10, 2], [10, 5, 10, 5]],
            ],
            np.float32,
        )
        / 1_000_000,
        "valid": np.asarray([[False, True, True], [True, True, True]]),
        "expert_phase": np.asarray([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], np.float32),
        "target_phase": np.asarray([0.25, -0.5], np.float32),
        "sample_token": np.asarray(["q0", "q1"]),
        "group_id": np.asarray(["g0", "g0"]),
        "role": np.asarray([ROLE, ROLE]),
    }


def _read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _population(
    observations: dict[str, np.ndarray], history: dict[str, np.ndarray] | None = None
) -> dict[str, object]:
    return normalizer_population_binding(
        observations,
        _history() if history is None else history,
        native_train40_rows_sha256="c" * 64,
        observation_receipt_sha256="d" * 64,
    )


class _FakeContext(nn.Module):
    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = shift
        self.location = nn.Linear(160, 1)
        self.width = nn.Linear(160, 2)
        with torch.no_grad():
            self.location.weight.fill_(0.001)
            self.location.bias.fill_(shift)
            self.width.weight.zero_()
            self.width.bias.copy_(torch.tensor([0.2, 0.4]))

    def forward(
        self, features: Tensor, timing: Tensor, valid: Tensor, experts: Tensor
    ) -> dict[str, Tensor]:
        del valid
        source = features[:, -1].mean(dim=-1, keepdim=True) + timing[:, -1, :1] + experts[:, :1]
        hidden = source.expand(-1, 160)
        raw = self.location(hidden).squeeze(-1)
        width = 0.03 * functional.softplus(self.width(hidden))
        return {
            "raw_location": raw,
            "point_phase": emitted_phase(raw),
            "q10": (raw - width[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (raw + width[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
        }


@pytest.mark.parametrize("modality", ["event", "rgb"])
def test_pair_cache_exact_schema_and_feature_order(tmp_path: Path, modality: str) -> None:
    output = tmp_path / f"pair_{modality}.npz"
    receipt = build_pair_cache(
        _pair_source(modality),
        output,
        modality=modality,
        parents=PARENT,
        native_train40_rows_sha256=NATIVE_ROWS_SHA256,
        compressed=False,
    )
    values = load_sealed_cache(
        output,
        schema="final_refit_pair_cache_v1",
        exact_keys={"features", "target_phase", "group_id", "sample_token", "sequence_id", "role"},
    )
    assert values["features"].shape == (2, 133)
    np.testing.assert_array_equal(values["features"][:, :128], _pair_source(modality)["token128"])
    np.testing.assert_array_equal(
        values["features"][:, 128], np.asarray([0.1, 0.25], dtype=np.float32)
    )
    np.testing.assert_allclose(values["features"][:, 129], np.log([0.1, 0.25]), rtol=1e-6)
    np.testing.assert_allclose(values["features"][:, 130], [10, 4], rtol=0, atol=0)
    np.testing.assert_allclose(values["features"][:, 131:], [[0.8, 0.7], [0.5, 0.5]])
    assert receipt["modality"] == modality
    assert set(values["role"]) == {ROLE}
    assert validate_cache_receipt(output, "final_refit_pair_cache_v1") == receipt


def test_normalizer_and_context_are_train_only_causal_and_target_independent(
    tmp_path: Path,
) -> None:
    observations_path = tmp_path / "rgb_observations.npz"
    build_observation_cache(
        _observation_source("rgb"),
        observations_path,
        modality="rgb",
        parents=PARENT,
        compressed=False,
    )
    observations = _read(observations_path)
    normalizer_path = tmp_path / "rgb_normalizer.json"
    normalizer = fit_train40_normalizer(
        observations,
        normalizer_path,
        modality="rgb",
        parents=PARENT,
        population_binding=_population(observations),
    )
    assert normalizer["targets_read"] is False
    context_path = tmp_path / "rgb_context.npz"
    receipt = build_context_cache(
        observations,
        _history(),
        normalizer_path,
        context_path,
        modality="rgb",
        parents=PARENT,
        population_binding=_population(observations),
        compressed=False,
    )
    values = _read(context_path)
    assert set(values) == {
        "features",
        "timing",
        "valid",
        "expert_phase",
        "target_phase",
        "group_id",
        "sample_token",
        "role",
    }
    assert values["features"].shape == (2, 3, 17)
    np.testing.assert_array_equal(values["features"][0, 0], np.zeros(17, np.float32))
    mean = np.asarray(normalizer["mean"], np.float32)
    scale = np.asarray(normalizer["scale"], np.float32)
    np.testing.assert_allclose(
        values["features"][0, 1], (observations["features"][0] - mean) / scale
    )
    assert receipt["history_slots"] == 3
    validate_cache_receipt(context_path, "final_refit_context_cache_v1")


def test_context_rejects_normalizer_from_different_parent(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.npz"
    build_observation_cache(
        _observation_source("event"),
        observations_path,
        modality="event",
        parents=PARENT,
        compressed=False,
    )
    observations = _read(observations_path)
    normalizer_path = tmp_path / "normalizer.json"
    fit_train40_normalizer(
        observations,
        normalizer_path,
        modality="event",
        parents=PARENT,
        population_binding=_population(observations),
    )
    with pytest.raises(ValueError, match="normalizer identity"):
        build_context_cache(
            observations,
            _history(),
            normalizer_path,
            tmp_path / "bad.npz",
            modality="event",
            parents={"frozen_endpoint": "b" * 64},
            population_binding=_population(observations),
        )


def test_context_rejects_mutated_normalizer_payload(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.npz"
    build_observation_cache(
        _observation_source("event"),
        observations_path,
        modality="event",
        parents=PARENT,
        compressed=False,
    )
    observations = _read(observations_path)
    normalizer_path = tmp_path / "normalizer.json"
    fit_train40_normalizer(
        observations,
        normalizer_path,
        modality="event",
        parents=PARENT,
        population_binding=_population(observations),
    )
    payload = json.loads(normalizer_path.read_text(encoding="utf-8"))
    payload["mean"][0] += 1
    normalizer_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normalizer identity"):
        build_context_cache(
            observations,
            _history(),
            normalizer_path,
            tmp_path / "bad.npz",
            modality="event",
            parents=PARENT,
            population_binding=_population(observations),
        )


def test_context_rejects_cross_group_or_inconsistent_clock(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.npz"
    build_observation_cache(
        _observation_source("rgb"),
        observations_path,
        modality="rgb",
        parents=PARENT,
        compressed=False,
    )
    observations = _read(observations_path)
    normalizer_path = tmp_path / "normalizer.json"
    fit_train40_normalizer(
        observations,
        normalizer_path,
        modality="rgb",
        parents=PARENT,
        population_binding=_population(observations),
    )
    crossed = _history()
    crossed["group_id"][0] = "g1"
    with pytest.raises(ValueError, match="group boundary"):
        build_context_cache(
            observations,
            crossed,
            normalizer_path,
            tmp_path / "crossed.npz",
            modality="rgb",
            parents=PARENT,
            population_binding=_population(observations, crossed),
        )
    wrong_clock = _history()
    wrong_clock["timing"][0, -1, 0] += 0.1
    with pytest.raises(ValueError, match="query clocks"):
        build_context_cache(
            observations,
            wrong_clock,
            normalizer_path,
            tmp_path / "wrong_clock.npz",
            modality="rgb",
            parents=PARENT,
            population_binding=_population(observations),
        )


def test_normalizer_rejects_same_size_foreign_observation_inventory(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.npz"
    build_observation_cache(
        _observation_source("rgb"),
        observations_path,
        modality="rgb",
        parents=PARENT,
        compressed=False,
    )
    observations = _read(observations_path)
    expected = _population(observations)
    foreign = dict(observations)
    foreign["observation_id"] = np.asarray(["o0", "o1", "foreign"])
    with pytest.raises(ValueError, match="sealed external inventory"):
        fit_train40_normalizer(
            foreign,
            tmp_path / "foreign.json",
            modality="rgb",
            parents=PARENT,
            population_binding=expected,
        )


def test_normalizer_rejects_extra_unused_observation_even_with_fresh_receipt(
    tmp_path: Path,
) -> None:
    source = _observation_source("rgb")
    extended: dict[str, np.ndarray] = {}
    for name, value in source.items():
        if name == "observation_id":
            tail = np.asarray(["unused"])
        elif name == "group_id":
            tail = np.asarray(["g0"])
        elif name == "role":
            tail = np.asarray([ROLE])
        else:
            tail = np.asarray(value[-1:]).copy()
        extended[name] = np.concatenate((value, tail), axis=0)
    observations_path = tmp_path / "observations_with_unused.npz"
    build_observation_cache(
        extended,
        observations_path,
        modality="rgb",
        parents=PARENT,
        compressed=False,
    )
    observations = _read(observations_path)
    with pytest.raises(ValueError, match="exact referenced history set"):
        normalizer_population_binding(
            observations,
            _history(),
            native_train40_rows_sha256=NATIVE_ROWS_SHA256,
            observation_receipt_sha256=sha256_file(observations_path.with_suffix(".json")),
        )


def test_context_rejects_same_size_foreign_query_inventory(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.npz"
    build_observation_cache(
        _observation_source("event"),
        observations_path,
        modality="event",
        parents=PARENT,
        compressed=False,
    )
    observations = _read(observations_path)
    expected = _population(observations)
    normalizer_path = tmp_path / "normalizer.json"
    fit_train40_normalizer(
        observations,
        normalizer_path,
        modality="event",
        parents=PARENT,
        population_binding=expected,
    )
    foreign = _history()
    foreign["sample_token"] = np.asarray(["q0", "foreign"])
    with pytest.raises(ValueError, match="sealed TRAIN40 population"):
        build_context_cache(
            observations,
            foreign,
            normalizer_path,
            tmp_path / "foreign.npz",
            modality="event",
            parents=PARENT,
            population_binding=expected,
        )


def test_phase_cache_exact_r170_e170_order_and_receipt(tmp_path: Path) -> None:
    base = _history()
    rows, length = 2, 3
    context = {
        "features": np.arange(rows * length * 17, dtype=np.float32).reshape(rows, length, 17)
        / 1000,
        "timing": base["timing"],
        "valid": base["valid"],
        "expert_phase": base["expert_phase"],
        "target_phase": base["target_phase"],
        "group_id": base["group_id"],
        "sample_token": base["sample_token"],
        "role": base["role"],
    }
    rgb = {name: value.copy() for name, value in context.items()}
    event = {name: value.copy() for name, value in context.items()}
    event["features"] += 1
    output = tmp_path / "phase.npz"
    receipt = build_phase_cache(
        event,
        rgb,
        _FakeContext(0.2),
        _FakeContext(-0.1),
        output,
        parents={"E_CTX": "c" * 64, "R_CTX": "d" * 64},
        normalizers={"event": "e" * 64, "rgb": "f" * 64},
        native_train40_rows_sha256=NATIVE_ROWS_SHA256,
        batch_size=1,
        compressed=False,
    )
    values = _read(output)
    assert values["features"].shape == (2, 340)
    assert values["rgb_width_logits"].shape == (2, 2)
    assert np.all(values["features"][:, :160] != values["features"][:, 170:330])
    assert receipt["feature_order"] == "R170_then_E170"
    assert receipt["phase_only_selected_columns"] == [[160, 170], [330, 340]]
    assert receipt["zero_correction_rgb_parity"] is True
    validate_cache_receipt(output, "final_refit_phase_cache_v1")


def test_receipt_detects_cache_mutation(tmp_path: Path) -> None:
    output = tmp_path / "pair.npz"
    build_pair_cache(
        _pair_source("rgb"),
        output,
        modality="rgb",
        parents=PARENT,
        native_train40_rows_sha256=NATIVE_ROWS_SHA256,
    )
    output.write_bytes(output.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_cache_receipt(output, "final_refit_pair_cache_v1")


def test_receipt_payload_is_canonical(tmp_path: Path) -> None:
    output = tmp_path / "pair.npz"
    build_pair_cache(
        _pair_source("rgb"),
        output,
        modality="rgb",
        parents=PARENT,
        native_train40_rows_sha256=NATIVE_ROWS_SHA256,
    )
    sidecar = output.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["group_ids"] = list(reversed(payload["group_ids"]))
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_cache_receipt(output, "final_refit_pair_cache_v1")


def test_invalid_native_inventory_does_not_publish_orphan_cache(tmp_path: Path) -> None:
    output = tmp_path / "pair.npz"
    with pytest.raises(ValueError, match="native TRAIN40"):
        build_pair_cache(
            _pair_source("rgb"),
            output,
            modality="rgb",
            parents=PARENT,
            native_train40_rows_sha256="invalid",
        )
    assert not output.exists()
    assert not output.with_suffix(".json").exists()
