"""Retained observations rebuild clocks instead of copying an old context tensor."""

import json
from decimal import Decimal

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_observation_cache.capture import CapturingCompetitionRuntime
from operational.phase_observation_cache.integrity import load_bound_observations, seal_observations
from operational.phase_observation_cache.replay import native_from_observations, replay
from scripts.build_phase_native_context_augmentations import native_vector, select_indices
from scripts.final_train40_transfer import HEAD
from scripts.pilot_phase_observation_cache import (
    sensor_row_receipt,
    serialize_observations,
    verify_replays,
)


def observations():
    return {
        "event": [
            {
                "anchor_us": anchor,
                "available_us": 1_000_000,
                "first_sensor_ts_us": anchor - 300_000,
                "timestamps_us": [anchor - 200_000, anchor - 100_000, anchor],
                "windows_us": [
                    [anchor - 300_000, anchor - 200_000],
                    [anchor - 200_000, anchor - 100_000],
                    [anchor - 100_000, anchor],
                ],
                "raw17": np.full(17, i, np.float32),
                "expert3": np.full(3, i + 1, np.float32),
            }
            for i, anchor in enumerate([700_000, 800_000, 900_000, 1_000_000])
        ],
        "rgb": [],
    }


def test_short_context_resets_first_retained_interarrival():
    normalizers = {k: (torch.zeros(17), torch.ones(17)) for k in ("event", "rgb")}
    native, query = native_from_observations(
        {}, normalizers, observations(), 1_000_000, "cpu", keep=2
    )
    features, clocks, valid, experts = native._context(query["event_history"], "event", 1_000_000)
    np.testing.assert_array_equal(clocks[0, -2:, 2].numpy(), [0.0, np.float32(0.1)])
    assert valid.sum() == 2
    np.testing.assert_array_equal(features[0, -2:, 0].numpy(), [2, 3])
    np.testing.assert_array_equal(experts.numpy(), [[4, 4, 4]])


def test_full_replay_and_single_subset_do_not_mutate_source_or_forge_sensor_restart():
    source = observations()
    normalizers = {k: (torch.zeros(17), torch.ones(17)) for k in ("event", "rgb")}
    full, query = native_from_observations({}, normalizers, source, 1_000_000, "cpu")
    short, reduced = native_from_observations({}, normalizers, source, 1_000_000, "cpu", keep=1)
    assert len(query["event_history"]) == len(source["event"]) == 4
    assert len(reduced["event_history"]) == 1
    assert source["event"][-1]["first_sensor_ts_us"] == 700_000
    full.event_values[query["event_history"][0]][0][0] = -100
    assert source["event"][0]["raw17"][0] == 0
    _, timing, _, _ = short._context(reduced["event_history"], "event", 1_000_000)
    assert timing[0, -1, 2] == 0


def test_sensor_receipt_preserves_decimal_timestamp_without_float_rounding():
    row = {"timestamp_s": Decimal("1.563198"), "timestamp_text": "1.563198", "anchor_us": 1563198}
    value = sensor_row_receipt(row)
    assert json.loads(json.dumps(value))["timestamp_s"] == "1.563198"
    assert row["timestamp_s"] == Decimal("1.563198")


def test_early_unavailable_query_clears_previous_capture():
    runtime = CapturingCompetitionRuntime({}, {}, device="cpu")
    runtime.observations = {"event": [{"stale": True}], "rgb": [{"stale": True}]}
    runtime.requested = {"event": [{"stale": True}]}
    result = runtime.predict([], [], 1_000_000)
    assert result.status == "unavailable"
    assert runtime.observations == {"event": [], "rgb": []}
    assert runtime.requested == {}


def test_missing_event_observations_replay_unavailable_without_fabricating_fusion():
    value, arrays = replay({}, {}, {"event": [], "rgb": []}, 1_000_000, "cpu")
    assert value is None
    assert arrays == {}


def test_event_only_disk_replay_preserves_original_fallback(tmp_path):
    models = {
        "E_CTX_MATCHED": TemporalRefiner(TemporalConfig(hidden=160)).eval(),
        "R_CTX": torch.nn.Identity(),
        HEAD: torch.nn.Identity(),
    }
    normalizers = {"event": (torch.zeros(17), torch.ones(17))}
    source = observations()
    arrays, metadata = serialize_observations(source)
    save_npz(tmp_path / "OBSERVATIONS.npz", arrays)
    atomic_receipt(tmp_path / "OBSERVATIONS.json", metadata)
    prediction, original = replay(models, normalizers, source, 1_000_000, "cpu")
    result = verify_replays(tmp_path, models, normalizers, source, 1_000_000, original, "cpu")
    assert result["status"] == "event_only"
    assert result["clean_ttc"] == prediction
    assert result["disk_roundtrip_exact"]
    assert set(result["truncations"]) == {"1", "2", "3"}
    assert all(key.startswith("E_CTX_MATCHED.") for key in original)


def test_unavailable_disk_replay_retains_rgb_without_fabricating_output(tmp_path):
    source = {"event": [], "rgb": observations()["event"]}
    arrays, metadata = serialize_observations(source)
    save_npz(tmp_path / "OBSERVATIONS.npz", arrays)
    atomic_receipt(tmp_path / "OBSERVATIONS.json", metadata)
    result = verify_replays(tmp_path, {}, {}, source, 1_000_000, {}, "cpu")
    assert result["status"] == "unavailable"
    assert result["clean_ttc"] is None
    assert result["clean_compared_arrays"] == 0
    assert result["truncations"] == {}
    assert result["disk_roundtrip_exact"]


def test_bound_shard_rejects_swapped_identity_and_payload_tamper(tmp_path):
    source = observations()
    arrays, metadata = serialize_observations(source)
    save_npz(tmp_path / "OBSERVATIONS.npz", arrays)
    atomic_receipt(tmp_path / "OBSERVATIONS.json", metadata)
    binding = {
        "token": "one",
        "group": "road",
        "query_us": 1_000_000,
        "query_sha256": "roi",
        "calibration_sha256": "calib",
        "model_sha256": "model",
        "source_sha256": "code",
    }
    seal_observations(tmp_path, binding)
    assert len(load_bound_observations(tmp_path, binding)["event"]) == 4
    with pytest.raises(ValueError):
        load_bound_observations(tmp_path, {**binding, "token": "other"})
    (tmp_path / "OBSERVATIONS.json").write_text("{}")
    with pytest.raises(ValueError):
        load_bound_observations(tmp_path, binding)


@pytest.mark.parametrize(
    "fault",
    [
        "nan",
        "float64",
        "fractional",
        "future",
        "span",
        "timestamps",
        "unknown",
        "missing_windows",
        "delta",
        "roi",
    ],
)
def test_bound_shard_rejects_invalid_payload_before_sealing(tmp_path, fault):
    source = observations()
    arrays, metadata = serialize_observations(source)
    if fault == "nan":
        arrays["event.raw17"][0, 0] = np.nan
    elif fault == "float64":
        arrays["event.raw17"] = arrays["event.raw17"].astype(np.float64)
    elif fault == "fractional":
        metadata["event"][0]["anchor_us"] += 0.5
    elif fault == "future":
        metadata["event"][0]["available_us"] = 1_000_001
    elif fault == "span":
        metadata["event"][0]["first_sensor_ts_us"] = 349_999
    elif fault == "timestamps":
        metadata["event"][0]["timestamps_us"][-1] -= 1
    elif fault == "missing_windows":
        metadata["event"][0].pop("windows_us")
    elif fault == "delta":
        metadata["event"][0]["delta_t_s"] = [0.2, 0.1]
    elif fault == "roi":
        metadata["event"][0]["square_xyxy"] = [1, 1, 0, 0]
    else:
        arrays["unknown"] = np.zeros(1)
    save_npz(tmp_path / "OBSERVATIONS.npz", arrays)
    atomic_receipt(tmp_path / "OBSERVATIONS.json", metadata)
    with pytest.raises(ValueError):
        seal_observations(tmp_path, {"query_us": 1_000_000})


def test_native_vector_retains_unavailable_and_event_only_semantics():
    assert np.isnan(native_vector({}, "cpu")).all()
    vector = native_vector(
        {"E_CTX_MATCHED.output_point_phase": np.array([0.025], np.float32)}, "cpu"
    )
    assert vector[0] == vector[2]
    assert np.isnan(vector[1])
    assert np.isfinite(vector[[0, 2]]).all()


def test_selection_uses_only_identity_and_temporal_quantiles():
    counts = [106] * 11 + [105] * 13
    groups, tokens, metadata = [], [], {}
    for group_index, count in enumerate(counts):
        for ordinal in range(count):
            token = f"road{group_index:02}:{ordinal}"
            groups.append(f"road{group_index:02}")
            tokens.append(token)
            metadata[token] = {"anchor_us": ordinal * 100_000}
    groups.extend(["Slider-500"] * 77)
    tokens.extend([f"Slider-500:{i}" for i in range(77)])
    selected = select_indices(np.array(groups), np.array(tokens), metadata)
    assert len(selected) == 149
    assert [tokens[i] for i in selected[:3]] == ["road00:26", "road00:53", "road00:79"]
    assert [tokens[i] for i in selected[-77:]] == [f"Slider-500:{i}" for i in range(77)]
