from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from operational.final_refit import test12 as test12_module
from operational.final_refit.export import HEAD_STAGE, MODEL_SEAL_SCHEMA
from operational.final_refit.recipe import canonical_sha256, sha256
from operational.final_refit.test12 import (
    PredictionFragments,
    run_test12_inference,
    validate_test12_inputs,
)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _inputs(path: Path) -> dict[str, Any]:
    tokens = [f"token-{index:04d}" for index in range(6_762)]
    media_root = path.parent / "media"
    raw = media_root / "data/test/sequence/events.h5"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"public-event-fixture")
    tensor = path.parent / "query_00000.npz"
    tensor.write_bytes(b"frozen-event-tensor")
    windows = [[100_000, 200_000], [200_000, 300_000], [300_000, 400_000]]
    event = {
        tokens[0]: {
            "sample_token": tokens[0],
            "sequence_id": "sequence",
            "track_id": "track",
            "anchor_us": 900_000,
            "available_us": 900_000,
            "input_span_us": 300_000,
            "sensor_anchor_us": 400_000,
            "windows_us": windows,
            "square_xyxy": [1.0, 2.0, 3.0, 4.0],
            "delta_t_s": 0.1,
            "raw_path": str(raw.resolve()),
            "current_tensor_path": str(tensor.resolve()),
            "current_tensor_sha256": sha256(tensor),
        }
    }
    rgb = {
        "rgb-current": {
            "observation_id": "rgb-current",
            "sample_token": tokens[0],
            "anchor_us": 950_000,
            "available_us": 975_000,
            "query_time_us": 950_000,
            "query_age_us": 0,
            "frame_ids": ["track|first", "track|second"],
            "producer_times_us": [850_000, 950_000],
            "producer_available_us": [875_000, 975_000],
            "producer_sensor_times_us": [100_000, 200_000],
            "producer_shards": [
                "data/test/sequence/rgb_shards/one.tar",
                "data/test/sequence/rgb_shards/one.tar",
            ],
            "producer_members": ["first.png", "second.png"],
            "producer_boxes_xyxy": [[1.0, 2.0, 3.0, 4.0]] * 2,
        }
    }
    queries = [
        {
            "sample_token": token,
            "sequence_id": "sequence",
            "track_id": "track",
            "query_time_us": 1_000_000,
            "event_history": [tokens[0]],
            "rgb_history": [] if index == 2 else ["rgb-current"],
        }
        for index, token in enumerate(tokens)
    ]
    # One duplicate official row is a physical alias.  Both rows retain their
    # official sample_token, but inference is executed only once.
    queries[1] = {**queries[0], "sample_token": tokens[1]}
    aliases = {token: token for token in tokens}
    aliases[tokens[1]] = tokens[0]
    value = {
        "media_root": str(media_root.resolve()),
        "queries": queries,
        "event_observations": event,
        "rgb_observations": rgb,
        "physical_aliases": aliases,
    }
    _write(path, value)
    return value


def _provenance(inputs_path: Path) -> tuple[Path, Path]:
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    event = inputs["event_observations"]["token-0000"]
    event_freeze = inputs_path.with_name("EVENT_INPUT_FREEZE.json")
    _write(
        event_freeze,
        {
            "schema": "garlttc_native_h8_test_inputs_v1",
            "input_sha256": "a" * 64,
            "test_labels_read": False,
            "rows": [
                {
                    "sample_token": "token-0000",
                    "sequence": event["sequence_id"],
                    "path": event["raw_path"],
                    "windows": event["windows_us"],
                    "square": event["square_xyxy"],
                    "delta": event["delta_t_s"],
                    "anchor": event["sensor_anchor_us"],
                }
            ],
        },
    )
    tensor = Path(event["current_tensor_path"])
    _write(
        tensor.with_suffix(".json"),
        {
            "input_freeze_sha256": sha256(event_freeze),
            "sample_token": "token-0000",
            "sha256": sha256(tensor),
            "raw_sha256": "b" * 64,
        },
    )
    model_freeze = inputs_path.with_name("MODEL_INPUT_FREEZE.json")
    _write(
        model_freeze,
        {
            "inputs_sha256": sha256(inputs_path),
            "test_targets_read": False,
            "external_submission": False,
            "test_input_sha256": "a" * 64,
        },
    )
    return model_freeze, event_freeze


def _official(path: Path) -> Path:
    official = path.with_name("test_inputs.parquet")
    pd.DataFrame({"sample_token": [f"token-{index:04d}" for index in range(6_762)]}).to_parquet(
        official, index=False
    )
    return official


def _trusted(inputs_path: Path, official_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "official_test_inputs_path": official_path,
        "expected_inputs_sha256": sha256(inputs_path),
        "expected_official_test_inputs_sha256": sha256(official_path),
    }
    model_freeze = inputs_path.with_name("MODEL_INPUT_FREEZE.json")
    event_freeze = inputs_path.with_name("EVENT_INPUT_FREEZE.json")
    if model_freeze.exists() and event_freeze.exists():
        result.update(
            expected_model_input_freeze_sha256=sha256(model_freeze),
            expected_event_input_freeze_sha256=sha256(event_freeze),
        )
    return result


def _seal(path: Path) -> None:
    _write(
        path,
        {
            "schema": MODEL_SEAL_SCHEMA,
            "status": "SEALED_NOT_EVALUATED",
            "candidate": HEAD_STAGE,
            "scope": {"private_targets_read": False},
        },
    )


class _Predictor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    def predict(self, query: Mapping[str, Any], inputs: Mapping[str, Any]) -> float:
        assert "target" not in inputs
        self.calls.append(str(query["sample_token"]))
        return 2.0 + len(self.calls) / 100_000

    def close(self) -> None:
        self.closed = True


def test_resumable_seal_bound_inference_preserves_alias_and_fallback(tmp_path: Path) -> None:
    inputs_path = tmp_path / "INPUTS.json"
    _inputs(inputs_path)
    official = _official(inputs_path)
    model_freeze, event_freeze = _provenance(inputs_path)
    seal_path = tmp_path / "MODEL_SEAL.json"
    _seal(seal_path)
    binding = canonical_sha256(
        {
            "model_seal_sha256": sha256(seal_path),
            "inputs_sha256": sha256(inputs_path),
            "inference_source_sha256": sha256(Path(test12_module.__file__)),
        }
    )
    fragments = PredictionFragments(tmp_path / "work/prediction_fragments", binding)
    fragments.write("token-0000", 1.25)
    predictor = _Predictor()

    receipt = run_test12_inference(
        inputs_path=inputs_path,
        **_trusted(inputs_path, official),
        model_seal_path=seal_path,
        predictor=predictor,
        work_directory=tmp_path / "work",
        predictions_output=tmp_path / "predictions.csv",
        receipt_output=tmp_path / "PREDICTIONS_SEALED.json",
        verify_rgb_sources=False,
        model_input_freeze_path=model_freeze,
        event_input_freeze_path=event_freeze,
    )

    assert receipt["status"] == "COMPLETE"
    assert receipt["population"] == 6_762
    assert receipt["physical_queries"] == 6_761
    assert receipt["fallback_count"] == 1
    assert len(predictor.calls) == 6_760
    assert predictor.closed
    frame = pd.read_csv(tmp_path / "predictions.csv")
    assert frame.loc[0, "prediction"] == frame.loc[1, "prediction"] == 1.25
    assert frame.loc[2, "fallback_source"] == "E_CTX"
    assert not bool(frame.loc[2, "rgb_available"])
    assert receipt["predictions_sha256"] == sha256(tmp_path / "predictions.csv")


def test_input_validation_rejects_private_fields_and_noncausal_history(tmp_path: Path) -> None:
    inputs_path = tmp_path / "INPUTS.json"
    value = _inputs(inputs_path)
    official = _official(inputs_path)
    model_freeze, event_freeze = _provenance(inputs_path)
    value["queries"][0]["target_ttc"] = 2.0
    _write(inputs_path, value)
    with pytest.raises(ValueError, match="target/label"):
        validate_test12_inputs(
            inputs_path,
            **_trusted(inputs_path, official),
            verify_rgb_sources=False,
            model_input_freeze_path=model_freeze,
            event_input_freeze_path=event_freeze,
        )

    del value["queries"][0]["target_ttc"]
    value["event_observations"]["token-0000"]["anchor_us"] = 400_000
    value["event_observations"]["token-0000"]["available_us"] = 400_000
    _write(inputs_path, value)
    with pytest.raises(ValueError, match="650 ms"):
        validate_test12_inputs(
            inputs_path,
            **_trusted(inputs_path, official),
            verify_rgb_sources=False,
            model_input_freeze_path=model_freeze,
            event_input_freeze_path=event_freeze,
        )


def test_input_validation_rejects_cross_object_and_future_rgb_clock(tmp_path: Path) -> None:
    inputs_path = tmp_path / "INPUTS.json"
    value = _inputs(inputs_path)
    official = _official(inputs_path)
    model_freeze, event_freeze = _provenance(inputs_path)
    value["queries"][2]["track_id"] = "another-track"
    _write(inputs_path, value)
    with pytest.raises(ValueError, match="object or sequence"):
        validate_test12_inputs(
            inputs_path,
            **_trusted(inputs_path, official),
            verify_rgb_sources=False,
            model_input_freeze_path=model_freeze,
            event_input_freeze_path=event_freeze,
        )

    value["queries"][2]["track_id"] = "track"
    value["rgb_observations"]["rgb-current"]["producer_available_us"][-1] = 1_000_001
    _write(inputs_path, value)
    with pytest.raises(ValueError, match="RGB observation"):
        validate_test12_inputs(
            inputs_path,
            **_trusted(inputs_path, official),
            verify_rgb_sources=False,
            model_input_freeze_path=model_freeze,
            event_input_freeze_path=event_freeze,
        )


def test_input_validation_accepts_rgb_explicit_input_span(tmp_path: Path) -> None:
    inputs_path = tmp_path / "INPUTS.json"
    value = _inputs(inputs_path)
    official = _official(inputs_path)
    value["rgb_observations"]["rgb-current"]["input_span_us"] = 100_000
    _write(inputs_path, value)
    model_freeze, event_freeze = _provenance(inputs_path)

    validated = validate_test12_inputs(
        inputs_path,
        **_trusted(inputs_path, official),
        verify_rgb_sources=False,
        model_input_freeze_path=model_freeze,
        event_input_freeze_path=event_freeze,
    )

    assert validated["input_provenance_verified"] is True


def test_input_validation_rejects_equal_size_nonofficial_token_inventory(
    tmp_path: Path,
) -> None:
    inputs_path = tmp_path / "INPUTS.json"
    value = _inputs(inputs_path)
    official = _official(inputs_path)
    displaced = str(value["queries"][-1]["sample_token"])
    value["queries"][-1]["sample_token"] = "alien-token"
    del value["physical_aliases"][displaced]
    value["physical_aliases"]["alien-token"] = "alien-token"
    _write(inputs_path, value)

    with pytest.raises(ValueError, match="official test_inputs.parquet"):
        validate_test12_inputs(
            inputs_path,
            **_trusted(inputs_path, official),
            verify_rgb_sources=False,
        )


def test_input_validation_rejects_unbound_event_tensor_receipt(tmp_path: Path) -> None:
    inputs_path = tmp_path / "INPUTS.json"
    value = _inputs(inputs_path)
    official = _official(inputs_path)
    model_freeze, event_freeze = _provenance(inputs_path)
    tensor = Path(value["event_observations"]["token-0000"]["current_tensor_path"])
    receipt = json.loads(tensor.with_suffix(".json").read_text(encoding="utf-8"))
    receipt["input_freeze_sha256"] = "f" * 64
    _write(tensor.with_suffix(".json"), receipt)
    with pytest.raises(ValueError, match="tensor receipt"):
        validate_test12_inputs(
            inputs_path,
            **_trusted(inputs_path, official),
            verify_rgb_sources=False,
            model_input_freeze_path=model_freeze,
            event_input_freeze_path=event_freeze,
        )


def test_resume_fragment_rejects_another_model_binding(tmp_path: Path) -> None:
    first = PredictionFragments(tmp_path, "a" * 64)
    first.write("token", 3.0)
    second = PredictionFragments(tmp_path, "b" * 64)
    with pytest.raises(ValueError, match="sealed run"):
        second.read("token")


def test_resume_fragment_rejects_private_target_claim(tmp_path: Path) -> None:
    fragments = PredictionFragments(tmp_path, "a" * 64)
    fragments.write("token", 3.0)
    path = fragments._path("token")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["private_targets_read"] = True
    _write(path, value)
    with pytest.raises(ValueError, match="sealed run"):
        fragments.read("token")
