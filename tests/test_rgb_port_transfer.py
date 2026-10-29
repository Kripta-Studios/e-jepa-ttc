from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.rgb_port.accounting import sha256_file
from operational.rgb_port.infer_experts import Fragments
from operational.rgb_port.transfer import (
    PREDICTION_FIELDS,
    _clock,
    _fragment_values,
    _manifest,
    _own_rgb_crop,
    _restore_fragment,
    _select_rgb_history,
    block_external,
    parser,
    score_dev32,
)


def _csv(path: Path, fields: list[str], rows: list[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_prediction_manifest_rejects_target_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "LABEL_FREE_MANIFEST",
                "rows": [{"query_id": "q", "target_ttc": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden target"):
        _manifest(manifest)


def test_clock_rejects_information_after_cutoff() -> None:
    with pytest.raises(ValueError, match="causally available"):
        _clock(np.asarray([100], np.int64), np.asarray([201], np.int64), 200)


def test_own_rgb_transfer_crop_matches_training_uint8_path() -> None:
    rows, columns = np.indices((48, 64))
    frame0 = np.stack((rows, columns, (3 * rows + 5 * columns) % 256), axis=-1).astype(np.uint8)
    frame1 = np.flip(frame0, axis=1).copy()
    frames = [frame0, frame1]
    boxes = [(8.5, 10.0, 24.5, 30.0), (18.0, 12.0, 40.0, 36.0)]

    actual, square = _own_rgb_crop(frames, boxes)
    expected_square = common_square_from_boxes(boxes, (0, 1), margin_fraction=0.25)
    expected_uint8 = np.stack([crop_uint8(frame, expected_square) for frame in frames])
    expected = np.ascontiguousarray(expected_uint8.astype(np.float32) / np.float32(255.0))

    assert square == expected_square
    assert actual.dtype == np.float32
    assert actual.shape == (2, 3, 128, 128)
    np.testing.assert_array_equal(actual, expected)


def test_score_requires_seal_then_preserves_public_comparators(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    own = {
        name: (
            "q0"
            if name == "query_id"
            else "s0"
            if name == "sequence_id"
            else "100"
            if name == "anchor_us"
            else "PREDICTED"
            if name == "prediction_status"
            else "True"
            if name == "rgb_available"
            else ""
            if name == "rgb_unavailable_reason"
            else "2.0"
        )
        for name in PREDICTION_FIELDS
    }
    _csv(predictions, list(PREDICTION_FIELDS), [own])
    seal = tmp_path / "seal.json"
    seal.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "labels_read": False,
                "predictions_path": str(predictions),
                "predictions_sha256": sha256_file(predictions),
            }
        ),
        encoding="utf-8",
    )
    public = tmp_path / "public.csv"
    _csv(
        public,
        ["query_id", "sequence_id", "anchor_us", "truth_ttc_seconds", "public_Garl_event_lhr"],
        [
            {
                "query_id": "q0",
                "sequence_id": "s0",
                "anchor_us": 100,
                "truth_ttc_seconds": 1.5,
                "public_Garl_event_lhr": 1.25,
            }
        ],
    )
    result = score_dev32(seal_path=seal, comparator_csv=public, output=tmp_path / "score")
    assert result["optimizer_updates"] == 0
    assert result["metrics"]["E_CTX_MATCHED"]["native_mae_s"] == pytest.approx(0.5)
    assert result["metrics"]["public_Garl_event_lhr"]["native_mae_s"] == pytest.approx(0.25)
    native = result["metrics"]["E_CTX_MATCHED"]["native"]
    common = result["metrics"]["E_CTX_MATCHED"]["common_cap60"]
    assert native["num_rows"] == common["num_rows"] == 1
    assert native["diagnostics"]["median_ae_s"] == pytest.approx(0.5)
    assert native["diagnostics"]["rte_pct"] == pytest.approx(100 / 3)
    assert "sign_error_rate" in common["diagnostics"]
    assert "p95_ae_s" in common["diagnostics"]
    assert result["row_accounting"] == {
        "population": 1,
        "prediction_status_counts": {"PREDICTED": 1},
        "rgb_available_count": 1,
        "rgb_unavailable_count": 0,
        "rgb_unavailable_reason_counts": {},
        "rows_omitted": 0,
    }
    preprocessing = result["preprocessing_contracts"]
    assert preprocessing["own_rgb_port"]["target_fields_used"] is False
    assert (
        preprocessing["published_garl_comparator"]["reused_for_own_rgb_port_predictions"] is False
    )


def test_external_block_is_branch_local_and_cli_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "FCWD_RGB_STATUS.json"
    result = block_external(
        output=path,
        branch="FCWD_RGB_AND_FUSION",
        reason="RGB_TO_EVENT_CALIBRATION_MISSING",
    )
    assert result["status"] == "BLOCKED_EXTERNAL"
    assert result["independent_p_h_v_unaffected"] is True
    parsed = parser().parse_args(
        ["block-external", "--output", str(path), "--branch", "rgb", "--reason", "missing"]
    )
    assert parsed.command == "block-external"


def test_rgb_history_cap_includes_the_complete_pair_span() -> None:
    feature = torch.zeros(17)
    expert = torch.zeros(3)
    timeline = [
        (349_999, 450_000, 450_000, feature, expert),
        (350_000, 450_001, 450_001, feature + 1, expert + 1),
        (900_000, 1_000_000, 1_000_000, feature + 2, expert + 2),
    ]
    selected = _select_rgb_history(timeline, 1_000_000)
    assert [item[0] for item in selected] == [350_000, 900_000]
    dense = [
        (900_000 + index, 910_000 + index, 910_000 + index, feature, expert) for index in range(10)
    ]
    assert len(_select_rgb_history(dense, 1_000_000)) == 8


def test_query_fragment_restores_prediction_and_bounded_rgb_state(tmp_path: Path) -> None:
    row = {"query_id": "q0", "sequence_id": "s0", "anchor_us": 1_000_000}
    prediction = {
        "query_id": "q0",
        "sequence_id": "s0",
        "anchor_us": 1_000_000,
        "prediction_status": "PREDICTED",
        "rgb_available": True,
        "rgb_unavailable_reason": "",
        "E_H1_MATCHED": 1.0,
        "E_CTX_MATCHED": 2.0,
        "R_H1": 3.0,
        "R_CTX": 4.0,
        "F_TRUE": 5.0,
        "F_ZERO": 6.0,
    }
    history = [(900_000, 1_000_000, 1_000_000, torch.arange(17).float(), torch.arange(3).float())]
    store = Fragments(tmp_path / "parts", {"parents": "frozen", "optimizer_updates": 0})
    store.write(0, _fragment_values(prediction, history))
    saved = store.read(0)
    assert saved is not None
    restored, restored_history = _restore_fragment(saved, row, torch.device("cpu"))
    assert restored == prediction
    assert len(restored_history) == 1
    assert restored_history[0][:3] == history[0][:3]
    torch.testing.assert_close(restored_history[0][3], history[0][3])
    torch.testing.assert_close(restored_history[0][4], history[0][4])
