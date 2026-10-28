from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.compare_garl_evttc_competition as comparison

COUNTS = (50, 43, 30, 46, 55, 59, 42, 40, 51, 18)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    phase, garl = tmp_path / "phase", tmp_path / "garl"
    phase.mkdir()
    garl.mkdir()
    phase_rows = []
    query_manifest = {}
    cursor = 0
    for sequence_id, count in zip(comparison.SEQUENCES, COUNTS, strict=True):
        identities = []
        for row_index in range(count):
            identity = {
                "sequence_id": sequence_id,
                "row_index": row_index,
                "index": row_index * 2,
                "timestamp_text": f"{cursor / 100:.2f}",
                "anchor_us": cursor * 10_000,
                "box_text": ["1", "2", "3", "4"],
            }
            identities.append(identity)
            phase_rows.append(
                {
                    "schema": "evttc_competition_row_v1",
                    "status": "ok",
                    "identity": identity,
                    "prediction_ttc_seconds": 1.0 + cursor / 200,
                    "model_wall_seconds": 0.02,
                    "cost_time_seconds": 0.03,
                }
            )
            cursor += 1
        query_manifest[sequence_id] = identities
    assert len(phase_rows) == comparison.PHASE_ROWS
    _json(phase / "RUN_FREEZE.json", {"query_manifest": query_manifest})
    _jsonl(phase / "all_rows.jsonl", phase_rows)
    _json(
        phase / "summary.json",
        {
            "status": "COMPLETE",
            "terminal_population_accounted": True,
            "recorded_rows": 434,
            "null_rows": 0,
        },
    )
    _json(phase / "independent_audit.json", {"valid": True})
    (phase / "results.zip").write_bytes(b"phase-results")

    garl_rows = []
    for phase_row in phase_rows:
        for variant_index, variant in enumerate(comparison.VARIANTS):
            unavailable = variant == "event" and phase_row["identity"]["row_index"] == 0
            prediction = (
                None if unavailable else phase_row["prediction_ttc_seconds"] + variant_index
            )
            garl_rows.append(
                {
                    "schema": "garl_evttc_competition_row_v1",
                    "variant": variant,
                    "identity": phase_row["identity"],
                    "status": "unavailable" if unavailable else "ok",
                    "prediction_ttc_seconds": prediction,
                    "preprocess_seconds": 0.01,
                    "model_wall_seconds": 0.02,
                    "cost_time_seconds": 0.03,
                    "reason": "missing_event_support" if unavailable else "available",
                    "input_details": {"delta_t_s": [0.1, 0.1], "windows_us": [[0, 1]]},
                    "prediction": None if unavailable else {"status": "ok"},
                }
            )
    _jsonl(garl / "all_rows.jsonl", garl_rows)
    variants = {}
    for variant in comparison.VARIANTS:
        path = garl / f"{variant}.zip"
        path.write_bytes(f"{variant}-terminal".encode())
        nulls = sum(r["variant"] == variant and r["status"] == "unavailable" for r in garl_rows)
        variants[variant] = {
            "count": 434,
            "nulls": nulls,
            ("partial_results_zip" if nulls else "results_zip"): path.name,
            (
                "partial_results_zip_sha256" if nulls else "results_zip_sha256"
            ): comparison.sha256_file(path),
        }
    _json(
        garl / "summary.json",
        {
            "status": "PARTIAL",
            "terminal_population_accounted": True,
            "expected_records": 1302,
            "recorded_records": 1302,
            "variants": variants,
        },
    )
    phase_hashes = {
        name: comparison.sha256_file(phase / name) for name in comparison.PHASE_ARTIFACTS
    }
    asset = tmp_path / "checkpoint.pt"
    asset.write_bytes(b"pinned-checkpoint")
    gt_audit = tmp_path / "ground_truth_identity_audit.json"
    _json(
        gt_audit,
        {
            "schema": "evttc_competition_gt_identity_inventory_v1",
            "status": "NO_VERIFIED_EXACT_QUERY_GROUND_TRUTH",
            "query_count": 434,
        },
    )
    _json(
        garl / "RUN_FREEZE.json",
        {
            "schema": "garl_evttc_competition_freeze_v1",
            "query_manifest": query_manifest,
            "phase_reference": {
                "path_resolved": str(phase.resolve()),
                "artifact_sha256": phase_hashes,
            },
            "executed_source_sha256": {
                "scripts/compare_garl_evttc_competition.py": comparison.sha256_file(
                    Path(comparison.__file__).resolve()
                )
            },
            "inventory": {
                "pinned": True,
                "asset": {
                    "path": str(asset.resolve()),
                    "sha256": comparison.sha256_file(asset),
                    "bytes": asset.stat().st_size,
                },
            },
            "ground_truth_identity_audit": {
                "path_resolved": str(gt_audit.resolve()),
                "sha256": comparison.sha256_file(gt_audit),
                "schema": "evttc_competition_gt_identity_inventory_v1",
                "status": "NO_VERIFIED_EXACT_QUERY_GROUND_TRUTH",
            },
            "timing": {
                "per_variant": (
                    "independent_preparation_plus_synchronized_model_to_host_wall_time"
                ),
                "no_explicit_tensor_or_preprocessing_reuse": True,
                "fixed_variant_order": list(comparison.VARIANTS),
                "shared_state": "OS_and_HDF5_page_cache_and_GPU_warm_state_can_be_shared",
                "cross_variant_latency_ranking_supported": False,
                "not_optimized_streaming_latency": True,
            },
        },
    )
    garl_summary = comparison.read_json(garl / "summary.json")
    garl_summary["freeze_sha256"] = comparison.sha256_file(garl / "RUN_FREEZE.json")
    garl_summary["ledger_sha256"] = comparison.sha256_file(garl / "all_rows.jsonl")
    _json(garl / "summary.json", garl_summary)
    return garl, phase


def test_builds_label_blind_disagreement_report_from_terminal_rows(tmp_path: Path) -> None:
    run, _phase = _fixture(tmp_path)
    output = tmp_path / "comparison"
    result = comparison.build_comparison(run, output, _phase)
    assert result["status"] == "COMPLETE_LABEL_BLIND_COMPARISON"
    assert result["ground_truth_available_for_matching_434_queries"] is False
    assert result["prediction_disagreement"]["full"]["absolute_disagreement_median_s"] == 0
    assert result["variant_stats"]["event"]["null_predictions"] == len(comparison.SEQUENCES)
    with (output / "per_query_prediction_disagreement.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == comparison.GARL_ROWS
    assert "prediction_disagreement_garl_minus_phase_s" in rows[0]
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "not an accuracy evaluation" in report
    assert "10.60% RTE" in report
    assert "does not support cross-variant latency ranking" in report
    assert "different history and recomputed offline adapter" in report
    assert (output / "predictions_by_sequence_common_scale.pdf").is_file()
    assert (output / "predictions_by_sequence_signed_symlog.png").is_file()
    assert (output / "offline_runtime_profile.png").is_file()
    with (output / "per_sequence_prediction_disagreement.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        sequence_rows = list(csv.DictReader(stream))
    assert len(sequence_rows) == 30


def test_refuses_nonterminal_summary_before_reading_predictions(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _json(
        run / "summary.json",
        {
            "status": "RUNNING",
            "terminal_population_accounted": False,
            "expected_records": 1302,
            "recorded_records": 4,
        },
    )
    (run / "all_rows.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not terminal"):
        comparison.build_comparison(run)


def test_rejects_garl_identity_drift(tmp_path: Path) -> None:
    run, _phase = _fixture(tmp_path)
    rows = comparison.read_jsonl(run / "all_rows.jsonl")
    rows[0]["identity"] = {**rows[0]["identity"], "timestamp_text": "changed"}
    _jsonl(run / "all_rows.jsonl", rows)
    summary = comparison.read_json(run / "summary.json")
    summary["ledger_sha256"] = comparison.sha256_file(run / "all_rows.jsonl")
    _json(run / "summary.json", summary)
    with pytest.raises(ValueError, match="identity differs"):
        comparison.build_comparison(run, expected_phase=_phase)


def test_garl_conversion_delta_is_counted_exactly_once_per_available_query() -> None:
    rows = [
        {
            "input_details": {
                "conversion_delta_t_s": value,
                "delta_t_s": [999.0, 999.0],
                "event_windows": [[0, 1], [1, 2]],
            },
            "prediction": {"conversion_delta_t_s": value},
        }
        for value in (0.08, 0.12)
    ]

    stats = comparison.history_summary(rows)

    assert stats["delta_count"] == 2
    assert stats["delta_seconds_min"] == 0.08
    assert stats["delta_seconds_median"] == 0.1
    assert stats["delta_seconds_max"] == 0.12
    assert stats["delta_semantics"] == "one input conversion_delta_t_s per available query"
