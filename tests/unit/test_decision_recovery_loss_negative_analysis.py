"""Paired full-cohort identity and additive negative-band accounting."""

import json

import numpy as np
import pandas as pd
import pytest

from operational.decision_recovery import loss_negative_analysis as analysis


def examples():
    truth = np.array([-2.0, -3.0, -4.0, -5.0, 1.0, 4.0, 8.0])
    base = pd.DataFrame(
        {
            "sample_token": [f"q{i}" for i in range(len(truth))],
            "group_id": ["a", "a", "a", "b", "a", "a", "a"],
            "sequence_id": ["s"] * len(truth),
            "target_ttc": truth,
            "role": "H",
        }
    )
    tables = {}
    for arm, pred in {"L0": [-2, -3, 4, 5, 1, 4, 8], "LW": [-1, 3, -4, 60, 1, 4, 8]}.items():
        frame = base.copy()
        frame["prediction_ttc"] = np.asarray(pred, float)
        frame["raw_phase"] = -np.log1p(-0.1 / frame.prediction_ttc)
        tables[arm] = frame
    return tables, base


def test_cells_reconstruct_native_group_macro_and_error_sums():
    tables, rows = examples()
    report, frame = analysis.compute(tables, rows, "H")
    assert len(frame) == 4
    assert all(cell["count"] == 1 for cell in report["cells"].values())
    assert report["cells"]["LW_only_correct"]["L0"]["absolute_error_sum_s"] == 8
    assert report["cells"]["LW_only_correct"]["LW"]["absolute_error_sum_s"] == 0
    assert report["negative_totals"]["L0"]["sign_error_count"] == 2
    assert report["negative_totals"]["LW"]["sign_error_count"] == 2
    for arm in analysis.ARMS:
        expected = report["complete_population_metrics"][arm]["bins"]["negative"]["group_macro_mid"]
        actual = sum(
            c[arm]["group_macro_negative_MiD_contribution"] for c in report["cells"].values()
        )
        assert actual == pytest.approx(expected)
        assert (
            sum(c[arm]["absolute_error_sum_s"] for c in report["cells"].values())
            == report["negative_totals"][arm]["absolute_error_sum_s"]
        )
    assert not report["candidate"] and not report["selection"]


@pytest.mark.parametrize("change", ["order", "target", "duplicate", "clock_identity", "coverage"])
def test_identity_and_coverage_changes_fail_closed(change):
    tables, rows = examples()
    if change == "order":
        tables["LW"] = tables["LW"].iloc[::-1]
    elif change == "target":
        tables["LW"].loc[0, "target_ttc"] = -2.01
    elif change == "duplicate":
        tables["LW"].loc[0, "sample_token"] = "q1"
    elif change == "clock_identity":
        tables["LW"].loc[0, "sample_token"] = "same_object_other_query_time"
    else:
        tables["LW"].loc[0, "prediction_ttc"] = np.nan
    with pytest.raises(ValueError):
        analysis.compute(tables, rows, "H")


def test_empty_cells_have_null_means_and_zero_contributions():
    tables, rows = examples()
    tables["LW"] = tables["L0"].copy()
    report, _ = analysis.compute(tables, rows, "H")
    for name in ("L0_only_correct", "LW_only_correct"):
        assert report["cells"][name]["count"] == 0
        assert report["cells"][name]["LW"]["mae_s"] is None
        assert report["cells"][name]["LW"]["MiD_sum"] == 0


def test_positive_queries_do_not_enter_negative_decomposition():
    tables, rows = examples()
    before, _ = analysis.compute(tables, rows, "H")
    tables["LW"].loc[4, "prediction_ttc"] = -60.0
    tables["LW"].loc[4, "raw_phase"] = -np.log1p(-0.1 / -60.0)
    after, _ = analysis.compute(tables, rows, "H")
    assert after["cells"] == before["cells"]
    assert after["complete_population_metrics"] != before["complete_population_metrics"]


@pytest.fixture
def admitted_files(tmp_path, monkeypatch):
    from operational.range_transfer.common import digest

    tables, rows = examples()
    rows["query_time_us"] = np.arange(len(rows), dtype=np.int64) * 100000 + 1000000
    rows["cutoff_us"] = rows.query_time_us
    rows["producer_times_us"] = [[100, 200]] * len(rows)
    monkeypatch.setattr(analysis.common, "RUN", tmp_path)
    monkeypatch.setitem(analysis.COUNTS, "H", len(rows))
    manifest_path, rows_path, sensor = (
        tmp_path / "manifest.json",
        tmp_path / "rows.parquet",
        tmp_path / "sensor",
    )
    monkeypatch.setattr(analysis.training, "manifest_path", lambda role: manifest_path)
    monkeypatch.setattr(analysis.training, "cache_root", lambda role: sensor)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    sha = analysis.common.sha
    rows.to_parquet(rows_path, index=False)
    write(
        manifest_path,
        {
            "status": "COMPLETE",
            "rows_path": str(rows_path),
            "rows_sha256": sha(rows_path),
            "population_size": len(rows),
        },
    )
    sensor_binding = {
        "manifest_sha256": sha(manifest_path),
        "source_identity": {"rows_sha256": sha(rows_path)},
    }
    write(sensor / "BINDING.json", sensor_binding)
    write(
        sensor / "COMPLETE.json",
        {"status": "COMPLETE", "binding": digest(sensor_binding), "rows": len(rows)},
    )
    registration = tmp_path / "evaluation/REGISTRATION.json"
    write(registration, {"endpoints": {arm: {"checkpoint_sha256": arm} for arm in analysis.ARMS}})
    write(tmp_path / "evaluation/REPORT.json", {})
    for arm, frame in tables.items():
        directory = tmp_path / f"evaluation/predictions/H/{arm}"
        directory.mkdir(parents=True)
        frame.to_parquet(directory / "predictions.parquet", index=False)
        write(
            directory / "METRICS.json",
            analysis.common.score(frame, frame.prediction_ttc.to_numpy()),
        )
        identity = {
            "arm": arm,
            "role": "H",
            "seed": 7,
            "endpoint": {"checkpoint_sha256": arm},
            "evaluation_registration_sha256": sha(registration),
            "manifest_sha256": sha(manifest_path),
            "rows_sha256": sha(rows_path),
            "sensor_complete_sha256": sha(sensor / "COMPLETE.json"),
        }
        write(directory / "IDENTITY.json", identity)
        write(
            directory / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "rows": len(rows),
                "binding": digest(identity),
                "sha256": sha(directory / "predictions.parquet"),
                "metrics_sha256": sha(directory / "METRICS.json"),
            },
        )
    return tmp_path, rows_path


def test_loading_requires_exact_clock_bearing_source_bytes(admitted_files):
    _, rows_path = admitted_files
    _, _, admission, _ = analysis.load("H")
    assert admission["full_metrics_reproduced_exactly"]
    assert "query_time_us" in admission["clocks"]["source_columns"]
    rows = pd.read_parquet(rows_path)
    rows.loc[0, "query_time_us"] += 1
    rows.to_parquet(rows_path, index=False)
    with pytest.raises(ValueError, match="Source population or sensor clock binding changed"):
        analysis.load("H")


def test_loading_rejects_prediction_clock_receipt_rebinding(admitted_files):
    root, _ = admitted_files
    path = root / "evaluation/predictions/H/LW/IDENTITY.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    identity["rows_sha256"] = "a_different_clock_population"
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="Changed endpoint/clock/coverage receipt"):
        analysis.load("H")


def test_saturation_counts_include_both_native_signs():
    tables, rows = examples()
    tables["LW"].loc[0, ["prediction_ttc", "raw_phase"]] = [60.0, 0.0]
    tables["LW"].loc[2, ["prediction_ttc", "raw_phase"]] = [-60.0, -1e-5]
    result, _ = analysis.compute(tables, rows, "H")
    assert result["negative_totals"]["LW"]["saturated_positive_count"] >= 1
    assert result["negative_totals"]["LW"]["saturated_negative_count"] == 1
