import numpy as np
import pytest

from operational.phase_submission.protocol import (
    eligible_population,
    interpolate_negative_predictions,
)


def test_release_bindings_reject_substituted_model_target_and_dependency(tmp_path):
    import hashlib
    import json

    from operational.phase_submission.verify import verify_artifact_bindings

    def put(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    model = put("FINAL_MODEL/MODEL.json", {"identity": "original"})
    archive = put("replay/results.zip", "fixture")
    population = put("inputs/population.npz", "population")
    excluded = put("inputs/exclusions.json", [])
    road = put("inputs/road.json", [])
    slider = [put(f"slider/{s}_gt.csv", []) for s in (750, 1000)]
    history = put("history/environment.json", {})
    dependency = put("src/model.py", "executed")
    put("replay/RUN_FREEZE.json", {"model_sha256": digest(model)})
    put(
        "evaluation/LOCAL_EVALUATION.json",
        {
            "model_sha256": digest(model),
            "target_sources": {str(p): digest(p) for p in [road, *slider]},
        },
    )
    put("training/CONFIG.json", {"inputs": {str(p): digest(p) for p in [population, excluded]}})
    put("audit/HISTORICAL_AUDIT.json", {"source_hashes": {history.name: digest(history)}})
    put("audit/SOURCE_SUPPLEMENT.json", {"files": {"src/model.py": digest(dependency)}})
    put(
        "FINAL_CHECKS.json",
        {
            "status": "passed",
            "model_sha256": digest(model),
            "results_zip_sha256": digest(archive),
            "git_commit": "commit",
        },
    )
    args = (tmp_path, history.parent, population, excluded, road, slider[0].parent, tmp_path)
    verify_artifact_bindings(*args, current_commit="commit")
    for path in (model, road, dependency, population):
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="binding"):
            verify_artifact_bindings(*args, current_commit="commit")
        path.write_bytes(original)
    with pytest.raises(ValueError, match="commit"):
        verify_artifact_bindings(*args, current_commit="other")


def test_package_install_metadata_includes_root_readme():
    from scripts.package_phase_submission import INSTALL_FILES

    assert "README.md" in INSTALL_FILES
    from pathlib import Path

    readme = Path("docs/evttc_submission_20261031/README.md").read_text(encoding="utf-8")
    assert "uv sync --group dev" in readme
    assert "uv sync --extra dev" not in readme


def test_submission_verifier_rejects_duplicate_identity_and_changed_cost(tmp_path):
    from operational.phase_submission.verify import verify_results
    from scripts.run_evttc_competition import deterministic_zip, write_signed_results

    expected = {"fixture": [(0, "0.010000"), (1, "0.020000")]}
    rows = [
        {
            "identity": {
                "sequence_id": "fixture",
                "row_index": i,
                "index": i,
                "timestamp_text": stamp,
            },
            "status": "ok",
            "prediction_ttc_seconds": float(i - 1),
            "cost_time_seconds": 0.25,
        }
        for i, (_, stamp) in enumerate(expected["fixture"])
    ]
    result = write_signed_results(tmp_path / "results", rows, expected)
    archive = tmp_path / "results.zip"
    deterministic_zip(archive, tmp_path / "results", result["files"])
    assert verify_results(archive, rows, expected)["rows"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        verify_results(archive, rows + rows[:1], expected)
    rows[0]["cost_time_seconds"] = 0.5
    with pytest.raises(ValueError, match="ledger"):
        verify_results(archive, rows, expected)


def test_official_export_preserves_signed_values_and_original_timestamp_text(tmp_path):
    from scripts.run_evttc_competition import validate_signed_results, write_signed_results

    expected = {"fixture": [(1, "0.0400822162628174"), (2, "0.0801641941070557")]}
    rows = [
        {
            "identity": {
                "sequence_id": "fixture",
                "row_index": i,
                "index": index,
                "timestamp_text": stamp,
            },
            "status": "ok",
            "prediction_ttc_seconds": prediction,
            "cost_time_seconds": 0.2,
        }
        for i, ((index, stamp), prediction) in enumerate(
            zip(expected["fixture"], [-2.0, 3.0], strict=True)
        )
    ]
    report = write_signed_results(tmp_path, rows, expected)
    assert report["valid"]
    assert "1\t0.0400822162628174\t-2\t" in (tmp_path / "fixture/results.txt").read_text()
    with pytest.raises(ValueError, match="incomplete"):
        write_signed_results(tmp_path, rows[:1], expected)
    (tmp_path / "fixture/results.txt").write_text("bad")
    assert not validate_signed_results(tmp_path, expected)["valid"]


def test_organizer_interpolation_uses_physical_time_and_keeps_original():
    times = np.array([0.0, 1.0, 3.0, 4.0])
    original = np.array([2.0, -8.0, -9.0, 6.0])
    values, changed = interpolate_negative_predictions(times, original)
    np.testing.assert_array_equal(values, [2.0, 3.0, 5.0, 6.0])
    np.testing.assert_array_equal(original, [2.0, -8.0, -9.0, 6.0])
    assert changed == [1, 2]


def test_negative_edges_do_not_invent_organizer_extrapolation():
    with pytest.raises(ValueError, match="surrounding"):
        interpolate_negative_predictions(np.array([0.0, 1.0]), np.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="increasing"):
        interpolate_negative_predictions(np.array([1.0, 0.0]), np.array([1.0, 2.0]))


def test_submission_fit_excludes_both_slider_benchmark_labels_and_rejects_overlap():
    a = {
        "group": np.array(["road", "Slider-500", "Slider-750", "Slider-1000"]),
        "family": np.array(["road", "Slider", "Slider", "Slider"]),
        "native": np.ones((4, 3), np.float32),
        "target": np.array([1.0, 2.0, 999.0, -999.0]),
        "geometry": np.ones(4),
        "quality": np.ones((4, 20)),
        "token": np.array(["a", "b", "c", "d"]),
    }
    filtered = eligible_population(a, {"overlap"})
    assert filtered["group"].tolist() == ["road", "Slider-500"]
    a["target"][2:] = 123456
    np.testing.assert_array_equal(eligible_population(a, {"overlap"})["target"], filtered["target"])
    a["group"][0] = "overlap"
    with pytest.raises(ValueError, match="overlap"):
        eligible_population(a, {"overlap"})
