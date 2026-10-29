"""New report gates: both seals before labels, physical cohorts and coverage."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import final_train40_transfer_report as report


def fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / name
        for name in ("manifest", "registration", "own", "garl", "candidate", "output")
    }
    candidate = paths["candidate"]
    report.write(candidate / "MODEL_SEAL.json", {"frozen": True})
    report.write(
        candidate / "PORTABLE_PATHS.json",
        {"components": {key: {"sha256": "frozen"} for key in report.MODEL_COMPONENTS}},
    )
    rows = []
    for index in range(3):
        target = tmp_path / f"gt{index}.csv"
        target.write_text(f"0 {index + 1} 0 0 2\n", encoding="utf-8")
        rows.append(
            {
                "query_id": f"q{index}",
                "raw_path": f"raw{index}",
                "anchor_us": (index + 1) * 1000000,
                "physical_sequence_id": f"s{index}",
                "strata": [report.HISTORICAL, report.EXTENDED] if index == 0 else [report.EXTENDED],
                "ttc_csv_path": str(target),
            }
        )
    report.write(paths["manifest"], {"rows": rows, "targets_read": False, "optimizer_updates": 0})
    manifest_sha = report.sha(paths["manifest"])
    report.write(
        paths["registration"],
        {
            "manifest_sha256": manifest_sha,
            "queries": 3,
            "targets_read": False,
            "optimizer_updates": 0,
        },
    )
    common = {
        "manifest_sha256": manifest_sha,
        "targets_read": False,
        "optimizer_updates": 0,
        "precision": "float32",
    }
    own = {
        **common,
        "model_name": report.OWN,
        "seed": 7,
        "modality": "RGB+EVENTS",
        "model_seal_sha256": report.sha(candidate / "MODEL_SEAL.json"),
        "model_hashes": {key: "frozen" for key in report.MODEL_COMPONENTS},
        "checkpoint_selection": False,
        "model": {"portable_paths_sha256": report.sha(candidate / "PORTABLE_PATHS.json")},
    }
    garl = {
        **common,
        "model_name": report.GARL,
        "checkpoint_sha256": report.GARL_WEIGHT,
        "config_sha256": report.GARL_CONFIG,
        "registration_sha256": report.sha(paths["registration"]),
        "native_delta_t_s": 0.1,
        "clipping": None,
    }
    for route, binding in (("own", own), ("garl", garl)):
        folder = paths[route]
        report.write(folder / "INFERENCE_FREEZE.json", binding)
        digest = report.sha(folder / "INFERENCE_FREEZE.json")
        fragments = {}
        for index in reversed(range(3)):
            row = {**common, "binding_sha256": digest, "query_id": f"q{index}"}
            if route == "own":
                row["predictions"] = {model: 2.5 for model in report.MODELS if model != report.GARL}
            else:
                row.update(prediction_ttc=3.0, failure_reason=None)
            name = f"row{2 - index}.json"
            report.write(folder / name, row)
            fragments[name] = report.sha(folder / name)
        report.write(
            folder / "PREDICTIONS_SEALED.json",
            {
                **common,
                "schema": "final_train40_transfer_predictions_v1",
                "status": "COMPLETE",
                "binding_sha256": digest,
                "queries": 3,
                "fragments": fragments,
            },
        )
    return paths


def run(paths: dict[str, Path]) -> dict:
    return report.build_report(
        paths["manifest"],
        paths["registration"],
        paths["own"],
        paths["garl"],
        paths["candidate"],
        paths["output"],
        figure=False,
    )


def reseal(paths: dict[str, Path], route: str, name: str, value: dict) -> None:
    folder = paths[route]
    report.write(folder / name, value)
    seal = report.read(folder / "PREDICTIONS_SEALED.json")
    seal["fragments"][name] = report.sha(folder / name)
    report.write(folder / "PREDICTIONS_SEALED.json", seal)


def test_both_seals_before_any_gt_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path)
    (paths["garl"] / "PREDICTIONS_SEALED.json").unlink()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("GT read before complete dual prediction seals")

    monkeypatch.setattr(np, "genfromtxt", forbidden)
    with pytest.raises(FileNotFoundError):
        run(paths)


def test_reordered_fragments_join_by_id_and_overlap_is_disjoint(tmp_path: Path) -> None:
    result = run(fixture(tmp_path))
    assert result["cohorts"]["historical46"]["sensor_queries"] == 1
    assert result["cohorts"]["extended_new"]["sensor_queries"] == 2
    metrics = result["cohorts"]["extended_new"]["metrics"]
    assert metrics["pooled"][report.OWN]["complete_cohort"]["RTE_pct"] == 25
    assert metrics["contrasts"][0]["RTE_pct"]["CI95"] == [-25, -25]


@pytest.mark.parametrize("corruption", ["bytes", "duplicate_id", "ancestry", "escape"])
def test_changed_prediction_is_rejected_before_labels(
    tmp_path: Path, corruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = fixture(tmp_path)
    folder = paths["garl"]
    row = report.read(folder / "row0.json")
    if corruption == "bytes":
        (folder / "row0.json").write_text("{}", encoding="utf-8")
    elif corruption == "escape":
        seal = report.read(folder / "PREDICTIONS_SEALED.json")
        seal["fragments"]["../row0.json"] = seal["fragments"].pop("row0.json")
        report.write(folder / "PREDICTIONS_SEALED.json", seal)
    else:
        row["query_id" if corruption == "duplicate_id" else "binding_sha256"] = (
            "q0" if corruption == "duplicate_id" else "changed"
        )
        reseal(paths, "garl", "row0.json", row)
    monkeypatch.setattr(np, "genfromtxt", lambda *a, **k: pytest.fail("Read GT before rejection"))
    with pytest.raises(ValueError):
        run(paths)


def test_missing_output_keeps_coverage_and_paired_subset(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    row = report.read(paths["garl"] / "row0.json")
    row.update(prediction_ttc=None, failure_reason="EMPTY_EVENT_SEGMENT")
    reseal(paths, "garl", "row0.json", row)
    metrics = run(paths)["cohorts"]["extended_new"]["metrics"]
    assert metrics["pooled"][report.GARL]["coverage"] == 0.5
    assert metrics["pooled"][report.GARL]["complete_cohort"] is None
    assert metrics["paired_common_finite"]["queries"] == 1
    assert metrics["contrasts"][0]["status"] == "INCOMPLETE"


def test_native_binding_rejected_before_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = fixture(tmp_path)
    freeze = report.read(paths["own"] / "INFERENCE_FREEZE.json")
    freeze["seed"] = 13
    report.write(paths["own"] / "INFERENCE_FREEZE.json", freeze)
    digest = report.sha(paths["own"] / "INFERENCE_FREEZE.json")
    seal = report.read(paths["own"] / "PREDICTIONS_SEALED.json")
    seal["binding_sha256"] = digest
    for name in seal["fragments"]:
        row = report.read(paths["own"] / name)
        row["binding_sha256"] = digest
        report.write(paths["own"] / name, row)
        seal["fragments"][name] = report.sha(paths["own"] / name)
    report.write(paths["own"] / "PREDICTIONS_SEALED.json", seal)
    monkeypatch.setattr(np, "genfromtxt", lambda *a, **k: pytest.fail("Read GT before rejection"))
    with pytest.raises(ValueError, match="candidate binding"):
        run(paths)


def test_descriptive_band_edges() -> None:
    assert [report.band(x) for x in (-20, -1, 0, 0.1, 3, 6, 10, 11)] == [
        "negative",
        "negative",
        "invalid",
        "positive_lt3",
        "positive_3_to_lt6",
        "positive_6_to_10",
        "positive_6_to_10",
        "positive_gt10",
    ]


def test_portable_export_inventory_is_not_a_model_parent(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    portable_path = paths["candidate"] / "PORTABLE_PATHS.json"
    portable = report.read(portable_path)
    portable["components"]["predictions.csv"] = {"sha256": "old-test12-evidence"}
    report.write(portable_path, portable)
    freeze_path = paths["own"] / "INFERENCE_FREEZE.json"
    freeze = report.read(freeze_path)
    freeze["model"]["portable_paths_sha256"] = report.sha(portable_path)
    report.write(freeze_path, freeze)
    binding = report.sha(freeze_path)
    seal = report.read(paths["own"] / "PREDICTIONS_SEALED.json")
    seal["binding_sha256"] = binding
    for name in seal["fragments"]:
        row = report.read(paths["own"] / name)
        row["binding_sha256"] = binding
        report.write(paths["own"] / name, row)
        seal["fragments"][name] = report.sha(paths["own"] / name)
    report.write(paths["own"] / "PREDICTIONS_SEALED.json", seal)
    assert run(paths)["sensor_queries"] == 3


def test_macro_bootstrap_samples_physical_sequences_not_query_rows() -> None:
    scorer, _ = report.legacy_functions()
    frame = pd.DataFrame(
        [
            {
                "group_id": "a",
                "target_ttc": 2.0,
                report.OWN: 2.0,
                report.GARL: 3.0,
                "E_CTX_TRAIN40": 2.0,
                "R_CTX_TRAIN40": 2.0,
            },
            {
                "group_id": "b",
                "target_ttc": 2.0,
                report.OWN: 4.0,
                report.GARL: 3.0,
                "E_CTX_TRAIN40": 2.0,
                "R_CTX_TRAIN40": 2.0,
            },
        ]
    )
    duplicated = pd.concat([frame, *[frame.iloc[[0]]] * 50], ignore_index=True)
    first = report.grouped_metrics(frame, scorer)
    second = report.grouped_metrics(duplicated, scorer)
    assert first["contrasts"] == second["contrasts"]
    assert first["equal_sequence_macro"] == second["equal_sequence_macro"]
    assert first["pooled"] != second["pooled"]
