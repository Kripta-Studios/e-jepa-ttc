"""No inherited prediction changes and no unequal-population accuracy claims."""

import csv
from pathlib import Path

import pytest

from operational.efficient_context.common import atomic_json, digest
from operational.evttc_rgb_transfer.run import read
from operational.evttc_rgb_transfer.score import MODELS, run


def fixture_pair(tmp_path: Path) -> tuple[Path, Path]:
    baseline, output = tmp_path / "old", tmp_path / "new"
    rows = [{"query_id": f"q{i}", "sequence_id": "seq"} for i in range(2)]
    atomic_json(baseline / "QUERY_MANIFEST.json", {"rows": rows})
    atomic_json(baseline / "INFERENCE_FREEZE.json", {"frozen": True})
    old_hashes = {}
    for i, row in enumerate(rows):
        name = f"predictions/query_{i:05d}.json"
        atomic_json(
            baseline / name,
            {
                **row,
                "binding_sha256": digest(baseline / "INFERENCE_FREEZE.json"),
                "ttc": {m: float(i + 2) for m in MODELS[:-1]},
            },
        )
        old_hashes[name] = digest(baseline / name)
    atomic_json(
        baseline / "PREDICTIONS_SEALED.json",
        {
            "status": "COMPLETE",
            "queries": 2,
            "fragments": old_hashes,
            "binding_sha256": digest(baseline / "INFERENCE_FREEZE.json"),
            "manifest_sha256": digest(baseline / "QUERY_MANIFEST.json"),
        },
    )
    with (baseline / "SCORED_PREDICTIONS.csv").open("w", newline="") as h:
        records = [
            {**r, "truth_ttc_seconds": 1.0, **{m: i + 2 for m in MODELS[:-1]}}
            for i, r in enumerate(rows)
        ]
        writer = csv.DictWriter(h, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    atomic_json(
        output / "BASELINE_PRESERVATION.json",
        {
            "key_files": {
                n: digest(baseline / n)
                for n in (
                    "QUERY_MANIFEST.json",
                    "INFERENCE_FREEZE.json",
                    "PREDICTIONS_SEALED.json",
                    "SCORED_PREDICTIONS.csv",
                )
            }
        },
    )
    atomic_json(output / "INFERENCE_FREEZE.json", {"new_frozen": True})
    hashes = {}
    for i, row in enumerate(rows):
        name = f"predictions/query_{i:05d}.json"
        atomic_json(
            output / name,
            {
                **row,
                "ttc": {**read(baseline / name)["ttc"], MODELS[-1]: 4.0 if i == 0 else None},
                "binding_sha256": digest(output / "INFERENCE_FREEZE.json"),
                "baseline_fragment_sha256": old_hashes[name],
                "unavailable_reason": None if i == 0 else "timing",
            },
        )
        hashes[name] = digest(output / name)
    atomic_json(
        output / "PREDICTIONS_SEALED.json",
        {
            "status": "COMPLETE",
            "queries": 2,
            "fragments": hashes,
            "binding_sha256": digest(output / "INFERENCE_FREEZE.json"),
            "manifest_sha256": digest(baseline / "QUERY_MANIFEST.json"),
        },
    )
    return baseline, output


def test_common_support_and_unavailability(tmp_path: Path) -> None:
    baseline, output = fixture_pair(tmp_path)
    run(output, baseline)
    result = read(output / "RESULT.json")
    assert result["metrics"][MODELS[-1]]["coverage"] == 0.5
    assert result["metrics"][MODELS[0]]["mae_seconds"] == 1.5
    assert result["common_support_metrics"][MODELS[0]]["mae_seconds"] == 1.0
    assert result["paired_sequence_comparison"][MODELS[0]]["difference_seconds"] == -2.0


def test_reject_changed_fragment(tmp_path: Path) -> None:
    baseline, output = fixture_pair(tmp_path)
    (output / "predictions/query_00000.json").write_text("{}")
    with pytest.raises(ValueError, match="prediction changed"):
        run(output, baseline)


def test_reject_resealed_changed_h8(tmp_path: Path) -> None:
    baseline, output = fixture_pair(tmp_path)
    name = "predictions/query_00000.json"
    value = read(output / name)
    value["ttc"][MODELS[0]] = 99.0
    atomic_json(output / name, value)
    seal = read(output / "PREDICTIONS_SEALED.json")
    seal["fragments"][name] = digest(output / name)
    atomic_json(output / "PREDICTIONS_SEALED.json", seal)
    with pytest.raises(ValueError, match="inherited output mismatch"):
        run(output, baseline)


def test_pair_support_does_not_depend_on_other_comparator(tmp_path: Path) -> None:
    baseline, output = fixture_pair(tmp_path)
    name = "predictions/query_00000.json"
    for folder in (baseline, output):
        value = read(folder / name)
        value["ttc"]["public_Garl_event_lhr"] = None
        if folder == output:
            value["baseline_fragment_sha256"] = digest(baseline / name)
        atomic_json(folder / name, value)
        seal = read(folder / "PREDICTIONS_SEALED.json")
        seal["fragments"][name] = digest(folder / name)
        atomic_json(folder / "PREDICTIONS_SEALED.json", seal)
    pin = read(output / "BASELINE_PRESERVATION.json")
    pin["key_files"]["PREDICTIONS_SEALED.json"] = digest(baseline / "PREDICTIONS_SEALED.json")
    atomic_json(output / "BASELINE_PRESERVATION.json", pin)
    run(output, baseline)
    result = read(output / "RESULT.json")
    assert result["common_support_metrics"][MODELS[0]]["queries"] == 0
    assert result["paired_sequence_comparison"][MODELS[0]]["common_queries"] == 1
