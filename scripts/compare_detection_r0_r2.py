"""Verify completed seed-7 runs and summarize fixed-budget development evaluations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def digest(path: Path) -> str:
    """Hash a source or evidence file with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    """Retain denominators and metrics, excluding per-prediction PR arrays."""
    return {
        "sample_count": metrics["sample_count"],
        "protocol_sha256": metrics["protocol_sha256"],
        "AP": metrics["dev_AP101_supported_classes"],
        "population": {
            key: value
            for key, value in metrics["population"].items()
            if key != "ground_truth_sha256_by_token"
        },
        "class_agnostic_recall": metrics["class_agnostic_recall"],
        "per_class": {
            category: {
                key: value
                for key, value in row.items()
                if key not in {"thresholds", "diagnostic_matches"}
            }
            | {
                "thresholds": {
                    threshold: {
                        key: value
                        for key, value in result.items()
                        if key not in {"pr_curve", "ignored_predictions"}
                    }
                    for threshold, result in row["thresholds"].items()
                }
            }
            for category, row in metrics["per_class"].items()
        },
    }


def write(path: Path, value: Any) -> None:
    """Write finite JSON evidence atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(args: argparse.Namespace) -> None:
    """Use the frozen queue verifier and capture compact metrics during its reads."""
    spec = importlib.util.spec_from_file_location("frozen_queue", args.verifier)
    assert spec is not None and spec.loader is not None
    helper: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original_load = helper._load_object
    evaluations: dict[str, Any] = {}

    def capture(path: Path) -> dict[str, Any]:
        payload = original_load(path)
        if "metrics_v2" in payload:
            # Recompute every serialized GT hash, beyond checking the stored identity map.
            expected = payload["metrics_v2"]["population"]["ground_truth_sha256_by_token"]
            for token, frame in payload["ground_truth_v2"].items():
                encoded = json.dumps(
                    frame, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                if hashlib.sha256(encoded).hexdigest() != expected[token]:
                    raise ValueError(f"GT contents/hash mismatch: {path}: {token}")
            evaluations[str(path)] = compact(payload["metrics_v2"])
        return payload

    helper._load_object = capture
    result: dict[str, Any] = {"schema": "phase_r0_r2_seed7_comparison_v1", "runs": {}}
    population_receipts = {}
    for name, output, start, resume in (
        ("r0", args.r0, 4096, "a187fd777b73d0386f186a1b86fa5483406762acc8d9aa65539e33eb46e4c6aa"),
        ("r2", args.r2, 0, None),
    ):
        summary = json.loads((output / "summary.json").read_text())
        config = helper.load_campaign_config(Path(summary["config"]))
        receipt = helper.verify_summary(
            output,
            config,
            expected_status="COMPLETED",
            expected_final_update=8192,
            expected_start_update=start,
            resume_sha256=resume,
        )
        full = helper.verify_full_evaluations(output, summary)
        population_receipts[name] = full
        result["runs"][name] = {
            "completion_receipt": receipt,
            "config": json.loads((output / "config_resolved.json").read_text()),
            "full": {
                update: {
                    key: value
                    for key, value in entry.items()
                    if key != "ground_truth_sha256_by_token"
                }
                | {"metrics": evaluations[str(entry["path"])]}
                for update, entry in full.items()
            },
        }
        write(args.output / f"{name}_verified.json", result["runs"][name])
        print(
            name,
            "verified",
            {u: row["metrics"]["AP"] for u, row in result["runs"][name]["full"].items()},
            flush=True,
        )
        gc.collect()
    helper.verify_population_same(population_receipts["r0"], population_receipts["r2"])
    result["same_full_population_and_protocol"] = True
    result["verifier_sha256"] = digest(args.verifier)
    result["analysis_script_sha256"] = digest(Path(__file__))
    result["scope"] = "Single-seed development comparison; not official eAP or independent test."
    write(args.output / "comparison.json", result)


def diagnose(args: argparse.Namespace) -> None:
    """Reevaluate final monitor panels and apply bounded GT-only diagnostic oracles."""
    sys.path.insert(0, str(args.project / "src"))
    from e_jepa_ttc.detection.error_diagnostics import (
        class_agnostic_correspondence,
        oracle_predictions,
    )
    from e_jepa_ttc.detection.evaluation_ground_truth import EvaluationBox, EvaluationGroundTruth
    from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
    from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2

    results: dict[str, Any] = {}
    identity = None
    rare = args.mode == "rare"
    suffix = "rare_full" if rare else "diagnostics"
    for name, output in (("r0", args.r0), ("r2", args.r2)):
        filename = "full_selection_update_008192.json" if rare else "selection_update_008192.json"
        path = output / "evaluation" / filename
        payload = json.loads(path.read_text())
        summary = json.loads((output / "summary.json").read_text())
        protocol = EvaluationProtocol.from_dict(summary["evaluation_protocol"])
        truth = {}
        for token, frame in payload["ground_truth_v2"].items():
            if rare and not any(
                box["class_name"] in {"bicycle", "other_vehicle"} for box in frame["eligible"]
            ):
                continue
            values = dict(frame)
            for key in ("eligible", "ignored"):
                values[key] = tuple(
                    EvaluationBox(
                        **(
                            box
                            | {
                                "translation_m": tuple(box["translation_m"]),
                                "size_lwh_m": tuple(box["size_lwh_m"]),
                            }
                        )
                    )
                    for box in frame[key]
                )
            values["ignored_by_reason"] = tuple(sorted(frame["ignored_by_reason"].items()))
            for key in ("intrinsic", "event_from_ego"):
                values[key] = tuple(tuple(row) for row in frame[key])
            values["image_wh"] = tuple(frame["image_wh"])
            truth[token] = EvaluationGroundTruth(**values)
        hashes = {token: frame.sha256 for token, frame in truth.items()}
        expected = payload["metrics_v2"]["population"]["ground_truth_sha256_by_token"]
        if hashes != {token: expected[token] for token in truth}:
            raise ValueError("Reconstructed GT identity differs")
        if identity is not None and identity != hashes:
            raise ValueError("Monitor populations differ")
        identity = hashes
        predictions = {token: payload["predictions"][token] for token in truth}
        baseline = payload["metrics_v2"]
        if not rare:
            baseline = evaluate_detections_v2(predictions, truth, protocol=protocol)
            if compact(baseline) != compact(payload["metrics_v2"]):
                raise ValueError("Independent reevaluation differs from frozen stored metrics")
        baseline_summary = compact(baseline)
        del payload, baseline
        gc.collect()
        pairs = class_agnostic_correspondence(
            predictions,
            truth,
            protocol=protocol,
            threshold_m=2.0,
            max_predictions_per_frame=protocol.max_predictions_per_frame,
        )
        by_class = {}
        for category in ("bicycle", "other_vehicle") if rare else protocol.classes:
            rows = [row for row in pairs if row["target_class"] == category]
            by_class[category] = {
                "eligible": sum(
                    box.class_name == category for frame in truth.values() for box in frame.eligible
                ),
                "matched_agnostic_2m": len(rows),
                "predicted_classes": dict(Counter(row["predicted_class"] for row in rows)),
                "conditional_mean_errors": {
                    key: mean(row[key] for row in rows) if rows else None
                    for key in (
                        "ego_xy_error_m",
                        "ego_z_vertical_error_m",
                        "camera_depth_error_m",
                        "camera_xy_error_m",
                        "yaw_rad",
                    )
                },
            }
        oracles = {}
        for kind in () if rare else ("class", "center", "geometry", "ranking"):
            changed = oracle_predictions(predictions, truth, protocol=protocol, kind=kind)
            metrics = evaluate_detections_v2(changed, truth, protocol=protocol)
            oracles[kind] = compact(metrics)
            print(name, kind, metrics["dev_AP101_supported_classes"], flush=True)
        results[name] = {
            "source": str(path),
            "sha256": digest(path),
            "baseline_parity": None if rare else True,
            "baseline": baseline_summary,
            "diagnostic_frame_count": len(truth),
            "class_agnostic_pairs_2m": by_class,
            "oracles": oracles,
        }
        write(args.output / f"{name}_{suffix}.json", results[name])
        print(name, suffix, by_class if rare else "complete", flush=True)
    results["source_hashes"] = {
        name: digest(args.project / "src/e_jepa_ttc/detection" / name)
        for name in (
            "error_diagnostics.py",
            "evaluation_v2.py",
            "evaluation_protocol.py",
            "evaluation_ground_truth.py",
            "geometry_contract.py",
        )
    }
    results["scope"] = (
        "Final 1024-frame monitor panel; GT diagnostic transformations at bounded 2m "
        "class-agnostic correspondence, then rematching. Not achievable performance or "
        "global oracle upper bounds. Geometry errors condition on matched objects."
    )
    if rare:
        results["scope"] = (
            "All eligible bicycle/other_vehicle annotations from full selection. "
            "Retain all objects and predictions in their frames for unchanged greedy "
            "one-to-one 2m matching, then aggregate only target rare classes. "
            "Baseline metrics are stored full-selection results, not reevaluated here. "
            "No oracle interventions. Geometry errors condition on matched objects."
        )
    results["analysis_script_sha256"] = digest(Path(__file__))
    write(args.output / f"{suffix}.json", results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "diagnose", "rare"))
    for name in ("project", "r0", "r2", "verifier", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(args.project)
    (verify if args.mode == "verify" else diagnose)(args)


if __name__ == "__main__":
    main()
