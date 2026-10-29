#!/usr/bin/env python
"""Evaluate pilot predictions against train-only controls under frozen detection v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.evaluation_controls import evaluate_controls  # noqa: E402
from e_jepa_ttc.detection.evaluation_ground_truth import (  # noqa: E402
    EvaluationBox,
    EvaluationGroundTruth,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol  # noqa: E402


def sha256(path: Path) -> str:
    """Hash an input incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object while refusing duplicate keys."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _box(value: dict[str, object]) -> EvaluationBox:
    expected = {
        "class_name",
        "translation_m",
        "size_lwh_m",
        "yaw_rad",
        "source_index",
        "ignore_reason",
    }
    if set(value) != expected:
        raise ValueError("Serialized EvaluationBox fields differ from v2")
    return EvaluationBox(
        class_name=str(value["class_name"]),
        translation_m=tuple(value["translation_m"]),  # type: ignore[arg-type]
        size_lwh_m=tuple(value["size_lwh_m"]),  # type: ignore[arg-type]
        yaw_rad=float(value["yaw_rad"]),
        source_index=int(value["source_index"]),
        ignore_reason=value["ignore_reason"],  # type: ignore[arg-type]
    )


def _ground_truth(value: dict[str, object]) -> EvaluationGroundTruth:
    expected = {
        "protocol_sha256",
        "eligible",
        "ignored",
        "raw_count",
        "ignored_by_reason",
        "intrinsic",
        "event_from_ego",
        "image_wh",
    }
    if set(value) != expected:
        raise ValueError("Serialized EvaluationGroundTruth fields differ from v2")
    ignored_by_reason = value["ignored_by_reason"]
    if not isinstance(ignored_by_reason, dict):
        raise ValueError("ignored_by_reason must be an object")
    return EvaluationGroundTruth(
        protocol_sha256=str(value["protocol_sha256"]),
        eligible=tuple(_box(row) for row in value["eligible"]),  # type: ignore[arg-type]
        ignored=tuple(_box(row) for row in value["ignored"]),  # type: ignore[arg-type]
        raw_count=int(value["raw_count"]),
        ignored_by_reason=tuple(
            sorted((str(reason), int(count)) for reason, count in ignored_by_reason.items())
        ),
        intrinsic=tuple(tuple(float(item) for item in row) for row in value["intrinsic"]),  # type: ignore[union-attr]
        event_from_ego=tuple(
            tuple(float(item) for item in row)
            for row in value["event_from_ego"]  # type: ignore[union-attr]
        ),
        image_wh=tuple(value["image_wh"]),  # type: ignore[arg-type]
    )


def render(report: dict[str, object]) -> str:
    """Render the compact result and its scientific limits."""
    bootstrap = report["paired_sequence_bootstrap"]
    assert isinstance(bootstrap, dict)
    intervals = bootstrap["observed_minus_control"]
    assert isinstance(intervals, dict)
    observed = report["observed_model_metrics"]
    constant = report["constant_prior"]
    derangement = report["frame_derangement"]
    assert isinstance(observed, dict)
    assert isinstance(constant, dict) and isinstance(constant["metrics"], dict)
    assert isinstance(derangement, dict) and isinstance(derangement["metrics"], dict)
    lines = [
        "# Controles S0 del detector bajo evaluación v2",
        "",
        "Los controles usan GT v2 idéntico, un prior ajustado solo con train y una "
        "permutación prefijada de listas completas. El resultado es local y no oficial.",
        "",
        f"- AP observado: `{observed['dev_AP101_supported_classes']}`.",
        f"- AP prior constante: `{constant['metrics']['dev_AP101_supported_classes']}`.",
        f"- AP permutación: `{derangement['metrics']['dev_AP101_supported_classes']}`.",
        f"- Clusters de bootstrap: `{bootstrap['sequence_count']}`.",
        f"- Réplicas: `{bootstrap['num_replicates']}`.",
        "",
        "## Diferencias emparejadas (observado − control)",
        "",
    ]
    for name, value in intervals.items():
        lines.append(f"- {name}: mediana `{value['median']}`, IC95 `{value['ci95']}`.")
    lines.extend(
        [
            "",
            "Con solo dos secuencias, los intervalos por cluster son débiles y no acreditan "
            "generalización. Las réplicas sin soporte de una clase permanecen `null`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    """Preregister the CPU budget, validate all inputs, then evaluate controls."""
    if args.output.exists():
        raise FileExistsError("Use a new controls output directory")
    if args.replicates < 1 or args.seed < 0:
        raise ValueError("replicates must be positive and seed nonnegative")
    input_paths = {
        "train_results": args.train_results.resolve(),
        "selection_predictions": args.selection_predictions.resolve(),
        "selection_ground_truth": args.selection_ground_truth.resolve(),
        "protocol": args.protocol.resolve(),
    }
    inputs = {name: sha256(path) for name, path in input_paths.items()}
    args.output.mkdir(parents=True)
    preregistration = {
        "schema": "phase_ttc_detection_controls_preregistration_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "train constant prior, first seed derangement, paired sequence bootstrap",
        "num_replicates": args.replicates,
        "seed": args.seed,
        "bootstrap_unit": "complete sequence",
        "inputs_sha256": inputs,
        "official_eap_score": False,
    }
    write_json(args.output / "preregistration.json", preregistration)
    protocol = EvaluationProtocol.from_dict(load_json(input_paths["protocol"]))
    bundle = load_json(input_paths["selection_ground_truth"])
    if bundle.get("schema") != "phase_ttc_detection_ground_truth_bundle_v2":
        raise ValueError("Unexpected v2 ground-truth bundle schema")
    if bundle.get("protocol_sha256") != protocol.sha256:
        raise ValueError("Ground-truth bundle protocol differs from the declared protocol")
    raw_ground_truth = bundle.get("ground_truth_by_token")
    sequence_by_token = bundle.get("sequence_id_by_token")
    if not isinstance(raw_ground_truth, dict) or not isinstance(sequence_by_token, dict):
        raise ValueError("Ground-truth bundle lacks populations or sequence assignments")
    ground_truth = {token: _ground_truth(value) for token, value in raw_ground_truth.items()}
    if any(not isinstance(value, str) or not value for value in sequence_by_token.values()):
        raise ValueError("Sequence assignments must be nonempty strings")
    frozen = load_json(input_paths["selection_predictions"])
    if (
        frozen.get("schema") != "monodgp_eap_label_free_inference_v1"
        or frozen.get("ground_truth_in_output") is not False
    ):
        raise ValueError("Selection input is not the frozen label-free inference receipt")
    predictions = frozen.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("Frozen inference lacks predictions")
    train = load_json(input_paths["train_results"])
    train_truth = train.get("ground_truth")
    if not isinstance(train_truth, dict):
        raise ValueError("Train result lacks trainer-admitted ground truth")
    report = evaluate_controls(
        train_truth,
        predictions,
        ground_truth,
        sequence_by_token,
        protocol=protocol,
        num_replicates=args.replicates,
        seed=args.seed,
    )
    report["provenance"] = {
        "inputs_sha256": inputs,
        "preregistration_sha256": sha256(args.output / "preregistration.json"),
        "script_sha256": sha256(Path(__file__)),
    }
    write_json(args.output / "controls.json", report)
    (args.output / "CONTROLS.md").write_text(render(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--selection-predictions", type=Path, required=True)
    parser.add_argument("--selection-ground-truth", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
