#!/usr/bin/env python
"""Fit or apply the preregistered fixed-pool detection quality ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.evaluation_ground_truth import (  # noqa: E402
    EvaluationBox,
    EvaluationGroundTruth,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol  # noqa: E402
from e_jepa_ttc.detection.quality_ranking import (  # noqa: E402
    TrainingProvenance,
    apply_quality_ranker,
    apply_score_control,
    build_training_rows,
    fit_quality_ranker,
    token_population_sha256,
    verify_train_token_ownership,
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _array(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _int(value: object, name: str) -> int:
    number = _float(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _box(value: Mapping[str, object]) -> EvaluationBox:
    translation = tuple(
        _float(item, "translation")
        for item in _array(value.get("translation_m"), "translation")
    )
    size = tuple(
        _float(item, "size") for item in _array(value.get("size_lwh_m"), "size")
    )
    if len(translation) != 3 or len(size) != 3:
        raise ValueError("Serialized ground-truth vectors must contain three values")
    return EvaluationBox(
        class_name=str(value["class_name"]),
        translation_m=cast(tuple[float, float, float], translation),
        size_lwh_m=cast(tuple[float, float, float], size),
        yaw_rad=_float(value["yaw_rad"], "yaw_rad"),
        source_index=_int(value["source_index"], "source_index"),
        ignore_reason=(
            None if value.get("ignore_reason") is None else str(value["ignore_reason"])
        ),
    )


def restore_ground_truth(
    path: Path, protocol: EvaluationProtocol
) -> tuple[dict[str, EvaluationGroundTruth], dict[str, object]]:
    """Restore and validate the exact serialized v2 training population."""
    bundle = read_json(path)
    if not isinstance(bundle, dict) or bundle.get("schema") != (
        "phase_ttc_detection_ground_truth_bundle_v2"
    ):
        raise ValueError("Unexpected ground-truth bundle schema")
    if bundle.get("protocol_sha256") != protocol.sha256:
        raise ValueError("Ground-truth bundle uses another protocol")
    raw = bundle.get("ground_truth_by_token")
    sequence_id_by_token = bundle.get("sequence_id_by_token")
    if not isinstance(raw, dict) or not isinstance(sequence_id_by_token, dict):
        raise ValueError("Ground-truth bundle lacks token or canonical sequence metadata")
    result: dict[str, EvaluationGroundTruth] = {}
    for token, value in raw.items():
        if not isinstance(token, str) or not isinstance(value, dict):
            raise ValueError("Ground-truth token entry is invalid")
        ignored_by_reason = value.get("ignored_by_reason")
        if not isinstance(ignored_by_reason, dict):
            raise ValueError("Ground-truth ignore counts are missing")
        eligible = _array(value.get("eligible"), "eligible")
        ignored = _array(value.get("ignored"), "ignored")
        intrinsic = _array(value.get("intrinsic"), "intrinsic")
        event_from_ego = _array(value.get("event_from_ego"), "event_from_ego")
        image_wh = tuple(
            _int(item, "image_wh") for item in _array(value.get("image_wh"), "image_wh")
        )
        if len(image_wh) != 2:
            raise ValueError("image_wh must contain two values")
        result[token] = EvaluationGroundTruth(
            protocol_sha256=str(value["protocol_sha256"]),
            eligible=tuple(_box(cast(Mapping[str, object], item)) for item in eligible),
            ignored=tuple(_box(cast(Mapping[str, object], item)) for item in ignored),
            raw_count=_int(value["raw_count"], "raw_count"),
            ignored_by_reason=tuple(
                sorted(
                    (str(reason), _int(count, "ignored count"))
                    for reason, count in ignored_by_reason.items()
                )
            ),
            intrinsic=tuple(
                tuple(_float(item, "intrinsic") for item in _array(row, "intrinsic row"))
                for row in intrinsic
            ),
            event_from_ego=tuple(
                tuple(_float(item, "transform") for item in _array(row, "transform row"))
                for row in event_from_ego
            ),
            image_wh=cast(tuple[int, int], image_wh),
        )
    return result, cast(dict[str, object], sequence_id_by_token)


def load_predictions_and_sidecar(
    predictions_path: Path, sidecar_path: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, object]]], dict[str, object]]:
    """Load a sidecar only when it is bound to the exact prediction file."""
    document = read_json(predictions_path)
    sidecar = read_json(sidecar_path)
    if not isinstance(document, dict) or not isinstance(sidecar, dict):
        raise ValueError("Prediction output and sidecar must be JSON objects")
    raw_predictions = document.get("predictions")
    if not isinstance(raw_predictions, dict):
        raise ValueError("Prediction output lacks its proposal mapping")
    if sidecar.get("prediction_output_sha256") != sha256_file(predictions_path):
        raise ValueError("Sidecar is not bound to the supplied prediction file")
    predictions = cast(dict[str, list[dict[str, object]]], raw_predictions)
    return document, predictions, cast(dict[str, object], sidecar)


def fit(args: argparse.Namespace) -> None:
    protocol = EvaluationProtocol.default()
    _, predictions, sidecar = load_predictions_and_sidecar(args.predictions, args.sidecar)
    ground_truth, sequence_id_by_token = restore_ground_truth(args.ground_truth, protocol)
    split_manifest = read_json(args.split_manifest)
    if not isinstance(split_manifest, dict):
        raise ValueError("Split manifest must be a JSON object")
    ownership_sha256 = verify_train_token_ownership(
        sorted(predictions), split_manifest, sequence_id_by_token
    )
    provenance = TrainingProvenance(
        dataset_role=args.dataset_role,
        split_manifest_sha256=sha256_file(args.split_manifest),
        predictions_sha256=sha256_file(args.predictions),
        sidecar_sha256=sha256_file(args.sidecar),
        ground_truth_sha256=sha256_file(args.ground_truth),
        checkpoint_sha256=str(sidecar.get("checkpoint_sha256")),
        protocol_sha256=protocol.sha256,
        token_population_sha256=token_population_sha256(sorted(predictions)),
        train_ownership_sha256=ownership_sha256,
    )
    rows = build_training_rows(predictions, sidecar, ground_truth, protocol=protocol)
    write_new_json(args.output, fit_quality_ranker(rows, provenance=provenance))


def apply(args: argparse.Namespace) -> None:
    protocol = EvaluationProtocol.default()
    document, predictions, sidecar = load_predictions_and_sidecar(
        args.predictions, args.sidecar
    )
    bundle = read_json(args.bundle)
    if not isinstance(bundle, dict):
        raise ValueError("Ranker bundle must be a JSON object")
    rescored = apply_quality_ranker(predictions, sidecar, bundle, protocol=protocol)
    output = dict(document)
    output.update(
        {
            "schema": "phase_ttc_detection_ranked_predictions_v1",
            "source_schema": document.get("schema"),
            "predictions": rescored,
            "quality_ranker_bundle_sha256": bundle.get("bundle_sha256"),
            "label_access_during_apply": False,
            "fixed_proposal_pool": True,
        }
    )
    write_new_json(args.output, output)


def control(args: argparse.Namespace) -> None:
    """Materialize one preregistered fixed-pool ranking control."""
    protocol = EvaluationProtocol.default()
    document, predictions, sidecar = load_predictions_and_sidecar(
        args.predictions, args.sidecar
    )
    rescored = apply_score_control(
        predictions, sidecar, protocol=protocol, control=args.control
    )
    output = dict(document)
    output.update(
        {
            "schema": "phase_ttc_detection_ranking_control_v1",
            "source_schema": document.get("schema"),
            "predictions": rescored,
            "ranking_control": args.control,
            "label_access_during_apply": False,
            "fixed_proposal_pool": True,
        }
    )
    write_new_json(args.output, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit", help="fit once on verified TRAIN outputs and GT")
    fit_parser.add_argument("--predictions", type=Path, required=True)
    fit_parser.add_argument("--sidecar", type=Path, required=True)
    fit_parser.add_argument("--ground-truth", type=Path, required=True)
    fit_parser.add_argument("--split-manifest", type=Path, required=True)
    fit_parser.add_argument("--dataset-role", choices=("train",), required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    apply_parser = commands.add_parser("apply", help="apply without a label-bearing input")
    apply_parser.add_argument("--predictions", type=Path, required=True)
    apply_parser.add_argument("--sidecar", type=Path, required=True)
    apply_parser.add_argument("--bundle", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    control_parser = commands.add_parser("control", help="materialize a frozen score control")
    control_parser.add_argument("--predictions", type=Path, required=True)
    control_parser.add_argument("--sidecar", type=Path, required=True)
    control_parser.add_argument(
        "--control",
        choices=("baseline", "scalar_uncertainty", "monotonic_temperature"),
        required=True,
    )
    control_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fit":
            fit(args)
        elif args.command == "apply":
            apply(args)
        else:
            control(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
