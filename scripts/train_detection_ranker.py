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


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def verify_fit_evidence(
    *,
    recipe_path: Path,
    receipt_path: Path,
    predictions_path: Path,
    sidecar_path: Path,
    ground_truth_path: Path,
    split_path: Path,
    predictions_document: Mapping[str, object],
    predictions: Mapping[str, object],
    sidecar: Mapping[str, object],
    sequence_id_by_token: Mapping[str, object],
    protocol: EvaluationProtocol,
) -> tuple[str, str, str]:
    """Verify the post-inference panel proof before admitting any fit labels."""
    recipe = read_json(recipe_path)
    receipt = read_json(receipt_path)
    split = read_json(split_path)
    if not isinstance(recipe, dict) or recipe.get("schema") != (
        "phase_ttc_detection_ranking_panel_recipe_v1"
    ):
        raise ValueError("Unexpected ranking-panel recipe schema")
    unhashed = dict(recipe)
    expected_recipe_hash = unhashed.pop("recipe_sha256", None)
    if expected_recipe_hash != canonical_sha256(unhashed):
        raise ValueError("Ranking-panel recipe self-hash differs")
    if not isinstance(receipt, dict) or receipt.get("schema") != (
        "phase_ttc_detection_ranking_gt_receipt_v1"
    ):
        raise ValueError("Unexpected ranking GT receipt schema")
    recipe_sha256 = sha256_file(recipe_path)
    receipt_sha256 = sha256_file(receipt_path)
    if receipt.get("panel_recipe_sha256") != recipe_sha256:
        raise ValueError("GT receipt is not bound to the ranking-panel recipe")
    if receipt.get("panel") != "train4096" or receipt.get("role") != "train":
        raise ValueError("Ranker fitting requires the train4096 GT receipt")
    if (
        receipt.get("label_free_predictions_verified_before_labels") is not True
        or receipt.get("labels_read") is not True
        or receipt.get("test_labels_read") is not False
    ):
        raise ValueError("GT receipt does not prove label-free separation")
    expected_hashes = {
        "prediction_sha256": sha256_file(predictions_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "protocol_sha256": protocol.sha256,
    }
    if any(receipt.get(name) != value for name, value in expected_hashes.items()):
        raise ValueError("GT receipt content hashes differ from fit inputs")
    if recipe.get("candidate_split_sha256") != sha256_file(split_path):
        raise ValueError("Fit split differs from the ranking-panel recipe")
    panels = recipe.get("panels")
    panel = panels.get("train4096") if isinstance(panels, Mapping) else None
    if not isinstance(panel, Mapping) or panel.get("role") != "train":
        raise ValueError("Ranking-panel recipe lacks train4096")
    tokens = sorted(predictions)
    if len(tokens) != panel.get("token_count") or token_population_sha256(tokens) != panel.get(
        "token_population_sha256"
    ):
        raise ValueError("Fit prediction population differs from train4096")
    if (
        predictions_document.get("schema") != "eap_detection_campaign_inference_v1"
        or predictions_document.get("targets_argument") is not None
        or predictions_document.get("annotations_returned") != 0
    ):
        raise ValueError("Fit prediction artifact is not proven label-free")
    checkpoint_sha256 = recipe.get("checkpoint_sha256")
    if (
        predictions_document.get("checkpoint_sha256") != checkpoint_sha256
        or sidecar.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("Fit artifacts differ from the recipe checkpoint")
    if sidecar.get("prediction_output_sha256") != sha256_file(predictions_path):
        raise ValueError("Fit sidecar is not bound to prediction bytes")
    if sidecar.get("claims") != {
        "changes_predictions": False,
        "ground_truth_access": False,
        "internal_unexposed_values_reconstructed": False,
    }:
        raise ValueError("Fit sidecar does not prove external label-free observation")
    frames = sidecar.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Fit sidecar must contain exactly one frame per train4096 token")
    raw_frame_tokens = [
        frame.get("sample_token") for frame in frames if isinstance(frame, Mapping)
    ]
    if len(raw_frame_tokens) != len(frames) or any(
        not isinstance(token, str) or not token for token in raw_frame_tokens
    ):
        raise ValueError("Fit sidecar must contain exactly one frame per train4096 token")
    frame_tokens = cast(list[str], raw_frame_tokens)
    if len(frame_tokens) != len(set(frame_tokens)) or sorted(frame_tokens) != tokens:
        raise ValueError("Fit sidecar must contain exactly one frame per train4096 token")
    if not isinstance(split, dict):
        raise ValueError("Fit split must be a JSON object")
    ownership_sha256 = verify_train_token_ownership(
        tokens, split, sequence_id_by_token
    )
    if receipt.get("train_ownership_sha256") != ownership_sha256:
        raise ValueError("GT receipt TRAIN ownership differs from recomputation")
    return recipe_sha256, receipt_sha256, ownership_sha256


def fit(args: argparse.Namespace) -> None:
    protocol = EvaluationProtocol.default()
    document, predictions, sidecar = load_predictions_and_sidecar(
        args.predictions, args.sidecar
    )
    ground_truth, sequence_id_by_token = restore_ground_truth(args.ground_truth, protocol)
    split_manifest = read_json(args.split_manifest)
    if not isinstance(split_manifest, dict):
        raise ValueError("Split manifest must be a JSON object")
    recipe_sha256, receipt_sha256, ownership_sha256 = verify_fit_evidence(
        recipe_path=args.panel_recipe,
        receipt_path=args.ground_truth_receipt,
        predictions_path=args.predictions,
        sidecar_path=args.sidecar,
        ground_truth_path=args.ground_truth,
        split_path=args.split_manifest,
        predictions_document=document,
        predictions=predictions,
        sidecar=sidecar,
        sequence_id_by_token=sequence_id_by_token,
        protocol=protocol,
    )
    provenance = TrainingProvenance(
        dataset_role=args.dataset_role,
        split_manifest_sha256=sha256_file(args.split_manifest),
        panel_recipe_sha256=recipe_sha256,
        ground_truth_receipt_sha256=receipt_sha256,
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
    fit_parser.add_argument("--panel-recipe", type=Path, required=True)
    fit_parser.add_argument("--ground-truth-receipt", type=Path, required=True)
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
