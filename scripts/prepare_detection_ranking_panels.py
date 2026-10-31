#!/usr/bin/env python
"""Prepare frozen R0 ranking panels and post-inference independent v2 GT bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.evaluation_ground_truth import (  # noqa: E402
    EvaluationGroundTruth,
    build_evaluation_ground_truth,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol  # noqa: E402
from e_jepa_ttc.detection.quality_ranking import (  # noqa: E402
    token_population_sha256,
    verify_train_token_ownership,
)

RECIPE_SCHEMA = "phase_ttc_detection_ranking_panel_recipe_v1"
GT_SCHEMA = "phase_ttc_detection_ground_truth_bundle_v2"
PANELS = {"train4096": "train", "selection1024": "selection", "selection_full": "selection"}


def sha256_file(path: Path) -> str:
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


def read_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _strings(value: object, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    result = list(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return cast(list[str], result)


def _split_owners(split: Mapping[str, object]) -> dict[str, str]:
    assignments = split.get("assignments")
    if not isinstance(assignments, Mapping) or not {"train", "selection"}.issubset(assignments):
        raise ValueError("Split lacks train/selection assignments")
    owner: dict[str, str] = {}
    for role, values in assignments.items():
        if not isinstance(role, str):
            raise ValueError("Split role names must be strings")
        for sequence in _strings(values, f"assignments.{role}"):
            if sequence in owner:
                raise ValueError("Split sequence ownership overlaps")
            owner[sequence] = role
    return owner


def create_recipe(
    *,
    campaign_manifest_path: Path,
    population_audit_path: Path,
    split_path: Path,
    public_files_manifest_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    train_count: int = 4096,
    selection_count: int = 1024,
    selection_panel_name: str = "selection1024",
) -> dict[str, object]:
    """Create label-free token files and bind all future inputs without reading labels."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    campaign = read_object(campaign_manifest_path, "campaign manifest")
    population_audit = read_object(population_audit_path, "population audit")
    split = read_object(split_path, "candidate split")
    public_files = read_object(public_files_manifest_path, "public-files manifest")
    if campaign.get("schema") != "eap_detection_campaign_manifest_v1":
        raise ValueError("Unexpected campaign manifest schema")
    if population_audit.get("schema") != "eap_detection_population_audit_v2":
        raise ValueError("Unexpected population-audit schema")
    if campaign.get("population_audit_sha256") != canonical_sha256(population_audit):
        raise ValueError("Campaign manifest is not bound to the population audit")
    if split.get("schema") != "eap_detection_sequence_split_v1":
        raise ValueError("Unexpected candidate split schema")
    if public_files.get("schema_version") != "eap_detection_public_files_v1":
        raise ValueError("Unexpected public-files manifest schema")
    if campaign.get("dataset_revision") != split.get("dataset_revision") or campaign.get(
        "dataset_revision"
    ) != public_files.get("revision"):
        raise ValueError("Dataset revisions differ across panel inputs")
    if campaign.get("candidate_split_sha256") != sha256_file(split_path):
        raise ValueError("Campaign manifest is not bound to the supplied split")
    media_path = data_root / "data/train.parquet"
    media_sha256 = sha256_file(media_path)
    if (
        campaign.get("source_media_sha256") != media_sha256
        or split.get("media_sha256") != media_sha256
    ):
        raise ValueError("Canonical metadata differs from campaign or split")
    files = public_files.get("files")
    media_receipt = files.get("data/train.parquet") if isinstance(files, Mapping) else None
    if not isinstance(media_receipt, Mapping) or media_receipt.get("sha256") != media_sha256:
        raise ValueError("Public-files manifest does not bind canonical metadata")
    owners = _split_owners(split)
    image_geometry = population_audit.get("image_geometry")
    if not isinstance(image_geometry, Mapping):
        raise ValueError("Population audit lacks image geometry")
    source_wh = image_geometry.get("source_wh")
    if source_wh != [1280, 720]:
        raise ValueError("Population audit source geometry differs from provisional v2")
    sampled = image_geometry.get("sampled_one_frame_per_sequence")
    if not isinstance(sampled, list):
        raise ValueError("Population audit lacks per-sequence image samples")
    sampled_by_sequence = {
        row.get("sequence_id"): row.get("image_wh") for row in sampled if isinstance(row, Mapping)
    }
    roles = campaign.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("Campaign manifest lacks roles")
    if selection_panel_name not in {"selection1024", "selection_full"}:
        raise ValueError("Unknown selection panel name")
    counts = {"train4096": train_count, selection_panel_name: selection_count}
    output_dir.mkdir(parents=True)
    panel_receipts: dict[str, object] = {}
    for panel, role in {"train4096": "train", selection_panel_name: "selection"}.items():
        entry = roles.get(role)
        if not isinstance(entry, Mapping):
            raise ValueError(f"Campaign manifest lacks role {role}")
        tokens = _strings(entry.get("sample_tokens"), f"roles.{role}.sample_tokens")
        sequences = _strings(entry.get("sequence_ids"), f"roles.{role}.sequence_ids")
        if len(tokens) != counts[panel] or len(tokens) != len(set(tokens)):
            raise ValueError(f"{panel} does not have its exact unique token budget")
        if {token.split(":", 1)[0] for token in tokens} != set(sequences):
            raise ValueError(f"{panel} token prefixes differ from declared sequences")
        if any(owners.get(sequence) != role for sequence in sequences):
            raise ValueError(f"{panel} sequences are not owned by split role {role}")
        if any(sampled_by_sequence.get(sequence) != source_wh for sequence in sequences):
            raise ValueError(f"{panel} lacks one audited source-size frame per sequence")
        token_path = output_dir / f"{panel}_tokens.json"
        write_new(token_path, {"sample_tokens": tokens})
        panel_receipts[panel] = {
            "role": role,
            "token_count": len(tokens),
            "sequence_ids": sorted(sequences),
            "tokens_file": token_path.name,
            "tokens_file_sha256": sha256_file(token_path),
            "token_population_sha256": token_population_sha256(tokens),
        }
    recipe: dict[str, object] = {
        "schema": RECIPE_SCHEMA,
        "dataset_revision": campaign["dataset_revision"],
        "source_split": "train",
        "source_media_sha256": media_sha256,
        "campaign_manifest_sha256": sha256_file(campaign_manifest_path),
        "population_audit_file_sha256": sha256_file(population_audit_path),
        "population_audit_canonical_sha256": canonical_sha256(population_audit),
        "source_image_wh": source_wh,
        "image_geometry_evidence": (
            "inherited public-release invariant plus one decoded RGB member per sequence; "
            "not every RGB header was decoded"
        ),
        "candidate_split_sha256": sha256_file(split_path),
        "public_files_manifest_sha256": sha256_file(public_files_manifest_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "panels": panel_receipts,
        "separation": {
            "inference_loader": "label-free; annotations_returned must equal zero",
            "ground_truth_stage": "permitted only after prediction and sidecar verification",
            "train_ownership": "split.assignments.train plus canonical sequence metadata",
            "test_labels_read": False,
        },
        "commands": {
            "inference": (
                "python scripts/infer_detection_campaign.py --checkpoint <checkpoint> "
                "--data-root <data_root> --tokens <panel_tokens.json> "
                "--output <predictions.json> --diagnostic-sidecar <sidecar.json>"
            ),
            "ground_truth": (
                "python scripts/prepare_detection_ranking_panels.py build-gt "
                "--recipe <recipe.json> --panel <panel> --predictions <predictions.json> "
                "--sidecar <sidecar.json> --data-root <data_root> --split <split.json> "
                "--public-files-manifest <manifest.json> --output <ground_truth_v2.json> "
                "--receipt <ground_truth_receipt.json>"
            ),
        },
    }
    recipe["recipe_sha256"] = canonical_sha256(recipe)
    write_new(output_dir / "ranking_panel_recipe.json", recipe)
    return recipe


def _verify_inference(
    recipe: Mapping[str, object],
    panel: Mapping[str, object],
    predictions_path: Path,
    sidecar_path: Path,
) -> tuple[dict[str, list[dict[str, object]]], list[str]]:
    predictions_document = read_object(predictions_path, "predictions")
    sidecar = read_object(sidecar_path, "diagnostic sidecar")
    if predictions_document.get("schema") != "eap_detection_campaign_inference_v1":
        raise ValueError("Unexpected campaign inference schema")
    if (
        predictions_document.get("targets_argument") is not None
        or predictions_document.get("annotations_returned") != 0
    ):
        raise ValueError("Prediction artifact is not proven label-free")
    if predictions_document.get("checkpoint_sha256") != recipe.get("checkpoint_sha256"):
        raise ValueError("Prediction checkpoint differs from panel recipe")
    raw_predictions = predictions_document.get("predictions")
    if not isinstance(raw_predictions, dict):
        raise ValueError("Prediction artifact lacks proposal mapping")
    tokens = sorted(raw_predictions)
    if len(tokens) != panel.get("token_count") or token_population_sha256(tokens) != panel.get(
        "token_population_sha256"
    ):
        raise ValueError("Prediction token population differs from panel recipe")
    if sidecar.get("schema") != "monodgp_external_diagnostic_sidecar_v1":
        raise ValueError("Unexpected diagnostic sidecar schema")
    if sidecar.get("checkpoint_sha256") != recipe.get("checkpoint_sha256"):
        raise ValueError("Sidecar checkpoint differs from panel recipe")
    if sidecar.get("prediction_output_sha256") != sha256_file(predictions_path):
        raise ValueError("Sidecar is not bound to prediction bytes")
    claims = sidecar.get("claims")
    if claims != {
        "changes_predictions": False,
        "ground_truth_access": False,
        "internal_unexposed_values_reconstructed": False,
    }:
        raise ValueError("Sidecar does not prove external label-free observation")
    frames = sidecar.get("frames")
    raw_frame_tokens = (
        [frame.get("sample_token") for frame in frames if isinstance(frame, Mapping)]
        if isinstance(frames, list)
        else []
    )
    if (
        not isinstance(frames, list)
        or len(raw_frame_tokens) != len(frames)
        or any(not isinstance(token, str) or not token for token in raw_frame_tokens)
    ):
        raise ValueError("Sidecar token population differs from predictions")
    frame_tokens = [token for token in raw_frame_tokens if isinstance(token, str)]
    if (
        len(frame_tokens) != len(set(frame_tokens))
        or len(frame_tokens) != len(tokens)
        or set(frame_tokens) != set(tokens)
    ):
        raise ValueError("Sidecar token population differs from predictions")
    return cast(dict[str, list[dict[str, object]]], raw_predictions), tokens


def _verify_pinned_file(data_root: Path, relative: str, public_files: Mapping[str, object]) -> str:
    files = public_files.get("files")
    receipt = files.get(relative) if isinstance(files, Mapping) else None
    if not isinstance(receipt, Mapping):
        raise ValueError(f"Pinned public-files manifest lacks {relative}")
    path = (data_root / relative).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise ValueError("Dataset path escapes its root")
    if path.stat().st_size != receipt.get("size_bytes"):
        raise ValueError(f"Pinned dataset size differs: {relative}")
    observed = sha256_file(path)
    if observed != receipt.get("sha256"):
        raise ValueError(f"Pinned dataset hash differs: {relative}")
    return observed


def build_ground_truth_bundle(
    *,
    recipe_path: Path,
    panel_name: str,
    predictions_path: Path,
    sidecar_path: Path,
    data_root: Path,
    split_path: Path,
    public_files_manifest_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Build independent GT only after frozen label-free artifacts pass verification."""
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite GT evidence")
    recipe = read_object(recipe_path, "panel recipe")
    unhashed = dict(recipe)
    expected_recipe_hash = unhashed.pop("recipe_sha256", None)
    if recipe.get("schema") != RECIPE_SCHEMA or expected_recipe_hash != canonical_sha256(unhashed):
        raise ValueError("Panel recipe schema or self-hash differs")
    panels = recipe.get("panels")
    panel = panels.get(panel_name) if isinstance(panels, Mapping) else None
    if panel_name not in PANELS or not isinstance(panel, Mapping):
        raise ValueError("Unknown ranking panel")
    _, tokens = _verify_inference(recipe, panel, predictions_path, sidecar_path)
    split = read_object(split_path, "candidate split")
    public_files = read_object(public_files_manifest_path, "public-files manifest")
    if sha256_file(split_path) != recipe.get("candidate_split_sha256") or sha256_file(
        public_files_manifest_path
    ) != recipe.get("public_files_manifest_sha256"):
        raise ValueError("Split or public-files manifest differs from panel recipe")
    media_sha256 = _verify_pinned_file(data_root, "data/train.parquet", public_files)
    if media_sha256 != recipe.get("source_media_sha256"):
        raise ValueError("Canonical metadata differs from panel recipe")
    media_rows = pq.read_table(
        data_root / "data/train.parquet",
        columns=[
            "sample_token",
            "sequence_id",
            "split",
            "labels_path",
            "K_event",
            "T_event_ego",
        ],
        filters=[("sample_token", "in", tokens)],
    ).to_pylist()
    if len(media_rows) != len(tokens) or {str(row["sample_token"]) for row in media_rows} != set(
        tokens
    ):
        raise ValueError("Canonical metadata is incomplete for the panel")
    sequence_id_by_token = {str(row["sample_token"]): str(row["sequence_id"]) for row in media_rows}
    owners = _split_owners(split)
    role = str(panel["role"])
    if any(
        row["split"] != "train" or owners.get(str(row["sequence_id"])) != role for row in media_rows
    ):
        raise ValueError("Canonical metadata ownership differs from the panel role")
    train_ownership = None
    if role == "train":
        train_ownership = verify_train_token_ownership(tokens, split, sequence_id_by_token)
    annotations_by_token: dict[str, list[dict[str, object]]] = {token: [] for token in tokens}
    label_hashes: dict[str, str] = {}
    labels_by_path: dict[str, list[str]] = {}
    for row in media_rows:
        labels_by_path.setdefault(str(row["labels_path"]), []).append(str(row["sample_token"]))
    for relative, requested in sorted(labels_by_path.items()):
        label_hashes[relative] = _verify_pinned_file(data_root, relative, public_files)
        table = pq.read_table(
            data_root / relative,
            columns=["sample_token", "split", "category", "bbox_3d_ego"],
            filters=[("sample_token", "in", requested)],
        )
        for row in table.to_pylist():
            token = str(row["sample_token"])
            if row["split"] != "train" or token not in annotations_by_token:
                raise ValueError("Label query escaped the frozen public-train panel")
            annotations_by_token[token].append(cast(dict[str, object], row))
    protocol = EvaluationProtocol.default()
    source_image_wh = recipe.get("source_image_wh")
    if not isinstance(source_image_wh, list) or source_image_wh != [1280, 720]:
        raise ValueError("Recipe source geometry differs from provisional v2")
    source_wh = cast(list[int], source_image_wh)
    ground_truth: dict[str, EvaluationGroundTruth] = {}
    for row in media_rows:
        token = str(row["sample_token"])
        ground_truth[token] = build_evaluation_ground_truth(
            annotations_by_token[token],
            intrinsic=np.stack([np.asarray(value, dtype=np.float64) for value in row["K_event"]]),
            event_from_ego=np.stack(
                [np.asarray(value, dtype=np.float64) for value in row["T_event_ego"]]
            ),
            image_wh=(source_wh[0], source_wh[1]),
            protocol=protocol,
        )
    bundle = {
        "schema": GT_SCHEMA,
        "protocol_sha256": protocol.sha256,
        "sequence_id_by_token": dict(sorted(sequence_id_by_token.items())),
        "ground_truth_by_token": {
            token: ground_truth[token].to_dict() for token in sorted(ground_truth)
        },
    }
    write_new(output_path, bundle)
    receipt = {
        "schema": "phase_ttc_detection_ranking_gt_receipt_v1",
        "panel": panel_name,
        "role": role,
        "panel_recipe_sha256": sha256_file(recipe_path),
        "label_free_predictions_verified_before_labels": True,
        "prediction_sha256": sha256_file(predictions_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "ground_truth_sha256": sha256_file(output_path),
        "protocol_sha256": protocol.sha256,
        "label_file_sha256": label_hashes,
        "train_ownership_sha256": train_ownership,
        "labels_read": True,
        "test_labels_read": False,
    }
    write_new(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write label-free panel token files and recipe")
    plan.add_argument("--campaign-manifest", type=Path, required=True)
    plan.add_argument("--population-audit", type=Path, required=True)
    plan.add_argument("--split", type=Path, required=True)
    plan.add_argument("--public-files-manifest", type=Path, required=True)
    plan.add_argument("--checkpoint", type=Path, required=True)
    plan.add_argument("--data-root", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--train-count", type=int, default=4096)
    plan.add_argument("--selection-count", type=int, default=1024)
    plan.add_argument(
        "--selection-panel-name",
        choices=("selection1024", "selection_full"),
        default="selection1024",
    )
    gt = commands.add_parser("build-gt", help="build v2 GT after label-free artifact verification")
    gt.add_argument("--recipe", type=Path, required=True)
    gt.add_argument("--panel", choices=tuple(PANELS), required=True)
    gt.add_argument("--predictions", type=Path, required=True)
    gt.add_argument("--sidecar", type=Path, required=True)
    gt.add_argument("--data-root", type=Path, required=True)
    gt.add_argument("--split", type=Path, required=True)
    gt.add_argument("--public-files-manifest", type=Path, required=True)
    gt.add_argument("--output", type=Path, required=True)
    gt.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            create_recipe(
                campaign_manifest_path=args.campaign_manifest,
                population_audit_path=args.population_audit,
                split_path=args.split,
                public_files_manifest_path=args.public_files_manifest,
                checkpoint_path=args.checkpoint,
                data_root=args.data_root,
                output_dir=args.output_dir,
                train_count=args.train_count,
                selection_count=args.selection_count,
                selection_panel_name=args.selection_panel_name,
            )
        else:
            build_ground_truth_bundle(
                recipe_path=args.recipe,
                panel_name=args.panel,
                predictions_path=args.predictions,
                sidecar_path=args.sidecar,
                data_root=args.data_root,
                split_path=args.split,
                public_files_manifest_path=args.public_files_manifest,
                output_path=args.output,
                receipt_path=args.receipt,
            )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
