from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from e_jepa_ttc.detection.quality_ranking import RankingRecipe, fit_quality_ranker

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _script_module(filename: str):
    path = ROOT / f"scripts/{filename}.py"
    spec = importlib.util.spec_from_file_location(f"{filename}_integration", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predictor_artifacts(
    root: Path, *, token: str, checkpoint_sha256: str
) -> tuple[Path, Path]:
    score = 0.8
    prediction = {
        "class_name": "car",
        "translation_m": [0.0, 0.0, 10.0],
        "size_lwh_m": [4.0, 1.8, 1.6],
        "yaw_rad": 0.0,
        "velocity_mps": [0.0, 0.0],
        "score": score,
    }
    predictions = root / "predictions.json"
    _write(
        predictions,
        {
            "schema": "eap_detection_campaign_inference_v1",
            "checkpoint_sha256": checkpoint_sha256,
            "targets_argument": None,
            "annotations_returned": 0,
            "sample_count": 1,
            "predictions": {token: [prediction]},
        },
    )
    logits = [-8.0] * 8
    logits[0] = math.log(score / (1 - score))
    sidecar = root / "sidecar.json"
    _write(
        sidecar,
        {
            "schema": "monodgp_external_diagnostic_sidecar_v1",
            "checkpoint_sha256": checkpoint_sha256,
            "prediction_output_sha256": _sha256(predictions),
            "frames": [
                {
                    "sample_token": token,
                    "tensors": {
                        "pred_logits": {"values": [logits]},
                        "pred_depth": {"values": [[10.0, 0.0]]},
                    },
                }
            ],
            "claims": {
                "changes_predictions": False,
                "ground_truth_access": False,
                "internal_unexposed_values_reconstructed": False,
            },
        },
    )
    return predictions, sidecar


@pytest.mark.parametrize("selection_panel", ["selection1024", "selection_full"])
def test_tiny_label_free_panels_fit_train_apply_selection_and_reject_selection_fit(
    tmp_path: Path,
    selection_panel: str,
) -> None:
    revision = "a" * 40
    train_token, selection_token = "train-sequence:frame", "selection-sequence:frame"
    data_root = tmp_path / "dataset"
    media = data_root / "data/train.parquet"
    media.parent.mkdir(parents=True)
    intrinsic = [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    label_paths = {
        train_token: "data/train/train-sequence/labels.parquet",
        selection_token: "data/train/selection-sequence/labels.parquet",
    }
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_token": token,
                    "sequence_id": token.split(":", 1)[0],
                    "split": "train",
                    "labels_path": label_paths[token],
                    "K_event": intrinsic,
                    "T_event_ego": transform,
                }
                for token in (train_token, selection_token)
            ]
        ),
        media,
    )
    for token, relative in label_paths.items():
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "sample_token": token,
                        "split": "train",
                        "category": "car",
                        "bbox_3d_ego": [0.0, 0.0, 10.0, 4.0, 1.6, 1.8, 0.0],
                    }
                ]
            ),
            path,
        )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen R0 fixture")
    checkpoint_sha256 = _sha256(checkpoint)
    split = tmp_path / "split.json"
    _write(
        split,
        {
            "schema": "eap_detection_sequence_split_v1",
            "dataset_revision": revision,
            "media_sha256": _sha256(media),
            "assignments": {
                "train": ["train-sequence"],
                "selection": ["selection-sequence"],
                "calibration": ["calibration-sequence"],
            },
        },
    )
    public_files = tmp_path / "public_files.json"
    _write(
        public_files,
        {
            "schema_version": "eap_detection_public_files_v1",
            "revision": revision,
            "files": {
                "data/train.parquet": {
                    "sha256": _sha256(media),
                    "size_bytes": media.stat().st_size,
                },
                **{
                    relative: {
                        "sha256": _sha256(data_root / relative),
                        "size_bytes": (data_root / relative).stat().st_size,
                    }
                    for relative in label_paths.values()
                },
            },
        },
    )
    population_audit = tmp_path / "population_audit.json"
    population_audit_value = {
        "schema": "eap_detection_population_audit_v2",
        "image_geometry": {
            "source_wh": [1280, 720],
            "campaign_wh": [640, 384],
            "sampled_one_frame_per_sequence": [
                {
                    "sequence_id": sequence,
                    "sample_token": token,
                    "image_wh": [1280, 720],
                    "member_sha256": character * 64,
                }
                for sequence, token, character in (
                    ("train-sequence", train_token, "1"),
                    ("selection-sequence", selection_token, "2"),
                )
            ],
        },
    }
    _write(population_audit, population_audit_value)
    campaign = tmp_path / "campaign.json"
    _write(
        campaign,
        {
            "schema": "eap_detection_campaign_manifest_v1",
            "dataset_revision": revision,
            "source_media_sha256": _sha256(media),
            "candidate_split_sha256": _sha256(split),
            "population_audit_sha256": _canonical(population_audit_value),
            "roles": {
                "train": {
                    "sample_tokens": [train_token],
                    "sequence_ids": ["train-sequence"],
                },
                "selection": {
                    "sample_tokens": [selection_token],
                    "sequence_ids": ["selection-sequence"],
                },
            },
        },
    )
    panel_dir = tmp_path / "panels"
    panel_module = _script_module("prepare_detection_ranking_panels")
    panel_module.create_recipe(
        campaign_manifest_path=campaign,
        population_audit_path=population_audit,
        split_path=split,
        public_files_manifest_path=public_files,
        checkpoint_path=checkpoint,
        data_root=data_root,
        output_dir=panel_dir,
        train_count=1,
        selection_count=1,
        selection_panel_name=selection_panel,
    )
    recipe = json.loads((panel_dir / "ranking_panel_recipe.json").read_text(encoding="utf-8"))
    assert recipe["panels"]["train4096"]["token_count"] == 1
    assert recipe["separation"]["test_labels_read"] is False

    train_predictions, train_sidecar = _predictor_artifacts(
        tmp_path / "train", token=train_token, checkpoint_sha256=checkpoint_sha256
    )
    selection_predictions, selection_sidecar = _predictor_artifacts(
        tmp_path / "selection", token=selection_token, checkpoint_sha256=checkpoint_sha256
    )
    train_gt, selection_gt = tmp_path / "train_gt.json", tmp_path / "selection_gt.json"
    train_receipt = tmp_path / "train_gt_receipt.json"
    selection_receipt = tmp_path / "selection_gt_receipt.json"
    panel_module.build_ground_truth_bundle(
        recipe_path=panel_dir / "ranking_panel_recipe.json",
        panel_name="train4096",
        predictions_path=train_predictions,
        sidecar_path=train_sidecar,
        data_root=data_root,
        split_path=split,
        public_files_manifest_path=public_files,
        output_path=train_gt,
        receipt_path=train_receipt,
    )
    panel_module.build_ground_truth_bundle(
        recipe_path=panel_dir / "ranking_panel_recipe.json",
        panel_name=selection_panel,
        predictions_path=selection_predictions,
        sidecar_path=selection_sidecar,
        data_root=data_root,
        split_path=split,
        public_files_manifest_path=public_files,
        output_path=selection_gt,
        receipt_path=selection_receipt,
    )
    duplicate_sidecar = tmp_path / "duplicate_sidecar.json"
    duplicate_value = json.loads(train_sidecar.read_text(encoding="utf-8"))
    duplicate_value["frames"].append(dict(duplicate_value["frames"][0]))
    _write(duplicate_sidecar, duplicate_value)
    with pytest.raises(ValueError, match="token population differs"):
        panel_module.build_ground_truth_bundle(
            recipe_path=panel_dir / "ranking_panel_recipe.json",
            panel_name="train4096",
            predictions_path=train_predictions,
            sidecar_path=duplicate_sidecar,
            data_root=data_root,
            split_path=split,
            public_files_manifest_path=public_files,
            output_path=tmp_path / "duplicate_gt.json",
            receipt_path=tmp_path / "duplicate_receipt.json",
        )
    bundle = tmp_path / "ranker.json"
    module = _script_module("train_detection_ranker")
    dynamic_module = cast(Any, module)
    dynamic_module.fit_quality_ranker = lambda rows, provenance: fit_quality_ranker(
        rows, provenance=provenance, recipe=RankingRecipe(epochs=1)
    )
    tampered_receipt = tmp_path / "tampered_receipt.json"
    tampered_value = json.loads(train_receipt.read_text(encoding="utf-8"))
    tampered_value["label_free_predictions_verified_before_labels"] = False
    _write(tampered_receipt, tampered_value)
    with pytest.raises(ValueError, match="label-free separation"):
        dynamic_module.fit(
            Namespace(
                predictions=train_predictions,
                sidecar=train_sidecar,
                ground_truth=train_gt,
                panel_recipe=panel_dir / "ranking_panel_recipe.json",
                ground_truth_receipt=tampered_receipt,
                split_manifest=split,
                dataset_role="train",
                output=tmp_path / "bypass.json",
            )
        )
    assert not (tmp_path / "bypass.json").exists()
    dynamic_module.fit(
        Namespace(
            predictions=train_predictions,
            sidecar=train_sidecar,
            ground_truth=train_gt,
            panel_recipe=panel_dir / "ranking_panel_recipe.json",
            ground_truth_receipt=train_receipt,
            split_manifest=split,
            dataset_role="train",
            output=bundle,
        )
    )
    ranked_selection = tmp_path / "ranked_selection.json"
    dynamic_module.apply(
        Namespace(
            predictions=selection_predictions,
            sidecar=selection_sidecar,
            bundle=bundle,
            output=ranked_selection,
        )
    )
    ranked = json.loads(ranked_selection.read_text(encoding="utf-8"))
    assert ranked["label_access_during_apply"] is False
    assert ranked["predictions"][selection_token][0]["translation_m"] == [0.0, 0.0, 10.0]

    with pytest.raises(ValueError, match="train4096 GT receipt"):
        dynamic_module.fit(
            Namespace(
                predictions=selection_predictions,
                sidecar=selection_sidecar,
                ground_truth=selection_gt,
                panel_recipe=panel_dir / "ranking_panel_recipe.json",
                ground_truth_receipt=selection_receipt,
                split_manifest=split,
                dataset_role="train",
                output=tmp_path / "forbidden.json",
            )
        )
    assert not (tmp_path / "forbidden.json").exists()
