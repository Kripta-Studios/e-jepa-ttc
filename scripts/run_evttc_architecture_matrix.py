"""Run the staged, non-cartesian EvTTC architecture selection matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.data.evttc import read_manifest, scan_evttc_root, write_manifest
from e_jepa_ttc.data.evttc_object_cache import (
    EvTTCObjectCacheConfig,
    materialize_evttc_object_cache,
)
from e_jepa_ttc.data.grouped_cv import write_grouped_cv_protocol
from e_jepa_ttc.data.storage_guard import StorageBudget, assert_storage_budget
from e_jepa_ttc.evaluation.oracle_geometry import (
    GTGeometryOracleConfig,
    evaluate_gt_geometry_oracle,
)
from e_jepa_ttc.models.garl_ttc_replica import GarlTTCConfig
from e_jepa_ttc.models.object_geo_jepa_ttc import OGEConfig
from e_jepa_ttc.training.checkpoints import (
    validate_external_eap_checkpoint,
    validate_external_eap_ttc_checkpoint,
    validate_external_ssl_checkpoint,
    validate_external_ttc_checkpoint,
)
from e_jepa_ttc.training.garl_ttc import train_garl_ttc
from e_jepa_ttc.training.object_geo_trainer import OGETrainerConfig, train_object_geo_ttc
from e_jepa_ttc.utils.io import write_structured

VariantFactory = Callable[
    [int],
    tuple[str, GarlTTCConfig | GTGeometryOracleConfig | OGEConfig],
]


def _variants(
    *,
    garl_backbone: str = "resnet50",
    base_encoder_checkpoint: str | None = None,
    allow_random_base_initialization: bool = False,
) -> dict[str, VariantFactory]:
    oge = partial(
        OGEConfig,
        in_channels=21,
        backbone="base_event_tubelet",
        base_encoder_checkpoint=base_encoder_checkpoint,
        allow_random_base_initialization=allow_random_base_initialization,
        dim=192,
        backbone_depth=6,
        heads=6,
    )

    return {
        "G0_RGB_DIRECT": lambda channels: (
            "garl",
            GarlTTCConfig(
                event_channels=channels,
                modality="rgb",
                objective="direct",
                backbone=garl_backbone,
                foreground_supervision=False,
            ),
        ),
        "G1_EVENT_DIRECT": lambda channels: (
            "garl",
            GarlTTCConfig(
                event_channels=channels,
                modality="event",
                objective="direct",
                backbone=garl_backbone,
                foreground_supervision=False,
            ),
        ),
        "G2_RGBE_DIRECT_EARLY": lambda channels: (
            "garl",
            GarlTTCConfig(
                event_channels=channels,
                modality="rgbe",
                fusion="early",
                objective="direct",
                backbone=garl_backbone,
                foreground_supervision=False,
            ),
        ),
        "G3_RGB_LHR": lambda channels: (
            "garl",
            GarlTTCConfig(
                event_channels=channels,
                modality="rgb",
                objective="height_ratio",
                backbone=garl_backbone,
                foreground_supervision=False,
            ),
        ),
        "G4_EVENT_LHR": lambda channels: (
            "garl",
            GarlTTCConfig(
                event_channels=channels,
                modality="event",
                objective="height_ratio",
                backbone=garl_backbone,
                foreground_supervision=False,
            ),
        ),
        "G5_RGBE_LHR_EARLY": lambda channels: (
            "garl",
            GarlTTCConfig(
                event_channels=channels,
                modality="rgbe",
                fusion="early",
                objective="height_ratio",
                backbone=garl_backbone,
                foreground_supervision=False,
            ),
        ),
        "G6_RGBE_LHR_LATE": lambda channels: (
            "garl",
            GarlTTCConfig(
                event_channels=channels,
                modality="rgbe",
                fusion="late",
                objective="height_ratio",
                backbone=garl_backbone,
                foreground_supervision=False,
            ),
        ),
        "G7_RGBE_LHR_LATE_FOREGROUND": lambda channels: (
            "garl",
            GarlTTCConfig(
                event_channels=channels,
                modality="rgbe",
                fusion="late",
                objective="height_ratio",
                backbone=garl_backbone,
                foreground_supervision=True,
            ),
        ),
        "A0_MATCHED_GLOBAL": lambda channels: (
            "oge",
            oge(head_mode="global"),
        ),
        "A1_MATCHED_DENSE_BLOCK": lambda channels: (
            "oge",
            oge(head_mode="dense"),
        ),
        "R1_MATCHED_BBOX_ROI": lambda channels: (
            "oge",
            oge(head_mode="bbox_roi"),
        ),
        "A2_MATCHED_DENSE_ATTNRES": lambda channels: (
            "oge",
            oge(
                head_mode="dense",
                use_attention_residuals=True,
            ),
        ),
        "K1_OBJECT_KDA": lambda channels: (
            "oge",
            oge(
                head_mode="dense",
                use_attention_residuals=True,
                temporal_mixer="object_kda",
            ),
        ),
        "K2_ALIGNED_PATCH_KDA": lambda channels: (
            "oge",
            oge(
                head_mode="dense",
                use_attention_residuals=True,
                temporal_mixer="aligned_patch_kda",
            ),
        ),
        "A3_TARGET_QUERY": lambda channels: (
            "oge",
            oge(
                head_mode="dense",
                use_attention_residuals=True,
                use_target_query=True,
            ),
        ),
        "A4_GT_GEOMETRY": lambda channels: (
            "oracle",
            GTGeometryOracleConfig(evaluate_yaw_derotation=True),
        ),
        "A5_PRED_GEOMETRY": lambda channels: (
            "oge",
            oge(
                head_mode="dense",
                use_attention_residuals=True,
                use_target_query=True,
                geometry_mode="deterministic",
                bbox_source="predicted",
            ),
        ),
        "A6_YAW_DEROTATION_DETERMINISTIC": lambda channels: (
            "oge",
            oge(
                head_mode="dense",
                use_attention_residuals=True,
                use_target_query=True,
                geometry_mode="deterministic",
                bbox_source="predicted",
                use_yaw_derotation=True,
            ),
        ),
        "A7_STABLE_ROUTER": lambda channels: (
            "oge",
            oge(
                head_mode="dense",
                use_attention_residuals=True,
                use_target_query=True,
                geometry_mode="router",
                bbox_source="predicted",
                use_yaw_derotation=True,
            ),
        ),
        "A8_RESIDUAL_UNCERTAINTY": lambda channels: (
            "oge",
            oge(
                head_mode="dense",
                use_attention_residuals=True,
                use_target_query=True,
                use_highres_refiner=True,
                geometry_mode="router",
                bbox_source="predicted",
                use_yaw_derotation=True,
                use_bounded_residual=True,
                use_uncertainty=True,
            ),
        ),
    }


DEFAULT_VARIANTS = (
    "G0_RGB_DIRECT",
    "G1_EVENT_DIRECT",
    "G2_RGBE_DIRECT_EARLY",
    "G3_RGB_LHR",
    "G4_EVENT_LHR",
    "G5_RGBE_LHR_EARLY",
    "G6_RGBE_LHR_LATE",
    "G7_RGBE_LHR_LATE_FOREGROUND",
    "A0_MATCHED_GLOBAL",
    "A1_MATCHED_DENSE_BLOCK",
    "R1_MATCHED_BBOX_ROI",
    "A2_MATCHED_DENSE_ATTNRES",
    "K1_OBJECT_KDA",
    "A4_GT_GEOMETRY",
)
ALL_VARIANTS = tuple(_variants())


def _profile(config_path: Path, mode: str) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {**payload["hardware"], **payload["profiles"][mode]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_manifest(dataset_root: Path, manifest_path: Path) -> None:
    if manifest_path.is_file():
        sequences = read_manifest(manifest_path)
    else:
        sequences = scan_evttc_root(dataset_root)
        write_manifest(sequences, manifest_path)
    if len(sequences) != 32:
        raise ValueError(f"EvTTC-32 protocol requires 32 sequences; found {len(sequences)}.")


def _prepare_cache(
    *,
    manifest_path: Path,
    cv_path: Path,
    cache_dir: Path,
    fold: int,
    profile: dict[str, Any],
    split_protocol: str,
    historical_split_path: Path,
    need_oge: bool,
    need_garl: bool,
    need_rgb: bool,
    include_diagnostic_test: bool,
) -> Path:
    if split_protocol == "historical_base":
        if fold != 0:
            raise ValueError("The historical BASE split has a single fold numbered zero.")
        split_payload = yaml.safe_load(historical_split_path.read_text(encoding="utf-8"))
        split_names = (
            ("train", "validation", "test") if include_diagnostic_test else ("train", "validation")
        )
        assignments = {
            sequence_id: split
            for split in split_names
            for sequence_id in split_payload["splits"][split]
        }
        selection_protocol_sha256 = _sha256(historical_split_path)
    else:
        protocol = write_grouped_cv_protocol(
            manifest_path=manifest_path,
            output_path=cv_path,
            folds=5,
            seed=7,
        )
        fold_payload = protocol["folds"][fold]
        assignments = {
            sequence_id: split
            for split in ("train", "validation")
            for sequence_id in fold_payload[split]
        }
        selection_protocol_sha256 = _sha256(cv_path)
    cache_config = EvTTCObjectCacheConfig(
        history_frames=3,
        # EvTTC labels/RGB are ~20 Hz while the official Garl protocol uses
        # three timestamps at 10 Hz. Select every second label so all matrix
        # arms receive t0, t+0.1 s and t+0.2 s.
        history_stride_frames=2,
        prediction_horizons_ms=(100, 250, 500),
        event_window_ms=100,
        maximum_history_gap_ms=120,
        width=int(profile["cache_resolution"][0]),
        height=int(profile["cache_resolution"][1]),
        event_bins=int(profile["event_bins"]),
        shard_size=256,
        include_rgb=need_rgb,
        include_segmentation_masks=False,
        include_context_events=need_oge,
        include_future_events=False,
        include_garl_pair=need_garl,
    )
    cache_manifest = cache_dir / "manifest.json"
    if cache_manifest.is_file():
        payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
        if payload.get("format") != "evttc_object_event_jepa_cache_v6":
            raise ValueError(
                f"{cache_manifest} is a stale cache. Use a new --cache-dir; "
                "historical artifacts are not overwritten."
            )
        if payload.get("sequence_splits") != assignments:
            raise ValueError("Existing cache does not match this train/validation protocol.")
        if payload.get("selection_protocol_sha256") != selection_protocol_sha256:
            raise ValueError("Existing cache was built from a different split manifest.")
        normalized_config = json.loads(json.dumps(asdict(cache_config)))
        if payload.get("config") != normalized_config:
            raise ValueError("Existing cache does not match the selected feature stage.")
        assert_storage_budget(
            cache_dir,
            budget=StorageBudget(
                maximum_cache_gib=float(profile["maximum_cache_gib"]),
                minimum_free_gib=float(profile["minimum_free_gib"]),
            ),
        )
        return cache_manifest
    materialize_evttc_object_cache(
        manifest_path=manifest_path,
        output_dir=cache_dir,
        sequence_splits=assignments,
        config=cache_config,
        max_windows_per_sequence=profile["max_windows_per_sequence"],
        workers=int(profile["cache_build_workers"]),
        maximum_cache_gib=float(profile["maximum_cache_gib"]),
        minimum_free_gib=float(profile["minimum_free_gib"]),
    )
    payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    payload["selection_protocol"] = split_protocol
    payload["selection_protocol_sha256"] = selection_protocol_sha256
    write_structured(cache_manifest, payload)
    return cache_manifest


def _metric(summary: dict[str, Any], name: str) -> float | None:
    value = summary["validation"].get(name)
    return float(value) if value is not None else None


def _gate_rows(
    summaries: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    comparisons = (
        ("G3_RGB_LHR", "G0_RGB_DIRECT", 0.0, "garl_rgb_height_ratio"),
        ("G4_EVENT_LHR", "G1_EVENT_DIRECT", 0.0, "learned_height_ratio"),
        ("G5_RGBE_LHR_EARLY", "G2_RGBE_DIRECT_EARLY", 0.0, "garl_rgbe_height_ratio"),
        ("G6_RGBE_LHR_LATE", "G5_RGBE_LHR_EARLY", 0.0, "garl_late_fusion"),
        (
            "G7_RGBE_LHR_LATE_FOREGROUND",
            "G6_RGBE_LHR_LATE",
            0.0,
            "foreground_supervision",
        ),
        ("A1_MATCHED_DENSE_BLOCK", "A0_MATCHED_GLOBAL", 0.05, "dense_patch"),
        (
            "A2_MATCHED_DENSE_ATTNRES",
            "A1_MATCHED_DENSE_BLOCK",
            0.03,
            "attention_residuals",
        ),
        ("K1_OBJECT_KDA", "A2_MATCHED_DENSE_ATTNRES", 0.03, "object_kda"),
        ("A5_PRED_GEOMETRY", "A4_GT_GEOMETRY", -0.10, "predicted_mask_geometry"),
        (
            "A6_YAW_DEROTATION_DETERMINISTIC",
            "A5_PRED_GEOMETRY",
            0.0,
            "camera_yaw_derotation",
        ),
        (
            "A7_STABLE_ROUTER",
            "A6_YAW_DEROTATION_DETERMINISTIC",
            0.02,
            "stable_router",
        ),
        ("A8_RESIDUAL_UNCERTAINTY", "A7_STABLE_ROUTER", 0.03, "residual_uncertainty"),
    )
    rows: list[dict[str, Any]] = []
    for candidate, reference, required, component in comparisons:
        if candidate not in summaries or reference not in summaries:
            continue
        candidate_error = float(
            summaries[candidate]["validation"]["sequence_macro_mean_relative_error"]
        )
        reference_error = float(
            summaries[reference]["validation"]["sequence_macro_mean_relative_error"]
        )
        improvement = (reference_error - candidate_error) / max(reference_error, 1e-12)
        passed = improvement >= required
        rows.append(
            {
                "component": component,
                "candidate": candidate,
                "reference": reference,
                "metric": "sequence_macro_mean_relative_error",
                "candidate_error": candidate_error,
                "reference_error": reference_error,
                "relative_improvement": improvement,
                "required_relative_improvement": required,
                "numeric_gate_passed": passed,
                "decision": (
                    "integration_only"
                    if mode == "smoke"
                    else ("shortlist_for_confirmation" if passed else "reject_at_this_stage")
                ),
            }
        )
    if "A3_TARGET_QUERY" in summaries:
        validation = summaries["A3_TARGET_QUERY"]["validation"]
        rows.append(
            {
                "component": "target_query_mask",
                "candidate": "A3_TARGET_QUERY",
                "mask_iou_mean": validation.get("mask_iou_mean"),
                "target_recall_iou_0_1": validation.get("target_recall_iou_0_1"),
                "center_error_fraction_diagonal": validation.get("center_error_fraction_diagonal"),
                "numeric_gate_passed": (
                    float(validation.get("mask_iou_mean", 0.0)) >= 0.75
                    and float(validation.get("target_recall_iou_0_1", 0.0)) >= 0.98
                    and float(validation.get("center_error_fraction_diagonal", 1.0)) <= 0.05
                ),
                "decision": "integration_only" if mode == "smoke" else "apply_numeric_gate",
            }
        )
    if "A4_GT_GEOMETRY" in summaries:
        # The analytic oracle is reported independently.  It has no learned
        # encoder/head and therefore is not a single-factor increment over a
        # neural arm.
        oracle_variants = summaries["A4_GT_GEOMETRY"].get("oracle_variants", {})
        for expert, expert_metrics in oracle_variants.items():
            candidate_error = float(expert_metrics["sequence_macro_mean_relative_error"])
            rows.append(
                {
                    "component": f"gt_geometry_{expert}",
                    "candidate": f"A4_GT_GEOMETRY:{expert}",
                    "metric": "sequence_macro_mean_relative_error",
                    "candidate_error": candidate_error,
                    "numeric_gate_passed": None,
                    "decision": "analytic_oracle_reported_separately",
                }
            )
    return rows


def _matched_fairness_audit(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Verify the invariants required for a single-factor neural comparison."""

    ordered = (
        "A0_MATCHED_GLOBAL",
        "A1_MATCHED_DENSE_BLOCK",
        "A2_MATCHED_DENSE_ATTNRES",
        "K1_OBJECT_KDA",
    )
    selected = {name: summaries[name] for name in ordered if name in summaries}
    if len(selected) < 2:
        return {
            "status": "not_applicable",
            "arms": list(selected),
        }
    invariant_fields = (
        "sample_selection_sha256",
        "initial_backbone_sha256",
        "initial_common_head_sha256",
        "train_samples",
        "validation_samples",
    )
    checks = {
        field: len(
            {json.dumps(summary.get(field), sort_keys=True) for summary in selected.values()}
        )
        == 1
        for field in invariant_fields
    }
    trainer_fields = (
        "epochs",
        "batch_size",
        "gradient_accumulation",
        "learning_rate",
        "weight_decay",
        "precision",
        "early_stopping_patience",
        "early_stopping_min_epochs",
        "early_stopping_min_delta_relative",
        "log_ttc_loss_weight",
        "inverse_nll_loss_weight",
        "risk_loss_weight",
        "backbone_learning_rate_scale",
        "latent_anchor_weight",
        "latent_anchor_ema_momentum",
    )
    trainer_checks = {
        field: len(
            {
                json.dumps(summary["trainer"].get(field), sort_keys=True)
                for summary in selected.values()
            }
        )
        == 1
        for field in trainer_fields
    }
    passed = all(checks.values()) and all(trainer_checks.values())
    audit = {
        "status": "passed" if passed else "failed",
        "arms": list(selected),
        "invariant_checks": checks,
        "trainer_checks": trainer_checks,
        "actual_epochs_may_differ_only_by_shared_early_stopping_rule": True,
        "scientific_scope": (
            "Matched object-cache comparison only; it does not replace the "
            "separate historical BASE reproduction."
        ),
    }
    if not passed:
        raise ValueError(f"Matched comparison invariants failed: {audit}")
    return audit


def _garl_fairness_audit(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Audit equal data and optimizer budgets across the Garl screen.

    Late fusion follows the source protocol and therefore loads the separately
    trained RGB and event branches.  That extra branch pretraining is declared
    explicitly instead of being hidden as equal total training compute.
    """

    selected = {name: summary for name, summary in summaries.items() if name.startswith("G")}
    if len(selected) < 2:
        return {"status": "not_applicable", "arms": list(selected)}
    invariant_fields = (
        "sample_selection_sha256",
        "train_samples",
        "validation_samples",
        "effective_batch_size",
        "optimizer_steps_per_epoch",
        "maximum_optimizer_steps",
    )
    checks = {
        field: len(
            {json.dumps(summary.get(field), sort_keys=True) for summary in selected.values()}
        )
        == 1
        for field in invariant_fields
    }
    trainer_fields = (
        "epochs",
        "learning_rate",
        "weight_decay",
        "precision",
        "early_stopping_patience",
        "early_stopping_min_epochs",
        "early_stopping_min_delta_relative",
    )
    trainer_checks = {
        field: len(
            {
                json.dumps(summary["trainer"].get(field), sort_keys=True)
                for summary in selected.values()
            }
        )
        == 1
        for field in trainer_fields
    }
    passed = all(checks.values()) and all(trainer_checks.values())
    audit = {
        "status": "passed_with_declared_source_branch_pretraining" if passed else "failed",
        "arms": list(selected),
        "invariant_checks": checks,
        "trainer_checks": trainer_checks,
        "actual_epochs_may_differ_only_by_shared_early_stopping_rule": True,
        "late_fusion_branch_pretraining": {
            name: bool(summary.get("branch_initialization")) for name, summary in selected.items()
        },
        "scientific_scope": (
            "Equal downstream sample/effective-batch/update budgets. G6/G7 "
            "add source-required G3/G4 branch pretraining, so total training "
            "compute is reported but not claimed equal."
        ),
    }
    if not passed:
        raise ValueError(f"Garl comparison invariants failed: {audit}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "screen", "confirm"), default="screen")
    parser.add_argument(
        "--stage-role",
        choices=("core", "garl", "all"),
        help=(
            "Isolate matrix summaries by scientific stage. The value must agree "
            "with the selected variants."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/evttc"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/evttc32_local.yaml"),
    )
    parser.add_argument(
        "--cv-protocol",
        type=Path,
        default=Path("data/splits/evttc32_grouped_cv.yaml"),
    )
    parser.add_argument(
        "--split-protocol",
        choices=("historical_base", "grouped_cv"),
        default="historical_base",
    )
    parser.add_argument(
        "--historical-split",
        type=Path,
        default=Path("data/splits/evttc_all32_article_family_holdout.yaml"),
    )
    parser.add_argument("--base-encoder-checkpoint", type=Path)
    parser.add_argument(
        "--base-initialization",
        choices=(
            "audited_ssl",
            "random_control",
            "external_ssl",
            "external_ttc",
            "external_eap_ssl",
            "external_eap_geo",
            "external_eap_ttc",
        ),
        default="audited_ssl",
        help=(
            "Use the audited SSL checkpoint or a disclosed random EventTubelet "
            "control. random_control is intended for leakage-free grouped CV when "
            "fold-specific SSL checkpoints do not yet exist. external_ssl accepts "
            "a label-free CARLA checkpoint; external_ttc is the separately disclosed "
            "CARLA synthetic-TTC ablation. external_eap_ssl, external_eap_geo, and "
            "external_eap_ttc accept the paired public eAP train-only arms. None may "
            "have EvTTC exposure."
        ),
    )
    parser.add_argument(
        "--external-pretraining-split",
        type=Path,
        default=Path("data/splits/carla_dvs_looming_blocked_v1.json"),
        help="Signed source split required by every external initialization.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument(
        "--backbone-learning-rate-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply the OGE backbone learning rate while keeping downstream "
            "mixers and TTC heads at the configured base learning rate."
        ),
    )
    parser.add_argument(
        "--latent-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only EMA latent-anchor weight. Zero preserves the "
            "published f4c87df behavior."
        ),
    )
    parser.add_argument(
        "--latent-anchor-ema-momentum",
        type=float,
        default=0.996,
    )
    parser.add_argument(
        "--include-diagnostic-test-cache",
        action="store_true",
        help=(
            "Materialize the labelled family-holdout test sequences in a historical-split "
            "cache. Training still opens train/validation only."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sota/evttc_architecture_selection.yaml"),
    )
    args = parser.parse_args()
    if not 0 <= args.fold < 5:
        raise ValueError("fold must be in [0,4].")
    assert_no_sealed_benchmark_paths(
        (
            args.dataset_root,
            args.manifest,
            args.cv_protocol,
            args.historical_split,
            args.config,
            args.external_pretraining_split,
        )
    )
    if args.base_initialization == "random_control":
        if args.base_encoder_checkpoint is not None:
            raise ValueError(
                "--base-encoder-checkpoint conflicts with random_control initialization."
            )
        base_encoder_checkpoint = None
    elif args.base_initialization in {
        "external_ssl",
        "external_ttc",
        "external_eap_ssl",
        "external_eap_geo",
        "external_eap_ttc",
    }:
        if args.base_encoder_checkpoint is None:
            raise ValueError(f"{args.base_initialization} requires --base-encoder-checkpoint.")
        base_encoder_checkpoint = args.base_encoder_checkpoint
    else:
        base_encoder_checkpoint = args.base_encoder_checkpoint or Path(
            "artifacts/runs/evttc32_article_ablation/"
            f"base/seed{args.seed}/ssl30/jepa_encoder_best.pt"
        )
    profile = _profile(args.config, args.mode)
    if args.workers is not None:
        profile["dataloader_workers"] = args.workers
        profile["cache_build_workers"] = args.workers
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive.")
        profile["batch_size"] = args.batch_size
    if args.gradient_accumulation is not None:
        if args.gradient_accumulation <= 0:
            raise ValueError("gradient-accumulation must be positive.")
        profile["gradient_accumulation"] = args.gradient_accumulation
    if args.include_diagnostic_test_cache and args.split_protocol != "historical_base":
        raise ValueError("--include-diagnostic-test-cache is only valid with historical_base.")
    selected_variants = tuple(args.variants or DEFAULT_VARIANTS)
    oge_selected = any(not variant.startswith("G") for variant in selected_variants)
    needs_base_encoder = any(
        not variant.startswith("G") and variant != "A4_GT_GEOMETRY" for variant in selected_variants
    )
    garl_selected = any(variant.startswith("G") for variant in selected_variants)
    rgb_garl_variants = {
        "G0_RGB_DIRECT",
        "G2_RGBE_DIRECT_EARLY",
        "G3_RGB_LHR",
        "G5_RGBE_LHR_EARLY",
        "G6_RGBE_LHR_LATE",
        "G7_RGBE_LHR_LATE_FOREGROUND",
    }
    rgb_selected = bool(set(selected_variants) & rgb_garl_variants)
    inferred_stage_role = (
        "all" if oge_selected and garl_selected else ("core" if oge_selected else "garl")
    )
    stage_role = args.stage_role or inferred_stage_role
    if stage_role != inferred_stage_role:
        raise ValueError(
            f"stage-role={stage_role!r} conflicts with selected variants; "
            f"expected {inferred_stage_role!r}."
        )
    cache_dir = args.cache_dir or Path(
        "artifacts/features/"
        f"evttc32_paper_v6_{args.split_protocol}_{args.mode}_{stage_role}_fold{args.fold}"
    )
    output_dir = args.output_dir or Path(
        "artifacts/runs/"
        f"evttc32_architecture_v4_{args.split_protocol}_{args.mode}/"
        f"{stage_role}/fold-{args.fold}"
    )
    assert_no_sealed_benchmark_paths((cache_dir, output_dir))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "stage_role": stage_role,
                    "split_protocol": args.split_protocol,
                    "fold": args.fold,
                    "variants": selected_variants,
                    "profile": profile,
                    "cache_dir": cache_dir.as_posix(),
                    "output_dir": output_dir.as_posix(),
                    "base_initialization": args.base_initialization,
                    "base_encoder_checkpoint": (
                        base_encoder_checkpoint.as_posix()
                        if base_encoder_checkpoint is not None
                        else None
                    ),
                    "include_diagnostic_test_cache": args.include_diagnostic_test_cache,
                    "benchmark10_opened": False,
                },
                indent=2,
            )
        )
        return 0
    _prepare_manifest(args.dataset_root, args.manifest)
    cache_manifest = _prepare_cache(
        manifest_path=args.manifest,
        cv_path=args.cv_protocol,
        cache_dir=cache_dir,
        fold=args.fold,
        profile=profile,
        split_protocol=args.split_protocol,
        historical_split_path=args.historical_split,
        need_oge=oge_selected,
        need_garl=garl_selected,
        need_rgb=rgb_selected,
        include_diagnostic_test=args.include_diagnostic_test_cache,
    )
    external_pretraining_audit: dict[str, Any] = {
        "enabled": False,
        "pretraining_regime": None,
    }
    if needs_base_encoder and base_encoder_checkpoint is not None:
        if not base_encoder_checkpoint.is_file():
            raise FileNotFoundError(
                f"Audited BASE encoder checkpoint is missing: {base_encoder_checkpoint}"
            )
        checkpoint = torch.load(
            base_encoder_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if args.base_initialization == "external_ssl":
            external_pretraining_audit = {
                "enabled": True,
                **validate_external_ssl_checkpoint(
                    base_encoder_checkpoint,
                    checkpoint,
                    source_split_path=args.external_pretraining_split,
                ),
            }
        elif args.base_initialization == "external_ttc":
            external_pretraining_audit = {
                "enabled": True,
                **validate_external_ttc_checkpoint(
                    base_encoder_checkpoint,
                    checkpoint,
                    source_split_path=args.external_pretraining_split,
                ),
            }
        elif args.base_initialization in {"external_eap_ssl", "external_eap_geo"}:
            expected_regime = (
                "eap_ssl" if args.base_initialization == "external_eap_ssl" else "eap_geo"
            )
            external_pretraining_audit = {
                "enabled": True,
                **validate_external_eap_checkpoint(
                    base_encoder_checkpoint,
                    checkpoint,
                    source_split_path=args.external_pretraining_split,
                    expected_regime=expected_regime,
                ),
            }
        elif args.base_initialization == "external_eap_ttc":
            external_pretraining_audit = {
                "enabled": True,
                **validate_external_eap_ttc_checkpoint(
                    base_encoder_checkpoint,
                    checkpoint,
                    source_split_path=args.external_pretraining_split,
                ),
            }
        else:
            expected_split_hash = (
                _sha256(args.historical_split)
                if args.split_protocol == "historical_base"
                else _sha256(args.cv_protocol)
            )
            if checkpoint.get("split_manifest_sha256") != expected_split_hash:
                raise ValueError(
                    "BASE initialization was pretrained on a different split. "
                    "Use historical_base for the audited checkpoint or provide a "
                    "fold-specific grouped-CV SSL checkpoint."
                )
    probe = EAPObjectCacheDataset(cache_manifest, splits=("train",))
    sample = probe[0]
    garl_channels = None
    if garl_selected:
        garl_events = sample.get("garl_event_roi")
        if not isinstance(garl_events, torch.Tensor):
            raise TypeError("Garl cache must contain a tensor garl_event_roi.")
        garl_channels = int(garl_events.shape[0])
    if oge_selected and not isinstance(sample.get("context_events"), torch.Tensor):
        raise TypeError("OGE cache must contain tensor context_events.")
    probe.close()
    trainer = OGETrainerConfig(
        epochs=int(profile["epochs"]),
        batch_size=int(profile["batch_size"]),
        gradient_accumulation=int(profile["gradient_accumulation"]),
        learning_rate=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
        precision=str(profile["precision"]),
        num_workers=int(profile["dataloader_workers"]),
        early_stopping_patience=int(profile["early_stopping_patience"]),
        early_stopping_min_epochs=int(profile["early_stopping_min_epochs"]),
        max_train_samples=profile["max_train_samples"],
        max_validation_samples=profile["max_validation_samples"],
        backbone_learning_rate_scale=args.backbone_learning_rate_scale,
        latent_anchor_weight=args.latent_anchor_weight,
        latent_anchor_ema_momentum=args.latent_anchor_ema_momentum,
        seed=args.seed,
    )
    summaries: dict[str, dict[str, Any]] = {}
    factories = _variants(
        garl_backbone=str(profile["garl_backbone"]),
        base_encoder_checkpoint=(
            base_encoder_checkpoint.as_posix() if base_encoder_checkpoint is not None else None
        ),
        allow_random_base_initialization=args.base_initialization == "random_control",
    )
    for variant in selected_variants:
        kind, model_config = factories[variant](int(garl_channels or 0))
        active_trainer = (
            replace(
                trainer,
                batch_size=int(
                    profile["garl_foreground_batch_size"]
                    if (
                        isinstance(model_config, GarlTTCConfig)
                        and model_config.foreground_supervision
                        and model_config.backbone == "resnet50"
                    )
                    else profile["garl_batch_size"]
                ),
                gradient_accumulation=int(
                    profile["garl_foreground_gradient_accumulation"]
                    if (
                        isinstance(model_config, GarlTTCConfig)
                        and model_config.foreground_supervision
                        and model_config.backbone == "resnet50"
                    )
                    else profile["garl_gradient_accumulation"]
                ),
                learning_rate=1e-3,
                weight_decay=0.0,
            )
            if kind == "garl"
            else trainer
        )
        run_dir = output_dir / variant / f"seed-{args.seed}"
        summary_path = run_dir / "summary.json"
        garl_branch_kwargs: dict[str, Path] = {}
        if variant in {"G6_RGBE_LHR_LATE", "G7_RGBE_LHR_LATE_FOREGROUND"}:
            garl_branch_kwargs = {
                "rgb_branch_checkpoint": (
                    output_dir / "G3_RGB_LHR" / f"seed-{args.seed}" / "best.pt"
                ),
                "event_branch_checkpoint": (
                    output_dir / "G4_EVENT_LHR" / f"seed-{args.seed}" / "best.pt"
                ),
            }
        if kind == "garl":
            assert isinstance(model_config, GarlTTCConfig)
            expected_fingerprint = train_garl_ttc(
                cache_manifest_path=cache_manifest,
                output_dir=run_dir,
                model_config=model_config,
                trainer_config=active_trainer,
                device_name=args.device,
                **garl_branch_kwargs,
                dry_run_fingerprint=True,
            )
        elif kind == "oge":
            assert isinstance(model_config, OGEConfig)
            expected_fingerprint = train_object_geo_ttc(
                cache_manifest_path=cache_manifest,
                output_dir=run_dir,
                model_config=model_config,
                trainer_config=active_trainer,
                device_name=args.device,
                dry_run_fingerprint=True,
            )
        else:
            assert isinstance(model_config, GTGeometryOracleConfig)
            expected_fingerprint = evaluate_gt_geometry_oracle(
                cache_manifest_path=cache_manifest,
                output_dir=run_dir,
                config=model_config,
                batch_size=active_trainer.batch_size,
                num_workers=active_trainer.num_workers,
                max_train_samples=active_trainer.max_train_samples,
                max_validation_samples=active_trainer.max_validation_samples,
                device_name=args.device,
                dry_run_fingerprint=True,
            )
        assert isinstance(expected_fingerprint, str)
        if args.resume and summary_path.is_file():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if previous.get("run_fingerprint") != expected_fingerprint:
                raise ValueError(
                    f"Completed run {run_dir} has a stale fingerprint. "
                    "Keep it as evidence and select a new --output-dir."
                )
            summaries[variant] = previous
            continue
        if kind == "garl":
            assert isinstance(model_config, GarlTTCConfig)
            result = train_garl_ttc(
                cache_manifest_path=cache_manifest,
                output_dir=run_dir,
                model_config=model_config,
                trainer_config=active_trainer,
                device_name=args.device,
                **garl_branch_kwargs,
                resume=args.resume and (run_dir / "resume.pt").is_file(),
            )
        elif kind == "oge":
            assert isinstance(model_config, OGEConfig)
            result = train_object_geo_ttc(
                cache_manifest_path=cache_manifest,
                output_dir=run_dir,
                model_config=model_config,
                trainer_config=active_trainer,
                device_name=args.device,
                resume=args.resume and (run_dir / "resume.pt").is_file(),
            )
        else:
            assert isinstance(model_config, GTGeometryOracleConfig)
            result = evaluate_gt_geometry_oracle(
                cache_manifest_path=cache_manifest,
                output_dir=run_dir,
                config=model_config,
                batch_size=active_trainer.batch_size,
                num_workers=active_trainer.num_workers,
                max_train_samples=active_trainer.max_train_samples,
                max_validation_samples=active_trainer.max_validation_samples,
                device_name=args.device,
            )
        assert isinstance(result, dict)
        summaries[variant] = result
        write_structured(
            output_dir / "matrix_progress.json",
            {
                "mode": args.mode,
                "fold": args.fold,
                "seed": args.seed,
                "completed": list(summaries),
                "benchmark10_opened": False,
            },
        )
    ranking = sorted(
        (
            {
                "variant": variant,
                "sequence_macro_selection_score": _metric(
                    summary,
                    "sequence_macro_selection_score",
                ),
                "sequence_macro_mean_relative_error": _metric(
                    summary,
                    "sequence_macro_mean_relative_error",
                ),
                "sequence_macro_mae_s": _metric(summary, "sequence_macro_mae_s"),
                "worst_sequence_mae_s": _metric(summary, "worst_sequence_mae_s"),
                "mask_iou_mean": _metric(summary, "mask_iou_mean"),
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "milliseconds_per_window": summary["validation"].get("milliseconds_per_window"),
                "peak_vram_gib": (
                    float(summary.get("peak_vram_bytes", 0)) / 1024**3
                    if summary.get("peak_vram_bytes") is not None
                    else None
                ),
            }
            for variant, summary in summaries.items()
        ),
        key=lambda row: float(row["sequence_macro_selection_score"] or float("inf")),
    )
    matrix = {
        "protocol": "evttc_architecture_selection_v4_stage_isolated",
        "mode": args.mode,
        "stage_role": stage_role,
        "split_protocol": args.split_protocol,
        "fold": args.fold,
        "seed": args.seed,
        "cache_manifest": cache_manifest.as_posix(),
        "trainer": asdict(trainer),
        "variants": list(selected_variants),
        "ranking": ranking,
        "gates": _gate_rows(summaries, mode=args.mode),
        "matched_fairness_audit": _matched_fairness_audit(summaries),
        "garl_fairness_audit": _garl_fairness_audit(summaries),
        "benchmark10_opened": False,
        "base_initialization": args.base_initialization,
        "selection_split": (
            "historical_validation_sequences"
            if args.split_protocol == "historical_base"
            else "validation_grouped_fold"
        ),
        "external_pretraining": external_pretraining_audit,
    }
    write_structured(output_dir / "matrix_summary.json", matrix)
    print(json.dumps(matrix, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
