"""Run the bounded, label-free Dense Level--Dynamics Tubelet JEPA runner.

Selection is intentionally out of scope here: the runner accepts only a signed
matched manifest and an eAP root containing raw event HDF5 files.  ``--dry-run``
validates all signatures/config/resource contracts without opening HDF5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.eap_highres_jepa import (  # noqa: E402
    EAPHighResLabelFreeDataset,
    make_label_free_loader,
)
from e_jepa_ttc.data.matched_eap_subset import validate_matched_manifest  # noqa: E402
from e_jepa_ttc.losses.level_dynamics_jepa import LevelDynamicsLossConfig  # noqa: E402
from e_jepa_ttc.models.dense_level_dynamics_jepa import DenseLevelDynamicsConfig  # noqa: E402
from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHRConfig  # noqa: E402
from e_jepa_ttc.training.eap_highres_jepa import (  # noqa: E402
    EAPHighResJEPATrainer,
    EAPHighResJEPATrainerConfig,
    LabelFreeManifestProvenance,
    load_signed_label_free_manifest,
)


def _bounded_positive(value: str, *, name: str, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer.") from exc
    if not 1 <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{name} must lie in [1,{maximum}].")
    return parsed


def _load_label_free_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dense Level-Dynamics config is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dense Level-Dynamics config must contain a mapping.")
    prohibited_fragments = ("ttc", "3d", "bbox", "box", "category", "mask", "rgb", "evttc")
    prohibited_truthy: dict[str, Any] = {}

    def visit(item: object, path_parts: tuple[str, ...]) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key)
                child_path = (*path_parts, key_text)
                lowered_key = key_text.lower()
                depth_label_key = (
                    lowered_key in {"depth", "depth_label", "depth_labels"}
                    or "depth_or_3d" in lowered_key
                )
                if (
                    (
                        depth_label_key
                        or any(fragment in lowered_key for fragment in prohibited_fragments)
                    )
                    and child is not False
                    and lowered_key != "signed_ttc_convention"
                ):
                    prohibited_truthy[".".join(child_path)] = child
                visit(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, (*path_parts, f"[{index}]"))

    visit(value, ())
    if prohibited_truthy:
        raise ValueError(
            "Dense Level-Dynamics config enables prohibited SSL-Pure fields: "
            + ", ".join(sorted(prohibited_truthy))
        )
    return dict(value)


def _approved_trainer_config(value: dict[str, Any], *, seed: int) -> EAPHighResJEPATrainerConfig:
    """Parse only the four explicit config sections; reject silent field drops."""

    expected = {"artifact_type", "architecture", "data", "training", "objective"}
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(
            "Approved Dense Level-Dynamics config schema mismatch: "
            f"missing={missing}, extra={unknown}"
        )
    if value.get("artifact_type") != "dense_level_dynamics_jepa_train_config_v1":
        raise ValueError("Config artifact_type must be dense_level_dynamics_jepa_train_config_v1.")
    architecture = value["architecture"]
    data = value["data"]
    training = value["training"]
    objective = value["objective"]
    if not all(isinstance(section, dict) for section in (architecture, data, training, objective)):
        raise ValueError("Approved Dense Level-Dynamics sections must be mappings.")
    expected_arch = {
        "encoder",
        "projection_dim",
        "predictor_dim",
        "predictor_layers",
        "predictor_heads",
        "predictor_mlp_ratio",
        "patch_query_chunk_size",
        "max_batch_size",
        "max_temporal_steps",
        "max_patches",
        "max_horizons",
        "ema_start",
        "ema_end",
        "ema_total_updates",
    }
    if set(architecture) != expected_arch:
        raise ValueError("Architecture keys must exactly match the Dense Level-Dynamics contract.")
    encoder_values = architecture["encoder"]
    if not isinstance(encoder_values, dict):
        raise ValueError("architecture.encoder must be a mapping.")
    encoder_expected = {
        "in_channels",
        "embed_dim",
        "patch_size",
        "spatial_window",
        "heads",
        "spatial_depth",
        "temporal_depth",
        "temporal_mixer",
        "merge_2x2",
        "global_attention",
    }
    if set(encoder_values) != encoder_expected:
        raise ValueError("architecture.encoder keys do not match the frozen backbone contract.")
    frozen_encoder = {
        "in_channels": 21,
        "embed_dim": 192,
        "patch_size": 16,
        "spatial_window": 8,
        "heads": 6,
        "spatial_depth": 1,
        "temporal_depth": 2,
        "temporal_mixer": "block_causal",
        "merge_2x2": False,
        "global_attention": False,
    }
    if {key: encoder_values[key] for key in encoder_expected} != frozen_encoder:
        raise ValueError("architecture.encoder must match the frozen LHR-small backbone exactly.")
    expected_data = {
        "stage",
        "modality",
        "input_policy",
        "sampler_policy",
        "calibration_mode",
        "signed_ttc_convention",
        "horizons_s",
        "horizon_tolerance_s",
        "exclusion_window_s",
        "width",
        "height",
        "temporal_steps",
        "bins",
    }
    expected_training = {
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "total_updates",
        "batch_size",
        "workers",
        "precision",
        "seeds",
    }
    expected_objective = {
        "name",
        "level_weight",
        "temporal_residual_weight",
        "dynamics_nce_weight",
        "residual_visreg_weight",
        "nce_temperature",
        "nce_exclusion_window_s",
        "nce_min_negatives",
        "visreg_projections",
        "visreg_temperature",
    }
    if (
        set(data) != expected_data
        or set(training) != expected_training
        or set(objective) != expected_objective
    ):
        raise ValueError("Data/training/objective keys do not match the frozen config contract.")
    if (
        int(data["width"]) != 320
        or int(data["height"]) != 192
        or int(data["temporal_steps"]) != 5
        or int(data["bins"]) != 5
    ):
        raise ValueError("Dense Level-Dynamics input policy must be 320x192, T=5, bins=5.")
    declared_seeds = tuple(int(item) for item in training["seeds"])
    if declared_seeds != (7, 13, 23):
        raise ValueError("The governing confirmation seed set is exactly [7,13,23].")
    if seed not in declared_seeds:
        raise ValueError(f"Requested seed {seed} is not in the declared seed set {declared_seeds}.")
    if str(training["precision"]) not in {"fp32", "bf16", "fp16"}:
        raise ValueError("The pilot precision must be exactly fp32, bf16 or fp16.")
    arch_encoder = EJEPATubeletLHRConfig(
        **{str(key): encoder_values[key] for key in encoder_expected}
    )
    model = DenseLevelDynamicsConfig(
        encoder=arch_encoder,
        **{str(key): architecture[key] for key in expected_arch if key != "encoder"},
    )
    loss = LevelDynamicsLossConfig(
        objective=str(objective["name"]),
        **{str(key): objective[key] for key in expected_objective if key != "name"},
    )
    return EAPHighResJEPATrainerConfig(
        model=model,
        loss=loss,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        total_updates=int(training["total_updates"]),
        seed=int(seed),
        precision=str(training["precision"]),
    )


def _config_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_shared_objective_configs(config_path: Path, current: Mapping[str, Any]) -> None:
    """Require the four objective YAMLs to share every non-objective section."""

    shared = {key: value for key, value in current.items() if key != "objective"}
    expected_hash = _config_hash(shared)
    siblings = sorted(config_path.parent.glob("dense_level_dynamics_*.yaml"))
    if len(siblings) < 4:
        raise ValueError("The four dense_level_dynamics objective configs are required.")
    for sibling in siblings:
        loaded = _load_label_free_config(sibling)
        candidate = {key: value for key, value in loaded.items() if key != "objective"}
        if _config_hash(candidate) != expected_hash:
            raise ValueError(
                "Dense Level-Dynamics objective configs differ outside objective weights/name: "
                f"{sibling.name}"
            )


def _validate_manifest_config_compatibility(
    manifest: Mapping[str, Any], config_value: Mapping[str, Any]
) -> None:
    """Fail closed when the signed manifest and YAML disagree on frozen inputs."""

    manifest_config = manifest.get("config")
    freeze = manifest.get("freeze")
    data = config_value.get("data")
    architecture = config_value.get("architecture")
    training = config_value.get("training")
    objective = config_value.get("objective")
    if not isinstance(manifest_config, Mapping):
        raise ValueError("Manifest/config compatibility requires manifest.config mapping.")
    if not isinstance(freeze, Mapping):
        raise ValueError("Manifest/config compatibility requires manifest.freeze mapping.")
    if not isinstance(data, Mapping):
        raise ValueError("Manifest/config compatibility requires config.data mapping.")
    if not isinstance(architecture, Mapping):
        raise ValueError("Manifest/config compatibility requires config.architecture mapping.")
    if not isinstance(training, Mapping):
        raise ValueError("Manifest/config compatibility requires config.training mapping.")
    if not isinstance(objective, Mapping):
        raise ValueError("Manifest/config compatibility requires mapping sections.")
    encoder = architecture.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("Manifest/config compatibility requires architecture.encoder.")

    def equal(name: str, actual: object, expected: object) -> None:
        if actual != expected:
            raise ValueError(
                f"Signed manifest/config mismatch for {name}: {actual!r} != {expected!r}."
            )

    def floats(value: object) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("Manifest/config horizons must be sequences.")
        return [float(item) for item in value]

    equal("horizons_s", floats(manifest_config["horizons_s"]), floats(data["horizons_s"]))
    equal(
        "horizon_tolerance_s",
        float(manifest_config["horizon_tolerance_s"]),
        float(data["horizon_tolerance_s"]),
    )
    equal(
        "exclusion_window_s",
        float(manifest_config["exclusion_window_s"]),
        float(data["exclusion_window_s"]),
    )
    equal(
        "objective.nce_exclusion_window_s",
        float(objective["nce_exclusion_window_s"]),
        float(data["exclusion_window_s"]),
    )
    equal("width", int(manifest_config["ssl_width"]), int(data["width"]))
    equal("height", int(manifest_config["ssl_height"]), int(data["height"]))
    equal("temporal_steps", int(manifest_config["temporal_steps"]), int(data["temporal_steps"]))
    equal("bins", int(manifest_config["bins"]), int(data["bins"]))
    equal("in_channels", int(encoder["in_channels"]), 21)
    equal("batch_size", int(manifest_config["batch_size"]), int(training["batch_size"]))
    equal("batch_size.freeze", int(freeze["batch_size"]), int(training["batch_size"]))
    equal("max_workers", int(manifest_config["max_workers"]), int(training["workers"]))
    equal("max_workers.freeze", int(freeze["max_workers"]), int(training["workers"]))
    equal("update_budget", int(manifest_config["update_budget"]), int(training["total_updates"]))
    equal("update_budget.freeze", int(freeze["update_budget"]), int(training["total_updates"]))
    equal(
        "ema_total_updates", int(architecture["ema_total_updates"]), int(training["total_updates"])
    )
    equal(
        "seeds", [int(item) for item in freeze["seeds"]], [int(item) for item in training["seeds"]]
    )
    requested_stage = str(data["stage"])
    available_stages = {
        str(item.get("stage")) for item in manifest.get("stages", []) if isinstance(item, Mapping)
    }
    if requested_stage not in available_stages:
        raise ValueError(
            f"Signed manifest/config mismatch for stage: {requested_stage!r} is unavailable."
        )
    equal("modality", list(freeze["modalities"]), [str(data["modality"])])
    equal("input_policy", str(freeze["ssl_input_policy"]), str(data["input_policy"]))
    equal("calibration_mode", str(freeze["calibration_mode"]), str(data["calibration_mode"]))
    equal(
        "signed_ttc_convention",
        str(manifest_config["signed_ttc_convention"]),
        str(data["signed_ttc_convention"]),
    )
    equal(
        "signed_ttc_convention.freeze",
        str(freeze["signed_ttc_convention"]),
        str(data["signed_ttc_convention"]),
    )
    equal("sampler_policy", str(manifest["selection_rule"]), str(data["sampler_policy"]))
    equal("max_batch_size", int(architecture["max_batch_size"]), int(training["batch_size"]))
    equal(
        "max_temporal_steps", int(architecture["max_temporal_steps"]), int(data["temporal_steps"])
    )
    equal("max_horizons", int(architecture["max_horizons"]), len(data["horizons_s"]))


def build_parser() -> argparse.ArgumentParser:
    """Build a bounded CLI without annotation, benchmark, or supervised-label inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True, help="Read-only raw eAP root.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Already-signed label-free matched-manifest JSON; selection is not rebuilt here.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Label-free Dense Level-Dynamics YAML."
    )
    parser.add_argument("--seed", type=int, default=7, help="Controlled initial seed (default: 7).")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Bounded run output directory."
    )
    parser.add_argument(
        "--batch-size",
        type=lambda value: _bounded_positive(value, name="batch-size", maximum=2),
        default=None,
        help="Optional assertion; production value comes from config (must match).",
    )
    parser.add_argument(
        "--temporal-steps",
        type=lambda value: _bounded_positive(value, name="temporal-steps", maximum=5),
        default=None,
        help="Optional assertion; production value comes from config (must match).",
    )
    parser.add_argument(
        "--horizons",
        type=lambda value: _bounded_positive(value, name="horizons", maximum=3),
        default=None,
        help="Optional assertion; production value comes from config (must match).",
    )
    parser.add_argument(
        "--patch-query-chunk-size",
        type=lambda value: _bounded_positive(value, name="patch-query-chunk-size", maximum=60),
        default=None,
        help="Optional assertion; production value comes from config (must match).",
    )
    parser.add_argument(
        "--max-updates",
        type=lambda value: _bounded_positive(value, name="max-updates", maximum=100_000),
        default=None,
        help="Optional assertion; production value comes from config.",
    )
    parser.add_argument(
        "--workers",
        type=lambda value: _bounded_positive(value, name="workers", maximum=8),
        default=None,
        help="Optional assertion; production value comes from config (must match).",
    )
    parser.add_argument("--device", default="auto", help="Requested bounded device identifier.")
    parser.add_argument(
        "--resume", action="store_true", help="Resume only from a matching core checkpoint."
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path used with --resume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify root/config/signature/resource arguments without opening event data.",
    )
    return parser


def _resolve_resource_args(
    args: argparse.Namespace, config_value: dict[str, Any]
) -> dict[str, int]:
    """Resolve bounded production resources and reject mismatched assertions."""

    architecture = config_value["architecture"]
    data = config_value["data"]
    training = config_value["training"]
    expected = {
        "batch_size": int(training["batch_size"]),
        "temporal_steps": int(data["temporal_steps"]),
        "horizons": len(data["horizons_s"]),
        "patch_query_chunk_size": int(architecture["patch_query_chunk_size"]),
        "max_updates": int(training["total_updates"]),
        "workers": int(training["workers"]),
    }
    for key, configured in expected.items():
        asserted = getattr(args, key, None)
        if asserted is not None and int(asserted) != configured:
            raise ValueError(
                f"CLI {key}={int(asserted)} differs from configured production value {configured}."
            )
    if expected["batch_size"] != 2:
        raise ValueError("The frozen NCE microbatch contract requires config batch_size=2.")
    if not 0 <= expected["workers"] <= 8:
        raise ValueError("Configured workers must lie in [0,8].")
    return expected


def _cycling_batches(batches: Iterable[Any], *, start_offset: int = 0) -> Iterator[Any]:
    """Cycle a re-iterable loader from an exact resume offset without caching tensors."""

    if start_offset < 0:
        raise ValueError("Cycling batch start_offset must be non-negative.")
    remaining_skip = int(start_offset)

    while True:
        seen = False
        for batch in batches:
            seen = True
            if remaining_skip:
                remaining_skip -= 1
                continue
            yield batch
        if not seen:
            raise RuntimeError("Label-free DataLoader produced no batches to cycle.")


def _metrics_history_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_metric_history(rows: Sequence[Mapping[str, Any]], expected_count: int) -> None:
    if len(rows) != int(expected_count):
        raise ValueError(
            f"Metrics history has {len(rows)} rows; expected exactly {int(expected_count)}."
        )
    updates: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping) or "update" not in row:
            raise ValueError("Metrics history rows must be mappings with update fields.")
        updates.append(int(round(float(row["update"]))))
    if updates != list(range(1, int(expected_count) + 1)):
        raise ValueError(
            "Metrics history updates must be the contiguous sequence 1..total_updates."
        )


def _read_metrics_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume metrics history is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Metrics history line {line_number} is invalid JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Metrics history line {line_number} must be an object.")
        rows.append(value)
    return rows


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_resume_metrics(
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, Any],
    provenance: LabelFreeManifestProvenance,
) -> list[dict[str, Any]]:
    metrics_path = output_dir / "metrics.jsonl"
    meta_path = output_dir / "metrics.jsonl.meta.json"
    rows = _read_metrics_history(metrics_path)
    _validate_metric_history(rows, int(checkpoint_payload.get("update_count", 0)))
    if not meta_path.is_file():
        raise FileNotFoundError(f"Resume metrics binding metadata is missing: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("Resume metrics binding metadata must be an object.")
    expected = {
        "update_count": int(checkpoint_payload.get("update_count", 0)),
        "history_hash": _metrics_history_hash(rows),
        "checkpoint_sha256": _sha256_path(checkpoint_path),
        "checkpoint_config_hash": checkpoint_payload.get("config_hash"),
        "matched_manifest_hash": provenance.matched_manifest_hash,
        "sampler_order_hash": provenance.sampler_order_hash,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError("Resume metrics history binding mismatch: " + ", ".join(mismatches))
    return rows


def _write_metrics_history(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, Any],
    provenance: LabelFreeManifestProvenance,
) -> tuple[Path, Path]:
    metrics_path = output_dir / "metrics.jsonl"
    metrics_tmp = metrics_path.with_suffix(".jsonl.tmp")
    with metrics_tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            )
    metrics_tmp.replace(metrics_path)
    metadata = {
        "artifact_type": "dense_level_dynamics_metrics_binding_v1",
        "update_count": len(rows),
        "history_hash": _metrics_history_hash(rows),
        "checkpoint_sha256": _sha256_path(checkpoint_path),
        "checkpoint_config_hash": checkpoint_payload.get("config_hash"),
        "matched_manifest_hash": provenance.matched_manifest_hash,
        "sampler_order_hash": provenance.sampler_order_hash,
    }
    meta_path = output_dir / "metrics.jsonl.meta.json"
    meta_tmp = meta_path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta_tmp.replace(meta_path)
    return metrics_path, meta_path


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a signed request and optionally execute its bounded update loop."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.eap_root.is_dir():
            raise FileNotFoundError(f"eAP root is missing or not a directory: {args.eap_root}")
        manifest, provenance = load_signed_label_free_manifest(args.manifest)
        if manifest.get("artifact_type") != "matched_eap_subset_v1":
            raise ValueError("Pretraining requires a signed matched_eap_subset_v1 manifest.")
        validate_matched_manifest(manifest)
        config_value = _load_label_free_config(args.config)
        if config_value.get("artifact_type") == "dense_level_dynamics_jepa_train_config_v1":
            trainer_config = _approved_trainer_config(config_value, seed=args.seed)
            _validate_shared_objective_configs(args.config, config_value)
        else:
            raise ValueError(
                "Pretraining requires one of the four dense_level_dynamics train YAMLs; "
                "legacy/generic configs are rejected."
            )
        _validate_manifest_config_compatibility(manifest, config_value)
        resources = _resolve_resource_args(args, config_value)
        if args.resume_checkpoint is not None and not args.resume:
            raise ValueError("--resume-checkpoint requires --resume.")
        resume_path = (
            args.resume_checkpoint
            if args.resume_checkpoint is not None
            else args.output_dir / "checkpoint.pt"
        )
        if args.resume and not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {resume_path}")
        stage = str(config_value["data"]["stage"])
        if not stage.startswith("matched_"):
            raise ValueError("Dense Level-Dynamics pretraining requires a matched manifest stage.")
        manifest_stage_names = {
            str(item.get("stage")) for item in manifest.get("stages", []) if isinstance(item, dict)
        }
        if stage not in manifest_stage_names:
            raise ValueError(f"Requested manifest stage {stage!r} is unavailable.")
        selected_stage = next(
            item
            for item in manifest.get("stages", [])
            if isinstance(item, dict) and str(item.get("stage")) == stage
        )
        stage_row_ids = {str(row_id) for row_id in selected_stage.get("row_ids", [])}
        selected_train_rows = sum(
            1
            for row in manifest.get("rows", [])
            if isinstance(row, dict)
            and str(row.get("row_id")) in stage_row_ids
            and str(row.get("role")) == "train"
        )
        summary = {
            "artifact_type": "dense_level_dynamics_jepa_preflight_v1",
            "status": "preflight_passed",
            "eap_root": args.eap_root.resolve().as_posix(),
            "manifest": args.manifest.resolve().as_posix(),
            "matched_manifest_hash": provenance.matched_manifest_hash,
            "split_hash": provenance.split_hash,
            "sampler_order_hash": provenance.sampler_order_hash,
            "selection_rule": provenance.selection_rule,
            "seed": args.seed,
            "role": "train",
            "selected_stage_rows": len(stage_row_ids),
            "selected_train_rows": selected_train_rows,
            "resident_limits": {
                **resources,
            },
            "manifest_top_level_fields": sorted(manifest),
            "raw_events_opened": False,
            "annotation_files_opened": False,
            "dense_disk_cache_created": False,
        }
        if args.dry_run:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        requested_device = str(args.device)
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this host.")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training_started = time.perf_counter()
        dataset: EAPHighResLabelFreeDataset | None = None
        trainer: EAPHighResJEPATrainer | None = None
        metrics: list[dict[str, Any]] = []
        existing_history: list[dict[str, Any]] = []
        metrics_path = args.output_dir / "metrics.jsonl"
        if not args.resume and (
            metrics_path.exists() or (args.output_dir / "checkpoint.pt").exists()
        ):
            raise FileExistsError(
                "Output directory already contains metrics/checkpoint; use --resume to continue."
            )
        try:
            dataset = EAPHighResLabelFreeDataset(
                manifest,
                eap_root=args.eap_root,
                role="train",
                stage=stage,
                temporal_steps=resources["temporal_steps"],
                width=int(config_value["data"]["width"]),
                height=int(config_value["data"]["height"]),
                bins=int(config_value["data"]["bins"]),
            )
            loader = make_label_free_loader(
                dataset,
                batch_size=resources["batch_size"],
                num_workers=resources["workers"],
                exclusion_window_s=float(config_value["data"]["exclusion_window_s"]),
                positive_tolerance_s=float(config_value["data"]["horizon_tolerance_s"]),
            )
            trainer = EAPHighResJEPATrainer(trainer_config, device=device)
            if args.resume:
                checkpoint_payload = torch.load(
                    resume_path, map_location=device, weights_only=False
                )
                if not isinstance(checkpoint_payload, dict):
                    raise ValueError("Resume checkpoint must be a mapping.")
                expected_provenance = {
                    "matched_manifest_hash": provenance.matched_manifest_hash,
                    "split_hash": provenance.split_hash,
                    "sampler_order_hash": provenance.sampler_order_hash,
                    "selection_rule": provenance.selection_rule,
                    "dataset_hashes": dict(provenance.dataset_hashes),
                    "label_family_provenance": dict(provenance.label_family_provenance),
                }
                mismatches = {
                    key: (checkpoint_payload.get(key), expected)
                    for key, expected in expected_provenance.items()
                    if checkpoint_payload.get(key) != expected
                }
                if mismatches:
                    raise ValueError(
                        "Resume checkpoint provenance differs from the signed manifest: "
                        + ", ".join(sorted(mismatches))
                    )
                existing_history = _validate_resume_metrics(
                    args.output_dir,
                    resume_path,
                    checkpoint_payload,
                    provenance,
                )
                trainer.load_checkpoint(resume_path)
                if trainer.update_count >= trainer_config.total_updates:
                    raise RuntimeError(
                        "Resume checkpoint already exhausted the configured update budget; "
                        "no early-complete result is emitted."
                    )
            uses_dynamics_nce = bool(
                getattr(trainer_config.loss.objective, "uses_dynamics_nce", False)
            )
            if uses_dynamics_nce and not trainer.nce_preflight_passed:
                trainer.preflight_nce_batches(loader, max_batches=None)
            remaining = trainer_config.total_updates - trainer.update_count
            loader_batches = len(loader)
            if loader_batches <= 0:
                raise RuntimeError("Label-free DataLoader has no complete batches.")
            batch_stream = _cycling_batches(
                loader,
                start_offset=trainer.update_count % loader_batches,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = args.output_dir / "checkpoint.pt"
            complete_history = list(existing_history)
            checkpoint_interval = 100
            while trainer.update_count < trainer_config.total_updates:
                until_boundary = checkpoint_interval - (trainer.update_count % checkpoint_interval)
                chunk = min(
                    until_boundary,
                    trainer_config.total_updates - trainer.update_count,
                )
                segment = trainer.train_batches(batch_stream, max_updates=chunk)
                metrics.extend(segment)
                complete_history.extend(segment)
                trainer.save_checkpoint(checkpoint, provenance)
                checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if not isinstance(checkpoint_payload, Mapping):
                    raise RuntimeError("Saved checkpoint payload is not a mapping.")
                _write_metrics_history(
                    args.output_dir,
                    complete_history,
                    checkpoint,
                    checkpoint_payload,
                    provenance,
                )
            if trainer.update_count != trainer_config.total_updates or len(metrics) != remaining:
                raise RuntimeError(
                    "Training stopped before the exact configured update budget: "
                    f"updates={trainer.update_count}, expected={trainer_config.total_updates}."
                )
            _validate_metric_history(complete_history, trainer_config.total_updates)
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint_payload, Mapping):
                raise RuntimeError("Saved checkpoint payload is not a mapping.")
            metrics_path, metrics_meta_path = _write_metrics_history(
                args.output_dir,
                complete_history,
                checkpoint,
                checkpoint_payload,
                provenance,
            )
            summary.update(
                {
                    "status": "completed",
                    "raw_events_opened": True,
                    "metrics_path": metrics_path.resolve().as_posix(),
                    "metrics_binding_path": metrics_meta_path.resolve().as_posix(),
                    "checkpoint_path": checkpoint.resolve().as_posix(),
                    "updates": trainer.update_count,
                    "device": str(device),
                    "precision": trainer_config.precision,
                    "checkpoint_interval_updates": checkpoint_interval,
                    "training_wall_seconds": time.perf_counter() - training_started,
                    "peak_vram_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                    ),
                    "peak_vram_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
                    ),
                }
            )
            summary["updates_per_second"] = trainer.update_count / max(
                float(summary["training_wall_seconds"]), 1e-9
            )
            summary_path = args.output_dir / "summary.json"
            summary_tmp = summary_path.with_suffix(".json.tmp")
            summary_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            summary_tmp.replace(summary_path)
        finally:
            if dataset is not None:
                dataset.close()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            f"Dense Level-Dynamics pretraining request rejected: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
