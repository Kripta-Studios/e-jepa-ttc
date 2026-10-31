"""Strict state and training primitives for an isolated S3 temporal campaign."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from e_jepa_ttc.detection.campaign_training import (
    CHECKPOINT_SCHEMA,
    CampaignConfig,
    UpdateLRScheduler,
    canonical_sha256,
    capture_rng_state,
    load_campaign_config,
    restore_rng_state,
    sha256_file,
)
from e_jepa_ttc.detection.temporal_campaign import (
    D4096_V5_MANIFEST_SHA256,
    FULL_SELECTION_V1_SHA256,
    HISTORY_MEDIA_RECEIPT_V2_SHA256,
    TEMPORAL_MEDIA_ADMISSION_V2_SHA256,
    TemporalCampaignBatch,
    TemporalCampaignConfig,
    TemporalCampaignSampler,
)
from e_jepa_ttc.detection.temporal_detector import TemporalDetectorAdapter
from e_jepa_ttc.detection.temporal_fusion import CausalTemporalFeatureFusion
from e_jepa_ttc.detection.temporal_source_patch import portable_tree_hashes

TEMPORAL_TRAINING_SCHEMA = "eap_detection_temporal_training_config_v1"
TEMPORAL_CONTINUATION_SCHEMA = "eap_detection_temporal_training_config_v2"
TEMPORAL_CHECKPOINT_SCHEMA = "eap_detection_temporal_checkpoint_v1"
FROZEN_SPLIT_SHA256 = "91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903"
FROZEN_DATASET_REVISION = "a814a946e60d37f7242358af5f344542177e50c5"

TRAINING_CODE_PATHS = (
    "scripts/smoke_monodgp_portable.py",
    "src/e_jepa_ttc/detection/campaign_resources.py",
    "scripts/train_detection_temporal_campaign.py",
    "src/e_jepa_ttc/detection/campaign_training.py",
    "src/e_jepa_ttc/detection/development_metrics.py",
    "src/e_jepa_ttc/detection/eap_rgb_data.py",
    "src/e_jepa_ttc/detection/evaluation_ap_fast.py",
    "src/e_jepa_ttc/detection/evaluation_ground_truth.py",
    "src/e_jepa_ttc/detection/evaluation_protocol.py",
    "src/e_jepa_ttc/detection/evaluation_v2.py",
    "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
    "src/e_jepa_ttc/detection/temporal_campaign.py",
    "src/e_jepa_ttc/detection/temporal_detector.py",
    "src/e_jepa_ttc/detection/temporal_fusion.py",
    "src/e_jepa_ttc/detection/temporal_source_patch.py",
    "src/e_jepa_ttc/detection/temporal_training.py",
)


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a full lowercase SHA-256")
    return value


def _exact(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    result = dict(value)
    if set(result) != keys:
        raise ValueError(
            f"{name} keys differ: unknown={sorted(set(result) - keys)}, "
            f"missing={sorted(keys - set(result))}"
        )
    return result


@dataclass(frozen=True)
class ParentCheckpointConfig:
    path: str
    sha256: str
    update: int
    runtime_provenance_sha256: str


@dataclass(frozen=True)
class TemporalRuntimeConfig:
    portable_root: str
    source_manifest: str
    source_manifest_sha256: str
    expected_output_tree_sha256: str
    parity_report: str
    parity_report_sha256: str
    temporal_media_admission: str
    temporal_media_admission_sha256: str
    history_media_receipt: str
    history_media_receipt_sha256: str


@dataclass(frozen=True)
class TemporalTrainingConfig:
    """Baseline recipe plus immutable temporal parent/runtime bindings."""

    schema: str
    baseline_config_path: str
    baseline: CampaignConfig
    temporal: TemporalCampaignConfig
    parent: ParentCheckpointConfig
    runtime: TemporalRuntimeConfig
    parent_config: CampaignConfig | None = None

    @property
    def sha256(self) -> str:
        value = asdict(self)
        value["baseline"] = asdict(self.baseline)
        if self.parent_config is None:
            value.pop("parent_config")
        return canonical_sha256(value)


def load_temporal_training_config(path: str | Path) -> TemporalTrainingConfig:
    """Load a strict wrapper without weakening the ordinary CampaignConfig."""
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    continuation = isinstance(document, Mapping) and document.get("schema") == (
        TEMPORAL_CONTINUATION_SCHEMA
    )
    raw = _exact(
        document,
        {"schema", "baseline_config", "temporal", "parent", "runtime"}
        | ({"parent_config"} if continuation else set()),
        "temporal_training",
    )
    if raw["schema"] not in (TEMPORAL_TRAINING_SCHEMA, TEMPORAL_CONTINUATION_SCHEMA):
        raise ValueError("Unknown temporal training schema")
    baseline_path = (path.parent / str(raw["baseline_config"])).resolve()
    baseline = load_campaign_config(baseline_path)
    if (
        (not continuation and baseline.optimization.updates != 8192)
        or baseline.optimization.batch_size != 4
        or baseline.optimization.precision != "fp32"
        or baseline.optimization.matmul_precision != "highest"
        or (not continuation and baseline.evaluation.full_selection_updates != (4096, 8192))
    ):
        raise ValueError("S3 fixes 8192 extra updates, batch 4, FP32-highest and full evals")
    parent_config = (
        load_campaign_config((path.parent / str(raw["parent_config"])).resolve())
        if continuation
        else None
    )
    if parent_config is not None:
        for field in (
            "backbone",
            "num_classes",
            "queries",
            "group_num",
            "predict_velocity",
            "velocity_loss_coef",
            "width",
            "height",
        ):
            if getattr(parent_config.model, field) != getattr(baseline.model, field):
                raise ValueError(f"Continuation changes parent model contract: {field}")
        if (
            parent_config.data.expected_dataset_revision != baseline.data.expected_dataset_revision
            or parent_config.data.expected_split_sha256 != baseline.data.expected_split_sha256
        ):
            raise ValueError("Continuation must preserve parent dataset release and split")
    if any(
        value != 0
        for value in (
            baseline.augmentation.brightness,
            baseline.augmentation.contrast,
            baseline.augmentation.saturation,
        )
    ):
        raise ValueError("First S3 campaign has no preregistered temporal photometric transform")
    temporal = TemporalCampaignConfig.from_mapping(
        _exact(
            raw["temporal"],
            {
                "schema",
                "variant",
                "producer_version",
                "maximum_history_age_us",
                "width",
                "height",
                "include_velocity",
                "generation",
                "media_failure_policy",
            },
            "temporal",
        )
    )
    if (
        temporal.width != baseline.model.width
        or temporal.height != baseline.model.height
        or temporal.include_velocity != baseline.model.predict_velocity
    ):
        raise ValueError("Temporal input/head settings differ from the baseline recipe")
    parent = _exact(
        raw["parent"], {"path", "sha256", "update", "runtime_provenance_sha256"}, "parent"
    )
    runtime = _exact(
        raw["runtime"],
        {
            "portable_root",
            "source_manifest",
            "source_manifest_sha256",
            "expected_output_tree_sha256",
            "parity_report",
            "parity_report_sha256",
            "temporal_media_admission",
            "temporal_media_admission_sha256",
            "history_media_receipt",
            "history_media_receipt_sha256",
        },
        "runtime",
    )
    if runtime["temporal_media_admission_sha256"] != TEMPORAL_MEDIA_ADMISSION_V2_SHA256:
        raise ValueError("Temporal admission hash is not the preregistered receipt")
    if runtime["history_media_receipt_sha256"] != HISTORY_MEDIA_RECEIPT_V2_SHA256:
        raise ValueError("History receipt hash is not the preregistered receipt")
    parent_update = parent["update"]
    if (
        isinstance(parent_update, bool)
        or not isinstance(parent_update, int)
        or not 1 <= parent_update <= (parent_config or baseline).optimization.updates
    ):
        raise ValueError("parent.update must identify a completed baseline monitor update")
    return TemporalTrainingConfig(
        schema=TEMPORAL_TRAINING_SCHEMA,
        baseline_config_path=str(baseline_path),
        baseline=baseline,
        temporal=temporal,
        parent=ParentCheckpointConfig(
            path=str(parent["path"]),
            sha256=_sha(parent["sha256"], "parent.sha256"),
            update=parent_update,
            runtime_provenance_sha256=_sha(
                parent["runtime_provenance_sha256"], "parent.runtime_provenance_sha256"
            ),
        ),
        runtime=TemporalRuntimeConfig(
            portable_root=str(runtime["portable_root"]),
            source_manifest=str(runtime["source_manifest"]),
            source_manifest_sha256=_sha(
                runtime["source_manifest_sha256"], "runtime.source_manifest_sha256"
            ),
            expected_output_tree_sha256=_sha(
                runtime["expected_output_tree_sha256"], "runtime.expected_output_tree_sha256"
            ),
            parity_report=str(runtime["parity_report"]),
            parity_report_sha256=_sha(
                runtime["parity_report_sha256"], "runtime.parity_report_sha256"
            ),
            temporal_media_admission=str(runtime["temporal_media_admission"]),
            temporal_media_admission_sha256=str(runtime["temporal_media_admission_sha256"]),
            history_media_receipt=str(runtime["history_media_receipt"]),
            history_media_receipt_sha256=str(runtime["history_media_receipt_sha256"]),
        ),
        parent_config=parent_config,
    )


def validate_temporal_parity_evidence(
    parity: Mapping[str, Any],
    *,
    parent_sha256: str,
    source_manifest_sha256: str,
    repository_root: Path,
) -> None:
    """Reject shallow PASS receipts without exact, immutable parity evidence."""
    comparisons = parity.get("comparisons")
    expected = {
        "original_vs_split",
        "split_vs_current_only",
        "split_vs_current_repeated",
    }
    runtime_hashes = parity.get("runtime_source_hashes")
    required_paths = {
        "scripts/smoke_monodgp_portable.py",
        "scripts/verify_temporal_detector_parity.py",
        "src/e_jepa_ttc/detection/backbone_initialization.py",
        "src/e_jepa_ttc/detection/eap_rgb_data.py",
        "src/e_jepa_ttc/detection/geometry_contract.py",
        "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
        "src/e_jepa_ttc/detection/portable_attention.py",
        "src/e_jepa_ttc/detection/temporal_detector.py",
        "src/e_jepa_ttc/detection/temporal_fusion.py",
        "src/e_jepa_ttc/detection/velocity_extension.py",
    }
    sources_valid = (
        isinstance(runtime_hashes, Mapping)
        and required_paths.issubset(runtime_hashes)
        and parity.get("runtime_source_hashes_sha256") == canonical_sha256(runtime_hashes)
        and all(
            sha256_file(repository_root / relative) == runtime_hashes[relative]
            for relative in required_paths
        )
    )
    comparisons_valid = (
        isinstance(comparisons, Mapping)
        and set(comparisons) == expected
        and all(
            isinstance(value, Mapping)
            and value.get("exact") is True
            and value.get("maximum_absolute_error") == 0.0
            and isinstance(value.get("tensor_count"), int)
            and value["tensor_count"] > 0
            and value.get("mismatches") == []
            for value in comparisons.values()
        )
    )
    workers = parity.get("workers")
    workers_valid = isinstance(workers, Mapping) and set(workers) == {"original", "temporal"}
    if workers_valid:
        for name, worker in cast(Mapping[str, object], workers).items():
            if not isinstance(worker, Mapping):
                workers_valid = False
                continue
            workers_valid = workers_valid and (
                worker.get("schema") == "temporal_detector_parity_worker_v1"
                and worker.get("checkpoint_sha256") == parent_sha256
                and worker.get("targets_argument") is None
                and worker.get("annotations_returned") == 0
                and worker.get("label_files_read") is False
                and worker.get("label_free_reader_override") is True
                and worker.get("optimizer_updates") == 0
                and worker.get("model_state_before_sha256")
                == worker.get("model_state_after_sha256")
                and isinstance(worker.get("tensor_output_sha256"), str)
            )
            if name == "temporal":
                workers_valid = workers_valid and (
                    worker.get("fusion_state_before_sha256")
                    == worker.get("fusion_state_after_sha256")
                    and worker.get("fusion_state_before_sha256") is not None
                )
    if (
        parity.get("schema") != "temporal_detector_engineering_parity_v1"
        or parity.get("status") != "PASS"
        or parity.get("checkpoint_sha256") != parent_sha256
        or parity.get("source_manifest_sha256") != source_manifest_sha256
        or parity.get("targets_read") is not False
        or parity.get("optimizer_updates") != 0
        or parity.get("decoded_predictions_exact") is not True
        or parity.get("sample_count") != 4
        or not comparisons_valid
        or not workers_valid
        or not sources_valid
    ):
        raise ValueError("Temporal GPU parity evidence differs from parent/source contract")


def validate_parent_checkpoint_payload(
    payload: Mapping[str, object], config: TemporalTrainingConfig
) -> None:
    """Require the selected parent to use the exact inherited baseline config."""
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("producer") != "scripts/train_detection_campaign.py"
        or payload.get("config_sha256") != (config.parent_config or config.baseline).sha256
        or payload.get("update") != config.parent.update
        or payload.get("runtime_provenance_sha256") != config.parent.runtime_provenance_sha256
    ):
        raise ValueError("Parent is not the configured selected baseline checkpoint")


def validate_temporal_runtime(config: TemporalTrainingConfig) -> dict[str, object]:
    """Bind parent, materialized source, and admission before model construction."""
    parent = Path(config.parent.path).resolve()
    source_manifest_path = Path(config.runtime.source_manifest).resolve()
    paths = {
        "parent_checkpoint": (parent, config.parent.sha256),
        "source_manifest": (source_manifest_path, config.runtime.source_manifest_sha256),
        "parity_report": (
            Path(config.runtime.parity_report).resolve(),
            config.runtime.parity_report_sha256,
        ),
        "temporal_media_admission": (
            Path(config.runtime.temporal_media_admission).resolve(),
            config.runtime.temporal_media_admission_sha256,
        ),
        "history_media_receipt": (
            Path(config.runtime.history_media_receipt).resolve(),
            config.runtime.history_media_receipt_sha256,
        ),
        "baseline_manifest": (
            Path(config.baseline.data.manifest).resolve(),
            config.baseline.data.expected_manifest_sha256,
        ),
        "baseline_split": (
            Path(config.baseline.data.split).resolve(),
            config.baseline.data.expected_split_sha256,
        ),
        "full_selection_tokens": (
            Path(config.baseline.data.full_selection_tokens).resolve(),
            config.baseline.data.expected_full_selection_sha256,
        ),
    }
    for name, (artifact, digest) in paths.items():
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"{name} differs from configured SHA-256")
    if parent.name != "selected.pt":
        raise ValueError("Temporal parent must be the baseline selected.pt checkpoint")
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    validate_parent_checkpoint_payload(payload, config)
    data = config.baseline.data
    inherited = {
        "train_role": data.train_role == "train",
        "selection_role": data.selection_role == "selection",
        "dataset_revision": data.expected_dataset_revision == FROZEN_DATASET_REVISION,
        "manifest": data.expected_manifest_sha256 == D4096_V5_MANIFEST_SHA256,
        "split": data.expected_split_sha256 == FROZEN_SPLIT_SHA256,
        "full_selection": data.expected_full_selection_sha256 == FULL_SELECTION_V1_SHA256,
        "prepared_cache_nonapplicable_fixed": data.cache_capacity == 8,
        "label_cache_applied": data.label_cache_sequences == 24,
    }
    if not all(inherited.values()):
        raise ValueError("Inherited baseline data declarations differ from S3 implementation")
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    parity = json.loads(Path(config.runtime.parity_report).read_text(encoding="utf-8"))
    portable_root = Path(config.runtime.portable_root).resolve()
    if not portable_root.is_dir():
        raise FileNotFoundError("Materialized temporal portable root is absent")
    portable_hashes = portable_tree_hashes(portable_root)
    observed_tree = canonical_sha256(portable_hashes)
    if (
        manifest.get("schema") != "monodgp_projected_temporal_source_v1"
        or manifest.get("variant") != "monodgp_projected_temporal_v1"
        or manifest.get("checkpoint_sha256") != config.parent.sha256
        or manifest.get("output_tree_sha256") != config.runtime.expected_output_tree_sha256
        or observed_tree != config.runtime.expected_output_tree_sha256
        or manifest.get("state_dict_module_names_changed") is not False
    ):
        raise ValueError("Materialized temporal source differs from parent/runtime contract")
    root = Path(__file__).resolve().parents[3]
    validate_temporal_parity_evidence(
        parity,
        parent_sha256=config.parent.sha256,
        source_manifest_sha256=config.runtime.source_manifest_sha256,
        repository_root=root,
    )
    training_code_hashes = {
        relative: sha256_file(root / relative) for relative in TRAINING_CODE_PATHS
    }
    return {
        "schema": "eap_detection_temporal_runtime_admission_v1",
        "config_sha256": config.sha256,
        "artifacts": {key: digest for key, (_, digest) in paths.items()},
        "parent_update": config.parent.update,
        "output_tree_sha256": config.runtime.expected_output_tree_sha256,
        "portable_tree_file_count": len(portable_hashes),
        "training_code_hashes": training_code_hashes,
        "training_code_hashes_sha256": canonical_sha256(training_code_hashes),
        "inherited_config_application": inherited,
        "status": "PASS_BEFORE_MODEL_CONSTRUCTION",
    }


def immutable_resource_contract(
    config: TemporalTrainingConfig, observed: Mapping[str, object]
) -> dict[str, object]:
    """Keep dynamic headroom as evidence while binding only stable resource requirements."""
    if observed.get("status") != "PASS":
        raise ValueError("Resource admission did not pass")
    return {
        "precision": config.baseline.optimization.precision,
        "matmul_precision": config.baseline.optimization.matmul_precision,
        "batch_size": config.baseline.optimization.batch_size,
        "num_cpu_threads": config.baseline.resources.num_cpu_threads,
        "maximum_seconds": config.baseline.resources.maximum_seconds,
        "minimum_available_ram_gb": config.baseline.resources.minimum_available_ram_gb,
        "minimum_available_vram_gb": config.baseline.resources.minimum_available_vram_gb,
        "visible_cuda_devices": 1,
        "admission_status": "PASS",
    }


def publish_selected_candidate(
    candidate: Path,
    *,
    checkpoints_dir: Path,
    update: int,
    expected_sha256: str,
) -> Path:
    """Publish or idempotently reuse an immutable selected checkpoint.

    A crash may occur after publication but before ``last.pt`` commits the
    selected lineage. Replaying that update reuses only byte-identical output.
    """
    digest = _sha(expected_sha256, "expected_sha256")
    if not candidate.is_file():
        raise FileNotFoundError(f"Selected checkpoint candidate is missing: {candidate}")
    if sha256_file(candidate) != digest:
        raise ValueError("Selected checkpoint candidate digest does not match save receipt")
    destination = checkpoints_dir / f"selected_update_{update:06d}_{digest}.pt"
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != digest:
            raise FileExistsError(
                "Immutable selected checkpoint path exists with different content"
            )
        candidate.unlink()
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate, destination)
    return destination


def load_parent_model_only(
    backend: torch.nn.Module, config: TemporalTrainingConfig
) -> Mapping[str, torch.Tensor]:
    """Strictly load only parent model tensors, never its optimizer or RNG."""
    payload = torch.load(config.parent.path, map_location="cpu", weights_only=True)
    state = cast(Mapping[str, torch.Tensor], payload["model"])
    backend.load_state_dict(dict(state), strict=True)
    return state


def seeded_fusion(seed: int) -> CausalTemporalFeatureFusion:
    """Create identical fusion initialization for every control without changing global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CausalTemporalFeatureFusion()


def temporal_optimizer_parameter_groups(
    adapter: TemporalDetectorAdapter, config: CampaignConfig
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Preserve the baseline backbone multiplier under the adapter name prefix."""
    backbone, detector, names = [], [], {"backbone": [], "detector_and_fusion": []}
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backend.backbone."):
            backbone.append(parameter)
            names["backbone"].append(name)
        else:
            detector.append(parameter)
            names["detector_and_fusion"].append(name)
    fusion_present = any(name.startswith("fusion.") for name in names["detector_and_fusion"])
    if not backbone or not detector or not fusion_present:
        raise ValueError("Temporal optimizer cannot identify backbone/detector/fusion parameters")
    optimization = config.optimization
    groups = [
        {
            "params": backbone,
            "lr": optimization.learning_rate * optimization.backbone_lr_multiplier,
            "weight_decay": optimization.weight_decay,
            "name": "backbone",
        },
        {
            "params": detector,
            "lr": optimization.learning_rate,
            "weight_decay": optimization.weight_decay,
            "name": "detector_and_fusion",
        },
    ]
    return groups, {
        key: {"names_sha256": canonical_sha256(value), "tensors": len(value)}
        for key, value in names.items()
    }


def temporal_train_step(
    adapter: TemporalDetectorAdapter,
    criterion: torch.nn.Module,
    batch: TemporalCampaignBatch,
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateLRScheduler,
    update: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    """Execute one update; histories remain detached inside the adapter."""
    if batch.targets is None:
        raise ValueError("Temporal training requires current-frame targets")
    scheduler.apply(update)
    optimizer.zero_grad(set_to_none=True)
    outputs = adapter(
        batch.current_images,
        batch.calibs,
        batch.targets,
        batch.image_sizes,
        **batch.detector_kwargs(),
    )
    losses = cast(Mapping[str, torch.Tensor], criterion(outputs, batch.targets))
    weight_dict = cast(Mapping[str, float], criterion.weight_dict)  # type: ignore[attr-defined]
    total = sum(losses[key] * weight_dict[key] for key in losses if key in weight_dict)
    if not isinstance(total, torch.Tensor) or not torch.isfinite(total):
        raise ValueError("Temporal weighted loss is absent or nonfinite")
    total.backward()
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), gradient_clip_norm)
    optimizer.step()
    return {
        "loss_total": float(total.detach()),
        **{key: float(value.detach()) for key, value in losses.items()},
    }


def temporal_eval_forward(
    adapter: TemporalDetectorAdapter, batch: TemporalCampaignBatch
) -> dict[str, Any]:
    """Run label-free model inference; GT must be produced by a separate reader."""
    if batch.targets is not None:
        raise ValueError("Temporal evaluation forward must not receive targets")
    with torch.no_grad():
        return adapter(
            batch.current_images,
            batch.calibs,
            None,
            batch.image_sizes,
            **batch.detector_kwargs(),
        )


def validate_evaluation_commit(
    update: int,
    *,
    monitor_complete: bool,
    full_selection_complete: bool,
    config: TemporalTrainingConfig,
) -> dict[str, bool]:
    """Permit a checkpoint only after every evaluation due at its update committed."""
    monitor_required = update in config.baseline.evaluation.updates
    full_required = update in config.baseline.evaluation.full_selection_updates
    if monitor_required and not monitor_complete:
        raise ValueError("Scheduled monitor evaluation is incomplete")
    if full_required and not full_selection_complete:
        raise ValueError("Scheduled full-selection evaluation is incomplete")
    if full_selection_complete and not monitor_complete:
        raise ValueError("Full-selection completion cannot precede monitor completion")
    return {
        "monitor_required": monitor_required,
        "monitor_complete": monitor_complete,
        "full_selection_required": full_required,
        "full_selection_complete": full_selection_complete,
    }


def save_temporal_checkpoint(
    path: str | Path,
    *,
    adapter: TemporalDetectorAdapter,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateLRScheduler,
    sampler: TemporalCampaignSampler,
    config: TemporalTrainingConfig,
    runtime_provenance: Mapping[str, object],
    update: int,
    best_metric: float | None = None,
    monitor_complete: bool = False,
    full_selection_complete: bool = False,
    elapsed_seconds: float = 0.0,
    selected_lineage: Mapping[str, object] | None = None,
) -> str:
    """Atomically save every state required for an exact temporal continuation."""
    evaluation_commit = validate_evaluation_commit(
        update,
        monitor_complete=monitor_complete,
        full_selection_complete=full_selection_complete,
        config=config,
    )
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    payload = {
        "schema": TEMPORAL_CHECKPOINT_SCHEMA,
        "producer": "scripts/train_detection_temporal_campaign.py",
        "config_sha256": config.sha256,
        "runtime_provenance": dict(runtime_provenance),
        "runtime_provenance_sha256": canonical_sha256(runtime_provenance),
        "parent_checkpoint_sha256": config.parent.sha256,
        "variant": config.temporal.variant,
        "update": update,
        "best_metric": best_metric,
        "evaluation_commit": evaluation_commit,
        "elapsed_seconds": elapsed_seconds,
        "selected_lineage": dict(selected_lineage) if selected_lineage is not None else None,
        "adapter": adapter.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": capture_rng_state(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return sha256_file(destination)


def load_temporal_checkpoint(
    path: str | Path,
    *,
    adapter: TemporalDetectorAdapter,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateLRScheduler,
    sampler: TemporalCampaignSampler,
    config: TemporalTrainingConfig,
    runtime_provenance: Mapping[str, object],
) -> tuple[int, float | None, float, dict[str, object] | None]:
    """Restore the complete variant-bound continuation state fail-closed."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != TEMPORAL_CHECKPOINT_SCHEMA
        or payload.get("producer") != "scripts/train_detection_temporal_campaign.py"
        or payload.get("config_sha256") != config.sha256
        or payload.get("parent_checkpoint_sha256") != config.parent.sha256
        or payload.get("variant") != config.temporal.variant
        or payload.get("runtime_provenance_sha256") != canonical_sha256(runtime_provenance)
        or payload.get("runtime_provenance") != dict(runtime_provenance)
    ):
        raise ValueError("Temporal checkpoint contract differs from this run")
    commit = payload.get("evaluation_commit")
    if not isinstance(commit, Mapping):
        raise ValueError("Temporal checkpoint lacks evaluation commit state")
    expected_commit = validate_evaluation_commit(
        cast(int, payload.get("update")),
        monitor_complete=commit.get("monitor_complete") is True,
        full_selection_complete=commit.get("full_selection_complete") is True,
        config=config,
    )
    if dict(commit) != expected_commit:
        raise ValueError("Temporal checkpoint evaluation commit payload differs")
    adapter.load_state_dict(payload["adapter"], strict=True)
    criterion.load_state_dict(payload["criterion"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    sampler.load_state_dict(payload["sampler"])
    restore_rng_state(payload["rng"])
    update = payload.get("update")
    if (
        isinstance(update, bool)
        or not isinstance(update, int)
        or not 0 <= update <= config.baseline.optimization.updates
    ):
        raise ValueError("Temporal checkpoint update is invalid")
    best_metric = payload.get("best_metric")
    if best_metric is not None and (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not math.isfinite(best_metric)
    ):
        raise ValueError("Temporal checkpoint best metric is invalid")
    elapsed_seconds = payload.get("elapsed_seconds")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("Temporal checkpoint elapsed time is invalid")
    selected = payload.get("selected_lineage")
    if selected is not None:
        if not isinstance(selected, Mapping) or set(selected) != {
            "path",
            "sha256",
            "update",
            "best_metric",
        }:
            raise ValueError("Temporal checkpoint selected lineage is invalid")
        selected_path = Path(cast(str, selected["path"]))
        if (
            not selected_path.is_file()
            or sha256_file(selected_path) != selected["sha256"]
            or selected["best_metric"] != best_metric
        ):
            raise ValueError("Temporal selected checkpoint lineage differs")
    return (
        update,
        None if best_metric is None else float(best_metric),
        float(elapsed_seconds),
        None if selected is None else dict(selected),
    )
