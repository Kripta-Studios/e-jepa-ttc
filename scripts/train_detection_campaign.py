"""Run a preregistered MonoDGP S1 campaign with exact continuation state.

This command is for new campaign checkpoints. Historical pilot checkpoints must
continue to use their frozen runner and source snapshot.
"""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree is supported.

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from smoke_monodgp_portable import build_detector, prepare_source, require_finite

from e_jepa_ttc.detection.backbone_initialization import load_imagenet_resnet50_backbone
from e_jepa_ttc.detection.campaign_training import (
    CampaignConfig,
    ResumablePermutationSampler,
    UpdateLRScheduler,
    canonical_sha256,
    dataset_cache_kwargs,
    inherit_selected_checkpoint,
    load_campaign_checkpoint,
    load_campaign_config,
    optimizer_parameter_groups,
    save_campaign_checkpoint,
    sha256_file,
    validate_campaign_manifest,
    validate_manifest_against_metadata,
    validate_stop_after_update,
)
from e_jepa_ttc.detection.development_metrics import evaluate_detections
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset, RawRGBSample
from e_jepa_ttc.detection.evaluation_ground_truth import build_evaluation_ground_truth
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.evaluation_v2 import evaluate_detections_v2
from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample


def write_json(path: Path, value: object) -> None:
    """Write one finite JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_protocol(value: str) -> EvaluationProtocol:
    """Load the fixed default or a fully serialized protocol."""
    if value == "default":
        return EvaluationProtocol.default()
    return EvaluationProtocol.from_dict(json.loads(Path(value).read_text(encoding="utf-8")))


class PreparedCache:
    """Bound raw and prepared sample residency without changing preparation."""

    def __init__(
        self,
        dataset: EAPRGBDataset,
        config: CampaignConfig,
        member_sha256: dict[str, str],
        *,
        label_free_model_input: bool = False,
    ) -> None:
        self.dataset, self.config = dataset, config
        self.member_sha256 = member_sha256
        self.label_free_model_input = label_free_model_input
        self.cache: OrderedDict[int, tuple[Any, RawRGBSample]] = OrderedDict()

    def get(self, index: int) -> tuple[Any, RawRGBSample]:
        """Get one immutable raw sample and its declared model preparation."""
        if index not in self.cache:
            raw = self.dataset[index]
            expected = self.member_sha256.get(raw.sample_token)
            if expected is not None and expected != raw.member_sha256:
                raise ValueError(f"RGB member bytes changed on resume: {raw.sample_token}")
            self.member_sha256[raw.sample_token] = raw.member_sha256
            prepared = (
                prepare_model_input(raw, self.config)
                if self.label_free_model_input
                else prepare_sample(
                    raw,
                    width=self.config.model.width,
                    height=self.config.model.height,
                    include_velocity=self.config.model.predict_velocity,
                )
            )
            self.cache[index] = prepared, raw
            if len(self.cache) > self.config.data.cache_capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def prepare_model_input(raw: RawRGBSample, config: CampaignConfig) -> Any:
    """Prepare image/calibration tensors from an annotation-empty immutable view."""
    prepared = prepare_sample(
        replace(raw, annotations=[]),
        width=config.model.width,
        height=config.model.height,
        include_velocity=config.model.predict_velocity,
    )
    if prepared.native_ground_truth or prepared.target["labels"].numel():
        raise AssertionError("Label-free model preparation unexpectedly produced targets")
    return prepared


def open_campaign_dataset(
    root: str | Path, tokens: tuple[str, ...] | list[str], config: CampaignConfig
) -> EAPRGBDataset:
    """Construct every campaign reader with the declared bounded label cache."""
    settings = dataset_cache_kwargs(config)
    return EAPRGBDataset(root, tokens, max_label_sequences=settings["max_label_sequences"])


def _photometric(images: torch.Tensor, config: CampaignConfig) -> torch.Tensor:
    """Apply declared photometric jitter in RGB space using the Torch RNG."""
    augmentation = config.augmentation
    if not any((augmentation.brightness, augmentation.contrast, augmentation.saturation)):
        return images
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = (images * std + mean).clamp(0, 1)

    def factors(magnitude: float) -> torch.Tensor:
        return 1 + (torch.rand(len(rgb), 1, 1, 1, device=rgb.device) * 2 - 1) * magnitude

    if augmentation.brightness:
        rgb = rgb * factors(augmentation.brightness)
    if augmentation.contrast:
        gray = rgb.mean(dim=(1, 2, 3), keepdim=True)
        rgb = gray + factors(augmentation.contrast) * (rgb - gray)
    if augmentation.saturation:
        gray = rgb.mean(dim=1, keepdim=True)
        rgb = gray + factors(augmentation.saturation) * (rgb - gray)
    return (rgb.clamp(0, 1) - mean) / std


def batch_to_cuda(
    rows: list[tuple[Any, RawRGBSample]], config: CampaignConfig, *, training: bool = False
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]], torch.Tensor]:
    """Transfer one contiguous batch with separately variable targets."""
    prepared = [row[0] for row in rows]
    images = torch.stack([item.image for item in prepared]).contiguous().cuda()
    return (
        _photometric(images, config) if training else images,
        torch.stack([item.calibration for item in prepared]).cuda(),
        [{key: value.cuda() for key, value in item.target.items()} for item in prepared],
        torch.stack([item.image_size for item in prepared]).cuda(),
    )


def model_inputs_to_cuda(prepared: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transfer only tensors consumed by a target-free detector forward pass."""
    return (
        prepared.image.unsqueeze(0).contiguous().cuda(),
        prepared.calibration.unsqueeze(0).cuda(),
        prepared.image_size.unsqueeze(0).cuda(),
    )


def legacy_ground_truth(
    raw: RawRGBSample, config: CampaignConfig
) -> tuple[list[Any] | None, str | None]:
    """Build the historical filtered GT without coupling it to v2 model input."""
    try:
        prepared = prepare_sample(
            raw,
            width=config.model.width,
            height=config.model.height,
            include_velocity=config.model.predict_velocity,
        )
    except ValueError as error:
        if "targets exceed" not in str(error):
            raise
        return None, str(error)
    return prepared.native_ground_truth, None


def evaluate(
    model: torch.nn.Module,
    cache: PreparedCache,
    protocol: EvaluationProtocol,
    config: CampaignConfig,
    output: Path,
) -> dict[str, object]:
    """Run label-free inference, then score raw detector-independent GT."""
    if not cache.label_free_model_input:
        raise ValueError("Evaluation requires an annotation-empty model-input cache")
    model.eval()
    predictions: dict[str, list[dict[str, object]]] = {}
    raw_truth = {}
    legacy_truth = {}
    legacy_unavailable = {}
    decode_counts = {}
    timings = []
    with torch.no_grad():
        for index in range(len(cache.dataset)):
            prepared, raw = cache.get(index)
            images, calibration, image_sizes = model_inputs_to_cuda(prepared)
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(images, calibration, None, image_sizes)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            require_finite(outputs)
            predictions[raw.sample_token], decode_counts[raw.sample_token] = decode_eap(
                outputs,
                0,
                prepared.calibration,
                raw.event_from_ego,
                prepared.image_size,
            )
            raw_truth[raw.sample_token] = build_evaluation_ground_truth(
                raw.annotations,
                intrinsic=raw.intrinsic,
                event_from_ego=raw.event_from_ego,
                image_wh=(raw.image.shape[1], raw.image.shape[0]),
                protocol=protocol,
            )
            if config.evaluation.legacy_metrics:
                historical, reason = legacy_ground_truth(raw, config)
                if reason is not None:
                    legacy_unavailable[raw.sample_token] = reason
                elif historical is not None:
                    legacy_truth[raw.sample_token] = historical
    result = evaluate_detections_v2(predictions, raw_truth, protocol=protocol)
    result["inference_seconds_median"] = statistics.median(timings)
    result["inference_seconds_total"] = sum(timings)
    payload: dict[str, object] = {
        "metrics_v2": result,
        "predictions": predictions,
        "ground_truth_v2": {token: item.to_dict() for token, item in raw_truth.items()},
        "decode_counts": decode_counts,
    }
    if config.evaluation.legacy_metrics:
        payload["legacy_metrics_separate"] = (
            {
                "status": "UNAVAILABLE",
                "reason": "historical target capacity exceeded; v2 evaluation remains valid",
                "tokens": legacy_unavailable,
            }
            if legacy_unavailable
            else evaluate_detections(predictions, legacy_truth)
        )
    write_json(output, payload)
    return result


def _metric(result: dict[str, object], name: str) -> float:
    value = result.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"Selection metric {name!r} is absent or nonfinite")
    return float(value)


def commit_evaluation_artifacts(
    *,
    run_full_evaluation: Callable[[], None],
    publish_selected: Callable[[], None] | None,
    publish_last: Callable[[], None],
) -> None:
    """Publish no checkpoint until every due evaluation has completed."""
    run_full_evaluation()
    if publish_selected is not None:
        publish_selected()
    publish_last()


def should_publish_periodic_checkpoint(
    update: int,
    *,
    checkpoint_interval: int,
    evaluation_updates: tuple[int, ...],
    stop_after_update: int | None,
) -> bool:
    """Exclude evaluation and pending-stop updates from periodic publication."""
    return (
        update % checkpoint_interval == 0
        and update not in evaluation_updates
        and update != stop_after_update
    )


def _snapshot(output: Path, paths: list[Path]) -> dict[str, str]:
    receipt = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = output / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        receipt[relative.as_posix()] = sha256_file(destination)
    return receipt


def snapshot_media_admission(output: Path, source: Path) -> dict[str, str]:
    """Freeze an external admission receipt at one location-independent code path."""
    if not source.is_file():
        raise FileNotFoundError(f"Media-admission receipt is missing: {source}")
    relative = Path("data/admission/media_admission.json")
    destination = output / "code" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Snapshot destination already exists: {destination}")
    shutil.copy2(source, destination)
    return {relative.as_posix(): sha256_file(destination)}


def _snapshot_portable_source(source: Path, output: Path) -> dict[str, str]:
    """Freeze and hash the complete patched third-party runtime."""
    destination = output / "code/portable_source"
    shutil.copytree(source, destination)
    return {
        path.relative_to(destination).as_posix(): sha256_file(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def resource_admission(config: CampaignConfig) -> dict[str, object]:
    """Refuse model construction when current host headroom is below the fixed gate."""
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        available_ram, total_ram = status.available_physical, status.total_physical
    else:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_ram = os.sysconf("SC_AVPHYS_PAGES") * page_size
        total_ram = os.sysconf("SC_PHYS_PAGES") * page_size
    free_vram, total_vram = torch.cuda.mem_get_info()
    gib = 1024**3
    minimum_ram = int(config.resources.minimum_available_ram_gb * gib)
    minimum_vram = int(config.resources.minimum_available_vram_gb * gib)
    receipt: dict[str, object] = {
        "available_ram_bytes": available_ram,
        "total_ram_bytes": total_ram,
        "available_vram_bytes": free_vram,
        "total_vram_bytes": total_vram,
        "minimum_available_ram_bytes": minimum_ram,
        "minimum_available_vram_bytes": minimum_vram,
    }
    if available_ram < minimum_ram:
        raise RuntimeError("Available RAM is below the preregistered campaign admission gate")
    if free_vram < minimum_vram:
        raise RuntimeError("Available VRAM is below the preregistered campaign admission gate")
    return receipt | {"status": "PASS"}


def run(
    config_path: Path,
    output: Path,
    resume: Path | None,
    stop_after_update: int | None = None,
    *,
    dataset_factory: (
        Callable[
            [str | Path, list[str] | tuple[str, ...], CampaignConfig], EAPRGBDataset
        ]
        | None
    ) = None,
    runtime_extension: Mapping[str, object] | None = None,
    extra_owned_paths: tuple[Path, ...] = (),
    performance_metrics: Callable[[], Mapping[str, object]] | None = None,
    performance_observer: Callable[[str, float], None] | None = None,
    stop_commit_validator: Callable[..., bool] | None = None,
) -> dict[str, object]:
    """Preflight, train, evaluate and preserve a complete campaign receipt."""
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    config = load_campaign_config(config_path)
    manifest = validate_campaign_manifest(config)
    metadata_receipt = validate_manifest_against_metadata(manifest, config)
    media_admission = dict(metadata_receipt["media_admission"])
    expected_member_sha256 = dict(media_admission.pop("data_member_sha256"))
    metadata_receipt = {**metadata_receipt, "media_admission": media_admission}
    full_selection_path = Path(config.data.full_selection_tokens)
    if sha256_file(full_selection_path) != config.data.expected_full_selection_sha256:
        raise ValueError("Full selection token artifact differs from configured SHA-256")
    full_selection = json.loads(full_selection_path.read_text(encoding="utf-8"))
    if (
        full_selection.get("schema") != "eap_detection_full_selection_v1"
        or full_selection.get("dataset_revision") != config.data.expected_dataset_revision
        or full_selection.get("candidate_split_sha256") != config.data.expected_split_sha256
        or full_selection.get("role") != config.data.selection_role
        or set(full_selection.get("sequence_ids", ()))
        != set(manifest["roles"][config.data.selection_role]["sequence_ids"])
        or full_selection.get("source_media_sha256") != manifest["source_media_sha256"]
    ):
        raise ValueError("Full selection artifact differs from the campaign data contract")
    full_selection_tokens = tuple(full_selection.get("sample_tokens", ()))
    if not full_selection_tokens or len(set(full_selection_tokens)) != len(full_selection_tokens):
        raise ValueError("Full selection tokens must be nonempty and unique")
    if sha256_file(Path(config.model.backbone_weights)) != config.model.backbone_weights_sha256:
        raise ValueError("Backbone weights differ from the declared SHA-256")
    protocol = load_protocol(config.evaluation.protocol)
    split_metadata = json.loads(Path(config.data.split).read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    report: dict[str, object] = {
        "schema": "eap_detection_campaign_run_v1",
        "status": "RUNNING",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "config": config_path.as_posix(),
        "config_sha256": config.sha256,
        "run_name": output.name,
        "seed": config.seed,
        "host": platform.node(),
        "dataset_manifest_hash": config.data.expected_manifest_sha256,
        "split_version": split_metadata.get("schema"),
        "split_hash": config.data.expected_split_sha256,
        "evaluation_protocol": protocol.to_dict(),
        "evaluation_protocol_sha256": protocol.sha256,
        "metadata_admission": metadata_receipt,
        "full_selection_sha256": config.data.expected_full_selection_sha256,
        "full_selection_count": len(full_selection_tokens),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    write_json(output / "config_resolved.json", asdict(config) | {"sha256": config.sha256})
    if not torch.cuda.is_available():
        raise RuntimeError("The campaign runner requires a CUDA device")
    report["device"] = torch.cuda.get_device_name()
    report["resource_admission"] = resource_admission(config)
    torch.set_num_threads(config.resources.num_cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision(config.optimization.matmul_precision)
    source = prepare_source(
        Path(config.model.source_root),
        Path(config.model.portable_root),
        output / "upstream.patch",
        predict_velocity=config.model.predict_velocity,
    )
    model, criterion, effective_model = build_detector(
        Path(config.model.portable_root),
        config.model.group_num,
        predict_velocity=config.model.predict_velocity,
        velocity_loss_coef=config.model.velocity_loss_coef,
    )
    expected_model_settings = {
        "backbone": config.model.backbone,
        "num_classes": config.model.num_classes,
        "num_queries": config.model.queries,
        "group_num": config.model.group_num,
        "predict_velocity": config.model.predict_velocity,
        "velocity_loss_coef": config.model.velocity_loss_coef,
        "device": "cuda",
    }
    if any(effective_model.get(key) != value for key, value in expected_model_settings.items()):
        raise ValueError("Effective MonoDGP settings differ from the declared campaign model")
    groups, ledger = optimizer_parameter_groups(model, config.optimization)
    optimizer = torch.optim.AdamW(groups)
    scheduler = UpdateLRScheduler(
        optimizer,
        total_updates=config.optimization.updates,
        kind=config.optimization.scheduler,
        warmup_fraction=config.optimization.warmup_fraction,
        minimum_ratio=config.optimization.minimum_lr_ratio,
    )
    train_tokens = manifest["roles"][config.data.train_role]["sample_tokens"]
    selection_tokens = manifest["roles"][config.data.selection_role]["sample_tokens"]
    pilot_tokens = manifest["historical_panels"]["pilot_selection256"]["sample_tokens"]
    sampler = ResumablePermutationSampler(len(train_tokens), config.seed)
    owned_paths = [
        Path(__file__),
        ROOT / "data/manifests/eap_detection_public_files_v1.json",
        ROOT / "scripts/infer_detection_campaign.py",
        ROOT / "scripts/smoke_monodgp_portable.py",
        ROOT / "src/e_jepa_ttc/__init__.py",
        ROOT / "src/e_jepa_ttc/detection/__init__.py",
        ROOT / "src/e_jepa_ttc/detection/campaign_training.py",
        ROOT / "src/e_jepa_ttc/detection/backbone_initialization.py",
        ROOT / "src/e_jepa_ttc/detection/development_metrics.py",
        ROOT / "src/e_jepa_ttc/detection/diagnostic_export.py",
        ROOT / "src/e_jepa_ttc/detection/eap_rgb_data.py",
        ROOT / "src/e_jepa_ttc/detection/evaluation_ap_fast.py",
        ROOT / "src/e_jepa_ttc/detection/evaluation_protocol.py",
        ROOT / "src/e_jepa_ttc/detection/evaluation_ground_truth.py",
        ROOT / "src/e_jepa_ttc/detection/evaluation_v2.py",
        ROOT / "src/e_jepa_ttc/detection/geometry_contract.py",
        ROOT / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py",
        ROOT / "src/e_jepa_ttc/detection/portable_attention.py",
        ROOT / "src/e_jepa_ttc/detection/velocity_extension.py",
    ] + list(extra_owned_paths)
    code_sha256 = _snapshot(output, owned_paths)
    code_sha256.update(snapshot_media_admission(output, Path(media_admission["receipt_path"])))
    portable_source_sha256 = _snapshot_portable_source(Path(config.model.portable_root), output)
    runtime_preregistration = {
        "schema": "eap_detection_campaign_runtime_preregistration_v1",
        "config": asdict(config),
        "config_sha256": config.sha256,
        "protocol_sha256": protocol.sha256,
        "source_receipt": source,
        "source_receipt_sha256": canonical_sha256(source),
        "effective_model": effective_model,
        "effective_model_sha256": canonical_sha256(effective_model),
        "code_sha256": code_sha256,
        "code_sha256_manifest": canonical_sha256(code_sha256),
        "portable_source_sha256": portable_source_sha256,
        "portable_source_sha256_manifest": canonical_sha256(portable_source_sha256),
        "dataset_media_sha256": manifest["source_media_sha256"],
        "media_admission_receipt_sha256": media_admission["receipt_sha256"],
        "media_admission_population_sha256": media_admission["train_token_population_sha256"],
        "matmul_precision": config.optimization.matmul_precision,
        "runtime_extension": dict(runtime_extension or {}),
    }
    write_json(output / "runtime_preregistration.json", runtime_preregistration)
    runtime_provenance = {
        "runtime_preregistration_sha256": sha256_file(output / "runtime_preregistration.json"),
        "source_receipt_sha256": runtime_preregistration["source_receipt_sha256"],
        "effective_model_sha256": runtime_preregistration["effective_model_sha256"],
        "code_sha256_manifest": runtime_preregistration["code_sha256_manifest"],
        "portable_source_sha256_manifest": runtime_preregistration[
            "portable_source_sha256_manifest"
        ],
        "matmul_precision": config.optimization.matmul_precision,
    }
    start_update, best_metric = 0, None
    data_member_sha256: dict[str, str] = dict(expected_member_sha256)
    ancestor_progress: dict[str, int | float] = {
        "training_seconds": 0.0,
        "empty_target_exposures": 0,
        "valid_velocity_object_exposures": 0,
    }
    initialization = None
    if resume is None:
        initialization = load_imagenet_resnet50_backbone(
            model, Path(config.model.backbone_weights), config.model.backbone_weights_sha256
        )
    else:
        start_update, best_metric, ancestor_progress, resumed_member_sha256 = (
            load_campaign_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                config=config,
                protocol_sha256=protocol.sha256,
                runtime_provenance=runtime_provenance,
            )
        )
        if resumed_member_sha256 != expected_member_sha256:
            raise ValueError("Resume RGB member ledger differs from media admission")
        data_member_sha256 = resumed_member_sha256
    selected_lineage = (
        inherit_selected_checkpoint(
            resume,
            output / "checkpoints/selected.pt",
            config=config,
            protocol_sha256=protocol.sha256,
            runtime_provenance=runtime_provenance,
            best_metric=best_metric,
        )
        if resume is not None
        else None
    )
    stop_after_update = validate_stop_after_update(
        stop_after_update,
        start_update=start_update,
        maximum_update=config.optimization.updates,
    )
    parent_run_chain: list[dict[str, object]] = []
    if resume is not None:
        parent_summary_path = resume.parent.parent / "summary.json"
        if parent_summary_path.is_file():
            parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
            parent_run_chain.extend(parent_summary.get("parent_run_chain", []))
            parent_run_chain.append(
                {
                    "summary_path": str(parent_summary_path.resolve()),
                    "summary_sha256": sha256_file(parent_summary_path),
                    "status": parent_summary.get("status"),
                    "final_update": parent_summary.get("final_update"),
                    "selected_checkpoint": parent_summary.get("selected_checkpoint"),
                }
            )
        else:
            parent_run_chain.append(
                {
                    "summary_path": None,
                    "summary_sha256": None,
                    "status": "PARENT_SUMMARY_UNAVAILABLE_INCOMPLETE_RUN",
                    "resume_checkpoint": str(resume.resolve()),
                }
            )
    report.update(
        source=source,
        effective_model=effective_model,
        parameter_ledger=ledger,
        initialization=initialization,
        resumed_from_sha256=sha256_file(resume) if resume else None,
        start_update=start_update,
        code_sha256=code_sha256,
        portable_source_sha256_manifest=runtime_preregistration["portable_source_sha256_manifest"],
        runtime_provenance=runtime_provenance,
        parent_run=(
            {
                "resume_checkpoint": str(resume.resolve()),
                "resume_checkpoint_sha256": sha256_file(resume),
                "selected_checkpoint": selected_lineage,
                "ancestor_progress": ancestor_progress,
            }
            if resume is not None
            else None
        ),
        parent_run_chain=parent_run_chain,
    )
    write_json(output / "preregistration.json", report)
    make_dataset = dataset_factory or open_campaign_dataset
    train_dataset = make_dataset(config.data.root, train_tokens, config)
    selection_dataset = make_dataset(config.data.root, selection_tokens, config)
    pilot_dataset = make_dataset(config.data.root, pilot_tokens, config)
    train_cache = PreparedCache(train_dataset, config, data_member_sha256)
    selection_cache = PreparedCache(
        selection_dataset, config, data_member_sha256, label_free_model_input=True
    )
    pilot_cache = PreparedCache(
        pilot_dataset, config, data_member_sha256, label_free_model_input=True
    )
    curve = output / "metrics.jsonl"
    started = time.perf_counter()
    step_times: list[float] = []
    preprocessing_times: list[float] = []
    final_update = start_update
    empty_target_exposures = int(ancestor_progress["empty_target_exposures"])
    valid_velocity_object_exposures = int(ancestor_progress["valid_velocity_object_exposures"])
    ancestor_training_seconds = float(ancestor_progress["training_seconds"])
    ancestor_empty_target_exposures = empty_target_exposures
    ancestor_valid_velocity_object_exposures = valid_velocity_object_exposures
    completed_evaluation_updates: set[int] = set()
    torch.cuda.reset_peak_memory_stats()
    with curve.open("w", encoding="utf-8") as stream:
        for update in range(start_update + 1, config.optimization.updates + 1):
            if (
                ancestor_training_seconds + time.perf_counter() - started
                >= config.resources.maximum_seconds
            ):
                report["stop_reason"] = "maximum_seconds"
                break
            indices = sampler.next_batch(config.optimization.batch_size)
            before_prepare = time.perf_counter()
            rows = [train_cache.get(index) for index in indices]
            preprocessing_times.append(time.perf_counter() - before_prepare)
            if performance_observer:
                performance_observer("prepare", preprocessing_times[-1])
            before_h2d = time.perf_counter()
            images, calibrations, targets, sizes = batch_to_cuda(rows, config, training=True)
            if performance_observer:
                torch.cuda.synchronize()
                performance_observer("h2d", time.perf_counter() - before_h2d)
            learning_rates = scheduler.apply(update)
            model.train()
            criterion.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            before_step = time.perf_counter()
            outputs = model(images, calibrations, targets, sizes)
            require_finite(outputs)
            losses = criterion(outputs, targets)
            require_finite(losses, "training_losses")
            weighted_losses = [
                value * criterion.weight_dict[key]
                for key, value in losses.items()
                if key in criterion.weight_dict
            ]
            if not weighted_losses:
                raise ValueError("Criterion produced no weighted training losses")
            loss = weighted_losses[0]
            for term in weighted_losses[1:]:
                loss = loss + term
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.optimization.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - before_step
            step_times.append(step_seconds)
            if performance_observer:
                performance_observer("compute", step_seconds)
            final_update = update
            batch_empty_targets = sum(len(target["labels"]) == 0 for target in targets)
            batch_valid_velocity = (
                sum(int(target["velocity_valid"].sum().item()) for target in targets)
                if config.model.predict_velocity
                else 0
            )
            empty_target_exposures += batch_empty_targets
            valid_velocity_object_exposures += batch_valid_velocity
            record = {
                "update": update,
                "epoch": sampler.epoch,
                "sample_tokens": [rows[index][1].sample_token for index in range(len(rows))],
                "loss": float(loss.detach()),
                "loss_center": float(losses["loss_center"].detach()),
                "losses": {key: float(value.detach()) for key, value in losses.items()},
                "empty_target_exposures": batch_empty_targets,
                "valid_velocity_object_exposures": batch_valid_velocity,
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rates": learning_rates,
                "preprocessing_seconds": preprocessing_times[-1],
                "forward_backward_update_seconds": step_seconds,
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            del images, calibrations, targets, sizes, outputs, losses, loss
            if should_publish_periodic_checkpoint(
                update,
                checkpoint_interval=config.optimization.checkpoint_interval,
                evaluation_updates=config.evaluation.updates,
                stop_after_update=stop_after_update,
            ):
                save_campaign_checkpoint(
                    output / "checkpoints/last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    config=config,
                    protocol_sha256=protocol.sha256,
                    runtime_provenance=runtime_provenance,
                    update=update,
                    best_metric=best_metric,
                    progress={
                        "training_seconds": ancestor_training_seconds
                        + time.perf_counter()
                        - started,
                        "empty_target_exposures": empty_target_exposures,
                        "valid_velocity_object_exposures": valid_velocity_object_exposures,
                    },
                    data_member_sha256=data_member_sha256,
                )
            if update in config.evaluation.updates:
                before_evaluation = time.perf_counter()
                result = evaluate(
                    model,
                    selection_cache,
                    protocol,
                    config,
                    output / f"evaluation/selection_update_{update:06d}.json",
                )
                value = _metric(result, config.evaluation.selection_metric)
                evaluate(
                    model,
                    pilot_cache,
                    protocol,
                    config,
                    output / f"evaluation/pilot_parity_update_{update:06d}.json",
                )
                improves = best_metric is None or value > best_metric
                committed_metric = value if improves else best_metric

                def publish_checkpoint(
                    path: Path,
                    *,
                    transaction_update: int = update,
                    transaction_metric: float | None = committed_metric,
                    transaction_empty: int = empty_target_exposures,
                    transaction_velocity: int = valid_velocity_object_exposures,
                ) -> None:
                    save_campaign_checkpoint(
                        path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler=sampler,
                        config=config,
                        protocol_sha256=protocol.sha256,
                        runtime_provenance=runtime_provenance,
                        update=transaction_update,
                        best_metric=transaction_metric,
                        progress={
                            "training_seconds": ancestor_training_seconds
                            + time.perf_counter()
                            - started,
                            "empty_target_exposures": transaction_empty,
                            "valid_velocity_object_exposures": transaction_velocity,
                        },
                        data_member_sha256=data_member_sha256,
                    )

                def run_full_evaluation(transaction_update: int = update) -> None:
                    if transaction_update in config.evaluation.full_selection_updates:
                        with make_dataset(
                            config.data.root, list(full_selection_tokens), config
                        ) as full_selection_dataset:
                            evaluate(
                                model,
                                PreparedCache(
                                    full_selection_dataset,
                                    config,
                                    data_member_sha256,
                                    label_free_model_input=True,
                                ),
                                protocol,
                                config,
                                output
                                / f"evaluation/full_selection_update_{transaction_update:06d}.json",
                            )

                commit_evaluation_artifacts(
                    run_full_evaluation=run_full_evaluation,
                    publish_selected=(
                        (lambda: publish_checkpoint(output / "checkpoints/selected.pt"))
                        if improves
                        else None
                    ),
                    publish_last=lambda: publish_checkpoint(output / "checkpoints/last.pt"),
                )
                best_metric = committed_metric
                if performance_observer:
                    performance_observer("evaluation", time.perf_counter() - before_evaluation)
                completed_evaluation_updates.add(update)
            if stop_after_update is not None and update == stop_after_update:
                report["external_execution_stop_requested"] = stop_after_update
                report["stop_reason"] = "external_execution_stop_after_completed_update"
                break
    if final_update > start_update and final_update not in config.evaluation.updates:
        evaluate(
            model,
            selection_cache,
            protocol,
            config,
            output / f"evaluation/stop_diagnostic_update_{final_update:06d}.json",
        )
        report["unscheduled_stop_evaluation"] = {
            "update": final_update,
            "selection_used": False,
            "reason": report.get("stop_reason", "unscheduled_completion"),
        }
        completed_evaluation_updates.add(final_update)
    save_campaign_checkpoint(
        output / "checkpoints/last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
        protocol_sha256=protocol.sha256,
        runtime_provenance=runtime_provenance,
        update=final_update,
        best_metric=best_metric,
        progress={
            "training_seconds": ancestor_training_seconds + time.perf_counter() - started,
            "empty_target_exposures": empty_target_exposures,
            "valid_velocity_object_exposures": valid_velocity_object_exposures,
        },
        data_member_sha256=data_member_sha256,
    )
    if stop_after_update is not None and final_update == stop_after_update:
        if stop_commit_validator is not None and not stop_commit_validator(
            final_update,
            stop_after_update,
            required_evaluation_complete=(final_update in completed_evaluation_updates),
            checkpoint_durable=(output / "checkpoints/last.pt").is_file(),
        ):
            raise AssertionError("External stop transaction was not committed")
        report["external_execution_stop_transaction"] = "COMMITTED_AFTER_REQUIRED_EVALUATION"
    selected_path = output / "checkpoints/selected.pt"
    selected_checkpoint = (
        {
            "status": "AVAILABLE",
            "path": str(selected_path),
            "sha256": sha256_file(selected_path),
            "best_metric": best_metric,
        }
        if selected_path.is_file()
        else {
            "status": "NOT_AVAILABLE_NO_SCHEDULED_SELECTION",
            "path": None,
            "sha256": None,
            "best_metric": best_metric,
        }
    )
    report.update(
        status="COMPLETED" if final_update == config.optimization.updates else "BUDGET_STOPPED",
        final_update=final_update,
        best_selection_metric=best_metric,
        stage_empty_target_exposures=(empty_target_exposures - ancestor_empty_target_exposures),
        cumulative_empty_target_exposures=empty_target_exposures,
        stage_valid_velocity_object_exposures=(
            valid_velocity_object_exposures - ancestor_valid_velocity_object_exposures
        ),
        cumulative_valid_velocity_object_exposures=valid_velocity_object_exposures,
        selected_checkpoint=selected_checkpoint,
        stage_training_seconds=time.perf_counter() - started,
        cumulative_training_seconds=ancestor_training_seconds + time.perf_counter() - started,
        stage_image_exposures=(final_update - start_update) * config.optimization.batch_size,
        cumulative_image_exposures=final_update * config.optimization.batch_size,
        step_seconds_median=statistics.median(step_times) if step_times else None,
        preprocessing_seconds_median=(
            statistics.median(preprocessing_times) if preprocessing_times else None
        ),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        last_checkpoint_sha256=sha256_file(output / "checkpoints/last.pt"),
        checkpoint_path=str(output / "checkpoints/last.pt"),
        metrics_path=str(output / "summary.json"),
        ended_at_utc=datetime.now(UTC).isoformat(),
        performance_metrics=(dict(performance_metrics()) if performance_metrics else None),
    )
    write_json(output / "summary.json", report)
    train_dataset.close()
    selection_dataset.close()
    pilot_dataset.close()
    return report


def main() -> int:
    """CLI with a durable failure receipt when an output directory was created."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args()
    output_existed = args.output.exists()
    try:
        report = run(args.config, args.output, args.resume, args.stop_after_update)
        print(json.dumps({"status": report["status"], "final_update": report["final_update"]}))
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level receipt must preserve unexpected failures
        traceback.print_exc()
        if args.output.exists() and not output_existed:
            write_json(
                args.output / "failure.json",
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at_utc": datetime.now(UTC).isoformat(),
                },
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
