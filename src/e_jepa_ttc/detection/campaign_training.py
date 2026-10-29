"""Strict, resumable training contracts for the eAP detection campaign.

The historical pilot runner remains the reproduction path for its checkpoints.
This module provides the state and configuration machinery for new S1 runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import torch
import yaml

from e_jepa_ttc.detection.eap_rgb_data import safe_dataset_path, safe_relative_path
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol

CONFIG_SCHEMA = "eap_detection_campaign_config_v1"
MANIFEST_SCHEMA = "eap_detection_campaign_manifest_v1"
CHECKPOINT_SCHEMA = "eap_detection_campaign_checkpoint_v1"

T = TypeVar("T")


def validate_stop_after_update(
    value: int | None, *, start_update: int, maximum_update: int
) -> int | None:
    """Validate an execution segment without changing the nominal training budget."""
    if value is None:
        return None
    stop = _integer(value, "stop_after_update", minimum=1)
    if stop <= start_update or stop > maximum_update:
        raise ValueError("stop_after_update must be after resume and within nominal budget")
    return stop


def canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value using a canonical representation."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return dict(value)


def _exact(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    result = _mapping(value, name)
    unknown, missing = set(result) - keys, keys - set(result)
    if unknown or missing:
        raise ValueError(
            f"{name} keys differ: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, lower: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= lower if strict else result < lower):
        operator = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {operator} {lower}")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class DataConfig:
    root: str
    manifest: str
    split: str
    train_role: str
    selection_role: str
    expected_dataset_revision: str
    expected_manifest_sha256: str
    expected_split_sha256: str
    full_selection_tokens: str
    expected_full_selection_sha256: str
    cache_capacity: int
    label_cache_sequences: int


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    num_classes: int
    queries: int
    group_num: int
    predict_velocity: bool
    velocity_loss_coef: float
    width: int
    height: int
    backbone_weights: str
    backbone_weights_sha256: str
    source_root: str
    portable_root: str


@dataclass(frozen=True)
class OptimizationConfig:
    updates: int
    checkpoint_interval: int
    batch_size: int
    precision: str
    matmul_precision: str
    learning_rate: float
    backbone_lr_multiplier: float
    weight_decay: float
    gradient_clip_norm: float
    scheduler: str
    warmup_fraction: float
    minimum_lr_ratio: float


@dataclass(frozen=True)
class AugmentationConfig:
    brightness: float
    contrast: float
    saturation: float


@dataclass(frozen=True)
class EvaluationConfig:
    protocol: str
    updates: tuple[int, ...]
    full_selection_updates: tuple[int, ...]
    selection_metric: str
    selection_mode: str
    legacy_metrics: bool


@dataclass(frozen=True)
class ResourceConfig:
    maximum_seconds: float
    num_cpu_threads: int
    minimum_available_ram_gb: float
    minimum_available_vram_gb: float


@dataclass(frozen=True)
class CampaignConfig:
    schema: str
    experiment_id: str
    seed: int
    data: DataConfig
    model: ModelConfig
    optimization: OptimizationConfig
    augmentation: AugmentationConfig
    evaluation: EvaluationConfig
    resources: ResourceConfig

    @property
    def sha256(self) -> str:
        """Return the hash of all effective settings."""
        return canonical_sha256(asdict(self))


def load_campaign_config(path: Path) -> CampaignConfig:
    """Load and strictly validate every supported campaign setting."""
    raw = _exact(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "config",
        {
            "schema",
            "experiment_id",
            "seed",
            "data",
            "model",
            "optimization",
            "augmentation",
            "evaluation",
            "resources",
        },
    )
    if raw["schema"] != CONFIG_SCHEMA:
        raise ValueError(f"schema must be {CONFIG_SCHEMA}")
    data = _exact(
        raw["data"],
        "data",
        {
            "root",
            "manifest",
            "split",
            "train_role",
            "selection_role",
            "expected_dataset_revision",
            "expected_manifest_sha256",
            "expected_split_sha256",
            "full_selection_tokens",
            "expected_full_selection_sha256",
            "cache_capacity",
            "label_cache_sequences",
        },
    )
    model = _exact(
        raw["model"],
        "model",
        {
            "backbone",
            "num_classes",
            "queries",
            "group_num",
            "predict_velocity",
            "velocity_loss_coef",
            "width",
            "height",
            "backbone_weights",
            "backbone_weights_sha256",
            "source_root",
            "portable_root",
        },
    )
    optimization = _exact(
        raw["optimization"],
        "optimization",
        {
            "updates",
            "checkpoint_interval",
            "batch_size",
            "precision",
            "matmul_precision",
            "learning_rate",
            "backbone_lr_multiplier",
            "weight_decay",
            "gradient_clip_norm",
            "scheduler",
            "warmup_fraction",
            "minimum_lr_ratio",
        },
    )
    augmentation = _exact(
        raw["augmentation"],
        "augmentation",
        {"brightness", "contrast", "saturation"},
    )
    evaluation = _exact(
        raw["evaluation"],
        "evaluation",
        {
            "protocol",
            "updates",
            "full_selection_updates",
            "selection_metric",
            "selection_mode",
            "legacy_metrics",
        },
    )
    resources = _exact(
        raw["resources"],
        "resources",
        {
            "maximum_seconds",
            "num_cpu_threads",
            "minimum_available_ram_gb",
            "minimum_available_vram_gb",
        },
    )
    if model["backbone"] != "resnet50" or model["num_classes"] != 8:
        raise ValueError("S1 screening supports exactly ResNet-50 and eight classes")
    if model["queries"] != 50 or model["group_num"] != 11:
        raise ValueError("S1 screening supports exactly 50 queries and 11 groups")
    if not isinstance(model["predict_velocity"], bool):
        raise ValueError("model.predict_velocity must be boolean")
    if any(_integer(model[key], f"model.{key}", minimum=1) % 64 for key in ("width", "height")):
        raise ValueError("model width and height must be divisible by 64")
    if optimization["precision"] != "fp32":
        raise ValueError("Initial S1 screening admits only measured FP32")
    if optimization["matmul_precision"] != "highest":
        raise ValueError("Initial S1 screening preserves highest FP32 matmul precision")
    if optimization["scheduler"] not in {"constant", "warmup_cosine"}:
        raise ValueError("optimization.scheduler must be constant or warmup_cosine")
    if optimization["scheduler"] == "constant" and any(
        float(optimization[key]) != expected
        for key, expected in (("warmup_fraction", 0.0), ("minimum_lr_ratio", 1.0))
    ):
        raise ValueError("constant scheduler requires zero warmup and minimum_lr_ratio=1")
    if not 0 <= float(optimization["warmup_fraction"]) <= 1:
        raise ValueError("optimization.warmup_fraction must be in [0,1]")
    if not 0 < float(optimization["minimum_lr_ratio"]) <= 1:
        raise ValueError("optimization.minimum_lr_ratio must be in (0,1]")
    if not 0 < float(optimization["backbone_lr_multiplier"]) <= 1:
        raise ValueError("optimization.backbone_lr_multiplier must be in (0,1]")
    if any(not 0 <= float(augmentation[key]) <= 1 for key in augmentation):
        raise ValueError("Photometric augmentation magnitudes must be in [0,1]")
    if evaluation["selection_mode"] != "max":
        raise ValueError("S1 supports only maximization of a preregistered detection metric")
    if not isinstance(evaluation["legacy_metrics"], bool):
        raise ValueError("evaluation.legacy_metrics must be boolean")
    eval_updates = tuple(
        _integer(item, "evaluation.updates[]", minimum=1) for item in evaluation["updates"]
    )
    maximum_updates = _integer(optimization["updates"], "optimization.updates", minimum=1)
    if not eval_updates or tuple(sorted(set(eval_updates))) != eval_updates:
        raise ValueError("evaluation.updates must be nonempty, sorted, and unique")
    if eval_updates[-1] != maximum_updates or eval_updates[-1] > maximum_updates:
        raise ValueError("evaluation.updates must end at optimization.updates")
    full_updates = tuple(
        _integer(item, "evaluation.full_selection_updates[]", minimum=1)
        for item in evaluation["full_selection_updates"]
    )
    if tuple(sorted(set(full_updates))) != full_updates:
        raise ValueError("evaluation.full_selection_updates must be sorted and unique")
    if not set(full_updates).issubset(eval_updates):
        raise ValueError("Full selection evaluations must be scheduled monitor evaluations")
    hashes = (
        (data["expected_manifest_sha256"], "data.expected_manifest_sha256"),
        (data["expected_split_sha256"], "data.expected_split_sha256"),
        (model["backbone_weights_sha256"], "model.backbone_weights_sha256"),
        (
            data["expected_full_selection_sha256"],
            "data.expected_full_selection_sha256",
        ),
    )
    if any(re.fullmatch(r"[0-9a-f]{64}", _string(value, name)) is None for value, name in hashes):
        raise ValueError("Expected artifact hashes must be full SHA-256 values")
    if (
        re.fullmatch(
            r"[0-9a-f]{40}",
            _string(data["expected_dataset_revision"], "data.expected_dataset_revision"),
        )
        is None
    ):
        raise ValueError("dataset revision must be a full commit SHA")
    if _integer(data["label_cache_sequences"], "data.label_cache_sequences", minimum=1) > 24:
        raise ValueError("Label cache cannot exceed the frozen 24-sequence train role")
    return CampaignConfig(
        schema=CONFIG_SCHEMA,
        experiment_id=_string(raw["experiment_id"], "experiment_id"),
        seed=_integer(raw["seed"], "seed"),
        data=DataConfig(
            root=_string(data["root"], "data.root"),
            manifest=_string(data["manifest"], "data.manifest"),
            split=_string(data["split"], "data.split"),
            train_role=_string(data["train_role"], "data.train_role"),
            selection_role=_string(data["selection_role"], "data.selection_role"),
            expected_dataset_revision=_string(
                data["expected_dataset_revision"], "data.expected_dataset_revision"
            ),
            expected_manifest_sha256=_string(
                data["expected_manifest_sha256"], "data.expected_manifest_sha256"
            ),
            expected_split_sha256=_string(
                data["expected_split_sha256"], "data.expected_split_sha256"
            ),
            full_selection_tokens=_string(
                data["full_selection_tokens"], "data.full_selection_tokens"
            ),
            expected_full_selection_sha256=_string(
                data["expected_full_selection_sha256"],
                "data.expected_full_selection_sha256",
            ),
            cache_capacity=_integer(data["cache_capacity"], "data.cache_capacity", minimum=1),
            label_cache_sequences=_integer(
                data["label_cache_sequences"], "data.label_cache_sequences", minimum=1
            ),
        ),
        model=ModelConfig(
            backbone="resnet50",
            num_classes=8,
            queries=50,
            group_num=11,
            predict_velocity=cast(bool, model["predict_velocity"]),
            velocity_loss_coef=_number(model["velocity_loss_coef"], "model.velocity_loss_coef"),
            width=_integer(model["width"], "model.width", minimum=1),
            height=_integer(model["height"], "model.height", minimum=1),
            backbone_weights=_string(model["backbone_weights"], "model.backbone_weights"),
            backbone_weights_sha256=_string(
                model["backbone_weights_sha256"], "model.backbone_weights_sha256"
            ),
            source_root=_string(model["source_root"], "model.source_root"),
            portable_root=_string(model["portable_root"], "model.portable_root"),
        ),
        optimization=OptimizationConfig(
            updates=maximum_updates,
            checkpoint_interval=_integer(
                optimization["checkpoint_interval"],
                "optimization.checkpoint_interval",
                minimum=1,
            ),
            batch_size=_integer(optimization["batch_size"], "optimization.batch_size", minimum=1),
            precision="fp32",
            matmul_precision="highest",
            learning_rate=_number(
                optimization["learning_rate"], "optimization.learning_rate", strict=True
            ),
            backbone_lr_multiplier=_number(
                optimization["backbone_lr_multiplier"],
                "optimization.backbone_lr_multiplier",
                strict=True,
            ),
            weight_decay=_number(optimization["weight_decay"], "optimization.weight_decay"),
            gradient_clip_norm=_number(
                optimization["gradient_clip_norm"], "optimization.gradient_clip_norm", strict=True
            ),
            scheduler=cast(str, optimization["scheduler"]),
            warmup_fraction=_number(
                optimization["warmup_fraction"], "optimization.warmup_fraction"
            ),
            minimum_lr_ratio=_number(
                optimization["minimum_lr_ratio"], "optimization.minimum_lr_ratio", strict=True
            ),
        ),
        augmentation=AugmentationConfig(
            **{
                key: _number(augmentation[key], f"augmentation.{key}")
                for key in ("brightness", "contrast", "saturation")
            }
        ),
        evaluation=EvaluationConfig(
            protocol=_string(evaluation["protocol"], "evaluation.protocol"),
            updates=eval_updates,
            full_selection_updates=full_updates,
            selection_metric=_string(evaluation["selection_metric"], "evaluation.selection_metric"),
            selection_mode="max",
            legacy_metrics=cast(bool, evaluation["legacy_metrics"]),
        ),
        resources=ResourceConfig(
            maximum_seconds=_number(
                resources["maximum_seconds"], "resources.maximum_seconds", strict=True
            ),
            num_cpu_threads=_integer(
                resources["num_cpu_threads"], "resources.num_cpu_threads", minimum=1
            ),
            minimum_available_ram_gb=_number(
                resources["minimum_available_ram_gb"],
                "resources.minimum_available_ram_gb",
                strict=True,
            ),
            minimum_available_vram_gb=_number(
                resources["minimum_available_vram_gb"],
                "resources.minimum_available_vram_gb",
                strict=True,
            ),
        ),
    )


def dataset_cache_kwargs(config: CampaignConfig) -> dict[str, int]:
    """Return bounded dataset cache settings applied by every campaign reader."""
    if config.data.label_cache_sequences > 24:
        raise ValueError("Label cache cannot exceed the frozen 24-sequence train role")
    return {"max_label_sequences": config.data.label_cache_sequences}


def validate_campaign_manifest(config: CampaignConfig) -> dict[str, Any]:
    """Bind and validate the exact data population before model construction."""
    path, split_path = Path(config.data.manifest), Path(config.data.split)
    if sha256_file(path) != config.data.expected_manifest_sha256:
        raise ValueError("Campaign manifest SHA-256 differs from the preregistered value")
    if sha256_file(split_path) != config.data.expected_split_sha256:
        raise ValueError("Campaign split SHA-256 differs from the preregistered value")
    manifest_keys = {
        "schema",
        "dataset_revision",
        "source_split",
        "source_media_sha256",
        "candidate_split_sha256",
        "population_audit_sha256",
        "sampling_seed",
        "roles",
        "historical_panels",
        "sampling",
        "limitations",
    }
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    unknown = set(raw) - (manifest_keys | {"media_admission"})
    missing = manifest_keys - set(raw)
    if unknown or missing:
        raise ValueError(f"manifest: unknown={sorted(unknown)}, missing={sorted(missing)}")
    if raw["schema"] != MANIFEST_SCHEMA or raw["source_split"] != "train":
        raise ValueError("Campaign manifest must use public train data and the v1 schema")
    if raw["dataset_revision"] != config.data.expected_dataset_revision:
        raise ValueError("Campaign manifest dataset revision differs from config")
    if raw["candidate_split_sha256"] != config.data.expected_split_sha256:
        raise ValueError("Campaign manifest does not bind the configured split")
    _integer(raw["sampling_seed"], "manifest.sampling_seed")
    roles = _mapping(raw["roles"], "manifest.roles")
    expected_roles = {config.data.train_role, config.data.selection_role}
    if set(roles) != expected_roles:
        raise ValueError(f"manifest roles must be exactly {sorted(expected_roles)}")
    seen_tokens: set[str] = set()
    seen_sequences: set[str] = set()
    normalized: dict[str, Any] = {}
    for role in (config.data.train_role, config.data.selection_role):
        entry = _exact(roles[role], f"manifest.roles.{role}", {"sample_tokens", "sequence_ids"})
        tokens = tuple(_string(item, f"{role}.sample_tokens[]") for item in entry["sample_tokens"])
        sequences = tuple(_string(item, f"{role}.sequence_ids[]") for item in entry["sequence_ids"])
        if not tokens or not sequences or len(set(tokens)) != len(tokens):
            raise ValueError(f"{role} needs unique tokens and at least one sequence")
        if len(set(sequences)) != len(sequences):
            raise ValueError(f"{role} contains duplicate sequence IDs")
        if seen_tokens.intersection(tokens) or seen_sequences.intersection(sequences):
            raise ValueError("Training and selection populations must be token/sequence disjoint")
        seen_tokens.update(tokens)
        seen_sequences.update(sequences)
        normalized[role] = {"sample_tokens": tokens, "sequence_ids": sequences}
    panels = _exact(
        raw["historical_panels"],
        "manifest.historical_panels",
        {"pilot_selection256"},
    )
    pilot = _exact(
        panels["pilot_selection256"],
        "manifest.historical_panels.pilot_selection256",
        {"sample_tokens", "source"},
    )
    pilot_tokens = tuple(
        _string(item, "pilot_selection256.sample_tokens[]") for item in pilot["sample_tokens"]
    )
    if not pilot_tokens or len(set(pilot_tokens)) != len(pilot_tokens):
        raise ValueError("pilot_selection256 must contain unique tokens")
    return {
        **raw,
        "roles": normalized,
        "historical_panels": {"pilot_selection256": {**pilot, "sample_tokens": pilot_tokens}},
        "sha256": config.data.expected_manifest_sha256,
    }


def validate_manifest_against_metadata(
    manifest: Mapping[str, Any], config: CampaignConfig
) -> dict[str, Any]:
    """Verify every declared token's canonical sequence and public-train ownership."""
    import pandas as pd

    audit_receipt = validate_population_audit(manifest, config)

    role_tokens = [
        token
        for role in (config.data.train_role, config.data.selection_role)
        for token in manifest["roles"][role]["sample_tokens"]
    ]
    pilot_tokens = list(manifest["historical_panels"]["pilot_selection256"]["sample_tokens"])
    full_selection_path = Path(config.data.full_selection_tokens)
    if sha256_file(full_selection_path) != config.data.expected_full_selection_sha256:
        raise ValueError("Full-selection token artifact SHA-256 differs from config")
    full_selection = _mapping(
        json.loads(full_selection_path.read_text(encoding="utf-8")), "full_selection"
    )
    full_tokens = tuple(
        _string(item, "full_selection.sample_tokens[]")
        for item in full_selection.get("sample_tokens", ())
    )
    selection_sequences = set(manifest["roles"][config.data.selection_role]["sequence_ids"])
    if (
        full_selection.get("schema") != "eap_detection_full_selection_v1"
        or full_selection.get("dataset_revision") != config.data.expected_dataset_revision
        or full_selection.get("candidate_split_sha256") != config.data.expected_split_sha256
        or full_selection.get("source_media_sha256") != manifest["source_media_sha256"]
        or full_selection.get("role") != config.data.selection_role
        or set(full_selection.get("sequence_ids", ())) != selection_sequences
        or not full_tokens
        or len(set(full_tokens)) != len(full_tokens)
    ):
        raise ValueError("Full-selection token artifact differs from the frozen campaign")
    tokens = sorted(set(role_tokens + pilot_tokens + list(full_tokens)))
    media_path = Path(config.data.root) / "data/train.parquet"
    table = pd.read_parquet(
        media_path,
        columns=[
            "sample_token",
            "sequence_id",
            "split",
            "rgb_shard_path",
            "rgb_member_path",
        ],
        filters=[("sample_token", "in", tokens)],
    )
    if len(table) != len(tokens) or set(table.sample_token) != set(tokens):
        raise ValueError(
            "At least one campaign token is absent or duplicated in canonical metadata"
        )
    if table.sample_token.duplicated().any() or not table.split.eq("train").all():
        raise ValueError("Campaign tokens must map uniquely to the public train split")
    owner = dict(zip(table.sample_token.astype(str), table.sequence_id.astype(str), strict=True))
    token_media = dict(
        zip(
            table.sample_token.astype(str),
            zip(
                table.rgb_shard_path.astype(str),
                table.rgb_member_path.astype(str),
                strict=True,
            ),
            strict=True,
        )
    )
    for role in (config.data.train_role, config.data.selection_role):
        declared = set(manifest["roles"][role]["sequence_ids"])
        observed = {owner[token] for token in manifest["roles"][role]["sample_tokens"]}
        if observed != declared:
            raise ValueError(f"{role} token ownership differs from declared sequence_ids")
    full_observed = {owner[token] for token in full_tokens}
    if full_observed != selection_sequences:
        raise ValueError("Full-selection tokens do not belong to exactly the frozen sequences")
    split = json.loads(Path(config.data.split).read_text(encoding="utf-8"))
    assignments = _mapping(split.get("assignments"), "split.assignments")
    if split.get("dataset_revision") != config.data.expected_dataset_revision:
        raise ValueError("Split dataset revision differs from campaign config")
    if split.get("media_sha256") != manifest["source_media_sha256"]:
        raise ValueError("Manifest media hash differs from the fixed split")
    if sha256_file(media_path) != manifest["source_media_sha256"]:
        raise ValueError("Canonical media bytes differ from the campaign manifest")
    for role in (config.data.train_role, config.data.selection_role):
        declared = set(manifest["roles"][role]["sequence_ids"])
        if not declared.issubset(set(assignments.get(role, []))):
            raise ValueError(f"{role} contains sequences outside its fixed split assignment")
    if {owner[token] for token in pilot_tokens} - set(assignments[config.data.selection_role]):
        raise ValueError("Historical pilot panel leaves the frozen selection sequences")
    media_admission = validate_media_admission(
        manifest,
        config,
        full_selection_tokens=full_tokens,
        token_owner=owner,
        token_media=token_media,
    )
    return {
        **audit_receipt,
        "canonical_media": str(media_path),
        "canonical_row_count": len(table),
        "pilot_selection256_count": len(pilot_tokens),
        "full_selection_count": len(full_tokens),
        "full_selection_ownership_sha256": canonical_sha256(
            sorted((token, owner[token]) for token in full_tokens)
        ),
        "token_ownership_sha256": canonical_sha256(sorted(owner.items())),
        "split_sha256": config.data.expected_split_sha256,
        "media_admission": media_admission,
    }


def validate_media_admission(
    manifest: Mapping[str, Any],
    config: CampaignConfig,
    *,
    full_selection_tokens: Sequence[str],
    token_owner: Mapping[str, str],
    token_media: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    """Verify the decoded-RGB admission and return hashes enforced on first read."""
    declaration = _exact(
        manifest.get("media_admission"),
        "manifest.media_admission",
        {
            "receipt_path",
            "receipt_sha256",
            "source_manifest_sha256",
            "train_token_population_sha256",
        },
    )
    relative = safe_relative_path(
        _string(declaration["receipt_path"], "media_admission.receipt_path")
    )
    manifest_parent = Path(config.data.manifest).parent.resolve()
    receipt_path = manifest_parent.joinpath(*relative.parts).resolve()
    if not receipt_path.is_relative_to(manifest_parent):
        raise ValueError("Media-admission receipt escapes the manifest directory")
    receipt_sha = _string(declaration["receipt_sha256"], "media_admission.receipt_sha256")
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha:
        raise ValueError("Media-admission receipt is absent or differs from manifest")
    receipt = _exact(
        json.loads(receipt_path.read_text(encoding="utf-8")),
        "media_admission_receipt",
        {
            "schema",
            "status",
            "source_manifest_sha256",
            "source_manifest_canonical_sha256",
            "source_full_preflight_summary_sha256",
            "source_full_preflight_token_ledger_sha256",
            "source_full_preflight_shard_ledger_sha256",
            "source_frame_ledger_sha256",
            "source_full_selection_sha256",
            "source_media_sha256",
            "source_pinned_inventory_sha256",
            "implementation_sha256",
            "excluded",
            "replacement",
            "rare_identity_retention",
            "new_train_token_population_sha256",
            "selection_1024_unchanged",
            "full_selection_23958_unchanged",
            "final_counts",
            "final_media",
        },
    )
    train_tokens = tuple(manifest["roles"][config.data.train_role]["sample_tokens"])
    full_tokens = tuple(full_selection_tokens)
    expected_tokens = set(train_tokens) | set(full_tokens)
    train_population_sha = canonical_sha256(list(train_tokens))
    inventory = Path(__file__).parents[3] / "data/manifests/eap_detection_public_files_v1.json"
    bindings = {
        "source_manifest_sha256": declaration["source_manifest_sha256"],
        "new_train_token_population_sha256": declaration["train_token_population_sha256"],
        "source_full_selection_sha256": config.data.expected_full_selection_sha256,
        "source_media_sha256": manifest["source_media_sha256"],
        "source_pinned_inventory_sha256": sha256_file(inventory),
    }
    if (
        receipt["schema"] != "eap_detection_media_admission_v1"
        or receipt["status"] != "PASS"
        or declaration["train_token_population_sha256"] != train_population_sha
        or any(receipt.get(key) != value for key, value in bindings.items())
        or receipt["selection_1024_unchanged"] is not True
        or receipt["full_selection_23958_unchanged"] is not True
    ):
        raise ValueError("Media-admission receipt differs from the frozen campaign population")
    final_media = _mapping(receipt["final_media"], "media_admission_receipt.final_media")
    if set(final_media) != expected_tokens:
        raise ValueError("Media-admission ledger must cover exactly train and full selection")
    counts = _exact(
        receipt["final_counts"],
        "media_admission_receipt.final_counts",
        {"train", "full_selection", "total"},
    )
    if counts != {
        "train": len(train_tokens),
        "full_selection": len(full_tokens),
        "total": len(expected_tokens),
    }:
        raise ValueError("Media-admission counts differ from its exact token population")
    train_set = set(train_tokens)
    expected_hashes: dict[str, str] = {}
    for token in sorted(expected_tokens):
        entry = _exact(
            final_media[token],
            f"media_admission.final_media[{token}]",
            {
                "role",
                "sequence_id",
                "rgb_shard_path",
                "rgb_member_path",
                "member_size_bytes",
                "member_sha256",
                "image_wh",
                "source_owner",
            },
        )
        member_hash = _string(entry["member_sha256"], f"final_media[{token}].member_sha256")
        if (
            entry["role"] != ("train" if token in train_set else "full_selection")
            or entry["sequence_id"] != token_owner.get(token)
            or token_media.get(token) != (entry["rgb_shard_path"], entry["rgb_member_path"])
            or _integer(entry["member_size_bytes"], "member_size_bytes", minimum=1)
            > 32 * 1024 * 1024
            or not re.fullmatch(r"[0-9a-f]{64}", member_hash)
            or entry["image_wh"] != [1280, 720]
            or entry["source_owner"] not in {"rgb_full_preflight_v1", "replacement_probe_v1"}
        ):
            raise ValueError(f"Media-admission entry differs from metadata: {token}")
        expected_hashes[token] = member_hash
    return {
        "schema": receipt["schema"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "train_token_population_sha256": train_population_sha,
        "admitted_token_count": len(expected_hashes),
        "data_member_sha256": expected_hashes,
    }


def validate_population_audit(
    manifest: Mapping[str, Any], config: CampaignConfig
) -> dict[str, Any]:
    """Revalidate the signed population audit and label bytes used by the campaign."""
    audit_path = Path(config.data.manifest).parent / "population_audit.json"
    if not audit_path.is_file():
        raise ValueError("Campaign population audit is missing")
    audit = _mapping(json.loads(audit_path.read_text(encoding="utf-8")), "population_audit")
    audit_semantic_sha = canonical_sha256(audit)
    if audit_semantic_sha != manifest.get("population_audit_sha256"):
        raise ValueError("Population audit canonical SHA-256 differs from manifest")

    protocol = (
        EvaluationProtocol.default()
        if config.evaluation.protocol == "default"
        else EvaluationProtocol.from_dict(
            json.loads(Path(config.evaluation.protocol).read_text(encoding="utf-8"))
        )
    )
    expected = {
        "schema": "eap_detection_population_audit_v2",
        "dataset_revision": config.data.expected_dataset_revision,
        "source_split": "train",
        "source_media_sha256": manifest.get("source_media_sha256"),
        "candidate_split_sha256": config.data.expected_split_sha256,
        "evaluation_protocol_sha256": protocol.sha256,
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ValueError(f"Population audit {field} differs from the current campaign")

    inventory_path = Path(__file__).parents[3] / "data/manifests/eap_detection_public_files_v1.json"
    if not inventory_path.is_file() or sha256_file(inventory_path) != audit.get(
        "pinned_inventory_sha256"
    ):
        raise ValueError("Population audit pinned inventory is absent or has changed")

    expected_sequences = {
        sequence
        for role in (config.data.train_role, config.data.selection_role)
        for sequence in manifest["roles"][role]["sequence_ids"]
    }
    verified: dict[str, str] = {}
    entries = audit.get("label_files")
    if not isinstance(entries, list):
        raise ValueError("Population audit label_files must be a list")
    for index, raw_entry in enumerate(entries):
        entry = _exact(
            raw_entry,
            f"population_audit.label_files[{index}]",
            {"path", "bytes", "sha256", "status"},
        )
        relative = _string(entry["path"], f"label_files[{index}].path")
        parts = relative.split("/")
        if len(parts) != 4 or parts[:2] != ["data", "train"] or parts[3] != "labels.parquet":
            raise ValueError(f"Unexpected population-audit label path: {relative}")
        sequence = parts[2]
        if sequence not in expected_sequences:
            continue
        if sequence in verified:
            raise ValueError(f"Duplicate label audit entry for sequence {sequence}")
        path = safe_dataset_path(Path(config.data.root), relative)
        size = _integer(entry["bytes"], f"label_files[{index}].bytes", minimum=1)
        digest = _string(entry["sha256"], f"label_files[{index}].sha256")
        if (
            entry["status"] != "verified"
            or not path.is_file()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ValueError(f"Label bytes differ from population audit for {sequence}")
        verified[sequence] = digest
    if set(verified) != expected_sequences:
        missing = sorted(expected_sequences - set(verified))
        raise ValueError(f"Population audit lacks campaign label files: {missing}")
    return {
        "population_audit": str(audit_path),
        "population_audit_sha256": audit_semantic_sha,
        "pinned_inventory_sha256": audit["pinned_inventory_sha256"],
        "label_files_sha256": canonical_sha256(sorted(verified.items())),
        "label_sequence_count": len(verified),
        "evaluation_protocol_sha256": protocol.sha256,
    }


class ResumablePermutationSampler:
    """Generate seeded permutations without repeating an item within an epoch."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = _integer(size, "sampler.size", minimum=1)
        self.seed = _integer(seed, "sampler.seed")
        self.epoch = 0
        self.position = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()

    def next_batch(self, batch_size: int) -> list[int]:
        """Return exactly ``batch_size`` indices, crossing epochs deterministically."""
        count = _integer(batch_size, "batch_size", minimum=1)
        result: list[int] = []
        while len(result) < count:
            available = min(count - len(result), self.size - self.position)
            result.extend(self.order[self.position : self.position + available])
            self.position += available
            if self.position == self.size:
                self.epoch += 1
                self.position = 0
                self.order = torch.randperm(self.size, generator=self.generator).tolist()
        return result

    def state_dict(self) -> dict[str, Any]:
        """Serialize the next-index position and generator state."""
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore only a state that belongs to the same sampler contract."""
        if state["size"] != self.size or state["seed"] != self.seed:
            raise ValueError("Sampler size or seed differs from checkpoint")
        order = list(state["order"])
        if sorted(order) != list(range(self.size)):
            raise ValueError("Checkpoint sampler order is not a permutation")
        position = _integer(state["position"], "sampler.position")
        if position >= self.size:
            raise ValueError("Checkpoint sampler position is outside the permutation")
        self.epoch = _integer(state["epoch"], "sampler.epoch")
        self.position, self.order = position, order
        self.generator.set_state(state["generator_state"])


class UpdateLRScheduler:
    """Apply a deterministic constant or linear-warmup/cosine schedule by update."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_updates: int,
        kind: str,
        warmup_fraction: float,
        minimum_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_updates = _integer(total_updates, "total_updates", minimum=1)
        if kind not in {"constant", "warmup_cosine"}:
            raise ValueError("Unknown scheduler")
        self.kind = kind
        self.warmup_updates = round(self.total_updates * warmup_fraction)
        self.minimum_ratio = float(minimum_ratio)
        self.update = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def ratio_for(self, update: int) -> float:
        """Return the LR multiplier used by the one-indexed update."""
        current = _integer(update, "update", minimum=1)
        if current > self.total_updates:
            raise ValueError("Scheduler update exceeds the fixed budget")
        if self.kind == "constant":
            return 1.0
        if self.warmup_updates and current <= self.warmup_updates:
            return current / self.warmup_updates
        denominator = max(1, self.total_updates - self.warmup_updates)
        progress = (current - self.warmup_updates) / denominator
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.minimum_ratio + (1 - self.minimum_ratio) * cosine

    def apply(self, update: int) -> list[float]:
        """Set and return the learning rates for the requested update."""
        ratio = self.ratio_for(update)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * ratio
        self.update = update
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Serialize effective schedule settings and position."""
        return {
            "total_updates": self.total_updates,
            "kind": self.kind,
            "warmup_updates": self.warmup_updates,
            "minimum_ratio": self.minimum_ratio,
            "update": self.update,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a schedule only when all immutable settings match."""
        expected = self.state_dict()
        for key in ("total_updates", "kind", "warmup_updates", "minimum_ratio", "base_lrs"):
            if state[key] != expected[key]:
                raise ValueError(f"Scheduler {key} differs from checkpoint")
        self.update = _integer(state["update"], "scheduler.update")
        if self.update:
            self.apply(self.update)


def optimizer_parameter_groups(
    model: torch.nn.Module, config: OptimizationConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Separate backbone and detector parameters with an auditable ledger."""
    backbone_module = getattr(model, "backbone", None)
    if not isinstance(backbone_module, torch.nn.Module):
        raise ValueError("Model must expose its backbone as a torch module")
    backbone_ids = {id(parameter) for parameter in backbone_module.parameters()}
    backbone: list[torch.nn.Parameter] = []
    detector: list[torch.nn.Parameter] = []
    frozen: list[str] = []
    names: dict[str, list[str]] = {"backbone": [], "detector": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(name)
        elif id(parameter) in backbone_ids:
            backbone.append(parameter)
            names["backbone"].append(name)
        else:
            detector.append(parameter)
            names["detector"].append(name)
    if not backbone or not detector:
        raise ValueError("Model must expose trainable backbone and detector parameters")
    groups = [
        {
            "params": backbone,
            "lr": config.learning_rate * config.backbone_lr_multiplier,
            "weight_decay": config.weight_decay,
            "name": "backbone",
        },
        {
            "params": detector,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "name": "detector",
        },
    ]
    ledger = {
        "groups": {
            key: {
                "parameter_tensors": len(value),
                "parameter_elements": sum(
                    dict(model.named_parameters())[name].numel() for name in value
                ),
                "names_sha256": canonical_sha256(value),
            }
            for key, value in names.items()
        },
        "frozen_parameter_tensors": len(frozen),
        "frozen_names_sha256": canonical_sha256(frozen),
    }
    return groups, ledger


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU and all CUDA generators."""
    numpy_state = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore all captured random generators."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_campaign_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateLRScheduler,
    sampler: ResumablePermutationSampler,
    config: CampaignConfig,
    protocol_sha256: str,
    runtime_provenance: Mapping[str, Any],
    update: int,
    best_metric: float | None,
    progress: Mapping[str, int | float],
    data_member_sha256: Mapping[str, str],
) -> str:
    """Atomically save every state required to continue the next update."""
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "producer": "scripts/train_detection_campaign.py",
        "inference_contract": "campaign_v1_strict_loader_required",
        "config_sha256": config.sha256,
        "protocol_sha256": protocol_sha256,
        "runtime_provenance": dict(runtime_provenance),
        "runtime_provenance_sha256": canonical_sha256(runtime_provenance),
        "update": _integer(update, "update"),
        "best_metric": best_metric,
        "progress": dict(progress),
        "data_member_sha256": dict(data_member_sha256),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": capture_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def load_campaign_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: UpdateLRScheduler,
    sampler: ResumablePermutationSampler,
    config: CampaignConfig,
    protocol_sha256: str,
    runtime_provenance: Mapping[str, Any],
) -> tuple[int, float | None, dict[str, int | float], dict[str, str]]:
    """Strictly restore a trusted local campaign checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unknown campaign checkpoint schema")
    if payload.get("producer") != "scripts/train_detection_campaign.py":
        raise ValueError("Checkpoint producer is not the S1 campaign runner")
    if payload.get("inference_contract") != "campaign_v1_strict_loader_required":
        raise ValueError("Checkpoint inference contract is unsupported")
    if payload.get("config_sha256") != config.sha256:
        raise ValueError("Checkpoint configuration differs from this run")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Checkpoint evaluation protocol differs from this run")
    expected_provenance = canonical_sha256(runtime_provenance)
    if payload.get("runtime_provenance_sha256") != expected_provenance:
        raise ValueError("Checkpoint executed runtime provenance differs from this run")
    if payload.get("runtime_provenance") != dict(runtime_provenance):
        raise ValueError("Checkpoint runtime provenance payload is inconsistent")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    sampler.load_state_dict(payload["sampler"])
    restore_rng_state(payload["rng"])
    progress = dict(payload.get("progress", {}))
    required_progress = {
        "training_seconds",
        "empty_target_exposures",
        "valid_velocity_object_exposures",
    }
    if set(progress) != required_progress:
        raise ValueError("Checkpoint campaign progress fields differ")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in progress.values()
    ):
        raise ValueError("Checkpoint campaign progress must be finite nonnegative numbers")
    member_hashes = dict(payload.get("data_member_sha256", {}))
    if any(
        not isinstance(token, str)
        or not token
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for token, digest in member_hashes.items()
    ):
        raise ValueError("Checkpoint data member hash ledger is malformed")
    return (
        _integer(payload["update"], "checkpoint.update"),
        payload["best_metric"],
        progress,
        member_hashes,
    )


def inherit_selected_checkpoint(
    resume: Path,
    destination: Path,
    *,
    config: CampaignConfig,
    protocol_sha256: str,
    runtime_provenance: Mapping[str, Any],
    best_metric: float | None,
) -> dict[str, object] | None:
    """Verify and preserve an ancestor's best checkpoint in a resumed run."""
    if best_metric is None:
        return None
    source = resume.parent / "selected.pt"
    if not source.is_file():
        raise ValueError("Resume records a best metric but has no sibling selected.pt")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    expected_provenance = canonical_sha256(runtime_provenance)
    checks = (
        payload.get("schema") == CHECKPOINT_SCHEMA,
        payload.get("config_sha256") == config.sha256,
        payload.get("protocol_sha256") == protocol_sha256,
        payload.get("runtime_provenance_sha256") == expected_provenance,
        payload.get("runtime_provenance") == dict(runtime_provenance),
        payload.get("best_metric") == best_metric,
    )
    if not all(checks):
        raise ValueError("Ancestor selected checkpoint differs from resumed campaign state")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_sha = sha256_file(source)
    if sha256_file(destination) != source_sha:
        raise RuntimeError("Inherited selected checkpoint copy differs from its ancestor")
    return {
        "source": str(source.resolve()),
        "source_sha256": source_sha,
        "selected_update": _integer(payload["update"], "selected.update"),
        "best_metric": best_metric,
        "status": "VERIFIED_INHERITED",
    }


def run_update_loop(
    *,
    start_update: int,
    maximum_update: int,
    batch_size: int,
    sampler: ResumablePermutationSampler,
    scheduler: UpdateLRScheduler,
    step: Callable[[int, Sequence[int], Sequence[float]], T],
) -> list[T]:
    """Run the deterministic state transition used by CPU tests and the CUDA CLI."""
    if start_update < 0 or start_update > maximum_update:
        raise ValueError("Invalid starting update")
    records = []
    for update in range(start_update + 1, maximum_update + 1):
        batch = sampler.next_batch(batch_size)
        lrs = scheduler.apply(update)
        records.append(step(update, batch, lrs))
    return records
