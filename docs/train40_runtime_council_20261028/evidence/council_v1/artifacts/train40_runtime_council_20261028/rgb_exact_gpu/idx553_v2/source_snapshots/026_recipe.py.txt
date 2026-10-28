"""Prospectively frozen RGB-PORT producer recipes derived from audited TRAIN40 JSON."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRODUCER_IDS = ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F")
AUTHORITY_SHA256 = "39fac190a2568757511de017fb7e7d55428506382a0066804af8241d4908a66b"
SCHEMA = "rgb_port_producers_v1"


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible scientific identity without platform whitespace."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProducerRecipe:
    """One fully expanded producer contract with a population-derived endpoint."""

    fit_id: str
    base_arm: str
    modality: str
    seed: int
    epochs: int
    effective_batch_size: int
    microbatch_size: int
    accumulation_steps: int
    updates: int
    model_config: dict[str, Any]
    loss_config: dict[str, Any]
    training_config: dict[str, Any]
    sampler_policy: dict[str, Any]
    teacher_policy: dict[str, Any]
    authority_sha256: str
    role_manifest_sha256: str
    producer_population: int

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            **self.__dict__,
            "random_initialization": True,
            "event_weights_loaded": False,
            "checkpoint_selection": "fixed_epoch18",
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.identity)


def _expected_arm(fit_id: str) -> tuple[str, str]:
    if fit_id not in PRODUCER_IDS:
        raise ValueError(f"Unknown RGB-PORT producer: {fit_id}")
    arm = "a5" if "A5" in fit_id else "c2f"
    modality = "event" if fit_id.startswith("E_") else "rgb"
    return arm, modality


def resolved_recipe(
    config_path: Path,
    *,
    fit_id: str,
    producer_population: int,
    role_manifest_sha256: str,
    microbatch_size: int | None = None,
) -> ProducerRecipe:
    """Load and validate a frozen expanded config before model construction."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("authority_sha256") != AUTHORITY_SHA256:
        raise ValueError("Producer recipe authority differs")
    arm, modality = _expected_arm(fit_id)
    entry = payload.get("producers", {}).get(fit_id)
    if not isinstance(entry, dict) or entry.get("base_arm") != arm:
        raise ValueError("Expanded producer entry is absent or bound to another arm")
    if not 0 < producer_population <= 88_744:
        raise ValueError("P population must be positive and bounded by TRAIN40")
    if len(role_manifest_sha256) != 64:
        raise ValueError("P role manifest SHA-256 is required")
    model = dict(entry["model_config"])
    if (model.get("modality"), int(model.get("in_channels", -1))) != (
        modality,
        12 if modality == "event" else 3,
    ):
        raise ValueError("Producer modality/channel contract differs")
    training = dict(entry["training_config"])
    required = {
        "seed": 7,
        "epochs": 18,
        "foreground_warmup_epochs": 3,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "minimum_learning_rate": 3e-5,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "checkpoint_selection_mode": "last_epoch",
    }
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError("Expanded producer training recipe differs from the campaign")
    effective = 32
    microbatch = int(microbatch_size or training.get("microbatch_size", effective))
    if microbatch <= 0 or effective % microbatch:
        raise ValueError("Microbatch must divide the fixed effective batch 32")
    updates = 18 * math.ceil(producer_population / effective)
    if updates > 49_932:
        raise ValueError("Producer scientific endpoint exceeds its fixed cap")
    teacher = dict(entry["teacher_policy"])
    if (
        teacher.get("kind") != "generic_frozen_dinov3_local_relational"
        or teacher.get("roles") != ["P"]
        or teacher.get("source_pixels") != "original_uint8_rgb"
        or teacher.get("teacher_is_model_input") is not False
    ):
        raise ValueError("DINO teacher privilege/input path differs")
    sampler = dict(entry.get("sampler_policy", {}))
    if sampler != {
        "kind": "shard_grouped_random_v1",
        "fallback_group_size": 256,
        "shuffle_groups": True,
        "shuffle_rows_within_group": True,
        "effective_batch_order_shared_between_modalities": True,
        "mixed_t_split_before_collate": True,
    }:
        raise ValueError("Producer cache-local sampler policy differs")
    return ProducerRecipe(
        fit_id=fit_id,
        base_arm=arm,
        modality=modality,
        seed=7,
        epochs=18,
        effective_batch_size=effective,
        microbatch_size=microbatch,
        accumulation_steps=effective // microbatch,
        updates=updates,
        model_config=model,
        loss_config=dict(entry["loss_config"]),
        training_config=training,
        sampler_policy=sampler,
        teacher_policy=teacher,
        authority_sha256=AUTHORITY_SHA256,
        role_manifest_sha256=role_manifest_sha256,
        producer_population=producer_population,
    )


def verify_authority(config_path: Path, authority_path: Path) -> dict[str, Any]:
    """Compare every inherited model/loss/training field with the audited JSON."""
    if file_sha256(authority_path) != AUTHORITY_SHA256:
        raise ValueError("Historical producer authority bytes changed")
    expanded = json.loads(config_path.read_text(encoding="utf-8"))
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    for fit_id in PRODUCER_IDS:
        arm, modality = _expected_arm(fit_id)
        candidate = expanded["producers"][fit_id]
        original = authority["producers"][arm]
        expected_model = dict(original["model_config"])
        expected_model.update(
            modality=modality,
            in_channels=12 if modality == "event" else 3,
        )
        if candidate["model_config"] != expected_model:
            raise ValueError(f"{fit_id} model differs beyond the registered RGB port")
        if candidate["loss_config"] != original["loss_config"]:
            raise ValueError(f"{fit_id} loss differs from audited {arm}")
        expected_training = dict(original["training_config"])
        expected_training.update(
            minimum_epochs=18,
            early_stopping_patience=0,
            checkpoint_selection_mode="last_epoch",
            microbatch_size=32,
        )
        if candidate["training_config"] != expected_training:
            raise ValueError(f"{fit_id} training fields differ from registered endpoint changes")
    return {
        "status": "MATCHED",
        "authority_sha256": AUTHORITY_SHA256,
        "producer_ids": list(PRODUCER_IDS),
        "registered_training_changes": [
            "minimum_epochs:8->18",
            "early_stopping_patience:5->0",
            "microbatch_size:explicit32",
        ],
    }


def assert_matched_pair(event: ProducerRecipe, rgb: ProducerRecipe) -> None:
    """Prove E/R differ only in modality, channels and ID-specific metadata."""
    if event.base_arm != rgb.base_arm or event.modality != "event" or rgb.modality != "rgb":
        raise ValueError("Expected matched event/RGB recipes from one base arm")
    event_model, rgb_model = dict(event.model_config), dict(rgb.model_config)
    differences = {
        key
        for key in set(event_model) | set(rgb_model)
        if event_model.get(key) != rgb_model.get(key)
    }
    if differences != {"modality", "in_channels"}:
        raise ValueError(f"Unregistered matched-model differences: {sorted(differences)}")
    if event.loss_config != rgb.loss_config:
        raise ValueError("Matched E/R loss JSON differs")
    ignored = {"modality", "fit_id"}
    if any(
        getattr(event, field) != getattr(rgb, field)
        for field in (
            "seed",
            "epochs",
            "effective_batch_size",
            "microbatch_size",
            "accumulation_steps",
            "updates",
            "training_config",
            "sampler_policy",
            "teacher_policy",
            "role_manifest_sha256",
            "producer_population",
        )
        if field not in ignored
    ):
        raise ValueError("Matched E/R execution fields differ")


__all__ = [
    "AUTHORITY_SHA256",
    "PRODUCER_IDS",
    "ProducerRecipe",
    "assert_matched_pair",
    "canonical_sha256",
    "file_sha256",
    "resolved_recipe",
    "verify_authority",
]
