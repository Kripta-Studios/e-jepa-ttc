"""Deterministic label-free core trainer for Dense Level--Dynamics JEPA.

This module deliberately consumes already materialized event-only batches.  It does
not select a subset, open Garl annotations, build an eAP manifest, or access EvTTC.
Those responsibilities belong to the separately owned manifest/data layer.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import torch

from e_jepa_ttc.losses.level_dynamics_jepa import (
    LevelDynamicsLossConfig,
    LevelDynamicsObjectiveOutput,
    WithinTrackNCEOutput,
    build_horizon_positive_weights,
    build_temporal_residual_target,
    dense_cosine_loss,
    residual_visreg_loss,
    temporal_residual_cosine_loss,
    validate_nce_preflight,
    within_track_nce_loss,
)
from e_jepa_ttc.models.dense_level_dynamics_jepa import (
    DenseLevelDynamicsConfig,
    DenseLevelDynamicsJEPA,
    DenseLevelDynamicsOutput,
)

PROHIBITED_SSL_LABEL_KEY_FRAGMENTS: tuple[str, ...] = (
    "ttc",
    "depth",
    "3d",
    "category",
    "bbox",
    "boxes",
    "mask",
    "rgb",
    "evttc",
    "garl",
)
SAFE_SSL_CONTROL_KEYS: frozenset[str] = frozenset({"nce_candidate_mask"})
LABEL_FAMILY_PROVENANCE_FIELDS: tuple[str, ...] = (
    "uses_ttc_labels",
    "uses_depth_or_3d",
    "uses_category_labels",
    "uses_boxes",
    "uses_masks",
    "uses_rgb",
    "uses_evttc",
)


def reject_prohibited_label_keys(value: Mapping[str, Any], *, path: str = "batch") -> None:
    """Reject label-family keys recursively before a batch can reach the SSL model."""

    for key, item in value.items():
        lowered = str(key).lower().replace("-", "_")
        if lowered not in SAFE_SSL_CONTROL_KEYS and any(
            fragment in lowered for fragment in PROHIBITED_SSL_LABEL_KEY_FRAGMENTS
        ):
            raise ValueError(
                f"SSL-Pure {path} contains prohibited label-family key {key!r}; "
                "only event volumes, timestamps, sequence boundaries and track identity "
                "are allowed."
            )
        if isinstance(item, Mapping):
            reject_prohibited_label_keys(cast(Mapping[str, Any], item), path=f"{path}.{key}")


@dataclass(frozen=True)
class LabelFreeBatch:
    """Typed event-only batch consumed by the high-resolution SSL core.

    ``future_events`` contains complete future event windows with shape
    ``[B,H,T,C,H,W]``.  Identity and timestamps exist only to enforce within-track
    temporal NCE; no supervised labels are represented by this contract.
    """

    context_events: torch.Tensor
    future_events: torch.Tensor
    horizon_delta_t_s: torch.Tensor
    sequence_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    reference_timestamps_s: torch.Tensor
    future_timestamps_s: torch.Tensor
    context_valid: torch.Tensor | None = None
    future_valid: torch.Tensor | None = None
    nce_candidate_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.context_events.ndim != 5 or self.future_events.ndim != 6:
            raise ValueError("LabelFreeBatch events must be [B,T,C,H,W] and [B,H,T,C,H,W].")
        batch, steps, channels, height, width = self.context_events.shape
        future_batch, horizons, future_steps, future_channels, future_height, future_width = (
            self.future_events.shape
        )
        if (future_batch, future_steps, future_channels, future_height, future_width) != (
            batch,
            steps,
            channels,
            height,
            width,
        ):
            raise ValueError(
                "LabelFreeBatch future event windows must align with context geometry."
            )
        if (
            self.horizon_delta_t_s.shape != (batch, horizons)
            or not self.horizon_delta_t_s.is_floating_point()
        ):
            raise ValueError("LabelFreeBatch horizon_delta_t_s must be floating [B,H].")
        if not bool(torch.isfinite(self.horizon_delta_t_s).all()) or bool(
            (self.horizon_delta_t_s <= 0.0).any()
        ):
            raise ValueError(
                "LabelFreeBatch horizon_delta_t_s must contain finite positive values."
            )
        if self.reference_timestamps_s.shape != (batch,) or self.future_timestamps_s.shape != (
            batch,
            horizons,
        ):
            raise ValueError("LabelFreeBatch timestamps must be [B] and [B,H].")
        if not (
            self.reference_timestamps_s.is_floating_point()
            and self.future_timestamps_s.is_floating_point()
        ):
            raise ValueError("LabelFreeBatch timestamps must be floating tensors.")
        if not bool(torch.isfinite(self.reference_timestamps_s).all()) or not bool(
            torch.isfinite(self.future_timestamps_s).all()
        ):
            raise ValueError("LabelFreeBatch timestamps must be finite.")
        if not (
            len(self.sequence_ids) == len(self.track_ids) == batch
            and all(str(value) for value in self.sequence_ids)
            and all(str(value) for value in self.track_ids)
        ):
            raise ValueError(
                "LabelFreeBatch sequence_ids and track_ids must be non-empty and align with B."
            )
        if self.context_valid is not None and (
            self.context_valid.dtype != torch.bool or self.context_valid.shape != (batch, steps)
        ):
            raise ValueError("LabelFreeBatch context_valid must be bool [B,T].")
        if self.future_valid is not None and (
            self.future_valid.dtype != torch.bool
            or self.future_valid.shape != (batch, horizons, steps)
        ):
            raise ValueError("LabelFreeBatch future_valid must be bool [B,H,T].")
        if self.nce_candidate_mask is not None and (
            self.nce_candidate_mask.dtype != torch.bool
            or self.nce_candidate_mask.shape != (batch, horizons, batch, horizons)
        ):
            raise ValueError("LabelFreeBatch nce_candidate_mask must be bool [B,H,B,H].")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LabelFreeBatch:
        """Create a typed batch while rejecting labels and unknown implicit inputs."""

        reject_prohibited_label_keys(value)
        required = {
            "context_events",
            "future_events",
            "horizon_delta_t_s",
            "sequence_ids",
            "track_ids",
            "reference_timestamps_s",
            "future_timestamps_s",
        }
        optional = {"context_valid", "future_valid", "nce_candidate_mask"}
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required - optional)
        if missing or extra:
            raise ValueError(
                f"LabelFreeBatch mapping schema mismatch; missing={missing}, extra={extra}."
            )
        sequence_ids = value["sequence_ids"]
        track_ids = value["track_ids"]
        if not isinstance(sequence_ids, Sequence) or isinstance(sequence_ids, str):
            raise TypeError("sequence_ids must be a sequence of strings.")
        if not isinstance(track_ids, Sequence) or isinstance(track_ids, str):
            raise TypeError("track_ids must be a sequence of strings.")
        tensors = {
            key: value[key]
            for key in (
                "context_events",
                "future_events",
                "horizon_delta_t_s",
                "reference_timestamps_s",
                "future_timestamps_s",
            )
        }
        if any(not isinstance(item, torch.Tensor) for item in tensors.values()):
            raise TypeError("LabelFreeBatch event/timestamp fields must be torch.Tensor values.")
        return cls(
            context_events=cast(torch.Tensor, value["context_events"]),
            future_events=cast(torch.Tensor, value["future_events"]),
            horizon_delta_t_s=cast(torch.Tensor, value["horizon_delta_t_s"]),
            sequence_ids=tuple(str(item) for item in sequence_ids),
            track_ids=tuple(str(item) for item in track_ids),
            reference_timestamps_s=cast(torch.Tensor, value["reference_timestamps_s"]),
            future_timestamps_s=cast(torch.Tensor, value["future_timestamps_s"]),
            context_valid=cast(torch.Tensor | None, value.get("context_valid")),
            future_valid=cast(torch.Tensor | None, value.get("future_valid")),
            nce_candidate_mask=cast(torch.Tensor | None, value.get("nce_candidate_mask")),
        )

    def to(self, device: torch.device) -> LabelFreeBatch:
        """Move tensors only; immutable identities remain host-resident metadata."""

        return LabelFreeBatch(
            context_events=self.context_events.to(device=device, dtype=torch.float32),
            future_events=self.future_events.to(device=device, dtype=torch.float32),
            horizon_delta_t_s=self.horizon_delta_t_s.to(device=device, dtype=torch.float64),
            sequence_ids=self.sequence_ids,
            track_ids=self.track_ids,
            reference_timestamps_s=self.reference_timestamps_s.to(
                device=device, dtype=torch.float64
            ),
            future_timestamps_s=self.future_timestamps_s.to(device=device, dtype=torch.float64),
            context_valid=(
                self.context_valid.to(device=device) if self.context_valid is not None else None
            ),
            future_valid=(
                self.future_valid.to(device=device) if self.future_valid is not None else None
            ),
            nce_candidate_mask=(
                self.nce_candidate_mask.to(device=device)
                if self.nce_candidate_mask is not None
                else None
            ),
        )


class LabelFreeDataset(Protocol):
    """Minimal dataset contract; a builder owns selection and raw-data decoding."""

    def __len__(self) -> int:
        """Return bounded sample count."""

    def __getitem__(self, index: int) -> LabelFreeBatch:
        """Return one event-only pretraining batch/sample."""


LabelFreeBatchLike: TypeAlias = LabelFreeBatch | Mapping[str, Any]
TrainMetricValue: TypeAlias = float | list[int]


@dataclass(frozen=True)
class LabelFreeManifestProvenance:
    """Frozen provenance copied into every pretraining/resume checkpoint."""

    matched_manifest_hash: str
    dataset_hashes: Mapping[str, str]
    split_hash: str
    sampler_order_hash: str
    selection_rule: str
    label_family_provenance: Mapping[str, bool]

    def __post_init__(self) -> None:
        required = set(LABEL_FAMILY_PROVENANCE_FIELDS)
        missing = sorted(required - set(self.label_family_provenance))
        truthy = sorted(
            key for key in required if bool(self.label_family_provenance.get(key, True))
        )
        if missing or truthy:
            raise ValueError(
                "SSL-Pure provenance must explicitly declare every prohibited label family false; "
                f"missing={missing}, truthy={truthy}."
            )
        values = (
            self.matched_manifest_hash,
            self.split_hash,
            self.sampler_order_hash,
            self.selection_rule,
        )
        if any(not str(value).strip() for value in values):
            raise ValueError("Manifest provenance requires non-empty hashes and selection_rule.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LabelFreeManifestProvenance:
        """Parse a minimal, already-signed manifest header without selecting rows."""

        provenance = value.get("label_family_provenance", value.get("provenance"))
        if not isinstance(provenance, Mapping):
            raise ValueError("Manifest must contain explicit label_family_provenance mapping.")
        manifest_hash = value.get("matched_manifest_hash", value.get("manifest_sha256"))
        dataset_hashes = value.get("dataset_hashes", {})
        if not isinstance(dataset_hashes, Mapping):
            raise TypeError("Manifest dataset_hashes must be a mapping.")
        return cls(
            matched_manifest_hash=str(manifest_hash or ""),
            dataset_hashes={str(key): str(item) for key, item in dataset_hashes.items()},
            split_hash=str(value.get("split_hash", "")),
            sampler_order_hash=str(value.get("sampler_order_hash", "")),
            selection_rule=str(value.get("selection_rule", "")),
            label_family_provenance={str(key): bool(item) for key, item in provenance.items()},
        )


def _canonical_manifest_hash(value: Mapping[str, Any], signature_key: str) -> str:
    payload = dict(value)
    payload.pop(signature_key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_signed_label_free_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], LabelFreeManifestProvenance]:
    """Read and verify a compact manifest header, never its source datasets or labels."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Label-free matched manifest is missing: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Label-free matched manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Label-free matched manifest must be a JSON object.")
    signature_key = next(
        (key for key in ("signature", "manifest_sha256", "artifact_sha256") if key in value),
        None,
    )
    if signature_key is None or not isinstance(value[signature_key], str):
        raise ValueError("Label-free matched manifest lacks a supported SHA-256 signature field.")
    if value[signature_key] != _canonical_manifest_hash(value, signature_key):
        raise ValueError("Label-free matched manifest signature mismatch.")
    return value, LabelFreeManifestProvenance.from_mapping(value)


@dataclass(frozen=True)
class EAPHighResJEPATrainerConfig:
    """Deterministic optimization and health settings for the bounded core trainer."""

    model: DenseLevelDynamicsConfig = field(default_factory=DenseLevelDynamicsConfig)
    loss: LevelDynamicsLossConfig = field(default_factory=LevelDynamicsLossConfig)
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    total_updates: int = 1_000
    seed: int = 7
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("Trainer learning rate, weight decay and grad norm must be valid.")
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive.")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16.")


def _config_hash(config: EAPHighResJEPATrainerConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "visreg_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any], generator: torch.Generator) -> None:
    required = {"python", "numpy", "torch", "visreg_generator"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Resume checkpoint RNG state is incomplete: {missing}")
    random.setstate(cast(tuple[Any, ...], state["python"]))
    np.random.set_state(cast(tuple[Any, ...], state["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, state["torch"]))
    generator.set_state(cast(torch.Tensor, state["visreg_generator"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cast(list[torch.Tensor], state["cuda"]))


def _masked_patch_pool(
    tokens: torch.Tensor, valid_patch_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 4 or valid_patch_mask.shape != tokens.shape[:3]:
        raise ValueError("NCE patch pooling requires [B,H,P,D] tokens and [B,H,P] masks.")
    weights = valid_patch_mask.to(tokens.dtype).unsqueeze(-1)
    pooled = (tokens * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    return pooled, valid_patch_mask.any(dim=2)


class EAPHighResJEPATrainer:
    """Small deterministic trainer that owns objective assembly, EMA and resume state."""

    def __init__(
        self,
        config: EAPHighResJEPATrainerConfig | None = None,
        *,
        model: DenseLevelDynamicsJEPA | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or EAPHighResJEPATrainerConfig()
        self.device = torch.device(device)
        self.model = (model or DenseLevelDynamicsJEPA(self.config.model)).to(self.device)
        if self.model.config != self.config.model:
            raise ValueError("Trainer/model DenseLevelDynamicsConfig must match exactly.")
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = scheduler or torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda _: 1.0,
        )
        self.visreg_generator = torch.Generator(device=self.device.type)
        self.visreg_generator.manual_seed(self.config.seed)
        self.epoch = 0
        self.update_count = 0
        self.nce_preflight_passed = False
        self.health_state: dict[str, Any] = {
            "finite_updates": 0,
            "non_finite_updates": 0,
            "nce_preflight_passed": False,
            "last_level_std": None,
            "last_dynamics_std": None,
        }

    @staticmethod
    def coerce_batch(batch: LabelFreeBatchLike) -> LabelFreeBatch:
        """Normalize a typed or mapping batch and reject prohibited data keys."""

        if isinstance(batch, LabelFreeBatch):
            return batch
        if isinstance(batch, Mapping):
            return LabelFreeBatch.from_mapping(cast(Mapping[str, Any], batch))
        raise TypeError(
            "Dense Level-Dynamics trainer accepts only LabelFreeBatch or mapping batches."
        )

    def _nce_output(
        self,
        output: DenseLevelDynamicsOutput,
        batch: LabelFreeBatch,
    ) -> WithinTrackNCEOutput:
        # A context-invalid patch cannot serve as an NCE anchor even when a target
        # future patch happens to exist.  Candidate validity remains target-only:
        # another valid context anchor from the same track may still identify it.
        context_patch_valid = output.valid_patch_mask.any(dim=1)
        anchor_patch_mask = output.valid_target_patch_mask & context_patch_valid.unsqueeze(1)
        predicted, anchor_valid = _masked_patch_pool(
            output.predicted_dynamics_tokens,
            anchor_patch_mask,
        )
        candidates, candidate_valid = _masked_patch_pool(
            output.target_dynamics_tokens,
            output.valid_target_patch_mask,
        )
        batch_size, horizons, dimension = predicted.shape
        anchors = batch_size * horizons
        sequence_ids = tuple(
            batch.sequence_ids[batch_index]
            for batch_index in range(batch_size)
            for _ in range(horizons)
        )
        track_ids = tuple(
            batch.track_ids[batch_index]
            for batch_index in range(batch_size)
            for _ in range(horizons)
        )
        reference_timestamps = batch.reference_timestamps_s.repeat_interleave(horizons)
        horizon_delta_t = batch.horizon_delta_t_s.reshape(anchors)
        desired_future_timestamps = reference_timestamps + horizon_delta_t
        candidate_timestamps = batch.future_timestamps_s.reshape(anchors)
        positive_weights = build_horizon_positive_weights(
            reference_timestamps,
            candidate_timestamps,
            horizon_delta_t,
            sequence_ids,
            sequence_ids,
            track_ids,
            track_ids,
            tolerance_s=self.config.loss.nce_positive_tolerance_s,
            distance_weighting=True,
        )
        return within_track_nce_loss(
            predicted.reshape(anchors, dimension),
            candidates.reshape(anchors, dimension),
            positive_weights,
            sequence_ids,
            sequence_ids,
            track_ids,
            track_ids,
            desired_future_timestamps,
            candidate_timestamps,
            candidate_mask=(
                batch.nce_candidate_mask.reshape(anchors, anchors)
                if batch.nce_candidate_mask is not None
                else None
            ),
            candidate_valid=candidate_valid.reshape(anchors),
            anchor_valid=anchor_valid.reshape(anchors),
            exclusion_window_s=self.config.loss.nce_exclusion_window_s,
            temperature=self.config.loss.nce_temperature,
            min_negatives=self.config.loss.nce_min_negatives,
        )

    def assemble_objective(
        self,
        output: DenseLevelDynamicsOutput,
        batch: LabelFreeBatch,
    ) -> LevelDynamicsObjectiveOutput:
        """Assemble exactly the loss components authorized by the configured arm."""

        config = self.config.loss
        level = dense_cosine_loss(
            output.predicted_level_tokens,
            output.target_level_tokens,
            output.valid_target_patch_mask,
        )
        residual_target = None
        residual = output.predicted_level_tokens.sum() * 0.0
        nce = None
        nce_loss = output.predicted_level_tokens.sum() * 0.0
        visreg = None
        visreg_loss = output.predicted_level_tokens.sum() * 0.0
        if config.objective.uses_temporal_residual:
            residual_target = build_temporal_residual_target(
                output.target_reference_dynamics_tokens,
                output.target_dynamics_tokens,
                output.target_reference_valid_patch_mask,
                output.valid_target_patch_mask,
            )
            residual = temporal_residual_cosine_loss(
                output.predicted_residual_tokens, residual_target
            )
        if config.objective.uses_dynamics_nce:
            nce = self._nce_output(output, batch)
            nce_loss = nce.loss
        if config.objective.uses_residual_visreg:
            visreg = residual_visreg_loss(
                output.dynamics_tokens,
                output.valid_patch_mask,
                generator=self.visreg_generator,
                projections=config.visreg_projections,
                temperature=config.visreg_temperature,
            )
            visreg_loss = visreg.loss
        total = (
            config.level_weight * level
            + config.temporal_residual_weight * residual
            + config.dynamics_nce_weight * nce_loss
            + config.residual_visreg_weight * visreg_loss
        )
        return LevelDynamicsObjectiveOutput(
            loss=total,
            level_loss=level,
            temporal_residual_loss=residual,
            dynamics_nce_loss=nce_loss,
            residual_visreg_loss=visreg_loss,
            residual_target=residual_target,
            nce=nce,
            visreg=visreg,
        )

    def _update_health(
        self, output: DenseLevelDynamicsOutput, objective: LevelDynamicsObjectiveOutput
    ) -> None:
        if not torch.isfinite(objective.loss):
            self.health_state["non_finite_updates"] = (
                int(self.health_state["non_finite_updates"]) + 1
            )
            raise FloatingPointError(
                "Dense Level-Dynamics objective is non-finite; optimizer step was skipped."
            )
        valid = output.valid_patch_mask
        level_values = output.level_tokens[valid]
        dynamics_values = output.dynamics_tokens[valid]
        self.health_state["finite_updates"] = int(self.health_state["finite_updates"]) + 1
        self.health_state["last_level_std"] = (
            float(level_values.std().detach().cpu()) if level_values.numel() else 0.0
        )
        self.health_state["last_dynamics_std"] = (
            float(dynamics_values.std().detach().cpu()) if dynamics_values.numel() else 0.0
        )

    def preflight_nce_batches(
        self,
        batches: Iterable[LabelFreeBatchLike],
        *,
        max_batches: int | None = None,
    ) -> None:
        """Verify bounded NCE coverage without retaining raw high-resolution tensors.

        Callers that will subsequently train must pass a re-iterable source.  This
        method intentionally streams one preflight pass and leaves reopening the
        source to ``train_batches``; it never materializes a batch list.
        """

        if not self.config.loss.objective.uses_dynamics_nce:
            return
        if max_batches is not None and max_batches <= 0:
            raise ValueError("NCE preflight max_batches must be positive when provided.")
        was_training = self.model.training
        self.model.eval()
        total_anchors = 0
        total_valid = 0
        minimum_negatives: int | None = None
        seen_batches = 0
        try:
            with torch.no_grad():
                for value in islice(batches, max_batches):
                    seen_batches += 1
                    typed = self.coerce_batch(value).to(self.device)
                    output = self.model(
                        typed.context_events,
                        typed.future_events,
                        typed.horizon_delta_t_s,
                        context_valid_temporal_mask=typed.context_valid,
                        future_valid_temporal_mask=typed.future_valid,
                    )
                    nce = self._nce_output(output, typed)
                    total_anchors += nce.valid_anchor_mask.numel()
                    total_valid += int(nce.valid_anchor_mask.sum().cpu())
                    if bool(nce.valid_anchor_mask.any()):
                        observed = int(nce.negatives_per_anchor[nce.valid_anchor_mask].min().cpu())
                        minimum_negatives = (
                            observed
                            if minimum_negatives is None
                            else min(minimum_negatives, observed)
                        )
        finally:
            self.model.train(was_training)
        if seen_batches == 0:
            raise RuntimeError("NCE preflight received no label-free batches.")
        valid_fraction = total_valid / max(total_anchors, 1)
        required_fraction = self.config.loss.nce_min_valid_anchor_fraction
        required_negatives = self.config.loss.nce_min_negatives
        if valid_fraction < required_fraction or (minimum_negatives or 0) < required_negatives:
            raise RuntimeError(
                "NCE preflight failed before optimizer step: "
                f"valid-anchor fraction={valid_fraction:.3f} (required {required_fraction:.3f}), "
                f"minimum negatives={minimum_negatives or 0} (required {required_negatives})."
            )
        self.nce_preflight_passed = True
        self.health_state["nce_preflight_passed"] = True
        self.health_state["nce_preflight_valid_anchor_fraction"] = valid_fraction
        self.health_state["nce_preflight_min_negatives"] = minimum_negatives

    def train_step(self, batch: LabelFreeBatchLike) -> dict[str, TrainMetricValue]:
        """Run one deterministic bounded update, preflighting NCE before any step."""

        if self.update_count >= self.config.total_updates:
            raise RuntimeError(
                "Dense Level-Dynamics total update budget is already exhausted; "
                "a resumed run cannot execute additional updates."
            )
        typed = self.coerce_batch(batch).to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(
            typed.context_events,
            typed.future_events,
            typed.horizon_delta_t_s,
            context_valid_temporal_mask=typed.context_valid,
            future_valid_temporal_mask=typed.future_valid,
        )
        objective = self.assemble_objective(output, typed)
        if self.config.loss.objective.uses_dynamics_nce and not self.nce_preflight_passed:
            assert objective.nce is not None
            validate_nce_preflight(
                objective.nce,
                minimum_valid_anchor_fraction=self.config.loss.nce_min_valid_anchor_fraction,
                minimum_negatives=self.config.loss.nce_min_negatives,
            )
            self.nce_preflight_passed = True
            self.health_state["nce_preflight_passed"] = True
        self._update_health(output, objective)
        objective.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        momentum = self.model.update_target_ema(
            self.update_count, total_updates=self.config.total_updates
        )
        self.update_count += 1
        result = {
            "loss": float(objective.loss.detach().cpu()),
            "level_loss": float(objective.level_loss.detach().cpu()),
            "temporal_residual_loss": float(objective.temporal_residual_loss.detach().cpu()),
            "dynamics_nce_loss": float(objective.dynamics_nce_loss.detach().cpu()),
            "residual_visreg_loss": float(objective.residual_visreg_loss.detach().cpu()),
            "ema_momentum": momentum,
            "update": float(self.update_count),
        }
        if objective.nce is not None:
            result["nce_valid_anchor_fraction"] = float(objective.nce.valid_anchor_fraction.cpu())
            result["nce_negatives_per_valid_anchor"] = float(
                objective.nce.mean_negatives_per_valid_anchor.cpu()
            )
            result["nce_negatives_per_anchor"] = [
                int(value) for value in objective.nce.negatives_per_anchor.detach().cpu().tolist()
            ]
        if objective.residual_target is not None:
            valid = objective.residual_target.valid_mask
            result["raw_residual_norm"] = float(
                objective.residual_target.raw_norm[valid].mean().cpu() if bool(valid.any()) else 0.0
            )
        return result

    def train_batches(
        self,
        batches: Iterable[LabelFreeBatchLike],
        *,
        max_updates: int | None = None,
    ) -> list[dict[str, TrainMetricValue]]:
        """Train a bounded iterable and return compact per-update metrics."""

        rows: list[dict[str, TrainMetricValue]] = []
        if max_updates is not None and max_updates <= 0:
            raise ValueError("max_updates must be positive.")
        remaining = self.config.total_updates - self.update_count
        if remaining <= 0:
            raise RuntimeError(
                "Dense Level-Dynamics total update budget is already exhausted; "
                "a resumed run cannot execute additional updates."
            )
        budget = remaining if max_updates is None else max_updates
        if budget > remaining:
            raise ValueError(
                "Requested updates exceed the remaining configured budget: "
                f"requested={budget}, remaining={remaining}, total={self.config.total_updates}."
            )
        if self.config.loss.objective.uses_dynamics_nce and not self.nce_preflight_passed:
            if isinstance(batches, Iterator):
                raise TypeError(
                    "NCE training requires a re-iterable batch source for a streaming "
                    "preflight pass; pass a DataLoader or sequence, not a one-shot iterator."
                )
            self.preflight_nce_batches(batches, max_batches=budget)
        for batch in islice(batches, budget):
            if len(rows) >= budget:
                break
            rows.append(self.train_step(batch))
        if not rows:
            raise RuntimeError("Dense Level-Dynamics trainer received no batches.")
        self.epoch += 1
        return rows

    def checkpoint_state(self, provenance: LabelFreeManifestProvenance) -> dict[str, Any]:
        """Return complete deterministic resume state and explicit transfer boundaries."""

        transfer = self.model.downstream_backbone_payload()
        full_encoder_state = self.model.online_representation.encoder.state_dict()
        transfer_keys = set(transfer["online_encoder_state_dict"])
        resume_only_encoder_state = {
            key: value.detach().clone()
            for key, value in full_encoder_state.items()
            if key not in transfer_keys
        }
        return {
            "artifact_type": "dense_level_dynamics_jepa_checkpoint_v1",
            "online_encoder_state_dict": transfer["online_encoder_state_dict"],
            "online_encoder_config": transfer["online_encoder_config"],
            # Query pooling is excluded from downstream transfer but retained here
            # solely so a resumed wrapper state is bitwise equivalent to its source.
            "online_encoder_resume_only_state_dict": resume_only_encoder_state,
            "online_level_head_state_dict": (
                self.model.online_representation.level_head.state_dict()
            ),
            "online_dynamics_head_state_dict": (
                self.model.online_representation.dynamics_head.state_dict()
            ),
            "target_representation_state_dict": self.model.target_representation.state_dict(),
            "predictor_state_dict": self.model.predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "rng_state": _capture_rng_state(self.visreg_generator),
            "epoch": self.epoch,
            "update_count": self.update_count,
            "objective": self.config.loss.objective.value,
            "resolved_config": asdict(self.config),
            "config_hash": _config_hash(self.config),
            "matched_manifest_hash": provenance.matched_manifest_hash,
            "dataset_hashes": dict(provenance.dataset_hashes),
            "split_hash": provenance.split_hash,
            "sampler_order_hash": provenance.sampler_order_hash,
            "selection_rule": provenance.selection_rule,
            "health_state": dict(self.health_state),
            "nce_preflight_passed": self.nce_preflight_passed,
            "label_family_provenance": dict(provenance.label_family_provenance),
            "uses_dense_disk_cache": False,
            "uses_ttc_labels": False,
            "uses_depth_or_3d": False,
            "uses_category_labels": False,
            "uses_boxes": False,
            "uses_masks": False,
            "uses_rgb": False,
            "uses_evttc": False,
            "downstream_transfer_state": transfer,
        }

    def save_checkpoint(self, path: str | Path, provenance: LabelFreeManifestProvenance) -> None:
        """Atomically save a checkpoint whose target/predictor state is resume-only."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self.checkpoint_state(provenance), temporary)
        temporary.replace(destination)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore an exact bounded run, including optimizer, scheduler and RNG state."""

        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("Dense Level-Dynamics resume checkpoint must be a mapping.")
        if payload.get("artifact_type") != "dense_level_dynamics_jepa_checkpoint_v1":
            raise ValueError("Resume checkpoint has the wrong Dense Level-Dynamics artifact type.")
        if payload.get("config_hash") != _config_hash(self.config):
            raise ValueError(
                "Resume checkpoint config hash differs from this trainer configuration."
            )
        if payload.get("objective") != self.config.loss.objective.value:
            raise ValueError(
                "Resume checkpoint objective arm differs from this trainer configuration."
            )
        encoder_state = payload.get("online_encoder_state_dict")
        encoder_config = payload.get("online_encoder_config")
        if not isinstance(encoder_state, Mapping) or not isinstance(encoder_config, Mapping):
            raise ValueError("Resume checkpoint lacks exact online backbone state/config.")
        self.model.online_representation.encoder.load_exact_backbone_state_dict(
            encoder_state,
            encoder_config,
        )
        resume_only_encoder_state = payload.get("online_encoder_resume_only_state_dict")
        if not isinstance(resume_only_encoder_state, Mapping):
            raise ValueError("Resume checkpoint lacks online_encoder_resume_only_state_dict.")
        expected_resume_only = set(self.model.online_representation.encoder.state_dict()) - set(
            encoder_state
        )
        received_resume_only = set(str(key) for key in resume_only_encoder_state)
        if received_resume_only != expected_resume_only:
            raise ValueError(
                "Resume-only encoder state keys differ; "
                f"missing={sorted(expected_resume_only - received_resume_only)}, "
                f"extra={sorted(received_resume_only - expected_resume_only)}."
            )
        resumed_encoder = self.model.online_representation.encoder.load_state_dict(
            resume_only_encoder_state,
            strict=False,
        )
        if resumed_encoder.unexpected_keys:
            raise RuntimeError(
                "Unexpected resume-only encoder keys: " + repr(resumed_encoder.unexpected_keys)
            )
        required_states = {
            "online_level_head_state_dict": self.model.online_representation.level_head,
            "online_dynamics_head_state_dict": self.model.online_representation.dynamics_head,
            "target_representation_state_dict": self.model.target_representation,
            "predictor_state_dict": self.model.predictor,
        }
        for key, module in required_states.items():
            state = payload.get(key)
            if not isinstance(state, Mapping):
                raise ValueError(f"Resume checkpoint lacks {key}.")
            module.load_state_dict(state, strict=True)
        optimizer_state = payload.get("optimizer_state_dict")
        scheduler_state = payload.get("scheduler_state_dict")
        rng_state = payload.get("rng_state")
        if not isinstance(optimizer_state, Mapping) or not isinstance(scheduler_state, Mapping):
            raise ValueError("Resume checkpoint lacks optimizer/scheduler state.")
        if not isinstance(rng_state, Mapping):
            raise ValueError("Resume checkpoint lacks RNG state.")
        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(scheduler_state)
        _restore_rng_state(rng_state, self.visreg_generator)
        self.epoch = int(payload.get("epoch", 0))
        self.update_count = int(payload.get("update_count", 0))
        if not 0 <= self.update_count <= self.config.total_updates:
            raise ValueError(
                "Resume checkpoint update_count lies outside the configured total update budget."
            )
        self.nce_preflight_passed = bool(payload.get("nce_preflight_passed", False))
        health = payload.get("health_state", {})
        if not isinstance(health, Mapping):
            raise ValueError("Resume checkpoint health_state must be a mapping.")
        self.health_state = dict(health)
        self.model.target_representation.requires_grad_(False)
        self.model.target_representation.eval()
        return dict(payload)


__all__ = [
    "EAPHighResJEPATrainer",
    "EAPHighResJEPATrainerConfig",
    "LABEL_FAMILY_PROVENANCE_FIELDS",
    "LabelFreeBatch",
    "LabelFreeDataset",
    "LabelFreeManifestProvenance",
    "PROHIBITED_SSL_LABEL_KEY_FRAGMENTS",
    "load_signed_label_free_manifest",
    "reject_prohibited_label_keys",
]
