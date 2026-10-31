"""Fail-closed campaign batching for the first causal RGB temporal candidate.

This module is deliberately separate from the single-frame campaign runner.  It
turns the frozen D4096-v5 current-frame population into stateless detector calls;
it never caches backbone features and it never reads labels for history frames.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pandas as pd
import torch

from e_jepa_ttc.detection.campaign_training import ResumablePermutationSampler
from e_jepa_ttc.detection.eap_rgb_data import (
    EAPRGBDataset,
    RawRGBSample,
    safe_dataset_path,
    sha256_file,
)
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample
from e_jepa_ttc.detection.temporal_data import TemporalRGBIndex
from e_jepa_ttc.detection.temporal_detector import (
    TemporalBatchContext,
    TemporalFrameIdentity,
)

TemporalCampaignVariant = Literal[
    "current_only", "current_repeated", "correct_past_1", "correct_past_2"
]

TEMPORAL_CAMPAIGN_SCHEMA = "eap_detection_temporal_campaign_v1"
TEMPORAL_MAXIMUM_HISTORY_AGE_US = 500_000
D4096_V5_MANIFEST_SHA256 = (
    "9beeb6a3c5b76bf994808eee0b3f0dfa2f7339f139bd120dbfa6318e414945f1"
)
FULL_SELECTION_V1_SHA256 = (
    "e4303685e39f635a47b76921b9c5f38821e8715f3aa5f202c73a2c32dae1a497"
)
TEMPORAL_MEDIA_ADMISSION_V2_SHA256 = (
    "1e00d240adba04011063341a3d8d6cea6c0afd3bce310a18d4a8710701e43d38"
)
HISTORY_MEDIA_RECEIPT_V2_SHA256 = (
    "0804d352451fbf5296dc95aa4f61c4c7335bf3b53a0d71181196ec7a9f4fb9c5"
)


class TemporalMediaError(RuntimeError):
    """A required current/history image failed its explicit media contract."""


class TemporalSampleReader(Protocol):
    """Token-addressable raw RGB reader with declared label behavior."""

    labels_enabled: bool

    def read(self, sample_token: str) -> RawRGBSample:
        """Read exactly one token or raise; implementations must not substitute."""
        raise NotImplementedError


class _LabelFreeEAPRGBDataset(EAPRGBDataset):
    def _frame_labels(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        return []


class EAPTokenReader:
    """Small token adapter over the audited bounded eAP reader."""

    def __init__(self, dataset: EAPRGBDataset, *, labels_enabled: bool) -> None:
        self.dataset = dataset
        self.labels_enabled = labels_enabled
        self._indices = {token: index for index, token in enumerate(dataset.sample_tokens)}

    def read(self, sample_token: str) -> RawRGBSample:
        try:
            index = self._indices[sample_token]
        except KeyError as error:
            raise TemporalMediaError(f"Token is outside this reader: {sample_token}") from error
        try:
            return self.dataset[index]
        except Exception as error:
            raise TemporalMediaError(
                f"RGB read failed without substitution: {sample_token}"
            ) from error


@dataclass(frozen=True)
class TemporalCampaignConfig:
    """Preregistered, strict settings for one temporal campaign condition."""

    schema: str
    variant: TemporalCampaignVariant
    producer_version: str
    maximum_history_age_us: int = TEMPORAL_MAXIMUM_HISTORY_AGE_US
    width: int = 640
    height: int = 384
    include_velocity: bool = True
    generation: int = 0
    media_failure_policy: str = "fail"

    def __post_init__(self) -> None:
        if self.schema != TEMPORAL_CAMPAIGN_SCHEMA:
            raise ValueError("Unknown temporal campaign schema")
        if self.variant not in (
            "current_only",
            "current_repeated",
            "correct_past_1",
            "correct_past_2",
        ):
            raise ValueError("Unknown temporal campaign variant")
        if not self.producer_version:
            raise ValueError("producer_version must be nonempty")
        if self.maximum_history_age_us != TEMPORAL_MAXIMUM_HISTORY_AGE_US:
            raise ValueError("First temporal campaign fixes maximum history age at 500000 us")
        if self.width <= 0 or self.height <= 0 or self.width % 64 or self.height % 64:
            raise ValueError("Temporal image dimensions must be positive multiples of 64")
        if type(self.include_velocity) is not bool:
            raise ValueError("include_velocity must be boolean")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if self.media_failure_policy != "fail":
            raise ValueError("Temporal media failures cannot be skipped or replaced")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TemporalCampaignConfig:
        """Parse a config while refusing ignored or misspelled settings."""
        allowed = {
            "schema",
            "variant",
            "producer_version",
            "maximum_history_age_us",
            "width",
            "height",
            "include_velocity",
            "generation",
            "media_failure_policy",
        }
        unknown = set(value) - allowed
        missing = {"schema", "variant", "producer_version"} - set(value)
        if unknown or missing:
            raise ValueError(
                "Temporal config keys differ "
                f"(unknown={sorted(unknown)}, missing={sorted(missing)})"
            )
        return cls(**cast(Any, dict(value)))

    @property
    def detector_control(self) -> Literal["current_only", "current_repeated", "correct_past"]:
        if self.variant.startswith("correct_past"):
            return "correct_past"
        return cast(Any, self.variant)

    @property
    def history_limit(self) -> int:
        return {
            "current_only": 0,
            "current_repeated": 2,
            "correct_past_1": 1,
            "correct_past_2": 2,
        }[self.variant]


@dataclass(frozen=True)
class HistoryMediaAdmission:
    """Exact readable-byte admission for one historical RGB member."""

    sample_token: str
    readable: bool
    member_sha256: str | None
    reason: str

    def __post_init__(self) -> None:
        if not self.sample_token:
            raise ValueError("History admission token must be nonempty")
        if not self.reason:
            raise ValueError("History admission reason must be nonempty")
        if self.readable != (self.member_sha256 is not None):
            raise ValueError("Readable history admission requires exactly one member hash")
        if self.member_sha256 is not None and (
            len(self.member_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.member_sha256)
        ):
            raise ValueError("History member SHA-256 is invalid")


@dataclass(frozen=True)
class TemporalAdmissionLedger:
    """Hash-bound media decisions shared by every temporal campaign variant."""

    history_media: Mapping[str, HistoryMediaAdmission]
    current_rows: Mapping[str, Mapping[str, object]]
    source_hashes: Mapping[str, str]


def load_temporal_admission_ledger(
    temporal_media_path: str | Path,
    history_media_path: str | Path,
) -> TemporalAdmissionLedger:
    """Load the exact 28,054-current/32,080-history admission receipts."""
    temporal_path = Path(temporal_media_path).resolve()
    history_path = Path(history_media_path).resolve()
    temporal_hash = sha256_file(temporal_path)
    history_hash = sha256_file(history_path)
    if temporal_hash != TEMPORAL_MEDIA_ADMISSION_V2_SHA256:
        raise ValueError("Temporal media admission SHA-256 differs")
    if history_hash != HISTORY_MEDIA_RECEIPT_V2_SHA256:
        raise ValueError("History media receipt SHA-256 differs")
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if (
        temporal.get("schema") != "eap_temporal_media_admission_v1"
        or temporal.get("revision") != "temporal_d4096_v5_admission_v2"
        or temporal.get("status") != "PASS"
        or temporal.get("maximum_history_age_us") != TEMPORAL_MAXIMUM_HISTORY_AGE_US
        or temporal.get("d4096_v5_manifest_sha256") != D4096_V5_MANIFEST_SHA256
        or temporal.get("full_selection_sha256") != FULL_SELECTION_V1_SHA256
    ):
        raise ValueError("Temporal media admission contract differs")
    if history.get("schema") != "eap_temporal_history_media_receipt_v1":
        raise ValueError("History media receipt schema differs")
    history_media: dict[str, HistoryMediaAdmission] = {}
    for token, record in history["history_media"].items():
        if token != record.get("sample_token"):
            raise ValueError("History media receipt token key differs from its record")
        readable = record.get("status") == "PASS"
        history_media[token] = HistoryMediaAdmission(
            sample_token=token,
            readable=readable,
            member_sha256=record.get("member_sha256") if readable else None,
            reason=str(record.get("readability") or record.get("reason") or record.get("status")),
        )
    if len(history_media) != 32_080:
        raise ValueError("History media receipt population count differs")
    current_rows: dict[str, Mapping[str, object]] = {}
    expected_counts = {"train": 4096, "selection": 23_958}
    for role, count in expected_counts.items():
        entries = temporal["roles"][role]["entries"]
        if len(entries) != count:
            raise ValueError("Temporal current admission population count differs")
        for entry in entries:
            token = entry.get("current_token")
            if not isinstance(token, str) or token in current_rows:
                raise ValueError("Temporal current admission identity is invalid or duplicated")
            if entry.get("shared_history_mask_across_variants") is not True:
                raise ValueError("Temporal history mask is not shared across variants")
            if entry.get("substitute_history_after_failure") is not False:
                raise ValueError("Temporal admission permits forbidden history substitution")
            current_rows[token] = entry
    if len(current_rows) != 28_054:
        raise ValueError("Temporal current admission union count differs")
    return TemporalAdmissionLedger(
        history_media=history_media,
        current_rows=current_rows,
        source_hashes={
            "temporal_media_admission": temporal_hash,
            "history_media_receipt": history_hash,
        },
    )


@dataclass(frozen=True)
class TemporalCampaignItem:
    """One prepared current frame and two explicit history slots."""

    current_image: torch.Tensor
    calibration: torch.Tensor
    image_size: torch.Tensor
    target: dict[str, torch.Tensor] | None
    current_identity: TemporalFrameIdentity
    history_images: torch.Tensor | None
    history_ages_us: torch.Tensor | None
    history_mask: torch.Tensor | None
    history_identities: tuple[TemporalFrameIdentity | None, TemporalFrameIdentity | None]
    current_member_sha256: str
    history_member_sha256: tuple[str | None, str | None]
    canonical_history_tokens: tuple[str | None, str | None]
    history_slot_reasons: tuple[str, str]


@dataclass(frozen=True)
class TemporalCampaignBatch:
    """Batch whose call arguments match :class:`TemporalDetectorAdapter`."""

    current_images: torch.Tensor
    calibs: torch.Tensor
    targets: tuple[dict[str, torch.Tensor], ...] | None
    image_sizes: torch.Tensor
    temporal_context: TemporalBatchContext
    history_images: torch.Tensor | None
    history_ages_us: torch.Tensor | None
    history_mask: torch.Tensor | None
    history_limit: int
    provenance: dict[str, object]

    def detector_kwargs(self) -> dict[str, object]:
        """Return exact keyword-only arguments for ``TemporalDetectorAdapter.forward``."""
        return {
            "temporal_context": self.temporal_context,
            "history_images": self.history_images,
            "history_ages_us": self.history_ages_us,
            "history_mask": self.history_mask,
            "history_limit": self.history_limit,
        }


class TemporalCampaignDataset:
    """Prepare D4096-v5 current frames and causal, label-free RGB histories."""

    def __init__(
        self,
        *,
        current_tokens: Sequence[str],
        exposure_rows: Sequence[Mapping[str, object]],
        current_reader: TemporalSampleReader,
        history_reader: TemporalSampleReader,
        history_admission: Mapping[str, HistoryMediaAdmission],
        config: TemporalCampaignConfig,
        current_labels: bool,
        role: str,
        manifest_sha256: str,
        token_generations: Mapping[str, int] | None = None,
        source_hashes: Mapping[str, str] | None = None,
        admission_rows: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        if role not in {"train", "monitor_selection", "full_selection"}:
            raise ValueError("Unknown temporal campaign population")
        if not current_tokens or len(set(current_tokens)) != len(current_tokens):
            raise ValueError("Current token population must be nonempty and unique")
        if current_reader.labels_enabled != current_labels:
            raise ValueError("Current reader label mode differs from the requested role mode")
        if history_reader.labels_enabled:
            raise ValueError("History reader must guarantee that label files are never opened")
        if role != "train" and current_labels:
            raise ValueError("Evaluation current frames must also be label-free")
        self.current_tokens = tuple(current_tokens)
        self.index = TemporalRGBIndex(exposure_rows)
        missing = set(self.current_tokens) - set(self.index.sample_tokens)
        if missing:
            raise ValueError(
                f"Current tokens are absent from canonical exposures: {sorted(missing)}"
            )
        self._rows = {str(row["sample_token"]): row for row in exposure_rows}
        generations = dict(token_generations or {})
        unknown_generations = set(generations) - set(self.index.sample_tokens)
        if unknown_generations:
            raise ValueError("Generation mapping contains unknown tokens")
        for token, generation in generations.items():
            if type(generation) is not int or generation < 0:
                raise ValueError(f"Invalid generation for {token}")
        self._generations = generations
        self.current_reader = current_reader
        self.history_reader = history_reader
        self.history_admission = dict(history_admission)
        self.config = config
        self.current_labels = current_labels
        self.role = role
        self.manifest_sha256 = manifest_sha256
        self.source_hashes = dict(source_hashes or {"population_contract": manifest_sha256})
        self.admission_rows = dict(admission_rows or {})
        if self.admission_rows and set(self.current_tokens) - set(self.admission_rows):
            raise ValueError("Temporal admission ledger does not cover every current token")

    def __len__(self) -> int:
        return len(self.current_tokens)

    def _generation(self, token: str) -> int:
        return self._generations.get(token, self.config.generation)

    def _read_checked(
        self,
        reader: TemporalSampleReader,
        token: str,
        *,
        history: bool,
    ) -> RawRGBSample:
        try:
            sample = reader.read(token)
        except Exception as error:
            if isinstance(error, TemporalMediaError):
                raise
            kind = "history" if history else "current"
            raise TemporalMediaError(f"Required {kind} RGB failed: {token}") from error
        row = self._rows[token]
        if (
            sample.sample_token != token
            or sample.sequence_id != str(row["sequence_id"])
            or sample.exposure_start_us
            != int(cast(Any, row["rgb_exposure_start_timestamp_us"]))
            or sample.exposure_end_us
            != int(cast(Any, row["rgb_exposure_end_timestamp_us"]))
        ):
            raise TemporalMediaError(f"RGB identity/timestamps differ from exposure index: {token}")
        if history:
            admission = self.history_admission.get(token)
            if admission is None or not admission.readable:
                raise TemporalMediaError(f"History lacks explicit readable admission: {token}")
            if admission.member_sha256 != sample.member_sha256:
                raise TemporalMediaError(f"History RGB bytes differ from admission: {token}")
            if sample.annotations:
                raise TemporalMediaError(
                    "History reader returned labels despite label-free contract"
                )
        return sample

    def __getitem__(self, index: int) -> TemporalCampaignItem:
        token = self.current_tokens[index]
        temporal = self.index.resolve(token, control=self.config.detector_control)
        admission_row = self.admission_rows.get(token)
        if self.config.variant.startswith("correct_past"):
            if admission_row is None:
                raise TemporalMediaError(f"Current has no frozen temporal admission row: {token}")
            canonical = tuple(
                cast(
                    Sequence[str | None],
                    admission_row["canonical_history_tokens_before_media"],
                )
            )
            if canonical != temporal.history_tokens:
                raise TemporalMediaError(f"Canonical history differs from admission: {token}")
            if (
                admission_row["sequence_id"] != temporal.sequence_id
                or admission_row["current_exposure_start_us"]
                != temporal.current_exposure_start_us
                or admission_row["current_exposure_end_us"] != temporal.current_exposure_end_us
                or admission_row["decision_sensor_us"] != temporal.decision_sensor_us
            ):
                raise TemporalMediaError(f"Current exposure differs from admission: {token}")
        current_generation = self._generation(token)
        current_raw = self._read_checked(self.current_reader, token, history=False)
        current = prepare_sample(
            current_raw,
            width=self.config.width,
            height=self.config.height,
            include_velocity=self.config.include_velocity,
        )
        identity = TemporalFrameIdentity(current_raw.sequence_id, token, current_generation)
        if self.config.history_limit == 0:
            return TemporalCampaignItem(
                current.image,
                current.calibration,
                current.image_size,
                current.target if self.current_labels else None,
                identity,
                None,
                None,
                None,
                (None, None),
                current_raw.member_sha256,
                (None, None),
                temporal.history_tokens,
                ("current_only", "current_only"),
            )

        history_tokens = list(temporal.history_tokens)
        history_ages = list(temporal.history_age_us)
        history_mask = list(temporal.history_mask)
        if self.config.variant == "correct_past_1":
            history_tokens[0], history_ages[0], history_mask[0] = None, None, False
        images: list[torch.Tensor] = []
        identities: list[TemporalFrameIdentity | None] = []
        hashes: list[str | None] = []
        reasons: list[str] = []
        for slot, history_token in enumerate(history_tokens):
            if history_token is None:
                images.append(torch.zeros_like(current.image))
                identities.append(None)
                hashes.append(None)
                reasons.append("sequence_or_generation_boundary")
                continue
            age = history_ages[slot]
            if age is None or age < 0:
                raise TemporalMediaError(f"History age is invalid: {history_token}")
            if age > self.config.maximum_history_age_us:
                images.append(torch.zeros_like(current.image))
                identities.append(None)
                hashes.append(None)
                reasons.append("age_exceeds_500000us")
                history_tokens[slot], history_ages[slot], history_mask[slot] = None, None, False
                continue
            if self._generation(history_token) != current_generation:
                # A generation reset is a real causal boundary, not missing media.
                images.append(torch.zeros_like(current.image))
                identities.append(None)
                hashes.append(None)
                reasons.append("generation_boundary")
                history_tokens[slot], history_ages[slot], history_mask[slot] = None, None, False
                continue
            if self.config.variant == "current_repeated":
                history_raw = current_raw
                history_prepared = current
                reasons.append("current_repeated_capacity_control")
            else:
                admission = self.history_admission.get(history_token)
                if admission is None:
                    raise TemporalMediaError(
                        f"History has no explicit media admission decision: {history_token}"
                    )
                if not admission.readable:
                    images.append(torch.zeros_like(current.image))
                    identities.append(None)
                    hashes.append(None)
                    reasons.append(admission.reason)
                    history_tokens[slot], history_ages[slot], history_mask[slot] = (
                        None,
                        None,
                        False,
                    )
                    continue
                history_raw = self._read_checked(self.history_reader, history_token, history=True)
                # Defensive erasure preserves the no-history-label invariant for custom readers.
                history_prepared = prepare_sample(
                    replace(history_raw, annotations=[]),
                    width=self.config.width,
                    height=self.config.height,
                    include_velocity=self.config.include_velocity,
                )
                reasons.append("readable_media_admission")
            if history_prepared.image.shape != current.image.shape:
                raise TemporalMediaError("Prepared history and current shapes differ")
            images.append(history_prepared.image)
            identities.append(
                TemporalFrameIdentity(current_raw.sequence_id, history_token, current_generation)
            )
            hashes.append(history_raw.member_sha256)
        mask_tensor = torch.tensor(history_mask, dtype=torch.bool)
        age_tensor = torch.tensor(
            [0 if age is None else age for age in history_ages], dtype=torch.float32
        )
        if admission_row is not None and self.config.variant.startswith("correct_past"):
            expected_mask = list(cast(Sequence[bool], admission_row["history_mask"]))
            expected_age = list(cast(Sequence[int | None], admission_row["history_age_us"]))
            canonical_tokens = cast(
                Sequence[str | None], admission_row["canonical_history_tokens_before_media"]
            )
            for slot, canonical_token in enumerate(canonical_tokens):
                if (
                    canonical_token is not None
                    and self._generation(canonical_token) != current_generation
                ):
                    expected_mask[slot], expected_age[slot] = False, None
            if self.config.variant == "correct_past_1":
                expected_mask[0], expected_age[0] = False, None
            if history_mask != expected_mask or [
                None if not valid else history_ages[slot]
                for slot, valid in enumerate(history_mask)
            ] != expected_age:
                raise TemporalMediaError(
                    f"Runtime history mask/ages differ from admission: {token}"
                )
            reasons = list(cast(Sequence[str], admission_row["history_mask_reasons"]))
            if self.config.variant == "correct_past_1":
                reasons[0] = "disabled_by_history_limit_1"
        return TemporalCampaignItem(
            current.image,
            current.calibration,
            current.image_size,
            current.target if self.current_labels else None,
            identity,
            torch.stack(images),
            age_tensor,
            mask_tensor,
            cast(Any, tuple(identities)),
            current_raw.member_sha256,
            cast(Any, tuple(hashes)),
            temporal.history_tokens,
            cast(Any, tuple(reasons)),
        )

    def collate(self, indices: Sequence[int]) -> TemporalCampaignBatch:
        """Read and collate exactly the requested indices with no feature cache."""
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Temporal batch indices must be nonempty and unique")
        items = [self[index] for index in indices]
        targets = (
            tuple(cast(dict[str, torch.Tensor], item.target) for item in items)
            if self.current_labels
            else None
        )
        histories = (
            torch.stack([cast(torch.Tensor, item.history_images) for item in items])
            if self.config.history_limit
            else None
        )
        ages = (
            torch.stack([cast(torch.Tensor, item.history_ages_us) for item in items])
            if self.config.history_limit
            else None
        )
        masks = (
            torch.stack([cast(torch.Tensor, item.history_mask) for item in items])
            if self.config.history_limit
            else None
        )
        context = TemporalBatchContext(
            producer_version=self.config.producer_version,
            control=self.config.detector_control,
            current=tuple(item.current_identity for item in items),
            history=tuple(item.history_identities for item in items),
        )
        context.validate(
            batch_size=len(items),
            producer_version=self.config.producer_version,
            history_mask=masks,
            history_ages_us=ages,
            history_limit=self.config.history_limit,
        )
        return TemporalCampaignBatch(
            current_images=torch.stack([item.current_image for item in items]),
            calibs=torch.stack([item.calibration for item in items]),
            targets=targets,
            image_sizes=torch.stack([item.image_size for item in items]),
            temporal_context=context,
            history_images=histories,
            history_ages_us=ages,
            history_mask=masks,
            history_limit=self.config.history_limit,
            provenance={
                "schema": "eap_detection_temporal_batch_v1",
                "role": self.role,
                "variant": self.config.variant,
                "manifest_sha256": self.manifest_sha256,
                "source_hashes": self.source_hashes,
                "producer_version": self.config.producer_version,
                "maximum_history_age_us": self.config.maximum_history_age_us,
                "current_labels": self.current_labels,
                "history_labels": False,
                "feature_cache": False,
                "current_tokens": [item.current_identity.frame_id for item in items],
                "current_member_sha256": [item.current_member_sha256 for item in items],
                "history_member_sha256": [list(item.history_member_sha256) for item in items],
                "canonical_history_tokens": [
                    list(item.canonical_history_tokens) for item in items
                ],
                "history_slot_reasons": [list(item.history_slot_reasons) for item in items],
            },
        )


class TemporalCampaignSampler:
    """Checkpointable sampler bound to one ordered temporal population."""

    def __init__(self, dataset: TemporalCampaignDataset, *, seed: int) -> None:
        self.dataset = dataset
        self._sampler = ResumablePermutationSampler(len(dataset), seed)

    def next_batch(self, batch_size: int) -> tuple[list[int], TemporalCampaignBatch]:
        indices = self._sampler.next_batch(batch_size)
        if len(set(indices)) != len(indices):
            raise ValueError("Batch crossed an epoch with duplicate current identities")
        return indices, self.dataset.collate(indices)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "eap_detection_temporal_sampler_state_v1",
            "manifest_sha256": self.dataset.manifest_sha256,
            "variant": self.dataset.config.variant,
            "current_tokens": self.dataset.current_tokens,
            "sampler": self._sampler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "schema",
            "manifest_sha256",
            "variant",
            "current_tokens",
            "sampler",
        }
        if set(state) != expected:
            raise ValueError("Temporal sampler state keys differ")
        if (
            state["schema"] != "eap_detection_temporal_sampler_state_v1"
            or state["manifest_sha256"] != self.dataset.manifest_sha256
            or state["variant"] != self.dataset.config.variant
            or tuple(cast(Sequence[str], state["current_tokens"]))
            != self.dataset.current_tokens
        ):
            raise ValueError("Temporal sampler state belongs to another campaign")
        self._sampler.load_state_dict(cast(Mapping[str, Any], state["sampler"]))


def build_v5_temporal_campaign(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    population: Literal["train", "monitor_selection", "full_selection"],
    config: TemporalCampaignConfig,
    admission_ledger: TemporalAdmissionLedger,
    current_labels: bool,
    full_selection_path: str | Path | None = None,
    max_label_sequences: int = 24,
    max_open_shards: int = 2,
) -> TemporalCampaignDataset:
    """Build one frozen train/monitor/full population from hash-bound artifacts."""
    manifest_path = Path(manifest_path).resolve()
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != D4096_V5_MANIFEST_SHA256:
        raise ValueError("Temporal campaign requires the exact readable D4096-v5 manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "eap_detection_campaign_manifest_v1":
        raise ValueError("Unexpected D4096 manifest schema")
    if current_labels != (population == "train"):
        raise ValueError("Only train currents carry labels; both evaluation populations do not")
    source_hashes = {
        "d4096_v5_manifest": manifest_hash,
        **admission_ledger.source_hashes,
    }
    if population == "train":
        current_tokens = manifest["roles"]["train"]["sample_tokens"]
        sequences = manifest["roles"]["train"]["sequence_ids"]
    elif population == "monitor_selection":
        current_tokens = manifest["roles"]["selection"]["sample_tokens"]
        sequences = manifest["roles"]["selection"]["sequence_ids"]
    else:
        if full_selection_path is None:
            raise ValueError("full_selection population requires its frozen token artifact")
        selection_path = Path(full_selection_path).resolve()
        selection_hash = sha256_file(selection_path)
        if selection_hash != FULL_SELECTION_V1_SHA256:
            raise ValueError("Full-selection token artifact SHA-256 differs")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            selection.get("schema") != "eap_detection_full_selection_v1"
            or selection.get("dataset_revision") != manifest.get("dataset_revision")
            or selection.get("source_media_sha256") != manifest.get("source_media_sha256")
            or selection.get("role") != "selection"
        ):
            raise ValueError("Full-selection artifact differs from the D4096-v5 contract")
        current_tokens = selection["sample_tokens"]
        sequences = selection["sequence_ids"]
        if not set(manifest["roles"]["selection"]["sample_tokens"]).issubset(
            current_tokens
        ):
            raise ValueError("D4096 monitor selection is not a subset of full selection")
        source_hashes["full_selection_v1"] = selection_hash
    expected_count = {"train": 4096, "monitor_selection": 1024, "full_selection": 23958}[
        population
    ]
    if len(current_tokens) != expected_count or len(set(current_tokens)) != expected_count:
        raise ValueError("Frozen temporal current-frame population count differs")
    root = Path(dataset_root).resolve()
    media_path = safe_dataset_path(root, "data/train.parquet")
    if sha256_file(media_path) != manifest["source_media_sha256"]:
        raise ValueError("Canonical exposure table differs from D4096-v5")
    media = pd.read_parquet(media_path, filters=[("sequence_id", "in", sequences)])
    if not media.split.eq("train").all():
        raise ValueError("Temporal exposure index admits only public train rows")
    media = media.sort_values(
        ["sequence_id", "rgb_exposure_start_timestamp_us", "sample_token"],
        kind="stable",
    )
    history_tokens: set[str] = set()
    index = TemporalRGBIndex(media.to_dict("records"))
    if config.variant.startswith("correct_past"):
        for token in current_tokens:
            history_tokens.update(
                item for item in index.resolve(token).history_tokens if item is not None
            )
    current_dataset_cls = EAPRGBDataset if current_labels else _LabelFreeEAPRGBDataset
    current_reader = EAPTokenReader(
        current_dataset_cls(
            root, current_tokens, max_label_sequences=max_label_sequences,
            max_open_shards=max_open_shards,
        ),
        labels_enabled=current_labels,
    )
    history_reader = EAPTokenReader(
        _LabelFreeEAPRGBDataset(
            root,
            sorted(history_tokens) or current_tokens[:1],
            max_label_sequences=max_label_sequences,
            max_open_shards=max_open_shards,
        ),
        labels_enabled=False,
    )
    contract_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return TemporalCampaignDataset(
        current_tokens=current_tokens,
        exposure_rows=media.to_dict("records"),
        current_reader=current_reader,
        history_reader=history_reader,
        history_admission=admission_ledger.history_media,
        config=config,
        current_labels=current_labels,
        role=population,
        manifest_sha256=contract_hash,
        source_hashes=source_hashes,
        admission_rows=admission_ledger.current_rows,
    )
