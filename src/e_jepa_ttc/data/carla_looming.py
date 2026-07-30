"""Lazy, pickle-free access to the CARLA DVS Looming dataset.

The public distribution contains one structured ``events.npy`` array and one
``sim_data.npz`` metadata archive per simulated sequence.  This module never
materializes a second event cache: windows are sliced from NumPy memory maps.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.types import EventBatch, TTCWindowSample
from e_jepa_ttc.data.validation import validate_event_batch
from e_jepa_ttc.utils.io import read_structured, write_structured

CARLA_LOOMING_DATASET_ID = "CARLA_DVS_LOOMING_1406"
CARLA_LOOMING_WIDTH = 640
CARLA_LOOMING_HEIGHT = 480
CARLA_LOOMING_SOURCE_URL = "https://doi.org/10.25377/sussex.29114609.v1"
CARLA_LOOMING_LICENSE = "CC BY 4.0"
CARLA_LOOMING_ARCHIVE_MD5 = "21a3e72a1c1d9c441a7426393f4e545f"
CARLA_LOOMING_ARCHIVE_BYTES = 15_096_280_525
CARLA_LOOMING_EVENT_DTYPE = np.dtype(
    [("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "<u2")]
)

_COLLISION_TYPE_ALIASES = {
    "cars": "car",
    "car": "car",
    "pedestrian": "pedestrian",
    "pedestrians": "pedestrian",
    "none": "none",
    "none_with_traffic": "none_with_traffic",
    "none_with_crossing": "none_with_crossing",
}
_POSITIVE_TYPES = frozenset({"car", "pedestrian"})


@dataclass(frozen=True)
class CarlaLoomingMetadata:
    """Safe scalar metadata for one simulated sequence."""

    collision_type: str
    collision_type_raw: str
    collision: bool
    t_end_ms: int
    dt_ms: float
    relative_velocity_mps: float | None
    object_diameter: float | None


@dataclass(frozen=True)
class CarlaLoomingSequence:
    """Audited CARLA sequence entry with paths relative to the dataset root."""

    sequence_id: str
    example_index: int
    relative_dir: str
    events_filename: str
    metadata_filename: str
    metadata: CarlaLoomingMetadata | None
    num_events: int
    first_event_ms: int | None
    last_event_ms: int | None
    valid: bool
    issues: tuple[str, ...]
    split_group: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize without NumPy scalars or non-finite JSON values."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CarlaLoomingSequence:
        """Reconstruct an entry from a generated manifest."""

        values = dict(payload)
        raw_metadata = values.get("metadata")
        values["metadata"] = (
            CarlaLoomingMetadata(**raw_metadata) if isinstance(raw_metadata, dict) else None
        )
        values["issues"] = tuple(values.get("issues", ()))
        return cls(**values)


def _python_scalar(value: np.ndarray) -> object:
    scalar = value.item()
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def _optional_finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _normalize_collision_type(value: str) -> str:
    return _COLLISION_TYPE_ALIASES.get(value.strip().lower(), value.strip().lower())


def load_carla_looming_metadata(path: str | Path) -> CarlaLoomingMetadata:
    """Load scalar metadata with ``allow_pickle=False`` in every code path.

    Some negative sequences store ``diameter_object`` as an object-array
    ``None``.  Accessing that member correctly fails when pickle is disabled;
    the field is then reported as unavailable instead of enabling pickle.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"coll_type", "t_end", "dt", "vel", "diameter_object"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"CARLA metadata {source} is missing fields: {missing}.")
        raw_type = str(_python_scalar(archive["coll_type"]))
        collision_type = _normalize_collision_type(raw_type)
        t_end_ms = int(_python_scalar(archive["t_end"]))
        dt_ms = float(_python_scalar(archive["dt"]))
        velocity = _optional_finite_float(_python_scalar(archive["vel"]))
        try:
            diameter = _optional_finite_float(_python_scalar(archive["diameter_object"]))
        except ValueError as error:
            if "Object arrays cannot be loaded" not in str(error):
                raise
            diameter = None
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError(f"CARLA metadata {source} has invalid dt={dt_ms!r}.")
    return CarlaLoomingMetadata(
        collision_type=collision_type,
        collision=collision_type in _POSITIVE_TYPES,
        collision_type_raw=raw_type,
        t_end_ms=t_end_ms,
        dt_ms=dt_ms,
        relative_velocity_mps=velocity,
        object_diameter=diameter,
    )


def _example_index(path: Path) -> int | None:
    prefix = "example_"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _event_path(sequence_dir: Path) -> Path:
    candidates = (sequence_dir / "events.npy", sequence_dir / "event.npy")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _validate_event_array_full(events: np.ndarray, *, chunk_events: int) -> list[str]:
    issues: list[str] = []
    previous_timestamp: int | None = None
    for start in range(0, int(events.shape[0]), chunk_events):
        chunk = events[start : start + chunk_events]
        timestamps = chunk["t"]
        if timestamps.size:
            if previous_timestamp is not None and int(timestamps[0]) < previous_timestamp:
                issues.append("timestamps_not_monotonic")
            if np.any(np.diff(timestamps.astype(np.int64, copy=False)) < 0):
                issues.append("timestamps_not_monotonic")
            previous_timestamp = int(timestamps[-1])
        if np.any(chunk["x"] >= CARLA_LOOMING_WIDTH):
            issues.append("x_out_of_bounds")
        if np.any(chunk["y"] >= CARLA_LOOMING_HEIGHT):
            issues.append("y_out_of_bounds")
        if np.any(chunk["p"] > 1):
            issues.append("unsupported_polarity")
        if issues:
            break
    return issues


def scan_carla_looming_root(
    root: str | Path,
    *,
    context_ms: int = 100,
    group_size: int = 25,
    full_event_validation: bool = False,
    validation_chunk_events: int = 4_000_000,
) -> list[CarlaLoomingSequence]:
    """Discover and validate all sequence directories under ``random_spawn``."""

    if context_ms <= 0 or group_size <= 0 or validation_chunk_events <= 0:
        raise ValueError("Context, group size and validation chunk size must be positive.")
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"CARLA DVS Looming root does not exist: {dataset_root}.")
    indexed = [
        (index, path)
        for path in dataset_root.iterdir()
        if (index := _example_index(path)) is not None
    ]
    sequences: list[CarlaLoomingSequence] = []
    for index, sequence_dir in sorted(indexed):
        issues: list[str] = []
        metadata_path = sequence_dir / "sim_data.npz"
        events_path = _event_path(sequence_dir)
        metadata: CarlaLoomingMetadata | None = None
        num_events = 0
        first_event_ms: int | None = None
        last_event_ms: int | None = None
        try:
            metadata = load_carla_looming_metadata(metadata_path)
        except (FileNotFoundError, KeyError, OSError, ValueError) as error:
            issues.append(f"metadata_error:{type(error).__name__}")
        try:
            events = np.load(events_path, mmap_mode="r", allow_pickle=False)
            num_events = int(events.shape[0]) if events.ndim == 1 else 0
            if events.ndim != 1:
                issues.append("events_not_one_dimensional")
            if events.dtype != CARLA_LOOMING_EVENT_DTYPE:
                issues.append(f"unexpected_event_dtype:{events.dtype.str}")
            if num_events == 0:
                issues.append("empty_events")
            elif events.dtype == CARLA_LOOMING_EVENT_DTYPE:
                first_event_ms = int(events["t"][0])
                last_event_ms = int(events["t"][-1])
                if full_event_validation:
                    issues.extend(
                        _validate_event_array_full(
                            events,
                            chunk_events=validation_chunk_events,
                        )
                    )
        except (FileNotFoundError, OSError, ValueError) as error:
            issues.append(f"events_error:{type(error).__name__}")
        if metadata is not None:
            if metadata.collision_type not in set(_COLLISION_TYPE_ALIASES.values()):
                issues.append(f"unsupported_collision_type:{metadata.collision_type}")
            if metadata.t_end_ms <= 0:
                issues.append("nonpositive_t_end")
            if first_event_ms is not None and metadata.t_end_ms - first_event_ms < context_ms:
                issues.append("shorter_than_context")
            if metadata.collision and last_event_ms is not None:
                if last_event_ms > metadata.t_end_ms:
                    issues.append("post_collision_events")
        sequences.append(
            CarlaLoomingSequence(
                sequence_id=f"example_{index}",
                example_index=index,
                relative_dir=f"example_{index}",
                events_filename=events_path.name,
                metadata_filename=metadata_path.name,
                metadata=metadata,
                num_events=num_events,
                first_event_ms=first_event_ms,
                last_event_ms=last_event_ms,
                valid=not issues,
                issues=tuple(dict.fromkeys(issues)),
                split_group=f"generation_block_{index // group_size:04d}",
            )
        )
    if not sequences:
        raise ValueError(f"No example_<index> directories found under {dataset_root}.")
    return sequences


def resolve_carla_sequence_paths(
    root: str | Path,
    sequence: CarlaLoomingSequence,
) -> tuple[Path, Path]:
    """Resolve the event and metadata files for a manifest entry."""

    directory = Path(root) / sequence.relative_dir
    return directory / sequence.events_filename, directory / sequence.metadata_filename


def read_carla_event_window(
    root: str | Path,
    sequence: CarlaLoomingSequence,
    *,
    start_us: int,
    end_us: int,
) -> EventBatch:
    """Read one exact half-open ``[start_us, end_us)`` window from mmap."""

    if start_us < 0 or end_us <= start_us:
        raise ValueError("CARLA event windows require 0 <= start_us < end_us.")
    events_path, _ = resolve_carla_sequence_paths(root, sequence)
    events = np.load(events_path, mmap_mode="r", allow_pickle=False)
    return _carla_event_window_from_array(
        events,
        sequence,
        start_us=start_us,
        end_us=end_us,
        source=events_path,
    )


def _carla_event_window_from_array(
    events: np.ndarray,
    sequence: CarlaLoomingSequence,
    *,
    start_us: int,
    end_us: int,
    source: Path,
) -> EventBatch:
    """Slice an already-open CARLA mmap without reopening it per horizon."""

    if start_us < 0 or end_us <= start_us:
        raise ValueError("CARLA event windows require 0 <= start_us < end_us.")
    if events.ndim != 1 or events.dtype != CARLA_LOOMING_EVENT_DTYPE:
        raise ValueError(f"Unexpected CARLA event array in {source}.")
    timestamps_ms = events["t"]
    left = int(np.searchsorted(timestamps_ms, start_us / 1000.0, side="left"))
    right = int(np.searchsorted(timestamps_ms, end_us / 1000.0, side="left"))
    selected = events[left:right]
    if selected.size == 0:
        return EventBatch.empty(
            width=CARLA_LOOMING_WIDTH,
            height=CARLA_LOOMING_HEIGHT,
            sequence_id=sequence.sequence_id,
            t_start_us=start_us,
            t_end_us=end_us,
        )
    batch = EventBatch(
        x=selected["x"].astype(np.int32, copy=True),
        y=selected["y"].astype(np.int32, copy=True),
        t_us=selected["t"].astype(np.int64, copy=True) * 1000,
        polarity=np.where(selected["p"] > 0, 1, -1).astype(np.int8),
        width=CARLA_LOOMING_WIDTH,
        height=CARLA_LOOMING_HEIGHT,
        sequence_id=sequence.sequence_id,
        t_start_us=start_us,
        t_end_us=end_us,
    )
    validate_event_batch(batch)
    return batch


def _coverage_end_ms(sequence: CarlaLoomingSequence) -> int:
    if sequence.metadata is None or sequence.last_event_ms is None:
        raise ValueError(f"Sequence {sequence.sequence_id} has no valid temporal metadata.")
    event_end = int(np.floor(sequence.last_event_ms + sequence.metadata.dt_ms))
    return min(event_end, sequence.metadata.t_end_ms)


def carla_window_references_ms(
    sequence: CarlaLoomingSequence,
    *,
    context_ms: int = 100,
    stride_ms: int = 50,
    minimum_positive_ttc_s: float = 0.1,
    max_windows: int | None = 64,
) -> np.ndarray:
    """Return deterministic causal endpoints without oversampling a sequence."""

    if not sequence.valid or sequence.metadata is None or sequence.first_event_ms is None:
        return np.empty(0, dtype=np.int64)
    if context_ms <= 0 or stride_ms <= 0 or minimum_positive_ttc_s < 0.0:
        raise ValueError("Invalid CARLA window parameters.")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive when provided.")
    first_reference = sequence.first_event_ms + context_ms
    last_reference = _coverage_end_ms(sequence)
    if sequence.metadata.collision:
        last_reference -= int(np.ceil(minimum_positive_ttc_s * 1000.0))
    if last_reference < first_reference:
        return np.empty(0, dtype=np.int64)
    references = np.arange(
        first_reference,
        last_reference + 1,
        stride_ms,
        dtype=np.int64,
    )
    if max_windows is not None and references.size > max_windows:
        selected = np.linspace(0, references.size - 1, max_windows, dtype=np.int64)
        references = references[selected]
    return np.unique(references)


def build_carla_window_sample(
    root: str | Path,
    sequence: CarlaLoomingSequence,
    *,
    reference_ms: int,
    context_ms: int = 100,
    horizons_ms: tuple[int, ...] = (),
    future_window_ms: int = 50,
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> TTCWindowSample:
    """Build a causal TTC sample; negatives carry a censored regression target."""

    if not sequence.valid or sequence.metadata is None:
        raise ValueError(f"Cannot sample invalid CARLA sequence {sequence.sequence_id}.")
    if context_ms <= 0 or future_window_ms <= 0:
        raise ValueError("Context and future window durations must be positive.")
    if any(horizon < 0 for horizon in horizons_ms):
        raise ValueError("Future horizons must be non-negative.")
    events_path, _ = resolve_carla_sequence_paths(root, sequence)
    events = np.load(events_path, mmap_mode="r", allow_pickle=False)
    reference_us = int(reference_ms) * 1000
    context = _carla_event_window_from_array(
        events,
        sequence,
        start_us=reference_us - context_ms * 1000,
        end_us=reference_us,
        source=events_path,
    )
    available_end_ms = _coverage_end_ms(sequence)
    future: dict[int, EventBatch] = {}
    for horizon_ms in horizons_ms:
        future_start_ms = int(reference_ms) + int(horizon_ms)
        future_end_ms = future_start_ms + future_window_ms
        if future_end_ms <= available_end_ms:
            future[horizon_ms] = _carla_event_window_from_array(
                events,
                sequence,
                start_us=future_start_ms * 1000,
                end_us=future_end_ms * 1000,
                source=events_path,
            )
    ttc_seconds = (
        (sequence.metadata.t_end_ms - reference_ms) / 1000.0
        if sequence.metadata.collision
        else None
    )
    collision_within = {
        float(threshold): bool(ttc_seconds is not None and ttc_seconds <= threshold)
        for threshold in risk_thresholds_s
    }
    return TTCWindowSample(
        context_events=context,
        future_events=future,
        ttc_seconds=ttc_seconds,
        collision_within=collision_within,
        object_bbox=None,
        object_mask=None,
        metadata={
            "dataset_id": CARLA_LOOMING_DATASET_ID,
            "collision": sequence.metadata.collision,
            "collision_type": sequence.metadata.collision_type,
            "reference_ms": int(reference_ms),
            "ttc_censored": not sequence.metadata.collision,
            "split_group": sequence.split_group,
        },
    )


class CarlaLoomingWindowDataset:
    """On-demand window dataset compatible with PyTorch ``DataLoader``."""

    def __init__(
        self,
        root: str | Path,
        sequences: list[CarlaLoomingSequence],
        *,
        context_ms: int = 100,
        stride_ms: int = 50,
        minimum_positive_ttc_s: float = 0.1,
        max_windows_per_sequence: int = 64,
        horizons_ms: tuple[int, ...] = (),
        future_window_ms: int = 50,
    ) -> None:
        self.root = Path(root)
        self.context_ms = context_ms
        self.horizons_ms = horizons_ms
        self.future_window_ms = future_window_ms
        self._samples: list[tuple[CarlaLoomingSequence, int]] = []
        for sequence in sequences:
            references = carla_window_references_ms(
                sequence,
                context_ms=context_ms,
                stride_ms=stride_ms,
                minimum_positive_ttc_s=minimum_positive_ttc_s,
                max_windows=max_windows_per_sequence,
            )
            self._samples.extend((sequence, int(reference)) for reference in references)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> TTCWindowSample:
        sequence, reference_ms = self._samples[index]
        return build_carla_window_sample(
            self.root,
            sequence,
            reference_ms=reference_ms,
            context_ms=self.context_ms,
            horizons_ms=self.horizons_ms,
            future_window_ms=self.future_window_ms,
        )


def summarize_carla_sequences(sequences: list[CarlaLoomingSequence]) -> dict[str, Any]:
    """Return counts suitable for manifests and console audit output."""

    valid = [sequence for sequence in sequences if sequence.valid]
    invalid = [sequence for sequence in sequences if not sequence.valid]
    class_counts = Counter(
        sequence.metadata.collision_type
        for sequence in valid
        if sequence.metadata is not None
    )
    return {
        "sequence_count": len(sequences),
        "valid_sequence_count": len(valid),
        "invalid_sequence_count": len(invalid),
        "valid_event_count": int(sum(sequence.num_events for sequence in valid)),
        "class_counts": dict(sorted(class_counts.items())),
        "invalid_sequence_ids": [sequence.sequence_id for sequence in invalid],
        "invalid_issue_counts": dict(
            sorted(Counter(issue for sequence in invalid for issue in sequence.issues).items())
        ),
    }


def write_carla_looming_manifest(
    path: str | Path,
    sequences: list[CarlaLoomingSequence],
    *,
    root_hint: str = "datasets/CARLA_DVS_Looming_Dataset/random_spawn",
    context_ms: int = 100,
) -> dict[str, Any]:
    """Write a signed, path-portable manifest for the local distribution."""

    payload: dict[str, Any] = {
        "artifact_type": "carla_dvs_looming_manifest",
        "format_version": 1,
        "dataset_id": CARLA_LOOMING_DATASET_ID,
        "source_url": CARLA_LOOMING_SOURCE_URL,
        "license": CARLA_LOOMING_LICENSE,
        "official_archive": {
            "filename": "random_spawn.tar.gz",
            "bytes": CARLA_LOOMING_ARCHIVE_BYTES,
            "md5": CARLA_LOOMING_ARCHIVE_MD5,
        },
        "root_hint": root_hint,
        "event_resolution": [CARLA_LOOMING_WIDTH, CARLA_LOOMING_HEIGHT],
        "timestamp_unit": "milliseconds_in_source; microseconds_after_adapter",
        "context_ms_used_for_validity": context_ms,
        "security": {"numpy_allow_pickle": False},
        "summary": summarize_carla_sequences(sequences),
        "sequences": [sequence.to_dict() for sequence in sequences],
    }
    write_structured(path, payload)
    return read_structured(path)


def read_carla_looming_manifest(path: str | Path) -> list[CarlaLoomingSequence]:
    """Load entries from a generated CARLA manifest."""

    payload = read_structured(path)
    if payload.get("dataset_id") != CARLA_LOOMING_DATASET_ID:
        raise ValueError(f"Not a {CARLA_LOOMING_DATASET_ID} manifest: {path}.")
    raw_sequences = payload.get("sequences")
    if not isinstance(raw_sequences, list):
        raise ValueError("CARLA manifest sequences must be a list.")
    return [CarlaLoomingSequence.from_dict(item) for item in raw_sequences]


def _split_statistics(
    sequence_ids: list[str],
    by_id: dict[str, CarlaLoomingSequence],
) -> dict[str, Any]:
    entries = [by_id[sequence_id] for sequence_id in sequence_ids]
    classes = Counter(
        entry.metadata.collision_type
        for entry in entries
        if entry.metadata is not None
    )
    return {
        "sequence_count": len(entries),
        "event_count": int(sum(entry.num_events for entry in entries)),
        "group_count": len({entry.split_group for entry in entries}),
        "class_counts": dict(sorted(classes.items())),
        "positive_count": sum(
            bool(entry.metadata and entry.metadata.collision) for entry in entries
        ),
        "negative_count": sum(
            bool(entry.metadata and not entry.metadata.collision) for entry in entries
        ),
    }


def create_carla_looming_splits(
    sequences: list[CarlaLoomingSequence],
    *,
    seed: int = 42,
    folds: int = 5,
) -> dict[str, Any]:
    """Create deterministic class-balanced splits over complete generation blocks."""

    if folds < 3:
        raise ValueError("At least three folds are required for train/validation/test.")
    valid = [sequence for sequence in sequences if sequence.valid]
    if not valid:
        raise ValueError("No valid CARLA sequences are available for splitting.")
    grouped: dict[str, list[CarlaLoomingSequence]] = defaultdict(list)
    for sequence in valid:
        grouped[sequence.split_group].append(sequence)
    if len(grouped) < folds:
        raise ValueError("The number of CARLA split groups must cover every fold.")
    labels = sorted(
        {
            sequence.metadata.collision_type
            for sequence in valid
            if sequence.metadata is not None
        }
    )
    label_index = {label: index for index, label in enumerate(labels)}
    total_counts = np.zeros(len(labels), dtype=np.float64)
    vectors: dict[str, np.ndarray] = {}
    for group, members in grouped.items():
        vector = np.zeros(len(labels), dtype=np.float64)
        for member in members:
            if member.metadata is not None:
                vector[label_index[member.metadata.collision_type]] += 1.0
        vectors[group] = vector
        total_counts += vector
    rng = np.random.default_rng(seed)
    tie_break = {group: float(rng.random()) for group in grouped}
    ordered = sorted(
        grouped,
        key=lambda group: (
            -float(np.max(vectors[group])),
            -len(grouped[group]),
            tie_break[group],
        ),
    )
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    fold_counts = np.zeros((folds, len(labels)), dtype=np.float64)
    fold_sizes = np.zeros(folds, dtype=np.float64)
    for group in ordered:
        vector = vectors[group]
        size = len(grouped[group])
        scores: list[tuple[float, float, int]] = []
        for fold in range(folds):
            fold_counts[fold] += vector
            fold_sizes[fold] += size
            class_fractions = fold_counts / np.maximum(total_counts[None, :], 1.0)
            class_imbalance = float(np.mean(np.std(class_fractions, axis=0)))
            size_imbalance = float(np.std(fold_sizes / max(len(valid), 1)))
            scores.append((class_imbalance, size_imbalance, fold))
            fold_counts[fold] -= vector
            fold_sizes[fold] -= size
        selected_fold = min(range(folds), key=lambda fold: scores[fold])
        fold_groups[selected_fold].append(group)
        fold_counts[selected_fold] += vector
        fold_sizes[selected_fold] += size
    validation_fold = folds - 2
    test_fold = folds - 1
    role_by_fold = {
        fold: (
            "validation"
            if fold == validation_fold
            else "test"
            if fold == test_fold
            else "train"
        )
        for fold in range(folds)
    }
    assignments: dict[str, list[str]] = {role: [] for role in ("train", "validation", "test")}
    groups_by_role: dict[str, list[str]] = {
        role: [] for role in ("train", "validation", "test")
    }
    for fold, groups in enumerate(fold_groups):
        role = role_by_fold[fold]
        groups_by_role[role].extend(sorted(groups))
        assignments[role].extend(
            member.sequence_id for group in groups for member in grouped[group]
        )
    assignments = {role: sorted(ids) for role, ids in assignments.items()}
    groups_by_role = {role: sorted(groups) for role, groups in groups_by_role.items()}
    sequence_sets = {role: set(ids) for role, ids in assignments.items()}
    group_sets = {role: set(groups) for role, groups in groups_by_role.items()}
    roles = ("train", "validation", "test")
    for left_index, left in enumerate(roles):
        if not sequence_sets[left] or not group_sets[left]:
            raise RuntimeError(f"CARLA split role {left!r} is empty.")
        for right in roles[left_index + 1 :]:
            if sequence_sets[left] & sequence_sets[right]:
                raise RuntimeError(f"CARLA split sequence leakage: {left}/{right}.")
            if group_sets[left] & group_sets[right]:
                raise RuntimeError(f"CARLA split group leakage: {left}/{right}.")
    if set().union(*sequence_sets.values()) != {sequence.sequence_id for sequence in valid}:
        raise RuntimeError("CARLA split assignments do not cover every valid sequence.")
    by_id = {sequence.sequence_id: sequence for sequence in valid}
    return {
        "protocol": "carla_dvs_looming_blocked_v1",
        "seed": seed,
        "fold_count": folds,
        "grouping": "contiguous example_index blocks declared by manifest",
        "scientific_role": (
            "synthetic pretraining and blocked out-of-sample diagnostics; "
            "EvTTC remains the real-domain supervised benchmark"
        ),
        "assignments": assignments,
        "groups": groups_by_role,
        "statistics": {
            role: _split_statistics(ids, by_id) for role, ids in assignments.items()
        },
        "excluded_invalid_sequence_ids": [
            sequence.sequence_id for sequence in sequences if not sequence.valid
        ],
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_carla_looming_splits(
    path: str | Path,
    *,
    manifest_path: str | Path,
    sequences: list[CarlaLoomingSequence],
    seed: int = 42,
    folds: int = 5,
) -> dict[str, Any]:
    """Write signed CARLA blocked splits tied to an exact manifest file."""

    split = create_carla_looming_splits(sequences, seed=seed, folds=folds)
    payload: dict[str, Any] = {
        "artifact_type": "carla_dvs_looming_split",
        "format_version": 1,
        "dataset_id": CARLA_LOOMING_DATASET_ID,
        "manifest": Path(manifest_path).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "benchmark10_opened": False,
        **split,
    }
    write_structured(path, payload)
    return read_structured(path)


__all__ = [
    "CARLA_LOOMING_ARCHIVE_BYTES",
    "CARLA_LOOMING_ARCHIVE_MD5",
    "CARLA_LOOMING_DATASET_ID",
    "CARLA_LOOMING_EVENT_DTYPE",
    "CARLA_LOOMING_HEIGHT",
    "CARLA_LOOMING_LICENSE",
    "CARLA_LOOMING_SOURCE_URL",
    "CARLA_LOOMING_WIDTH",
    "CarlaLoomingMetadata",
    "CarlaLoomingSequence",
    "CarlaLoomingWindowDataset",
    "build_carla_window_sample",
    "carla_window_references_ms",
    "create_carla_looming_splits",
    "load_carla_looming_metadata",
    "read_carla_event_window",
    "read_carla_looming_manifest",
    "resolve_carla_sequence_paths",
    "scan_carla_looming_root",
    "summarize_carla_sequences",
    "write_carla_looming_manifest",
    "write_carla_looming_splits",
]
