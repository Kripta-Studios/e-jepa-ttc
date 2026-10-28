"""Hard storage and closed-dataset guards for the v6 experimental protocol."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

GIB = 1024**3


class DatasetId(StrEnum):
    """Only dataset identities admitted by the closed v6 protocol."""

    CARLA_DVS_LOOMING_1406 = "CARLA_DVS_LOOMING_1406"
    EAP_HF_TRAIN40 = "EAP_HF_TRAIN40"
    EVTTC32_LABELLED = "EVTTC32_LABELLED"
    BENCHMARK10_SEALED = "BENCHMARK10_SEALED"


class StorageBudgetError(RuntimeError):
    """Raised before a derived artifact can exceed a hard storage budget."""


@dataclass(frozen=True)
class StorageBudget:
    """Byte budgets used before materializing a derived cache."""

    maximum_cache_gib: float = 8.0
    minimum_free_gib: float = 100.0

    @property
    def maximum_cache_bytes(self) -> int:
        return int(self.maximum_cache_gib * GIB)

    @property
    def minimum_free_bytes(self) -> int:
        return int(self.minimum_free_gib * GIB)


def _existing_anchor(path: Path) -> Path:
    anchor = path.resolve(strict=False)
    while not anchor.exists() and anchor.parent != anchor:
        anchor = anchor.parent
    if not anchor.exists():
        raise StorageBudgetError(f"No existing filesystem anchor for {path}.")
    return anchor


def directory_size_bytes(path: str | Path) -> int:
    """Return the materialized size of a directory without following symlinks."""

    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(
        entry.stat().st_size
        for entry in root.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )


def estimate_dense_voxel_cache_bytes(
    *,
    samples: int,
    frames_per_sample: int,
    channels: int,
    height: int,
    width: int,
) -> int:
    """Conservative uncompressed upper bound for dense float32 voxel tensors."""

    if min(samples, frames_per_sample, channels, height, width) <= 0:
        raise ValueError("Cache dimensions and sample count must be positive.")
    tensor_bytes = samples * frames_per_sample * channels * height * width * 4
    # Boxes, masks, actions, strings and NPZ metadata are intentionally overestimated.
    return int(tensor_bytes * 1.10)


def assert_storage_budget(
    output_dir: str | Path,
    *,
    budget: StorageBudget,
    planned_write_bytes: int = 0,
) -> None:
    """Reject a cache before writing if either the cache or free-space guard fails."""

    if planned_write_bytes < 0:
        raise ValueError("planned_write_bytes cannot be negative.")
    output = Path(output_dir)
    current = directory_size_bytes(output)
    projected = current + planned_write_bytes
    if projected > budget.maximum_cache_bytes:
        raise StorageBudgetError(
            "Projected derived cache exceeds the hard budget: "
            f"{projected / GIB:.2f} GiB > {budget.maximum_cache_gib:.2f} GiB."
        )
    free = shutil.disk_usage(_existing_anchor(output)).free
    free_after_write = free - planned_write_bytes
    if free_after_write < budget.minimum_free_bytes:
        raise StorageBudgetError(
            "Insufficient free space after the planned write: "
            f"{free_after_write / GIB:.2f} GiB < {budget.minimum_free_gib:.2f} GiB."
        )


def assert_bounded_cache_request(
    max_windows_per_sequence: int | None,
    *,
    allow_full_dataset_cache: bool = False,
) -> int:
    """Forbid accidental full-dataset voxel materialization."""

    if max_windows_per_sequence is None and not allow_full_dataset_cache:
        raise StorageBudgetError(
            "Full-dataset voxel caches are forbidden. Set a bounded "
            "max_windows_per_sequence or use the future on-demand loader."
        )
    if max_windows_per_sequence is None:
        return 0
    if max_windows_per_sequence <= 0:
        raise ValueError("max_windows_per_sequence must be positive.")
    return max_windows_per_sequence


def validate_closed_dataset_roots(
    roots: dict[DatasetId | str, str | Path],
    *,
    for_training: bool,
) -> dict[DatasetId, Path]:
    """Validate identities, unique roots and sealed-benchmark isolation."""

    normalized: dict[DatasetId, Path] = {}
    for raw_id, raw_path in roots.items():
        try:
            dataset_id = DatasetId(str(raw_id))
        except ValueError as error:
            allowed = ", ".join(item.value for item in DatasetId)
            raise ValueError(
                f"Dataset {raw_id!r} is not allowed; expected one of {allowed}."
            ) from error
        path = Path(raw_path).resolve(strict=False)
        if path in normalized.values():
            raise ValueError("Closed dataset roots must be independent.")
        normalized[dataset_id] = path
    if for_training and DatasetId.BENCHMARK10_SEALED in normalized:
        raise ValueError("BENCHMARK10_SEALED cannot be registered by a training pipeline.")
    return normalized


def assert_no_forbidden_cache_features(features: Iterable[str]) -> None:
    """Reject storage-expensive cache products forbidden by PLAN_v6."""

    forbidden = {
        "dino_all_layers_full_dataset",
        "sam_full_resolution_logits",
        "full_dataset_event_voxels",
        "bulk_rgb_tar_extraction",
    }
    requested = set(features)
    invalid = sorted(requested & forbidden)
    if invalid:
        raise StorageBudgetError(f"Forbidden cache features requested: {invalid}.")


__all__ = [
    "DatasetId",
    "StorageBudget",
    "StorageBudgetError",
    "assert_bounded_cache_request",
    "assert_no_forbidden_cache_features",
    "assert_storage_budget",
    "directory_size_bytes",
    "estimate_dense_voxel_cache_bytes",
    "validate_closed_dataset_roots",
]
