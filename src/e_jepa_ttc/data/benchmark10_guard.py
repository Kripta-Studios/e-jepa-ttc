"""Fail-closed guard for the sealed official benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

SEALED_MARKERS = (
    "evttc_official_benchmark_sealed",
    "benchmark10_sealed",
    "benchmark-10-sealed",
)


def is_sealed_benchmark_path(path: str | Path) -> bool:
    """Check path text without opening or enumerating the target."""

    normalized = str(Path(path).resolve(strict=False)).replace("\\", "/").lower()
    return any(marker in normalized for marker in SEALED_MARKERS)


def assert_no_sealed_benchmark_paths(paths: Iterable[str | Path]) -> None:
    """Reject training, cache or selection inputs that point into the sealed root."""

    rejected = [str(path) for path in paths if is_sealed_benchmark_path(path)]
    if rejected:
        raise ValueError(
            "Sealed Benchmark-10 paths are forbidden during training/selection: "
            + ", ".join(rejected)
        )


def assert_benchmark_inference_authorized(
    path: str | Path,
    *,
    final_freeze_manifest: str | Path | None,
    explicit_authorization: bool,
) -> None:
    """Allow a sealed path only for an explicitly frozen final inference."""

    if not is_sealed_benchmark_path(path):
        raise ValueError("The requested final benchmark path is not marked as sealed.")
    if not explicit_authorization:
        raise PermissionError("Explicit final benchmark authorization was not provided.")
    if final_freeze_manifest is None or not Path(final_freeze_manifest).is_file():
        raise FileNotFoundError("A final_freeze_manifest.json is required before benchmark use.")


__all__ = [
    "assert_benchmark_inference_authorized",
    "assert_no_sealed_benchmark_paths",
    "is_sealed_benchmark_path",
]
