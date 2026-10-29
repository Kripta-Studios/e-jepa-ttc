#!/usr/bin/env python3
"""Sanitize three mixed v4.30 OOF files into seed-isolated v4.31 evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from e_jepa_ttc.data.object_event_v4_31 import AtomicDirectory, sha256_file, strict_json  # noqa: E402, I001

ALLOW = {
    "row_index": ("oof_row_index", "row_index"),
    "log_eta": ("log_eta",),
    "endpoint_swap_log_eta": ("endpoint_swap_log_eta",),
    "unknown": ("unknown",),
    "row_identity": ("row_identity", "sample_identity"),
}
SEEDS = (7, 13, 23)


def _read(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        names = {
            key: next((item for item in choices if item in keys), None)
            for key, choices in ALLOW.items()
        }
        if any(item is None for item in names.values()):
            raise ValueError(f"missing stage2 allowlisted key in {path}: {names}")
        value = {key: np.asarray(archive[str(name)]) for key, name in names.items()}
    if any(item.ndim != 1 or item.shape != (2048,) for item in value.values()):
        raise ValueError("every stage2 seed must have exactly 2048 rows")
    if not np.issubdtype(value["row_index"].dtype, np.integer) or not np.array_equal(
        value["row_index"].astype(np.int64), np.arange(2048)
    ):
        raise ValueError("oof_row_index must be exactly 0..2047")
    for key in ("log_eta", "endpoint_swap_log_eta"):
        if not np.issubdtype(value[key].dtype, np.floating) or not np.isfinite(value[key]).all():
            raise ValueError(f"stage2 {key} must be finite floating point")
    unknown = value["unknown"]
    if not (
        np.issubdtype(unknown.dtype, np.bool_)
        or (
            np.issubdtype(unknown.dtype, np.number)
            and np.isfinite(unknown).all()
            and np.isin(unknown, (0, 1)).all()
        )
    ):
        raise ValueError("stage2 unknown must be boolean or 0/1")
    identities = value["row_identity"]
    if identities.dtype.kind not in {"U", "S"} or not all(str(item) for item in identities):
        raise ValueError("stage2 requires nonempty stable row_identity metadata")
    if len(set(str(item) for item in identities)) != 2048:
        raise ValueError("stage2 row_identity values must be unique")
    canonical = {
        "row_index": value["row_index"].astype(np.int64, copy=False),
        "log_eta": value["log_eta"].astype(np.float32, copy=False),
        "endpoint_swap_log_eta": value["endpoint_swap_log_eta"].astype(np.float32, copy=False),
        "unknown": value["unknown"].astype(bool, copy=False),
        "row_identity": identities.astype("U", copy=False),
    }
    return canonical, {"sha256": sha256_file(path), "keys": sorted(keys), "mapping": names}


def sanitize(sources: dict[int, Path], output: Path, *, force: bool = False) -> dict[str, object]:
    if tuple(sorted(sources)) != SEEDS:
        raise ValueError("--source must specify exactly 7, 13, and 23")
    canonical_paths = [sources[seed].resolve() for seed in SEEDS]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError("stage2 sources must be three distinct canonical paths")
    output_resolved = output.resolve()
    if any(output_resolved == path or output_resolved in path.parents for path in canonical_paths):
        raise ValueError("stage2 output must not overlap a mixed source path")
    loaded = {seed: _read(path) for seed, path in sources.items()}
    if len({str(metadata["sha256"]) for _, metadata in loaded.values()}) != len(SEEDS):
        raise ValueError("stage2 sources must have three distinct content SHA256 values")
    reference = loaded[7][0]["row_index"]
    if any(not np.array_equal(reference, loaded[seed][0]["row_index"]) for seed in SEEDS[1:]):
        raise ValueError("stage2 seed row-index arrays are not identical")
    reference_identity = loaded[7][0]["row_identity"]
    if any(
        not np.array_equal(reference_identity, loaded[seed][0]["row_identity"])
        for seed in SEEDS[1:]
    ):
        raise ValueError("stage2 stable row identities must agree across seeds")
    with AtomicDirectory(output, force=force) as stage:
        for seed, (values, _) in loaded.items():
            np.savez(stage / f"seed_{seed}.npz", **values)  # type: ignore[reportArgumentType]
        manifest: dict[str, object] = {
            "artifact_type": "object_event_v4_31_stage2_v1",
            "source_contains_forbidden_fields": any(
                bool(
                    set(cast(list[str], metadata["keys"]))
                    - {alias for names in ALLOW.values() for alias in names}
                )
                for _, metadata in loaded.values()
            ),
            "count_per_seed": 2048,
            "seeds": {
                str(seed): {**metadata, "canonical_path": str(sources[seed].resolve())}
                for seed, (_, metadata) in loaded.items()
            },
            "opened_paths": [str(sources[seed].resolve()) for seed in SEEDS],
            "outputs": {
                str(seed): {
                    "path": f"seed_{seed}.npz",
                    "sha256": sha256_file(stage / f"seed_{seed}.npz"),
                }
                for seed in SEEDS
            },
        }
        (stage / "manifest.json").write_text(strict_json(manifest), encoding="utf-8")
    return manifest


def _parse_source(value: str) -> tuple[int, Path]:
    try:
        seed, path = value.split("=", 1)
        return int(seed), Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source must be SEED=PATH") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        print(strict_json(sanitize(dict(args.source), args.output_dir, force=args.force)))
    except Exception as exc:
        print(
            strict_json(
                {
                    "artifact_type": "object_event_v4_31_stage2_v1",
                    "status": "invalid_incomplete",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
