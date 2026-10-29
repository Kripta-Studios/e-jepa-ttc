"""Materialize the official Garl-TTC cache one auditable shard at a time.

The canonical cache builder is intentionally left untouched.  This adapter
reuses its row materializer and creates a bounded scratch shard, records its
content and row-identity hashes, optionally hands it to a consumer, and only
then allows the consumer to delete it.  It never treats a partial run as a
complete cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.eap import EAPEventReader  # noqa: E402
from e_jepa_ttc.data.garlttc_calibration import CalibrationMode  # noqa: E402
from e_jepa_ttc.data.garlttc_eap import (  # noqa: E402
    GARLTTC_JOIN_KEYS,
    load_garlttc_train_index,
    validate_garlttc_train_index,
)
from e_jepa_ttc.data.garlttc_lhr_cache import (  # noqa: E402  # noqa: E402
    CalibrationResolver,
    GarlTTCLHRCacheConfig,
    _atomic_json,
    _atomic_torch_save,
    _cache_row_identity,
    _init_cache_worker,
    _materialize_cache_worker,
    _materialize_row,
    _RGBTarReader,
    _sha256_file,
)
from e_jepa_ttc.data.garlttc_sampling import select_balanced_cache_rows  # noqa: E402
from e_jepa_ttc.utils.io import read_structured  # noqa: E402

Consumer = Callable[[Path, dict[str, Any]], None]
SAFE_DEFAULT_SHARD_SIZE = 256
STORAGE_RESERVE_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class RotatingShard:
    """A deterministic group of rows that can be consumed independently."""

    role: str
    shard_index: int
    rows: tuple[dict[str, object], ...]

    @property
    def identities(self) -> tuple[tuple[str, ...], ...]:
        return tuple(_cache_row_identity(row) for row in self.rows)

    @property
    def identity_sha256(self) -> str:
        return identity_hash(self.identities)


def identity_hash(identities: Sequence[tuple[str, ...]]) -> str:
    """Hash ordered five-key identities without serializing media or labels."""

    payload = "\n".join(
        json.dumps(list(identity), ensure_ascii=False, separators=(",", ":"))
        for identity in identities
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _planned_storage_bytes(
    plan: Mapping[str, Any],
    *,
    max_shards: int | None,
    retained: bool,
) -> int:
    """Return the planned retained total or rotating peak before media reads."""

    raw_shards = plan.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("Rotating plan has no shard storage estimates")
    selected = raw_shards if max_shards is None else raw_shards[:max_shards]
    estimates = [int(shard.get("estimated_bytes", -1)) for shard in selected]
    if not estimates or min(estimates) <= 0:
        raise ValueError("Every selected rotating shard needs a positive byte estimate")
    return sum(estimates) if retained else max(estimates)


def _storage_anchor(path: Path) -> Path:
    """Find an existing ancestor suitable for a disk-free-space query."""

    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"No existing storage ancestor for {path}")
    return candidate


def _validate_storage_capacity(
    output_dir: Path,
    plan: Mapping[str, Any],
    *,
    max_shards: int | None,
    retained: bool,
) -> int:
    """Reject a run that would consume the configured drive reserve."""

    required = _planned_storage_bytes(plan, max_shards=max_shards, retained=retained)
    free = shutil.disk_usage(_storage_anchor(output_dir)).free
    usable = max(free - STORAGE_RESERVE_BYTES, 0)
    if required > usable:
        mode = "retained" if retained else "rotating-peak"
        raise OSError(
            f"Planned {mode} cache needs {required / 1024**3:.2f} GiB but only "
            f"{usable / 1024**3:.2f} GiB is available after the 10 GiB reserve."
        )
    return required


def partition_selected_rows(
    selected: pd.DataFrame,
    sequence_roles: Mapping[str, str],
    *,
    shard_size: int,
) -> list[RotatingShard]:
    """Partition already-selected rows without changing their deterministic order."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    expected = set(GARLTTC_JOIN_KEYS)
    missing = sorted(expected - set(selected.columns))
    if missing:
        raise ValueError(f"Selected rows miss join keys: {missing}")
    identities = [_cache_row_identity(row) for row in selected.to_dict(orient="records")]
    if len(set(identities)) != len(identities):
        raise ValueError("Selected rows contain duplicate five-key identities")

    shards: list[RotatingShard] = []
    for role in ("train", "validation"):
        role_mask = selected["sequence_id"].astype(str).map(sequence_roles.get) == role
        role_rows = selected.loc[role_mask].to_dict(orient="records")
        for shard_index, start in enumerate(range(0, len(role_rows), shard_size)):
            rows = tuple(
                cast(dict[str, object], row) for row in role_rows[start : start + shard_size]
            )
            shards.append(RotatingShard(role, shard_index, rows))
    if sum(len(shard.rows) for shard in shards) != len(selected):
        raise RuntimeError("Rotating shards do not cover the selected rows")
    return shards


def _validate_plan(
    plan: Mapping[str, Any],
    shards: Sequence[RotatingShard],
    *,
    data_sha256: str,
    annotations_sha256: str,
) -> None:
    """Fail before media reads if the persisted plan differs from the source index."""

    if plan.get("artifact_type") != "garlttc_rotating_cache_plan_v1":
        raise ValueError("Unexpected rotating-cache plan artifact type")
    if plan.get("status") != "pass" or plan.get("materialization_started") is not False:
        raise ValueError("Rotating plan is not a clean, pre-materialization plan")
    if plan.get("garlttc_data_sha256") != data_sha256:
        raise ValueError("Garl data parquet hash differs from the frozen plan")
    if plan.get("garlttc_annotations_sha256") != annotations_sha256:
        raise ValueError("Garl annotations parquet hash differs from the frozen plan")
    expected = plan.get("shards")
    if not isinstance(expected, list) or len(expected) != len(shards):
        raise ValueError("Rotating plan shard count does not match selected rows")
    for actual, raw in zip(shards, expected, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("Rotating plan contains a non-object shard")
        if (
            raw.get("role") != actual.role
            or int(raw.get("shard_index", -1)) != actual.shard_index
            or int(raw.get("row_count", -1)) != len(actual.rows)
            or raw.get("row_identity_sha256") != actual.identity_sha256
        ):
            raise ValueError(
                f"Rotating plan identity mismatch for {actual.role}/{actual.shard_index}"
            )


def _source_rows(
    *,
    garlttc_root: Path,
    split_path: Path,
    seed: int,
    shard_size: int,
    expected_rows: int,
) -> tuple[list[RotatingShard], dict[str, Any], Any]:
    split = read_structured(split_path)
    assignments = split.get("assignments")
    if not isinstance(assignments, Mapping):
        raise ValueError("Split artifact has no assignments mapping")
    sequence_roles = {
        str(sequence): role
        for role in ("train", "validation")
        for sequence in assignments.get(role, [])
    }
    index = load_garlttc_train_index(garlttc_root, sorted(sequence_roles))
    validate_garlttc_train_index(
        index,
        expected_rows=expected_rows,
        allow_version_change=False,
    )
    rows = index.merged.sort_values(
        ["sequence_id", "timestamp_us", "track_id", "sample_token"],
        kind="mergesort",
    )
    selected, selection_report = select_balanced_cache_rows(
        rows,
        sequence_roles,
        seed=seed,
        max_samples_per_split=None,
    )
    return (
        partition_selected_rows(selected, sequence_roles, shard_size=shard_size),
        selection_report,
        index,
    )


def _materialize_records(
    rows: Sequence[dict[str, object]],
    *,
    eap_root: Path,
    config: GarlTTCLHRCacheConfig,
    first_track_lookup: Mapping[tuple[object, object], object],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tasks: list[tuple[int, dict[str, object], int]] = []
    for order, row in enumerate(rows):
        fallback = row["timestamp_us"]
        first_timestamp = first_track_lookup.get((row["sequence_id"], row["track_id"]), fallback)
        tasks.append((order, row, int(cast(Any, first_timestamp))))
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cache_worker,
            initargs=(eap_root.as_posix(), config),
        ) as executor:
            results = list(executor.map(_materialize_cache_worker, tasks, chunksize=1))
    else:
        event_readers: dict[Path, EAPEventReader] = {}
        calibration = CalibrationResolver(
            cast(CalibrationMode, config.calibration_mode),
            eap_root=eap_root,
        )
        rgb_reader = _RGBTarReader(eap_root) if config.include_rgb else None
        results = []
        for order, row, first_timestamp in tasks:
            try:
                sample = _materialize_row(
                    row,
                    eap_root=eap_root,
                    config=config,
                    event_readers=event_readers,
                    rgb_reader=rgb_reader,
                    calibration=calibration,
                    first_track_timestamp_us=first_timestamp,
                )
            except Exception as exc:  # explicit accounting; never silently substitute labels
                results.append((order, None, f"{type(exc).__name__}:{str(exc)[:120]}"))
            else:
                results.append((order, sample, None))
    results.sort(key=lambda item: item[0])
    records = [sample for _, sample, error in results if sample is not None]
    errors = Counter(error for _, sample, error in results if sample is None and error)
    return records, dict(sorted(errors.items()))


def execute_rotating_cache(
    *,
    eap_root: Path,
    garlttc_root: Path,
    split_path: Path,
    plan_path: Path,
    output_dir: Path,
    config: GarlTTCLHRCacheConfig,
    selection_seed: int,
    shard_size: int,
    expected_rows: int,
    workers: int,
    resume: bool = False,
    delete_after_consume: bool = False,
    consumer: Consumer | None = None,
    max_shards: int | None = None,
    allow_retained_full_cache: bool = False,
) -> dict[str, Any]:
    """Execute a lossless rotating materialization with optional consumption."""

    if delete_after_consume and consumer is None:
        raise ValueError("delete_after_consume requires a consumer callback")
    if max_shards is not None and max_shards <= 0:
        raise ValueError("max_shards must be positive when provided")
    retained = consumer is None or not delete_after_consume
    if max_shards is None and retained and not allow_retained_full_cache:
        raise ValueError(
            "Unbounded retained-cache execution is unsafe. Provide a consumer with "
            "delete_after_consume=True, bound the run with max_shards, or explicitly "
            "set allow_retained_full_cache=True after verifying capacity."
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = cast(dict[str, Any], read_structured(plan_path))
    shards, selection_report, index = _source_rows(
        garlttc_root=garlttc_root,
        split_path=split_path,
        seed=selection_seed,
        shard_size=shard_size,
        expected_rows=expected_rows,
    )
    _validate_plan(
        plan,
        shards,
        data_sha256=index.data_sha256,
        annotations_sha256=index.annotations_sha256,
    )
    planned_storage_bytes = _validate_storage_capacity(
        output_dir,
        plan,
        max_shards=max_shards,
        retained=retained,
    )
    planned_shard_count = len(shards)
    planned_selected_row_count = sum(len(shard.rows) for shard in shards)
    if max_shards is not None:
        shards = shards[:max_shards]
    lookup = (
        index.merged.groupby(["sequence_id", "track_id"], dropna=False)["timestamp_us"]
        .min()
        .to_dict()
    )
    run_path = output_dir / "run_manifest.json"
    state_path = output_dir / "state.json"
    existing_state: dict[str, Any] = {}
    if resume and state_path.is_file():
        existing_state = cast(dict[str, Any], read_structured(state_path))
    completed = {
        str(item["key"]): item
        for item in existing_state.get("shards", [])
        if isinstance(item, Mapping) and isinstance(item.get("key"), str)
    }
    started = datetime.now(UTC).isoformat()
    _atomic_json(
        run_path,
        {
            "artifact_type": "garlttc_rotating_cache_execution_v1",
            "status": "running",
            "started_at": started,
            "git_commit": plan.get("git_commit"),
            "plan_sha256": _sha256_file(plan_path),
            "split_sha256": _sha256_file(split_path),
            "garlttc_data_sha256": index.data_sha256,
            "garlttc_annotations_sha256": index.annotations_sha256,
            "selection_report": selection_report,
            "split_counts": {
                role: sum(len(shard.rows) for shard in shards if shard.role == role)
                for role in ("train", "validation")
            },
            "shard_count": len(shards),
            "planned_shard_count": planned_shard_count,
            "bounded_smoke": max_shards is not None,
            "workers": workers,
            "config": asdict(config),
            "delete_after_consume": delete_after_consume,
            "planned_storage_bytes": planned_storage_bytes,
            "storage_mode": "retained" if retained else "rotating_peak",
            "host": platform.node(),
        },
    )
    _atomic_json(
        state_path,
        {
            "artifact_type": "garlttc_rotating_cache_state_v1",
            "status": "running",
            "shards": list(completed.values()),
        },
    )
    try:
        for shard in shards:
            key = f"{shard.role}/{shard.shard_index:05d}"
            previous = completed.get(key)
            shard_path = output_dir / shard.role / f"shard-{shard.shard_index:05d}.pt.gz"
            metadata_path = output_dir / shard.role / f"shard-{shard.shard_index:05d}.meta.json"
            if previous is not None and previous.get("status") == "deleted":
                if previous.get("row_identity_sha256") != shard.identity_sha256:
                    raise ValueError(f"Deleted shard identity changed: {key}")
                continue
            if previous is not None and shard_path.is_file() and metadata_path.is_file():
                if previous.get("sha256") != _sha256_file(shard_path):
                    raise ValueError(f"Existing shard hash changed: {key}")
                metadata = dict(previous)
                if consumer is not None:
                    consumer(shard_path, metadata)
                if delete_after_consume:
                    if not shard_path.is_file() or _sha256_file(shard_path) != metadata["sha256"]:
                        raise RuntimeError(f"Shard changed before deletion: {key}")
                    shard_path.unlink()
                    metadata["status"] = "deleted"
                    metadata["deleted_after_consume"] = True
                    completed[key] = metadata
                    _atomic_json(
                        state_path,
                        {
                            "artifact_type": "garlttc_rotating_cache_state_v1",
                            "status": "running",
                            "shards": [completed[item] for item in sorted(completed)],
                        },
                    )
                continue
            if previous is not None:
                raise FileNotFoundError(
                    f"Recorded shard is missing or incomplete and cannot be resumed: {key}"
                )

            records, discard_reasons = _materialize_records(
                shard.rows,
                eap_root=eap_root,
                config=config,
                first_track_lookup=lookup,
                workers=workers,
            )
            if discard_reasons or len(records) != len(shard.rows):
                raise RuntimeError(
                    f"Shard {key} discarded rows: requested={len(shard.rows)} "
                    f"materialized={len(records)} reasons={discard_reasons}"
                )
            _atomic_torch_save(
                records,
                shard_path,
                compression="gzip",
                compression_level=config.compression_level,
            )
            record_count = len(records)
            del records
            digest = _sha256_file(shard_path)
            metadata = {
                "artifact_type": "garlttc_rotating_shard_v1",
                "key": key,
                "role": shard.role,
                "shard_index": shard.shard_index,
                "count": record_count,
                "row_identity_sha256": shard.identity_sha256,
                "sha256": digest,
                "path": shard_path.relative_to(output_dir).as_posix(),
                "discard_count": 0,
                "status": "materialized",
            }
            _atomic_json(metadata_path, metadata)
            if consumer is not None:
                consumer(shard_path, metadata)
            if delete_after_consume:
                if not shard_path.is_file() or _sha256_file(shard_path) != digest:
                    raise RuntimeError(f"Shard changed before deletion: {key}")
                shard_path.unlink()
                metadata["status"] = "deleted"
                metadata["deleted_after_consume"] = True
            completed[key] = metadata
            _atomic_json(
                state_path,
                {
                    "artifact_type": "garlttc_rotating_cache_state_v1",
                    "status": "running",
                    "shards": [completed[item] for item in sorted(completed)],
                },
            )
    except BaseException as exc:
        failure = {
            "artifact_type": "garlttc_rotating_cache_failure_v1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "state_path": state_path.as_posix(),
            "completed_shards": len(completed),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(output_dir / "FAILURE.json", failure)
        raise

    completed_values = [completed[key] for key in sorted(completed)]
    result = {
        "artifact_type": "garlttc_rotating_cache_execution_v1",
        "status": "completed" if len(completed_values) == planned_shard_count else "partial",
        "completed_at": datetime.now(UTC).isoformat(),
        "plan_sha256": _sha256_file(plan_path),
        "split_sha256": _sha256_file(split_path),
        "garlttc_data_sha256": index.data_sha256,
        "garlttc_annotations_sha256": index.annotations_sha256,
        "selected_row_count": sum(len(shard.rows) for shard in shards),
        "planned_selected_row_count": planned_selected_row_count,
        "split_counts": {
            role: sum(len(shard.rows) for shard in shards if shard.role == role)
            for role in ("train", "validation")
        },
        "shard_count": len(shards),
        "planned_shard_count": planned_shard_count,
        "coverage_complete": len(completed_values) == planned_shard_count,
        "bounded_smoke": max_shards is not None,
        "shards": completed_values,
        "all_shard_hashes_recorded": all(item.get("sha256") for item in completed_values),
        "all_deleted_after_consume": all(
            item.get("status") == "deleted" for item in completed_values
        ),
        "selection_report": selection_report,
    }
    _atomic_json(run_path, result)
    _atomic_json(
        state_path,
        {
            "artifact_type": "garlttc_rotating_cache_state_v1",
            "status": result["status"],
            "shards": completed_values,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=SAFE_DEFAULT_SHARD_SIZE)
    parser.add_argument("--expected-rows", type=int, default=88_744)
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Bound a smoke run to the first N planned shards; it is not a full-cache result.",
    )
    parser.add_argument(
        "--allow-large-shards",
        action="store_true",
        help="Acknowledge the RAM risk of shards larger than the verified 256-sample size.",
    )
    parser.add_argument(
        "--allow-retained-full-cache",
        action="store_true",
        help="Acknowledge retaining every shard; capacity preflight still applies.",
    )
    args = parser.parse_args()
    if args.shard_size > SAFE_DEFAULT_SHARD_SIZE and not args.allow_large_shards:
        parser.error(f"--shard-size above {SAFE_DEFAULT_SHARD_SIZE} requires --allow-large-shards")
    config = GarlTTCLHRCacheConfig(
        include_rgb=args.include_rgb,
        shard_size=args.shard_size,
        workers=args.workers,
        compression="gzip",
        compression_level=args.compression_level,
    )
    result = execute_rotating_cache(
        eap_root=args.eap_root.resolve(),
        garlttc_root=args.garlttc_root.resolve(),
        split_path=args.split.resolve(),
        plan_path=args.plan.resolve(),
        output_dir=args.output_dir.resolve(),
        config=config,
        selection_seed=args.selection_seed,
        shard_size=args.shard_size,
        expected_rows=args.expected_rows,
        workers=args.workers,
        resume=args.resume,
        max_shards=args.max_shards,
        allow_retained_full_cache=args.allow_retained_full_cache,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
