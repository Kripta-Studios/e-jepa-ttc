"""Physical event-H5 inventory and admission for independent raw-history V4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, cast

import h5py
import hdf5plugin  # noqa: F401  # Registers the public eAP compression filters.
import numpy as np
import pandas as pd

from .lineage import _load_rotation_role_unverified
from .population import canonical_sha256, file_sha256, identity_sha256
from .raw_history import ROLE_KEYS, verify_raw_history_index
from .raw_history import SCHEMA as RAW_SCHEMA

PLAN_SCHEMA = "independent_phase_event_h5_inventory_v2"
PREFLIGHT_SCHEMA = "independent_phase_event_h5_metadata_preflight_v4"
ADMISSION_SCHEMA = "independent_phase_event_h5_admission_v1"
PUBLIC_MANIFEST_PATH = Path("data/manifests/eap_detection_public_files_v1.json")
REQUIRED_DATASETS = ("events/x", "events/y", "events/t", "events/p", "ms_to_idx")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _self_hash(value: dict[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("receipt_sha256", None)
    if claimed != canonical_sha256(payload):
        raise ValueError("H5 receipt canonical identity changed")


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _history_counts(frame: pd.DataFrame) -> tuple[int, int]:
    """Return valid and invalid row counts without ambiguous pandas scalars."""
    values = cast(pd.Series, frame["history_valid"]).to_numpy(dtype=np.bool_)
    valid = int(np.count_nonzero(values))
    return valid, int(values.size - valid)


def _safe_event_path(root: Path, sequence: str, relative: object) -> Path:
    posix = PurePosixPath(str(relative).replace("\\", "/"))
    expected = PurePosixPath("data") / "train" / sequence / "events.h5"
    if posix != expected or posix.is_absolute() or ".." in posix.parts:
        raise ValueError("Event H5 path differs from canonical per-sequence path")
    path = (root / Path(*posix.parts)).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Event H5 escapes the pinned eAP root")
    return path


def _required_files(raw_receipt_path: Path, eap_root: Path) -> list[dict[str, Any]]:
    raw_path = raw_receipt_path.resolve(strict=True)
    verified = verify_raw_history_index(raw_path)
    if verified != {"status": "PASS", "queries": 53_047, "groups": 24, "roles": 12}:
        raise ValueError("Raw-history V4 verification result differs")
    receipt = _json(raw_path)
    if receipt.get("schema") != RAW_SCHEMA or receipt.get("training_admitted") is not False:
        raise ValueError("H5 admission requires the metadata-only raw-history V4 contract")
    index_path = (raw_path.parent / str(receipt["index_path"])).resolve(strict=True)
    frame = pd.read_parquet(
        index_path,
        columns=[
            "sample_token",
            "sequence_id",
            "events_path",
            "observation_source_start_us",
            "event_history_anchor_us",
            "observation_windows_us",
        ],
    )
    role_groups: dict[str, set[str]] = {}
    for key in ROLE_KEYS:
        ref = receipt["roles"][key]
        role = pd.read_parquet(raw_path.parent / str(ref["path"]), columns=["sequence_id"])
        role_groups[key] = set(role["sequence_id"].astype(str))
    files: list[dict[str, Any]] = []
    for sequence, rows in frame.groupby("sequence_id", sort=True):
        paths = set(rows["events_path"].astype(str))
        if len(paths) != 1:
            raise ValueError("One TRAIN24 sequence maps to multiple event streams")
        relative = next(iter(paths))
        path = _safe_event_path(eap_root, str(sequence), relative)
        windows = [
            [int(value) for value in pair]
            for nested in rows["observation_windows_us"]
            for pair in nested
        ]
        if any(right - left != 20_000 for left, right in windows):
            raise ValueError("Raw-history observation interval duration differs")
        files.append(
            {
                "sequence_id": str(sequence),
                "relative_path": PurePosixPath(relative).as_posix(),
                "absolute_path": str(path),
                "required_start_us": int(
                    np.asarray(rows["observation_source_start_us"], dtype=np.int64).min()
                ),
                "required_end_us": int(
                    np.asarray(rows["event_history_anchor_us"], dtype=np.int64).max()
                ),
                "queries": len(rows),
                "observation_intervals": len(windows),
                "sample_token_set_sha256": identity_sha256(
                    rows["sample_token"].astype(str).tolist(), ordered=False
                ),
                "roles": sorted(key for key, groups in role_groups.items() if sequence in groups),
            }
        )
    if len(files) != 24 or any(len(item["roles"]) != 5 for item in files):
        raise ValueError("H5 inventory does not preserve exact six-rotation producer/refiner roles")
    return files


def build_h5_inventory(
    raw_receipt_path: Path,
    eap_root: Path,
    output_dir: Path,
    public_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Write a stat-only execution plan; this artifact never admits training."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    root = eap_root.resolve(strict=True)
    public_path = (public_manifest_path or PUBLIC_MANIFEST_PATH).resolve(strict=True)
    public = _json(public_path)
    public_files = public.get("files")
    if not isinstance(public_files, dict):
        raise ValueError("Public eAP manifest has no file inventory")
    files = _required_files(raw_receipt_path, root)
    for item in files:
        stat = Path(item["absolute_path"]).stat()
        release = public_files.get(item["relative_path"])
        if not isinstance(release, dict) or not isinstance(release.get("sha256"), str):
            raise ValueError("Required event H5 is absent from public eAP manifest")
        if stat.st_size != release.get("size_bytes"):
            raise ValueError("Local event H5 size differs from public eAP release")
        item.update(
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            expected_public_sha256=release["sha256"],
            expected_public_size_bytes=release["size_bytes"],
        )
    total = sum(int(item["size_bytes"]) for item in files)
    receipt: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "INVENTORIED_NOT_HASHED_NOT_ADMITTED",
        "raw_history_receipt_path": str(raw_receipt_path.resolve(strict=True)),
        "raw_history_receipt_sha256": file_sha256(raw_receipt_path),
        "eap_root": str(root),
        "public_manifest_path": str(public_path),
        "public_manifest_sha256": file_sha256(public_path),
        "public_manifest_api_sha256": public.get("api_sha256"),
        "files": files,
        "file_count": 24,
        "total_bytes": total,
        "hash_read_bytes_pending": total,
        "eta_seconds_at_200_mib_s": round(total / (200 * 1024**2), 1),
        "eta_seconds_at_500_mib_s": round(total / (500 * 1024**2), 1),
        "h5_files_opened": False,
        "full_hashes_computed": False,
        "stream_bounds_checked": False,
        "training_admitted": False,
        "next_step": "RUN_ADMISSION_SEQUENTIALLY_ONLY_WHEN_E_DRIVE_IS_IDLE",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    output_dir.mkdir(parents=True)
    (output_dir / "INVENTORY.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _inspect_h5(path: Path, required_start: int, required_end: int) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"Event H5 schema missing datasets: {missing}")
        datasets = {name: cast(h5py.Dataset, handle[name]) for name in REQUIRED_DATASETS}
        event_shapes = [datasets[f"events/{key}"].shape for key in ("x", "y", "t", "p")]
        if any(len(shape) != 1 for shape in event_shapes) or len(set(event_shapes)) != 1:
            raise ValueError("Event H5 event arrays are not equal one-dimensional vectors")
        events = int(event_shapes[0][0])
        if events <= 0 or not np.issubdtype(datasets["events/t"].dtype, np.integer):
            raise ValueError("Event H5 timestamp vector is empty or non-integer")
        first, last = int(datasets["events/t"][0]), int(datasets["events/t"][-1])
        ms_index = np.asarray(datasets["ms_to_idx"][:], dtype=np.int64)
        start_ms = required_start // 1000
        end_ms = (required_end + 999) // 1000
        if (
            ms_index.ndim != 1
            or len(ms_index) < 2
            or np.any(np.diff(ms_index) < 0)
            or int(ms_index[0]) < 0
            or int(ms_index[-1]) > events
            or required_start < 0
            or end_ms > len(ms_index) - 1
            or last // 1000 > len(ms_index) + 1
        ):
            raise ValueError("Event H5 timestamp bounds/index do not cover causal intervals")
        return {
            "datasets": {
                name: {"shape": list(datasets[name].shape), "dtype": str(datasets[name].dtype)}
                for name in REQUIRED_DATASETS
            },
            "events": events,
            "first_timestamp_us": first,
            "last_timestamp_us": last,
            "stream_half_open_end_us": last + 1,
            "ms_index_origin_us": 0,
            "required_start_ms": start_ms,
            "required_end_ms": end_ms,
            "required_intervals_covered": True,
            "ms_to_idx_nondecreasing": True,
        }


def _inspect_h5_metadata(path: Path, required_start: int, required_end: int) -> dict[str, Any]:
    """Read only headers and the two boundary timestamps, never full event arrays."""
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"Event H5 schema missing datasets: {missing}")
        datasets = {name: cast(h5py.Dataset, handle[name]) for name in REQUIRED_DATASETS}
        shapes = [datasets[f"events/{key}"].shape for key in ("x", "y", "t", "p")]
        schema_valid = (
            all(len(shape) == 1 for shape in shapes)
            and len(set(shapes)) == 1
            and int(shapes[0][0]) > 0
            and np.issubdtype(datasets["events/t"].dtype, np.integer)
            and len(datasets["ms_to_idx"].shape) == 1
            and np.issubdtype(datasets["ms_to_idx"].dtype, np.integer)
        )
        if not schema_valid:
            raise ValueError("Event H5 metadata schema differs")
        first = int(datasets["events/t"][0])
        last = int(datasets["events/t"][-1])
        declared_start = handle.attrs.get("t_start_us", handle["events"].attrs.get("t_start_us"))
        declared_end = handle.attrs.get("t_end_us", handle["events"].attrs.get("t_end_us"))
        admission_start = int(declared_start) if declared_start is not None else first
        admission_end = int(declared_end) if declared_end is not None else last + 1
        start_ms = required_start // 1000
        end_ms = (required_end + 999) // 1000
        maximum_ms = int(datasets["ms_to_idx"].shape[0] - 1)
        timebase_consistent = first >= 0 and last // 1000 <= maximum_ms + 2
        index_covered = required_start >= 0 and 0 <= start_ms <= end_ms <= maximum_ms
        start_index = int(datasets["ms_to_idx"][start_ms]) if index_covered else None
        end_index = int(datasets["ms_to_idx"][end_ms]) if index_covered else None
        indexed_range_valid = (
            start_index is not None
            and end_index is not None
            and 0 <= start_index <= end_index <= int(shapes[0][0])
        )
        return {
            "datasets": {
                name: {"shape": list(datasets[name].shape), "dtype": str(datasets[name].dtype)}
                for name in REQUIRED_DATASETS
            },
            "first_timestamp_us": first,
            "last_timestamp_us": last,
            "stream_half_open_end_us": last + 1,
            "declared_recording_start_us": (
                int(declared_start) if declared_start is not None else None
            ),
            "declared_recording_end_us": int(declared_end) if declared_end is not None else None,
            "admission_start_us": admission_start,
            "admission_end_us": admission_end,
            "admission_start_authority": (
                "H5_T_START_US_ATTRIBUTE"
                if declared_start is not None
                else "FIRST_EVENT_CONSERVATIVE_NO_RECORDING_START_AUTHORITY"
            ),
            "event_first_minus_required_start_us": first - required_start,
            "event_last_margin_us": last + 1 - required_end,
            "ms_index_origin_us": 0,
            "ms_index_maximum_ms": maximum_ms,
            "required_start_ms": start_ms,
            "required_end_ms": end_ms,
            "required_start_event_index": start_index,
            "required_end_event_index": end_index,
            "timebase_consistent": timebase_consistent,
            "required_bounds_covered": (
                index_covered and timebase_consistent and indexed_range_valid
            ),
        }


def build_h5_metadata_preflight(inventory_path: Path, output_dir: Path) -> dict[str, Any]:
    """Cheaply inspect all H5 headers/bounds and fail closed before full hashing."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    inventory = verify_h5_inventory(inventory_path)
    raw_receipt = Path(inventory["raw_history_receipt_path"])
    raw = _json(raw_receipt)
    frame = pd.read_parquet(
        raw_receipt.parent / str(raw["index_path"]),
        columns=[
            "fit_ordinal",
            "sample_token",
            "sequence_id",
            "observation_source_start_us",
            "event_history_anchor_us",
        ],
    )
    frame["history_valid"] = True
    frame["invalid_reason"] = ""
    files: list[dict[str, Any]] = []
    for item in inventory["files"]:
        metadata = _inspect_h5_metadata(
            Path(item["absolute_path"]), item["required_start_us"], item["required_end_us"]
        )
        rows = frame.loc[frame["sequence_id"].astype(str) == item["sequence_id"]]
        start_invalid = rows["observation_source_start_us"] < int(metadata["admission_start_us"])
        end_invalid = rows["event_history_anchor_us"] > int(metadata["admission_end_us"])
        start_uncovered, end_uncovered = int(start_invalid.sum()), int(end_invalid.sum())
        frame.loc[start_invalid.index[start_invalid], "history_valid"] = False
        frame.loc[start_invalid.index[start_invalid], "invalid_reason"] = (
            "NO_AUTHORITATIVE_RECORDING_START_BEFORE_FIRST_EVENT"
        )
        frame.loc[end_invalid.index[end_invalid], "history_valid"] = False
        frame.loc[end_invalid.index[end_invalid], "invalid_reason"] = "AFTER_RECORDING_END"
        files.append(
            {
                **item,
                "h5_metadata": metadata,
                "start_uncovered_queries": start_uncovered,
                "end_uncovered_queries": end_uncovered,
            }
        )
    output_dir.mkdir(parents=True)
    mask = cast(
        pd.DataFrame,
        frame[["fit_ordinal", "sample_token", "sequence_id", "history_valid", "invalid_reason"]],
    ).sort_values(by="fit_ordinal")
    valid_count, blocked_queries = _history_counts(mask)
    invalid_sequences = cast(
        pd.Series, mask.loc[~cast(pd.Series, mask["history_valid"]), "sequence_id"]
    )
    blocked_sequences = int(invalid_sequences.astype(str).nunique())
    mask_path = output_dir / "history_admissibility.parquet"
    mask.to_parquet(mask_path, index=False)
    role_masks: dict[str, Any] = {}
    for key in ROLE_KEYS:
        reference = raw["roles"][key]
        role = pd.read_parquet(raw_receipt.parent / str(reference["path"]))
        view = role[["rotation", "rotation_role", "fit_ordinal", "sample_token"]].merge(
            mask[["fit_ordinal", "sample_token", "history_valid", "invalid_reason"]],
            on=["fit_ordinal", "sample_token"],
            how="left",
            validate="one_to_one",
        )
        role_path = output_dir / "roles" / f"{key}.parquet"
        role_path.parent.mkdir(parents=True, exist_ok=True)
        view.to_parquet(role_path, index=False)
        valid_rows, invalid_rows = _history_counts(view)
        role_masks[key] = {
            "path": role_path.relative_to(output_dir).as_posix(),
            "sha256": file_sha256(role_path),
            "rows": len(view),
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
        }
    generation_masks: dict[str, Any] = {}
    generated_valid: set[str] = set()
    fold_path = Path(raw["fold_contract_path"])
    for rotation in range(6):
        generation = _load_rotation_role_unverified(fold_path, rotation=rotation, role="generation")
        view = generation[["rotation", "rotation_role", "fit_ordinal", "sample_token"]].merge(
            mask[["fit_ordinal", "sample_token", "history_valid", "invalid_reason"]],
            on=["fit_ordinal", "sample_token"],
            how="left",
            validate="one_to_one",
        )
        valid_tokens = set(view.loc[view["history_valid"], "sample_token"].astype(str))
        if generated_valid & valid_tokens:
            raise ValueError("Valid generation mask repeats a query")
        generated_valid |= valid_tokens
        key = f"rotation_{rotation}_generation"
        generation_path = output_dir / "generation" / f"{key}.parquet"
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        view.to_parquet(generation_path, index=False)
        _, invalid_rows = _history_counts(view)
        generation_masks[key] = {
            "path": generation_path.relative_to(output_dir).as_posix(),
            "sha256": file_sha256(generation_path),
            "rows": len(view),
            "valid_rows": len(valid_tokens),
            "invalid_rows": invalid_rows,
        }
    expected_valid_tokens = set(mask.loc[mask["history_valid"], "sample_token"].astype(str))
    if generated_valid != expected_valid_tokens:
        raise ValueError("Valid generation masks do not cover admissible queries exactly once")
    receipt: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": (
            "PASS_ALL_HISTORY_VALID_NOT_HASHED"
            if blocked_queries == 0
            else "PASS_WITH_INVALID_HISTORY_ROWS_EXCLUDED_NOT_HASHED"
        ),
        "inventory_path": str(inventory_path.resolve(strict=True)),
        "inventory_sha256": file_sha256(inventory_path),
        "raw_history_receipt_path": inventory["raw_history_receipt_path"],
        "raw_history_receipt_sha256": inventory["raw_history_receipt_sha256"],
        "files": files,
        "file_count": 24,
        "total_bytes": inventory["total_bytes"],
        "blocked_sequences": blocked_sequences,
        "blocked_queries": blocked_queries,
        "invalid_history_queries": blocked_queries,
        "admissible_queries": valid_count,
        "history_admissibility_path": mask_path.name,
        "history_admissibility_sha256": file_sha256(mask_path),
        "role_masks": role_masks,
        "generation_masks": generation_masks,
        "valid_generation_coverage": "EACH_ADMISSIBLE_QUERY_EXACTLY_ONCE",
        "full_hashes_computed": False,
        "event_extrema_checked": True,
        "ms_index_bounds_checked": True,
        "recording_acquisition_bounds_verified": False,
        "stream_bounds_checked": False,
        "training_admitted": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (output_dir / "PREFLIGHT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def build_h5_admission(
    preflight_path: Path, output_dir: Path, *, resume_dir: Path | None = None
) -> dict[str, Any]:
    """Sequentially hash and inspect all 24 streams, with per-file resumable receipts."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    preflight = verify_h5_metadata_preflight(preflight_path)
    if preflight["status"] not in {
        "PASS_ALL_HISTORY_VALID_NOT_HASHED",
        "PASS_WITH_INVALID_HISTORY_ROWS_EXCLUDED_NOT_HASHED",
    }:
        raise RuntimeError("Physical H5 admission blocked by metadata/bounds preflight")
    inventory_path = Path(preflight["inventory_path"])
    inventory = verify_h5_inventory(inventory_path)
    work = resume_dir or output_dir.with_name(f".{output_dir.name}.work")
    work.mkdir(parents=True, exist_ok=True)
    admitted: list[dict[str, Any]] = []
    for ordinal, item in enumerate(inventory["files"]):
        part = work / f"{ordinal:02d}_{item['sequence_id']}.json"
        stat = Path(item["absolute_path"]).stat()
        if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("Event H5 was substituted after inventory")
        expected_base: dict[str, Any] = {
            **item,
            "ordinal": ordinal,
            "h5": _inspect_h5(
                Path(item["absolute_path"]), item["required_start_us"], item["required_end_us"]
            ),
        }
        if part.exists():
            previous = _json(part)
            _self_hash(previous)
            previous_base = {
                k: v for k, v in previous.items() if k not in {"receipt_sha256", "file_sha256"}
            }
            if (
                previous_base != expected_base
                or previous.get("file_sha256") != item["expected_public_sha256"]
            ):
                raise ValueError("Resumed H5 admission part differs")
            admitted.append(previous)
            continue
        expected_base["file_sha256"] = _sha256_stream(Path(item["absolute_path"]))
        if expected_base["file_sha256"] != item["expected_public_sha256"]:
            raise ValueError("Event H5 digest differs from public eAP release")
        expected_base["receipt_sha256"] = canonical_sha256(expected_base)
        pending = part.with_suffix(".pending")
        pending.write_text(
            json.dumps(expected_base, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pending.replace(part)
        admitted.append(expected_base)
    receipt: dict[str, Any] = {
        "schema": ADMISSION_SCHEMA,
        "status": "PHYSICAL_H5_HASHES_AND_BOUNDS_VERIFIED",
        "inventory_path": str(inventory_path.resolve(strict=True)),
        "inventory_sha256": file_sha256(inventory_path),
        "preflight_path": str(preflight_path.resolve(strict=True)),
        "preflight_sha256": file_sha256(preflight_path),
        "raw_history_receipt_path": inventory["raw_history_receipt_path"],
        "raw_history_receipt_sha256": inventory["raw_history_receipt_sha256"],
        "history_admissibility_path": preflight["history_admissibility_path"],
        "history_admissibility_sha256": preflight["history_admissibility_sha256"],
        "admissible_queries": preflight["admissible_queries"],
        "invalid_history_queries": preflight["blocked_queries"],
        "files": admitted,
        "file_count": 24,
        "total_bytes": inventory["total_bytes"],
        "full_hashes_computed": True,
        "stream_bounds_checked": True,
        "training_admitted": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    output_dir.mkdir(parents=True)
    (output_dir / "ADMISSION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_h5_admission(output_dir / "ADMISSION.json")
    parts_dir = output_dir / "parts"
    if work.resolve() != parts_dir.resolve():
        work.replace(parts_dir)
    return receipt


def verify_h5_inventory(path: Path) -> dict[str, Any]:
    receipt = _json(path.resolve(strict=True))
    _self_hash(receipt)
    if (
        receipt.get("schema") != PLAN_SCHEMA
        or receipt.get("status") != "INVENTORIED_NOT_HASHED_NOT_ADMITTED"
        or receipt.get("file_count") != 24
        or len(receipt.get("files", [])) != 24
        or receipt.get("h5_files_opened") is not False
        or receipt.get("full_hashes_computed") is not False
        or receipt.get("stream_bounds_checked") is not False
        or receipt.get("training_admitted") is not False
    ):
        raise ValueError("H5 inventory receipt differs")
    raw = Path(receipt["raw_history_receipt_path"]).resolve(strict=True)
    if file_sha256(raw) != receipt["raw_history_receipt_sha256"]:
        raise ValueError("Raw-history receipt changed after H5 inventory")
    raw_receipt = _json(raw)
    public_path = Path(receipt["public_manifest_path"]).resolve(strict=True)
    if file_sha256(public_path) != receipt.get("public_manifest_sha256"):
        raise ValueError("Public eAP manifest changed after H5 inventory")
    public = _json(public_path)
    public_files = public.get("files")
    if not isinstance(public_files, dict):
        raise ValueError("Public eAP manifest has no file inventory")
    index_path = (raw.parent / str(raw_receipt["index_path"])).resolve(strict=True)
    if file_sha256(index_path) != raw_receipt["index_sha256"]:
        raise ValueError("Raw-history index changed after H5 inventory")
    role_groups: dict[str, set[str]] = {}
    for key in ROLE_KEYS:
        reference = raw_receipt["roles"][key]
        role_path = (raw.parent / str(reference["path"])).resolve(strict=True)
        if file_sha256(role_path) != reference["sha256"]:
            raise ValueError("Raw-history role changed after H5 inventory")
        role = pd.read_parquet(role_path, columns=["sequence_id"])
        role_groups[key] = set(role["sequence_id"].astype(str))
    frame = pd.read_parquet(
        index_path,
        columns=["sequence_id", "observation_source_start_us", "event_history_anchor_us"],
    )
    observed_sequences = {item["sequence_id"] for item in receipt["files"]}
    if observed_sequences != set(frame["sequence_id"].astype(str)):
        raise ValueError("H5 inventory sequence identity differs from raw history")
    total = 0
    root = Path(receipt["eap_root"]).resolve(strict=True)
    for item in receipt["files"]:
        sequence = item["sequence_id"]
        event_path = _safe_event_path(root, sequence, item["relative_path"])
        stat = event_path.stat()
        rows = frame.loc[frame["sequence_id"].astype(str) == sequence]
        expected_roles = sorted(key for key, groups in role_groups.items() if sequence in groups)
        release = public_files.get(item["relative_path"])
        if (
            not isinstance(release, dict)
            or item.get("expected_public_sha256") != release.get("sha256")
            or item.get("expected_public_size_bytes") != release.get("size_bytes")
            or str(event_path) != item["absolute_path"]
            or stat.st_size != item["size_bytes"]
            or stat.st_mtime_ns != item["mtime_ns"]
            or int(rows["observation_source_start_us"].min()) != item["required_start_us"]
            or int(rows["event_history_anchor_us"].max()) != item["required_end_us"]
            or len(rows) != item["queries"]
            or item["roles"] != expected_roles
        ):
            raise ValueError("H5 inventory file/bounds/role binding differs")
        total += stat.st_size
    if total != receipt.get("total_bytes") or total != receipt.get("hash_read_bytes_pending"):
        raise ValueError("H5 inventory byte total differs")
    return receipt


def verify_h5_metadata_preflight(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    receipt = _json(path.resolve(strict=True))
    _self_hash(receipt)
    inventory_path = Path(receipt.get("inventory_path", "")).resolve(strict=True)
    inventory = verify_h5_inventory(inventory_path)
    if (
        receipt.get("schema") != PREFLIGHT_SCHEMA
        or receipt.get("status")
        not in {
            "PASS_ALL_HISTORY_VALID_NOT_HASHED",
            "PASS_WITH_INVALID_HISTORY_ROWS_EXCLUDED_NOT_HASHED",
        }
        or receipt.get("inventory_sha256") != file_sha256(inventory_path)
        or receipt.get("raw_history_receipt_sha256") != inventory["raw_history_receipt_sha256"]
        or receipt.get("file_count") != 24
        or receipt.get("total_bytes") != inventory["total_bytes"]
        or len(receipt.get("files", [])) != 24
        or receipt.get("full_hashes_computed") is not False
        or receipt.get("event_extrema_checked") is not True
        or receipt.get("ms_index_bounds_checked") is not True
        or receipt.get("recording_acquisition_bounds_verified") is not False
        or receipt.get("stream_bounds_checked") is not False
        or receipt.get("training_admitted") is not False
    ):
        raise ValueError("H5 metadata preflight receipt differs")
    blocked = sum(
        item["start_uncovered_queries"] + item["end_uncovered_queries"] for item in receipt["files"]
    )
    population_queries = sum(item["queries"] for item in receipt["files"])
    if (
        blocked != receipt.get("blocked_queries")
        or receipt.get("admissible_queries") != population_queries - blocked
        or (blocked == 0) != (receipt["status"] == "PASS_ALL_HISTORY_VALID_NOT_HASHED")
    ):
        raise ValueError("H5 metadata preflight coverage summary differs")
    raw = _json(Path(receipt["raw_history_receipt_path"]))
    frame = pd.read_parquet(
        Path(receipt["raw_history_receipt_path"]).parent / str(raw["index_path"]),
        columns=[
            "fit_ordinal",
            "sample_token",
            "sequence_id",
            "observation_source_start_us",
            "event_history_anchor_us",
        ],
    )
    expected_valid = pd.Series(True, index=frame.index)
    expected_reason = pd.Series("", index=frame.index, dtype="str")
    for item in receipt["files"]:
        metadata = _inspect_h5_metadata(
            Path(item["absolute_path"]), item["required_start_us"], item["required_end_us"]
        )
        rows = frame.loc[frame["sequence_id"].astype(str) == item["sequence_id"]]
        if (
            metadata != item["h5_metadata"]
            or int(
                (rows["observation_source_start_us"] < int(metadata["admission_start_us"])).sum()
            )
            != item["start_uncovered_queries"]
            or int((rows["event_history_anchor_us"] > int(metadata["admission_end_us"])).sum())
            != item["end_uncovered_queries"]
        ):
            raise ValueError("Physical H5 metadata/bounds changed after preflight")
        start_bad = rows["observation_source_start_us"] < int(metadata["admission_start_us"])
        end_bad = rows["event_history_anchor_us"] > int(metadata["admission_end_us"])
        expected_valid.loc[start_bad.index[start_bad]] = False
        expected_valid.loc[end_bad.index[end_bad]] = False
        expected_reason.loc[start_bad.index[start_bad]] = (
            "NO_AUTHORITATIVE_RECORDING_START_BEFORE_FIRST_EVENT"
        )
        expected_reason.loc[end_bad.index[end_bad]] = "AFTER_RECORDING_END"
    mask_path = (path.parent / str(receipt["history_admissibility_path"])).resolve(strict=True)
    mask = pd.read_parquet(mask_path)
    expected_mask = frame[["fit_ordinal", "sample_token", "sequence_id"]].copy()
    expected_mask["history_valid"] = expected_valid
    expected_mask["invalid_reason"] = expected_reason
    if (
        mask_path.parent != path.parent
        or file_sha256(mask_path) != receipt.get("history_admissibility_sha256")
        or not mask.equals(expected_mask)
    ):
        raise ValueError("H5 history-admissibility mask differs")
    if set(receipt.get("role_masks", {})) != set(ROLE_KEYS):
        raise ValueError("H5 role mask inventory differs")
    for key in ROLE_KEYS:
        reference = receipt["role_masks"][key]
        role_path = (path.parent / str(reference["path"])).resolve(strict=True)
        role = pd.read_parquet(role_path)
        raw_role = pd.read_parquet(
            Path(receipt["raw_history_receipt_path"]).parent / str(raw["roles"][key]["path"])
        )
        expected_role = raw_role[
            ["rotation", "rotation_role", "fit_ordinal", "sample_token"]
        ].merge(
            mask[["fit_ordinal", "sample_token", "history_valid", "invalid_reason"]],
            on=["fit_ordinal", "sample_token"],
            how="left",
            validate="one_to_one",
        )
        valid_rows, invalid_rows = _history_counts(role)
        if (
            file_sha256(role_path) != reference["sha256"]
            or len(role) != reference["rows"]
            or not role.equals(expected_role)
            or valid_rows != reference["valid_rows"]
            or invalid_rows != reference["invalid_rows"]
        ):
            raise ValueError("H5 role admissibility mask differs")
    if (
        set(receipt.get("generation_masks", {}))
        != {f"rotation_{rotation}_generation" for rotation in range(6)}
        or receipt.get("valid_generation_coverage") != "EACH_ADMISSIBLE_QUERY_EXACTLY_ONCE"
    ):
        raise ValueError("H5 generation mask inventory differs")
    generated_valid: set[str] = set()
    fold_path = Path(raw["fold_contract_path"])
    for rotation in range(6):
        key = f"rotation_{rotation}_generation"
        reference = receipt["generation_masks"][key]
        generation_path = (path.parent / str(reference["path"])).resolve(strict=True)
        generation = pd.read_parquet(generation_path)
        raw_generation = _load_rotation_role_unverified(
            fold_path, rotation=rotation, role="generation"
        )
        expected_generation = raw_generation[
            ["rotation", "rotation_role", "fit_ordinal", "sample_token"]
        ].merge(
            mask[["fit_ordinal", "sample_token", "history_valid", "invalid_reason"]],
            on=["fit_ordinal", "sample_token"],
            how="left",
            validate="one_to_one",
        )
        tokens = set(generation.loc[generation["history_valid"], "sample_token"].astype(str))
        valid_rows, invalid_rows = _history_counts(generation)
        if (
            file_sha256(generation_path) != reference["sha256"]
            or not generation.equals(expected_generation)
            or generated_valid & tokens
            or valid_rows != reference["valid_rows"]
            or invalid_rows != reference["invalid_rows"]
        ):
            raise ValueError("H5 generation admissibility mask differs")
        generated_valid |= tokens
    if generated_valid != set(mask.loc[mask["history_valid"], "sample_token"].astype(str)):
        raise ValueError("Valid generation coverage differs")
    return receipt


def verify_h5_admission(path: Path, *, full_hash: bool = False) -> dict[str, Any]:
    """Verify admission bindings; re-read full H5 bytes only when explicitly requested."""
    receipt = _json(path.resolve(strict=True))
    _self_hash(receipt)
    inventory_path = Path(receipt.get("inventory_path", "")).resolve(strict=True)
    inventory = verify_h5_inventory(inventory_path)
    preflight_path = Path(receipt.get("preflight_path", "")).resolve(strict=True)
    preflight = verify_h5_metadata_preflight(preflight_path)
    if (
        receipt.get("schema") != ADMISSION_SCHEMA
        or receipt.get("status") != "PHYSICAL_H5_HASHES_AND_BOUNDS_VERIFIED"
        or receipt.get("inventory_sha256") != file_sha256(inventory_path)
        or receipt.get("preflight_sha256") != file_sha256(preflight_path)
        or preflight.get("status")
        not in {
            "PASS_ALL_HISTORY_VALID_NOT_HASHED",
            "PASS_WITH_INVALID_HISTORY_ROWS_EXCLUDED_NOT_HASHED",
        }
        or receipt.get("raw_history_receipt_sha256") != inventory["raw_history_receipt_sha256"]
        or receipt.get("file_count") != 24
        or receipt.get("total_bytes") != inventory["total_bytes"]
        or receipt.get("full_hashes_computed") is not True
        or receipt.get("stream_bounds_checked") is not True
        or receipt.get("training_admitted") is not True
        or receipt.get("history_admissibility_sha256")
        != preflight.get("history_admissibility_sha256")
        or receipt.get("admissible_queries") != preflight.get("admissible_queries")
        or len(receipt.get("files", [])) != 24
    ):
        raise ValueError("Physical H5 admission receipt differs")
    for expected, observed in zip(inventory["files"], receipt["files"], strict=True):
        path_value = Path(observed["absolute_path"]).resolve(strict=True)
        if (
            observed["sequence_id"] != expected["sequence_id"]
            or observed["roles"] != expected["roles"]
            or path_value.stat().st_size != expected["size_bytes"]
            or path_value.stat().st_mtime_ns != expected["mtime_ns"]
            or observed.get("file_sha256") != expected["expected_public_sha256"]
            or (full_hash and _sha256_stream(path_value) != expected["expected_public_sha256"])
            or _inspect_h5(path_value, expected["required_start_us"], expected["required_end_us"])
            != observed.get("h5")
        ):
            raise ValueError("Physical H5 file/hash/bounds/role identity differs")
    return {
        "status": "PASS",
        "files": 24,
        "bytes": receipt["total_bytes"],
        "full_hash_reverified": full_hash,
    }


__all__ = [
    "build_h5_admission",
    "build_h5_inventory",
    "build_h5_metadata_preflight",
    "verify_h5_admission",
    "verify_h5_inventory",
    "verify_h5_metadata_preflight",
]
