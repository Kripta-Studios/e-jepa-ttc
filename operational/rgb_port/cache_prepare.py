"""Durably materialize lossless RGB producer inputs in homogeneous T2/T3 shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from e_jepa_ttc.rgb_port.data import TarFrameReader, decode_query
from operational.rgb_port.prepare import _resource_snapshot, atomic_json, file_sha256

SHARD_SIZE = 32
CACHE_FIELDS = (
    "rgb_uint8",
    "frame_times_us",
    "sensor_frame_times_us",
    "annotation_frame_times_us",
    "delta_t_s",
    "frame_valid",
    "foreground_mask",
    "boxes_in_crop_xyxy",
    "visible_heights_px",
    "log_visible_heights",
    "roi_xyxy",
    "requested_roi_xyxy",
)
_READER: TarFrameReader | None = None
_EAP_ROOT: Path | None = None


def _init_worker(eap_root: str) -> None:
    global _READER, _EAP_ROOT
    _EAP_ROOT = Path(eap_root)
    _READER = TarFrameReader(_EAP_ROOT, max_open=4, max_frame_cache_bytes=256 * 1024**2)


def _write_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name, suffix=".pending", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            savez = cast(Any, np.savez_compressed)
            savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def _task(
    rows: list[dict[str, Any]], indices: list[int], path: str, role_manifest_sha256: str
) -> dict[str, Any]:
    if _READER is None or _EAP_ROOT is None:
        raise RuntimeError("RGB cache worker was not initialized")
    decoded = [decode_query(row, eap_root=_EAP_ROOT, reader=_READER) for row in rows]
    values = {
        "indices": np.asarray(indices, dtype=np.int64),
        "tokens": np.asarray([str(row["sample_token"]) for row in rows]),
        **{field: np.stack([item[field] for item in decoded]) for field in CACHE_FIELDS},
    }
    target = Path(path)
    _write_npz(target, values)
    with zipfile.ZipFile(target) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise OSError(f"NPZ CRC failed for {bad_member}")
    receipt = {
        "schema": "rgb_port_input_shard_receipt_v1",
        "status": "COMPLETE",
        "path": str(target.resolve()),
        "sha256": file_sha256(target),
        "bytes": target.stat().st_size,
        "indices": indices,
        "tokens": values["tokens"].astype(str).tolist(),
        "T": int(values["rgb_uint8"].shape[1]),
        "role_manifest_sha256": role_manifest_sha256,
        "npz_crc_verified": True,
    }
    atomic_json(target.with_suffix(".json"), receipt)
    return receipt


def _existing(
    path: Path, indices: list[int], tokens: list[str], role_sha: str
) -> dict[str, Any] | None:
    receipt_path = path.with_suffix(".json")
    if not path.is_file() or not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("indices") != indices
        or receipt.get("tokens") != tokens
        or receipt.get("role_manifest_sha256") != role_sha
        or file_sha256(path) != receipt.get("sha256")
    ):
        return None
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            return None
    return cast(dict[str, Any], receipt)


def _partitions(rows: pd.DataFrame) -> list[tuple[list[int], list[dict[str, Any]], str]]:
    result: list[tuple[list[int], list[dict[str, Any]], str]] = []
    records = cast(list[dict[str, Any]], rows.to_dict(orient="records"))
    for count in (2, 3):
        indices = np.flatnonzero(rows.producer_frame_count.to_numpy() == count).tolist()
        for shard_number, start in enumerate(range(0, len(indices), SHARD_SIZE)):
            selected = list(map(int, indices[start : start + SHARD_SIZE]))
            result.append(
                (
                    selected,
                    [records[index] for index in selected],
                    f"T{count}/shard_{shard_number:05d}.npz",
                )
            )
    return result


def _parity(
    rows: pd.DataFrame, shards: list[dict[str, Any]], eap_root: Path, cache_root: Path
) -> dict[str, Any]:
    records = cast(list[dict[str, Any]], rows.to_dict(orient="records"))
    selected: list[int] = []
    for count in (2, 3):
        candidates = np.flatnonzero(rows.producer_frame_count.to_numpy() == count)
        selected.extend([int(candidates[0]), int(candidates[-1])])
    lookup = {
        int(index): (shard, local)
        for shard in shards
        for local, index in enumerate(shard["indices"])
    }
    reader = TarFrameReader(eap_root)
    checks: list[dict[str, Any]] = []
    try:
        for index in selected:
            expected = decode_query(records[index], eap_root=eap_root, reader=reader)
            shard, local = lookup[index]
            with np.load(Path(shard["path"]), allow_pickle=False) as stored:
                equal = {
                    field: bool(np.array_equal(expected[field], stored[field][local]))
                    for field in CACHE_FIELDS
                }
            if not all(equal.values()):
                raise ValueError(f"prepared RGB cache differs at dataset index {index}")
            checks.append(
                {
                    "index": index,
                    "token": str(records[index]["sample_token"]),
                    "T": int(records[index]["producer_frame_count"]),
                    "fields_equal": equal,
                }
            )
    finally:
        reader.close()
    receipt = {
        "schema": "rgb_port_cache_parity_v1",
        "status": "COMPLETE",
        "scope": "first_and_last_query_of_each_T_bucket",
        "checks": checks,
        "all_fields_bit_exact": True,
        "optimizer_updates": 0,
    }
    atomic_json(cache_root / "CACHE_PARITY.json", receipt)
    return receipt


def materialize(role_manifest_path: Path, cache_root: Path, *, workers: int) -> dict[str, Any]:
    """Resume or create the complete lossless cache for one immutable role."""
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    role_manifest_path = role_manifest_path.resolve(strict=True)
    role = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    if role.get("status") != "COMPLETE" or role.get("role") != "P":
        raise ValueError("full prepared RGB cache is authorized for COMPLETE role P only")
    role_sha = file_sha256(role_manifest_path)
    rows = pd.read_parquet(role["rows_path"])
    if len(rows) != int(role["population_size"]):
        raise ValueError("role rows and population size differ")
    cache_root.mkdir(parents=True, exist_ok=True)
    pending_work: list[tuple[list[int], list[dict[str, Any]], Path]] = []
    completed: list[dict[str, Any]] = []
    partitions = _partitions(rows)
    total_shards = len(partitions)
    for indices, records, relative in partitions:
        path = cache_root / relative
        existing = _existing(path, indices, [str(row["sample_token"]) for row in records], role_sha)
        if existing is None:
            pending_work.append((indices, records, path))
        else:
            completed.append(existing)
    failures: list[dict[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(Path(role["eap_root"]).resolve(strict=True)),),
    ) as pool:
        pending: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
        iterator = iter(pending_work)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < 8 and not exhausted:
                allowed, resources = _resource_snapshot(cache_root)
                if not allowed:
                    atomic_json(
                        cache_root / "RESOURCE_PAUSE.json",
                        {"status": "PAUSED_RESOURCE", **resources},
                    )
                    raise RuntimeError(f"resource guard paused RGB cache: {resources}")
                try:
                    indices, records, path = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = pool.submit(_task, records, indices, str(path), role_sha)
                pending[future] = str(path)
            if pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    path = pending.pop(future)
                    try:
                        completed.append(future.result())
                    except Exception as error:
                        failures.append({"path": path, "error": repr(error)})
            atomic_json(
                cache_root / "PROGRESS.json",
                {
                    "status": "RUNNING" if not failures else "FAILED",
                    "completed_shards": len(completed),
                    "total_shards": total_shards,
                    "failures": failures,
                },
            )
    if failures:
        raise RuntimeError(f"RGB cache preparation had {len(failures)} failed shards")
    completed.sort(key=lambda item: (int(item["T"]), int(item["indices"][0])))
    covered = sorted(index for item in completed for index in map(int, item["indices"]))
    if covered != list(range(len(rows))):
        raise ValueError("complete RGB cache does not cover every P dataset index exactly once")
    parity = _parity(rows, completed, Path(role["eap_root"]), cache_root)
    manifest = {
        "schema": "rgb_port_prepared_input_cache_v1",
        "status": "COMPLETE",
        "role": "P",
        "role_manifest_path": str(role_manifest_path),
        "role_manifest_sha256": role_sha,
        "population_size": len(rows),
        "shard_size": SHARD_SIZE,
        "compression": "numpy_savez_compressed_lossless",
        "stored_pixels": "uint8_RGB",
        "float_conversion": "derived_on_read_exact_uint8_float32_div255",
        "no_frame_padding_or_duplication": True,
        "decode_parity_verified": parity["all_fields_bit_exact"],
        "cache_fields": list(CACHE_FIELDS),
        "shards": completed,
        "compressed_bytes": sum(int(item["bytes"]) for item in completed),
    }
    manifest_path = cache_root / "CACHE_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    binding = {
        "schema": "rgb_port_cache_binding_v1",
        "status": "COMPLETE",
        "role_manifest_sha256": role_sha,
        "cache_manifest_path": str(manifest_path.resolve()),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "equivalence": "bit_exact_decoded_inputs_no_recipe_change",
    }
    atomic_json(role_manifest_path.with_name("P_RGB_CACHE_BINDING.json"), binding)
    atomic_json(
        cache_root / "PROGRESS.json",
        {
            "status": "COMPLETE",
            "completed_shards": len(completed),
            "total_shards": len(completed),
            "failures": [],
        },
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--role-manifest", type=Path, required=True)
    result.add_argument("--cache-root", type=Path, required=True)
    result.add_argument("--workers", type=int, default=1)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = materialize(args.role_manifest, args.cache_root, workers=args.workers)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "population_size", "compressed_bytes")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
