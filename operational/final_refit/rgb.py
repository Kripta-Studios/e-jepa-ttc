"""Exact all-TRAIN40 RGB inputs and frozen generic-DINO teacher lookup."""

from __future__ import annotations

import concurrent.futures
import ctypes
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.rgb_port.data import RGBPortDataset, as_list
from operational.rgb_port.recipe import file_sha256

ROLES = ("P", "H", "V")
_CACHE_MIN_PHYSICAL_BYTES = 4 * 1024**3
_CACHE_MIN_COMMIT_BYTES = 8 * 1024**3
_CACHE_MAX_TREE_RSS_BYTES = 4 * 1024**3
_CACHE_MIN_DISK_BYTES = 10 * 1024**3


def _windows_commit_headroom() -> int | None:
    if os.name != "nt":
        return None

    class Performance(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "CommitTotal",
                    "CommitLimit",
                    "CommitPeak",
                    "PhysicalTotal",
                    "PhysicalAvailable",
                    "SystemCache",
                    "KernelTotal",
                    "KernelPaged",
                    "KernelNonpaged",
                    "PageSize",
                )
            ],
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    counters = Performance()
    counters.cb = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(counters), counters.cb):
        raise OSError("Windows commit counters unavailable")
    return int((counters.CommitLimit - counters.CommitTotal) * counters.PageSize)


def _cache_resources_allowed(values: Mapping[str, int | None]) -> bool:
    """Apply the measured single-worker cache envelope.

    The interrupted H resume retained roughly 14.2 GiB of commit while its
    process tree remained near 2 GiB RSS. A 4 GiB tree ceiling therefore stays
    conservative without pausing on unrelated host fluctuations above it.
    """
    commit = values["windows_commit_headroom_bytes"]
    return bool(
        int(values["host_available_bytes"] or 0) >= _CACHE_MIN_PHYSICAL_BYTES
        and int(values["tree_rss_bytes"] or 0) <= _CACHE_MAX_TREE_RSS_BYTES
        and (commit is None or int(commit) >= _CACHE_MIN_COMMIT_BYTES)
        and int(values["disk_free_bytes"] or 0) >= _CACHE_MIN_DISK_BYTES
    )


def _cache_resource_snapshot(destination: Path) -> tuple[bool, dict[str, int | None]]:
    """Guard physical RAM, Windows commit and disk before each sole in-flight shard."""
    import psutil

    process = psutil.Process()
    tree = [process, *process.children(recursive=True)]
    rss = sum(item.memory_info().rss for item in tree if item.is_running())
    available = int(psutil.virtual_memory().available)
    commit = _windows_commit_headroom()
    disk = int(__import__("shutil").disk_usage(destination).free)
    values: dict[str, int | None] = {
        "tree_rss_bytes": rss,
        "host_available_bytes": available,
        "windows_commit_headroom_bytes": commit,
        "disk_free_bytes": disk,
        "minimum_host_available_bytes": _CACHE_MIN_PHYSICAL_BYTES,
        "minimum_windows_commit_headroom_bytes": _CACHE_MIN_COMMIT_BYTES,
        "maximum_tree_rss_bytes": _CACHE_MAX_TREE_RSS_BYTES,
        "minimum_disk_free_bytes": _CACHE_MIN_DISK_BYTES,
        "maximum_inflight_shards": 1,
    }
    return _cache_resources_allowed(values), values


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TeacherShardCache:
    """Bounded, hash-checking reader over the complete native TRAIN40 teacher."""

    def __init__(self, manifest_path: Path, *, max_bytes: int = 64 * 1024**2) -> None:
        self.manifest_path = manifest_path.resolve(strict=True)
        self.manifest_sha256 = file_sha256(self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "train40_complete_fragment_manifest_v1"
            or manifest.get("status") != "COMPLETE_VERIFIED"
            or manifest.get("kind") != "TEACHER"
            or int(manifest.get("row_count", -1)) != 88_744
        ):
            raise ValueError("Teacher manifest is not the complete verified TRAIN40 cache")
        directory = Path(str(manifest["directory"])).resolve(strict=True)
        self.rows_sha256 = str(manifest["rows_sha256"])
        self.freeze_sha256 = str(manifest["freeze_sha256"])
        self.shards = tuple(
            {
                **item,
                "path": str((directory / str(item["name"])).resolve(strict=True)),
            }
            for item in manifest["ordered_shards"]
        )
        if not self.shards or int(self.shards[0]["start"]) != 0:
            raise ValueError("Teacher shards do not begin at native ordinal zero")
        if (
            any(
                int(left["stop"]) != int(right["start"])
                for left, right in zip(self.shards, self.shards[1:], strict=False)
            )
            or int(self.shards[-1]["stop"]) != 88_744
        ):
            raise ValueError("Teacher shards are not a contiguous TRAIN40 partition")
        self._by_ordinal: list[int] = [-1] * 88_744
        for shard_index, item in enumerate(self.shards):
            for ordinal in range(int(item["start"]), int(item["stop"])):
                self._by_ordinal[ordinal] = shard_index
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("Teacher cache byte bound must be positive")
        self._loaded: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._sizes: dict[int, int] = {}
        self.retained_bytes = 0
        self.peak_retained_bytes = 0
        self.reads = 0
        self.hits = 0

    def _load(self, shard_index: int) -> dict[str, np.ndarray]:
        cached = self._loaded.get(shard_index)
        if cached is not None:
            self.hits += 1
            self._loaded.move_to_end(shard_index)
            return cached
        item = self.shards[shard_index]
        path = Path(str(item["path"]))
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Teacher shard changed: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            stored = {name: archive[name] for name in archive.files}
        required = {"ordinals", "tokens", "relation_targets", "relation_valid"}
        if not required <= set(stored):
            raise ValueError("Teacher shard fields differ")
        expected = np.arange(int(item["start"]), int(item["stop"]), dtype=np.int64)
        if not np.array_equal(stored["ordinals"], expected):
            raise ValueError("Teacher shard ordinal identity differs")
        size = sum(value.nbytes for value in stored.values())
        self.reads += 1
        if size <= self.max_bytes:
            while self._loaded and self.retained_bytes + size > self.max_bytes:
                stale, _ = self._loaded.popitem(last=False)
                self.retained_bytes -= self._sizes.pop(stale)
            self._loaded[shard_index] = stored
            self._sizes[shard_index] = size
            self.retained_bytes += size
            self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        return stored

    def get(self, ordinals: Sequence[int], tokens: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if len(ordinals) != len(tokens) or not ordinals:
            raise ValueError("Teacher lookup requires aligned nonempty ordinals and tokens")
        values: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for ordinal, token in zip(ordinals, tokens, strict=True):
            if not 0 <= int(ordinal) < len(self._by_ordinal):
                raise IndexError(f"Native teacher ordinal out of range: {ordinal}")
            shard_index = self._by_ordinal[int(ordinal)]
            item = self.shards[shard_index]
            stored = self._load(shard_index)
            local = int(ordinal) - int(item["start"])
            if int(stored["ordinals"][local]) != int(ordinal) or str(
                stored["tokens"][local]
            ) != str(token):
                raise ValueError("Teacher row token/ordinal identity differs")
            values.append(stored["relation_targets"][local])
            valid.append(stored["relation_valid"][local])
        return np.stack(values), np.stack(valid)

    def snapshot(self) -> dict[str, int]:
        return {
            "reads": self.reads,
            "hits": self.hits,
            "retained_shards": len(self._loaded),
            "retained_bytes": self.retained_bytes,
            "peak_retained_bytes": self.peak_retained_bytes,
            "max_bytes": self.max_bytes,
        }


class AllTrainRGBSource:
    """Explicit P+H+V union; role identities are preserved rather than relabelled as P."""

    def __init__(
        self,
        role_manifests: Mapping[str, Path],
        teacher_manifest: Path,
        *,
        cache_bindings: Mapping[str, Path] | None = None,
    ) -> None:
        if set(role_manifests) != set(ROLES):
            raise ValueError("All-TRAIN RGB source requires exactly P, H and V manifests")
        self.role_manifest_paths = {
            role: role_manifests[role].resolve(strict=True) for role in ROLES
        }
        self.datasets: dict[str, RGBPortDataset] = {
            role: RGBPortDataset(self.role_manifest_paths[role], role=role) for role in ROLES
        }
        if cache_bindings:
            from e_jepa_ttc.rgb_port.data import PreparedRGBCache

            for role, binding in cache_bindings.items():
                if role not in self.datasets:
                    raise ValueError(f"Unknown RGB cache role: {role}")
                self.datasets[role].prepared_cache = PreparedRGBCache(
                    binding.resolve(strict=True), self.role_manifest_paths[role]
                )
        self.teacher = TeacherShardCache(teacher_manifest)
        self._rows: list[tuple[str, int, dict[str, Any]]] = []
        role_identities: dict[str, dict[str, Any]] = {}
        native: dict[int, str] = {}
        for role in ROLES:
            manifest = json.loads(self.role_manifest_paths[role].read_text(encoding="utf-8"))
            dataset = self.datasets[role]
            if len(dataset) != int(manifest["population_size"]):
                raise ValueError(f"{role} manifest population differs from rows")
            role_identities[role] = {
                "manifest_sha256": file_sha256(self.role_manifest_paths[role]),
                "rows_sha256": str(manifest["rows_sha256"]),
                "population_size": len(dataset),
                "group_ids": sorted(map(str, manifest["group_ids"])),
            }
            for local, row in enumerate(dataset.rows):
                ordinal = int(row["source_ordinal"])
                token = str(row["sample_token"])
                if ordinal in native:
                    raise ValueError("P/H/V contain duplicate native TRAIN40 ordinals")
                native[ordinal] = token
                self._rows.append((role, local, row))
        if sorted(native) != list(range(88_744)) or len(set(native.values())) != 88_744:
            raise ValueError("P/H/V are not the exact unique 88,744-row TRAIN40 union")
        # The complete event branch is indexed in native TRAIN40 ordinal order.
        # Sorting the disjoint historical views back to that order preserves the
        # registered E/R shared sampler stream; role-local indices remain intact.
        self._rows.sort(key=lambda item: int(item[2]["source_ordinal"]))
        if [int(row["source_ordinal"]) for _, _, row in self._rows] != list(range(88_744)):
            raise ValueError("RGB union could not be restored to native TRAIN40 ordinal order")
        if self.teacher.rows_sha256 not in {
            str(json.loads(path.read_text(encoding="utf-8"))["source_sha256"])
            for path in self.role_manifest_paths.values()
        }:
            raise ValueError("Teacher rows hash is not bound by the role manifests")
        self.population_size = len(self._rows)
        self.batch_keys = [
            (role, len(as_list(row["producer_members"]))) for role, _, row in self._rows
        ]
        self.frame_counts: Sequence[int] = tuple(
            len(as_list(row["producer_members"])) for _, _, row in self._rows
        )
        if set(self.frame_counts) - {2, 3}:
            raise ValueError("RGB producer histories must contain exactly two or three frames")
        self.input_span_us: Sequence[int] = tuple(
            int(as_list(row["producer_sensor_times_us"])[-1])
            - int(as_list(row["producer_sensor_times_us"])[0])
            for _, _, row in self._rows
        )
        # Cache-local grouping changes only shuffled access locality, never membership.
        self.sampling_groups = tuple(
            tuple(range(start, min(start + 256, self.population_size)))
            for start in range(0, self.population_size, 256)
        )
        group_ids = {
            str(row["group_id"] if "group_id" in row else row["sequence_id"])
            for _, _, row in self._rows
        }
        if len(group_ids) != 40:
            raise ValueError("All-TRAIN RGB source must contain exactly 40 groups")
        self.identity: Mapping[str, Any] = {
            "schema": "final_refit_all_train_rgb_source_v1",
            "role": "TRAIN",
            "roles_preserved": list(ROLES),
            "role_identities": role_identities,
            "population_size": self.population_size,
            "group_count": len(group_ids),
            "native_rows_sha256": self.teacher.rows_sha256,
            "teacher_manifest_path": str(self.teacher.manifest_path),
            "teacher_manifest_sha256": self.teacher.manifest_sha256,
            "teacher_freeze_sha256": self.teacher.freeze_sha256,
            "teacher_lookup": "native_source_ordinal_plus_sample_token",
            "teacher_scope": "complete_TRAIN40_88744",
            "union_order": "native_TRAIN40_source_ordinal_0_to_88743",
            "event_rgb_sampler_order_shared": True,
            "target_anchor": "query_time_us",
            "cache_bindings": {role: self._cache_binding_sha(role) for role in ROLES},
        }
        self.identity = {**self.identity, "identity_sha256": _canonical_sha256(self.identity)}

    def _cache_binding_sha(self, role: str) -> str | None:
        cache = self.datasets[role].prepared_cache
        return None if cache is None else cache.binding_sha256

    def batch(self, indices: Sequence[int], modality: str) -> ObjectEventV4Batch:
        """Decode one homogeneous role/T subset and attach teacher targets by native ordinal."""
        if modality != "rgb" or not indices:
            raise ValueError("AllTrainRGBSource only serves nonempty RGB batches")
        selected = [self._rows[int(index)] for index in indices]
        roles = {role for role, _, _ in selected}
        counts = {self.frame_counts[int(index)] for index in indices}
        if len(roles) != 1 or len(counts) != 1:
            raise ValueError("RGB microbatch must be homogeneous by source role and T")
        role = selected[0][0]
        samples = [self.datasets[role][local] for _, local, _ in selected]
        tensor_fields = (
            "rgb",
            "delta_t_s",
            "boxes_in_crop_xyxy",
            "visible_heights_px",
            "roi_xyxy",
            "target_ttc",
        )
        collated = {
            name: torch.stack([torch.as_tensor(sample[name]) for sample in samples])
            for name in tensor_fields
        }
        rows = [row for _, _, row in selected]
        tokens = [str(row["sample_token"]) for row in rows]
        values, valid = self.teacher.get([int(row["source_ordinal"]) for row in rows], tokens)
        return ObjectEventV4Batch(
            events=collated["rgb"].float(),
            delta_t_s=collated["delta_t_s"].float(),
            observable_motion=torch.empty((len(indices), 0), dtype=torch.float32),
            visible_heights_px=collated["visible_heights_px"].float(),
            target_ttc_s=collated["target_ttc"].float(),
            boxes_xyxy=collated["boxes_in_crop_xyxy"].float(),
            common_square_xyxy=collated["roi_xyxy"].float(),
            sequence_ids=[str(row["sequence_id"]) for row in rows],
            sample_tokens=tokens,
            track_ids=[str(row["track_id"]) for row in rows],
            dinov3_relation_targets=torch.from_numpy(values.astype(np.float32, copy=False)),
            dinov3_relation_valid=torch.from_numpy(valid.astype(np.bool_, copy=False)),
        )

    def row(self, index: int) -> Mapping[str, Any]:
        """Return one immutable role row with explicit native identity."""
        role, local, row = self._rows[int(index)]
        return {**row, "historical_role": role, "historical_role_ordinal": local}

    def metadata(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Return label-free clocks and IDs without decoding RGB pixels."""
        return [
            {
                "ordinal": int(index),
                "native_source_ordinal": int(self._rows[int(index)][2]["source_ordinal"]),
                "historical_role": self._rows[int(index)][0],
                "sample_token": str(self._rows[int(index)][2]["sample_token"]),
                "sequence_id": str(self._rows[int(index)][2]["sequence_id"]),
                "group_id": str(self._rows[int(index)][2]["group_id"]),
                "query_time_us": int(self._rows[int(index)][2]["query_time_us"]),
                "rgb_anchor_us": int(self._rows[int(index)][2]["rgb_anchor_us"]),
                "query_delay_us": int(self._rows[int(index)][2]["query_delay_us"]),
            }
            for index in indices
        ]

    def history(self, index: int) -> list[dict[str, Any]]:
        """Expose the canonical frozen RGB observation history for upper-stage extraction."""
        from e_jepa_ttc.rgb_port.data import history_observations

        return history_observations(self._rows[int(index)][2])

    def close(self) -> None:
        for dataset in self.datasets.values():
            if dataset.reader is not None:
                dataset.reader.close()


def load_all_train_source(
    descriptor_path: Path,
    teacher_manifest: Path | None = None,
) -> AllTrainRGBSource:
    """Load a path-only source descriptor; no role is silently rewritten."""
    descriptor_path = descriptor_path.resolve(strict=True)
    value = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if value.get("schema") != "final_refit_rgb_source_descriptor_v1":
        raise ValueError("Unknown final-refit RGB source descriptor")
    base = descriptor_path.parent

    def resolve(raw: str) -> Path:
        path = Path(raw)
        return (path if path.is_absolute() else base / path).resolve(strict=True)

    role_manifests = {role: resolve(str(value["role_manifests"][role])) for role in ROLES}
    caches = {role: resolve(str(path)) for role, path in value.get("cache_bindings", {}).items()}
    teacher = (
        teacher_manifest.resolve(strict=True)
        if teacher_manifest is not None
        else resolve(str(value["teacher_manifest"]))
    )
    return AllTrainRGBSource(role_manifests, teacher, cache_bindings=caches)


def materialize_role_cache(
    role_manifest_path: Path,
    cache_root: Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Resume a lossless role cache without modifying its historical manifest directory."""
    import pandas as pd

    from operational.rgb_port.cache_prepare import (
        _existing,
        _init_worker,
        _parity,
        _partitions,
        _task,
    )
    from operational.rgb_port.prepare import atomic_json

    if workers != 1:
        raise ValueError("Final RGB cache materialization requires exactly one sequential worker")
    role_manifest_path = role_manifest_path.resolve(strict=True)
    manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    role = str(manifest.get("role"))
    if manifest.get("status") != "COMPLETE" or role not in ROLES:
        raise ValueError("RGB cache source must be a complete historical P/H/V manifest")
    role_sha = file_sha256(role_manifest_path)
    rows = pd.read_parquet(manifest["rows_path"])
    if len(rows) != int(manifest["population_size"]):
        raise ValueError("RGB role rows and population differ")
    cache_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    work: list[tuple[list[int], list[dict[str, Any]], Path]] = []
    partitions = _partitions(rows)
    for indices, records, relative in partitions:
        path = cache_root / relative
        existing = _existing(path, indices, [str(row["sample_token"]) for row in records], role_sha)
        if existing is None:
            work.append((indices, records, path))
        else:
            completed.append(existing)
    failures: list[dict[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(Path(manifest["eap_root"]).resolve(strict=True)),),
    ) as pool:
        pending: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
        iterator = iter(work)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < 1 and not exhausted:
                allowed, resources = _cache_resource_snapshot(cache_root)
                if not allowed:
                    pause = {"status": "PAUSED_RESOURCE", **resources}
                    atomic_json(
                        cache_root / "RESOURCE_PAUSE.json",
                        pause,
                    )
                    atomic_json(
                        cache_root / "PROGRESS.json",
                        {
                            "status": "PAUSED_RESOURCE",
                            "role": role,
                            "completed_shards": len(completed),
                            "total_shards": len(partitions),
                            "failures": [],
                            "resources": resources,
                        },
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
                    except Exception as error:  # pragma: no cover - worker failures are data/IO
                        failures.append({"path": path, "error": repr(error)})
                        exhausted = True
                        for remaining in pending:
                            remaining.cancel()
                        break
            atomic_json(
                cache_root / "PROGRESS.json",
                {
                    "status": "FAILED" if failures else "RUNNING",
                    "role": role,
                    "completed_shards": len(completed),
                    "total_shards": len(partitions),
                    "failures": failures,
                },
            )
    if failures:
        raise RuntimeError(f"RGB {role} cache preparation had {len(failures)} failed shards")
    completed.sort(key=lambda item: (int(item["T"]), int(item["indices"][0])))
    covered = sorted(index for item in completed for index in map(int, item["indices"]))
    if covered != list(range(len(rows))):
        raise ValueError(f"RGB {role} cache does not cover every role row exactly once")
    parity = _parity(rows, completed, Path(manifest["eap_root"]), cache_root)
    cache_manifest = {
        "schema": "rgb_port_prepared_input_cache_v1",
        "status": "COMPLETE",
        "role": role,
        "role_manifest_path": str(role_manifest_path),
        "role_manifest_sha256": role_sha,
        "population_size": len(rows),
        "shard_size": 32,
        "compression": "numpy_savez_compressed_lossless",
        "stored_pixels": "uint8_RGB",
        "float_conversion": "derived_on_read_exact_uint8_float32_div255",
        "no_frame_padding_or_duplication": True,
        "decode_parity_verified": parity["all_fields_bit_exact"],
        "shards": completed,
        "compressed_bytes": sum(int(item["bytes"]) for item in completed),
    }
    manifest_path = cache_root / "CACHE_MANIFEST.json"
    atomic_json(manifest_path, cache_manifest)
    binding = {
        "schema": "rgb_port_cache_binding_v1",
        "status": "COMPLETE",
        "role": role,
        "role_manifest_sha256": role_sha,
        "cache_manifest_path": str(manifest_path.resolve()),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "equivalence": "bit_exact_decoded_inputs_no_recipe_change",
    }
    binding_path = cache_root / "CACHE_BINDING.json"
    atomic_json(binding_path, binding)
    atomic_json(
        cache_root / "PROGRESS.json",
        {
            "status": "COMPLETE",
            "role": role,
            "completed_shards": len(completed),
            "total_shards": len(completed),
            "failures": [],
        },
    )
    return {**cache_manifest, "binding_path": str(binding_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    result = materialize_role_cache(args.role_manifest, args.cache_root, workers=args.workers)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "role", "population_size", "compressed_bytes", "binding_path")
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AllTrainRGBSource",
    "TeacherShardCache",
    "load_all_train_source",
    "materialize_role_cache",
]
