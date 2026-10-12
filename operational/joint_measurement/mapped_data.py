"""Lossless uncompressed P packs, prepared separately from the active trainer."""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from .common import ROOT, RUN, require_resources
from .data import SensorDataset, cache_root

PACK_SIZE = 256


def mapped_root() -> Path:
    """Keep the additional sensor cache outside both Git and the results bundle."""
    return cache_root("P").parent.parent / "sensor_uncompressed/P"


def fingerprint(path: Path) -> tuple[int, int]:
    """Guard admitted files against replacement between durable boundaries."""
    status = path.stat()
    return status.st_size, status.st_mtime_ns


def write_pack(folder: Path, records: list[dict[str, np.ndarray]], binding: dict) -> dict:
    """Write and reread every field; publish a receipt only after exact parity."""
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "COMPLETE.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["binding"] != binding:
            raise ValueError("Mapped pack binding changed")
        verify_pack(folder, receipt)
        return receipt
    if not records or any(set(row) != set(records[0]) for row in records):
        raise ValueError("Empty or inconsistent source fields")
    fields = {}
    for key in sorted(records[0]):
        if not key.isidentifier():
            raise ValueError("Unsafe array field")
        values = np.stack([row[key] for row in records])
        if values.dtype.hasobject:
            raise ValueError("Object arrays are not sensor-cache inputs")
        path = folder / f"{key}.npy"
        pending = path.with_suffix(".pending")
        with pending.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        with pending.open("rb") as stream:
            replay = np.load(stream, allow_pickle=False)
        if replay.dtype != values.dtype or not np.array_equal(replay, values):
            raise ValueError("Mapped cache serialization changed sensor values")
        os.replace(pending, path)
        fields[key] = {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "shape": list(values.shape),
            "dtype": values.dtype.str,
        }
        del values, replay
    receipt = {
        "schema": "v18_uncompressed_pack_v1",
        "binding": binding,
        "rows": len(records),
        "fields": fields,
        "all_fields_exact": True,
    }
    freeze(receipt_path, receipt)
    return receipt


def verify_pack(folder: Path, receipt: dict) -> dict[str, tuple[int, int]]:
    """Hash every admitted field once and capture its stable filesystem stamp."""
    stamps = {}
    for key, field in receipt["fields"].items():
        if not key.isidentifier():
            raise ValueError("Unsafe mapped field")
        path = folder / f"{key}.npy"
        before = fingerprint(path)
        if before[0] != field["bytes"] or sha(path) != field["sha256"]:
            raise ValueError("Mapped sensor field changed")
        after = fingerprint(path)
        if after != before:
            raise ValueError("Mapped sensor field mutated during admission")
        stamps[key] = after
    return stamps


class MappedRows:
    """Bounded read-only mappings with cached admission and mutation guards."""

    def __init__(self, folder: Path, population: int, *, max_open_packs: int = 8) -> None:
        if population < 1 or max_open_packs < 1:
            raise ValueError("Invalid population or mapping limit")
        self.folder = folder
        self.population = population
        self.max_open_packs = max_open_packs
        self.verified: dict[int, dict] = {}
        self.maps: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def admit(self, number: int) -> None:
        """Admit original bytes; repeat admission only when explicitly requested."""
        folder = self.folder / f"pack_{number:05d}"
        receipt = read(folder / "COMPLETE.json")
        expected_rows = min(PACK_SIZE, self.population - number * PACK_SIZE)
        if expected_rows <= 0 or receipt["rows"] != expected_rows:
            raise ValueError("Mapped pack population mismatch")
        self.verified[number] = {
            "receipt": receipt,
            "receipt_sha256": sha(folder / "COMPLETE.json"),
            "stamps": verify_pack(folder, receipt),
        }

    def admit_all(self) -> None:
        """Perform the full byte admission outside any GPU scope."""
        for number in range((self.population + PACK_SIZE - 1) // PACK_SIZE):
            self.admit(number)

    @staticmethod
    def _close(values: dict[str, Any]) -> None:
        for array in values.values():
            array._mmap.close()

    def row(self, index: int) -> dict[str, np.ndarray]:
        """Return owned arrays so eviction cannot invalidate a caller's batch."""
        if not 0 <= index < self.population:
            raise IndexError("Mapped row outside admitted population")
        number, offset = divmod(index, PACK_SIZE)
        if number not in self.verified:
            self.admit(number)
        folder = self.folder / f"pack_{number:05d}"
        admission = self.verified[number]
        for key, stamp in admission["stamps"].items():
            if fingerprint(folder / f"{key}.npy") != stamp:
                raise ValueError("Mapped field changed after admission")
        if number not in self.maps:
            if sha(folder / "COMPLETE.json") != admission["receipt_sha256"]:
                raise ValueError("Mapped receipt changed after admission")
            if len(self.maps) >= self.max_open_packs:
                self._close(self.maps.popitem(last=False)[1])
            arrays = {}
            try:
                for key, field in admission["receipt"]["fields"].items():
                    array = np.load(folder / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                    arrays[key] = array
                    if list(array.shape) != field["shape"] or array.dtype.str != field["dtype"]:
                        raise ValueError("Mapped array metadata changed")
            except BaseException:
                self._close(arrays)
                raise
            self.maps[number] = arrays
        self.maps.move_to_end(number)
        return {key: np.array(array[offset], copy=True) for key, array in self.maps[number].items()}

    def close(self) -> None:
        """Release mappings while retaining verified immutable file identities."""
        for values in self.maps.values():
            self._close(values)
        self.maps.clear()


def prepare() -> dict:
    """Build P pack by pack without updating a model or changing legacy data."""
    torch.set_num_threads(1)
    dataset = SensorDataset("P")
    folder = mapped_root()
    source_complete = read(cache_root("P") / "COMPLETE.json")
    binding = {
        "schema": "v18_exact_uncompressed_p_v1",
        "role": "P",
        "population": dataset.population,
        "pack_size": PACK_SIZE,
        "source_complete_sha256": sha(cache_root("P") / "COMPLETE.json"),
        "source_pack_receipts_hash": source_complete["pack_receipts_hash"],
        "source_code_sha256": sha(Path(__file__)),
        "original_loader_sha256": sha(ROOT / "operational/joint_measurement/data.py"),
        "storage_only": True,
        "optimizer_updates": 0,
    }
    freeze(folder / "BINDING.json", binding)
    directory = RUN / "diagnostics/UNCOMPRESSED_P_CACHE"
    freeze(directory / "REGISTRATION.json", binding)
    receipts = []
    try:
        # Reserve 1.65 MB per remaining row plus 250 MB for metadata and padding.
        # Completed packs are validated on resume; partial pack files are replaced individually.
        for number, start in enumerate(range(0, dataset.population, PACK_SIZE)):
            target = folder / f"pack_{number:05d}"
            source_receipt = cache_root("P") / f"pack_{number:05d}.json"
            pack_binding = {
                "campaign": digest(binding),
                "source_receipt_sha256": sha(source_receipt),
                "start": start,
                "stop": min(start + PACK_SIZE, dataset.population),
            }
            if (target / "COMPLETE.json").exists():
                receipt = read(target / "COMPLETE.json")
                if receipt["binding"] != pack_binding:
                    raise ValueError("Resumed mapped pack source changed")
                verify_pack(target, receipt)
            else:
                require_resources(
                    disk_reserve=(dataset.population - start) * 1_650_000 + 250_000_000
                )
                records = [dataset._row(index) for index in range(start, pack_binding["stop"])]
                receipt = write_pack(target, records, pack_binding)
                del records
            receipts.append(receipt)
            write(
                directory / "PROGRESS.json",
                {
                    "utc": utc(),
                    "completed_rows": pack_binding["stop"],
                    "population": dataset.population,
                    "packs": len(receipts),
                },
            )
            print(f"UNCOMPRESSED P {pack_binding['stop']}/{dataset.population}", flush=True)
        result = {
            "status": "COMPLETE",
            "rows": dataset.population,
            "packs": len(receipts),
            "pack_receipts_hash": digest(receipts),
            "binding": digest(binding),
            "bytes": sum(f["bytes"] for r in receipts for f in r["fields"].values()),
            "all_fields_exact": True,
            "optimizer_updates": 0,
            "production_changed": False,
        }
        freeze(folder / "COMPLETE.json", result)
        write(directory / "COMPLETE.json", result)
        if torch.cuda.is_initialized():
            raise RuntimeError("Cache preparation unexpectedly initialized CUDA")
        return result
    finally:
        dataset.close()


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(prepare())
