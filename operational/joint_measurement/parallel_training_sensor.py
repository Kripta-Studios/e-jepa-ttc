"""Lossless parallel random-row decoding for the existing sampled training batches."""

from __future__ import annotations

import io
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.range_transfer.common import read, sha

from .fast_runtime import PersistentSensors
from .prefetched_sensor import stored_array


class ParallelTrainingSensors(PersistentSensors):
    """Preserve the sampler/prefetch/collator; decode independent ZIP rows concurrently.

    Cache state is owned by the single prefetch task. Decoder workers only read
    admitted ZipFile members; ZipFile serializes file seeks, while zlib decoding
    occurs outside that file lock. No detached model features are stored.
    """

    def __init__(self, *, limit_bytes: int = 8 * 1024**3, workers: int = 8) -> None:
        if not 1 <= workers <= 8:
            raise ValueError("One to eight random-row decoder workers required")
        super().__init__(limit_bytes=limit_bytes)
        self.workers = workers
        self.decoder = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p2-random")
        self.tokens = self.rows.sample_token.astype(str).tolist()
        self.current_rows: dict[int, dict[str, np.ndarray]] | None = None
        self.duplicate_draws = 0
        self.decoder_closed = False
        self.load_lock = threading.Lock()

    def _verify_pack(self, number: int) -> None:
        path = self.folder / f"pack_{number:05d}.zip"
        receipt = read(path.with_suffix(".json"))
        begin, end = number * 256, min((number + 1) * 256, self.population)
        if receipt["start"] != begin or receipt["stop"] != end:
            raise ValueError("Random training pack population changed")
        if sha(path) != receipt["sha256"]:
            raise ValueError("Random training sensor pack failed SHA admission")

    def _decode(self, index: int) -> dict[str, np.ndarray]:
        # The complete outer member read checks CRC, including all supervision arrays.
        raw = self.packs[index // 256].read(f"{index:06d}.npz")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(not name.endswith(".npy") for name in names):
                raise ValueError("Duplicate or unknown inner training member")
            result = {name[:-4]: stored_array(raw, archive, name[:-4]) for name in names}
        if str(result["token"]) != self.tokens[index]:
            raise ValueError("Random training token mismatch")
        pixels, clock = result["events"], result["delta"]
        if pixels.dtype != np.float32 or pixels.shape != (2, 12, 128, 128):
            raise ValueError("Native random training pixels changed")
        if clock.dtype != np.float32 or clock.size != 1 or not np.isfinite(clock).all():
            raise ValueError("Native random training clock changed")
        if not np.isfinite(pixels).all() or float(clock.reshape(-1)[0]) <= 0:
            raise ValueError("Invalid native random training input")
        return result

    @staticmethod
    def _size(row: dict[str, np.ndarray]) -> int:
        # Include a conservative allowance for the small stored NPY/ZIP headers.
        return sum(value.nbytes for value in row.values()) + 8192

    def _collect(self, indices: list[int]) -> dict[int, dict[str, np.ndarray]]:
        if not indices or any(
            not isinstance(index, int) or not 0 <= index < self.population for index in indices
        ):
            raise ValueError("Invalid random training row index")
        unique = list(dict.fromkeys(indices))
        self.duplicate_draws += len(indices) - len(unique)
        numbers = list(dict.fromkeys(index // 256 for index in unique))
        fingerprints = {number: self._fingerprint(number) for number in numbers}
        for number, value in fingerprints.items():
            if number in self.fingerprints and self.fingerprints[number] != value:
                raise ValueError("Previously admitted training pack or receipt changed")
        fresh = [number for number in numbers if number not in self.verified]
        list(self.decoder.map(self._verify_pack, fresh))
        for number in fresh:
            if self._fingerprint(number) != fingerprints[number]:
                raise ValueError("Training pack changed during SHA admission")
            path = self.folder / f"pack_{number:05d}.zip"
            archive = zipfile.ZipFile(path)
            expected = [
                f"{index:06d}.npz"
                for index in range(number * 256, min((number + 1) * 256, self.population))
            ]
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
                archive.close()
                raise ValueError("Duplicate, missing or extra outer training member")
            self.packs[number] = archive
            self.verified.add(number)
            self.fingerprints[number] = fingerprints[number]
        result, missing = {}, []
        for index in unique:
            if index in self.rows_cached:
                self.hits += 1
                self.rows_cached.move_to_end(index)
                result[index] = self.rows_cached[index]
            else:
                self.misses += 1
                missing.append(index)
        decoded = list(self.decoder.map(self._decode, missing))
        for index, row in zip(missing, decoded, strict=True):
            result[index] = row
            size = self._size(row)
            available = psutil.virtual_memory().available
            while self.rows_cached and (
                self.retained_bytes + size > self.limit_bytes or available < 4 * 1024**3
            ):
                _, old = self.rows_cached.popitem(last=False)
                self.retained_bytes -= self._size(old)
                available = psutil.virtual_memory().available
            if size <= self.limit_bytes and available >= 4 * 1024**3:
                self.rows_cached[index] = row
                self.retained_bytes += size
                self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        if any(self._fingerprint(number) != value for number, value in fingerprints.items()):
            raise ValueError("Training pack changed while consuming batch")
        return result

    def _row(self, index: int) -> dict[str, np.ndarray]:
        if self.current_rows is None or index not in self.current_rows:
            raise RuntimeError("Random rows must be admitted as a complete sampled batch")
        return self.current_rows[index]

    def load(self, ids: list[int]) -> ObjectEventV4Batch:
        """Return the original collator output, including duplicate draws and all targets."""
        if self.decoder_closed or not self.load_lock.acquire(blocking=False):
            raise RuntimeError("Random decoder is closed or called concurrently")
        try:
            self.current_rows = self._collect(ids)
            return super().load(ids)
        finally:
            self.current_rows = None
            self.load_lock.release()

    def shutdown(self) -> None:
        """Join the inherited prefetch before closing worker and archive resources."""
        self.close()
        self.decoder.shutdown(wait=True, cancel_futures=True)
        self.decoder_closed = True
        super().shutdown()

    def counters(self) -> dict:
        return {
            **super().counters(),
            "decoder_workers": self.workers,
            "duplicate_draws": self.duplicate_draws,
            "cache_counter_scope": "Unique rows; byte bound includes8192 header allowance per row",
        }
