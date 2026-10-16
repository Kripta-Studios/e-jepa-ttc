"""Bounded lossless P2 inference prefetch, independent of frozen training readers."""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import read

from .data import SensorDataset


@dataclass(frozen=True)
class SensorInputs:
    """Inference-only tensors; targets and teacher arrays are never decoded."""

    events: torch.Tensor
    delta_t_s: torch.Tensor


def decode_pack(path: Path, tokens: list[str], workers: int = 8) -> tuple[SensorInputs, dict]:
    """Hash compressed bytes once, decode disjoint stripes with private ZIP cursors."""
    start = time.perf_counter()
    receipt = read(path.with_suffix(".json"))
    begin, stop = receipt["start"], receipt["stop"]
    if (
        stop - begin != len(tokens)
        or not 0 < len(tokens) <= 256
        or begin != int(path.stem.split("_")[-1]) * 256
        or not 1 <= workers <= 8
    ):
        raise ValueError("Pack population mismatch")
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != receipt["sha256"]:
        raise ValueError("Sensor pack changed before prefetch")
    loaded = time.perf_counter()
    events = np.empty((len(tokens), 2, 12, 128, 128), dtype=np.float32)
    clocks = np.empty(len(tokens), dtype=np.float32)

    def stripe(offset: int) -> None:
        with zipfile.ZipFile(io.BytesIO(compressed)) as archive:
            expected = [f"{i:06d}.npz" for i in range(begin, stop)]
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
                raise ValueError("Duplicate, missing or extra sensor member")
            for position in range(offset, len(tokens), workers):
                raw = archive.read(expected[position])  # ZIP validates each complete member CRC.
                with np.load(io.BytesIO(raw), allow_pickle=False) as inner:
                    token, pixels, dt = str(inner["token"]), inner["events"], inner["delta"]
                if token != tokens[position]:
                    raise ValueError("Sensor row token mismatch")
                if pixels.dtype != np.float32 or pixels.shape != (2, 12, 128, 128):
                    raise ValueError("P2 FP32 pixel contract changed")
                if dt.dtype != np.float32 or dt.size != 1 or not np.isfinite(dt).all():
                    raise ValueError("Native FP32 scalar clock contract changed")
                if not np.isfinite(pixels).all() or float(dt.reshape(-1)[0]) <= 0:
                    raise ValueError("Nonfinite pixels or invalid native clock")
                events[position] = pixels
                clocks[position] = dt.reshape(-1)[0]

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p2-decode") as pool:
        list(pool.map(stripe, range(workers)))
    return SensorInputs(torch.from_numpy(events), torch.from_numpy(clocks)), {
        "pack": str(path),
        "sha256": receipt["sha256"],
        "rows": len(tokens),
        "read_hash_s": loaded - start,
        "decode_s": time.perf_counter() - loaded,
        "compressed_bytes": len(compressed),
        "decoded_bytes": events.nbytes + clocks.nbytes,
    }


class PrefetchedSensor:
    """Serve exact consecutive B4/tail2 inputs with at most current plus next pack."""

    def __init__(self, role: str, *, workers: int = 8, ahead: bool = True) -> None:
        source = SensorDataset(role)
        self.rows, self.folder, self.population = source.rows, source.folder, source.population
        source.close()
        self.role, self.workers, self.ahead = role, workers, ahead
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p2-prefetch")
        self.pending: tuple[int, Future] | None = None
        self.current_number = -1
        self.current: SensorInputs | None = None
        self.timings: list[dict] = []
        self.closed = False

    def _submit(self, number: int) -> Future:
        begin, stop = number * 256, min((number + 1) * 256, self.population)
        return self.pool.submit(
            decode_pack,
            self.folder / f"pack_{number:05d}.zip",
            self.rows.iloc[begin:stop].sample_token.astype(str).tolist(),
            self.workers,
        )

    def batch(self, indices: list[int]) -> SensorInputs:
        """Preserve population, sequence order, original batch size and native clocks."""
        if self.closed:
            raise ValueError("Reader is closed")
        if (
            not indices
            or indices != list(range(indices[0], indices[0] + len(indices)))
            or len(indices) not in {2, 4}
            or indices[0] % 4 != 0
            or len(indices) != min(4, self.population - indices[0])
            or indices[0] < 0
            or indices[-1] >= self.population
            or indices[0] // 256 != indices[-1] // 256
        ):
            raise ValueError("Inference requires an original consecutive B4/tail2 batch")
        number = indices[0] // 256
        if number != self.current_number:
            if self.pending is not None and self.pending[0] != number:
                # A resume can skip durable packs. Drain the one bounded future before replacement.
                self.pending[1].result()
                self.pending = None
            if self.pending is None:
                self.pending = (number, self._submit(number))
            self.current = None
            self.current, timing = self.pending[1].result()
            self.timings.append(timing)
            self.current_number, self.pending = number, None
            if self.ahead and (number + 1) * 256 < self.population:
                self.pending = (number + 1, self._submit(number + 1))
        assert self.current is not None
        start, stop = indices[0] % 256, indices[-1] % 256 + 1
        # Own only B4 storage: a caller retaining its previous batch must not pin an old 400MB pack.
        return SensorInputs(
            self.current.events[start:stop].clone(), self.current.delta_t_s[start:stop].clone()
        )

    def close(self) -> None:
        """Join the bounded producer and release its buffers without touching source packs."""
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.current, self.pending, self.closed = None, None, True
