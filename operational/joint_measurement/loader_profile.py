"""CPU-only breakdown of the immutable P2 sensor reader, without model updates."""

from __future__ import annotations

import io
import statistics
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

from operational.range_transfer.common import read, sha, utc, write

from .common import RUN
from .data import SensorDataset


def main() -> dict:
    """Profile fixed P rows, including ZIP decompression and inner NPZ extraction."""
    data = SensorDataset("P")
    rows = list(range(128))
    path = data.folder / "pack_00000.zip"
    measurements = []
    start = time.perf_counter()
    actual = sha(path)
    hash_s = time.perf_counter() - start
    if actual != read(path.with_suffix(".json"))["sha256"]:
        raise ValueError("Sensor pack hash mismatch")
    with zipfile.ZipFile(path) as archive:
        for index in rows:
            start = time.perf_counter()
            raw = archive.read(f"{index:06d}.npz")
            decoded = time.perf_counter()
            with np.load(io.BytesIO(raw), allow_pickle=False) as inner:
                record = {key: inner[key] for key in inner.files}
            end = time.perf_counter()
            measurements.append(
                {
                    "zip_read_inflate_s": decoded - start,
                    "inner_npz_s": end - decoded,
                    "bytes": len(raw),
                }
            )
            if str(record["token"]) != str(data.rows.iloc[index].sample_token):
                raise ValueError("Native token mismatch")

    # Private archive handles prevent lock contention and shared cursor races.
    def load(index: int) -> tuple[np.ndarray, np.ndarray, str]:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read(f"{index:06d}.npz")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            return archive["events"], archive["delta"], str(archive["token"])

    timings = {}
    for workers in (1, 4, 8):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (_, _, token) in zip(rows, pool.map(load, rows), strict=True):
                if token != str(data.rows.iloc[index].sample_token):
                    raise ValueError("Parallel token mismatch")
        timings[str(workers)] = (time.perf_counter() - start) / (len(rows) / 4)
    original = []
    for begin in range(0, 128, 4):
        start = time.perf_counter()
        data.batch(list(range(begin, begin + 4)))
        original.append(time.perf_counter() - start)
    data.close()
    result = {
        "utc": utc(),
        "role": "P",
        "indices": rows,
        "scope": "CPU-only profiling while graph evaluation is live; OS cache uncontrolled",
        "pack_bytes": path.stat().st_size,
        "pack_sha256": actual,
        "hash_s": hash_s,
        "median_per_batch4_s": {
            key: 4 * statistics.median(item[key] for item in measurements)
            for key in ("zip_read_inflate_s", "inner_npz_s")
        },
        "parallel_selective_seconds_per_batch4": timings,
        "original_median_batch_s": statistics.median(original),
        "rss_bytes": psutil.Process().memory_info().rss,
        "measurements": measurements,
        "optimizer_updates": 0,
    }
    write(RUN / "runtime/loader_profile_v1/RESULT.json", result)
    print({key: value for key, value in result.items() if key not in {"measurements", "indices"}})
    return result


if __name__ == "__main__":
    main()
