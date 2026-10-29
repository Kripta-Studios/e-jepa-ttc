"""Real TRAIN parity for the new H8 workers, without using labels or updating weights."""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from operational.efficient_context.common import atomic_json, digest
from operational.train40_system.contracts import read
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.history_features import prepare, worker_init


def run(output: Path, raw_root: Path) -> None:
    """Compare union mapping to the original encoder and the retained full-precision cache."""
    import h5py

    from e_jepa_ttc.data.eap import _require_hdf5plugin
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
    from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union

    worker_init()
    _require_hdf5plugin()
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        index = {key: stored[key] for key in ("windows_us", "square_xyxy", "sequences")}
    cache = Path(read(output / "PREPARE_FREEZE.json")["cache_root"])
    jobs = []
    for row in (0, 31):
        sequence = str(index["sequences"][row])
        raw = raw_root / "data/train" / sequence / "events.h5"
        with h5py.File(raw, "r") as handle:
            timestamps = handle["events/t"]
            if not isinstance(timestamps, h5py.Dataset):
                raise ValueError("Raw timestamps must be a dataset")
            start = int(timestamps[0])
        windows = index["windows_us"][row]
        jobs.append(
            {
                "path": str(raw),
                "sequence": sequence,
                "windows": windows,
                "square": index["square_xyxy"][row],
                "valid": windows.min() - np.arange(7, -1, -1) * 50000 >= start,
            }
        )
    rows = []
    pool = ReaderPool()
    try:
        with ProcessPoolExecutor(max_workers=4, initializer=worker_init) as executor:
            futures = [executor.submit(prepare, job) for job in jobs]
            for row, job, future in zip((0, 31), jobs, futures, strict=True):
                actual = future.result()
                reference = encode_context_union(
                    pool.get(job["path"]),
                    job["windows"],
                    np.arange(15, -1, -1, dtype=np.int64) * 50000,
                    np.concatenate((np.zeros(8, bool), job["valid"])),
                    tuple(job["square"]),
                    sequence_id=job["sequence"],
                    roi_size=128,
                    event_pixel_diff=5,
                ).numpy()
                with np.load(cache / "shard_00000.npz", allow_pickle=False) as stored:
                    current = stored["events"][row]
                if not np.array_equal(actual, reference) or not np.array_equal(actual[-1], current):
                    raise ValueError("Real H8 raw/current-cache parity failed")
                rows.append(
                    {
                        "TRAIN_ordinal": row,
                        "all_16_slots_exact": True,
                        "current_cache_exact": True,
                        "sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                    }
                )
        atomic_json(
            output / "REAL_HISTORY_INPUT_ADMISSION.json",
            {
                "status": "PASSED",
                "source_sha256": digest(Path(__file__)),
                "worker_source_sha256": digest(Path(__file__).with_name("history_features.py")),
                "observations": rows,
                "optimizer_updates": 0,
                "targets_read": False,
                "reduced_unpadded_GPU_dispatch_allowed": False,
            },
        )
    finally:
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.raw_root.resolve())
