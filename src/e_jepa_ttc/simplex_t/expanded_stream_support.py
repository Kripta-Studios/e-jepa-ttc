"""Check expanded lag masks against actual TRAIN stream endpoints before replay."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np


def verify_expanded_stream_support(
    index: dict[str, np.ndarray],
    manifest: dict,
    *,
    pool: str,
    raw_train_root: Path,
    allowed_sequences: set[str],
    resource_ok: Callable[[], bool],
) -> list[dict]:
    """Recompute every validity bit without decoding targets or event windows.

    Authority and index hashes must already be checked by the caller. D1 uses
    its conservative last timestamp; DENSE retains D0's exclusive last+1 bound.
    Reading two scalars can decompress HDF5 chunks. This is not a full raw hash
    or monotonicity scan, nor proof of online ROI annotation availability.
    Returned size/mtime receipts support lightweight subsequent boundaries.
    """
    if pool not in {"D1", "DENSE_OLD"}:
        raise ValueError("unknown expanded stream pool")
    sequences = index["sequences"]
    count = len(sequences)
    if sequences.ndim != 1 or set(map(str, sequences)) != allowed_sequences:
        raise ValueError("stream support requires the exact authorized sequence set")
    lag, windows, anchors, valid = (
        index[name] for name in ("lag_us", "base_windows_us", "anchor_us", "valid")
    )
    if (
        lag.dtype != np.int64
        or not np.array_equal(lag, np.arange(15, -1, -1, dtype=np.int64) * 50_000)
        or windows.dtype != np.int64
        or windows.shape != (count, 3, 2)
        or anchors.dtype != np.int64
        or anchors.shape != (count,)
        or valid.dtype != bool
        or valid.shape != (count, 16)
        or (windows < 0).any()
        or not (windows[:, :, 1] > windows[:, :, 0]).all()
        or not np.array_equal(windows[:, 2, 1], anchors)
    ):
        raise ValueError("expanded stream support index schema or anchor differs")
    root = raw_train_root.resolve(strict=True)
    receipts = []
    for sequence in sorted(allowed_sequences):
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: expanded stream endpoint audit")
        if Path(sequence).name != sequence or sequence in {".", ".."}:
            raise ValueError("sequence is not a direct TRAIN directory")
        path = (root / sequence / "events.h5").resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("raw event path escapes permitted TRAIN root")
        before = path.stat()
        with h5py.File(path, "r") as stream:
            times = stream["events/t"]
            if (
                not isinstance(times, h5py.Dataset)
                or times.ndim != 1
                or len(times) == 0
                or times.dtype.kind not in {"i", "u"}
            ):
                raise ValueError("nonempty integer timestamp stream required")
            first, last = int(times[0]), int(times[-1])
            event_count = len(times)
        if not 0 <= first <= last < np.iinfo(np.int64).max:
            raise ValueError("raw timestamp endpoint range invalid")
        bounds = [first, last + int(pool == "DENSE_OLD")]
        if manifest["stream_bounds_us"][sequence] != bounds:
            raise ValueError("raw endpoints differ from indexed stream bounds")
        selected = sequences == sequence
        selected_windows = windows[selected]
        lower = selected_windows.min(axis=(1, 2))[:, None] - lag
        upper = selected_windows.max(axis=(1, 2))[:, None] - lag
        recomputed = (lower >= bounds[0]) & (upper <= bounds[1])
        if not np.array_equal(recomputed, valid[selected]) or not recomputed[:, -1].all():
            raise ValueError("expanded lag mask differs from actual stream support")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("raw stream changed during endpoint audit")
        receipts.append(
            {
                "sequence_id": sequence,
                "path": str(path),
                "bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "events": event_count,
                "support_bounds_us": bounds,
                "queries": int(selected.sum()),
                "valid_slots": int(recomputed.sum()),
            }
        )
    return receipts


def verify_stream_receipts(receipts: list[dict]) -> None:
    """Reject source size/mtime changes; do not claim a full raw-media hash."""
    for receipt in receipts:
        current = Path(receipt["path"]).stat()
        if (current.st_size, current.st_mtime_ns) != (receipt["bytes"], receipt["mtime_ns"]):
            raise ValueError("raw stream changed after endpoint audit")
