"""Prepare native sensor-only inference inputs in four CPU workers, without neural evaluation."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from .common import Campaign
from .parallel_inputs import worker_init

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def prepare_inference(
    native: dict, raw_root: str, identity: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """Run the unchanged native inference encoder with no training labels or model access."""
    import psutil

    from e_jepa_ttc.efficient_context.garl_input import inference_record

    from . import parallel_inputs

    if psutil.virtual_memory().available < 2 * 1024**3:
        raise InterruptedError("CPU inference preparation requires at least2GiB host available RAM")
    if set(native) != {"sequence_id", "boxes_xyxy", "event_windows_us"}:
        raise ValueError("native inference workers accept sensor/ROI/time fields only")
    root = Path(raw_root)
    path = root / native["sequence_id"] / "events.h5"
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != identity:
        raise ValueError("raw identity changed before native inference preparation")
    if parallel_inputs._pool is None:
        raise RuntimeError("native CPU inference reader was not initialized")
    sensor = inference_record(native, parallel_inputs._pool, root)
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != identity:
        raise ValueError("raw identity changed during native inference preparation")
    return sensor.numpy()


class InferenceWorkers:
    """At most eight original inputs queued per query; the main process owns every GPU forward."""

    def __init__(self, c: Campaign) -> None:
        self.c = c
        self.executor = ProcessPoolExecutor(max_workers=4, initializer=worker_init)

    def submit(self, native: dict, identity: tuple[int, int]) -> Future:
        """Return only the original40-channel FP32 sensor tensor, without labels or writes."""
        self.c.require_resources()
        return self.executor.submit(prepare_inference, native, str(self.c.raw), identity)

    def close(self) -> None:
        """Join CPU-only preparation while preserving all committed per-query fragments."""
        self.executor.shutdown(wait=True, cancel_futures=True)
