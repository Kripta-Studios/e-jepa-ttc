"""Bounded observational summaries of existing admission samples, without new probes."""

from __future__ import annotations

import time
from collections.abc import Callable


class ResourceObservations:
    """Record sampled extrema, not continuous peaks or an optimizer-work ledger.

    No admission policy, RNG, tensor, model or checkpoint state is modified.
    The caller supplies the same samples and decisions it already uses. The
    elapsed interval includes admission, loading, fits and publication, not just
    optimizer time. A hard process termination can prevent the final receipt.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = clock()
        self.samples = 0
        self.denied = 0
        self.missing_counter_samples = 0
        self.max_rss: int | None = None
        self.min_ram: int | None = None
        self.min_disk: int | None = None

    def observe(self, snapshot: dict, *, allowed: bool) -> None:
        """Consume one existing decision; retain no growing per-sample list."""
        self.samples += 1
        self.denied += not allowed
        rss = snapshot.get("process_tree_rss_bytes")
        ram = snapshot.get("host_available_bytes")
        disks = snapshot.get("written_volume_free_bytes")
        if (
            type(rss) is not int
            or rss < 0
            or type(ram) is not int
            or ram < 0
            or not isinstance(disks, list)
            or not disks
            or any(type(value) is not int or value < 0 for value in disks)
        ):
            self.missing_counter_samples += 1
            return
        disk = min(disks)
        self.max_rss = rss if self.max_rss is None else max(self.max_rss, rss)
        self.min_ram = ram if self.min_ram is None else min(self.min_ram, ram)
        self.min_disk = disk if self.min_disk is None else min(self.min_disk, disk)

    def summary(self) -> dict:
        """Snapshot elapsed time and extrema without converting them to work counts."""
        return dict(
            schema="simplex_t_resource_observations_v1",
            elapsed_seconds=self._clock() - self._started,
            admission_samples=self.samples,
            denied_admission_samples=self.denied,
            missing_counter_samples=self.missing_counter_samples,
            sampled_process_tree_rss_max_bytes=self.max_rss,
            sampled_host_available_min_bytes=self.min_ram,
            sampled_written_volume_free_min_bytes=self.min_disk,
            continuous_peak_measurement=False,
            vram_measurement=None,
            timing_scope="INVOCATION_ADMISSION_LOADING_FITS_AND_PUBLICATION_NOT_OPTIMIZER_ONLY",
            hard_termination_receipt_guaranteed=False,
            observer_optimizer_updates=0,
        )
