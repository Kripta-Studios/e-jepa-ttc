"""Label-free generic DSEC-style HDF5 event adapter.

This adapter only discovers and reads event fields.  It intentionally has no
sampling path for flow, disparity, segmentation or TTC labels, which keeps it
safe for the SSL-Pure protocol.
"""

from __future__ import annotations

from pathlib import Path

from e_jepa_ttc.data.evttc import read_events_window
from e_jepa_ttc.data.types import EventBatch


class DSECEventReader:
    """Read event windows from a DSEC-compatible HDF5 file without labels."""

    def __init__(
        self,
        path: str | Path,
        *,
        sequence_id: str,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.sequence_id = sequence_id
        self.width = width
        self.height = height

    def read_window(self, start_us: int, end_us: int) -> EventBatch:
        """Read one causal event interval."""

        return read_events_window(
            self.path,
            t_start_us=start_us,
            t_end_us=end_us,
            sequence_id=self.sequence_id,
            width=self.width,
            height=self.height,
        )


__all__ = ["DSECEventReader"]
