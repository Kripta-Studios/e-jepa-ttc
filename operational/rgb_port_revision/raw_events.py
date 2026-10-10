"""Actual eAP HDF5-to-native-event preprocessing for batch-one route timings."""

# ruff: noqa: ANN401 -- adapters bridge the retained dynamic source protocol

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.data.garlttc_calibration import CalibrationResolver
from e_jepa_ttc.data.garlttc_lhr_cache import _materialize_row
from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader
from operational.rgb_port.accounting import sha256_file
from operational.train40_system.prepare import configuration


class RawEventSource:
    """Recompute voxels each call, without teacher or prepared-voxel cache reads."""

    def __init__(self, prepared: Any, eap_root: Path) -> None:
        self.root = eap_root
        path = prepared._inputs.output / "TRAIN40_ROWS.parquet"
        self.rows = pd.read_parquet(path).to_dict(orient="records")
        self.indices = list(prepared._event_indices)
        self.readers: dict[Path, Any] = {}
        self.config = configuration()
        self.calibration = CalibrationResolver()
        self.identity = {
            "rows_sha256": sha256_file(path),
            "eap_root": str(eap_root),
            "prepared_voxels_read": False,
            "teacher_read": False,
            "raw_reader_cache": "at_most_two_CachedEventReader_handles",
        }

    def batch(self, indices: list[int], modality: str) -> Any:
        if modality != "event" or not indices:
            raise ValueError("Nonempty event indices required")
        records = []
        for index in indices:
            row = self.rows[self.indices[index]]
            path = self.root / str(row["events_path"])
            if path not in self.readers:
                while len(self.readers) >= 2:
                    self.readers.pop(next(iter(self.readers))).close()
                self.readers[path] = CachedEventReader(path)
            records.append(
                _materialize_row(
                    row,
                    eap_root=self.root,
                    config=self.config,
                    event_readers=self.readers,
                    rgb_reader=None,
                    mask_reader=None,
                    first_track_timestamp_us=None,
                    calibration=self.calibration,
                )
            )
        return SimpleNamespace(
            events=torch.from_numpy(np.stack([row["event_v4_common_roi"] for row in records])),
            delta_t_s=torch.tensor([row["garl_delta_t_s"] for row in records], dtype=torch.float32),
        )

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()
