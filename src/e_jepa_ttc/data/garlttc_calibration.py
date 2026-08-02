"""Calibration resolution for the official Garl-TTC object cache.

The public Garl-TTC annotation rows do not contain ``K_event``.  The official
loader uses one fixed event-camera focal length, while the optional physical
ablation joins the eAP frame table by the stable sequence/member identity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

OFFICIAL_FY = 1694.1323524131867
CalibrationMode = Literal["official_constant_fy", "per_sample_eap_intrinsics"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_python(value: object) -> object:
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        value = as_py()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _intrinsic_fy(value: object) -> float:
    value = _as_python(value)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 9:
        raise ValueError("K_event must contain exactly nine values for a 3x3 matrix.")
    matrix = matrix.reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[1, 1] <= 0.0:
        raise ValueError("K_event must be finite and have positive fy.")
    return float(matrix[1, 1])


@dataclass(frozen=True)
class CalibrationResolution:
    """Resolved focal length and its provenance for one endpoint."""

    fy: float
    calibration_source: str
    join_key: str | None
    source_sha256: str | None


class CalibrationResolver:
    """Resolve Garl visible-height calibration under an explicit mode."""

    def __init__(
        self,
        mode: CalibrationMode = "official_constant_fy",
        *,
        eap_root: str | Path | None = None,
    ) -> None:
        if mode not in {"official_constant_fy", "per_sample_eap_intrinsics"}:
            raise ValueError(f"Unsupported calibration mode: {mode!r}.")
        self.mode = mode
        self.eap_root = Path(eap_root).resolve() if eap_root is not None else None
        self._index: dict[tuple[str, str], object] | None = None
        self._source_sha256: str | None = None
        if mode == "per_sample_eap_intrinsics":
            if self.eap_root is None:
                raise ValueError("per_sample_eap_intrinsics requires eap_root.")
            path = self.eap_root / "data" / "train.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"eAP train parquet not found: {path}")
            frame = pd.read_parquet(path, columns=["sequence_id", "rgb_member_path", "K_event"])
            if frame.duplicated(["sequence_id", "rgb_member_path"]).any():
                raise ValueError("eAP calibration join is not unique.")
            self._index = {
                (str(sequence_id), str(rgb_member_path)): k_event
                for sequence_id, rgb_member_path, k_event in frame.itertuples(
                    index=False, name=None
                )
            }
            self._source_sha256 = _sha256_file(path)

    @property
    def calibration_source(self) -> str:
        """Return the stable provenance label used in manifests."""

        return self.mode

    def resolve(
        self,
        row: Mapping[str, object],
        endpoint_index: int,
    ) -> CalibrationResolution:
        """Resolve ``fy`` for an endpoint without reading a Garl row fallback."""

        if self.mode == "official_constant_fy":
            return CalibrationResolution(
                fy=OFFICIAL_FY,
                calibration_source=self.mode,
                join_key=None,
                source_sha256=None,
            )
        members = row.get("rgb_member_paths")
        as_py = getattr(members, "as_py", None)
        if callable(as_py):
            members = as_py()
        if not isinstance(members, (list, tuple, np.ndarray)):
            raise ValueError("per-sample calibration requires rgb_member_paths.")
        members_list = list(members)
        if endpoint_index < 0 or endpoint_index >= len(members_list):
            raise IndexError(f"Calibration endpoint index out of range: {endpoint_index}.")
        key = (str(row["sequence_id"]), str(_as_python(members_list[endpoint_index])))
        if self._index is None or key not in self._index:
            raise KeyError(f"No unique eAP calibration row for join key {key!r}.")
        return CalibrationResolution(
            fy=_intrinsic_fy(self._index[key]),
            calibration_source=self.mode,
            join_key=f"{key[0]}|{key[1]}",
            source_sha256=self._source_sha256,
        )

    def resolve_pair(
        self,
        row: Mapping[str, object],
        endpoint_indices: tuple[int, int],
    ) -> tuple[CalibrationResolution, CalibrationResolution]:
        """Resolve both endpoint focal lengths and require complete coverage."""

        return self.resolve(row, endpoint_indices[0]), self.resolve(row, endpoint_indices[1])


__all__ = ["CalibrationResolution", "CalibrationResolver", "OFFICIAL_FY"]
