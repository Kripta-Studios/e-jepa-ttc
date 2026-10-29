"""Bounded RGB and raw-label access for the public eAP train release.

This module does not interpret dimensions, yaw or object identity across datasets.
The model input is the complete RGB image, never an annotation-derived ROI.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import pandas as pd
from PIL import Image


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    """Accept normalized relative manifest paths, refusing traversal or drives."""
    if not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe manifest path: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe manifest path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"Absolute manifest path: {value!r}")
    return path


def safe_dataset_path(root: Path, value: str) -> Path:
    """Resolve a dataset entry and reject symlink escapes from the root."""
    relative = safe_relative_path(value)
    root = root.resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes root: {value!r}")
    return path


def finite_matrix(
    value: Sequence[Sequence[float]] | np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Convert Parquet's nested arrays without guessing calibration semantics."""
    matrix = np.stack([np.asarray(row, dtype=np.float64) for row in value])
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"Calibration must be finite with shape {shape}")
    return matrix


@dataclass(frozen=True)
class RawRGBSample:
    """A complete image with raw HF labels and its unmodified calibration."""

    sample_token: str
    sequence_id: str
    image: np.ndarray
    annotations: list[dict[str, Any]]
    intrinsic: np.ndarray
    event_from_ego: np.ndarray
    exposure_start_us: int
    exposure_end_us: int
    media: dict[str, Any]
    member_sha256: str


class EAPRGBDataset:
    """Read selected public train tokens with bounded open shards and label caches.

    Images are RGB uint8 HWC arrays owned by the returned sample. Annotation
    records preserve HF values; downstream code must apply an explicit geometry
    adapter. Create one instance per worker rather than sharing tar handles.
    """

    def __init__(
        self,
        root: Path | str,
        sample_tokens: Sequence[str],
        media_path: str = "data/train.parquet",
        max_open_shards: int = 2,
        max_label_sequences: int = 2,
    ) -> None:
        if min(max_open_shards, max_label_sequences) < 1:
            raise ValueError("Cache capacities must be positive")
        self.root = Path(root).resolve()
        self.sample_tokens = tuple(sample_tokens)
        if len(set(self.sample_tokens)) != len(self.sample_tokens):
            raise ValueError("Requested sample tokens must be unique")
        # Filter in Arrow before pandas materializes per-row calibration arrays.
        # Full-release structure is audited separately; this runtime validates
        # the complete requested population, not unrelated manifest rows.
        table = pd.read_parquet(
            safe_dataset_path(self.root, media_path),
            filters=[("sample_token", "in", list(self.sample_tokens))],
        )
        if table.sample_token.isna().any() or table.sample_token.duplicated().any():
            raise ValueError("Media manifest contains missing or duplicate sample tokens")
        selected = cast(pd.DataFrame, table.loc[table.sample_token.isin(self.sample_tokens), :])
        if set(selected.sample_token) != set(self.sample_tokens):
            raise ValueError("At least one requested token is absent from the train manifest")
        if not selected.split.eq("train").all():
            raise ValueError("This runtime admits only public train samples")
        self.rows = {row["sample_token"]: row for row in selected.to_dict("records")}
        self.max_open_shards = max_open_shards
        self.max_label_sequences = max_label_sequences
        self._shards: OrderedDict[str, tarfile.TarFile] = OrderedDict()
        self._labels: OrderedDict[str, dict[str, list[dict[str, Any]]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def _shard(self, relative: str) -> tarfile.TarFile:
        if relative in self._shards:
            self._shards.move_to_end(relative)
            return self._shards[relative]
        path = safe_dataset_path(self.root, relative)
        shard = tarfile.open(path, "r:")
        # Duplicate names would make member identity ambiguous.
        names = shard.getnames()
        if len(names) != len(set(names)):
            shard.close()
            raise ValueError(f"TAR contains duplicate member names: {relative}")
        self._shards[relative] = shard
        if len(self._shards) > self.max_open_shards:
            _, old = self._shards.popitem(last=False)
            old.close()
        return shard

    def _frame_labels(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        relative = str(row["labels_path"])
        if relative not in self._labels:
            requested = [
                token
                for token, media in self.rows.items()
                if media["sequence_id"] == row["sequence_id"]
            ]
            table = pd.read_parquet(
                safe_dataset_path(self.root, relative),
                filters=[("sample_token", "in", requested)],
            )
            if not table.split.eq("train").all():
                raise ValueError("Label file contains non-train rows")
            if not table.sequence_id.eq(row["sequence_id"]).all():
                raise ValueError("Label sequence differs from media sequence")
            if table.duplicated(["sample_token", "instance_id"]).any():
                raise ValueError("Duplicate object ID within one label frame")
            self._labels[relative] = {
                str(token): frame.to_dict("records")
                for token, frame in table.groupby("sample_token", sort=False)
            }
            if len(self._labels) > self.max_label_sequences:
                self._labels.popitem(last=False)
        self._labels.move_to_end(relative)
        # Shallow record copies prevent a downstream adapter from adding fields
        # to our cache. Numeric arrays are copied because training transforms
        # commonly operate in place.
        return [
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in item.items()
            }
            for item in self._labels[relative].get(str(row["sample_token"]), [])
        ]

    def __getitem__(self, index: int) -> RawRGBSample:
        token = self.sample_tokens[index]
        row = self.rows[token]
        start = int(row["rgb_exposure_start_timestamp_us"])
        end = int(row["rgb_exposure_end_timestamp_us"])
        if end < start:
            raise ValueError(f"Reversed exposure interval: {token}")
        member_name = str(safe_relative_path(str(row["rgb_member_path"])))
        member = self._shard(str(row["rgb_shard_path"])).getmember(member_name)
        if not member.isfile() or not 0 < member.size <= 32 * 1024 * 1024:
            raise ValueError(f"RGB member must be a regular image of at most 32 MiB: {token}")
        stream = self._shards[str(row["rgb_shard_path"])].extractfile(member)
        if stream is None:
            raise ValueError(f"Cannot read RGB member: {token}")
        with stream:
            raw = stream.read()
        if len(raw) != member.size:
            raise ValueError(f"Truncated RGB member: {token}")
        with Image.open(io.BytesIO(raw)) as decoded:
            decoded.load()
            image = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unexpected RGB shape: {token}")
        return RawRGBSample(
            sample_token=token,
            sequence_id=str(row["sequence_id"]),
            image=image,
            annotations=self._frame_labels(row),
            intrinsic=finite_matrix(row["K_event"], (3, 3)),
            event_from_ego=finite_matrix(row["T_event_ego"], (4, 4)),
            exposure_start_us=start,
            exposure_end_us=end,
            media=dict(row),
            member_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def close(self) -> None:
        """Close shard handles and clear bounded caches."""
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()
        self._labels.clear()

    def __enter__(self) -> EAPRGBDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def select_pilot_tokens(
    media: pd.DataFrame,
    assignments: Mapping[str, Sequence[str]],
    *,
    seed: int = 7,
    frames_per_sequence: int = 128,
) -> dict[str, Any]:
    """Select two sequences per role without reading labels or image content."""
    if frames_per_sequence < 8:
        raise ValueError("At least eight frames per sequence are needed for overfit16")
    roles = ("train", "selection", "calibration")
    groups = {role: set(assignments[role]) for role in roles}
    for i, role in enumerate(roles):
        if any(groups[role] & groups[other] for other in roles[i + 1 :]):
            raise ValueError("Candidate sequence groups overlap")
    if not media.split.eq("train").all():
        raise ValueError("Pilot may use only the public train manifest")
    if media.sample_token.duplicated().any():
        raise ValueError("Duplicate media tokens")

    result: dict[str, Any] = {"seed": seed, "frames_per_sequence": frames_per_sequence}
    for role in ("train", "selection"):
        ordered = sorted(
            groups[role],
            key=lambda seq: hashlib.sha256(f"pilot:{seed}:{role}:{seq}".encode()).hexdigest(),
        )
        if len(ordered) < 2:
            raise ValueError(f"Need at least two candidate {role} sequences")
        chosen = ordered[:2]
        tokens = []
        for sequence in chosen:
            rows = cast(pd.DataFrame, media.loc[media.sequence_id.eq(sequence), :]).sort_values(
                ["rgb_exposure_start_timestamp_us", "sample_token"]
            )
            if len(rows) < frames_per_sequence:
                raise ValueError(f"Insufficient frames in selected sequence: {sequence}")
            positions = np.linspace(0, len(rows) - 1, frames_per_sequence, dtype=int)
            tokens.extend(rows.iloc[positions].sample_token.tolist())
        result[role] = {"sequences": chosen, "sample_tokens": tokens}
    overfit = []
    for sequence in result["train"]["sequences"]:
        subset = [t for t in result["train"]["sample_tokens"] if t.startswith(sequence + ":")]
        positions = np.linspace(0, len(subset) - 1, 8, dtype=int)
        overfit.extend(subset[i] for i in positions)
    result["overfit16"] = {"sample_tokens": overfit}
    return result
