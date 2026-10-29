"""Fail-closed indexed access to RGB members in uncompressed eAP TAR shards."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from e_jepa_ttc.detection.eap_rgb_data import (
    EAPRGBDataset,
    RawRGBSample,
    finite_matrix,
    safe_dataset_path,
    safe_relative_path,
    sha256_file,
)

MAX_RGB_MEMBER_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class TarSourceIdentity:
    """Filesystem identity bound when a TAR index is constructed."""

    device: int
    inode: int
    size: int
    modified_ns: int

    @classmethod
    def from_path(cls, path: Path) -> TarSourceIdentity:
        stat = path.stat()
        if not path.is_file():
            raise ValueError(f"TAR source is not a regular file: {path}")
        return cls(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


@dataclass(frozen=True)
class IndexedTarMember:
    """Exact byte range and admitted digest for one token-owned member."""

    token: str
    shard_relative: str
    name: str
    offset_data: int
    size: int
    sha256: str


def _digest(value: str, field: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


class IndexedTarRGBReader:
    """Index selected TAR shards once and seek directly to admitted RGB bytes."""

    def __init__(
        self,
        root: Path | str,
        ownership: Mapping[str, tuple[str, str, str]],
        shard_sha256: Mapping[str, str],
        *,
        max_open_handles: int = 2,
        max_member_bytes: int = MAX_RGB_MEMBER_BYTES,
    ) -> None:
        if max_open_handles < 1 or max_member_bytes < 1:
            raise ValueError("Reader limits must be positive")
        if not ownership:
            raise ValueError("At least one admitted token is required")
        self.root = Path(root).resolve()
        self.max_open_handles = max_open_handles
        self.max_member_bytes = max_member_bytes
        self._handles: OrderedDict[str, io.BufferedReader] = OrderedDict()
        self._identities: dict[str, TarSourceIdentity] = {}
        self._paths: dict[str, Path] = {}
        self._members: dict[str, IndexedTarMember] = {}

        requested: dict[str, dict[str, tuple[str, str]]] = {}
        owners: dict[tuple[str, str], str] = {}
        for token, (shard_raw, member_raw, member_digest) in ownership.items():
            if not token:
                raise ValueError("Token cannot be empty")
            shard = str(safe_relative_path(shard_raw))
            member = str(safe_relative_path(member_raw))
            key = (shard, member)
            if key in owners:
                raise ValueError(
                    f"TAR member has multiple token owners: {owners[key]!r}, {token!r}"
                )
            owners[key] = token
            requested.setdefault(shard, {})[member] = (
                token,
                _digest(member_digest, f"member SHA for {token}"),
            )

        if set(requested) != set(shard_sha256):
            raise ValueError("Shard digest keys must exactly match selected TAR shards")
        for shard, wanted in requested.items():
            self._index_shard(shard, wanted, _digest(shard_sha256[shard], "shard SHA"))
        if set(self._members) != set(ownership):
            raise ValueError("Indexed token ownership is incomplete")

    @property
    def open_handle_count(self) -> int:
        """Return current descriptor count for bounded-resource evidence."""
        return len(self._handles)

    def _index_shard(
        self,
        relative: str,
        wanted: Mapping[str, tuple[str, str]],
        expected_sha256: str,
    ) -> None:
        path = safe_dataset_path(self.root, relative)
        before = TarSourceIdentity.from_path(path)
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"TAR source SHA-256 differs from admission: {relative}")
        found: set[str] = set()
        names: set[str] = set()
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                name = str(safe_relative_path(member.name))
                if name in names:
                    raise ValueError(f"TAR contains duplicate member names: {relative}")
                names.add(name)
                if not member.isfile() or not 0 <= member.size <= self.max_member_bytes:
                    raise ValueError(
                        f"TAR member must be regular and within size limit: {relative}:{name}"
                    )
                if member.offset_data < 0 or member.offset_data + member.size > before.size:
                    raise ValueError(f"TAR member byte range is invalid: {relative}:{name}")
                if name in wanted:
                    if member.size == 0:
                        raise ValueError(f"Admitted TAR member is empty: {relative}:{name}")
                    token, digest = wanted[name]
                    found.add(name)
                    self._members[token] = IndexedTarMember(
                        token, relative, name, member.offset_data, member.size, digest
                    )
        if found != set(wanted):
            missing = sorted(set(wanted) - found)
            raise ValueError(f"Admitted TAR members are absent from {relative}: {missing}")
        after = TarSourceIdentity.from_path(path)
        if after != before:
            raise ValueError(f"TAR source changed while indexing: {relative}")
        self._paths[relative] = path
        self._identities[relative] = before

    def _assert_source(self, relative: str) -> None:
        if TarSourceIdentity.from_path(self._paths[relative]) != self._identities[relative]:
            self.close()
            raise ValueError(f"TAR source identity changed after indexing: {relative}")

    def _handle(self, relative: str) -> io.BufferedReader:
        self._assert_source(relative)
        if relative in self._handles:
            self._handles.move_to_end(relative)
            return self._handles[relative]
        handle = self._paths[relative].open("rb")
        self._handles[relative] = handle
        if len(self._handles) > self.max_open_handles:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def read(self, token: str) -> bytes:
        """Read and re-hash exactly one admitted token's RGB payload."""
        try:
            member = self._members[token]
        except KeyError as error:
            raise KeyError(f"Token is not owned by this indexed reader: {token}") from error
        handle = self._handle(member.shard_relative)
        handle.seek(member.offset_data, os.SEEK_SET)
        raw = handle.read(member.size)
        self._assert_source(member.shard_relative)
        if len(raw) != member.size:
            raise ValueError(f"Truncated indexed RGB member: {token}")
        if hashlib.sha256(raw).hexdigest() != member.sha256:
            raise ValueError(f"RGB member SHA-256 differs from admission: {token}")
        return raw

    def close(self) -> None:
        """Close every bounded file handle; the immutable index remains reusable."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


class IndexedEAPRGBDataset(EAPRGBDataset):
    """EAP dataset with identical samples and indexed, receipt-bound RGB access."""

    def __init__(
        self,
        root: Path | str,
        sample_tokens: Sequence[str],
        *,
        member_sha256_by_token: Mapping[str, str],
        shard_sha256_by_path: Mapping[str, str],
        media_path: str = "data/train.parquet",
        max_open_shards: int = 2,
        max_label_sequences: int = 2,
    ) -> None:
        super().__init__(
            root,
            sample_tokens,
            media_path=media_path,
            max_open_shards=max_open_shards,
            max_label_sequences=max_label_sequences,
        )
        if set(member_sha256_by_token) != set(self.sample_tokens):
            raise ValueError("Member digest keys must exactly match requested tokens")
        ownership = {
            token: (
                str(self.rows[token]["rgb_shard_path"]),
                str(self.rows[token]["rgb_member_path"]),
                member_sha256_by_token[token],
            )
            for token in self.sample_tokens
        }
        self.indexed_reader = IndexedTarRGBReader(
            self.root,
            ownership,
            shard_sha256_by_path,
            max_open_handles=max_open_shards,
        )

    def __getitem__(self, index: int) -> RawRGBSample:
        token = self.sample_tokens[index]
        row = self.rows[token]
        start = int(row["rgb_exposure_start_timestamp_us"])
        end = int(row["rgb_exposure_end_timestamp_us"])
        if end < start:
            raise ValueError(f"Reversed exposure interval: {token}")
        raw = self.indexed_reader.read(token)
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
        """Close indexed payload handles and inherited bounded label caches."""
        if hasattr(self, "indexed_reader"):
            self.indexed_reader.close()
        super().close()
