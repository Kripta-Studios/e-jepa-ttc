"""Lossless, resumable feature extraction sink with explicit producer genealogy.

The caller supplies model outputs from registered frozen parents. No labels,
normalizer fitting, model inference or scientific selection happen here.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from operational.range_transfer.common import digest, freeze, read, sha

from .fusion_features import FeatureBlock, FeatureIdentity, load_feature_cache

FIELDS = ("sample_tokens", "features", "prediction_ttc", "available", "sensor_pair_us")


def save_block(path: Path, block: FeatureBlock) -> None:
    """Publish a lossless NPZ atomically, with only the declared feature fields."""
    metadata = {
        "schema": "v18_fusion_feature_cache_v1",
        "identity": asdict(block.identity),
        "role": block.role,
        "manifest_sha256": block.manifest_sha256,
    }
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
            **{name: getattr(block, name) for name in FIELDS},
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


class FusionCacheWriter:
    """One caller-owned writer; a receipt admits each exact ordered query slice."""

    def __init__(
        self,
        directory: Path,
        identity: FeatureIdentity,
        *,
        role: str,
        manifest_sha256: str,
        sample_tokens: np.ndarray,
        feature_count: int,
        input_binding: dict,
        chunk_size: int = 256,
    ) -> None:
        identity.validate()
        if (
            role not in {"P", "H", "V"}
            or not input_binding
            or not isinstance(feature_count, int)
            or isinstance(feature_count, bool)
            or not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or min(feature_count, chunk_size) < 1
            or len(manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in manifest_sha256)
            or sample_tokens.ndim != 1
            or sample_tokens.dtype.kind not in {"U", "S"}
            or len(sample_tokens) == 0
            or len(np.unique(sample_tokens)) != len(sample_tokens)
        ):
            raise ValueError("Invalid externally admitted feature extraction population")
        self.directory, self.identity, self.role = directory, identity, role
        self.manifest_sha256 = manifest_sha256
        self.tokens = sample_tokens.copy()
        self.width, self.chunk_size = feature_count, chunk_size
        self.binding = {
            "identity": asdict(identity),
            "role": role,
            "manifest_sha256": manifest_sha256,
            "ordered_tokens_sha256": hashlib.sha256(self.tokens.tobytes()).hexdigest(),
            "token_dtype": str(self.tokens.dtype),
            "population": len(self.tokens),
            "feature_count": feature_count,
            "chunk_size": chunk_size,
            "input_binding": json.loads(json.dumps(input_binding, sort_keys=True)),
            "writer_sha256": sha(Path(__file__)),
            "admission_sha256": sha(Path(__file__).with_name("fusion_features.py")),
        }
        directory.mkdir(parents=True, exist_ok=True)
        freeze(directory / "BINDING.json", self.binding)
        self.key = digest(self.binding)

    @property
    def chunks(self) -> int:
        return (len(self.tokens) + self.chunk_size - 1) // self.chunk_size

    def bounds(self, index: int) -> tuple[int, int]:
        """Reject arbitrary or negative shard identifiers before resolving a path."""
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.chunks:
            raise ValueError("Invalid fusion feature chunk index")
        start = index * self.chunk_size
        return start, min(start + self.chunk_size, len(self.tokens))

    def load(self, path: Path, checksum: str, tokens: np.ndarray) -> FeatureBlock:
        """Admit exact bytes, expected producer, query order and native clocks."""
        return load_feature_cache(
            path,
            expected_sha256=checksum,
            expected_identity=self.identity,
            role=self.role,
            manifest_sha256=self.manifest_sha256,
            sample_tokens=tokens,
            feature_count=self.width,
        )

    def read_chunk(self, index: int) -> FeatureBlock | None:
        """Return a verified shard, or None if no durable receipt admitted it."""
        start, stop = self.bounds(index)
        path = self.directory / f"chunk_{index:05d}.npz"
        if not path.with_suffix(".json").exists():
            return None
        receipt = read(path.with_suffix(".json"))
        if receipt["binding"] != self.key or receipt["start"] != start or receipt["stop"] != stop:
            raise ValueError("Fusion feature shard receipt changed")
        return self.load(path, receipt["sha256"], self.tokens[start:stop])

    def write_chunk(self, index: int, block: FeatureBlock) -> dict:
        """Validate a complete slice before publication; never overwrite an admitted shard."""
        start, stop = self.bounds(index)
        if self.read_chunk(index) is not None:
            raise FileExistsError("Feature shard already admitted; reuse the verified result")
        block.validate(
            expected_identity=self.identity,
            role=self.role,
            manifest_sha256=self.manifest_sha256,
            sample_tokens=self.tokens[start:stop],
            feature_count=self.width,
        )
        path = self.directory / f"chunk_{index:05d}.npz"
        save_block(path, block)
        receipt = {"binding": self.key, "start": start, "stop": stop, "sha256": sha(path)}
        freeze(path.with_suffix(".json"), receipt)
        return receipt

    def complete(self) -> dict:
        """Publish a complete ordered cohort, preserving missing RGB queries and FP32 bytes."""
        final = self.directory / "features.npz"
        receipt_path = self.directory / "COMPLETE.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt["binding"] != self.key or receipt["rows"] != len(self.tokens):
                raise ValueError("Completed feature cohort identity changed")
            self.load(final, receipt["sha256"], self.tokens)
            return receipt
        blocks = [self.read_chunk(index) for index in range(self.chunks)]
        if any(block is None for block in blocks):
            raise ValueError("Complete every feature shard before publishing the cohort")
        arrays = {
            name: np.concatenate([getattr(block, name) for block in blocks]) for name in FIELDS
        }
        block = FeatureBlock(
            identity=self.identity, role=self.role, manifest_sha256=self.manifest_sha256, **arrays
        )
        block.validate(
            expected_identity=self.identity,
            role=self.role,
            manifest_sha256=self.manifest_sha256,
            sample_tokens=self.tokens,
            feature_count=self.width,
        )
        save_block(final, block)
        checksum = sha(final)
        self.load(final, checksum, self.tokens)
        receipt = {
            "status": "COMPLETE",
            "binding": self.key,
            "sha256": checksum,
            "rows": len(self.tokens),
            "available_rows": int(block.available.sum()),
            "feature_count": self.width,
            "chunks": self.chunks,
            "lossless": True,
            "contains_targets": False,
        }
        freeze(receipt_path, receipt)
        return receipt
