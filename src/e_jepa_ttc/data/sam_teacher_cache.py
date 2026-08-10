"""Train-only SAM masks joined to the event cache by exact sample token."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SAMTeacherMaskDataset(Dataset[dict[str, Any]]):
    """Attach filtered SAM masks to a train dataset without changing model inputs.

    The wrapper eagerly loads only the compact packed masks. Event tensors remain
    lazy in the wrapped dataset. Every retrieved record is checked against the
    teacher sequence and common crop before its masks are exposed to the loss.
    """

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        manifest_path: str | Path,
        expected_artifact_sha256: str,
        expected_manifest_sha256: str,
    ) -> None:
        self.dataset = dataset
        path = Path(manifest_path).resolve(strict=True)
        if _sha256(path) != expected_manifest_sha256:
            raise ValueError("SAM teacher manifest file hash differs from protocol")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not verify_artifact_hash(manifest):
            raise ValueError("SAM teacher manifest signature is invalid")
        if manifest.get("artifact_sha256") != expected_artifact_sha256:
            raise ValueError("SAM teacher artifact identity differs from protocol")
        if manifest.get("status") != "passed":
            raise ValueError("SAM teacher cache did not pass its preregistered gates")
        scope = manifest.get("scope", {})
        claims = manifest.get("claim_boundary", {})
        if (
            scope.get("public_train_only") is not True
            or scope.get("validation_or_test_opened") is not False
            or claims.get("validation_or_test_teacher_generation") is not False
            or claims.get("ttc_labels_read") is not False
        ):
            raise ValueError("SAM teacher cache violates the train-only boundary")
        cache = manifest.get("cache", {})
        if int(cache.get("roi_size", 0)) <= 0:
            raise ValueError("SAM teacher cache has an invalid ROI size")
        self.roi_size = int(cache["roi_size"])
        self.bitorder = str(cache.get("packbits_bitorder", ""))
        if self.bitorder not in {"big", "little"}:
            raise ValueError("SAM teacher cache has an invalid packbits bit order")
        self.artifact_sha256 = expected_artifact_sha256
        self.manifest_sha256 = expected_manifest_sha256
        self._teacher: dict[str, tuple[str, np.ndarray, np.ndarray, np.ndarray]] = {}
        root = path.parent
        shards = cache.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("SAM teacher manifest has no shards")
        for shard in shards:
            npz_path = root / str(shard["npz_path"])
            if _sha256(npz_path) != str(shard["npz_sha256"]):
                raise ValueError(f"SAM teacher shard hash mismatch: {npz_path.name}")
            with np.load(npz_path, allow_pickle=False) as arrays:
                tokens = arrays["sample_tokens"].astype(str)
                sequences = arrays["sequence_ids"].astype(str)
                packed = arrays["masks_packbits"].astype(np.uint8, copy=False)
                valid = arrays["training_mask_valid"].astype(np.bool_, copy=False)
                squares = arrays["common_square_xyxy"].astype(np.float32, copy=False)
                expected_shape = (len(tokens), 2, self.roi_size * self.roi_size // 8)
                if packed.shape != expected_shape or valid.shape != (len(tokens), 2):
                    raise ValueError(f"SAM teacher shard shape mismatch: {npz_path.name}")
                if squares.shape != (len(tokens), 4):
                    raise ValueError(f"SAM teacher crop shape mismatch: {npz_path.name}")
                for index, token in enumerate(tokens.tolist()):
                    if token in self._teacher:
                        raise ValueError(f"duplicate SAM teacher token: {token}")
                    self._teacher[token] = (
                        str(sequences[index]),
                        packed[index].copy(),
                        valid[index].copy(),
                        squares[index].copy(),
                    )
        expected_rows = int(scope.get("row_count", -1))
        if len(self._teacher) != expected_rows or len(self._teacher) != len(dataset):
            raise ValueError("SAM teacher and event train dataset counts differ")

    def __len__(self) -> int:
        return len(self.dataset)

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        provider = getattr(self.dataset, "shard_index_groups", None)
        if not callable(provider):
            raise TypeError("wrapped event dataset does not expose shard groups")
        return provider()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.dataset[index])
        token = str(record["sample_token"])
        try:
            sequence, packed, valid, teacher_square = self._teacher[token]
        except KeyError as exc:
            raise KeyError(f"event train token missing from SAM teacher cache: {token}") from exc
        if sequence != str(record["sequence_id"]):
            raise ValueError(f"SAM teacher sequence mismatch for {token}")
        event_square = np.asarray(record["event_v4_common_square_xyxy"], dtype=np.float32)
        if not np.allclose(event_square, teacher_square, rtol=0.0, atol=1.0e-4):
            raise ValueError(f"SAM teacher common crop mismatch for {token}")
        unpacked = np.unpackbits(packed, axis=-1, bitorder=self.bitorder)
        masks = unpacked.reshape(2, 1, self.roi_size, self.roi_size).astype(np.float32)
        record["sam_teacher_masks"] = torch.from_numpy(masks)
        record["sam_teacher_mask_valid"] = torch.from_numpy(valid.copy())
        return record


__all__ = ["SAMTeacherMaskDataset"]
