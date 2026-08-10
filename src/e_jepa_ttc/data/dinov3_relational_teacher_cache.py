"""Train-only DINOv3 relational teacher maps joined by exact sample token.

Each sample_token corresponds to a specific object/track within a frame, with
its own bounding-box ROI and TTC.  The teacher cache is keyed by sample_token
and additionally verifies track_id — two samples from the same RGB frame but
targeting different objects must never share or deduplicate teacher entries.

The common-square crop is verified per-object to ensure spatial alignment
between the event model's dense features and the DINO relation maps.

The wrapper never exposes DINO features to validation or inference — only
the pre-computed six local cosine relation maps for endpoints t1/t2.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.distillation.dinov3_relational import A4_RELATION_OFFSETS

# Expected scientific teacher for A4
_EXPECTED_MODEL_ID = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
_EXPECTED_GRID = (32, 32)
_EXPECTED_ENDPOINTS = 2
_EXPECTED_OFFSETS = [[dy, dx] for dy, dx in A4_RELATION_OFFSETS]
_NUM_OFFSETS = len(A4_RELATION_OFFSETS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DINOv3RelationalTeacherDataset(Dataset[dict[str, Any]]):
    """Attach pre-computed DINO relational maps to a train dataset.

    The wrapper eagerly loads compact relation maps from shards.  Event
    tensors remain lazy in the wrapped dataset.  Every retrieved record
    is checked against the teacher token, track_id, sequence and common
    crop before its relation maps are exposed to the loss.

    The relation maps must never appear in ``event_inputs()`` or
    ``model_inputs()`` — they are accessed via dedicated batch fields
    ``dinov3_relation_targets`` and ``dinov3_relation_valid``.

    Multi-object invariant
    ----------------------
    A single RGB frame may contain multiple objects, each with its own
    bbox/ROI and TTC.  The cache stores ``track_id`` per row and the
    wrapper verifies it matches the base dataset record.  Entries are
    never deduplicated by RGB frame/member path.
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

        # --- Manifest file hash ---
        if _sha256(path) != expected_manifest_sha256:
            raise ValueError("DINO teacher manifest file hash differs from protocol")

        manifest = json.loads(path.read_text(encoding="utf-8"))

        # --- Artifact signature ---
        if not verify_artifact_hash(manifest):
            raise ValueError("DINO teacher manifest signature is invalid")

        # --- Artifact identity ---
        if manifest.get("artifact_sha256") != expected_artifact_sha256:
            raise ValueError("DINO teacher artifact identity differs from protocol")

        # --- Status ---
        if manifest.get("status") != "passed":
            raise ValueError("DINO teacher cache did not pass its preregistered gates")

        # --- Train-only scope ---
        scope = manifest.get("scope", {})
        claim = manifest.get("claim_boundary", {})
        if scope.get("public_train_only") is not True:
            raise ValueError("DINO teacher cache must be public_train_only")
        if scope.get("validation_or_test_opened") is not False:
            raise ValueError("DINO teacher cache must not open validation/test")
        if scope.get("ttc_labels_read") is not False:
            raise ValueError("DINO teacher cache must not read TTC labels")
        if claim.get("teacher_is_model_input") is not False:
            raise ValueError("DINO teacher must not be a model input")
        if claim.get("validation_teacher_generation") is not False:
            raise ValueError("DINO teacher must not be generated for validation")
        if claim.get("ttc_labels_read") is not False:
            raise ValueError("DINO teacher claim_boundary violates no-TTC constraint")

        # --- Teacher model identity ---
        teacher = manifest.get("teacher", {})
        if teacher.get("model_id") != _EXPECTED_MODEL_ID:
            raise ValueError(
                f"DINO teacher model must be {_EXPECTED_MODEL_ID}, "
                f"got {teacher.get('model_id')}"
            )

        # --- Relation config ---
        relations = manifest.get("relations", {})
        if relations.get("type") != "local_cosine":
            raise ValueError("DINO teacher relation type must be local_cosine")
        if relations.get("offsets_dy_dx") != _EXPECTED_OFFSETS:
            raise ValueError("DINO teacher offsets do not match A4 protocol")
        if (relations.get("grid_height"), relations.get("grid_width")) != _EXPECTED_GRID:
            raise ValueError(
                f"DINO teacher grid must be {_EXPECTED_GRID}, "
                f"got ({relations.get('grid_height')}, {relations.get('grid_width')})"
            )

        # --- Row/endpoint counts ---
        expected_rows = int(scope.get("row_count", -1))
        if int(scope.get("endpoint_count_per_row", -1)) != _EXPECTED_ENDPOINTS:
            raise ValueError("DINO teacher must have 2 endpoints per row")

        self.artifact_sha256 = expected_artifact_sha256
        self.manifest_sha256 = expected_manifest_sha256

        # --- Load shards ---
        # Keyed by sample_token → (track_id, sequence_id, relations, valid, crop)
        self._teacher: dict[
            str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        root = path.parent
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("DINO teacher manifest has no shards")

        for shard in shards:
            npz_path = root / str(shard["npz_path"])
            if _sha256(npz_path) != str(shard["npz_sha256"]):
                raise ValueError(f"DINO teacher shard hash mismatch: {npz_path.name}")

            with np.load(npz_path, allow_pickle=False) as arrays:
                tokens = arrays["sample_tokens"].astype(str)
                track_ids = arrays["track_ids"].astype(str)
                sequences = arrays["sequence_ids"].astype(str)
                relation_targets = arrays["relation_targets"]
                relation_valid = arrays["relation_valid"]
                squares = arrays["common_square_xyxy"].astype(np.float32, copy=False)

                expected_rel_shape = (
                    len(tokens),
                    _EXPECTED_ENDPOINTS,
                    _NUM_OFFSETS,
                    _EXPECTED_GRID[0],
                    _EXPECTED_GRID[1],
                )
                if relation_targets.shape != expected_rel_shape:
                    raise ValueError(
                        f"DINO teacher shard relation shape mismatch: "
                        f"{relation_targets.shape} != {expected_rel_shape} "
                        f"in {npz_path.name}"
                    )
                if relation_valid.shape != expected_rel_shape:
                    raise ValueError(
                        f"DINO teacher shard valid shape mismatch: "
                        f"{relation_valid.shape} != {expected_rel_shape} "
                        f"in {npz_path.name}"
                    )
                if squares.shape != (len(tokens), 4):
                    raise ValueError(
                        f"DINO teacher crop shape mismatch in {npz_path.name}"
                    )
                if track_ids.shape != tokens.shape:
                    raise ValueError(
                        f"DINO teacher track_ids shape mismatch in {npz_path.name}"
                    )

                # Verify finiteness where valid
                valid_bool = relation_valid.astype(bool)
                if not np.isfinite(relation_targets[valid_bool]).all():
                    raise ValueError(
                        f"DINO teacher shard contains non-finite values in valid "
                        f"region: {npz_path.name}"
                    )

                for idx, token in enumerate(tokens.tolist()):
                    if token in self._teacher:
                        raise ValueError(f"duplicate DINO teacher token: {token}")
                    self._teacher[token] = (
                        str(track_ids[idx]),
                        str(sequences[idx]),
                        relation_targets[idx].copy(),
                        relation_valid[idx].copy(),
                        squares[idx].copy(),
                    )

        # --- Count validation ---
        if len(self._teacher) != expected_rows:
            raise ValueError(
                f"DINO teacher has {len(self._teacher)} tokens, "
                f"expected {expected_rows}"
            )
        if len(self._teacher) != len(dataset):
            raise ValueError(
                f"DINO teacher ({len(self._teacher)} tokens) and "
                f"event train dataset ({len(dataset)} rows) counts differ"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        """Delegate to base dataset for deterministic sampling."""

        provider = getattr(self.dataset, "shard_index_groups", None)
        if not callable(provider):
            raise TypeError("wrapped event dataset does not expose shard groups")
        return provider()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.dataset[index])
        token = str(record["sample_token"])
        try:
            track_id, sequence, rel_targets, rel_valid, teacher_square = (
                self._teacher[token]
            )
        except KeyError as exc:
            raise KeyError(
                f"event train token missing from DINO teacher cache: {token}"
            ) from exc

        # --- Per-object identity validation ---
        if track_id != str(record["track_id"]):
            raise ValueError(
                f"DINO teacher track_id mismatch for {token}: "
                f"cache={track_id!r}, dataset={record['track_id']!r}"
            )
        if sequence != str(record["sequence_id"]):
            raise ValueError(f"DINO teacher sequence mismatch for {token}")

        event_square = np.asarray(
            record["event_v4_common_square_xyxy"], dtype=np.float32
        )
        if not np.allclose(event_square, teacher_square, rtol=0.0, atol=1.0e-4):
            raise ValueError(f"DINO teacher common crop mismatch for {token}")

        record["dinov3_relation_targets"] = torch.from_numpy(
            rel_targets.astype(np.float32, copy=False)
        )
        record["dinov3_relation_valid"] = torch.from_numpy(
            rel_valid.astype(np.bool_, copy=False).copy()
        )
        return record


__all__ = ["DINOv3RelationalTeacherDataset"]
