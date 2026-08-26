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
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.data.canonical_token_identity import (
    hash_ordered_token_ids,
    hash_sorted_token_strings,
)
from e_jepa_ttc.distillation.dinov3_relational import A4_RELATION_OFFSETS
from e_jepa_ttc.scientific_provenance import refuse_scientific_bypass_env

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


class CompleteDinoTeacherCache:
    """Verified read-only complete DINO teacher. Training never materializes DINO."""

    def __init__(
        self,
        teacher: dict[str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
        *,
        artifact_sha256: str,
        manifest_sha256: str,
        source_row_count: int,
    ) -> None:
        self._teacher = teacher
        self.artifact_sha256 = artifact_sha256
        self.manifest_sha256 = manifest_sha256
        self.source_teacher_row_count = source_row_count

    @classmethod
    def open_verified(
        cls,
        manifest_path: str | Path,
        *,
        expected_artifact_sha256: str,
        expected_manifest_sha256: str,
        allowed_sample_tokens: set[str] | frozenset[str] | None = None,
    ) -> CompleteDinoTeacherCache:
        refuse_scientific_bypass_env()
        path = Path(manifest_path).resolve(strict=True)
        if _sha256(path) != expected_manifest_sha256:
            raise ValueError("DINO teacher manifest file hash differs from protocol")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not verify_artifact_hash(manifest):
            raise ValueError("DINO teacher manifest signature is invalid")
        if manifest.get("artifact_sha256") != expected_artifact_sha256:
            raise ValueError("DINO teacher artifact identity differs from protocol")
        _assert_teacher_scientific_contract(manifest)
        if (path.parent / "teacher.npy").is_file():
            teacher, count = _load_mmap_teacher(path.parent, manifest, allowed_sample_tokens)
            return cls(
                teacher,
                artifact_sha256=expected_artifact_sha256,
                manifest_sha256=expected_manifest_sha256,
                source_row_count=count,
            )
        wrapped = DINOv3RelationalTeacherDataset(
            _TokenProbeDataset(allowed_sample_tokens, manifest),
            manifest_path=path,
            expected_artifact_sha256=expected_artifact_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            allowed_sample_tokens=(
                set(allowed_sample_tokens) if allowed_sample_tokens is not None else None
            ),
        )
        return cls(
            wrapped._teacher,
            artifact_sha256=wrapped.artifact_sha256,
            manifest_sha256=wrapped.manifest_sha256,
            source_row_count=wrapped.source_teacher_row_count,
        )

    def __getitem__(self, token: str) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
        try:
            return self._teacher[str(token)]
        except KeyError as exc:
            raise KeyError(f"event train token missing from DINO teacher cache: {token}") from exc

    def tokens(self) -> frozenset[str]:
        return frozenset(self._teacher)


class _TokenProbeDataset:
    def __init__(
        self,
        allowed: set[str] | frozenset[str] | None,
        manifest: dict[str, Any],
    ) -> None:
        if allowed is not None:
            self._tokens = tuple(sorted(str(token) for token in allowed))
        else:
            row_count = int(manifest["scope"]["row_count"])
            self._tokens = tuple(f"probe-{index}" for index in range(row_count))

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, index: int) -> dict[str, Any]:
        token = self._tokens[index]
        return {
            "sample_token": token,
            "track_id": "",
            "sequence_id": "",
            "event_v4_common_square_xyxy": [0.0, 0.0, 1.0, 1.0],
        }


def write_complete_mmap_cache(
    output_dir: Path,
    *,
    tokens: list[str],
    track_ids: list[str],
    sequence_ids: list[str],
    relation_targets: np.ndarray,
    relation_valid: np.ndarray,
    crops: np.ndarray,
    manifest: dict[str, Any],
    rgb_sha256: np.ndarray | None = None,
    rgb_endpoint_indices: np.ndarray | None = None,
    rgb_frame_timestamps_us: np.ndarray | None = None,
    event_windows_us: np.ndarray | None = None,
    rgb_shard_paths: np.ndarray | None = None,
    rgb_member_paths: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write one indivisible mmap cache. Callers must already reject incomplete inputs."""

    import os

    refuse_scientific_bypass_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(tokens) != len(set(tokens)):
        raise ValueError("complete DINO cache cannot contain duplicate tokens")
    expected_shape = (
        len(tokens),
        _EXPECTED_ENDPOINTS,
        _NUM_OFFSETS,
        _EXPECTED_GRID[0],
        _EXPECTED_GRID[1],
    )
    if relation_targets.shape != expected_shape or relation_valid.shape != expected_shape:
        raise ValueError("complete DINO cache tensor shape mismatch")
    if crops.shape != (len(tokens), 4):
        raise ValueError("complete DINO cache crop shape mismatch")
    teacher_path = output_dir / "teacher.npy"
    valid_path = output_dir / "valid.npy"
    crops_path = output_dir / "crops.npy"
    index_path = output_dir / "index.json"
    np.save(teacher_path, np.asarray(relation_targets, dtype=np.float16))
    np.save(valid_path, np.asarray(relation_valid, dtype=np.uint8))
    np.save(crops_path, np.asarray(crops, dtype=np.float32))
    provenance_payload: dict[str, str] = {}
    provenance = {
        "rgb_sha256": rgb_sha256,
        "rgb_endpoint_indices": rgb_endpoint_indices,
        "rgb_frame_timestamps_us": rgb_frame_timestamps_us,
        "event_windows_us": event_windows_us,
        "rgb_shard_paths": rgb_shard_paths,
        "rgb_member_paths": rgb_member_paths,
    }
    provided = {name: array for name, array in provenance.items() if array is not None}
    if provided and len(provided) != len(provenance):
        raise ValueError("complete DINO cache provenance arrays must all be provided together")
    endpoint_shape = (len(tokens), _EXPECTED_ENDPOINTS)
    if rgb_sha256 is not None:
        if rgb_sha256.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_sha256 shape mismatch")
        if rgb_endpoint_indices is None or rgb_endpoint_indices.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_endpoint_indices shape mismatch")
        if rgb_frame_timestamps_us is None or rgb_frame_timestamps_us.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_frame_timestamps_us shape mismatch")
        if event_windows_us is None or event_windows_us.shape != (
            len(tokens),
            _EXPECTED_ENDPOINTS,
            2,
        ):
            raise ValueError("complete DINO cache event_windows_us shape mismatch")
        if rgb_shard_paths is None or rgb_shard_paths.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_shard_paths shape mismatch")
        if rgb_member_paths is None or rgb_member_paths.shape != endpoint_shape:
            raise ValueError("complete DINO cache rgb_member_paths shape mismatch")
        for name, array in provided.items():
            path = output_dir / f"{name}.npy"
            np.save(path, array)
            provenance_payload[f"{name}_sha256"] = _sha256(path)
    rows = [
        {
            "token_id": str(token),
            "track_id": str(track_ids[idx]),
            "sequence_id": str(sequence_ids[idx]),
            "row_index": idx,
        }
        for idx, token in enumerate(tokens)
    ]
    index_path.write_text(
        json.dumps(
            {"rows": rows, "ordered_token_ids_sha256": hash_ordered_token_ids(tokens)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = dict(manifest)
    payload.update(
        {
            "schema_version": "complete_dino_teacher_mmap_v1",
            "coverage": 1.0,
            "duplicates": 0,
            "missing": 0,
            "unexpected": 0,
            "row_count_expected": len(tokens),
            "row_count_observed": len(tokens),
            "ordered_token_identity_hash": hash_sorted_token_strings(tokens),
            "cache_path": teacher_path.name,
            "cache_sha256": _sha256(teacher_path),
            "valid_sha256": _sha256(valid_path),
            "crops_sha256": _sha256(crops_path),
            "index_path": index_path.name,
            "index_sha256": _sha256(index_path),
            "tensor_shape": list(expected_shape),
            "dtype": "float16",
            **provenance_payload,
        }
    )
    sign_artifact(payload)
    manifest_path = output_dir / "manifest.json"
    tmp = manifest_path.with_name(".manifest.json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    return payload


def _assert_teacher_scientific_contract(manifest: dict[str, Any]) -> None:
    """Fail-closed checks shared by complete mmap caches and legacy NPZ shards."""

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
    if claim.get("teacher_source_modality") != "rgb":
        raise ValueError("DINO teacher must explicitly declare RGB source modality")
    if claim.get("event_tensor_used_as_teacher_input") is not False:
        raise ValueError("DINO teacher artifact may not use event tensors as teacher input")
    teacher = manifest.get("teacher", {})
    if teacher.get("model_id") != _EXPECTED_MODEL_ID:
        raise ValueError(
            f"DINO teacher model must be {_EXPECTED_MODEL_ID}, got {teacher.get('model_id')}"
        )
    if teacher.get("source_modality") != "rgb":
        raise ValueError("DINO teacher manifest does not prove RGB source modality")
    source_rgb = manifest.get("source_rgb", {})
    endpoint_stored = source_rgb.get("endpoint_metadata_stored_in_shards") is True or (
        source_rgb.get("endpoint_metadata_stored_in_complete_cache") is True
    )
    if not endpoint_stored:
        raise ValueError("DINO teacher shards must store endpoint provenance metadata")
    if source_rgb.get("raw_rgb_sha256_stored_per_endpoint") is not True:
        raise ValueError("DINO teacher shards must store raw RGB SHA256 per endpoint")
    code_identity = manifest.get("code_identity", {})
    if code_identity.get("git_dirty") is not False:
        raise ValueError("DINO teacher cache must be materialized from a clean Git worktree")
    if not isinstance(code_identity.get("git_commit"), str) or not code_identity["git_commit"]:
        raise ValueError("DINO teacher cache must record the materialization Git commit")
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
    if int(scope.get("endpoint_count_per_row", -1)) != _EXPECTED_ENDPOINTS:
        raise ValueError("DINO teacher must have 2 endpoints per row")


def _load_mmap_teacher(
    root: Path,
    manifest: dict[str, Any],
    allowed_sample_tokens: set[str] | frozenset[str] | None,
) -> tuple[dict[str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]], int]:
    if manifest.get("status") not in {"passed", "completed"}:
        raise ValueError("DINO teacher cache did not pass its preregistered gates")
    if float(manifest.get("coverage", -1)) != 1.0:
        raise ValueError("DINO teacher cache coverage must be exactly 1.0")
    code_identity = manifest.get("code_identity", {})
    if code_identity.get("git_dirty") is not False:
        raise ValueError("DINO teacher cache must be materialized from a clean Git worktree")
    if int(manifest.get("unexpected", 0)) != 0:
        raise ValueError("DINO complete cache contains unexpected tokens")
    if int(manifest.get("missing", 0)) != 0:
        raise ValueError("DINO complete cache is missing expected tokens")
    if int(manifest.get("duplicates", 0)) != 0:
        raise ValueError("DINO complete cache contains duplicate tokens")
    for path, key in (
        (root / "index.json", "index_sha256"),
        (root / "teacher.npy", "cache_sha256"),
        (root / "valid.npy", "valid_sha256"),
        (root / "crops.npy", "crops_sha256"),
    ):
        expected = manifest.get(key)
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"DINO complete-cache file hash mismatch: {path.name}")
    for name in (
        "rgb_sha256",
        "rgb_endpoint_indices",
        "rgb_frame_timestamps_us",
        "event_windows_us",
        "rgb_shard_paths",
        "rgb_member_paths",
    ):
        key = f"{name}_sha256"
        if key not in manifest:
            continue
        path = root / f"{name}.npy"
        expected = manifest.get(key)
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"DINO complete-cache file hash mismatch: {path.name}")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    tokens = [str(item["token_id"]) for item in index["rows"]]
    if index.get("ordered_token_ids_sha256") != hash_ordered_token_ids(tokens):
        raise ValueError(
            "DINO complete cache ordered_token_ids_sha256 is not the canonical JSON hash"
        )
    if len(tokens) != len(set(tokens)):
        raise ValueError("DINO complete cache contains duplicate tokens")
    expected_rows = int(manifest.get("row_count_observed", -1))
    if len(tokens) != expected_rows:
        raise ValueError(f"DINO complete cache has {len(tokens)} tokens, expected {expected_rows}")
    teacher = np.load(root / "teacher.npy", mmap_mode="r")
    valid = np.load(root / "valid.npy", mmap_mode="r")
    crops = np.load(root / "crops.npy", mmap_mode="r")
    expected_shape = (
        len(tokens),
        _EXPECTED_ENDPOINTS,
        _NUM_OFFSETS,
        _EXPECTED_GRID[0],
        _EXPECTED_GRID[1],
    )
    if tuple(teacher.shape) != expected_shape or tuple(valid.shape) != expected_shape:
        raise ValueError("DINO complete-cache tensor shape mismatch")
    allowed = (
        frozenset(str(token) for token in allowed_sample_tokens)
        if allowed_sample_tokens is not None
        else None
    )
    loaded: dict[str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = {}
    for idx, token in enumerate(tokens):
        if allowed is not None and token not in allowed:
            continue
        row = index["rows"][idx]
        loaded[token] = (
            str(row["track_id"]),
            str(row["sequence_id"]),
            np.asarray(teacher[idx]),
            np.asarray(valid[idx]),
            np.asarray(crops[idx], dtype=np.float32),
        )
    if allowed is not None and set(loaded) != set(allowed):
        raise ValueError("allowed teacher tokens are unavailable from the cache")
    return loaded, len(tokens)


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
        allowed_sample_tokens: set[str] | None = None,
    ) -> None:
        self.dataset = dataset
        self._dataset_size = len(cast(Sized, dataset))
        allowed = (
            frozenset(str(token) for token in allowed_sample_tokens)
            if allowed_sample_tokens is not None
            else None
        )
        if allowed is not None and len(allowed) != self._dataset_size:
            raise ValueError("allowed teacher tokens differ from dataset size")
        refuse_scientific_bypass_env()
        path = Path(manifest_path).resolve(strict=True)
        if (path.parent / "teacher.npy").is_file():
            cache = CompleteDinoTeacherCache.open_verified(
                path,
                expected_artifact_sha256=expected_artifact_sha256,
                expected_manifest_sha256=expected_manifest_sha256,
                allowed_sample_tokens=allowed,
            )
            self._teacher = cache._teacher
            self.artifact_sha256 = cache.artifact_sha256
            self.manifest_sha256 = cache.manifest_sha256
            self.source_teacher_row_count = cache.source_teacher_row_count
            if len(self._teacher) != self._dataset_size:
                raise ValueError(
                    f"DINO teacher ({len(self._teacher)} tokens) and "
                    f"event train dataset ({self._dataset_size} rows) counts differ"
                )
            return

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

        _assert_teacher_scientific_contract(manifest)

        # --- Row/endpoint counts ---
        expected_rows = int(manifest.get("scope", {}).get("row_count", -1))

        self.artifact_sha256 = expected_artifact_sha256
        self.manifest_sha256 = expected_manifest_sha256

        # --- Load shards ---
        # Keyed by sample_token → (track_id, sequence_id, relations, valid, crop)
        self._teacher: dict[str, tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = {}
        source_tokens: set[str] = set()
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
                rgb_sha256 = arrays["rgb_sha256"].astype(str)
                rgb_endpoint_indices = arrays["rgb_endpoint_indices"]
                rgb_frame_timestamps_us = arrays["rgb_frame_timestamps_us"]
                event_windows_us = arrays["event_windows_us"]
                rgb_shard_paths = arrays["rgb_shard_paths"].astype(str)
                rgb_member_paths = arrays["rgb_member_paths"].astype(str)

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
                    raise ValueError(f"DINO teacher crop shape mismatch in {npz_path.name}")
                if track_ids.shape != tokens.shape:
                    raise ValueError(f"DINO teacher track_ids shape mismatch in {npz_path.name}")
                expected_endpoint_shape = (len(tokens), _EXPECTED_ENDPOINTS)
                provenance_shapes = {
                    "rgb_sha256": rgb_sha256.shape,
                    "rgb_endpoint_indices": rgb_endpoint_indices.shape,
                    "rgb_frame_timestamps_us": rgb_frame_timestamps_us.shape,
                    "rgb_shard_paths": rgb_shard_paths.shape,
                    "rgb_member_paths": rgb_member_paths.shape,
                }
                for field, shape in provenance_shapes.items():
                    if shape != expected_endpoint_shape:
                        raise ValueError(
                            f"DINO teacher {field} shape mismatch: "
                            f"{shape} != {expected_endpoint_shape} in {npz_path.name}"
                        )
                if event_windows_us.shape != (len(tokens), _EXPECTED_ENDPOINTS, 2):
                    raise ValueError(
                        "DINO teacher event_windows_us shape mismatch: "
                        f"{event_windows_us.shape} in {npz_path.name}"
                    )
                if (rgb_endpoint_indices < 0).any():
                    raise ValueError(
                        f"DINO teacher endpoint indices must be non-negative in {npz_path.name}"
                    )
                if not np.all(rgb_frame_timestamps_us[:, 1] > rgb_frame_timestamps_us[:, 0]):
                    raise ValueError(
                        f"DINO teacher endpoint timestamps must increase in {npz_path.name}"
                    )
                sha_lengths_ok = np.vectorize(
                    lambda value: (
                        len(str(value)) == 64
                        and all(c in "0123456789abcdef" for c in str(value).lower())
                    )
                )(rgb_sha256)
                if not bool(np.all(sha_lengths_ok)):
                    raise ValueError(
                        f"DINO teacher RGB SHA256 metadata is malformed in {npz_path.name}"
                    )
                if np.any(rgb_shard_paths == "") or np.any(rgb_member_paths == ""):
                    raise ValueError(f"DINO teacher RGB path metadata is empty in {npz_path.name}")

                # Verify finiteness where valid
                valid_bool = relation_valid.astype(bool)
                if not np.isfinite(relation_targets[valid_bool]).all():
                    raise ValueError(
                        f"DINO teacher shard contains non-finite values in valid "
                        f"region: {npz_path.name}"
                    )

                for idx, token in enumerate(tokens.tolist()):
                    token = str(token)
                    if token in source_tokens:
                        raise ValueError(f"duplicate DINO teacher token: {token}")
                    source_tokens.add(token)
                    if allowed is not None and token not in allowed:
                        continue
                    self._teacher[token] = (
                        str(track_ids[idx]),
                        str(sequences[idx]),
                        relation_targets[idx].copy(),
                        relation_valid[idx].copy(),
                        squares[idx].copy(),
                    )

        # --- Count validation ---
        self.source_teacher_row_count = len(source_tokens)
        if self.source_teacher_row_count != expected_rows:
            raise ValueError(
                f"DINO teacher has {self.source_teacher_row_count} source tokens, "
                f"expected {expected_rows}"
            )
        if allowed is not None and set(self._teacher) != set(allowed):
            raise ValueError("allowed teacher tokens are unavailable from the cache")
        if len(self._teacher) != self._dataset_size:
            raise ValueError(
                f"DINO teacher ({len(self._teacher)} tokens) and "
                f"event train dataset ({self._dataset_size} rows) counts differ"
            )

    def __len__(self) -> int:
        return self._dataset_size

    def teacher_sample_tokens(self) -> frozenset[str]:
        """Return the exact teacher-token subset exposed by this wrapper."""

        return frozenset(self._teacher)

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        """Delegate to base dataset for deterministic sampling."""

        provider = getattr(self.dataset, "shard_index_groups", None)
        if not callable(provider):
            raise TypeError("wrapped event dataset does not expose shard groups")
        return cast(tuple[tuple[int, ...], ...], provider())

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.dataset[index])
        token = str(record["sample_token"])
        try:
            track_id, sequence, rel_targets, rel_valid, teacher_square = self._teacher[token]
        except KeyError as exc:
            raise KeyError(f"event train token missing from DINO teacher cache: {token}") from exc

        # --- Per-object identity validation ---
        if track_id != str(record["track_id"]):
            raise ValueError(
                f"DINO teacher track_id mismatch for {token}: "
                f"cache={track_id!r}, dataset={record['track_id']!r}"
            )
        if sequence != str(record["sequence_id"]):
            raise ValueError(f"DINO teacher sequence mismatch for {token}")

        event_square = np.asarray(record["event_v4_common_square_xyxy"], dtype=np.float32)
        if not np.allclose(event_square, teacher_square, rtol=0.0, atol=1.0e-4):
            raise ValueError(f"DINO teacher common crop mismatch for {token}")

        record["dinov3_relation_targets"] = torch.from_numpy(
            rel_targets.astype(np.float32, copy=False)
        )
        record["dinov3_relation_valid"] = torch.from_numpy(
            rel_valid.astype(np.bool_, copy=False).copy()
        )
        return record


__all__ = [
    "CompleteDinoTeacherCache",
    "DINOv3RelationalTeacherDataset",
    "write_complete_mmap_cache",
]
