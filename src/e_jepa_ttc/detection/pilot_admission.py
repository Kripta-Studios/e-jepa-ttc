"""Fail-closed admission checks for the bounded public-train RGB pilot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_ROLES = ("train", "selection", "calibration")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _strings(value: object, field: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{field} must be a{' nonempty' if nonempty else ''} list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must contain nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} contains duplicates")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _token_sequence(token: str) -> str:
    parts = token.split(":")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid sample token: {token!r}")
    return parts[0]


def _relative_path(value: object, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe {field}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe {field}: {value!r}")
    return path


def _role_sequences(candidate: dict[str, Any]) -> dict[str, list[str]]:
    if candidate.get("schema") != "eap_detection_sequence_split_v1":
        raise ValueError("Unexpected candidate split schema")
    if candidate.get("source_split") != "train":
        raise ValueError("Candidate split must derive only from public train")
    assignments = candidate.get("assignments")
    if not isinstance(assignments, dict) or set(assignments) != set(_ROLES):
        raise ValueError("Candidate assignments must contain exactly train/selection/calibration")
    roles = {role: _strings(assignments[role], f"candidate.assignments.{role}") for role in _ROLES}
    for index, left in enumerate(_ROLES):
        for right in _ROLES[index + 1 :]:
            if overlap := set(roles[left]) & set(roles[right]):
                raise ValueError(f"Candidate roles overlap ({left}/{right}): {sorted(overlap)}")
    return roles


def validate_pilot_admission(
    manifest_path: Path, candidate_split_path: Path, data_root: Path
) -> dict[str, Any]:
    """Validate pilot identities and canonical Parquet metadata before GPU setup.

    This reads JSON plus selected rows from ``data/train.parquet``. It never opens
    RGB archives, event HDF5 files, labels, model weights, Torch, or CUDA.
    """
    manifest_path = manifest_path.resolve()
    candidate_split_path = candidate_split_path.resolve()
    media_path = (data_root / "data" / "train.parquet").resolve()
    manifest = _load_json(manifest_path)
    candidate = _load_json(candidate_split_path)

    if manifest.get("schema") != "eap_rgb_pilot_v1":
        raise ValueError("Unexpected pilot manifest schema")
    if manifest.get("source_split") != "train":
        raise ValueError("Pilot manifest must derive only from public train")
    revision = manifest.get("dataset_revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("dataset_revision must be a lowercase 40-character Git revision")
    if candidate.get("dataset_revision") != revision:
        raise ValueError("Pilot and candidate dataset revisions differ")
    if candidate.get("seed") != manifest.get("seed"):
        raise ValueError("Pilot and candidate seeds differ")

    candidate_hash = _digest(manifest.get("candidate_split_sha256"), "candidate_split_sha256")
    if _sha256(candidate_split_path) != candidate_hash:
        raise ValueError("Candidate split file differs from the pilot declaration")
    media_hash = _digest(manifest.get("source_media_sha256"), "source_media_sha256")
    if _digest(candidate.get("media_sha256"), "candidate.media_sha256") != media_hash:
        raise ValueError("Pilot and candidate media hashes differ")
    if _sha256(media_path) != media_hash:
        raise ValueError("Canonical data/train.parquet differs from the pilot declaration")

    candidate_roles = _role_sequences(candidate)
    selected: dict[str, list[str]] = {}
    declared_sequences: dict[str, list[str]] = {}
    for role in ("train", "selection"):
        section = manifest.get(role)
        if not isinstance(section, dict):
            raise ValueError(f"Missing pilot role: {role}")
        tokens = _strings(section.get("sample_tokens"), f"manifest.{role}.sample_tokens")
        sequences = _strings(section.get("sequences"), f"manifest.{role}.sequences")
        prefixes = {_token_sequence(token) for token in tokens}
        if prefixes != set(sequences):
            raise ValueError(f"Token prefixes and declared sequences differ for {role}")
        if not set(sequences).issubset(candidate_roles[role]):
            raise ValueError(f"Pilot {role} sequences do not belong to candidate {role}")
        if set(sequences) & set(candidate_roles["calibration"]):
            raise ValueError(f"Pilot {role} contains a calibration sequence")
        selected[role], declared_sequences[role] = tokens, sequences
    if overlap := set(selected["train"]) & set(selected["selection"]):
        raise ValueError(f"Train/selection tokens overlap: {sorted(overlap)}")
    if overlap := set(declared_sequences["train"]) & set(declared_sequences["selection"]):
        raise ValueError(f"Train/selection sequences overlap: {sorted(overlap)}")

    overfit = manifest.get("overfit16")
    if not isinstance(overfit, dict):
        raise ValueError("Missing overfit16 section")
    overfit_tokens = _strings(overfit.get("sample_tokens"), "manifest.overfit16.sample_tokens")
    if not set(overfit_tokens).issubset(selected["train"]):
        raise ValueError("overfit16 must be a subset of pilot train tokens")

    all_tokens = selected["train"] + selected["selection"]
    columns = [
        "sample_token",
        "split",
        "sequence_id",
        "rgb_shard_path",
        "rgb_member_path",
        "events_path",
        "labels_path",
    ]
    table = pq.read_table(media_path, columns=columns, filters=[("sample_token", "in", all_tokens)])
    rows = table.to_pylist()
    if len(rows) != len(all_tokens):
        raise ValueError("Canonical media has missing or duplicate selected sample tokens")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = row["sample_token"]
        if token in indexed:
            raise ValueError(f"Canonical media repeats selected token: {token}")
        indexed[token] = row
    if set(indexed) != set(all_tokens):
        raise ValueError("Canonical media token set differs from the pilot selection")
    for token, row in indexed.items():
        sequence = _token_sequence(token)
        if row["split"] != "train" or row["sequence_id"] != sequence:
            raise ValueError(f"Canonical media identity differs for {token}")
        prefix = ("data", "train", sequence)
        for field in ("rgb_shard_path", "events_path", "labels_path"):
            if _relative_path(row[field], field).parts[:3] != prefix:
                raise ValueError(f"{field} does not belong to public train sequence {sequence}")
        member = _relative_path(row["rgb_member_path"], "rgb_member_path")
        if member.parts[0] != "rgb" or member.stem != token.split(":", 1)[1]:
            raise ValueError(f"RGB member identity differs for {token}")

    return {
        "schema": "eap_rgb_pilot_admission_v1",
        "status": "PASS",
        "scope": "Metadata identity and sequence separation only; not geometry or media bytes",
        "manifest_sha256": _sha256(manifest_path),
        "candidate_split_sha256": candidate_hash,
        "source_media_sha256": media_hash,
        "dataset_revision": revision,
        "source_split": "train",
        "seed": manifest["seed"],
        "train_sequence_count": len(declared_sequences["train"]),
        "selection_sequence_count": len(declared_sequences["selection"]),
        "train_token_count": len(selected["train"]),
        "selection_token_count": len(selected["selection"]),
        "overfit_token_count": len(overfit_tokens),
        "canonical_rows_verified": len(rows),
        "token_overlap_count": 0,
        "sequence_overlap_count": 0,
        "calibration_sequences_selected": 0,
    }
