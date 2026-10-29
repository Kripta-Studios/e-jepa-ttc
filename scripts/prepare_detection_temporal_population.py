#!/usr/bin/env python
"""Admit causal RGB history media for frozen detection train/selection populations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
from PIL import Image, ImageFile

from e_jepa_ttc.detection.temporal_data import TemporalRGBIndex

SCHEMA = "eap_temporal_media_admission_v1"
KNOWN_BAD_TOKEN = "OYgB6RGWcq:000000661_000000160"
DEFAULT_MAXIMUM_HISTORY_AGE_US = 500_000
EXPECTED_CURRENT_COUNT = 28_054
EXPECTED_TRAIN_CURRENT_COUNT = 4_096
EXPECTED_SELECTION_CURRENT_COUNT = 23_958
EXPECTED_UNIQUE_HISTORY_COUNT = 32_080
EXPECTED_NEW_HISTORY_COUNT = 8_109
EXPECTED_NEW_HISTORY_SHARD_COUNT = 78
EXPECTED_CANONICAL_SLOT_COUNTS = (28_014, 28_022)
EXPECTED_ROLE_SUPPORT = {
    "train": {"history_0": 24, "history_1": 0, "history_2": 4_072},
    "selection": {"history_0": 8, "history_1": 8, "history_2": 23_942},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _strings(value: object, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    result = list(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return cast(list[str], result)


def _load_jsonl_by(path: Path, key: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get(key), str):
                raise ValueError(f"Invalid JSONL row {line_number}: {path}")
            identity = str(value[key])
            if identity in result:
                raise ValueError(f"Duplicate {key} in {path}: {identity}")
            result[identity] = cast(dict[str, object], value)
    return result


def _split_owners(split: Mapping[str, object]) -> dict[str, str]:
    if split.get("schema") != "eap_detection_sequence_split_v1":
        raise ValueError("Unexpected split schema")
    assignments = split.get("assignments")
    if not isinstance(assignments, Mapping) or set(assignments) != {
        "train",
        "selection",
        "calibration",
    }:
        raise ValueError("Split must preserve exact train/selection/calibration roles")
    owner: dict[str, str] = {}
    for role, raw in assignments.items():
        for sequence in _strings(raw, f"assignments.{role}"):
            if sequence in owner:
                raise ValueError("Split sequence roles overlap")
            owner[sequence] = str(role)
    return owner


def build_temporal_entries(
    *,
    index: TemporalRGBIndex,
    current_tokens: Sequence[str],
    role: str,
    sequence_owner: Mapping[str, str],
    media_by_token: Mapping[str, Mapping[str, object]],
    maximum_history_age_us: int,
) -> tuple[list[dict[str, object]], Counter[str]]:
    """Resolve canonical histories and apply age/media masks without backfilling."""
    if maximum_history_age_us <= 0:
        raise ValueError("maximum_history_age_us must be positive")
    entries: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for current_token in current_tokens:
        sample = index.resolve(current_token, control="correct_past")
        if sequence_owner.get(sample.sequence_id) != role:
            raise ValueError("Current token leaves its frozen sequence role")
        before_tokens = list(sample.history_tokens)
        before_mask = list(sample.history_mask)
        for slot, valid in enumerate(before_mask):
            if valid:
                counts[f"canonical_valid_slot_{slot}"] += 1
        final_tokens: list[str | None] = []
        final_mask: list[bool] = []
        final_ends: list[int | None] = []
        final_ages: list[int | None] = []
        history_media: list[dict[str, object] | None] = []
        mask_reasons: list[str | None] = []
        for slot, token in enumerate(sample.history_tokens):
            age = sample.history_age_us[slot]
            end = sample.history_exposure_end_us[slot]
            if token is None:
                final_tokens.append(None)
                final_mask.append(False)
                final_ends.append(None)
                final_ages.append(None)
                history_media.append(None)
                mask_reasons.append("sequence_boundary")
                counts["masked_sequence_boundary"] += 1
                continue
            if age is None or age > maximum_history_age_us:
                final_tokens.append(None)
                final_mask.append(False)
                final_ends.append(None)
                final_ages.append(None)
                history_media.append(None)
                mask_reasons.append("maximum_history_age_exceeded")
                counts["masked_maximum_age"] += 1
                continue
            media = media_by_token.get(token)
            if media is None or media.get("status") != "PASS":
                final_tokens.append(None)
                final_mask.append(False)
                final_ends.append(None)
                final_ages.append(None)
                history_media.append(dict(media) if media is not None else None)
                reason = (
                    "known_unreadable_history" if token == KNOWN_BAD_TOKEN else "media_unavailable"
                )
                mask_reasons.append(reason)
                counts[f"masked_{reason}"] += 1
                continue
            if media.get("sequence_id") != sample.sequence_id:
                raise ValueError("History media crosses sequence identity")
            final_tokens.append(token)
            final_mask.append(True)
            final_ends.append(end)
            final_ages.append(age)
            history_media.append(dict(media))
            mask_reasons.append(None)
            counts["valid_history_slots"] += 1
        counts[f"history_{sum(final_mask)}"] += 1
        entries.append(
            {
                "current_token": current_token,
                "sequence_id": sample.sequence_id,
                "current_exposure_start_us": sample.current_exposure_start_us,
                "current_exposure_end_us": sample.current_exposure_end_us,
                "decision_sensor_us": sample.decision_sensor_us,
                "canonical_history_tokens_before_media": before_tokens,
                "history_mask_before_media": before_mask,
                "history_tokens": final_tokens,
                "history_mask": final_mask,
                "history_exposure_end_us": final_ends,
                "history_age_us": final_ages,
                "history_media": history_media,
                "history_mask_reasons": mask_reasons,
                "shared_history_mask_across_variants": True,
                "substitute_history_after_failure": False,
                "availability_scope": sample.availability_scope,
            }
        )
    return entries, counts


def admit_history_media(
    *,
    data_root: Path,
    metadata_by_token: Mapping[str, Mapping[str, object]],
    history_tokens: Sequence[str],
    current_media: Mapping[str, Mapping[str, object]],
    shard_receipts: Mapping[str, Mapping[str, object]],
    pinned_files: Mapping[str, object],
    known_bad_receipt: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Reuse admitted current media and strict-decode only new unique histories."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    admitted: dict[str, dict[str, object]] = {}
    negatives: list[dict[str, object]] = []
    by_shard: dict[str, list[str]] = defaultdict(list)
    shard_bindings: dict[str, object] = {}

    def bind_tar(relative: str) -> tuple[Path, str, int, str]:
        tar_path = (data_root / relative).resolve()
        if not tar_path.is_relative_to(data_root.resolve()):
            raise ValueError("RGB TAR path escapes data root")
        pinned = pinned_files.get(relative)
        if not isinstance(pinned, Mapping):
            raise ValueError(f"Pinned inventory lacks RGB TAR: {relative}")
        prior = shard_receipts.get(relative)
        observed_size = tar_path.stat().st_size
        if prior is not None:
            if (
                prior.get("observed_sha256") != pinned.get("sha256")
                or prior.get("observed_size_bytes") != pinned.get("size_bytes")
                or observed_size != pinned.get("size_bytes")
            ):
                raise ValueError(f"Prior TAR receipt differs from pinned inventory: {relative}")
            return (
                tar_path,
                str(prior["observed_sha256"]),
                observed_size,
                "rgb_full_preflight_v1_shard_receipt",
            )
        if observed_size != pinned.get("size_bytes"):
            raise ValueError(f"New history TAR size differs from pinned inventory: {relative}")
        tar_sha256 = sha256_file(tar_path)
        if tar_sha256 != pinned.get("sha256"):
            raise ValueError(f"New history TAR hash differs from pinned inventory: {relative}")
        return tar_path, tar_sha256, observed_size, "new_full_tar_sha256"

    for token in sorted(set(history_tokens)):
        if token in current_media:
            row = dict(current_media[token])
            row.update(
                {
                    "sample_token": token,
                    "status": "PASS",
                    "readability": "strict_decode_reused_from_v5_current_receipt",
                }
            )
            admitted[token] = row
            continue
        metadata = metadata_by_token.get(token)
        if metadata is None:
            failure: dict[str, object] = {
                "sample_token": token,
                "status": "FAIL",
                "reason": "metadata_missing",
            }
            admitted[token] = failure
            negatives.append(failure)
            continue
        if token == KNOWN_BAD_TOKEN:
            relative = str(metadata["rgb_shard_path"])
            member_path = str(metadata["rgb_member_path"])
            tar_path, tar_sha256, tar_size, verification_source = bind_tar(relative)
            with tarfile.open(tar_path, mode="r:") as archive:
                stream = archive.extractfile(archive.getmember(member_path))
                if stream is None:
                    raise OSError("Known unreadable TAR member cannot be extracted")
                raw = stream.read()
            observed_error: Exception | None = None
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
            except Exception as error:
                observed_error = error
            if observed_error is None:
                raise ValueError("Known unreadable RGB member unexpectedly decoded")
            if type(observed_error).__name__ != known_bad_receipt.get("error_type"):
                raise ValueError("Known unreadable RGB exception type changed")
            failure = {
                "sample_token": token,
                "sequence_id": metadata["sequence_id"],
                "rgb_shard_path": relative,
                "rgb_member_path": member_path,
                "member_size_bytes": len(raw),
                "member_sha256": hashlib.sha256(raw).hexdigest(),
                "status": "UNREADABLE",
                "readability": "strict_decode_failed",
                "reason": "known_truncated_png",
                "error_type": type(observed_error).__name__,
                "error": str(observed_error),
                "tar_sha256": tar_sha256,
                "tar_size_bytes": tar_size,
                "tar_verification_source": verification_source,
                "known_bad_receipt_sha256": canonical_sha256(known_bad_receipt),
                "source_failure_receipt": dict(known_bad_receipt),
            }
            admitted[token] = failure
            negatives.append(failure)
            shard_bindings[relative] = {
                "status": "PASS_WITH_UNREADABLE_MEMBER",
                "tar_sha256": tar_sha256,
                "tar_size_bytes": tar_size,
                "verification_source": verification_source,
                "requested_history_members": 1,
                "unreadable_history_members": 1,
            }
            continue
        by_shard[str(metadata["rgb_shard_path"])].append(token)
    for relative, tokens in sorted(by_shard.items()):
        tar_path, tar_sha256, tar_size, verification_source = bind_tar(relative)
        previous_binding = shard_bindings.get(relative)
        unreadable_count = 0
        if isinstance(previous_binding, Mapping):
            unreadable_count = int(previous_binding.get("unreadable_history_members", 0))
        shard_bindings[relative] = {
            "status": "PASS_WITH_UNREADABLE_MEMBER" if unreadable_count else "PASS",
            "tar_sha256": tar_sha256,
            "tar_size_bytes": tar_size,
            "verification_source": verification_source,
            "requested_history_members": len(tokens) + unreadable_count,
            "unreadable_history_members": unreadable_count,
        }
        with tarfile.open(tar_path, mode="r:") as archive:
            for token in sorted(tokens):
                metadata = metadata_by_token[token]
                member_path = str(metadata["rgb_member_path"])
                try:
                    member = archive.getmember(member_path)
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise OSError("TAR member cannot be extracted")
                    raw = stream.read()
                    member_sha256 = hashlib.sha256(raw).hexdigest()
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        image_wh = [int(image.width), int(image.height)]
                    admitted[token] = {
                        "sample_token": token,
                        "sequence_id": metadata["sequence_id"],
                        "rgb_shard_path": relative,
                        "rgb_member_path": member_path,
                        "member_size_bytes": len(raw),
                        "member_sha256": member_sha256,
                        "image_wh": image_wh,
                        "status": "PASS",
                        "readability": "strict_decode_new_history",
                        "tar_sha256": tar_sha256,
                    }
                except Exception as error:
                    failure = {
                        "sample_token": token,
                        "sequence_id": metadata["sequence_id"],
                        "rgb_shard_path": relative,
                        "rgb_member_path": member_path,
                        "status": "FAIL",
                        "readability": "strict_decode_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "tar_sha256": tar_sha256,
                    }
                    admitted[token] = failure
                    negatives.append(failure)
    return admitted, negatives, shard_bindings


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    split = read_object(args.split, "split")
    owner = _split_owners(split)
    manifest = read_object(args.v5_manifest, "v5 manifest")
    v5_receipt = read_object(args.v5_receipt, "v5 receipt")
    media_admission = read_object(args.v5_media_admission, "v5 media admission")
    full_selection = read_object(args.full_selection, "full selection")
    preflight_summary = read_object(args.preflight_summary, "RGB preflight summary")
    pinned = read_object(args.pinned_manifest, "pinned public inventory")
    if manifest.get("schema") != "eap_detection_campaign_manifest_v1":
        raise ValueError("Unexpected v5 manifest schema")
    if v5_receipt.get("schema") != "eap_detection_readable_campaign_revision_v1":
        raise ValueError("Unexpected v5 receipt schema")
    if media_admission.get("schema") != "eap_detection_media_admission_v1":
        raise ValueError("Unexpected v5 media-admission schema")
    if full_selection.get("schema") != "eap_detection_full_selection_v1":
        raise ValueError("Unexpected full-selection schema")
    if pinned.get("schema_version") != "eap_detection_public_files_v1":
        raise ValueError("Unexpected pinned public inventory schema")
    media_path = args.data_root / "data/train.parquet"
    source_media_sha256 = sha256_file(media_path)
    expected_identities = {
        "d4096_v5_manifest_sha256": (args.v5_manifest, v5_receipt.get("new_manifest_sha256")),
        "v5_media_admission_sha256": (
            args.v5_media_admission,
            v5_receipt.get("media_admission_sha256"),
        ),
        "full_selection_sha256": (args.full_selection, v5_receipt.get("full_selection_sha256")),
        "candidate_split_sha256": (args.split, manifest.get("candidate_split_sha256")),
        "pinned_inventory_sha256": (
            args.pinned_manifest,
            media_admission.get("source_pinned_inventory_sha256"),
        ),
        "preflight_summary_sha256": (
            args.preflight_summary,
            media_admission.get("source_full_preflight_summary_sha256"),
        ),
        "preflight_shard_ledger_sha256": (
            args.preflight_shard_ledger,
            media_admission.get("source_full_preflight_shard_ledger_sha256"),
        ),
    }
    for name, (path, expected) in expected_identities.items():
        if sha256_file(path) != expected:
            raise ValueError(f"{name} differs from its frozen receipt")
    if source_media_sha256 != split.get("media_sha256") or source_media_sha256 != manifest.get(
        "source_media_sha256"
    ):
        raise ValueError("Canonical metadata differs from split/v5 manifest")
    train_tokens = _strings(manifest["roles"]["train"]["sample_tokens"], "v5 train tokens")
    selection_tokens = _strings(full_selection.get("sample_tokens"), "full selection tokens")
    if (
        len(train_tokens) != EXPECTED_TRAIN_CURRENT_COUNT
        or len(selection_tokens) != EXPECTED_SELECTION_CURRENT_COUNT
    ):
        raise ValueError("Frozen temporal current populations have unexpected counts")
    if set(train_tokens) & set(selection_tokens):
        raise ValueError("Temporal current roles overlap")
    current_tokens = train_tokens + selection_tokens
    if len(current_tokens) != EXPECTED_CURRENT_COUNT:
        raise ValueError("Frozen temporal current population differs from its audit")
    if KNOWN_BAD_TOKEN in current_tokens:
        raise ValueError("Known unreadable RGB cannot be admitted as a current frame")
    sequences = sorted({token.split(":", 1)[0] for token in current_tokens})
    rows = pq.read_table(
        media_path,
        columns=[
            "sample_token",
            "split",
            "sequence_id",
            "rgb_shard_path",
            "rgb_member_path",
            "rgb_exposure_start_timestamp_us",
            "rgb_exposure_end_timestamp_us",
        ],
        filters=[("sequence_id", "in", sequences)],
    ).to_pylist()
    rows.sort(
        key=lambda row: (
            str(row["sequence_id"]),
            int(row["rgb_exposure_start_timestamp_us"]),
            str(row["sample_token"]),
        )
    )
    if any(
        row["split"] != "train" or owner.get(str(row["sequence_id"])) not in PANELS_ROLES
        for row in rows
    ):
        raise ValueError("Temporal metadata leaves frozen public-train roles")
    metadata_by_token = {str(row["sample_token"]): cast(dict[str, object], row) for row in rows}
    if any(token not in metadata_by_token for token in current_tokens):
        raise ValueError("Temporal metadata misses a frozen current token")
    index = TemporalRGBIndex(rows)
    canonical_samples = {
        token: index.resolve(token, control="correct_past") for token in current_tokens
    }
    history_tokens = sorted(
        {
            token
            for sample in canonical_samples.values()
            for token, age in zip(sample.history_tokens, sample.history_age_us, strict=True)
            if token is not None and age is not None and age <= args.maximum_history_age_us
        }
    )
    if len(history_tokens) != EXPECTED_UNIQUE_HISTORY_COUNT:
        raise ValueError("Canonical temporal history population differs from its audit")
    final_media = media_admission.get("final_media")
    if not isinstance(final_media, Mapping) or set(final_media) != set(current_tokens):
        raise ValueError("v5 media receipt does not exactly cover current populations")
    for token, decision in final_media.items():
        if not isinstance(decision, Mapping):
            raise ValueError(f"Current-frame media decision is malformed: {token}")
        member_sha256 = decision.get("member_sha256")
        member_size = decision.get("member_size_bytes")
        if (
            not isinstance(member_sha256, str)
            or len(member_sha256) != 64
            or isinstance(member_size, bool)
            or not isinstance(member_size, int)
            or member_size <= 0
        ):
            raise ValueError(f"Current-frame media was not strictly admitted: {token}")
    shard_receipts = _load_jsonl_by(args.preflight_shard_ledger, "rgb_shard_path")
    pinned_files = pinned.get("files")
    if not isinstance(pinned_files, Mapping):
        raise ValueError("Pinned inventory lacks files mapping")
    bad_rows = preflight_summary.get("bad_tokens")
    if (
        not isinstance(bad_rows, list)
        or len(bad_rows) != 1
        or bad_rows[0].get("sample_token") != KNOWN_BAD_TOKEN
    ):
        raise ValueError("Frozen RGB preflight anomaly differs from the known single failure")
    known_bad_receipt = cast(dict[str, object], bad_rows[0])
    media_by_token, negatives, shard_bindings = admit_history_media(
        data_root=args.data_root,
        metadata_by_token=metadata_by_token,
        history_tokens=history_tokens,
        current_media=cast(Mapping[str, Mapping[str, object]], final_media),
        shard_receipts=shard_receipts,
        pinned_files=pinned_files,
        known_bad_receipt=known_bad_receipt,
    )
    new_history_tokens = [token for token in history_tokens if token not in final_media]
    if len(new_history_tokens) != EXPECTED_NEW_HISTORY_COUNT:
        raise ValueError("New temporal history population differs from its audit")
    if len(shard_bindings) != EXPECTED_NEW_HISTORY_SHARD_COUNT:
        raise ValueError("New temporal history shard population differs from its audit")
    if any(
        owner.get(str(metadata_by_token[token]["sequence_id"])) != "train"
        for token in new_history_tokens
    ):
        raise ValueError("New temporal history unexpectedly leaves the train sequence role")
    roles: dict[str, object] = {}
    aggregate_counts: Counter[str] = Counter()
    for role, tokens in (("train", train_tokens), ("selection", selection_tokens)):
        entries, counts = build_temporal_entries(
            index=index,
            current_tokens=tokens,
            role=role,
            sequence_owner=owner,
            media_by_token=media_by_token,
            maximum_history_age_us=args.maximum_history_age_us,
        )
        roles[role] = {
            "current_token_count": len(tokens),
            "current_token_population_sha256": canonical_sha256(sorted(tokens)),
            "counts": dict(sorted(counts.items())),
            "entries": entries,
        }
        aggregate_counts.update(counts)
    if (
        aggregate_counts["canonical_valid_slot_0"],
        aggregate_counts["canonical_valid_slot_1"],
    ) != EXPECTED_CANONICAL_SLOT_COUNTS:
        raise ValueError("Canonical history slot support differs from its audit")
    if aggregate_counts["masked_maximum_age"] != 0:
        raise ValueError("Unexpected maximum-age rejection in the frozen population")
    for role, expected in EXPECTED_ROLE_SUPPORT.items():
        role_counts = cast(Mapping[str, object], cast(Mapping[str, object], roles[role])["counts"])
        if any(int(cast(int, role_counts.get(key, 0))) != value for key, value in expected.items()):
            raise ValueError(f"Temporal support differs from the frozen {role} audit")
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "revision": "temporal_d4096_v5_admission_v2",
        "status": "PASS_WITH_EXPLICIT_HISTORY_MASKS" if negatives else "PASS",
        "source_media_sha256": source_media_sha256,
        "d4096_v5_manifest_sha256": sha256_file(args.v5_manifest),
        "full_selection_sha256": sha256_file(args.full_selection),
        "candidate_split_sha256": sha256_file(args.split),
        "pinned_inventory_sha256": sha256_file(args.pinned_manifest),
        "maximum_history_age_us": args.maximum_history_age_us,
        "history_rule": (
            "canonical previous two same-sequence frames with exposure_end <= current_start; "
            "then fixed age/media masking without alternative backfill"
        ),
        "availability_scope": "sensor exposure timestamps only; arrival unavailable",
        "known_bad_policy": {
            "sample_token": KNOWN_BAD_TOKEN,
            "receipt_sha256": canonical_sha256(known_bad_receipt),
            "action": "shared unavailable-history mask; no substitute",
        },
        "roles": roles,
        "aggregate_counts": dict(sorted(aggregate_counts.items())),
        "scope": {
            "labels_read": False,
            "events_h5_opened": False,
            "gpu_used": False,
            "test_opened": False,
            "current_cohort_changed": False,
        },
    }
    media_receipt = {
        "schema": "eap_temporal_history_media_receipt_v1",
        "unique_canonical_history_tokens_within_age": len(history_tokens),
        "reused_v5_current_receipts": sum(token in final_media for token in history_tokens),
        "new_history_receipts": sum(token not in final_media for token in history_tokens),
        "history_media": {token: media_by_token[token] for token in sorted(media_by_token)},
        "shard_bindings": shard_bindings,
        "source_shard_ledger_sha256": sha256_file(args.preflight_shard_ledger),
    }
    negative_audit = {
        "schema": "eap_temporal_history_negative_audit_v1",
        "negative_count": len(negatives),
        "rows": negatives,
        "all_failures_retained": True,
        "substitution_permitted": False,
    }
    args.output_dir.mkdir(parents=True)
    receipt_path = args.output_dir / "temporal_media_admission.json"
    media_path_out = args.output_dir / "history_media_receipt.json"
    negative_path = args.output_dir / "negative_audit.json"
    write_new(receipt_path, receipt)
    write_new(media_path_out, media_receipt)
    write_new(negative_path, negative_audit)
    provenance = {
        "schema": "eap_temporal_media_admission_provenance_v1",
        "implementation_sha256": sha256_file(Path(__file__)),
        "temporal_data_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src/e_jepa_ttc/detection/temporal_data.py"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, (path, _) in expected_identities.items()
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (receipt_path, media_path_out, negative_path)
        },
        "peak_memory_contract": "stream one TAR member; no image cache; target <1GiB",
    }
    write_new(args.output_dir / "provenance.json", provenance)
    summary = {
        "schema": "eap_temporal_media_admission_summary_v1",
        "status": receipt["status"],
        "train_current_tokens": len(train_tokens),
        "selection_current_tokens": len(selection_tokens),
        "unique_history_tokens": len(history_tokens),
        "negative_count": len(negatives),
        "aggregate_counts": receipt["aggregate_counts"],
        "temporal_media_admission_sha256": sha256_file(receipt_path),
        "history_media_receipt_sha256": sha256_file(media_path_out),
        "negative_audit_sha256": sha256_file(negative_path),
    }
    write_new(args.output_dir / "summary.json", summary)
    return summary


PANELS_ROLES = {"train", "selection"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--v5-manifest", type=Path, required=True)
    parser.add_argument("--v5-receipt", type=Path, required=True)
    parser.add_argument("--v5-media-admission", type=Path, required=True)
    parser.add_argument("--full-selection", type=Path, required=True)
    parser.add_argument("--preflight-summary", type=Path, required=True)
    parser.add_argument("--preflight-shard-ledger", type=Path, required=True)
    parser.add_argument("--pinned-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--maximum-history-age-us", type=int, default=DEFAULT_MAXIMUM_HISTORY_AGE_US
    )
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
