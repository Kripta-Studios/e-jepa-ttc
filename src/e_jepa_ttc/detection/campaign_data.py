"""Pinned eAP population audit and deterministic detection campaign sampling.

Only the public ``train`` release is admitted.  Label-aware selection is limited
to the training role; selection panels are chosen from timestamps before labels
are read so their natural prevalence is retained.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from e_jepa_ttc.detection.evaluation_ground_truth import build_evaluation_ground_truth
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
from e_jepa_ttc.detection.geometry_contract import original_bundle_corners, validate_calibration

CLASSES = EvaluationProtocol.default().classes
RARE_COVERAGE_CLASSES = ("bicycle", "tricycle", "other_vehicle")
CAMPAIGN_IMAGE_WH = (640, 384)
SOURCE_IMAGE_WH = (1280, 720)


def sha256_file(path: Path) -> str:
    """Hash a file without reading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value using its canonical encoding."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_candidate_split(path: Path) -> dict[str, Any]:
    """Load the frozen 24/8/8 public-train assignment and reject reinterpretation."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "eap_detection_sequence_split_v1":
        raise ValueError("Unexpected candidate split schema")
    if value.get("source_split") != "train":
        raise ValueError("Campaign data must derive only from public train")
    roles = value.get("assignments")
    if not isinstance(roles, dict) or set(roles) != {"train", "selection", "calibration"}:
        raise ValueError("Candidate roles must be exactly train/selection/calibration")
    if [len(roles[name]) for name in ("train", "selection", "calibration")] != [24, 8, 8]:
        raise ValueError("This campaign preserves the reviewed 24/8/8 split")
    owners: dict[str, str] = {}
    for role, sequences in roles.items():
        if not isinstance(sequences, list) or len(sequences) != len(set(sequences)):
            raise ValueError(f"Invalid or duplicate sequences in {role}")
        for sequence in sequences:
            if not isinstance(sequence, str) or not sequence or sequence in owners:
                raise ValueError("Candidate sequence roles overlap or are malformed")
            owners[sequence] = role
    if not re.fullmatch(r"[0-9a-f]{40}", str(value.get("dataset_revision", ""))):
        raise ValueError("Candidate split needs a pinned dataset revision")
    return value


def evenly_spaced(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    """Choose deterministic temporal coverage, including both endpoints when possible."""

    def timestamp(row: Mapping[str, Any]) -> int:
        value = row.get("rgb_exposure_start_timestamp_us", row.get("exposure_start_us"))
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Each row needs an integer exposure-start timestamp")
        return value

    ordered = sorted(rows, key=lambda row: (timestamp(row), str(row["sample_token"])))
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count >= len(ordered):
        return [dict(row) for row in ordered]
    if count == 0:
        return []
    if count == 1:
        return [dict(ordered[len(ordered) // 2])]
    indices = [(index * (len(ordered) - 1)) // (count - 1) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("Temporal selection produced duplicate indices")
    return [dict(ordered[index]) for index in indices]


def _training_reason(annotation: Mapping[str, Any], media: Mapping[str, Any]) -> str | None:
    """Reproduce the current trainer's per-object admission after 640x384 resize."""
    box = np.asarray(annotation["bbox_3d_ego"], dtype=np.float64)
    if box.shape != (7,) or not np.isfinite(box).all() or (box[3:6] <= 0).any():
        raise ValueError("Malformed bbox_3d_ego")
    intrinsic = _resized_intrinsic(np.asarray(media["K_event"], dtype=np.float64))
    transform = np.asarray(media["T_event_ego"], dtype=np.float64)
    validate_calibration(intrinsic, transform)
    corners = original_bundle_corners(box.tolist())
    camera = corners @ transform[:3, :3].T + transform[:3, 3]
    center = transform[:3, :3] @ box[:3] + transform[:3, 3]
    if not 0.1 <= center[2] < 60.0:
        return "center_depth_outside_profile"
    if (camera[:, 2] <= 0.1).any():
        return "corner_behind_or_near_camera"
    projected = camera @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    uv_h = intrinsic @ center
    uv = uv_h[:2] / uv_h[2]
    width, height = CAMPAIGN_IMAGE_WH
    if not (0 <= uv[0] < width and 0 <= uv[1] < height):
        return "projected_center_outside_image"
    minimum = np.maximum(pixels.min(0), [0, 0])
    maximum = np.minimum(pixels.max(0), [width - 1, height - 1])
    if min(maximum - minimum) <= 1:
        return "projected_box_too_small"
    distances = np.array(
        [uv[0] - minimum[0], maximum[0] - uv[0], uv[1] - minimum[1], maximum[1] - uv[1]]
    )
    if (distances < 0).any():
        return "projected_center_outside_clipped_box"
    return None


def _resized_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    """Apply the exact Pillow pixel-center mapping used by ``prepare_sample``."""
    source_w, source_h = SOURCE_IMAGE_WH
    width, height = CAMPAIGN_IMAGE_WH
    sx, sy = width / source_w, height / source_h
    affine = np.asarray([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return affine @ intrinsic


def _summarize_frames(frame_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw, evaluable, trainable = Counter(), Counter(), Counter()
    identities: dict[str, set[tuple[str, str]]] = defaultdict(set)
    overflow = 0
    negatives = 0
    for frame in frame_records:
        raw.update(frame["raw_counts"])
        evaluable.update(frame["evaluable_counts"])
        trainable.update(frame["trainable_counts"])
        overflow += bool(frame["trainer_overflow"])
        negatives += sum(frame["raw_counts"].values()) == 0
        for category, values in frame["trainable_identities"].items():
            identities[category].update((frame["sequence_id"], value) for value in values)
    return {
        "frame_count": len(frame_records),
        "raw_annotations": {name: raw[name] for name in CLASSES},
        "evaluable_annotations_v2": {name: evaluable[name] for name in CLASSES},
        "trainable_annotations_legacy_profile": {name: trainable[name] for name in CLASSES},
        "trainable_identity_count": {name: len(identities[name]) for name in CLASSES},
        "trainer_overflow_frames": overflow,
        "raw_negative_frames": negatives,
    }


def audit_population(
    *, data_root: Path, candidate_split_path: Path, pinned_manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit all forty train sequences and return per-frame records and media rows."""
    split = load_candidate_split(candidate_split_path)
    pinned = json.loads(pinned_manifest_path.read_text(encoding="utf-8"))
    if pinned.get("revision") != split["dataset_revision"]:
        raise ValueError("Pinned file inventory and candidate split revisions differ")
    media_path = data_root / "data/train.parquet"
    if sha256_file(media_path) != split["media_sha256"]:
        raise ValueError("Canonical train metadata SHA-256 differs")
    expected_sequences = {s for values in split["assignments"].values() for s in values}
    protocol = EvaluationProtocol.default()
    records: list[dict[str, Any]] = []
    file_receipts: list[dict[str, Any]] = []
    rgb_paths: set[str] = set()
    seen_tokens: set[str] = set()
    media_columns = [
        "sample_token",
        "split",
        "sequence_id",
        "rgb_shard_path",
        "rgb_member_path",
        "rgb_exposure_start_timestamp_us",
        "K_event",
        "T_event_ego",
    ]
    sample_media: dict[str, dict[str, Any]] = {}
    for sequence in sorted(expected_sequences):
        relative = f"data/train/{sequence}/labels.parquet"
        label_path = data_root / relative
        expected = pinned["files"].get(relative)
        digest = sha256_file(label_path)
        if (
            not expected
            or digest != expected.get("sha256")
            or label_path.stat().st_size != expected.get("size_bytes")
        ):
            raise ValueError(f"Pinned label bytes differ: {sequence}")
        labels = pq.read_table(label_path).to_pylist()
        file_receipts.append(
            {
                "path": relative,
                "bytes": label_path.stat().st_size,
                "sha256": digest,
                "status": "verified",
            }
        )
        media_rows = pq.read_table(
            media_path, columns=media_columns, filters=[("sequence_id", "=", sequence)]
        ).to_pylist()
        if not media_rows:
            raise ValueError(f"Frozen sequence absent from media: {sequence}")
        indexed = {str(row["sample_token"]): row for row in media_rows}
        if len(indexed) != len(media_rows) or any(
            row["split"] != "train" or row["sequence_id"] != sequence for row in media_rows
        ):
            raise ValueError(f"Media identity differs for {sequence}")
        if seen_tokens & set(indexed):
            raise ValueError("Media tokens repeat across sequences")
        seen_tokens.update(indexed)
        rgb_paths.update(str(row["rgb_shard_path"]) for row in media_rows)
        sample_media[sequence] = min(
            media_rows,
            key=lambda row: (int(row["rgb_exposure_start_timestamp_us"]), str(row["sample_token"])),
        )
        labels_by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in labels:
            token = str(row["sample_token"])
            if (
                token not in indexed
                or indexed[token]["sequence_id"] != sequence
                or row["split"] != "train"
            ):
                raise ValueError(f"Label identity leaves its train sequence: {token}")
            labels_by_token[token].append(row)
        role = next(name for name, values in split["assignments"].items() if sequence in values)
        for token, media in indexed.items():
            annotations = labels_by_token.get(token, [])
            if annotations:
                gt = build_evaluation_ground_truth(
                    annotations,
                    intrinsic=media["K_event"],
                    event_from_ego=media["T_event_ego"],
                    image_wh=(1280, 720),
                    protocol=protocol,
                )
                evaluable = Counter(box.class_name for box in gt.eligible)
                evaluation_ignored = dict(gt.ignored_by_reason)
            else:
                evaluable, evaluation_ignored = Counter(), {}
            trainable = Counter()
            trainable_ids: dict[str, set[str]] = defaultdict(set)
            training_ignored = Counter()
            raw = Counter(str(row["category"]) for row in annotations)
            for annotation in annotations:
                reason = _training_reason(annotation, media)
                if reason is None:
                    category = str(annotation["category"])
                    trainable[category] += 1
                    trainable_ids[category].add(str(annotation["track_id"]))
                else:
                    training_ignored[reason] += 1
            records.append(
                {
                    "sample_token": token,
                    "sequence_id": sequence,
                    "role": role,
                    "exposure_start_us": int(media["rgb_exposure_start_timestamp_us"]),
                    "raw_counts": dict(raw),
                    "evaluable_counts": dict(evaluable),
                    "trainable_counts": dict(trainable),
                    "trainable_identities": {
                        key: sorted(value) for key, value in trainable_ids.items()
                    },
                    "evaluation_ignored": evaluation_ignored,
                    "training_ignored": dict(training_ignored),
                    "trainer_overflow": sum(trainable.values()) > 50,
                }
            )

    by_role = {
        role: _summarize_frames([r for r in records if r["role"] == role])
        for role in split["assignments"]
    }
    by_sequence = {
        sequence: _summarize_frames([r for r in records if r["sequence_id"] == sequence])
        for sequence in sorted(expected_sequences)
    }
    rgb_receipts = []
    for relative in sorted(rgb_paths):
        path = data_root / relative
        expected = pinned["files"].get(relative)
        present = path.is_file()
        size_matches = bool(
            present and expected and path.stat().st_size == expected.get("size_bytes")
        )
        rgb_receipts.append(
            {
                "path": relative,
                "present": present,
                "size_matches_pinned": size_matches,
                "bytes": path.stat().st_size if present else None,
            }
        )
    if any(not row["present"] or not row["size_matches_pinned"] for row in rgb_receipts):
        raise ValueError(
            "At least one required public-train RGB shard is absent or has a wrong size"
        )
    image_samples = []
    for sequence, row in sorted(sample_media.items()):
        shard_path = data_root / str(row["rgb_shard_path"])
        member_name = str(row["rgb_member_path"])
        with tarfile.open(shard_path, "r:") as archive:
            member = archive.getmember(member_name)
            if not member.isfile() or not 0 < member.size <= 32 * 1024 * 1024:
                raise ValueError(f"Invalid sampled RGB member for {sequence}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read sampled RGB member for {sequence}")
            raw = stream.read()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            actual_wh = image.size
        if actual_wh != SOURCE_IMAGE_WH:
            raise ValueError(f"Sampled RGB shape differs for {sequence}: {actual_wh}")
        image_samples.append(
            {
                "sequence_id": sequence,
                "sample_token": row["sample_token"],
                "image_wh": list(actual_wh),
                "member_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    report = {
        "schema": "eap_detection_population_audit_v2",
        "dataset_revision": split["dataset_revision"],
        "source_split": "train",
        "candidate_split_sha256": sha256_file(candidate_split_path),
        "source_media_sha256": sha256_file(media_path),
        "pinned_inventory_sha256": sha256_file(pinned_manifest_path),
        "evaluation_protocol_sha256": protocol.sha256,
        "split_policy": "reviewed 24/8/8 retained; acquisition linkage remains unverified",
        "image_geometry": {
            "source_wh": list(SOURCE_IMAGE_WH),
            "campaign_wh": list(CAMPAIGN_IMAGE_WH),
            "sampled_one_frame_per_sequence": image_samples,
        },
        "roles": by_role,
        "sequences": by_sequence,
        "label_files": file_receipts,
        "rgb_shards": {
            "count": len(rgb_receipts),
            "present": sum(r["present"] for r in rgb_receipts),
            "size_matches_pinned": sum(r["size_matches_pinned"] for r in rgb_receipts),
            "files": rgb_receipts,
        },
    }
    return report, records, []


def _select_train(records: Sequence[Mapping[str, Any]], budget: int) -> list[str]:
    eligible = [r for r in records if r["role"] == "train" and not r["trainer_overflow"]]
    sequences = sorted({str(r["sequence_id"]) for r in eligible})
    if budget < len(sequences) or budget > len(eligible):
        raise ValueError("Training budget cannot cover the requested population")
    by_sequence = {
        sequence: [r for r in eligible if r["sequence_id"] == sequence] for sequence in sequences
    }
    required: set[str] = set()
    for category in CLASSES:
        category_rows = [r for r in eligible if r["trainable_counts"].get(category, 0)]
        if not category_rows:
            raise ValueError(f"Frozen train role has no trainable positive for {category}")
        required.add(str(evenly_spaced(category_rows, 1)[0]["sample_token"]))
    for sequence in sequences:
        for category in RARE_COVERAGE_CLASSES:
            identities: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in by_sequence[sequence]:
                for identity in row["trainable_identities"].get(category, []):
                    identities[identity].append(row)
            for rows in identities.values():
                required.add(str(evenly_spaced(rows, 1)[0]["sample_token"]))
    if len(required) > budget:
        raise ValueError("Rare-identity coverage exceeds D4096 budget")
    targets = {sequence: budget // len(sequences) for sequence in sequences}
    for sequence in sequences[: budget % len(sequences)]:
        targets[sequence] += 1
    token_owners: dict[str, str] = {}
    for row in eligible:
        token, sequence = str(row["sample_token"]), str(row["sequence_id"])
        if token in token_owners and token_owners[token] != sequence:
            raise ValueError(f"Sample token has multiple sequence owners: {token}")
        token_owners[token] = sequence
    required_by_sequence = {
        sequence: {token for token in required if token_owners[token] == sequence}
        for sequence in sequences
    }
    for sequence in sequences:
        targets[sequence] = max(targets[sequence], len(required_by_sequence[sequence]))
    while sum(targets.values()) > budget:
        choices = [
            sequence
            for sequence in sequences
            if targets[sequence] > max(1, len(required_by_sequence[sequence]))
        ]
        if not choices:
            raise ValueError("Required coverage exceeds balanced D4096 budget")
        sequence = max(choices, key=lambda item: (targets[item], item))
        targets[sequence] -= 1

    selected: set[str] = set()
    for sequence in sequences:
        ordered = sorted(
            by_sequence[sequence],
            key=lambda row: (
                int(row.get("exposure_start_us", row.get("rgb_exposure_start_timestamp_us"))),
                str(row["sample_token"]),
            ),
        )
        chosen = {str(row["sample_token"]) for row in evenly_spaced(ordered, targets[sequence])}
        chosen.update(required_by_sequence[sequence])
        positions = {str(row["sample_token"]): index for index, row in enumerate(ordered)}
        # Remove only redundant non-required points.  The closest-neighbour rule
        # preserves broad temporal span and needs at most one pass per rare frame.
        while len(chosen) > targets[sequence]:
            removable = chosen - required_by_sequence[sequence]
            if not removable:
                raise RuntimeError("Cannot meet sequence quota without dropping required coverage")

            chosen.remove(
                min(
                    removable,
                    key=lambda token: _redundancy_key(
                        token, chosen=chosen, positions=positions, length=len(ordered)
                    ),
                )
            )
        selected.update(chosen)
    if len(selected) != budget:
        raise RuntimeError("Balanced temporal selection did not meet the exact budget")
    return sorted(selected)


def _redundancy_key(
    token: str, *, chosen: set[str], positions: Mapping[str, int], length: int
) -> tuple[int, int, str]:
    others = [positions[value] for value in chosen if value != token]
    distance = min(abs(positions[token] - value) for value in others)
    endpoint = int(positions[token] in (0, length - 1))
    return (distance, endpoint, token)


def build_campaign_manifest(
    *,
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    pilot_manifest: Mapping[str, Any],
    train_budget: int = 4096,
    selection_budget: int = 1024,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build D4096 and a label-blind natural-prevalence selection panel."""
    train_tokens = _select_train(records, train_budget)
    selection_rows = [r for r in records if r["role"] == "selection"]
    sequences = sorted(split["assignments"]["selection"])
    per_sequence = selection_budget // len(sequences)
    remainder = selection_budget % len(sequences)
    selection_tokens: list[str] = []
    for index, sequence in enumerate(sequences):
        rows = [r for r in selection_rows if r["sequence_id"] == sequence]
        selection_tokens.extend(
            str(r["sample_token"]) for r in evenly_spaced(rows, per_sequence + (index < remainder))
        )
    pilot_selection = list(pilot_manifest["selection"]["sample_tokens"])
    manifest = {
        "schema": "eap_detection_campaign_manifest_v1",
        "dataset_revision": report["dataset_revision"],
        "source_split": "train",
        "source_media_sha256": report["source_media_sha256"],
        "candidate_split_sha256": report["candidate_split_sha256"],
        "population_audit_sha256": canonical_sha256(report),
        "sampling_seed": 7,
        "roles": {
            "train": {
                "sample_tokens": train_tokens,
                "sequence_ids": sorted(split["assignments"]["train"]),
            },
            "selection": {"sample_tokens": sorted(selection_tokens), "sequence_ids": sequences},
        },
        "historical_panels": {
            "pilot_selection256": {
                "sample_tokens": pilot_selection,
                "source": "frozen pilot manifest; development-exposed",
            }
        },
        "sampling": {
            "train": (
                "unique non-overflow frames; one representative per trainable rare "
                "(sequence,track_id), then temporal/sequence balancing"
            ),
            "selection": (
                "label-blind, evenly spaced by exposure time per frozen selection sequence; "
                "natural prevalence"
            ),
            "calibration": (
                "excluded from campaign train and selection; audited only as a frozen "
                "population role"
            ),
        },
        "limitations": [
            "sequence_id grouping only; cross-sequence acquisition linkage unverified",
            "selection is development-exposed, not final audit",
            "no public test labels read",
        ],
    }
    selected_records = [r for r in records if r["sample_token"] in set(train_tokens)]
    selection_records = [r for r in records if r["sample_token"] in set(selection_tokens)]
    coverage = {
        "schema": "eap_detection_campaign_coverage_v1",
        "manifest_sha256": canonical_sha256(manifest),
        "train": _summarize_frames(selected_records),
        "selection": _summarize_frames(selection_records),
        "train_sequence_frames": dict(Counter(r["sequence_id"] for r in selected_records)),
        "selection_sequence_frames": dict(Counter(r["sequence_id"] for r in selection_records)),
    }
    if any(
        coverage["train"]["trainable_annotations_legacy_profile"][name] == 0 for name in CLASSES
    ):
        raise ValueError("D4096 lacks a trainable positive for at least one class")
    if set(train_tokens) & set(selection_tokens) or len(train_tokens) != len(set(train_tokens)):
        raise RuntimeError("Campaign token roles overlap or repeat")
    return manifest, coverage


def build_full_selection_manifest(*, data_root: Path, candidate_split_path: Path) -> dict[str, Any]:
    """List every frame in the frozen eight-sequence selection role, label-blind."""
    split = load_candidate_split(candidate_split_path)
    media_path = data_root / "data/train.parquet"
    if sha256_file(media_path) != split["media_sha256"]:
        raise ValueError("Canonical train metadata SHA-256 differs")
    sequences = sorted(split["assignments"]["selection"])
    rows = pq.read_table(
        media_path,
        columns=["sample_token", "split", "sequence_id", "rgb_exposure_start_timestamp_us"],
        filters=[("sequence_id", "in", sequences)],
    ).to_pylist()
    if any(row["split"] != "train" or row["sequence_id"] not in sequences for row in rows):
        raise ValueError("Full selection leaves its frozen public-train role")
    tokens = [
        str(row["sample_token"])
        for row in sorted(
            rows,
            key=lambda row: (
                str(row["sequence_id"]),
                int(row["rgb_exposure_start_timestamp_us"]),
                str(row["sample_token"]),
            ),
        )
    ]
    if len(tokens) != len(set(tokens)) or {token.split(":", 1)[0] for token in tokens} != set(
        sequences
    ):
        raise ValueError("Full selection token ownership is invalid")
    return {
        "schema": "eap_detection_full_selection_v1",
        "dataset_revision": split["dataset_revision"],
        "source_split": "train",
        "source_media_sha256": split["media_sha256"],
        "candidate_split_sha256": sha256_file(candidate_split_path),
        "role": "selection",
        "sequence_ids": sequences,
        "sample_tokens": tokens,
        "selection_rule": (
            "all frames from the frozen eight selection sequences; no label or prediction selection"
        ),
        "use": (
            "full development evaluation at fixed campaign checkpoints; not an independent "
            "final audit"
        ),
    }
