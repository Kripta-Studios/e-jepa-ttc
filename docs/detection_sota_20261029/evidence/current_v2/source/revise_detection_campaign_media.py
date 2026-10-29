"""Create a minimal D4096 revision replacing strictly unreadable RGB media."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageFile

from scripts.preflight_detection_rgb_assets import canonical_sha256, read_jsonl, sha256_file

EXPECTED_WH = (1280, 720)
RARE_CLASSES = ("bicycle", "other_vehicle")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_frame_records(
    path: Path, *, selected_tokens: set[str], replacement_sequence: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    candidates = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            token = str(row["sample_token"])
            if token in selected_tokens:
                selected[token] = row
            if (
                row["role"] == "train"
                and row["sequence_id"] == replacement_sequence
                and not bool(row["trainer_overflow"])
            ):
                candidates.append(row)
    if set(selected) != selected_tokens:
        raise ValueError("Frame ledger does not cover the original campaign")
    return selected, candidates


def _coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frame_count": len(records),
        "raw_negative_frames": sum(not row["raw_counts"] for row in records),
        "trainer_overflow_frames": sum(bool(row["trainer_overflow"]) for row in records),
    }
    mappings = {
        "raw_counts": "raw_annotations",
        "evaluable_counts": "evaluable_annotations_v2",
        "trainable_counts": "trainable_annotations_legacy_profile",
    }
    for source, target in mappings.items():
        counts: Counter[str] = Counter()
        for row in records:
            counts.update({str(key): int(value) for key, value in row[source].items()})
        result[target] = dict(sorted(counts.items()))
    identities: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in records:
        for category, track_ids in row["trainable_identities"].items():
            identities[str(category)].update(
                (str(row["sequence_id"]), str(track_id)) for track_id in track_ids
            )
    result["trainable_identity_count"] = {
        category: len(values) for category, values in sorted(identities.items())
    }
    result["trainable_identities"] = {
        category: [list(value) for value in sorted(values)]
        for category, values in sorted(identities.items())
    }
    return result


def _media_rows(media_path: Path, tokens: list[str]) -> dict[str, dict[str, Any]]:
    table = pq.read_table(
        media_path,
        columns=["sample_token", "sequence_id", "rgb_shard_path", "rgb_member_path"],
        filters=[("sample_token", "in", tokens)],
    )
    rows = {str(row["sample_token"]): row for row in table.to_pylist()}
    if set(rows) != set(tokens):
        raise ValueError("Media metadata is missing candidate tokens")
    return rows


def _admit_candidate(
    *, root: Path, row: dict[str, Any], pinned: dict[str, Any], tar_hash_cache: dict[str, str]
) -> dict[str, Any]:
    shard_relative = str(row["rgb_shard_path"])
    member_relative = str(row["rgb_member_path"])
    shard = (root / shard_relative).resolve()
    if not shard.is_relative_to(root) or not shard.is_file():
        raise ValueError("Replacement shard is absent or escapes root")
    expected = pinned.get(shard_relative)
    if not isinstance(expected, dict):
        raise ValueError("Replacement shard lacks pinned inventory entry")
    if shard_relative not in tar_hash_cache:
        tar_hash_cache[shard_relative] = sha256_file(shard)
    observed_tar_sha = tar_hash_cache[shard_relative]
    if shard.stat().st_size != int(expected["size_bytes"]) or observed_tar_sha != str(
        expected["sha256"]
    ):
        raise ValueError("Replacement shard differs from pinned inventory")
    with tarfile.open(shard, "r:") as archive:
        member = archive.getmember(member_relative)
        if not member.isfile() or not 0 < member.size <= 32 * 1024 * 1024:
            raise ValueError("Replacement member is not a bounded regular file")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Replacement member cannot be extracted")
        with stream:
            raw = stream.read()
    if len(raw) != member.size:
        raise ValueError("Replacement member byte count differs from TAR header")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        actual_wh = image.size
    if actual_wh != EXPECTED_WH:
        raise ValueError(f"Replacement RGB resolution is {actual_wh}")
    return {
        "sample_token": str(row["sample_token"]),
        "sequence_id": str(row["sequence_id"]),
        "rgb_shard_path": shard_relative,
        "rgb_member_path": member_relative,
        "member_size_bytes": len(raw),
        "member_sha256": hashlib.sha256(raw).hexdigest(),
        "image_wh": list(actual_wh),
        "tar_size_bytes": shard.stat().st_size,
        "tar_sha256": observed_tar_sha,
        "tar_expected_sha256": str(expected["sha256"]),
        "strict_decode": True,
    }


def revise(  # noqa: PLR0913, PLR0915
    *,
    root: Path,
    old_manifest_path: Path,
    frame_ledger_path: Path,
    old_population_audit_path: Path,
    old_coverage_path: Path,
    full_selection_path: Path,
    rgb_preflight_dir: Path,
    media_path: Path,
    pinned_inventory_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build one readable campaign revision without changing population size."""
    if output_dir.exists():
        raise FileExistsError("Readable campaign revision output already exists")
    output_dir.mkdir(parents=True)
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    train_tokens = list(old_manifest["roles"]["train"]["sample_tokens"])
    selection_tokens = list(old_manifest["roles"]["selection"]["sample_tokens"])
    full_selection = json.loads(full_selection_path.read_text(encoding="utf-8"))
    full_selection_tokens = list(full_selection["sample_tokens"])
    token_records = read_jsonl(rgb_preflight_dir / "token_status.jsonl")
    failures = [record for record in token_records if record["status"] == "FAIL"]
    if len(failures) != 1 or failures[0]["role"] != "train":
        raise ValueError("Revision requires exactly one known train media failure")
    bad_token = str(failures[0]["sample_token"])
    if bad_token not in train_tokens:
        raise ValueError("Known bad token is absent from D4096 train")
    if any(
        record["status"] != "PASS"
        for record in token_records
        if record["role"] == "selection"
    ):
        raise ValueError("Full selection contains unreadable media")
    old_index = train_tokens.index(bad_token)
    selected_records, candidates = _load_frame_records(
        frame_ledger_path,
        selected_tokens=set(train_tokens + selection_tokens),
        replacement_sequence=str(failures[0]["sequence_id"]),
    )
    bad_exposure = int(selected_records[bad_token]["exposure_start_us"])
    train_set = set(train_tokens)
    unused = [row for row in candidates if row["sample_token"] not in train_set]
    unused.sort(
        key=lambda row: (
            abs(int(row["exposure_start_us"]) - bad_exposure),
            int(row["exposure_start_us"]) > bad_exposure,
            int(row["exposure_start_us"]),
            str(row["sample_token"]),
        )
    )
    candidate_media = _media_rows(media_path, [str(row["sample_token"]) for row in unused])
    pinned = json.loads(pinned_inventory_path.read_text(encoding="utf-8"))["files"]
    attempts = []
    admitted: dict[str, Any] | None = None
    replacement_record: dict[str, Any] | None = None
    tar_hash_cache: dict[str, str] = {}
    for rank, candidate in enumerate(unused, 1):
        token = str(candidate["sample_token"])
        distance = abs(int(candidate["exposure_start_us"]) - bad_exposure)
        try:
            evidence = _admit_candidate(
                root=root,
                row=candidate_media[token],
                pinned=pinned,
                tar_hash_cache=tar_hash_cache,
            )
        except Exception as exc:  # noqa: BLE001 - preserve rejected candidates
            attempts.append(
                {
                    "rank": rank,
                    "sample_token": token,
                    "distance_us": distance,
                    "status": "REJECTED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        admitted = evidence
        replacement_record = candidate
        attempts.append(
            {
                "rank": rank,
                "sample_token": token,
                "distance_us": distance,
                "is_earlier": int(candidate["exposure_start_us"]) < bad_exposure,
                "status": "ADMITTED",
            }
        )
        break
    if admitted is None or replacement_record is None:
        raise ValueError("No strictly readable same-sequence replacement exists")
    new_train_tokens = train_tokens.copy()
    new_train_tokens[old_index] = str(admitted["sample_token"])
    if len(new_train_tokens) != 4096 or len(set(new_train_tokens)) != 4096:
        raise ValueError("Readable train revision is not exactly 4096 unique frames")
    old_train_records = [selected_records[token] for token in train_tokens]
    records_by_token = {**selected_records, str(admitted["sample_token"]): replacement_record}
    new_train_records = [records_by_token[token] for token in new_train_tokens]
    old_train_coverage = _coverage(old_train_records)
    new_train_coverage = _coverage(new_train_records)
    rare_receipt = {}
    for category in RARE_CLASSES:
        old_ids = {
            tuple(value)
            for value in old_train_coverage["trainable_identities"].get(category, [])
        }
        new_ids = {
            tuple(value)
            for value in new_train_coverage["trainable_identities"].get(category, [])
        }
        if not old_ids.issubset(new_ids):
            raise ValueError(f"Replacement loses rare identity coverage: {category}")
        rare_receipt[category] = {
            "old_identity_count": len(old_ids),
            "new_identity_count": len(new_ids),
            "all_original_identities_retained": True,
        }
    pass_records = {
        str(record["sample_token"]): record
        for record in token_records
        if record["status"] == "PASS"
    }
    final_tokens = new_train_tokens + full_selection_tokens
    new_train_set = set(new_train_tokens)
    final_media: dict[str, dict[str, Any]] = {}
    for token in final_tokens:
        if token == admitted["sample_token"]:
            source = admitted
            source_owner = "replacement_probe_v1"
        else:
            source = pass_records[token]
            source_owner = "rgb_full_preflight_v1"
        final_media[token] = {
            "role": "train" if token in new_train_set else "full_selection",
            "sequence_id": source["sequence_id"],
            "rgb_shard_path": source["rgb_shard_path"],
            "rgb_member_path": source["rgb_member_path"],
            "member_size_bytes": source["member_size_bytes"],
            "member_sha256": source["member_sha256"],
            "image_wh": source["image_wh"],
            "source_owner": source_owner,
        }
    if len(final_media) != 28_054:
        raise ValueError("Final media ledger must contain 28,054 unique tokens")
    replacement = {
        "old_index": old_index,
        "old_sample_token": bad_token,
        "old_exposure_start_us": bad_exposure,
        "new": admitted,
        "new_exposure_start_us": int(replacement_record["exposure_start_us"]),
        "selection_rule": (
            "same sequence, unused, trainer_overflow=false, minimum absolute exposure "
            "distance, ties prefer earlier, strict media admission"
        ),
        "attempts": attempts,
    }
    media_admission = {
        "schema": "eap_detection_media_admission_v1",
        "status": "PASS",
        "source_manifest_sha256": sha256_file(old_manifest_path),
        "source_manifest_canonical_sha256": canonical_sha256(old_manifest),
        "source_full_preflight_summary_sha256": sha256_file(
            rgb_preflight_dir / "summary.json"
        ),
        "source_full_preflight_token_ledger_sha256": sha256_file(
            rgb_preflight_dir / "token_status.jsonl"
        ),
        "source_full_preflight_shard_ledger_sha256": sha256_file(
            rgb_preflight_dir / "shard_status.jsonl"
        ),
        "source_frame_ledger_sha256": sha256_file(frame_ledger_path),
        "source_full_selection_sha256": sha256_file(full_selection_path),
        "source_media_sha256": sha256_file(media_path),
        "source_pinned_inventory_sha256": sha256_file(pinned_inventory_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "excluded": failures,
        "replacement": replacement,
        "rare_identity_retention": rare_receipt,
        "new_train_token_population_sha256": canonical_sha256(new_train_tokens),
        "selection_1024_unchanged": selection_tokens
        == list(old_manifest["roles"]["selection"]["sample_tokens"]),
        "full_selection_23958_unchanged": full_selection_tokens
        == list(full_selection["sample_tokens"]),
        "final_counts": {"train": 4096, "full_selection": 23_958, "total": 28_054},
        "final_media": final_media,
    }
    media_admission_path = output_dir / "media_admission.json"
    _write_json(media_admission_path, media_admission)
    new_manifest = copy.deepcopy(old_manifest)
    new_manifest["roles"]["train"]["sample_tokens"] = new_train_tokens
    new_manifest["media_admission"] = {
        "receipt_path": "media_admission.json",
        "receipt_sha256": sha256_file(media_admission_path),
        "source_manifest_sha256": sha256_file(old_manifest_path),
        "train_token_population_sha256": canonical_sha256(new_train_tokens),
    }
    new_manifest["limitations"] = list(new_manifest["limitations"]) + [
        "one native malformed train PNG replaced by a versioned same-sequence readable frame"
    ]
    manifest_path = output_dir / "d4096_manifest.json"
    _write_json(manifest_path, new_manifest)
    selection_records = [selected_records[token] for token in selection_tokens]
    coverage = {
        "schema": "eap_detection_campaign_coverage_v2",
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_canonical_sha256": canonical_sha256(new_manifest),
        "train": new_train_coverage,
        "selection": _coverage(selection_records),
        "train_sequence_frames": dict(
            sorted(Counter(row["sequence_id"] for row in new_train_records).items())
        ),
        "selection_sequence_frames": dict(
            sorted(Counter(row["sequence_id"] for row in selection_records).items())
        ),
        "replacement": replacement,
        "rare_identity_retention": rare_receipt,
    }
    coverage["train"].pop("trainable_identities")
    coverage["selection"].pop("trainable_identities")
    coverage_path = output_dir / "d4096_coverage.json"
    _write_json(coverage_path, coverage)
    copied_audit = output_dir / "population_audit.json"
    shutil.copyfile(old_population_audit_path, copied_audit)
    if sha256_file(copied_audit) != sha256_file(old_population_audit_path):
        raise ValueError("Population audit copy is not byte exact")
    receipt = {
        "schema": "eap_detection_readable_campaign_revision_v1",
        "status": "PASS",
        "old_manifest_sha256": sha256_file(old_manifest_path),
        "new_manifest_sha256": sha256_file(manifest_path),
        "new_manifest_canonical_sha256": canonical_sha256(new_manifest),
        "media_admission_sha256": sha256_file(media_admission_path),
        "coverage_sha256": sha256_file(coverage_path),
        "population_audit_sha256": sha256_file(copied_audit),
        "population_audit_byte_exact": True,
        "old_coverage_sha256": sha256_file(old_coverage_path),
        "full_selection_sha256": sha256_file(full_selection_path),
        "train_frames": len(new_train_tokens),
        "selection_frames": len(selection_tokens),
        "full_selection_frames": len(full_selection_tokens),
        "replacement": {
            "old": bad_token,
            "new": admitted["sample_token"],
            "index": old_index,
        },
    }
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--old-dir", type=Path, default=Path("output/campaign_data/d4096_v4_final")
    )
    parser.add_argument(
        "--full-selection",
        type=Path,
        default=Path("output/campaign_data/full_selection_v1/full_selection_tokens.json"),
    )
    parser.add_argument(
        "--rgb-preflight",
        type=Path,
        default=Path("output/campaign_data/rgb_full_preflight_v1"),
    )
    parser.add_argument("--media", type=Path, default=Path("<LOCAL_PATH>"))
    parser.add_argument(
        "--pinned-inventory",
        type=Path,
        default=Path("data/manifests/eap_detection_public_files_v1.json"),
    )
    args = parser.parse_args()
    old = args.old_dir.resolve()
    receipt = revise(
        root=args.root.resolve(),
        old_manifest_path=old / "d4096_manifest.json",
        frame_ledger_path=old / "frame_population.jsonl",
        old_population_audit_path=old / "population_audit.json",
        old_coverage_path=old / "d4096_coverage.json",
        full_selection_path=args.full_selection.resolve(),
        rgb_preflight_dir=args.rgb_preflight.resolve(),
        media_path=args.media.resolve(),
        pinned_inventory_path=args.pinned_inventory.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
