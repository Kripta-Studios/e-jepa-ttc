"""Strict, resumable RGB preflight for D4096 train and full selection."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq
from PIL import Image, ImageFile

EXPECTED_WH = (1280, 720)
MAX_MEMBER_BYTES = 32 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash one file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value independently of indentation."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a normalized POSIX path below root."""
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or "\\" in relative
        or ":" in relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"Unsafe dataset path: {relative!r}")
    path = root.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes root: {relative!r}")
    return path


def append_jsonl(path: Path, value: object) -> None:
    """Append and flush one durable progress record."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete progress records, rejecting malformed lines."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL record at {path}:{line_number}")
            records.append(value)
    return records


def _load_population(train_manifest: Path, full_selection: Path) -> dict[str, str]:
    train = json.loads(train_manifest.read_text(encoding="utf-8"))
    selection = json.loads(full_selection.read_text(encoding="utf-8"))
    if train.get("schema") != "eap_detection_campaign_manifest_v1":
        raise ValueError("Unexpected D4096 manifest schema")
    if selection.get("schema") != "eap_detection_full_selection_v1":
        raise ValueError("Unexpected full-selection manifest schema")
    train_tokens = list(train["roles"]["train"]["sample_tokens"])
    selection_tokens = list(selection["sample_tokens"])
    combined = train_tokens + selection_tokens
    if len(set(combined)) != len(combined):
        duplicates = sorted(token for token, count in Counter(combined).items() if count > 1)
        raise ValueError(f"Duplicate population tokens: {duplicates[:5]}")
    return {**dict.fromkeys(train_tokens, "train"), **dict.fromkeys(selection_tokens, "selection")}


def _load_rows(media_path: Path, roles: dict[str, str]) -> list[dict[str, str]]:
    columns = ["sample_token", "split", "sequence_id", "rgb_shard_path", "rgb_member_path"]
    table = pq.read_table(
        media_path,
        columns=columns,
        filters=[("sample_token", "in", list(roles))],
    )
    rows = table.to_pylist()
    tokens = [str(row["sample_token"]) for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Media metadata contains duplicate selected tokens")
    if set(tokens) != set(roles):
        missing = sorted(set(roles) - set(tokens))
        raise ValueError(f"Selected tokens absent from media metadata: {missing[:5]}")
    result = []
    member_owners: dict[tuple[str, str], str] = {}
    for row in rows:
        token = str(row["sample_token"])
        if str(row["split"]) != "train":
            raise ValueError(f"Selected token is not in public train metadata: {token}")
        shard = str(row["rgb_shard_path"])
        member = str(row["rgb_member_path"])
        key = (shard, member)
        if key in member_owners:
            raise ValueError(f"Duplicate RGB member ownership: {member_owners[key]} and {token}")
        member_owners[key] = token
        result.append(
            {
                "sample_token": token,
                "role": roles[token],
                "sequence_id": str(row["sequence_id"]),
                "rgb_shard_path": shard,
                "rgb_member_path": member,
            }
        )
    return result


def _decode_member(
    archive: tarfile.TarFile, row: dict[str, str]
) -> dict[str, Any]:
    token = row["sample_token"]
    member_name = row["rgb_member_path"]
    member = archive.getmember(member_name)
    if not member.isfile() or not 0 < member.size <= MAX_MEMBER_BYTES:
        raise ValueError("RGB member is not a bounded regular file")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError("RGB member cannot be extracted")
    digest = hashlib.sha256()
    raw_parts: list[bytes] = []
    observed_size = 0
    with extracted:
        while True:
            block = extracted.read(1024 * 1024)
            if not block:
                break
            observed_size += len(block)
            digest.update(block)
            raw_parts.append(block)
    if observed_size != member.size:
        raise ValueError(f"RGB member is truncated: header={member.size}, read={observed_size}")
    raw = b"".join(raw_parts)
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        actual_wh = image.size
    if actual_wh != EXPECTED_WH:
        raise ValueError(f"RGB resolution is {actual_wh}, expected {EXPECTED_WH}")
    return {
        **row,
        "status": "PASS",
        "member_size_bytes": observed_size,
        "member_sha256": digest.hexdigest(),
        "image_wh": list(actual_wh),
        "strict_truncated_images": True,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "token_echo": token,
    }


def run_preflight(
    *,
    root: Path,
    train_manifest: Path,
    full_selection: Path,
    media_path: Path,
    pinned_inventory: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Audit every selected image and every used TAR, resuming from JSONL."""
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError("Completed preflight output already exists; choose a new directory")
    source_receipt = {
        "schema": "eap_rgb_full_preflight_inputs_v1",
        "root": str(root),
        "train_manifest_sha256": sha256_file(train_manifest),
        "full_selection_sha256": sha256_file(full_selection),
        "media_sha256": sha256_file(media_path),
        "pinned_inventory_sha256": sha256_file(pinned_inventory),
        "implementation_sha256": sha256_file(Path(__file__)),
        "pil_load_truncated_images": False,
        "workers": 1,
        "maximum_simultaneous_tar_readers": 1,
    }
    run_manifest_path = output_dir / "run_manifest.json"
    if run_manifest_path.exists():
        existing = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if existing != source_receipt:
            raise ValueError("Existing progress belongs to different sources or implementation")
    else:
        run_manifest_path.write_text(
            json.dumps(source_receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    roles = _load_population(train_manifest, full_selection)
    rows = _load_rows(media_path, roles)
    pinned = json.loads(pinned_inventory.read_text(encoding="utf-8"))["files"]
    by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_shard[row["rgb_shard_path"]].append(row)
    token_log = output_dir / "token_status.jsonl"
    shard_log = output_dir / "shard_status.jsonl"
    completed_tokens = {str(record["sample_token"]) for record in read_jsonl(token_log)}
    completed_shards = {str(record["rgb_shard_path"]) for record in read_jsonl(shard_log)}
    for shard_relative in sorted(by_shard):
        if shard_relative in completed_shards:
            continue
        shard_rows = sorted(by_shard[shard_relative], key=lambda row: row["sample_token"])
        shard_path = safe_path(root, shard_relative)
        expected = pinned.get(shard_relative)
        shard_record: dict[str, Any] = {
            "rgb_shard_path": shard_relative,
            "requested_token_count": len(shard_rows),
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        token_failures = 0
        shard_failures = 0
        try:
            if not shard_path.is_file() or not isinstance(expected, dict):
                raise FileNotFoundError("Used TAR or pinned inventory entry is absent")
            stat_before = shard_path.stat()
            actual_sha = sha256_file(shard_path)
            shard_record.update(
                {
                    "expected_size_bytes": int(expected["size_bytes"]),
                    "observed_size_bytes": stat_before.st_size,
                    "expected_sha256": str(expected["sha256"]),
                    "observed_sha256": actual_sha,
                    "size_matches_pinned": stat_before.st_size == int(expected["size_bytes"]),
                    "sha256_matches_pinned": actual_sha == str(expected["sha256"]),
                }
            )
            if not shard_record["size_matches_pinned"] or not shard_record[
                "sha256_matches_pinned"
            ]:
                shard_failures += 1
            with tarfile.open(shard_path, "r:") as archive:
                names = archive.getnames()
                duplicate_names = sorted(
                    name for name, count in Counter(names).items() if count > 1
                )
                if duplicate_names:
                    raise ValueError(f"TAR has duplicate member names: {duplicate_names[:5]}")
                for row in shard_rows:
                    if row["sample_token"] in completed_tokens:
                        continue
                    try:
                        token_record = _decode_member(archive, row)
                    except Exception as exc:  # noqa: BLE001 - audit must retain every bad asset
                        token_failures += 1
                        token_record = {
                            **row,
                            "status": "FAIL",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "checked_at_utc": datetime.now(UTC).isoformat(),
                        }
                    append_jsonl(token_log, token_record)
                    completed_tokens.add(row["sample_token"])
            stat_after = shard_path.stat()
            shard_record["stable_during_audit"] = (
                stat_before.st_size == stat_after.st_size
                and stat_before.st_mtime_ns == stat_after.st_mtime_ns
            )
            if not shard_record["stable_during_audit"]:
                shard_failures += 1
        except Exception as exc:  # noqa: BLE001 - persist shard-level corruption
            shard_failures += 1
            shard_record.update({"error_type": type(exc).__name__, "error": str(exc)})
            for row in shard_rows:
                if row["sample_token"] in completed_tokens:
                    continue
                token_failures += 1
                append_jsonl(
                    token_log,
                    {
                        **row,
                        "status": "FAIL",
                        "error_type": "ShardUnavailable",
                        "error": f"{type(exc).__name__}: {exc}",
                        "checked_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
                completed_tokens.add(row["sample_token"])
        shard_record.update(
            {
                "status": "PASS" if token_failures == 0 and shard_failures == 0 else "FAIL",
                "token_failure_count": token_failures,
                "shard_failure_count": shard_failures,
                "finished_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        append_jsonl(shard_log, shard_record)
        completed_shards.add(shard_relative)
    token_records = read_jsonl(token_log)
    shard_records = read_jsonl(shard_log)
    token_counts = Counter(str(record["status"]) for record in token_records)
    shard_counts = Counter(str(record["status"]) for record in shard_records)
    summary = {
        "schema": "eap_rgb_full_preflight_v1",
        "status": (
            "PASS"
            if len(token_records) == len(rows)
            and token_counts["FAIL"] == 0
            and len(shard_records) == len(by_shard)
            and shard_counts["FAIL"] == 0
            else "FAIL"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "population": {
            "train_tokens": sum(role == "train" for role in roles.values()),
            "full_selection_tokens": sum(role == "selection" for role in roles.values()),
            "total_unique_tokens": len(roles),
            "used_shards": len(by_shard),
        },
        "token_status_counts": dict(sorted(token_counts.items())),
        "shard_status_counts": dict(sorted(shard_counts.items())),
        "bad_tokens": [record for record in token_records if record["status"] == "FAIL"],
        "bad_shards": [record for record in shard_records if record["status"] == "FAIL"],
        "progress_files": {
            "run_manifest": str(run_manifest_path),
            "token_status_jsonl": str(token_log),
            "shard_status_jsonl": str(shard_log),
            "token_status_sha256": sha256_file(token_log),
            "shard_status_sha256": sha256_file(shard_log),
        },
        "source_receipt_sha256": canonical_sha256(source_receipt),
        "limitations": [
            (
                "Member SHA-256 values are observed from local TAR bytes; no public "
                "per-member hash exists."
            ),
            "Full TAR SHA-256 is compared with the pinned public inventory.",
            (
                "No labels, annotations, test media, GPU, repair, overwrite, resampling "
                "or manifest mutation."
            ),
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("output/campaign_data/d4096_v4_final/d4096_manifest.json"),
    )
    parser.add_argument(
        "--full-selection",
        type=Path,
        default=Path("output/campaign_data/full_selection_v1/full_selection_tokens.json"),
    )
    parser.add_argument("--media", type=Path, default=Path("<LOCAL_PATH>"))
    parser.add_argument(
        "--pinned-inventory",
        type=Path,
        default=Path("data/manifests/eap_detection_public_files_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_preflight(
        root=args.root,
        train_manifest=args.train_manifest.resolve(),
        full_selection=args.full_selection.resolve(),
        media_path=args.media.resolve(),
        pinned_inventory=args.pinned_inventory.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"status": summary["status"], "population": summary["population"]}))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
