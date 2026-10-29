"""Verify or resumably restore the exact 41-file EVENT raw manifest from HF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

REPO = "NAIL-HNU/eAP-dataset"
REVISION = "a814a946e60d37f7242358af5f344542177e50c5"
EXPECTED_FILES = 41
EXPECTED_BYTES = 449_003_803_674
MIN_FINAL_RESERVE = 50 * 1024**3


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"Refusing to replace immutable HF receipt: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")


def _rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in manifest["files"]:
        relative = (
            PureWindowsPath(item["source"])
            .relative_to(PureWindowsPath("E:/eAP_dataset"))
            .as_posix()
        )
        rows.append(
            {
                "relative": relative,
                "bytes": int(item["bytes"]),
                "sha256": str(item["sha256"]),
            }
        )
    if len(rows) != EXPECTED_FILES or sum(row["bytes"] for row in rows) != EXPECTED_BYTES:
        raise ValueError("Raw manifest population or bytes changed")
    return rows


def verify_remote(manifest_path: Path, output: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi

    manifest_path = manifest_path.resolve(strict=True)
    manifest = _read(manifest_path)
    rows = _rows(manifest)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    for row in rows:
        item = remote.get(row["relative"])
        if (
            item is None
            or item.size != row["bytes"]
            or item.lfs is None
            or item.lfs.sha256 != row["sha256"]
        ):
            raise ValueError(f"HF byte identity differs: {row['relative']}")
    result = {
        "schema": "ejepa_event_hf_remote_identity_v1",
        "status": "PASS",
        "repository": REPO,
        "revision": REVISION,
        "observed_revision": info.sha,
        "raw_deferred_manifest_sha256": _digest(manifest_path),
        "content_identity_sha256": manifest["content_identity_sha256"],
        "files": EXPECTED_FILES,
        "total_bytes": EXPECTED_BYTES,
        "exact_size_and_lfs_sha256_matches": EXPECTED_FILES,
        "download_executed": False,
        "authentication_secret_required": False,
        "verified_utc": datetime.now(UTC).isoformat(),
    }
    result["receipt_sha256"] = _canonical(result)
    _write_once(output, result)
    return result


def restore(manifest_path: Path, remote_receipt: Path, root: Path) -> dict[str, Any]:
    manifest_path, remote_receipt = map(Path.resolve, (manifest_path, remote_receipt))
    manifest, remote = _read(manifest_path), _read(remote_receipt)
    rows = sorted(_rows(manifest), key=lambda item: item["bytes"], reverse=True)
    if (
        remote.get("schema") != "ejepa_event_hf_remote_identity_v1"
        or remote.get("status") != "PASS"
        or remote.get("revision") != REVISION
        or remote.get("raw_deferred_manifest_sha256") != _digest(manifest_path)
        or remote.get("receipt_sha256")
        != _canonical({key: value for key, value in remote.items() if key != "receipt_sha256"})
    ):
        raise ValueError("Pinned HF remote identity receipt is absent or changed")
    root = root.resolve()
    if not root.is_relative_to(Path("/workspace")):
        raise ValueError("EVENT raw restoration must stay on the /workspace volume")
    root.mkdir(parents=True, exist_ok=True)
    cli = shutil.which("hf")
    if cli is None:
        raise FileNotFoundError("The existing hf CLI is required")
    remaining = sum(
        row["bytes"]
        for row in rows
        if not (root / row["relative"]).is_file()
        or (root / row["relative"]).stat().st_size != row["bytes"]
    )
    largest = max((row["bytes"] for row in rows), default=0)
    volume = _read(Path("/workspace/ejepa_event_runs/EVENT_VOLUME_SNAPSHOT.json"))
    age = time.time() - float(volume["observed_timestamp"])
    if not (
        volume.get("schema") == "ejepa_event_volume_snapshot_v1"
        and volume.get("quota_bytes") == 600_000_000_000
        and 0 <= int(volume.get("used_bytes", -1)) <= 600_000_000_000
        and 0 <= age <= 120
    ):
        raise RuntimeError("Fresh 600 GB event-volume snapshot is required")
    declared_free = 600_000_000_000 - int(volume["used_bytes"])
    usable_free = min(shutil.disk_usage(root).free, declared_free)
    if usable_free - remaining - largest < MIN_FINAL_RESERVE:
        raise RuntimeError("HF restore cannot preserve reserve plus one largest in-flight file")
    env = dict(os.environ)
    env.update(
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        HF_HUB_DISABLE_UPDATE_CHECK="1",
        HF_XET_HIGH_PERFORMANCE="0",
        HF_XET_CHUNK_CACHE_SIZE_BYTES="0",
        HF_XET_SHARD_CACHE_SIZE_LIMIT="0",
        HF_HOME=str(root / ".hf_control"),
    )
    receipts = root / ".event_raw_receipts"
    for row in rows:
        destination = root / row["relative"]
        receipt_path = receipts / (row["relative"].replace("/", "__") + ".json")
        if destination.is_file() and destination.stat().st_size == row["bytes"]:
            if _digest(destination) == row["sha256"]:
                continue
            raise ValueError(f"Existing raw file has the wrong SHA-256: {destination}")
        subprocess.run(
            [
                cli,
                "download",
                REPO,
                row["relative"],
                "--type",
                "dataset",
                "--revision",
                REVISION,
                "--local-dir",
                str(root),
                "--max-workers",
                "1",
                "--quiet",
            ],
            check=True,
            env=env,
        )
        if destination.stat().st_size != row["bytes"] or _digest(destination) != row["sha256"]:
            raise ValueError(f"Downloaded raw file differs: {destination}")
        _write_once(
            receipt_path,
            {
                "relative": row["relative"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "revision": REVISION,
                "mtime_ns": destination.stat().st_mtime_ns,
            },
        )
    coverage = {
        "schema": "ejepa_event_raw_coverage_v1",
        "status": "VERIFIED",
        "raw_deferred_manifest_sha256": _digest(manifest_path),
        "content_identity_sha256": manifest["content_identity_sha256"],
        "repository": REPO,
        "revision": REVISION,
        "files": EXPECTED_FILES,
        "sequences": 40,
        "total_bytes": EXPECTED_BYTES,
        "hashes_verified": EXPECTED_FILES,
        "availability_verified": True,
    }
    coverage["receipt_sha256"] = _canonical(coverage)
    _write_once(root / "EVENT_RAW_COVERAGE_RECEIPT.json", coverage)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-remote")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--remote-receipt", type=Path, required=True)
    download.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = (
        verify_remote(args.manifest, args.output)
        if args.command == "verify-remote"
        else restore(args.manifest, args.remote_receipt, args.root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
