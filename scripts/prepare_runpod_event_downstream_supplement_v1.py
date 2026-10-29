"""Pack the exact 13-file EVENT downstream metadata supplement."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.runpod_migration.bundle import safe_member
from operational.runpod_migration.inventory import digest, read


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"Refusing to replace immutable supplement record: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")


def prepare(coverage_path: Path, inventory_path: Path, output: Path) -> dict[str, Any]:
    coverage_path = coverage_path.resolve(strict=True)
    inventory_path = inventory_path.resolve(strict=True)
    coverage, inventory = read(coverage_path), read(inventory_path)
    missing = coverage.get("missing_required")
    if (
        coverage.get("schema") != "ejepa_event_downstream_coverage_plan_v1"
        or coverage.get("status") != "MISSING_REQUIRED_METADATA"
        or coverage.get("missing_bytes") != 81_182_038
        or not isinstance(missing, dict)
        or len(missing) != 13
        or coverage.get("source_inventory_sha256") != digest(inventory_path)
    ):
        raise ValueError("Coverage plan is not the reviewed 13-file EVENT closure")
    inventory_by_source = {
        str(Path(item["source"]).resolve(strict=True)): item for item in inventory["files"]
    }
    files: list[dict[str, Any]] = []
    for source_name, expected in sorted(missing.items()):
        source = Path(source_name).resolve(strict=True)
        item = inventory_by_source.get(str(source))
        actual = digest(source)
        if (
            item is None
            or int(expected["bytes"]) != source.stat().st_size
            or expected["sha256"] != actual
            or item["sha256"] != actual
        ):
            raise ValueError(f"Supplement source changed: {source}")
        files.append(
            {
                "source": str(source),
                "relative": safe_member(str(item["relative"]).replace("\\", "/")),
                "bytes": source.stat().st_size,
                "sha256": actual,
                "kind": expected["kind"],
            }
        )
    if sum(item["bytes"] for item in files) != 81_182_038:
        raise ValueError("Supplement byte total changed")
    content_identity = _canonical(
        [{key: item[key] for key in ("relative", "bytes", "sha256")} for item in files]
    )
    manifest = {
        "schema": "ejepa_event_downstream_supplement_manifest_v1",
        "status": "COMPLETE_VERIFIED_LOCAL",
        "created_utc": datetime.now(UTC).isoformat(),
        "scientific_config_sha256": coverage["scientific_config_sha256"],
        "coverage_plan_sha256": digest(coverage_path),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "content_identity_sha256": content_identity,
        "prepared_rgb_pixel_shards_included": False,
    }
    manifest["manifest_payload_sha256"] = _canonical(manifest)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "SUPPLEMENT_MANIFEST.json"
    _write_once(manifest_path, manifest)
    archive = output / "EVENT_DOWNSTREAM_SUPPLEMENT.tar"
    if archive.exists():
        raise FileExistsError(archive)
    pending = archive.with_suffix(".tar.partial")
    try:
        with tarfile.open(pending, "w") as target:
            for item in files:
                source = Path(item["source"])
                before = source.stat()
                info = target.gettarinfo(str(source), arcname=item["relative"])
                with source.open("rb") as stream:
                    target.addfile(info, stream)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Supplement source changed during pack: {source}")
            payload = manifest_path.read_bytes()
            info = tarfile.TarInfo("SUPPLEMENT_MANIFEST.json")
            info.size = len(payload)
            target.addfile(info, io.BytesIO(payload))
        os.replace(pending, archive)
    finally:
        pending.unlink(missing_ok=True)
    with tarfile.open(archive, "r") as packed:
        for item in files:
            stream = packed.extractfile(item["relative"])
            checksum = hashlib.sha256()
            if stream is None:
                raise ValueError(f"Packed supplement member missing: {item['relative']}")
            with stream:
                while block := stream.read(8 * 1024**2):
                    checksum.update(block)
            if checksum.hexdigest() != item["sha256"]:
                raise ValueError(f"Packed supplement member changed: {item['relative']}")
    archive_receipt = {
        "schema": "ejepa_event_downstream_supplement_archive_v1",
        "status": "COMPLETE_VERIFIED_LOCAL",
        "archive": str(archive.resolve()),
        "archive_sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "manifest_sha256": digest(manifest_path),
        "content_identity_sha256": content_identity,
        "files": 13,
        "payload_bytes": 81_182_038,
    }
    archive_receipt["receipt_sha256"] = _canonical(archive_receipt)
    receipt_path = output / "EVENT_DOWNSTREAM_SUPPLEMENT.tar.json"
    _write_once(receipt_path, archive_receipt)
    seal = {
        "schema": "ejepa_event_supplement_archive_seal_v1",
        "status": "SEALED_FOR_TRANSFER",
        "archive_sha256": archive_receipt["archive_sha256"],
        "archive_bytes": archive_receipt["archive_bytes"],
        "manifest_sha256": archive_receipt["manifest_sha256"],
        "content_identity_sha256": content_identity,
        "source_files_rehashed_before_pack": 13,
        "packed_members_rehashed_after_pack": 13,
        "remote_filesystem_identity_claimed": False,
    }
    seal["seal_payload_sha256"] = _canonical(seal)
    _write_once(output / "EVENT_DOWNSTREAM_SUPPLEMENT.ARCHIVE_SEAL.json", seal)
    return {"manifest": manifest, "archive": archive_receipt, "seal": seal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.coverage, args.inventory, args.output), indent=2))


if __name__ == "__main__":
    main()
