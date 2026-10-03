"""Publish recovered N3 bytes against the already verified essential delivery."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import psutil

from .compact_delivery import coordinator_guard

ROOT = Path(__file__).resolve().parents[2]
NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
BASE_SHA = "6689fca6c5f3323db2f306db20122ac59d081760a68240a35f980e07c630fdf9"


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def idle() -> None:
    for name in ("DRIVER.lock", "ACTIVE_CHILD.json", "execution/WRITER.lock"):
        path = NIGHT / name
        if not path.exists():
            continue
        owner = read(path)
        try:
            process = psutil.Process(owner["pid"])
            if abs(process.create_time() - owner["create_time"]) < 0.1:
                raise RuntimeError(f"live campaign owner: {name}")
        except psutil.NoSuchProcess:
            pass


def main() -> None:
    """Verify fresh physical admission, then persist and stream-check a delta ZIP."""
    idle()
    observed = coordinator_guard(heavy_child=False)
    inventory_path = NIGHT / "ENDPOINT_INVENTORY.json"
    reconciliation_path = NIGHT / "CHECKPOINT_JOURNAL_RECONCILIATION.json"
    inventory = read(inventory_path)
    if inventory["scientific_saved_updates"] <= 47132:
        raise RuntimeError("no newly recovered scientific bytes to deliver")
    if read(reconciliation_path)["status"] != "READ_ONLY_PHYSICAL_CHECKPOINT_ADMISSION":
        raise RuntimeError("fresh physical admission missing")
    # The canonical publisher physically admitted every checkpoint before its
    # publication reservation. Bind exactly those bytes and unchanged journals.
    attempts = sorted((NIGHT / "driver_receipts/N5_DELIVERY").glob("attempt_*.json"))
    latest = read(attempts[-1])
    if digest(attempts[-1].with_suffix(".log")) != latest["log_sha256"]:
        raise RuntimeError("physical admission log changed")
    start = datetime.fromisoformat(latest["started_utc"]).timestamp()
    end = datetime.fromisoformat(latest["ended_utc"]).timestamp()
    if not all(start <= p.stat().st_mtime <= end for p in (inventory_path, reconciliation_path)):
        raise RuntimeError("inventory is not from the latest physical admission")
    for row in inventory["fits"]:
        if row["checkpoint"] and digest(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed after physical admission: {row['id']}")
    journal = read(NIGHT / "execution/PHYSICAL_WORK.json")
    if journal != inventory["journals"]["N2_N3"]:
        raise RuntimeError("work journal changed after physical admission")
    historical = ROOT / "artifacts/simplex_t/h16_replication_20261003/PHYSICAL_WORK.json"
    if read(historical) != inventory["journals"]["N1"]:
        raise RuntimeError("historical H16 journal changed after admission")
    base = read(NIGHT / "FINAL_DELIVERY_COMPACT.json")
    if base["sha256"] != BASE_SHA or digest(Path(base["archive"])) != BASE_SHA:
        raise RuntimeError("verified base archive changed")
    files: dict[str, Path] = {}
    for row in inventory["fits"]:
        if row["family"] == "N3" and row["checkpoint"]:
            folder = Path(row["checkpoint"]).parent
            for path in folder.iterdir():
                if path.is_file():
                    files[path.relative_to(NIGHT).as_posix()] = path
    for name in (
        "ENDPOINT_INVENTORY.json",
        "CHECKPOINT_JOURNAL_RECONCILIATION.json",
        "INFORME_NOCTURNO_SIMPLEX_T.md",
        "NEXT_DECISION_NOCTURNA.json",
        "PROGRESS.json",
        "ACTIVITY.md",
        "WINDOW_AUTHORIZATION.json",
        "PROTOCOL_COST_CONTEXT.json",
        "QUEUE_AUTHORIZED.json",
        "execution/PHYSICAL_WORK.json",
        "execution/UPDATE_PROGRESS.json",
        "execution/ENDPOINT_PROGRESS.json",
        "execution/STATUS.json",
        "execution/RESOURCES.json",
        "COMMIT_RECOVERY_PREFLIGHT.json",
        "compact_delivery/VERIFICATION_019e098e7466.json",
        "compact_delivery/FINAL_NUMERICAL_RECONCILIATION.json",
    ):
        path = NIGHT / name
        if path.exists():
            files[name] = path
    files["provenance/delivery_delta.py"] = Path(__file__).resolve()
    files["receipts/N5_PHYSICAL_ADMISSION.json"] = attempts[-1]
    manifest = {
        "schema": "simplex_t_incremental_recovery_bundle_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "base_archive_sha256": BASE_SHA,
        "base_archive": str(base["archive"]),
        "h16_archive_sha256": "08ee2aae60d56006a73fdb8b249a6467cee0704b0a258bb9ad5659f863b3e348",
        "scientific_saved_updates": inventory["scientific_saved_updates"],
        "complete_endpoints": inventory["endpoints"],
        "optimizer_updates_in_publication": 0,
        "scope": "Apply these recovered N3 files over the verified base delivery. "
        "Full model/optimizer/RNG/sampler state and losses are preserved; N3 has no "
        "evaluation unless its three folds completed. Raw/TRAIN sources remain external.",
        "files": {
            n: {"bytes": p.stat().st_size, "sha256": digest(p)} for n, p in sorted(files.items())
        },
    }
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    # No extraction or backup copy is created. Use a conservative uncompressed
    # bound, allowing ZIP metadata plus durable receipts; never prune old files.
    reserve = sum(v["bytes"] for v in manifest["files"].values()) + len(raw) + 2 * 1024**2
    if observed["own_artifact_bytes"] + reserve > 2 * 1024**3:
        raise RuntimeError("incremental archive exceeds the own-artifact cap")
    output = (
        NIGHT
        / "compact_delivery"
        / f"E_JEPA_TTC_SIMPLEX_T_RECOVERY_{inventory['scientific_saved_updates']}.zip"
    )
    if output.exists():
        raise FileExistsError(output)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("CONTENT_MANIFEST.json", raw)
        for number, (name, path) in enumerate(sorted(files.items()), 1):
            coordinator_guard(heavy_child=False)
            archive.write(path, name)
            if number % 8 == 0:
                assert archive.fp is not None
                archive.fp.flush()
                os.fsync(archive.fp.fileno())
    with output.open("r+b") as stream:
        os.fsync(stream.fileno())
    with ZipFile(output) as archive:
        if len(archive.namelist()) != len(files) + 1:
            raise RuntimeError("unexpected archive membership")
        if archive.read("CONTENT_MANIFEST.json") != raw:
            raise RuntimeError("manifest bytes differ")
        for name, expected in manifest["files"].items():
            with archive.open(name) as stream:
                counter = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    counter.update(chunk)
                actual = counter.hexdigest()
            if actual != expected["sha256"] or archive.getinfo(name).file_size != expected["bytes"]:
                raise RuntimeError(f"archive bytes differ: {name}")
    receipt = dict(
        status="VERIFIED_INCREMENTAL_RECOVERY_DELIVERY",
        archive=str(output),
        sha256=digest(output),
        bytes=output.stat().st_size,
        members=len(files) + 1,
        scientific_saved_updates=inventory["scientific_saved_updates"],
        endpoints=inventory["endpoints"],
        base_archive_sha256=BASE_SHA,
        physical_admission_receipt_sha256=digest(attempts[-1]),
        checkpoint_inventory_sha256=digest(inventory_path),
        optimizer_updates=0,
        no_new_n3_inference_claim=True,
        all_member_hashes_and_crc_verified=True,
    )
    target = NIGHT / "FINAL_DELIVERY_RECOVERY.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".zip.sha256").write_text(
        str(receipt["sha256"]) + "  " + output.name + "\n", encoding="ascii"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
