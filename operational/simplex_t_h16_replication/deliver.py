"""Export independent H16 delivery and verify all six heads from extracted bytes."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from common import (
    OUT,
    ROOT,
    Lease,
    Resources,
    atomic_bytes,
    atomic_json,
    digest,
    protocol,
    publish_json,
    record,
    validate_pins,
)


def main() -> None:
    p, pin = protocol()
    resource = Resources()
    resource.check()
    validate_pins(p, full=True)
    with Lease():
        result = record(OUT / "analysis/RESULTS.json")
        seal = record(OUT / "ENDPOINTS.json")
        if result["protocol_sha256"] != pin or seal["protocol_sha256"] != pin:
            raise ValueError("delivery protocol differs")
        payload = OUT / "essential_delivery"
        payload.mkdir(exist_ok=True)
        selected = [
            OUT / k
            for k in (
                "PROTOCOL.json",
                "PROTOCOL.sha256",
                "PROTOCOL.md",
                "PREFLIGHT.json",
                "STATIC_TRAIN_AUDIT.json",
                "ENDPOINTS.json",
                "ACCOUNTING.json",
                "PHYSICAL_WORK.json",
                "FINAL_REPORT_H16.md",
                "NEXT_DECISION.json",
            )
        ]
        selected += [
            f
            for d in ("analysis", "cached_inputs", "weights", "verification", "publication", "fits")
            for f in (OUT / d).rglob("*")
            if f.is_file() and f.suffix not in {".pending", ".lock"}
        ]
        for file in selected:
            resource.check()
            target = payload / file.relative_to(OUT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != digest(file):
                    raise ValueError("existing essential delivery member changed")
            else:
                shutil.copyfile(file, target)
        controls = {}
        for key, row in p["historical_controls"].items():
            target = payload / "historical_controls" / (key.replace("/", "__") + ".parquet")
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copyfile(row["prediction"], target)
            if digest(target) != row["prediction_sha256"]:
                raise ValueError("bundled historical control differs")
            controls[key] = dict(
                original_endpoint=row["endpoint"],
                prediction_path=str(target.relative_to(payload)),
                prediction_sha256=digest(target),
                checkpoint_reference=row["checkpoint"],
            )
        publish_json(payload / "HISTORICAL_CONTROLS.json", controls)
        # Only frozen scientific source files are included, without raw data or expert weights.
        for row in record(Path(p["launch"]["freeze"]))["files"]:
            if (
                row["category"] != "code"
                or row["root"] != "work"
                or not row["relative_path"].startswith("src/")
            ):
                continue
            source = ROOT / row["relative_path"]
            target = payload / "provenance" / row["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if digest(target) != row["sha256"]:
                raise ValueError("delivery scientific code changed")
        for name in ("regenerate.py", "README.md"):
            target = payload / name
            source = Path(__file__).parent / name
            if not target.exists():
                shutil.copyfile(source, target)
            elif digest(target) != digest(source):
                raise ValueError("delivery regeneration source changed")
        manifest = {
            str(f.relative_to(payload)).replace("\\", "/"): dict(
                bytes=f.stat().st_size, sha256=digest(f)
            )
            for f in sorted(payload.rglob("*"))
            if f.is_file() and f.name != "CONTENT_MANIFEST.json"
        }
        publish_json(
            payload / "CONTENT_MANIFEST.json",
            dict(
                members=manifest,
                protocol_sha256=pin,
                scope=(
                    "Six new CPU FP32 heads from included normalized OLD_DEV inputs; "
                    "historical controls reused."
                ),
                raw_reconstruction_included=False,
                train_inputs_included=False,
                expert_weights_included=False,
            ),
        )
        archive = OUT / "E_JEPA_TTC_H16_REPLICATION_20261003.zip"
        if not archive.exists():
            pending = archive.with_suffix(".zip.pending")
            with zipfile.ZipFile(
                pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as z:
                for f in sorted(payload.rglob("*")):
                    if f.is_file():
                        resource.check()
                        z.write(f, str(f.relative_to(payload)).replace("\\", "/"))
            pending.replace(archive)
        sha = digest(archive)
        atomic_bytes(
            archive.with_suffix(".zip.sha256"), (sha + "  " + archive.name + "\n").encode()
        )
        extraction = OUT / "independent_extraction"
        extraction.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != len(set(z.namelist())) or z.testzip() is not None:
                raise ValueError("duplicate members or bad CRC")
            for member in z.infolist():
                resource.check()
                dest = (extraction / member.filename).resolve()
                if not dest.is_relative_to(extraction.resolve()):
                    raise ValueError("unsafe ZIP path")
                if dest.exists():
                    with z.open(member) as stream:
                        import hashlib

                        h = hashlib.sha256()
                        while block := stream.read(1024 * 1024):
                            h.update(block)
                        if h.hexdigest() != digest(dest):
                            raise ValueError("extracted member changed")
                else:
                    z.extract(member, extraction)
        # Parent exits the scientific environment; only the extracted package is supplied.
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(extraction / "regenerate.py"),
                "--root",
                str(extraction),
                "--output",
                str(OUT / "REGENERATION.json"),
            ],
            cwd=extraction,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        atomic_bytes(OUT / "REGENERATION.log", (completed.stdout + completed.stderr).encode())
        if completed.returncode:
            raise RuntimeError("independent regeneration failed; outputs preserved")
        publish_json(
            OUT / "FINAL_DELIVERY.json",
            dict(
                status="DELIVERED_REGENERATED_INDEPENDENT_H16",
                archive=str(archive),
                sha256=sha,
                bytes=archive.stat().st_size,
                protocol_sha256=pin,
                regeneration_sha256=digest(OUT / "REGENERATION.json"),
                scientific_saved_updates=15000,
                conclusion=result["conclusion"],
                no_push=True,
            ),
        )
        atomic_json(
            OUT / "STATUS.json",
            dict(
                status="CLOSED_INDEPENDENT_H16_REPLICATION_DELIVERED",
                endpoint_count=6,
                scientific_saved_updates=15000,
                archive_sha256=sha,
                conclusion=result["conclusion"],
                future_optimizer_updates_authorized=0,
            ),
        )
        print(json.dumps(record(OUT / "FINAL_DELIVERY.json")), flush=True)


if __name__ == "__main__":
    main()
