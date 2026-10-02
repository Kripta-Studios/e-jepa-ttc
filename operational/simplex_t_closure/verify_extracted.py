"""Extract the declared bundle and verify using only its included scientific code."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import psutil
from runtime import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/simplex_t/closure_20261002"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pre", "final"], required=True)
    args = parser.parse_args()
    delivery = ART / "essential_delivery"
    if args.stage == "pre":
        archive = delivery / "PREDELIVERY_REGENERATION_INPUT.zip"
        extraction = ART / "predelivery_extracted"
        proof = ART / "PREDELIVERY_REGENERATION.json"
    else:
        candidates = list(delivery.glob("E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_*.zip"))
        if len(candidates) != 1:
            raise ValueError("exactly one final delivery ZIP required")
        archive = candidates[0]
        extraction = ART / "final_delivery_extracted"
        proof = ART / "FINAL_REGENERATION.json"
    sha = digest(archive)
    if archive.with_suffix(".zip.sha256").read_text(encoding="ascii").split()[0] != sha:
        raise ValueError("ZIP sidecar hash mismatch")
    policy = json.loads(Path(__file__).with_name("resource_policy.json").read_text())
    if psutil.virtual_memory().available < policy["host_available_ram_floor_bytes"]:
        raise InterruptedError("PAUSED_RESOURCE: extracted regeneration RAM admission")
    extraction.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or bundle.testzip() is not None:
            raise ValueError("ZIP duplicate member or CRC mismatch")
        manifest = json.loads(bundle.read("CONTENT_MANIFEST.json"))
        if set(names) != {*manifest["members"], "CONTENT_MANIFEST.json"}:
            raise ValueError("ZIP inventory differs from its content manifest")
        for member in bundle.infolist():
            destination = (extraction / member.filename).resolve()
            if not destination.is_relative_to(extraction.resolve()) or member.is_dir():
                raise ValueError("ZIP member must be a file inside extraction root")
            if destination.exists():
                pin = manifest["members"].get(member.filename)
                if pin is not None and (
                    destination.stat().st_size != pin["bytes"]
                    or digest(destination) != pin["sha256"]
                ):
                    raise ValueError("preserve conflicting prior extraction")
                if pin is None and destination.read_bytes() != bundle.read(member):
                    raise ValueError("preserve differing extracted manifest")
                continue
            if (
                shutil.disk_usage(extraction).free - member.file_size - 512 * 1024**2
                < policy["written_volume_emergency_floor_bytes"]
            ):
                raise InterruptedError("PAUSED_RESOURCE: extraction storage admission")
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_name(destination.name + ".pending")
            with bundle.open(member) as source, pending.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024**2)
                target.flush()
                os.fsync(target.fileno())
            os.replace(pending, destination)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(extraction / "provenance/work/src")
    environment["PYTHONUTF8"] = "1"
    command = [
        sys.executable,
        "-B",
        "-u",
        str(extraction / "operational/regenerate.py"),
        "--bundle-root",
        str(extraction),
        "--output",
        str(proof),
    ]
    log = ART / (args.stage.upper() + "_REGENERATION.log")
    with log.open("a", encoding="utf-8") as stream:
        code = subprocess.run(
            command, cwd=extraction, env=environment, stdout=stream, stderr=subprocess.STDOUT
        ).returncode
    if code != 0:
        print(json.dumps({"status": "REGENERATION_FAILED", "log": str(log)}), flush=True)
        return code
    result = json.loads(proof.read_text(encoding="utf-8"))
    if result["status"] != "EXTRACTED_BUNDLE_REGENERATION_VERIFIED":
        raise ValueError("included regeneration did not establish its declared scope")
    if args.stage == "final":
        if digest(archive) != sha:
            raise ValueError("ZIP changed during final regeneration")
        receipt = {
            "status": "FINAL_ZIP_HASH_CRC_INVENTORY_AND_EXTRACTED_REGENERATION_VERIFIED",
            "archive": str(archive),
            "sha256": sha,
            "archive_bytes": archive.stat().st_size,
            "proof": str(proof),
            "proof_sha256": digest(proof),
            "members_hashed": result["members_hashed"],
            "heads_regenerated": len(result["heads"]),
            "scores_regenerated": len(result["scores"]),
            "scope": result["scope"],
            "raw_autonomous": False,
            "optimizer_updates": 0,
        }
        atomic_json(delivery / "FINAL_DELIVERY.json", receipt)
    print(json.dumps({"status": result["status"], "proof": str(proof)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
