"""Package revision evidence as an additive bundle for the existing V12 repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from operational.efficient_context.common import digest
from operational.train40_system.durable_io import atomic_json


def package(root: Path) -> Path:
    """Require complete evidence and verify every archived member's SHA-256."""
    root = root.resolve()
    campaign = root / "artifacts/ttc_revision_20261009"
    destination = campaign / "REVISION_ESSENTIAL.zip"
    if destination.exists():
        raise FileExistsError("preserve existing delivery bundle")
    checks = json.loads((campaign / "CPU_COMPLETION.json").read_text(encoding="utf-8"))
    if checks["status"] != "COMPLETE" or any(c["exit_code"] for c in checks["checks"]):
        raise ValueError("successful final checks required before packaging")
    if (campaign / "V13_PAUSE_OWNERSHIP.json").exists():
        resumed = json.loads((campaign / "V13_RESUME_VERIFIED.json").read_text(encoding="utf-8"))
        if resumed["status"] != "RESUMED_AND_VERIFIED":
            raise ValueError("verify restoration of user training before delivery")
    original = root / "docs/sota_campaign_20261008"
    for line in (original / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (original / name).resolve()
        if not path.is_relative_to(original) or digest(path) != expected:
            raise ValueError(f"original campaign changed: {name}")
    for cohort in ("dev32", "fcwd"):
        receipt = json.loads(
            (campaign / f"replay_{cohort}/REPLAY_COMPLETE.json").read_text(encoding="utf-8")
        )
        if receipt["status"] != "COMPLETE":
            raise ValueError("complete population replay required")
        replay = campaign / f"replay_{cohort}"
        identity_path = replay / "REPLAY_IDENTITY.json"
        if digest(identity_path) != receipt["identity_sha256"]:
            raise ValueError("replay identity changed")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        for name, expected in identity["source_sha256"].items():
            if digest(root / "operational/ttc_revision" / name) != expected:
                raise ValueError(f"replay source changed: {name}")
        if len(receipt["fragments"]) != receipt["population"]:
            raise ValueError("incomplete replay fragment count")
        for name, expected in receipt["fragments"].items():
            if digest(replay / "fragments" / name) != expected:
                raise ValueError(f"replay fragment changed: {cohort}/{name}")
    report = root / "docs/ttc_revision_20261009"
    receipt = json.loads((report / "REPORT_SOURCES.json").read_text(encoding="utf-8"))
    if receipt["report_sha256"] != digest(report / "README.md"):
        raise ValueError("report modified after generation")
    if receipt["generator_sha256"] != digest(root / "operational/ttc_revision/report.py"):
        raise ValueError("report generator changed after generation")
    for name, expected in receipt["tables"].items():
        if digest(report / "tables" / name) != expected:
            raise ValueError(f"report table changed: {name}")
    for seed in (7, 13, 23):
        path = campaign / f"direct_seed{seed}.pt"
        if (
            digest(path)
            != json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["sha256"]
        ):
            raise ValueError("trained checkpoint bytes changed")
    members = set()
    for directory in (root / "operational/ttc_revision", campaign, report):
        members.update(
            p
            for p in directory.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in {".pyc", ".tmp", ".zip", ".sha256"}
            and p.name
            not in {
                "BUNDLE_MANIFEST.json",
                "BUNDLE_VERIFICATION.json",
                "V13_RESUME.stdout.log",
                "V13_RESUME.stderr.log",
            }
        )
    members.add(root / "configs/experiment/ttc_revision_20261009.json")
    members.update((root / "tests/unit").glob("test_ttc_revision_*.py"))
    manifest = {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in sorted(members)}
    manifest_path = campaign / "BUNDLE_MANIFEST.json"
    atomic_json(
        manifest_path,
        {
            "scope": (
                "Additive V12 revision; requires existing repository, frozen producers and raw data"
            ),
            "files": manifest,
            "original_campaign_modified": False,
        },
    )
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in manifest:
            archive.write(root / name, name)
        archive.write(manifest_path, str(manifest_path.relative_to(root)).replace("\\", "/"))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("bundle CRC validation failed")
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"bundle hash mismatch: {name}")
    temporary.replace(destination)
    sha = digest(destination)
    destination.with_suffix(".zip.sha256").write_text(
        f"{sha}  {destination.name}\n", encoding="utf-8"
    )
    atomic_json(
        campaign / "BUNDLE_VERIFICATION.json",
        {
            "status": "PASSED",
            "members_verified": len(manifest),
            "archive_sha256": sha,
            "archive_bytes": destination.stat().st_size,
        },
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(package(parser.parse_args().root))
