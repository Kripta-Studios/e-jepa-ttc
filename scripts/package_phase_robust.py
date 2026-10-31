"""Assemble local models, reproducible evidence and source; verify every ZIP member."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new delivery path; existing archives are immutable")
    checks = read(args.root / "FINAL_CHECKS.json")
    if checks["status"] != "passed":
        raise ValueError("Final verification has not passed")
    files: dict[str, Path] = {}
    for path in args.root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name in {"native_sensor_manifest.json", "pyrightconfig.json", "DELIVERY.json"}:
            continue
        if path.suffix in {".pyc", ".tmp"}:
            continue
        files["phase_robust/" + path.relative_to(args.root).as_posix()] = path
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "src", "operational", "scripts", "configs"]
    )
    for item in raw.decode("utf-8").split("\0"):
        if not item:
            continue
        path = Path(item)
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json", ".toml"}:
            files["repository/" + path.as_posix()] = path
    for item in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "README.md",
        "LICENSE",
        "Makefile",
    ):
        path = Path(item)
        if path.is_file():
            files["repository/" + item] = path
    for path in Path("tests/unit").glob("test_phase_robust*.py"):
        files["repository/" + path.as_posix()] = path
    for path in Path("docs/evttc_transfer_v2/evidence/phase_robust").rglob("*"):
        if path.is_file():
            files["repository/" + path.as_posix()] = path
    plan = Path("docs/superpowers/plans/2026-10-31-phase-robust-fusion.md")
    files["repository/" + plan.as_posix()] = plan
    for path in args.root.parent.glob("phase_robust_*"):
        if path.is_file() and path.suffix in {".log", ".xml"}:
            files["logs/" + path.name] = path
    # Prior input metadata required by the grouped training command; no raw data.
    for relative in ("F1_grouped/CONFIG.json", "diagnosis/REPORT.json"):
        files["previous_campaign/" + relative] = Path("output/phase_campaign") / relative
    manifest = {
        name: {"sha256": file_sha(path), "bytes": path.stat().st_size}
        for name, path in sorted(files.items())
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr(
            "START_HERE.txt",
            (
                "PHASE-TTC robust fusion, 2026-10-31\n"
                "Read phase_robust/report/REPORT.md and phase_robust/FINAL_MODEL/MODEL_CARD.md.\n"
                "evaluated is the default checked model; "
                "all_data_refit includes every eligible label.\n"
                "The latter's metrics are resubstitution, "
                "with no remaining held-out confirmation.\n"
                "Install repository dependencies using Python3.11 and uv sync --extra dev.\n"
                "Run from repository with PYTHONPATH=src;. and PYTHONUTF8=1.\n"
                "Weights and evidence: ../phase_robust/FINAL_MODEL. Raw sensors are not bundled.\n"
                "Slider CLI: python -m scripts.predict_phase_robust --help\n"
                "Native road CLI: python -m scripts.predict_phase_robust_native --help\n"
                "Retrain fixed gates: scripts/train_phase_robust.py "
                "with --diagnosis ../phase_robust/diagnosis "
                "--campaign ../previous_campaign --output NEW_OUTPUT.\n"
            ),
        )
    with zipfile.ZipFile(temporary) as archive:
        for name, entry in manifest.items():
            value = archive.read(name)
            if len(value) != entry["bytes"] or hashlib.sha256(value).hexdigest() != entry["sha256"]:
                raise ValueError(f"Archive member failed verification: {name}")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC validation failed")
    temporary.replace(args.output)
    atomic_receipt(
        args.root / "DELIVERY.json",
        {
            "archive": str(args.output),
            "sha256": file_sha(args.output),
            "bytes": args.output.stat().st_size,
            "verified_members": len(files),
            "all_members_rehashed": True,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    )
    print(f"COMPLETE verified delivery: {args.output}", flush=True)


if __name__ == "__main__":
    main()
