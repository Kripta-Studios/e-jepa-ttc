"""Package trained candidates, every experiment, source and reproducible feature inputs."""

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
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "previous", "data", "campaign", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Existing deliveries are immutable")
    if read(args.root / "FINAL_CHECKS.json")["status"] != "passed":
        raise ValueError("Final checks are incomplete")
    files: dict[str, Path] = {}
    for path in args.root.rglob("*"):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in (".pyc", ".tmp")
        ):
            if path.name not in ("DELIVERY.json", "pyrightconfig.json"):
                files["phase_startup/" + path.relative_to(args.root).as_posix()] = path
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "src", "operational", "scripts", "configs"]
    )
    for item in raw.decode("utf-8").split("\0"):
        if (
            item
            and Path(item).is_file()
            and Path(item).suffix in (".py", ".yaml", ".yml", ".json", ".toml")
        ):
            files["repository/" + Path(item).as_posix()] = Path(item)
    for name in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "README.md",
        "LICENSE",
        "Makefile",
    ):
        files["repository/" + name] = Path(name)
    for pattern in (
        "tests/unit/test_phase_startup*.py",
        "docs/superpowers/plans/2026-10-31-phase-startup-router.md",
        "docs/evttc_transfer_v2/STARTUP_ROUTER_HANDOFF.md",
        "docs/evttc_transfer_v2/evidence/phase_startup/*",
    ):
        for path in Path().glob(pattern):
            if path.is_file():
                files["repository/" + path.as_posix()] = path
    for path in args.root.parent.glob("phase_startup_*"):
        if path.is_file() and path.suffix in (".log", ".xml"):
            files["logs/" + path.name] = path
    # Exact small dependencies needed to regenerate startup features/training.
    for relative in (
        "all_data_refit/PREDICTIONS.npz",
        "training/POPULATION.npz",
        "training/CV_PREDICTIONS.npz",
        "training/CONFIG.json",
        "training/SELECTION.json",
        "training/FINAL_FREEZE.json",
    ):
        files["prior_feature_inputs/" + relative] = args.previous / relative
    for path in (args.previous / "training/final").rglob("*"):
        if path.is_file():
            files["prior_feature_inputs/" + path.relative_to(args.previous).as_posix()] = path
    for speed in (750, 1000):
        for relative in (
            f"external/Slider-{speed}/PREDICTIONS.npz",
            f"external_contract/{speed}_bbox.csv",
        ):
            files["prior_feature_inputs/" + relative] = args.previous / relative
    files["native_metadata/POPULATION.json"] = args.data / "POPULATION.json"
    for relative in ("Slider500/FREEZE.json", "Slider500/inputs/Slider-500/bbox.csv"):
        files["campaign_inputs/" + relative] = args.campaign / relative
    manifest = {
        name: {"sha256": file_sha(path), "bytes": path.stat().st_size}
        for name, path in sorted(files.items())
    }
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
                "PHASE-TTC startup candidate, 2026-10-31.\n"
                "Read phase_startup/report/REPORT.md. Previous robust release remains default.\n"
                "This package adds an evaluated research candidate and an all-data fitted model;\n"
                "neither establishes SOTA or clean independent generalization.\n"
                "Install repository dependencies (Python3.11, uv sync --extra dev).\n"
                "Set PYTHONPATH=src;. on Windows. Run from repository.\n"
                "Model: ../phase_startup/FINAL_MODEL, "
                "CLI: python -m scripts.predict_phase_startup --help\n"
                "Retrain cached sensors: python -m scripts.train_phase_startup "
                "--previous ../prior_feature_inputs --data ../native_metadata "
                "--campaign ../campaign_inputs --output NEW_DIRECTORY --coordinate inverse\n"
                "Raw sensor replays and source88744 validation require the original "
                "licensed local data/cache.\n"
                "All current and failed experiment checkpoints/results are included.\n"
            ),
        )
    with zipfile.ZipFile(temporary) as archive:
        for name, item in manifest.items():
            content = archive.read(name)
            if (
                len(content) != item["bytes"]
                or hashlib.sha256(content).hexdigest() != item["sha256"]
            ):
                raise ValueError(f"ZIP member integrity failed: {name}")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failed")
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
    print(f"COMPLETE verified {len(files)} members: {args.output}", flush=True)


if __name__ == "__main__":
    main()
