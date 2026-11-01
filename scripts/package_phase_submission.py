"""Package the verified model, measured predictions, evidence and reproducible source."""

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
from operational.phase_submission.runtime import load_submission_model
from operational.phase_submission.verify import verify_artifact_bindings, verify_results

INSTALL_FILES = ("pyproject.toml", "uv.lock", ".python-version", "LICENSE", "README.md")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root",
        "historical_run",
        "population",
        "exclusions",
        "road_targets",
        "slider_contract",
        "output",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Existing releases are immutable")
    checks = read(args.root / "FINAL_CHECKS.json")
    if checks["status"] != "passed":
        raise ValueError("Final checks not passed")
    verify_artifact_bindings(
        args.root,
        args.historical_run,
        args.population,
        args.exclusions,
        args.road_targets,
        args.slider_contract,
        Path.cwd(),
        current_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    load_submission_model(args.root / "FINAL_MODEL")
    replay = args.root / "replay"
    complete = read(replay / "COMPLETE.json")
    freeze = read(replay / "RUN_FREEZE.json")
    evaluation = read(args.root / "evaluation/LOCAL_EVALUATION.json")
    if file_sha(replay / "results.zip") != evaluation["prediction_seal_before_scoring"]:
        raise ValueError("Evaluation is not bound to this submission")
    for name, key in (
        ("results.zip", "results_zip_sha256"),
        ("all_rows.jsonl", "ledger_sha256"),
        ("RUN_FREEZE.json", "freeze_sha256"),
    ):
        if file_sha(replay / name) != complete[key]:
            raise ValueError("Changed replay artifacts")
    for name, digest in freeze["sources"].items():
        if file_sha(Path(name)) != digest:
            raise ValueError("Changed executed source")
    rows = [json.loads(line) for line in (replay / "all_rows.jsonl").read_text().splitlines()]
    expected = {
        seq: [(r["index"], r["timestamp_text"]) for r in queries]
        for seq, queries in freeze["queries"].items()
    }
    verify_results(replay / "results.zip", rows, expected)
    files = {}
    for directory in ("FINAL_MODEL", "training", "replay", "audit", "evaluation", "validation"):
        for path in (args.root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files["phase_submission/" + path.relative_to(args.root).as_posix()] = path
    files["phase_submission/FINAL_CHECKS.json"] = args.root / "FINAL_CHECKS.json"
    files["inputs/POPULATION.npz"] = args.population
    files["inputs/exclusions.json"] = args.exclusions
    files["inputs/road_targets.json"] = args.road_targets
    for speed in (750, 1000):
        files[f"inputs/slider/{speed}_gt.csv"] = args.slider_contract / f"{speed}_gt.csv"
    for name in ("all_rows.jsonl", "RUN_FREEZE.json", "manifest.json", "environment.json"):
        files["historical_run/" + name] = args.historical_run / name
    tracked = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
    for name in tracked:
        path = Path(name)
        if (
            path.is_file()
            and (
                name.startswith(("src/", "operational/", "scripts/", "configs/"))
                or name in INSTALL_FILES
                or name.startswith("docs/evttc_submission_20261031/")
                or name == "tests/unit/test_phase_submission.py"
            )
            and path.suffix not in (".pyc", ".pt", ".npz")
        ):
            files["repository/" + path.as_posix()] = path
    for path in args.root.parent.glob("phase_submission_*"):
        if path.is_file() and path.suffix in (".log", ".xml"):
            files["verification_logs/" + path.name] = path
    files["START_HERE.md"] = Path("docs/evttc_submission_20261031/README.md")
    manifest = {
        name: {"sha256": file_sha(path), "bytes": path.stat().st_size}
        for name, path in sorted(files.items())
    }
    temporary = args.output.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
        z.writestr("MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2))
    with zipfile.ZipFile(temporary) as z:
        if len(z.namelist()) != len(manifest) + 1 or z.testzip() is not None:
            raise ValueError("Wrong ZIP inventory or CRC")
        for name, record in manifest.items():
            content = z.read(name)
            if (
                len(content) != record["bytes"]
                or hashlib.sha256(content).hexdigest() != record["sha256"]
            ):
                raise ValueError("ZIP member integrity failed: " + name)
    temporary.replace(args.output)
    atomic_receipt(
        args.root / "DELIVERY.json",
        {
            "path": str(args.output),
            "sha256": file_sha(args.output),
            "bytes": args.output.stat().st_size,
            "verified_members": len(manifest),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    )
    print(f"COMPLETE {len(manifest)} rehashed members: {args.output}")


if __name__ == "__main__":
    main()
