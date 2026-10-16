"""Deliver complete result tables and portable examples under a strict 100 MB cap."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from operational.range_transfer.common import read, utc, write

from .bundle import require_quiescent, verify_archive, write_archive
from .common import ROOT, RUN

LIMIT = 100_000_000
TEXT = {".json", ".csv", ".md", ".txt", ".log", ".yaml", ".yml"}


def collect() -> tuple[dict[str, Path], list[dict]]:
    """Retain positive and negative evidence; omit raw caches and optimizer payloads."""
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    )
    files = {
        "source/" + name: ROOT / name
        for name in tracked
        if name and not name.startswith(("artifacts/", "data/cache/"))
    }
    omitted = []
    for path in sorted(RUN.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(RUN)
        ephemeral = (
            "__pycache__" in relative.parts
            or path.suffix in {".lock", ".pyc", ".pending"}
            or ".pending" in path.name
        )
        portable = "exports" in relative.parts and path.suffix in {".pt", ".npz", ".npy"}
        table = path.name == "predictions.parquet"
        include = not ephemeral and (path.suffix in TEXT | {".png", ".pdf"} or portable or table)
        if include:
            files["campaign/" + relative.as_posix()] = path
        else:
            omitted.append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "reason": "runtime file" if ephemeral else "nonportable tensor/cache/fragment",
                }
            )
    files["README.md"] = ROOT / "docs/joint_measurement_20261012/RESEARCH_STATUS.md"
    return files, omitted


def build(output: Path) -> dict:
    """Verify a private candidate before publishing a ZIP of at most 100,000,000 bytes."""
    output = output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("A new .zip destination is required")
    candidate = output.with_suffix(".candidate")
    if any(
        path.exists()
        for path in (output, candidate, candidate.with_suffix(candidate.suffix + ".pending"))
    ):
        raise FileExistsError("Never overwrite a previous delivery or interrupted candidate")
    require_quiescent(ROOT, RUN)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Commit all source and report changes before delivery")
    conclusion = read(RUN / "CAMPAIGN_CONCLUSION.json")
    if not isinstance(conclusion.get("remaining_work"), list):
        raise ValueError("An explicit current conclusion with remaining work is required")
    files, omitted = collect()
    context = {
        "created_utc": utc(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "conclusion": conclusion,
        "max_zip_bytes": LIMIT,
        "selection": "All metadata, curves, consolidated predictions and portable exports",
        "omitted_files": omitted,
        "optimizer_checkpoint_payloads_included": False,
        "raw_sensor_or_feature_caches_included": False,
        "external_historical_parents_embedded": False,
        "reproduction_scope": (
            "Recompute reported metrics from consolidated predictions and replay included "
            "portable examples. Training and raw-sensor inference require external data, "
            "historical parents and local full checkpoints. Metadata can reference omitted "
            "tensor payloads. Fusion head exports alone do not include their producers."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = sum(path.stat().st_size for path in files.values())
    if shutil.disk_usage(output.parent).free < payload_bytes + 10_000_000_000:
        raise RuntimeError("Delivery would violate the 10 GB free disk reserve")
    write_archive(candidate, files, context)
    if candidate.stat().st_size > LIMIT:
        measured = candidate.stat().st_size
        candidate.unlink()
        raise RuntimeError(f"Unpublished candidate removed: {measured} bytes exceeds {LIMIT}")
    verified = verify_archive(candidate)
    os.replace(candidate, output)
    output.with_suffix(".zip.sha256").write_text(
        f"{verified['zip_sha256']}  {output.name}\n", encoding="utf-8"
    )
    write(output.with_suffix(".zip.verification.json"), verified)
    return {key: verified[key] for key in ("status", "zip_bytes", "zip_sha256", "payload_members")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(build(arguments.output))
