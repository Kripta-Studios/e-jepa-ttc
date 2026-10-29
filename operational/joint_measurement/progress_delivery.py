"""Package closed results and a labelled live-state snapshot without stopping C1."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .bundle import verify_archive, write_archive

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/joint_measurement_20261012"
C1 = RUN / "adaptive/C1_geometry/fits/PAIR-CORR-GEOMETRY-S7"


def stable_copy(source: Path, target: Path) -> None:
    """Copy a stable file, rejecting concurrent mutation rather than hiding it."""
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Changed during snapshot: {source}")


def prepare(stage: Path) -> dict:
    """Capture selected evidence outside the live accounting tree."""
    if stage.exists() or stage.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Use a new snapshot directory outside the worktree")
    if shutil.disk_usage(stage.parent).free < 10_000_000_000:
        raise RuntimeError("Insufficient free disk reserve")
    stage.mkdir()
    started = datetime.now(UTC).isoformat()
    full = (
        "fits",
        "predictions",
        "exports",
        "bootstrap",
        "cost",
        "primary_analysis",
        "primary_evidence",
        "primary_milestone",
        "report",
        "system_compare",
        "rgb",
    )
    metadata = ("diagnostics", "runtime", "qa", "verification")
    omitted = []
    for folder in (*full, *metadata):
        for source in sorted((RUN / folder).rglob("*")):
            if not source.is_file():
                continue
            rel = source.relative_to(RUN)
            skip = (
                source.suffix in {".lock", ".pyc"}
                or "__pycache__" in rel.parts
                or "hourly" in rel.parts
                or ".pending" in source.name
                or (
                    folder in metadata
                    and source.suffix not in {".json", ".csv", ".md", ".txt", ".log"}
                )
                or (folder == "verification" and source.suffix == ".log")
            )
            if skip:
                omitted.append(rel.as_posix())
                continue
            stable_copy(source, stage / "campaign" / rel)
    # These are time-labelled observations, not an atomic checkpoint of the campaign.
    observations = [
        "ACCOUNTING.json",
        "GPU_BUDGET.json",
        "NEXT_DECISION.json",
        "BUDGET_AUTHORIZATION_20261014.json",
        "PRIMARY_ENDPOINTS.json",
        "PRIMARY_SELECTION_FINAL.json",
        "NEXT_DECISION_MECHANISM_20261014.json",
        "CAMPAIGN_CONCLUSION.json",
        "adaptive/C1_geometry/REGISTRATION.json",
        "adaptive/C1_geometry/RUNNER_STATE.json",
    ]
    for name in observations:
        stable_copy(RUN / name, stage / "campaign" / name)
    # Open the immutable version selected by the pointer, never checkpoint_last.pt.
    pointer = json.loads((C1 / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8"))
    version = pointer["version"]
    if Path(version).name != version:
        raise ValueError("Unsafe checkpoint version")
    target = stage / "live_C1_checkpoint" / version
    stable_copy(C1 / "checkpoint_versions" / version, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != pointer["checkpoint_sha256"]:
        raise ValueError("C1 immutable checkpoint hash mismatch")
    (target.parent / "CHECKPOINT_POINTER.json").write_text(
        json.dumps(pointer, indent=2) + "\n", encoding="utf-8"
    )
    journal = json.loads((C1 / "UPDATE_JOURNAL.json").read_text(encoding="utf-8"))
    snapshot = {
        "schema": "v18_progress_delivery_v1",
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "status": "RESEARCH_ONGOING",
        "full_objective_complete": False,
        "primary": "NO_SYSTEM_PROMOTION",
        "rgb": "NO_CANDIDATE_RGB_PAIR",
        "C1": {"endpoint_target": 12000, "checkpoint": pointer, "observed_journal": journal},
        "accounting_scope": (
            "Per-file observations; active GPU scope may not yet be settled; "
            "not an atomic campaign snapshot"
        ),
        "omitted_files": omitted,
        "omitted_categories": [
            "raw sensors",
            "external historical parents",
            "large diagnostic/QA tensor payloads",
            "live logs",
            "live C1 checkpoint aliases",
        ],
        "remaining_work": [
            "C1 fixed endpoint and complete P/H evidence",
            "evidence-enabled adaptive/fusion decisions",
            "conditional replication/refit",
            "final campaign bundle",
        ],
        "reproduction_scope": (
            "Closed predictions, weights and portable examples included; "
            "full retraining needs external data/parents. "
            "Metadata can reference omitted QA payloads."
        ),
    }
    (stage / "SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def build(stage: Path, output: Path) -> dict:
    """Build a results-only delivery with an enforced 100 MB decimal limit."""
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Commit source before bundling")
    context = json.loads((stage / "SNAPSHOT.json").read_text(encoding="utf-8"))
    context["commit"] = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    )
    files = {}
    compact_omissions = []
    for p in stage.rglob("*"):
        if not p.is_file():
            continue
        name = p.relative_to(stage).as_posix()
        if (
            p.suffix in {".pt", ".npz", ".npy"}
            or p.name.startswith("chunk_")
            or name.startswith("live_C1_checkpoint/")
            or p.name in {"SNAPSHOT.json", "COMPACT_SNAPSHOT.json"}
        ):
            compact_omissions.append(name)
            continue
        files[name] = p
    context["max_zip_bytes"] = 100_000_000
    context["checkpoint_payloads_included"] = False
    context["portable_model_payloads_included"] = False
    context["compact_omissions"] = compact_omissions
    context["reproduction_scope"] = (
        "Consolidated predictions, metrics, curves, documentation and verification receipts. "
        "No model weights, checkpoint tensors, sensor data or portable input arrays. "
        "Metadata may reference omitted payloads; this is a results-only delivery."
    )
    compact = stage / "COMPACT_SNAPSHOT.json"
    compact.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    files["SNAPSHOT.json"] = compact
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    )
    for name in tracked:
        if name and not name.startswith(("artifacts/", "data/cache/")):
            files["source/" + name] = ROOT / name
    files["README.md"] = (
        ROOT / "docs/joint_measurement_20261012/RESULTADOS_Y_CONTINUACION_20261014.md"
    )
    required = sum(p.stat().st_size for p in files.values())
    if shutil.disk_usage(output.parent).free < required + 10_000_000_000:
        raise RuntimeError("Bundle would violate free disk reserve")
    write_archive(output, files, context)
    if output.stat().st_size > context["max_zip_bytes"]:
        output.unlink()
        raise RuntimeError("Removed bundle: exceeded the user limit of 100,000,000 bytes")
    result = verify_archive(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{result['zip_sha256']}  {output.name}\n", encoding="utf-8"
    )
    output.with_suffix(output.suffix + ".verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return {k: result[k] for k in ("status", "zip_sha256", "zip_bytes", "payload_members")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "build"])
    parser.add_argument("stage", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.stage)
        print(
            json.dumps(
                {k: result[k] for k in ("status", "started_utc", "finished_utc", "C1")}, indent=2
            )
        )
    else:
        if args.output is None:
            parser.error("build requires --output")
        print(json.dumps(build(args.stage, args.output), indent=2))
