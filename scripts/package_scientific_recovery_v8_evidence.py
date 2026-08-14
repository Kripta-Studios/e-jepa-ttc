#!/usr/bin/env python
"""Create an immutable, deterministic Scientific Recovery V8 evidence package."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Running ``python scripts/package_...py`` places ``scripts/`` rather than the
# repository root on sys.path.  Keep this explicit import path so the command
# works both as a script and as ``python -m scripts.package_...``.
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from scripts.build_scientific_recovery_v8_report import build_report, sha256_file  # noqa: E402

MAX_COMPACT_LOG_BYTES = 5 * 1024 * 1024
EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx"}
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _is_checkpoint(path: Path) -> bool:
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def _is_dataset_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts & {"datasets", "dataset", "raw", "cache"})


def _include_artifact(path: Path) -> bool:
    if _is_checkpoint(path) or _is_dataset_path(path):
        return False
    suffix = path.suffix.lower()
    if suffix == ".log":
        return path.stat().st_size <= MAX_COMPACT_LOG_BYTES
    return suffix in {".json", ".csv", ".parquet", ".md", ".png", ".svg", ".pdf", ".txt", ".npz"}


def _candidate_files(repo_root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    include: set[Path] = set()
    excluded: list[dict[str, str]] = []

    direct_files = [
        repo_root / "configs" / "protocol" / "scientific_recovery_v8_temporal.json",
        repo_root / "docs" / "SCIENTIFIC_RECOVERY_V8_STATUS.md",
        repo_root / "docs" / "SCIENTIFIC_RECOVERY_V8_REPORT.md",
    ]
    for path in direct_files:
        if path.is_file():
            include.add(path)
    for directory in (
        repo_root / "configs" / "experiment" / "scientific_recovery_v8_fold_chain",
        repo_root / "artifacts" / "scientific_recovery_v8",
    ):
        if directory.is_dir():
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                if _include_artifact(path):
                    include.add(path)
                else:
                    excluded.append(
                        {
                            "path": path.relative_to(repo_root).as_posix(),
                            "reason": "dataset/cache/checkpoint/oversize-log exclusion",
                        }
                    )
    model_dir = repo_root / "configs" / "model"
    if model_dir.is_dir():
        include.update(path for path in model_dir.glob("e_jepa_causal_scale_event_v8_*.yaml"))
    runs_root = repo_root / "artifacts" / "runs"
    if runs_root.is_dir():
        for run in sorted(
            path for path in runs_root.iterdir() if "scientific_recovery_v8" in path.name
        ):
            if not run.is_dir():
                continue
            for path in run.rglob("*"):
                if not path.is_file():
                    continue
                if _include_artifact(path):
                    include.add(path)
                else:
                    excluded.append(
                        {
                            "path": path.relative_to(repo_root).as_posix(),
                            "reason": "dataset/cache/checkpoint/oversize-log exclusion",
                        }
                    )
    return sorted(include, key=lambda item: item.relative_to(repo_root).as_posix()), excluded


def checkpoint_manifest(repo_root: Path) -> list[dict[str, Any]]:
    """Hash V8 checkpoints while keeping checkpoint bytes outside the evidence ZIP."""

    records: list[dict[str, Any]] = []
    roots = (repo_root / "artifacts" / "runs", repo_root / "artifacts" / "checkpoints")
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not _is_checkpoint(path):
                continue
            if "scientific_recovery_v8" not in path.as_posix().lower():
                continue
            records.append(
                {
                    "path": path.relative_to(repo_root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "included_in_zip": False,
                }
            )
    return records


def _zip_is_valid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _writestr(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compresslevel=9)


def package_evidence(repo_root: Path, output_path: Path) -> dict[str, Any]:
    """Build an immutable V8 ZIP after regenerating its signed report."""

    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        if not _zip_is_valid(output_path):
            raise FileExistsError(f"refusing to overwrite corrupt package: {output_path}")
        raise FileExistsError(f"refusing to overwrite immutable package: {output_path}")

    report_dir = repo_root / "artifacts" / "scientific_recovery_v8" / "report"
    report = build_report(repo_root, report_dir)
    files, excluded = _candidate_files(repo_root)
    manifest = {
        "artifact_type": "scientific_recovery_v8_evidence_package_manifest_v1",
        "status": (
            "blocked" if report["status"] == "blocked" else "evidence_indexed_not_claim_ready"
        ),
        "report_artifact_sha256": report["artifact_sha256"],
        "package_scope": (
            "configs, manifests, compact logs, predictions, aggregates, audits, figures and docs"
        ),
        "excluded_scope": "datasets, caches and checkpoint bytes",
        "files": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
        "excluded": sorted(excluded, key=lambda item: item["path"]),
        "checkpoints": checkpoint_manifest(repo_root),
    }
    sign_artifact(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in files:
                _writestr(archive, path.relative_to(repo_root).as_posix(), path.read_bytes())
            _writestr(
                archive,
                "evidence_manifest.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        if not _zip_is_valid(temporary):
            raise RuntimeError("created ZIP did not pass integrity validation")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    outer_manifest = dict(manifest)
    outer_manifest["package_path"] = output_path.relative_to(repo_root).as_posix()
    outer_manifest["package_sha256"] = sha256_file(output_path)
    sign_artifact(outer_manifest)
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(outer_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return outer_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/scientific_recovery_v8/package/scientific_recovery_v8_evidence.zip"
        ),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    try:
        manifest = package_evidence(args.repo_root, output)
    except Exception as error:
        parser.exit(2, f"V8 evidence package failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"package_sha256": manifest["package_sha256"], "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
