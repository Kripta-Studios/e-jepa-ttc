"""Audit the local, immutable Garl-TTC release and its input roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CHECKPOINT_NAMES = (
    "paper_event_only_lhr.pth",
    "paper_visual_only_lhr.pth",
    "paper_ours_full.pth",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(release_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(release_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _check_imports(release_root: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(release_root) + os.pathsep + current
    code = (
        "import garl_ttc.config, garl_ttc.datasets.event_representation, "
        "garl_ttc.models.ttc_network"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
        "status": "pass" if result.returncode == 0 else "fail",
    }


def _sequence_ids_from_parquet(path: Path) -> set[str]:
    """Read only the sequence-id column needed by the coverage audit."""

    frame = pd.read_parquet(path, columns=["sequence_id"])
    if "sequence_id" not in frame.columns:
        raise ValueError(f"Parquet has no sequence_id column: {path}")
    return {str(value) for value in frame["sequence_id"].dropna().unique().tolist()}


def audit_dataset_coverage(
    release_root: Path,
    *,
    eap_root: Path,
    garlttc_root: Path,
) -> dict[str, Any]:
    """Audit public train-sequence coverage without downloading or inferring labels.

    The official split and the locally mounted public parquets are deliberately
    reported separately.  An incomplete public snapshot is a valid audit
    outcome, but it must block any claim that a local run reproduces the
    official train46 experiment.
    """

    split_path = release_root / "configs" / "splits" / "train.txt"
    eap_path = eap_root.resolve() / "data" / "train.parquet"
    garl_path = garlttc_root.resolve() / "data" / "train.parquet"
    required = (split_path, eap_path, garl_path)
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        return {
            "artifact_type": "garl_dataset_coverage_v1",
            "status": "fail",
            "coverage_complete": False,
            "train_sequence_coverage": "0/0",
            "missing_files": missing,
            "errors": [f"coverage input missing: {path}" for path in missing],
            "retraining_claim": "blocked",
        }

    expected = {
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    observed = {
        "eap": _sequence_ids_from_parquet(eap_path),
        "garlttc": _sequence_ids_from_parquet(garl_path),
    }
    missing_by_root = {
        name: sorted(expected - sequence_ids) for name, sequence_ids in observed.items()
    }
    extra_by_root = {
        name: sorted(sequence_ids - expected) for name, sequence_ids in observed.items()
    }
    complete_by_root = {name: sequence_ids == expected for name, sequence_ids in observed.items()}
    return {
        "artifact_type": "garl_dataset_coverage_v1",
        "status": "pass",
        "coverage_complete": all(complete_by_root.values()),
        "coverage_status": (
            "complete_train46" if all(complete_by_root.values()) else "incomplete_public_train"
        ),
        "train_sequence_coverage": f"{len(observed['garlttc'] & expected)}/{len(expected)}",
        "official_split": {
            "path": split_path.as_posix(),
            "sha256": _sha256_file(split_path),
            "sequence_count": len(expected),
            "sequence_ids": sorted(expected),
        },
        "local_parquets": {
            "eap": {
                "path": eap_path.as_posix(),
                "sha256": _sha256_file(eap_path),
                "sequence_count": len(observed["eap"]),
                "sequence_ids": sorted(observed["eap"]),
            },
            "garlttc": {
                "path": garl_path.as_posix(),
                "sha256": _sha256_file(garl_path),
                "sequence_count": len(observed["garlttc"]),
                "sequence_ids": sorted(observed["garlttc"]),
            },
        },
        "missing_by_root": missing_by_root,
        "extra_by_root": extra_by_root,
        "complete_by_root": complete_by_root,
        "snapshot_check": {
            "status": "not_performed_no_download",
            "download_authorization_required": True,
        },
        "retraining_claim": (
            "official_train46_reproduction"
            if all(complete_by_root.values())
            else "public_train40_retraining_only"
        ),
        "retraining_claim_allowed": all(complete_by_root.values()),
        "errors": [],
    }


def audit_release(
    release_root: Path,
    *,
    eap_root: Path | None = None,
    garlttc_root: Path | None = None,
    expected_commit: str | None = None,
    minimum_free_gib: float = 1.0,
) -> dict[str, Any]:
    """Return a reproducible release audit without downloading or mutating data."""

    release_root = release_root.resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    if not release_root.is_dir():
        raise FileNotFoundError(f"Garl-TTC release root not found: {release_root}")
    try:
        commit = _git(release_root, "rev-parse", "HEAD")
        dirty = _git(release_root, "status", "--porcelain")
        checks["git"] = {
            "commit": commit,
            "expected_commit": expected_commit,
            "dirty": bool(dirty),
            "status": "pass" if not dirty else "fail",
        }
        if dirty:
            errors.append("official release worktree is dirty")
        if expected_commit is not None and commit != expected_commit:
            errors.append(f"official release commit mismatch: {commit} != {expected_commit}")
    except (OSError, subprocess.CalledProcessError) as exc:
        checks["git"] = {"status": "fail", "error": str(exc)}
        errors.append(f"official release is not a usable git repository: {exc}")

    config_path = release_root / "configs" / "garl_ttc_eventdecoder.yaml"
    split_paths = [release_root / "configs" / "splits" / name for name in ("train.txt", "test.txt")]
    checkpoints = {}
    for name in CHECKPOINT_NAMES:
        path = release_root / "checkpoints" / name
        if not path.is_file():
            errors.append(f"missing official checkpoint: {path}")
            checkpoints[name] = {"status": "fail", "path": path.as_posix()}
        else:
            checkpoints[name] = {
                "status": "pass",
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    checks["config"] = {
        "path": config_path.as_posix(),
        "exists": config_path.is_file(),
        "split_files": {path.name: path.is_file() for path in split_paths},
        "status": "pass"
        if config_path.is_file() and all(path.is_file() for path in split_paths)
        else "fail",
    }
    if not config_path.is_file() or not all(path.is_file() for path in split_paths):
        errors.append("official config or split asset is missing")
    checks["checkpoints"] = checkpoints

    input_roots = {}
    for name, root, required in (
        ("eap", eap_root, ("data/train.parquet",)),
        ("garlttc", garlttc_root, ("data/train.parquet", "data/test_inputs.parquet")),
    ):
        if root is None:
            input_roots[name] = {"status": "not_requested"}
            continue
        resolved = root.resolve()
        missing = [relative for relative in required if not (resolved / relative).is_file()]
        input_roots[name] = {
            "root": resolved.as_posix(),
            "required_files": list(required),
            "missing": missing,
            "status": "pass" if not missing else "fail",
        }
        if missing:
            errors.append(f"{name} input root missing: {missing}")
    checks["input_roots"] = input_roots
    if eap_root is not None and garlttc_root is not None:
        try:
            checks["dataset_coverage"] = audit_dataset_coverage(
                release_root,
                eap_root=eap_root,
                garlttc_root=garlttc_root,
            )
        except (OSError, ValueError, KeyError) as exc:
            checks["dataset_coverage"] = {
                "artifact_type": "garl_dataset_coverage_v1",
                "status": "fail",
                "coverage_complete": False,
                "errors": [str(exc)],
            }
            errors.append(f"dataset coverage audit failed: {exc}")
    else:
        checks["dataset_coverage"] = {
            "artifact_type": "garl_dataset_coverage_v1",
            "status": "not_requested",
        }
    checks["imports"] = _check_imports(release_root)
    if checks["imports"]["status"] != "pass":
        errors.append("official release imports failed in the active environment")

    disk = shutil.disk_usage(release_root)
    free_gib = disk.free / 1024**3
    checks["runtime"] = {
        "python": sys.version,
        "platform": sys.platform,
        "free_disk_gib": round(free_gib, 3),
        "minimum_free_gib": minimum_free_gib,
        "status": "pass" if free_gib >= minimum_free_gib else "fail",
    }
    if free_gib < minimum_free_gib:
        errors.append("insufficient free disk for an audit run")
    return {
        "artifact_type": "garl_official_release_audit_v1",
        "status": "pass" if not errors else "fail",
        "release_root": release_root.as_posix(),
        "checks": checks,
        "errors": errors,
    }


def audit_official_release(checkpoint_path: Path) -> dict[str, Any]:
    """Backward-compatible passive audit for one checkpoint file."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return {
        "checkpoint_path": checkpoint_path.as_posix(),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256_file(checkpoint_path),
        "top_level_keys": None,
        "keys_inspected": False,
        "audit_mode": "passive_file_inventory",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--eap-root", type=Path)
    parser.add_argument("--garlttc-root", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--minimum-free-gib", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.release_root is None:
        if args.checkpoint is None:
            parser.error("--release-root or legacy --checkpoint is required")
        report = audit_official_release(args.checkpoint)
    else:
        report = audit_release(
            args.release_root,
            eap_root=args.eap_root,
            garlttc_root=args.garlttc_root,
            expected_commit=args.expected_commit,
            minimum_free_gib=args.minimum_free_gib,
        )
    destination = args.output or (args.output_dir / "audit.json" if args.output_dir else None)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        coverage = report.get("checks", {}).get("dataset_coverage")
        if isinstance(coverage, dict) and args.output_dir is not None:
            coverage_path = args.output_dir / "dataset_coverage.json"
            coverage_path.write_text(
                json.dumps(coverage, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(text, end="")
    return 0 if report.get("status") in {"pass", None} else 1


if __name__ == "__main__":
    raise SystemExit(main())
