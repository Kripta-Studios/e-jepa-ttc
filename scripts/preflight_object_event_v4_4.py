#!/usr/bin/env python3
"""Fail-closed preflight for Object Event TTC v4.4 geometry CV."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEAD = "488b433857090e525c447bf2974ac72639f25194"
EXPECTED_HASHES = {
    "scripts/train_e_jepa_object_event_v4_2.py": "c7cd7498cd490f3dacb4b1f4c478133b585787aed5a95ba556fae68838f928df",
    "src/e_jepa_ttc/models/object_event_v4_2.py": "6f802c15d1a9a33d9f5ae10c4321e9ab6ed43a845627b99bb316c6e89d6d35ad",
    "src/e_jepa_ttc/training/object_event_v4_2.py": "7a805e57b83191e883e9a2726607d4b5df20c36725e3e624885e82bae69ad2d0",
    "configs/experiment/e_jepa_garl_object_event_screen_v4_2.yaml": "fd3d7d3cbaca5242840ee6c8bc5e8b004821f23292940512969813de5b16173c",
}
REQUIRED_COLUMNS = {
    "sequence_id",
    "sample_token",
    "track_id",
    "delta_t_s",
    "target_ttc_s",
    "target_expansion",
    "prediction_expansion",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git rev-parse HEAD failed")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    head = _head()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"Expected HEAD {EXPECTED_HEAD}, got {head}")

    checked_hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Hash mismatch for {relative}: expected {expected}, got {actual}")
        checked_hashes[relative] = actual

    cache_manifest = args.cache_manifest.resolve()
    if not cache_manifest.exists():
        raise FileNotFoundError(cache_manifest)
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if not isinstance(cache, dict):
        raise ValueError("cache manifest must be a JSON object")

    config = args.config.resolve()
    if not config.exists():
        raise FileNotFoundError(config)

    run_root = args.run_root.resolve()
    run_checks: dict[str, object] = {}
    for seed in (7, 13, 23):
        seed_dir = run_root / f"seed-{seed}"
        summary_path = seed_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        split_checks: dict[str, object] = {}
        for split in ("train", "validation"):
            path = seed_dir / f"{split}_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
            if missing:
                raise ValueError(f"Missing columns in {path}: {missing}")
            if frame.empty or frame.duplicated(["sequence_id", "sample_token", "track_id"]).any():
                raise ValueError(f"Invalid prediction identities in {path}")
            split_checks[split] = {
                "path": path.as_posix(),
                "rows": int(len(frame)),
                "sequences": int(frame["sequence_id"].nunique()),
            }
        run_checks[str(seed)] = {
            "status": summary.get("status"),
            "screen_passed": bool(summary.get("screen_passed", False)),
            "splits": split_checks,
        }

    result = {
        "status": "passed",
        "head": head,
        "expected_head": EXPECTED_HEAD,
        "critical_hashes": checked_hashes,
        "cache_manifest": cache_manifest.as_posix(),
        "cache_manifest_sha256": _sha256(cache_manifest),
        "config": config.as_posix(),
        "run_root": run_root.as_posix(),
        "seed_outputs": run_checks,
        "scientific_contract": {
            "seed_23_failure_is_allowed_as_input": True,
            "validation_labels_will_not_be_used_for_fitting": True,
            "critical_v4_2_files_are_hash_pinned": True,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
