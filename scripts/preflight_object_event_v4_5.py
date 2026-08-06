#!/usr/bin/env python3
"""Fail-closed preflight for Object Event TTC v4.5 paired reciprocal MiD."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "488b433857090e525c447bf2974ac72639f25194"
EXPECTED_HASHES = {
    "scripts/train_e_jepa_object_event_v4_2.py": "c7cd7498cd490f3dacb4b1f4c478133b585787aed5a95ba556fae68838f928df",
    "src/e_jepa_ttc/models/object_event_v4_2.py": "6f802c15d1a9a33d9f5ae10c4321e9ab6ed43a845627b99bb316c6e89d6d35ad",
    "src/e_jepa_ttc/training/object_event_v4_2.py": "7a805e57b83191e883e9a2726607d4b5df20c36725e3e624885e82bae69ad2d0",
    "configs/experiment/e_jepa_garl_object_event_screen_v4_2.yaml": "fd3d7d3cbaca5242840ee6c8bc5e8b004821f23292940512969813de5b16173c",
}
REQUIRED_V44_FILES = (
    "src/e_jepa_ttc/object_event_v4_4.py",
    "artifacts/debug/object_event_v4_4_geometry_cv/summary.json",
)
REQUIRED_PREDICTION_COLUMNS = {
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


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _head() -> str:
    result = _git("rev-parse", "HEAD")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git rev-parse HEAD failed")
    return result.stdout.strip()


def _require_base_ancestor(head: str) -> None:
    result = _git("merge-base", "--is-ancestor", BASE_COMMIT, head)
    if result.returncode != 0:
        raise RuntimeError(f"HEAD {head} does not descend from v4.3 base {BASE_COMMIT}")


def _checkpoint_for_seed(run_root: Path, seed: int) -> Path:
    seed_dir = run_root / f"seed-{seed}"
    eligible = seed_dir / "eligible.pt"
    best = seed_dir / "best_observed.pt"
    if eligible.exists():
        return eligible
    if best.exists():
        return best
    raise FileNotFoundError(f"No v4.2 checkpoint for seed {seed}: {seed_dir}")


def _checkpoint_metadata(path: Path, seed: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    checkpoint_seed = int(payload.get("train_config", {}).get("seed", -1))
    if checkpoint_seed != seed:
        raise ValueError(
            f"Checkpoint seed mismatch for {path}: expected {seed}, got {checkpoint_seed}"
        )
    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "epoch": int(payload.get("epoch", 0)),
        "artifact_type": payload.get("artifact_type"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v42-run-root", type=Path, required=True)
    parser.add_argument("--v44-summary", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--aggregate-config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 13, 23))
    args = parser.parse_args()

    head = _head()
    _require_base_ancestor(head)
    checked_hashes: dict[str, str] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        checked_hashes[relative] = actual

    for relative in REQUIRED_V44_FILES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)

    cache_manifest = args.cache_manifest.resolve()
    if not cache_manifest.exists():
        raise FileNotFoundError(cache_manifest)
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if not isinstance(cache, dict):
        raise ValueError("cache manifest must be a JSON object")

    v44_summary_path = args.v44_summary.resolve()
    v44 = json.loads(v44_summary_path.read_text(encoding="utf-8"))
    if v44.get("artifact_type") != "object_event_v4_4_train_only_geometry_cv":
        raise ValueError(f"Unexpected v4.4 artifact: {v44.get('artifact_type')}")
    validation = v44.get("validation_metrics", {})
    hybrid_mid = (
        validation.get("hybrid", {}).get("official_eap", {}).get("weighted_mid")
    )
    if hybrid_mid is None or not 100.0 < float(hybrid_mid) < 400.0:
        raise ValueError(f"Invalid v4.4 hybrid weighted MiD: {hybrid_mid}")

    for path in (args.train_config.resolve(), args.aggregate_config.resolve()):
        if not path.exists():
            raise FileNotFoundError(path)

    run_root = args.v42_run_root.resolve()
    checkpoints: dict[str, Any] = {}
    for seed in args.seeds:
        seed_dir = run_root / f"seed-{seed}"
        summary_path = seed_dir / "summary.json"
        predictions_path = seed_dir / "validation_predictions.csv"
        if not summary_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(f"Missing v4.2 seed outputs: {seed_dir}")
        frame = pd.read_csv(predictions_path)
        missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {predictions_path}: {missing}")
        if frame.empty or frame.duplicated(["sequence_id", "sample_token", "track_id"]).any():
            raise ValueError(f"Invalid prediction identities in {predictions_path}")
        checkpoint = _checkpoint_for_seed(run_root, seed)
        checkpoints[str(seed)] = _checkpoint_metadata(checkpoint, seed)

    result = {
        "status": "passed",
        "head": head,
        "required_base_ancestor": BASE_COMMIT,
        "critical_v4_2_hashes": checked_hashes,
        "cache_manifest": cache_manifest.as_posix(),
        "cache_manifest_sha256": _sha256(cache_manifest),
        "v4_4_summary": v44_summary_path.as_posix(),
        "v4_4_status": v44.get("status"),
        "v4_4_hybrid_weighted_mid": float(hybrid_mid),
        "matching_seed_checkpoints": checkpoints,
        "scientific_contract": {
            "v4_2_architecture_hash_pinned": True,
            "v4_4_failed_geometry_is_baseline_only": True,
            "matching_seed_initialisation": True,
            "official_test_and_evttc_are_not_opened": True,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
