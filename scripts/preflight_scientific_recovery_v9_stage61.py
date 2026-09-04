"""Fail-closed preflight for the Stage 61/62 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    expected_package = "0d09475424a797fc1bbe029afd44c94afb39635f701eba02810711c38b2e824c"
    if compute_file_hash(str(args.package)) != expected_package:
        raise ValueError("handoff package SHA-256 mismatch")
    pins = json.loads((args.handoff_root / "SOURCE_PINS.json").read_text(encoding="utf-8"))
    if (
        _git(repo, "rev-parse", "6c9cd1ef5f85c6b9b7fb5c2ccbbdde5c11a39181^{tree}")
        != pins["base_tree"]
    ):
        raise ValueError("starting tree mismatch")
    dirty = bool(_git(repo, "status", "--porcelain"))
    if args.require_clean and dirty:
        raise ValueError("training requires a clean frozen worktree")
    for relative, expected in pins["relevant_git_blob_shas"].items():
        if (
            _git(repo, "rev-parse", f"6c9cd1ef5f85c6b9b7fb5c2ccbbdde5c11a39181:{relative}")
            != expected
        ):
            raise ValueError(f"pinned source blob mismatch: {relative}")
    manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    if not verify_artifact_hash(manifest):
        raise ValueError("cache manifest signature mismatch")
    checkpoints = sorted(args.router_root.glob("outer_fold*_seed7/a5/*/train/model_best.pt"))
    c2f = sorted(args.router_root.glob("outer_fold*_seed7/c2f/*/expert_artifact.json"))
    if len(checkpoints) != 12 or len(c2f) != 12:
        raise ValueError("V8 nested prerequisites are not 12/12 A5 and 12/12 C2F")
    free = shutil.disk_usage(repo).free
    if free < 100 * 1024**3:
        raise ValueError("preflight requires at least 100 GiB free")
    result = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v9_stage61_preflight_v1",
            "status": "passed",
            "training_commit": _git(repo, "rev-parse", "HEAD"),
            "git_dirty": dirty,
            "package_sha256": expected_package,
            "base_commit": pins["base_commit"],
            "base_tree": pins["base_tree"],
            "source_pins_sha256": compute_file_hash(str(args.handoff_root / "SOURCE_PINS.json")),
            "cache_manifest_sha256": compute_file_hash(str(args.cache_root / "manifest.json")),
            "a5_producers": len(checkpoints),
            "c2f_producers": len(c2f),
            "disk_free_bytes": free,
            "sealed_paths_opened": False,
            "sha256_implementation": hashlib.sha256().name,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
