"""Freeze the three from-scratch Garl grouped-development commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from scripts.run_garl_matched_screen import EXPECTED_RELEASE_COMMIT  # noqa: E402

PROTOCOL_PATH = ROOT / "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
CACHE_MANIFEST = ROOT / "artifacts/cache/garl_budget_matched_s1_8192_v2/manifest.json"
IDENTITY_METADATA = (
    ROOT / "artifacts/scientific_recovery_master_v3/garl_budget_subset/train_data.parquet"
)
TRAINING_SCRIPT = ROOT / "scripts/train_garl_matched_from_cache.py"
OUTPUT_PATH = ROOT / "configs/protocol/scientific_recovery_v5_garl_grouped_runs.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    """Build commands without reading performance values or public validation rows."""

    if not verify_artifact_hash(protocol):
        raise ValueError("grouped-development protocol signature is invalid")
    if protocol.get("status") != "frozen_before_a8_results":
        raise ValueError("grouped-development protocol is not frozen")
    checks = protocol.get("checks", {})
    if checks.get("public_validation_used_for_selection") is not False:
        raise ValueError("public validation may not select grouped Garl")
    if checks.get("private_test_opened") is not False:
        raise ValueError("private/test must remain closed")
    common = (
        ".\\.venv\\Scripts\\python.exe scripts/train_garl_matched_from_cache.py "
        "--release-root E:\\Garl-TTC "
        "--cache-manifest artifacts/cache/garl_budget_matched_s1_8192_v2/manifest.json "
        "--development-protocol "
        "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json "
        "--identity-metadata "
        "artifacts/scientific_recovery_master_v3/garl_budget_subset/train_data.parquet "
        "--device cuda --seed 7 --epochs 18 --batch-size 32 --num-workers 0 "
        "--minimum-epochs 8 --early-stopping-patience 5 "
        "--maximum-runtime-hours 8"
    )
    runs = []
    for fold in protocol["folds"]:
        index = int(fold["fold"])
        output = f"artifacts/runs/scientific_recovery_v5_garl_fold_chain_fold{index}_seed7"
        runs.append(
            {
                "fold": index,
                "run_name": Path(output).name,
                "train_rows": fold["train_rows"],
                "dev_rows": fold["dev_rows"],
                "train_sample_tokens_sha256": fold["train_sample_tokens_sha256"],
                "dev_sample_tokens_sha256": fold["dev_sample_tokens_sha256"],
                "seed": 7,
                "from_scratch": True,
                "pretrained_checkpoint": None,
                "command": f"{common} --fold {index} --output-dir {output}",
            }
        )
    return {
        "artifact_type": "scientific_recovery_v5_garl_grouped_frozen_runs_v1",
        "status": "frozen_before_grouped_garl_training",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "protocol": {
            "path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "file_sha256": _sha256(PROTOCOL_PATH),
            "artifact_sha256": protocol["artifact_sha256"],
        },
        "sources": {
            "training_script": {
                "path": TRAINING_SCRIPT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(TRAINING_SCRIPT),
            },
            "cache_manifest": {
                "path": CACHE_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": _sha256(CACHE_MANIFEST),
                "artifact_sha256": json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))[
                    "artifact_sha256"
                ],
            },
            "identity_metadata": {
                "path": IDENTITY_METADATA.relative_to(ROOT).as_posix(),
                "sha256": _sha256(IDENTITY_METADATA),
            },
            "official_release_commit": EXPECTED_RELEASE_COMMIT,
        },
        "runs": runs,
        "contracts": {
            "sequential_cuda_runs": True,
            "num_workers": 0,
            "same_frozen_folds_as_ejepa": True,
            "from_scratch_per_fold": True,
            "pretrained_release_checkpoint_used": False,
            "event_only": True,
            "oracle_roi_preprocessing": True,
            "preprocessing_identical_to_ejepa": False,
            "comparison_scope": ("exact-sample, target, budget, metric and oracle-ROI matched"),
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.resolve(strict=True).read_text(encoding="utf-8"))
    manifest = build_manifest(protocol)
    sign_artifact(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(args.output), "runs": len(manifest["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
