"""Freeze a CV-complete EvTTC candidate before any sealed-benchmark inference."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.utils.io import write_structured


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-role", choices=("SINGLE_REALTIME", "ENSEMBLE_ACCURACY"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    status = _git("status", "--porcelain")
    if status and not args.allow_dirty:
        raise RuntimeError("Final freeze requires a clean Git worktree.")
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    row = next(
        (candidate for candidate in aggregate["ranking"] if candidate["variant"] == args.variant),
        None,
    )
    if row is None:
        raise ValueError(f"Variant {args.variant!r} is absent from the aggregate.")
    if not row["complete_for_final_selection"]:
        raise ValueError("Variant is not complete across the required folds and seeds.")
    checkpoints: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_config = checkpoint["model_config"]
        if reference_config is None:
            reference_config = model_config
        elif model_config != reference_config:
            raise ValueError("Ensemble checkpoints do not share an identical model config.")
        checkpoints.append(
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "epoch": int(checkpoint["epoch"]),
                "run_fingerprint": checkpoint["run_fingerprint"],
            }
        )
    if args.candidate_role == "SINGLE_REALTIME" and len(checkpoints) != 1:
        raise ValueError("SINGLE_REALTIME must contain exactly one checkpoint.")
    if args.candidate_role == "ENSEMBLE_ACCURACY" and len(checkpoints) < 2:
        raise ValueError("ENSEMBLE_ACCURACY must contain multiple checkpoints.")
    payload: dict[str, Any] = {
        "artifact_type": "evttc_final_freeze_manifest_v1",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "candidate_name": args.candidate_name,
        "candidate_role": args.candidate_role,
        "variant": args.variant,
        "git_commit": _git("rev-parse", "HEAD"),
        "dirty_worktree": bool(status),
        "aggregate_path": args.aggregate.as_posix(),
        "aggregate_sha256": _sha256(args.aggregate),
        "aggregate_protocol": aggregate["protocol"],
        "cv_run_count": row["run_count"],
        "cv_sequence_macro_selection_score_mean": row[
            "sequence_macro_selection_score_mean"
        ],
        "cv_sequence_macro_selection_score_std": row[
            "sequence_macro_selection_score_std"
        ],
        "cv_sequence_macro_mean_relative_error_mean": row[
            "sequence_macro_mean_relative_error_mean"
        ],
        "cv_sequence_macro_mean_relative_error_std": row[
            "sequence_macro_mean_relative_error_std"
        ],
        "cv_sequence_macro_mae_s_mean": row["sequence_macro_mae_s_mean"],
        "cv_sequence_macro_mae_s_std": row["sequence_macro_mae_s_std"],
        "model_config": reference_config,
        "checkpoints": checkpoints,
        "benchmark10_opened": False,
        "predictions_frozen_before_benchmark": True,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["freeze_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_structured(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
