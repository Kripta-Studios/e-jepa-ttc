"""Build a signed validation-only summary for the FlowMimic pilot matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity
from e_jepa_ttc.utils.io import write_structured


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def _validation_metrics(downstream: dict[str, Any]) -> dict[str, float]:
    if downstream.get("final_test_opened") is not False:
        raise ValueError("Pilot inputs must explicitly keep final_test_opened=false.")
    evaluation_splits = set(downstream.get("evaluation_splits", []))
    if "test" in evaluation_splits:
        raise ValueError("FlowMimic pilot summary cannot include test evaluation.")
    metrics = downstream["splits"]["validation"]["metrics"]
    return {
        "mae_s": float(metrics["mae_s"]),
        "mean_abs_relative_error_pct": float(metrics["mean_abs_relative_error_pct"]),
        "rmse_s": float(metrics["rmse_s"]),
        "median_abs_error_s": float(metrics["median_abs_error_s"]),
    }


def _reduction_pct(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def summarize_flowmimic_pilot(
    *,
    scratch_metrics_path: Path,
    variant_paths: dict[str, tuple[Path, Path]],
    output_path: Path,
) -> dict[str, Any]:
    """Validate and summarize scratch/E0/E1/E2 validation artifacts."""

    scratch = _read_json(scratch_metrics_path)
    scratch_validation = _validation_metrics(scratch)
    cache_sha256 = str(scratch["cache_sha256"])
    downstream_seed = int(scratch["seed"])
    rows: list[dict[str, Any]] = [
        {
            "id": "scratch",
            "flowmimic_alignment_weight": 0.0,
            "flowmimic_inverse_ttc_weight": 0.0,
            "pretrain": None,
            "downstream": {
                "metrics_path": scratch_metrics_path.as_posix(),
                "metrics_sha256": compute_file_hash(str(scratch_metrics_path)),
                "run_fingerprint": scratch["run_fingerprint"],
                "best_epoch": int(scratch["best_epoch"]),
                "seed": downstream_seed,
                "validation": scratch_validation,
            },
        }
    ]

    fingerprints = {str(scratch["run_fingerprint"])}
    for variant_id, (pretrain_path, downstream_path) in variant_paths.items():
        pretrain = _read_json(pretrain_path)
        downstream = _read_json(downstream_path)
        if pretrain["cache_sha256"] != cache_sha256 or downstream["cache_sha256"] != cache_sha256:
            raise ValueError(f"{variant_id} cache hash differs from scratch.")
        if int(downstream["seed"]) != downstream_seed:
            raise ValueError(f"{variant_id} downstream seed differs from scratch.")
        downstream_validation = _validation_metrics(downstream)
        checkpoint_path = Path(pretrain["best_checkpoint"])
        checkpoint_sha256 = compute_file_hash(str(checkpoint_path))
        recorded_checkpoint_sha256 = downstream["pretrained_encoder"]["checkpoint_sha256"]
        if checkpoint_sha256 != recorded_checkpoint_sha256:
            raise ValueError(f"{variant_id} downstream checkpoint hash does not match SSL best.")
        fingerprint = str(downstream["run_fingerprint"])
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate downstream fingerprint for {variant_id}.")
        fingerprints.add(fingerprint)
        rows.append(
            {
                "id": variant_id,
                "flowmimic_alignment_weight": float(pretrain["flowmimic_alignment_weight"]),
                "flowmimic_inverse_ttc_weight": float(pretrain["flowmimic_inverse_ttc_weight"]),
                "pretrain": {
                    "metrics_path": pretrain_path.as_posix(),
                    "metrics_sha256": compute_file_hash(str(pretrain_path)),
                    "run_fingerprint": pretrain["run_fingerprint"],
                    "best_epoch": int(pretrain["best_epoch"]),
                    "best_validation_loss": float(pretrain["best_loss"]),
                    "elapsed_seconds": float(pretrain["elapsed_seconds"]),
                    "checkpoint_sha256": checkpoint_sha256,
                    "seed": int(pretrain["seed"]),
                },
                "downstream": {
                    "metrics_path": downstream_path.as_posix(),
                    "metrics_sha256": compute_file_hash(str(downstream_path)),
                    "run_fingerprint": fingerprint,
                    "best_epoch": int(downstream["best_epoch"]),
                    "seed": int(downstream["seed"]),
                    "validation": downstream_validation,
                },
            }
        )

    row_by_id = {row["id"]: row for row in rows}
    required = {"scratch", "E0", "E1", "E2"}
    if set(row_by_id) != required:
        raise ValueError(f"Expected variants {sorted(required)}, got {sorted(row_by_id)}.")
    scratch_mae = scratch_validation["mae_s"]
    e0_mae = row_by_id["E0"]["downstream"]["validation"]["mae_s"]
    for row in rows:
        mae = row["downstream"]["validation"]["mae_s"]
        row["mae_reduction_vs_scratch_pct"] = _reduction_pct(scratch_mae, mae)
        row["mae_reduction_vs_e0_pct"] = _reduction_pct(e0_mae, mae)

    best_row = min(rows, key=lambda row: row["downstream"]["validation"]["mae_s"])
    protocol_version, protocol_sha256 = get_current_protocol_identity()
    payload: dict[str, Any] = {
        "artifact_type": "flowmimic_validation_pilot_summary",
        "schema_version": "1.0",
        "evidence_type": "validation_pilot",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "cache_sha256": cache_sha256,
        "validation_split": "validation",
        "final_test_opened": False,
        "single_seed_pilot": True,
        "three_seed_complete": False,
        "promotable_claim": False,
        "rows": rows,
        "selection": {
            "best_validation_variant": best_row["id"],
            "best_validation_mae_s": best_row["downstream"]["validation"]["mae_s"],
            "next_gate": "repeat E0 and E1 with independent SSL/downstream seeds 7,13,21",
        },
        "limitations": [
            "single SSL/downstream seed",
            "CUDA execution is not bit-deterministic in the current environment",
            "validation-only pilot; CPLA-high test is physically absent from the cache",
            "not comparable to official Garl-TTC without matching ROI and sequence protocol",
        ],
    }
    sign_artifact(payload)
    write_structured(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    for variant in ("e0", "e1", "e2"):
        parser.add_argument(f"--{variant}-pretrain", type=Path, required=True)
        parser.add_argument(f"--{variant}-downstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize_flowmimic_pilot(
        scratch_metrics_path=args.scratch,
        variant_paths={
            "E0": (args.e0_pretrain, args.e0_downstream),
            "E1": (args.e1_pretrain, args.e1_downstream),
            "E2": (args.e2_pretrain, args.e2_downstream),
        },
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
