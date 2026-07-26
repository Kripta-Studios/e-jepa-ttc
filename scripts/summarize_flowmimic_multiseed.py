"""Build the signed E0/E1 full-schedule validation gate summary."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.artifacts.hashing import (
    compute_file_hash,
    sign_artifact,
    verify_artifact_hash,
)
from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity
from e_jepa_ttc.evaluation.bootstrap import paired_sequence_bootstrap_difference
from e_jepa_ttc.utils.io import read_structured, write_structured


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def _validation_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if payload.get("final_test_opened") is not False:
        raise ValueError("Every downstream run must record final_test_opened=false.")
    if "test" in set(payload.get("evaluation_splits", [])):
        raise ValueError("The E0/E1 validation gate must not evaluate test.")
    metrics = payload["splits"]["validation"]["metrics"]
    return {
        "mae_s": float(metrics["mae_s"]),
        "mean_abs_relative_error_pct": float(metrics["mean_abs_relative_error_pct"]),
        "rmse_s": float(metrics["rmse_s"]),
        "median_abs_error_s": float(metrics["median_abs_error_s"]),
    }


def _prediction_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as predictions:
        required = {
            "validation_pred",
            "validation_true",
            "validation_sequence_id",
            "validation_timestamp_us",
            "validation_global_index",
        }
        missing = required - set(predictions.files)
        if missing:
            raise ValueError(f"Prediction artifact {path} is missing {sorted(missing)}.")
        return {key: np.asarray(predictions[key]).copy() for key in required}


def _seed_bootstrap(values: np.ndarray, *, iterations: int = 10000) -> dict[str, Any]:
    if values.size < 2:
        raise ValueError("Paired seed bootstrap requires at least two seed differences.")
    rng = np.random.default_rng(20260726)
    samples = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "confidence": 0.95,
        "iterations": iterations,
        "paired_seed_count": int(values.size),
        "status": "paired_seed_bootstrap",
    }


def _run_paths(run_root: Path, variant: str, seed: int) -> tuple[Path, Path]:
    stem = variant.lower()
    return (
        run_root / f"flowmimic_full_{stem}_seed{seed}_ssl30" / "metrics.json",
        run_root / f"flowmimic_full_{stem}_seed{seed}_ft30" / "metrics.json",
    )


def summarize_flowmimic_multiseed(
    *,
    config_path: Path,
    output_path: Path | None = None,
    robustness_path: Path | None = None,
) -> dict[str, Any]:
    """Validate all paired runs and produce a signed validation-only summary."""

    config = read_structured(config_path)
    data_config = config["data"]
    experiment = config["experiment"]
    outputs = config["outputs"]
    run_root = Path(outputs["run_root"])
    seeds = [int(seed) for seed in experiment["seeds"]]
    variants = tuple(str(variant) for variant in experiment["variants"])
    if variants != ("E0", "E1") or len(seeds) != 3:
        raise ValueError("The frozen gate requires variants E0/E1 and exactly three seeds.")

    expected_cache_sha256 = str(data_config["cache_sha256"])
    protocol_version, protocol_sha256 = get_current_protocol_identity()
    if str(config["protocol"]["sha256"]) != protocol_sha256:
        raise ValueError("Config protocol SHA-256 differs from the verified frozen protocol.")

    rows: list[dict[str, Any]] = []
    predictions_by_variant_seed: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for variant in variants:
        variant_config = experiment["variants"][variant]
        for seed in seeds:
            pretrain_path, downstream_path = _run_paths(run_root, variant, seed)
            if not pretrain_path.exists() or not downstream_path.exists():
                raise FileNotFoundError(f"Incomplete {variant} seed {seed} full-schedule run.")
            pretrain = _read_json(pretrain_path)
            downstream = _read_json(downstream_path)
            if (
                str(pretrain["cache_sha256"]) != expected_cache_sha256
                or str(downstream["cache_sha256"]) != expected_cache_sha256
            ):
                raise ValueError(f"{variant} seed {seed} used a different physical cache.")
            if int(pretrain["seed"]) != seed or int(downstream["seed"]) != seed:
                raise ValueError(f"{variant} seed identity mismatch for seed {seed}.")
            if int(pretrain["epochs"]) != 30 or int(downstream["epochs"]) != 30:
                raise ValueError(f"{variant} seed {seed} did not complete the 30/30 schedule.")
            if float(pretrain["flowmimic_alignment_weight"]) != float(
                variant_config["flowmimic_alignment_weight"]
            ):
                raise ValueError(f"{variant} seed {seed} has the wrong alignment weight.")
            if float(pretrain["flowmimic_inverse_ttc_weight"]) != 0.0:
                raise ValueError(f"{variant} seed {seed} unexpectedly uses inverse-TTC SSL.")

            ssl_checkpoint = Path(pretrain["best_checkpoint"])
            checkpoint_sha256 = compute_file_hash(str(ssl_checkpoint))
            if checkpoint_sha256 != downstream["pretrained_encoder"]["checkpoint_sha256"]:
                raise ValueError(f"{variant} seed {seed} SSL/downstream checkpoint mismatch.")
            prediction_path = Path(downstream["predictions_path"])
            predictions_by_variant_seed[(variant, seed)] = _prediction_payload(prediction_path)
            health = pretrain["last"]["validation"]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "flowmimic_alignment_weight": float(pretrain["flowmimic_alignment_weight"]),
                    "pretrain": {
                        "metrics_path": pretrain_path.as_posix(),
                        "metrics_sha256": compute_file_hash(str(pretrain_path)),
                        "checkpoint_sha256": checkpoint_sha256,
                        "run_fingerprint": str(pretrain["run_fingerprint"]),
                        "best_epoch": int(pretrain["best_epoch"]),
                        "best_validation_loss": float(pretrain["best_loss"]),
                        "elapsed_seconds": float(pretrain["elapsed_seconds"]),
                        "last_validation_embedding_health": {
                            "context_std": float(health["context_embedding_std"]),
                            "prediction_std": float(health["pred_embedding_std"]),
                            "target_std": float(health["target_embedding_std"]),
                            "context_effective_rank": float(health["context_effective_rank"]),
                            "prediction_effective_rank": float(health["pred_effective_rank"]),
                            "target_effective_rank": float(health["target_effective_rank"]),
                        },
                    },
                    "downstream": {
                        "metrics_path": downstream_path.as_posix(),
                        "metrics_sha256": compute_file_hash(str(downstream_path)),
                        "prediction_path": prediction_path.as_posix(),
                        "prediction_sha256": compute_file_hash(str(prediction_path)),
                        "run_fingerprint": str(downstream["run_fingerprint"]),
                        "best_epoch": int(downstream["best_epoch"]),
                        "validation": _validation_metrics(downstream),
                    },
                }
            )

    paired_rows: list[dict[str, Any]] = []
    e0_mae: list[float] = []
    e1_mae: list[float] = []
    ensemble: dict[str, list[np.ndarray]] = {"E0": [], "E1": []}
    reference_identity: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for seed in seeds:
        e0_prediction = predictions_by_variant_seed[("E0", seed)]
        e1_prediction = predictions_by_variant_seed[("E1", seed)]
        for identity_field in (
            "validation_true",
            "validation_sequence_id",
            "validation_timestamp_us",
            "validation_global_index",
        ):
            np.testing.assert_array_equal(
                e0_prediction[identity_field],
                e1_prediction[identity_field],
                err_msg=f"E0/E1 are not paired on {identity_field} for seed {seed}.",
            )
        identity = (
            e0_prediction["validation_true"],
            e0_prediction["validation_sequence_id"],
            e0_prediction["validation_timestamp_us"],
        )
        if reference_identity is None:
            reference_identity = identity
        else:
            for expected, observed in zip(reference_identity, identity, strict=True):
                np.testing.assert_array_equal(expected, observed)
        e0_row = next(row for row in rows if row["variant"] == "E0" and row["seed"] == seed)
        e1_row = next(row for row in rows if row["variant"] == "E1" and row["seed"] == seed)
        e0_value = float(e0_row["downstream"]["validation"]["mae_s"])
        e1_value = float(e1_row["downstream"]["validation"]["mae_s"])
        e0_mae.append(e0_value)
        e1_mae.append(e1_value)
        ensemble["E0"].append(e0_prediction["validation_pred"].astype(np.float64))
        ensemble["E1"].append(e1_prediction["validation_pred"].astype(np.float64))
        paired_rows.append(
            {
                "seed": seed,
                "e0_mae_s": e0_value,
                "e1_mae_s": e1_value,
                "e1_minus_e0_mae_s": e1_value - e0_value,
                "e1_mae_reduction_vs_e0_pct": 100.0 * (e0_value - e1_value) / e0_value,
                "paired_sequence_bootstrap_mae_difference": (
                    paired_sequence_bootstrap_difference(
                        e0_prediction["validation_true"],
                        e0_prediction["validation_pred"],
                        e1_prediction["validation_pred"],
                        e0_prediction["validation_sequence_id"],
                        seed=seed,
                    )
                ),
            }
        )

    if reference_identity is None:
        raise ValueError("No paired predictions were loaded.")
    e0_values = np.asarray(e0_mae, dtype=np.float64)
    e1_values = np.asarray(e1_mae, dtype=np.float64)
    seed_differences = e1_values - e0_values
    ensemble_bootstrap = paired_sequence_bootstrap_difference(
        reference_identity[0],
        np.mean(np.stack(ensemble["E0"]), axis=0),
        np.mean(np.stack(ensemble["E1"]), axis=0),
        reference_identity[1],
        seed=20260726,
    )

    variant_summary = {}
    for variant, values in (("E0", e0_values), ("E1", e1_values)):
        variant_summary[variant] = {
            "validation_mae_s_mean": float(np.mean(values)),
            "validation_mae_s_std": float(np.std(values, ddof=1)),
            "validation_mae_s_values": values.tolist(),
        }

    robustness: dict[str, Any] | None = None
    if robustness_path is not None:
        robustness = _read_json(robustness_path)
        if not verify_artifact_hash(robustness):
            raise ValueError("Robustness artifact signature is invalid.")
        if robustness.get("final_test_opened") is not False:
            raise ValueError("Robustness artifact opened final test.")
        if str(robustness.get("cache_sha256")) != expected_cache_sha256:
            raise ValueError("Robustness artifact used a different clean cache.")
    robustness_gate_passed = bool(
        robustness is not None and robustness.get("gate", {}).get("passed") is True
    )

    sequence_count = int(np.unique(reference_identity[1].astype(str)).size)
    all_e1_wins = bool(np.all(seed_differences < 0.0))
    sequence_inference_informative = sequence_count > 1
    payload: dict[str, Any] = {
        "artifact_type": "flowmimic_e0_e1_multiseed_validation_summary",
        "schema_version": "1.0",
        "evidence_type": "validation_multiseed",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "config_path": config_path.as_posix(),
        "config_sha256": compute_file_hash(str(config_path)),
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "cache_sha256": expected_cache_sha256,
        "validation_split": "validation",
        "validation_sequence_count": sequence_count,
        "final_test_opened": False,
        "three_seed_complete": True,
        "rows": rows,
        "variant_summary": variant_summary,
        "paired_seed_results": paired_rows,
        "paired_seed_mae_difference_bootstrap": _seed_bootstrap(seed_differences),
        "ensemble_paired_sequence_bootstrap_mae_difference": ensemble_bootstrap,
        "robustness_artifact": (
            {
                "path": robustness_path.as_posix(),
                "sha256": compute_file_hash(str(robustness_path)),
            }
            if robustness_path is not None
            else None
        ),
        "selection": {
            "all_seed_point_estimates_favor_e1": all_e1_wins,
            "mean_e1_minus_e0_mae_s": float(np.mean(seed_differences)),
            "mean_e1_reduction_vs_e0_pct": float(
                100.0 * (np.mean(e0_values) - np.mean(e1_values)) / np.mean(e0_values)
            ),
            "sequence_bootstrap_informative": sequence_inference_informative,
            "robustness_complete": robustness is not None,
            "robustness_gate_passed": robustness_gate_passed,
        },
        "promotable_claim": bool(
            all_e1_wins and sequence_inference_informative and robustness_gate_passed
        ),
        "limitations": [
            "validation contains one complete sequence, so sequence bootstrap is degenerate",
            "CUDA execution is not bit-deterministic in the current environment",
            "CPLA-high is physically excluded and remains closed",
            "the local protocol is not directly comparable to official Garl-TTC/eAP results",
        ],
    }
    sign_artifact(payload)
    destination = output_path or Path(outputs["summary"])
    write_structured(destination, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the paired E0/E1 30-epoch, three-seed validation gate."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--robustness", type=Path)
    args = parser.parse_args()
    payload = summarize_flowmimic_multiseed(
        config_path=args.config,
        output_path=args.output,
        robustness_path=args.robustness,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
