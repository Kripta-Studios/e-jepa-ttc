"""Aggregate CARLA JEPA smoke/throughput evidence from generated metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e_jepa_ttc.data.carla_looming import CARLA_LOOMING_DATASET_ID  # noqa: E402
from e_jepa_ttc.utils.io import read_structured, write_structured  # noqa: E402

DEFAULT_RUNS = (
    Path("artifacts/runs/carla_jepa_smoke_seed42_v1"),
    Path("artifacts/runs/carla_jepa_throughput_probe_singleopen_v1"),
    Path("artifacts/runs/carla_jepa_throughput_probe_batch16_v1"),
    Path("artifacts/runs/carla_jepa_throughput_probe_batch32_v1"),
    Path("artifacts/runs/carla_jepa_throughput_probe_batch48_v1"),
    Path("artifacts/runs/carla_jepa_throughput_probe_batch96_v1"),
    Path("artifacts/runs/carla_jepa_throughput_probe_workers6_v1"),
    Path("artifacts/runs/carla_jepa_throughput_probe_workers12_v1"),
)


def _row(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    payload = read_structured(metrics_path)
    if payload.get("pretraining_dataset_id") != CARLA_LOOMING_DATASET_ID:
        raise ValueError(f"Run is not CARLA JEPA evidence: {metrics_path}.")
    train = int(payload["selected_train_pair_count"])
    validation = int(payload["selected_validation_pair_count"])
    elapsed = float(payload["elapsed_seconds"])
    config = payload["trainer_config"]
    row = {
        "run_dir": run_dir.as_posix(),
        "metrics_artifact_sha256": payload["artifact_sha256"],
        "best_checkpoint_sha256": payload["best_checkpoint_sha256"],
        "epochs": int(payload["epochs_completed"]),
        "batch_size": int(config["batch_size"]),
        "gradient_accumulation": int(config["gradient_accumulation"]),
        "workers": int(config["num_workers"]),
        "train_pairs": train,
        "validation_pairs": validation,
        "pair_observations": train + validation,
        "elapsed_seconds": elapsed,
        "pair_observations_per_second": (train + validation) / elapsed,
        "peak_vram_bytes": int(payload["peak_vram_bytes"]),
        "best_validation_loss": float(payload["best_validation_loss"]),
        "collapsed_dimension_fraction": float(
            payload["history"][-1]["validation"][
                "context_collapsed_dimension_fraction"
            ]
        ),
    }
    evaluations: dict[str, Any] = {}
    for role in ("validation", "test"):
        evaluation_path = run_dir / f"{role}_evaluation.json"
        if not evaluation_path.is_file():
            continue
        evaluation = read_structured(evaluation_path)
        if evaluation.get("checkpoint_sha256") != payload["best_checkpoint_sha256"]:
            raise ValueError(f"CARLA {role} evaluation uses a different checkpoint.")
        evaluations[role] = {
            "artifact_sha256": evaluation["artifact_sha256"],
            "evaluated_pair_count": int(evaluation["evaluated_pair_count"]),
            "sequence_count": int(evaluation["sequence_count"]),
            "loss": float(evaluation["metrics"]["loss"]),
            "collapsed_dimension_fraction": float(
                evaluation["metrics"]["context_collapsed_dimension_fraction"]
            ),
            "used_for_model_selection": bool(evaluation["used_for_model_selection"]),
        }
    if evaluations:
        row["holdout_evaluations"] = evaluations
    return row


def main() -> int:
    """Write a signed compact summary and recommend only the fastest probe."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/carla_jepa_smoke_and_throughput_v1.json"),
    )
    args = parser.parse_args()
    rows = [_row(path) for path in (args.run_dir or DEFAULT_RUNS)]
    throughput_rows = [row for row in rows if row["pair_observations"] >= 600]
    if not throughput_rows:
        raise ValueError("At least one probe with 600 pair observations is required.")
    selected = max(
        throughput_rows,
        key=lambda row: float(row["pair_observations_per_second"]),
    )
    payload = {
        "artifact_type": "carla_dvs_looming_jepa_probe_aggregate_v1",
        "dataset_id": CARLA_LOOMING_DATASET_ID,
        "scientific_role": "integration and throughput only; never EvTTC model selection",
        "runs": rows,
        "recommended_profile": {
            "batch_size": selected["batch_size"],
            "gradient_accumulation": selected["gradient_accumulation"],
            "workers": selected["workers"],
            "pair_observations_per_second": selected[
                "pair_observations_per_second"
            ],
            "source_run_dir": selected["run_dir"],
        },
        "benchmark10_opened": False,
    }
    write_structured(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
