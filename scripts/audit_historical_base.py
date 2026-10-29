"""Reproduce the historical BASE exactly without retraining or opening Benchmark-10."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.models import build_regressor
from e_jepa_ttc.training.supervised import evaluate_supervised_checkpoint
from e_jepa_ttc.utils.io import write_structured


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_differences(
    recorded: dict[str, Any],
    reproduced: dict[str, Any],
) -> dict[str, float]:
    names = (
        "mae_s",
        "rmse_s",
        "mean_abs_relative_error_pct",
        "median_abs_relative_error_pct",
        "log_mae",
        "log_rmse",
    )
    return {name: abs(float(recorded[name]) - float(reproduced[name])) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/features/evttc32_trainval/cache.npz"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/runs/evttc32_article_ablation/base/seed7/ft30"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audit/oge_sota/historical_base_reproduction.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    checkpoint_path = args.run_dir / "tiny_cnn_best.pt"
    metrics_path = args.run_dir / "metrics.json"
    recorded_predictions_path = args.run_dir / "predictions.npz"
    assert_no_sealed_benchmark_paths(
        (
            args.cache,
            args.run_dir,
            checkpoint_path,
            metrics_path,
            recorded_predictions_path,
            args.output,
        )
    )
    required = (
        args.cache,
        checkpoint_path,
        metrics_path,
        recorded_predictions_path,
    )
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Historical BASE evidence is incomplete: {missing}")

    recorded = json.loads(metrics_path.read_text(encoding="utf-8"))
    declared_cache_hash = str(recorded["cache_sha256"])
    actual_cache_hash = _sha256(args.cache)
    if declared_cache_hash != actual_cache_hash:
        raise ValueError("Historical BASE cache hash differs from its signed metrics.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "event-tubelet-transformer":
        raise ValueError("Historical BASE checkpoint is not the expected tubelet transformer.")
    model = build_regressor(
        "event-tubelet-transformer",
        in_channels=int(checkpoint["in_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    encoder_parameter_count = sum(parameter.numel() for parameter in model.encoder.parameters())

    reproduced = evaluate_supervised_checkpoint(
        cache_path=args.cache,
        checkpoint_path=checkpoint_path,
        output_path=args.output,
        batch_size=args.batch_size,
        device_name=args.device,
        evaluation_splits=("train", "validation"),
        allow_final_test_evaluation=False,
    )
    differences = _metric_differences(
        recorded["splits"]["validation"]["metrics"],
        reproduced["splits"]["validation"]["metrics"],
    )
    metric_parity = max(differences.values(), default=0.0) <= 1e-12

    reproduced_predictions_path = args.output.with_suffix(".predictions.npz")
    with (
        np.load(recorded_predictions_path, allow_pickle=False) as expected,
        np.load(reproduced_predictions_path, allow_pickle=False) as actual,
    ):
        keys_match = expected.files == actual.files
        array_parity = keys_match and all(
            np.array_equal(expected[key], actual[key]) for key in expected.files
        )

    audit = {
        **reproduced,
        "audit_status": "passed" if metric_parity and array_parity else "failed",
        "historical_role": "B0_HISTORICAL_BASE_EXACT",
        "scientific_scope": (
            "Exact historical checkpoint reproduction on its original cache and "
            "historical validation split; not a matched object-cache ablation."
        ),
        "architecture": {
            "encoder": "EventTubeletTransformerEncoder",
            "input_channels": int(checkpoint["in_channels"]),
            "event_channels": 10,
            "auxiliary_channels": int(checkpoint["in_channels"]) - 10,
            "embedding_dim": int(model.encoder.output_dim),
            "depth": len(model.encoder.layers),
            "heads": 6,
            "patch_size": int(model.encoder.patch_size),
            "pooling": "mean_over_final_dense_tokens",
            "head": "LayerNorm-Linear(192,96)-GELU-Dropout(0.1)-Linear(96,1)",
            "prediction_space": "log_ttc",
            "parameter_count": parameter_count,
            "encoder_parameter_count": encoder_parameter_count,
        },
        "training_record": {
            "epochs_maximum": int(recorded["epochs"]),
            "epochs_completed": int(recorded["epochs"]),
            "best_epoch": int(recorded["best_epoch"]),
            "batch_size": int(recorded["batch_size"]),
            "learning_rate": float(recorded["learning_rate"]),
            "weight_decay": float(recorded["weight_decay"]),
            "effective_train_count": int(recorded["effective_train_count"]),
            "checkpoint_selected_by": checkpoint.get("checkpoint_selected_by"),
            "pretrained_encoder": recorded["pretrained_encoder"],
        },
        "integrity": {
            "cache_sha256": actual_cache_hash,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "recorded_metrics_sha256": _sha256(metrics_path),
            "recorded_predictions_sha256": _sha256(recorded_predictions_path),
            "metric_absolute_differences": differences,
            "metric_parity_at_1e-12": metric_parity,
            "prediction_arrays_byte_equivalent": array_parity,
            "benchmark10_opened": False,
        },
    }
    write_structured(args.output, audit)
    if audit["audit_status"] != "passed":
        raise RuntimeError(f"Historical BASE reproduction failed: {audit['integrity']}")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
