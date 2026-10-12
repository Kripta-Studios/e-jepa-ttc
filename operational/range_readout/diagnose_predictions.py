"""Reproduce persisted metrics and retain per-query losses and TTC subranges."""

from __future__ import annotations

import numpy as np
import pandas as pd

from operational.range_transfer.analyze import metrics

from .common import RUN, read, sha, write


def main() -> None:
    """No inference, fitting or selection: derive tables from frozen predictions."""
    checks = []
    summaries = []
    for path in sorted((RUN / "evaluation").glob("*/*/PREDICTIONS.csv")):
        receipt = read(path.parent / "RECEIPT.json")
        if sha(path) != receipt["predictions_sha256"]:
            raise ValueError("Predictions changed: " + str(path))
        frame = pd.read_csv(path, converters={"target_ttc": float, "prediction_ttc": np.float32})
        target = frame.target_ttc.to_numpy(np.float64)
        # CSV uses the shortest decimal that round-trips to the native FP32 output.
        native_pred = frame.prediction_ttc.to_numpy(np.float32)
        pred = native_pred.astype(np.float64)
        frame["prediction_ttc_float32_recovered"] = pred
        valid = np.isfinite(pred) & ((pred < 0) | (pred > 0.1))
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["MiD"] = np.where(
                valid, 10000 * np.abs(np.log1p(-0.1 / pred) - np.log1p(-0.1 / target)), np.nan
            )
            frame["native_aux_MiD"] = np.where(
                valid,
                10000 * np.abs(np.log1p(-0.1 / native_pred) - np.log1p(-0.1 / target)),
                np.nan,
            )
        frame["absolute_error_s"] = np.abs(pred - target)
        frame["signed_error_s"] = pred - target
        frame["sign_error"] = np.sign(pred) != np.sign(target)
        frame["warning_le_1s"] = (pred > 0.1) & (pred <= 1)
        frame["valid"] = valid
        frame["subrange"] = np.select(
            [
                target < 0,
                (target > 0.1) & (target <= 1),
                (target > 1) & (target <= 2),
                (target > 2) & (target <= 3),
                (target > 3) & (target <= 6),
                target > 6,
            ],
            ["negative", "0.1-1", "1-2", "2-3", "3-6", "6+"],
            default="invalid_target",
        )
        frame.to_csv(path.parent / "LOSSES_PER_QUERY.csv", index=False)
        computed = metrics(target, native_pred, frame.group_id.to_numpy())
        saved = read(path.parent / "METRICS.json")
        errors = []

        def compare(a: object, b: object, errors: list[float] = errors) -> None:
            if isinstance(a, dict) and isinstance(b, dict):
                if a.keys() != b.keys():
                    raise ValueError("Metric keys differ")
                for key in a:
                    compare(a[key], b[key])
            elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
                errors.append(abs(float(a) - float(b)))
                if not np.isclose(a, b, rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Recomputed metric differs: {a} vs {b}")
            elif a != b:
                raise ValueError(f"Metric value differs: {a} vs {b}")

        compare(computed, saved)
        checks.append(
            {
                "fit": path.parent.name,
                "rows": len(frame),
                "max_abs_error": max(errors, default=0),
                "sha256": sha(path),
                "prediction_precision": "native_float32_bits_recovered_from_CSV",
            }
        )
        grouped = (
            frame.groupby(["group_id", "subrange"], observed=True)
            .agg(
                queries=("sample_token", "size"),
                MiD=("MiD", "mean"),
                MAE_s=("absolute_error_s", "mean"),
                bias_s=("signed_error_s", "mean"),
                coverage=("valid", "mean"),
                warning_rate=("warning_le_1s", "mean"),
                sign_error_rate=("sign_error", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "fit", path.parent.name)
        grouped.insert(1, "stage", path.parent.parent.name)
        summaries.append(grouped)
    if not checks:
        raise ValueError("No completed evaluation predictions")
    pd.concat(summaries, ignore_index=True).to_csv(RUN / "SUBRANGES_BY_GROUP.csv", index=False)
    write(
        RUN / "qa/METRICS_REPRODUCTION.json",
        {
            "status": "PASS",
            "scope": "metrics_from_predictions_not_new_inference",
            "optimizer_updates": 0,
            "fits": checks,
        },
    )


if __name__ == "__main__":
    main()
