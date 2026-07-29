#!/usr/bin/env python
"""
Diagnóstico offline de todas las secuencias de validation para comprobar:
- si el modelo aprende una relación TTC útil;
- si comprime el rango de TTC;
- si el fallo es específico de CCRm-medium-0;
- si las semillas fallan en las mismas ventanas.

No usa GPU, no carga checkpoints y no abre el holdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def one_dim(value: np.ndarray) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def load_validation(label: str, path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as npz:
        required = [
            "validation_pred",
            "validation_true",
            "validation_sequence_id",
            "validation_global_index",
            "validation_timestamp_us",
            "validation_context_start_us",
            "validation_context_end_us",
        ]
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise KeyError(f"{path}: faltan claves {missing}")
        return {
            "label": np.asarray([label]),
            "pred": one_dim(npz["validation_pred"]).astype(np.float64),
            "true": one_dim(npz["validation_true"]).astype(np.float64),
            "sequence_id": one_dim(npz["validation_sequence_id"]).astype(str),
            "global_index": one_dim(npz["validation_global_index"]).astype(np.int64),
            "timestamp_us": one_dim(npz["validation_timestamp_us"]).astype(np.int64),
            "context_start_us": one_dim(
                npz["validation_context_start_us"]
            ).astype(np.int64),
            "context_end_us": one_dim(npz["validation_context_end_us"]).astype(
                np.int64
            ),
        }


def identity(data: dict[str, np.ndarray], i: int) -> tuple[Any, ...]:
    return (
        str(data["sequence_id"][i]),
        int(data["global_index"][i]),
        int(data["timestamp_us"][i]),
        int(data["context_start_us"][i]),
        int(data["context_end_us"][i]),
    )


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def middle_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def sequence_metrics(
    label: str,
    sequence: str,
    pred: np.ndarray,
    true: np.ndarray,
    timestamp_us: np.ndarray,
) -> dict[str, Any]:
    residual = pred - true
    abs_error = np.abs(residual)

    if true.size >= 2 and np.std(true) > 1e-12:
        slope, intercept = np.polyfit(true, pred, 1)
    else:
        slope, intercept = math.nan, math.nan

    true_p05, true_p95 = np.percentile(true, [5, 95])
    pred_p05, pred_p95 = np.percentile(pred, [5, 95])
    true_range = float(true_p95 - true_p05)
    pred_range = float(pred_p95 - pred_p05)
    range_ratio = pred_range / true_range if true_range > 1e-12 else math.nan

    order = np.argsort(timestamp_us)
    chunks = np.array_split(order, min(10, order.size))
    first = chunks[0]
    last = chunks[-1]
    true_temporal_change = float(true[last].mean() - true[first].mean())
    pred_temporal_change = float(pred[last].mean() - pred[first].mean())
    temporal_change_ratio = (
        pred_temporal_change / true_temporal_change
        if abs(true_temporal_change) > 1e-12
        else math.nan
    )

    return {
        "label": label,
        "sequence_id": sequence,
        "count": int(true.size),
        "mae_s": float(abs_error.mean()),
        "rmse_s": float(np.sqrt(np.mean(residual**2))),
        "bias_s": float(residual.mean()),
        "median_abs_error_s": float(np.median(abs_error)),
        "pred_true_pearson": safe_corr(pred, true),
        "calibration_slope_pred_on_true": float(slope),
        "calibration_intercept_s": float(intercept),
        "true_p05_s": float(true_p05),
        "true_p95_s": float(true_p95),
        "pred_p05_s": float(pred_p05),
        "pred_p95_s": float(pred_p95),
        "p05_p95_range_ratio": float(range_ratio),
        "first_bin_true_mean_s": float(true[first].mean()),
        "first_bin_pred_mean_s": float(pred[first].mean()),
        "last_bin_true_mean_s": float(true[last].mean()),
        "last_bin_pred_mean_s": float(pred[last].mean()),
        "temporal_change_ratio": float(temporal_change_ratio),
        "underestimate_pct": float(np.mean(residual < 0) * 100.0),
        "large_error_gt_0_5s_pct": float(np.mean(abs_error > 0.5) * 100.0),
        "large_error_gt_1s_pct": float(np.mean(abs_error > 1.0) * 100.0),
    }


def seed_agreement(
    sequence: str,
    label_a: str,
    data_a: dict[str, np.ndarray],
    label_b: str,
    data_b: dict[str, np.ndarray],
) -> dict[str, Any]:
    idx_a = np.where(data_a["sequence_id"] == sequence)[0]
    idx_b = np.where(data_b["sequence_id"] == sequence)[0]
    map_a = {identity(data_a, int(i)): int(i) for i in idx_a}
    map_b = {identity(data_b, int(i)): int(i) for i in idx_b}
    keys = sorted(set(map_a).intersection(map_b))
    ia = np.asarray([map_a[key] for key in keys], dtype=np.int64)
    ib = np.asarray([map_b[key] for key in keys], dtype=np.int64)

    pred_a = data_a["pred"][ia]
    pred_b = data_b["pred"][ib]
    true = data_a["true"][ia]
    err_a = pred_a - true
    err_b = pred_b - true
    abs_a = np.abs(err_a)
    abs_b = np.abs(err_b)
    top_n = min(30, len(keys))
    hard_a = set(np.argsort(abs_a)[::-1][:top_n].tolist())
    hard_b = set(np.argsort(abs_b)[::-1][:top_n].tolist())
    ensemble = (pred_a + pred_b) / 2.0

    return {
        "sequence_id": sequence,
        "label_a": label_a,
        "label_b": label_b,
        "aligned_count": len(keys),
        "prediction_pearson": safe_corr(pred_a, pred_b),
        "residual_pearson": safe_corr(err_a, err_b),
        "absolute_error_pearson": safe_corr(abs_a, abs_b),
        "absolute_error_spearman": safe_corr(
            middle_rank(abs_a), middle_rank(abs_b)
        ),
        "residual_sign_agreement_pct": float(
            np.mean(np.sign(err_a) == np.sign(err_b)) * 100.0
        ),
        "top30_overlap_pct": float(
            len(hard_a.intersection(hard_b)) / max(top_n, 1) * 100.0
        ),
        "ensemble_mae_s": float(np.mean(np.abs(ensemble - true))),
        "ensemble_rmse_s": float(np.sqrt(np.mean((ensemble - true) ** 2))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_slopes(rows: list[dict[str, Any]], output: Path) -> None:
    sequences = sorted({str(row["sequence_id"]) for row in rows})
    labels = sorted({str(row["label"]) for row in rows})
    width = 0.8 / max(len(labels), 1)
    x = np.arange(len(sequences), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, label in enumerate(labels):
        lookup = {
            str(row["sequence_id"]): float(
                row["calibration_slope_pred_on_true"]
            )
            for row in rows
            if row["label"] == label
        }
        ax.bar(
            x + offset * width,
            [lookup.get(sequence, np.nan) for sequence in sequences],
            width=width,
            label=label,
        )
    ax.axhline(1.0, linewidth=1)
    ax.set_xticks(x + width * (len(labels) - 1) / 2.0)
    ax.set_xticklabels(sequences, rotation=25, ha="right")
    ax.set_ylabel("Pendiente predicción vs TTC real")
    ax.set_title("Pendiente < 1 indica compresión del rango TTC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def make_report(
    metrics_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    lines = [
        "# Diagnóstico de todas las secuencias de validation",
        "",
        "Pendiente cercana a 1 y correlación alta indican que el modelo sigue "
        "correctamente la escala TTC. Una pendiente claramente menor que 1 "
        "indica compresión hacia la media.",
        "",
        "## Métricas por semilla",
        "",
        "| Secuencia | Seed | MAE | Bias | Corr. pred-real | Pendiente | "
        "Ratio rango | Ratio cambio temporal |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        metrics_rows, key=lambda x: (str(x["sequence_id"]), str(x["label"]))
    ):
        lines.append(
            "| {sequence_id} | {label} | {mae_s:.4f} | {bias_s:.4f} | "
            "{pred_true_pearson:.4f} | "
            "{calibration_slope_pred_on_true:.4f} | "
            "{p05_p95_range_ratio:.4f} | "
            "{temporal_change_ratio:.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Acuerdo entre semillas",
            "",
            "| Secuencia | Corr. residuo | Corr. error absoluto | Top-30 común | "
            "MAE ensemble |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(agreement_rows, key=lambda x: str(x["sequence_id"])):
        lines.append(
            "| {sequence_id} | {residual_pearson:.4f} | "
            "{absolute_error_pearson:.4f} | {top30_overlap_pct:.2f}% | "
            "{ensemble_mae_s:.4f} |".format(**row)
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_prediction(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Formato: etiqueta=ruta/predictions.npz")
    label, value = raw.split("=", 1)
    return label.strip(), Path(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/all_validation_now"),
    )
    args = parser.parse_args()

    specs = [parse_prediction(value) for value in args.prediction]
    loaded = {label: load_validation(label, path) for label, path in specs}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    sequences = sorted(
        set.intersection(
            *[
                set(data["sequence_id"].astype(str).tolist())
                for data in loaded.values()
            ]
        )
    )

    metrics_rows: list[dict[str, Any]] = []
    for label, data in loaded.items():
        for sequence in sequences:
            mask = data["sequence_id"] == sequence
            metrics_rows.append(
                sequence_metrics(
                    label,
                    sequence,
                    data["pred"][mask],
                    data["true"][mask],
                    data["timestamp_us"][mask],
                )
            )

    agreement_rows: list[dict[str, Any]] = []
    if len(specs) >= 2:
        label_a, _ = specs[0]
        label_b, _ = specs[1]
        for sequence in sequences:
            agreement_rows.append(
                seed_agreement(
                    sequence,
                    label_a,
                    loaded[label_a],
                    label_b,
                    loaded[label_b],
                )
            )

    write_csv(output / "validation_calibration.csv", metrics_rows)
    write_csv(output / "validation_seed_agreement.csv", agreement_rows)
    payload = {
        "prediction_files": {label: str(path) for label, path in specs},
        "sequences": sequences,
        "metrics": metrics_rows,
        "seed_agreement": agreement_rows,
    }
    (output / "diagnostic.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    make_report(metrics_rows, agreement_rows, output / "REPORT.md")
    plot_slopes(metrics_rows, output / "validation_calibration_slopes.png")

    print("ALL_VALIDATION_DIAGNOSTIC_COMPLETE")
    print(f"Output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
