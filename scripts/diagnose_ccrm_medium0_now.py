#!/usr/bin/env python
"""
Diagnóstico offline de CCRm-medium-0 usando predictions.npz ya existentes.

No carga checkpoints, no usa CUDA, no modifica el protocolo y no abre el holdout.
Puede ejecutarse mientras continúa la matriz experimental.

Ejemplo:
    python scripts/diagnose_ccrm_medium0_now.py ^
      --prediction seed7=artifacts/runs/evttc32_article_ablation/base/seed7/ft30/predictions.npz ^
      --prediction seed13=artifacts/runs/evttc32_article_ablation/base/seed13/ft30/predictions.npz ^
      --output artifacts/diagnostics/ccrm_medium0_now
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TARGET_DEFAULT = "CCRm-medium-0-overlap-0"
COMPARATORS_DEFAULT = (
    "CCRm-low-0-overlap-0",
    "CCRm-medium-50-overlap-50",
    "CCRm-medium-100-overlap-100",
)
TTC_BINS = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 15.0, np.inf])


@dataclass(frozen=True)
class PredictionBlock:
    label: str
    split: str
    pred: np.ndarray
    true: np.ndarray
    sequence_id: np.ndarray
    global_index: np.ndarray
    timestamp_us: np.ndarray
    context_start_us: np.ndarray
    context_end_us: np.ndarray

    def select(self, sequence: str) -> "PredictionBlock":
        mask = self.sequence_id.astype(str) == sequence
        return PredictionBlock(
            label=self.label,
            split=self.split,
            pred=self.pred[mask],
            true=self.true[mask],
            sequence_id=self.sequence_id[mask],
            global_index=self.global_index[mask],
            timestamp_us=self.timestamp_us[mask],
            context_start_us=self.context_start_us[mask],
            context_end_us=self.context_end_us[mask],
        )


def _as_1d(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    return arr.reshape(-1)


def _require(npz: Any, key: str) -> np.ndarray:
    if key not in npz.files:
        raise KeyError(
            f"Falta la clave {key!r}. Claves disponibles: {sorted(npz.files)}"
        )
    return _as_1d(npz[key])


def load_prediction_file(label: str, path: Path) -> dict[str, PredictionBlock]:
    if not path.is_file():
        raise FileNotFoundError(path)

    blocks: dict[str, PredictionBlock] = {}
    with np.load(path, allow_pickle=True) as npz:
        for split in ("train", "validation"):
            pred_key = f"{split}_pred"
            if pred_key not in npz.files:
                continue

            pred = _require(npz, pred_key).astype(np.float64)
            true = _require(npz, f"{split}_true").astype(np.float64)
            sequence_id = _require(npz, f"{split}_sequence_id").astype(str)
            global_index = _require(npz, f"{split}_global_index").astype(np.int64)
            timestamp_us = _require(npz, f"{split}_timestamp_us").astype(np.int64)
            context_start_us = _require(
                npz, f"{split}_context_start_us"
            ).astype(np.int64)
            context_end_us = _require(
                npz, f"{split}_context_end_us"
            ).astype(np.int64)

            lengths = {
                pred.size,
                true.size,
                sequence_id.size,
                global_index.size,
                timestamp_us.size,
                context_start_us.size,
                context_end_us.size,
            }
            if len(lengths) != 1:
                raise ValueError(
                    f"Longitudes incompatibles en {path} split={split}: {lengths}"
                )

            blocks[split] = PredictionBlock(
                label=label,
                split=split,
                pred=pred,
                true=true,
                sequence_id=sequence_id,
                global_index=global_index,
                timestamp_us=timestamp_us,
                context_start_us=context_start_us,
                context_end_us=context_end_us,
            )

    if not blocks:
        raise ValueError(f"No se encontraron splits de predicción en {path}")
    return blocks


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    if pred.size == 0:
        return {
            "count": 0,
            "mae_s": math.nan,
            "rmse_s": math.nan,
            "bias_s": math.nan,
            "median_abs_error_s": math.nan,
            "mean_abs_relative_error_pct": math.nan,
            "underestimate_pct": math.nan,
            "overestimate_pct": math.nan,
            "large_error_gt_0_5s_pct": math.nan,
            "large_error_gt_1s_pct": math.nan,
        }

    residual = pred - true
    abs_error = np.abs(residual)
    denom = np.maximum(np.abs(true), 1e-8)
    return {
        "count": int(pred.size),
        "mae_s": float(abs_error.mean()),
        "rmse_s": float(np.sqrt(np.mean(residual**2))),
        "bias_s": float(residual.mean()),
        "median_abs_error_s": float(np.median(abs_error)),
        "mean_abs_relative_error_pct": float(np.mean(abs_error / denom) * 100.0),
        "underestimate_pct": float(np.mean(residual < 0.0) * 100.0),
        "overestimate_pct": float(np.mean(residual > 0.0) * 100.0),
        "large_error_gt_0_5s_pct": float(np.mean(abs_error > 0.5) * 100.0),
        "large_error_gt_1s_pct": float(np.mean(abs_error > 1.0) * 100.0),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    """Rangos medios para empates, sin depender de SciPy."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        mean_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = mean_rank
        i = j
    return ranks


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def identity_key(block: PredictionBlock, index: int) -> tuple[Any, ...]:
    return (
        str(block.sequence_id[index]),
        int(block.global_index[index]),
        int(block.timestamp_us[index]),
        int(block.context_start_us[index]),
        int(block.context_end_us[index]),
    )


def align_blocks(
    a: PredictionBlock, b: PredictionBlock
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[Any, ...]]]:
    map_a = {identity_key(a, i): i for i in range(a.pred.size)}
    map_b = {identity_key(b, i): i for i in range(b.pred.size)}
    keys = sorted(set(map_a).intersection(map_b))
    if not keys:
        raise ValueError(f"No hay ventanas comunes entre {a.label} y {b.label}")

    ia = np.asarray([map_a[key] for key in keys], dtype=np.int64)
    ib = np.asarray([map_b[key] for key in keys], dtype=np.int64)

    true_a = a.true[ia]
    true_b = b.true[ib]
    if not np.allclose(true_a, true_b, rtol=0.0, atol=1e-7):
        raise ValueError("Los targets no coinciden en las ventanas alineadas.")

    return a.pred[ia], b.pred[ib], true_a, keys


def bin_rows(label: str, block: PredictionBlock) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lo, hi in zip(TTC_BINS[:-1], TTC_BINS[1:]):
        mask = (block.true >= lo) & (block.true < hi)
        item: dict[str, Any] = {
            "label": label,
            "split": block.split,
            "sequence_id": (
                str(block.sequence_id[0]) if block.sequence_id.size else ""
            ),
            "ttc_bin_lower_s": float(lo),
            "ttc_bin_upper_s": None if np.isinf(hi) else float(hi),
        }
        item.update(metrics(block.pred[mask], block.true[mask]))
        rows.append(item)
    return rows


def temporal_rows(label: str, block: PredictionBlock, bins: int = 10) -> list[dict[str, Any]]:
    if block.pred.size == 0:
        return []
    order = np.argsort(block.timestamp_us)
    chunks = np.array_split(order, min(bins, order.size))
    rows: list[dict[str, Any]] = []
    t0 = int(block.timestamp_us[order[0]])
    for index, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        item: dict[str, Any] = {
            "label": label,
            "split": block.split,
            "sequence_id": str(block.sequence_id[chunk[0]]),
            "temporal_bin": index + 1,
            "start_offset_s": float(
                (int(block.timestamp_us[chunk[0]]) - t0) / 1_000_000.0
            ),
            "end_offset_s": float(
                (int(block.timestamp_us[chunk[-1]]) - t0) / 1_000_000.0
            ),
            "true_ttc_mean_s": float(block.true[chunk].mean()),
        }
        item.update(metrics(block.pred[chunk], block.true[chunk]))
        rows.append(item)
    return rows


def hard_window_rows(label: str, block: PredictionBlock, count: int = 30) -> list[dict[str, Any]]:
    residual = block.pred - block.true
    order = np.argsort(np.abs(residual))[::-1][:count]
    rows: list[dict[str, Any]] = []
    for rank, i in enumerate(order, start=1):
        rows.append(
            {
                "label": label,
                "rank": rank,
                "split": block.split,
                "sequence_id": str(block.sequence_id[i]),
                "global_index": int(block.global_index[i]),
                "timestamp_us": int(block.timestamp_us[i]),
                "context_start_us": int(block.context_start_us[i]),
                "context_end_us": int(block.context_end_us[i]),
                "ttc_true_s": float(block.true[i]),
                "ttc_pred_s": float(block.pred[i]),
                "residual_s": float(residual[i]),
                "abs_error_s": float(abs(residual[i])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_prediction_series(blocks: list[PredictionBlock], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    reference = blocks[0]
    order = np.argsort(reference.timestamp_us)
    t0 = reference.timestamp_us[order[0]]
    x = (reference.timestamp_us[order] - t0) / 1_000_000.0
    ax.plot(x, reference.true[order], label="TTC real", linewidth=2)
    for block in blocks:
        lookup = {
            identity_key(block, i): block.pred[i] for i in range(block.pred.size)
        }
        pred = np.asarray(
            [lookup.get(identity_key(reference, i), np.nan) for i in order]
        )
        ax.plot(x, pred, label=f"Pred {block.label}", alpha=0.8)
    ax.set_xlabel("Tiempo desde el inicio de la secuencia (s)")
    ax.set_ylabel("TTC (s)")
    ax.set_title("CCRm-medium-0: TTC real y predicho")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_abs_error_by_true_ttc(blocks: list[PredictionBlock], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for block in blocks:
        ax.scatter(
            block.true,
            np.abs(block.pred - block.true),
            s=12,
            alpha=0.45,
            label=block.label,
        )
    ax.set_xlabel("TTC real (s)")
    ax.set_ylabel("Error absoluto (s)")
    ax.set_title("Error según el TTC real")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_sequence_mae(summary_rows: list[dict[str, Any]], output: Path) -> None:
    labels = sorted({str(row["label"]) for row in summary_rows})
    sequences = sorted({str(row["sequence_id"]) for row in summary_rows})
    width = 0.8 / max(len(labels), 1)
    x = np.arange(len(sequences), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, label in enumerate(labels):
        values = []
        lookup = {
            str(row["sequence_id"]): float(row["mae_s"])
            for row in summary_rows
            if row["label"] == label
        }
        for sequence in sequences:
            values.append(lookup.get(sequence, np.nan))
        ax.bar(x + offset * width, values, width=width, label=label)

    ax.set_xticks(x + width * (len(labels) - 1) / 2.0)
    ax.set_xticklabels(sequences, rotation=25, ha="right")
    ax.set_ylabel("MAE (s)")
    ax.set_title("Comparación CCRm por secuencia y semilla")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def build_report(
    target: str,
    summary_rows: list[dict[str, Any]],
    agreement: dict[str, Any],
    output: Path,
) -> None:
    target_rows = [row for row in summary_rows if row["sequence_id"] == target]
    lines = [
        "# Diagnóstico provisional de CCRm-medium-0",
        "",
        "Este informe utiliza únicamente predicciones de train/validation ya guardadas.",
        "No modifica checkpoints, no reentrena y no abre el holdout.",
        "",
        "## Resultado por semilla",
        "",
        "| Semilla | MAE (s) | RMSE (s) | Bias (s) | >0,5 s (%) | >1 s (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in target_rows:
        lines.append(
            "| {label} | {mae_s:.6f} | {rmse_s:.6f} | {bias_s:.6f} | "
            "{large_error_gt_0_5s_pct:.2f} | {large_error_gt_1s_pct:.2f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Acuerdo entre semillas",
            "",
            f"- Ventanas alineadas: {agreement.get('aligned_count', 0)}",
            f"- Correlación de errores absolutos: "
            f"{agreement.get('absolute_error_pearson', math.nan):.4f}",
            f"- Spearman de errores absolutos: "
            f"{agreement.get('absolute_error_spearman', math.nan):.4f}",
            f"- Coincidencia del signo del error: "
            f"{agreement.get('residual_sign_agreement_pct', math.nan):.2f} %",
            f"- Solapamiento de las 30 peores ventanas: "
            f"{agreement.get('top30_hard_window_overlap_pct', math.nan):.2f} %",
            f"- MAE del ensemble de dos semillas: "
            f"{agreement.get('ensemble_mae_s', math.nan):.6f} s",
            "",
            "## Interpretación",
            "",
            "- Correlación alta y gran solapamiento de ventanas difíciles: "
            "el problema es estructural o de datos, no solo de inicialización.",
            "- Correlación baja y ensemble claramente mejor: domina la varianza "
            "de optimización entre semillas.",
            "- Bias negativo: el modelo subestima TTC (predice colisión demasiado pronto).",
            "- Bias positivo: el modelo sobreestima TTC (predice colisión demasiado tarde).",
            "- Error concentrado en TTC bajos: problema de seguridad cerca de la colisión.",
            "- Error concentrado en TTC altos: problema de calibración o escala.",
            "",
            "Los CSV `ttc_bins.csv`, `temporal_bins.csv` y `hard_windows.csv` "
            "permiten localizar exactamente dónde aparece el fallo.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_prediction_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "Formato esperado: etiqueta=ruta/a/predictions.npz"
        )
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("La etiqueta no puede estar vacía.")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Repetible: etiqueta=ruta/predictions.npz",
    )
    parser.add_argument("--target", default=TARGET_DEFAULT)
    parser.add_argument(
        "--comparator",
        action="append",
        default=None,
        help="Secuencia CCRm comparadora. Puede repetirse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/ccrm_medium0_now"),
    )
    args = parser.parse_args()

    prediction_specs = [parse_prediction_arg(raw) for raw in args.prediction]
    if len(prediction_specs) < 1:
        raise SystemExit("Se requiere al menos un predictions.npz.")

    comparators = tuple(args.comparator or COMPARATORS_DEFAULT)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    loaded = {
        label: load_prediction_file(label, path)
        for label, path in prediction_specs
    }

    target_blocks: list[PredictionBlock] = []
    summary_rows: list[dict[str, Any]] = []
    ttc_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []

    sequences_to_analyze = (args.target,) + comparators
    for label, blocks in loaded.items():
        for sequence in sequences_to_analyze:
            found: PredictionBlock | None = None
            for split in ("validation", "train"):
                block = blocks.get(split)
                if block is None:
                    continue
                selected = block.select(sequence)
                if selected.pred.size:
                    found = selected
                    break
            if found is None:
                print(f"AVISO: {label}: no se encontró {sequence}")
                continue

            row: dict[str, Any] = {
                "label": label,
                "split": found.split,
                "sequence_id": sequence,
            }
            row.update(metrics(found.pred, found.true))
            summary_rows.append(row)

            if sequence == args.target:
                target_blocks.append(found)
                ttc_rows.extend(bin_rows(label, found))
                time_rows.extend(temporal_rows(label, found))
                hard_rows.extend(hard_window_rows(label, found))

    if not target_blocks:
        raise SystemExit(f"No se encontró la secuencia objetivo {args.target!r}.")

    agreement: dict[str, Any] = {}
    if len(target_blocks) >= 2:
        a, b = target_blocks[:2]
        pred_a, pred_b, true, keys = align_blocks(a, b)
        err_a = pred_a - true
        err_b = pred_b - true
        abs_a = np.abs(err_a)
        abs_b = np.abs(err_b)
        top_count = min(30, len(keys))
        hard_a = set(np.argsort(abs_a)[::-1][:top_count].tolist())
        hard_b = set(np.argsort(abs_b)[::-1][:top_count].tolist())
        ensemble = (pred_a + pred_b) / 2.0

        agreement = {
            "labels": [a.label, b.label],
            "aligned_count": len(keys),
            "prediction_pearson": safe_corr(pred_a, pred_b),
            "residual_pearson": safe_corr(err_a, err_b),
            "absolute_error_pearson": safe_corr(abs_a, abs_b),
            "absolute_error_spearman": safe_corr(rankdata(abs_a), rankdata(abs_b)),
            "residual_sign_agreement_pct": float(
                np.mean(np.sign(err_a) == np.sign(err_b)) * 100.0
            ),
            "top30_hard_window_overlap_pct": float(
                len(hard_a.intersection(hard_b)) / max(top_count, 1) * 100.0
            ),
            "seed_a_metrics": metrics(pred_a, true),
            "seed_b_metrics": metrics(pred_b, true),
            "ensemble_mae_s": float(np.mean(np.abs(ensemble - true))),
            "ensemble_rmse_s": float(np.sqrt(np.mean((ensemble - true) ** 2))),
        }

    write_csv(output / "sequence_summary.csv", summary_rows)
    write_csv(output / "ttc_bins.csv", ttc_rows)
    write_csv(output / "temporal_bins.csv", time_rows)
    write_csv(output / "hard_windows.csv", hard_rows)

    payload = {
        "target_sequence": args.target,
        "comparators": list(comparators),
        "prediction_files": {
            label: str(path) for label, path in prediction_specs
        },
        "sequence_summary": summary_rows,
        "seed_agreement": agreement,
    }
    (output / "diagnostic.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_report(args.target, summary_rows, agreement, output / "REPORT.md")
    plot_prediction_series(target_blocks, output / "ttc_time_series.png")
    plot_abs_error_by_true_ttc(target_blocks, output / "error_vs_true_ttc.png")
    plot_sequence_mae(summary_rows, output / "ccrm_sequence_mae.png")

    print("DIAGNOSTIC_COMPLETE")
    print(f"Output: {output.resolve()}")
    if agreement:
        print(json.dumps(agreement, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
