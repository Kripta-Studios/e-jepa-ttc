"""Regenerate matched public-label comparisons without fitting or choosing weights."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from operational.sota_evidence.metrics import paired_macro
from operational.sota_evidence.run import digest, json_safe, read
from operational.train40_system.durable_io import atomic_json

METHODS = ("H8_median3", "Direct_median3", "public_Garl_event_lhr", "public_Garl_rgb_event_full")


def measure(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Retain missing predictions and make full-cohort metrics unavailable on failure."""
    if truth.ndim != 1 or truth.shape != prediction.shape or not np.isfinite(truth).all():
        raise ValueError("Aligned finite ground truth is required")
    if (truth == 0).any():
        raise ValueError("Ground truth zero is outside the declared eligibility contract")
    finite = np.isfinite(prediction)
    urgent = (truth > 0) & (truth <= 1)
    missed = urgent & (~finite | (prediction <= 0) | (prediction > 1))
    result: dict[str, Any] = {
        "n": len(truth),
        "finite_n": int(finite.sum()),
        "coverage": float(finite.mean()) if len(truth) else None,
        "mae_s": None,
        "median_ae_s": None,
        "p95_ae_s": None,
        "bias_s": None,
        "underestimate_percent": None,
        "wrong_sign_percent": None,
        "rte_percent": None,
        "top1percent_error_share": None,
        "urgent_gt_n": int(urgent.sum()),
        "urgent_miss_n": int(missed.sum()),
        "urgent_miss_percent": float(missed.sum() / urgent.sum() * 100) if urgent.any() else None,
    }
    if not len(truth) or not finite.all():
        return result
    error = prediction - truth
    absolute = np.abs(error)
    count = max(1, int(np.ceil(len(truth) * 0.01)))
    result.update(
        mae_s=float(absolute.mean()),
        median_ae_s=float(np.median(absolute)),
        p95_ae_s=float(np.quantile(absolute, 0.95)),
        bias_s=float(error.mean()),
        underestimate_percent=float((error < 0).mean() * 100),
        wrong_sign_percent=float((np.sign(prediction) != np.sign(truth)).mean() * 100),
        rte_percent=float((absolute / np.abs(truth)).mean() * 100),
        top1percent_error_share=float(np.sort(absolute)[-count:].sum() / absolute.sum())
        if absolute.sum() > 0
        else 0.0,
    )
    return result


def run(root: Path, output: Path) -> None:
    """Verify checkpoint and prediction identities, then publish descriptive comparisons."""
    output.mkdir(parents=True, exist_ok=True)
    revision = root / "artifacts/ttc_revision_20261009"
    historical = root / "artifacts/sota_campaign_20261008"
    prior_inputs = read(root / "artifacts/sota_evidence_20261010/INPUTS.json")
    native = read(historical / "dev32_expanded/INFERENCE_FREEZE.json")["models"]["garl"]
    full = read(historical / "dev32_expanded_rgb/INFERENCE_FREEZE.json")["model"]
    identities = {}
    for name, binding, checkpoint in (
        (
            "event_only",
            native,
            root / "artifacts/train40_system_20261005/public_garl/paper_event_only_lhr.pth",
        ),
        ("rgb_event", full, Path(full["checkpoint"])),
    ):
        sha = digest(checkpoint)
        if sha != binding["checkpoint_sha256"]:
            raise ValueError(f"Downloaded {name} checkpoint changed")
        identities[name] = {
            "path": str(checkpoint),
            "sha256": sha,
            "historical_strict_load": binding["strict_state_load"],
        }
    summary, by_band, pairs, worst = [], [], [], []
    for dataset in ("DEV32", "FCWD"):
        path = revision / f"{dataset}_PREDICTIONS.csv"
        expected = prior_inputs[str(path.relative_to(root))]
        if digest(path) != expected or read(path.with_suffix(".json"))["output_sha256"] != expected:
            raise ValueError("Frozen prediction bytes changed")
        frame = pd.read_csv(path)
        if frame.query_id.duplicated().any():
            raise ValueError("Duplicate matched query")
        eligible = np.isfinite(frame.truth_ttc_seconds) & (frame.truth_ttc_seconds != 0)
        scored = frame.loc[eligible]
        truth = scored.truth_ttc_seconds.to_numpy(float)
        sequence = scored.sequence_id.to_numpy(str)
        labels = np.select(
            [truth < 0, truth <= 1, truth <= 3, truth <= 6, truth <= 10],
            ["negative", "(0,1]", "(1,3]", "(3,6]", "(6,10]"],
            default=">10",
        )
        for policy in ("native", "equal_clip_60_sensitivity"):
            predictions = {}
            for method in METHODS:
                prediction = scored[method].to_numpy(float)
                if policy != "native":
                    prediction = np.where(
                        np.isfinite(prediction), np.clip(prediction, -60, 60), prediction
                    )
                predictions[method] = prediction
                item = {
                    "dataset": dataset,
                    "policy": policy,
                    "method": method,
                    "population_n": len(frame),
                    "gt_ineligible_n": int((~eligible).sum()),
                    "sequences": len(np.unique(sequence)),
                    **measure(truth, prediction),
                }
                summary.append(item)
                for label in np.unique(labels):
                    mask = labels == label
                    by_band.append(
                        {
                            "dataset": dataset,
                            "policy": policy,
                            "method": method,
                            "band": label,
                            **measure(truth[mask], prediction[mask]),
                        }
                    )
                if policy == "native":
                    errors = np.abs(prediction - truth)
                    for index in np.argsort(np.where(np.isfinite(errors), errors, -1))[-10:][::-1]:
                        if not np.isfinite(errors[index]):
                            continue
                        worst.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "query_id": scored.iloc[index].query_id,
                                "sequence_id": sequence[index],
                                "truth_s": truth[index],
                                "prediction_s": prediction[index],
                                "absolute_error_s": errors[index],
                            }
                        )
            for ours in METHODS[:2]:
                for comparator in METHODS[2:]:
                    pairs.append(
                        {
                            "dataset": dataset,
                            "policy": policy,
                            "ours": ours,
                            "comparator": comparator,
                            **paired_macro(
                                truth, predictions[ours], predictions[comparator], sequence
                            ),
                        }
                    )
    for name, values in (
        ("METRICS", summary),
        ("ERROR_BANDS", by_band),
        ("PAIRED_SEQUENCE_BOOTSTRAP", pairs),
        ("LARGEST_ERRORS", worst),
    ):
        pd.DataFrame(values).to_csv(output / f"{name}.csv", index=False)
    latency_source = revision / "cache512/LATENCY_RAW.csv"
    latency = pd.read_csv(latency_source)
    latency = latency.loc[~latency.warmup.astype(str).str.lower().eq("true")]
    latency_rows = []
    for (mode, system), values in latency.groupby(["mode", "system"]):
        for field in ("cpu_ms", "e2e_ms", "cuda_stream_ms"):
            latency_rows.append(
                {
                    "mode": mode,
                    "system": system,
                    "stage": field,
                    "observations": len(values),
                    "queries": values.query_id.nunique(),
                    "mean_ms": values[field].mean(),
                    "median_ms": values[field].median(),
                    "p95_ms": values[field].quantile(0.95),
                }
            )
    pd.DataFrame(latency_rows).to_csv(output / "LATENCY.csv", index=False)
    atomic_json(
        output / "RESULT.json",
        cast(
            dict,
            json_safe(
                {
                    "status": "COMPLETE_PUBLIC_LABEL_COMPARISON",
                    "utc": datetime.now(UTC).isoformat(),
                    "checkpoint_identities": identities,
                    "optimizer_updates": 0,
                    "gpu_seconds": 0,
                    "predictions_reused_after_hash_verification": True,
                    "official_test_score_available": False,
                    "code_sha256": digest(Path(__file__)),
                    "latency_input_sha256": digest(latency_source),
                    "prediction_inputs": {
                        d: digest(revision / f"{d}_PREDICTIONS.csv") for d in ("DEV32", "FCWD")
                    },
                }
            ),
        ),
    )
    lines = [
        "# Comparación con los checkpoints GarlTTC descargados",
        "",
        "Resultados recalculados de predicciones congeladas con SHA-256 verificado. "
        "Dev32 es exploratorio y ya expuesto; FCWD tiene solo tres secuencias. "
        "No es una puntuación oficial de test12 ni demuestra SOTA.",
        "",
        "El [informe MiD](MID_REPORT.md) añade el scorer oficial, FR, bandas TTC, "
        "casos extremos y bootstrap pareado por secuencia. El orden por MAE no "
        "equivale al orden por MiD.",
        "",
        "| Datos | Método | n | Cobertura | MAE (s) | Mediana AE (s) | p95 AE (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["policy"] != "native":
            continue
        values = [
            f"{row[key]:.3f}" if row[key] is not None else "N/D"
            for key in ("coverage", "mae_s", "median_ae_s", "p95_ae_s")
        ]
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['n']} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "Sensibilidad separada: mismo límite ±60 s para todos (Dev32).",
        "",
        "| Método | MAE (s) |",
        "|---|---:|",
    ]
    for row in summary:
        if row["dataset"] == "DEV32" and row["policy"] == "equal_clip_60_sensitivity":
            lines.append(f"| {row['method']} | {row['mae_s']:.3f} |")
    lines += [
        "",
        "Coste medido anteriormente: ruta cronológica con caché cruda de 512 MiB, "
        "40 consultas y 200 observaciones por sistema tras excluir warmups.",
        "",
        "| Sistema | CPU media (ms) | GPU media (ms) | Total medio (ms) | p95 total (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for system in ("h8_fast_three", "garl_event_only", "garl_full"):
        timing = {r["stage"]: r for r in latency_rows if r["system"] == system}
        lines.append(
            f"| {system} | {timing['cpu_ms']['mean_ms']:.1f} | "
            f"{timing['cuda_stream_ms']['mean_ms']:.1f} | "
            f"{timing['e2e_ms']['mean_ms']:.1f} | {timing['e2e_ms']['p95_ms']:.1f} |"
        )
    lines += [
        "",
        "Aviso urgente: GT positivo ≤1 s y predicción positiva ≤1 s. "
        "Una salida no finita o de signo incorrecto cuenta como fallo.",
        "",
        "| Método | Urgentes Dev32 | Fallos | Tasa de fallo (%) |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        if row["dataset"] == "DEV32" and row["policy"] == "native":
            lines.append(
                f"| {row['method']} | {row['urgent_gt_n']} | {row['urgent_miss_n']} | "
                f"{row['urgent_miss_percent']:.2f} |"
            )
    lines += [
        "",
        "Condiciones y lectura:",
        "",
        "- H8 usa tres cabezas sobre productores compartidos y más contexto temporal. "
        "Direct es una variante experimental; H8 sigue siendo el candidato fijado.",
        "- Garl conserva su conversión nativa de alturas a TTC, sin clipping. H8 tiene "
        "un límite nativo ±60 s. La sensibilidad con ±60 s para todos está separada en los CSV.",
        "- No se eliminan errores extremos ni se descartan predicciones faltantes. "
        "Las métricas completas quedan N/D si falta alguna predicción de la cohorte elegible.",
        "- ERROR_BANDS.csv muestra sesgo y subestimación por rango. LARGEST_ERRORS.csv "
        "permite identificar las consultas extremas. El bootstrap usa secuencias enteras.",
        "- FCWD RGB+eventos no es puntuable: falta una transformación geométrica verificada.",
        "- LATENCY.csv procede de las mediciones locales anteriores, sin warmups, "
        "separando las rutas de caché. H8 incluye sus tres cabezas. No equivale al FPS del paper.",
        "- Test12 contiene etiquetas privadas: comparar entre predicciones no mide exactitud. "
        "La puntuación oficial sigue pendiente de CodaBench.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the comparison from existing immutable artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
