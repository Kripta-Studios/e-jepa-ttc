"""Regenerate public external-transfer diagnostics from sealed prediction CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file

OWN = ("E_H1_MATCHED", "E_CTX_MATCHED", "R_H1", "R_CTX", "F_TRUE", "F_ZERO")
REFERENCES = (
    "H8_seed7",
    "H8_seed13",
    "H8_seed23",
    "public_Garl_event_lhr",
    "public_Garl_rgb_event_full",
)
BANDS = (("c", 0, 3), ("s", 3, 6), ("l", 6, 10), ("n", -10, 0))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _numbers(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) if row[key] else np.nan for row in rows], np.float64)


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Independent direct calculation; every missing or invalid case is counted."""
    labelled = np.isfinite(truth)
    paired = labelled & np.isfinite(prediction)
    error = prediction[paired] - truth[paired]
    nonzero = paired & (truth != 0)
    bands = {}
    for name, lo, hi in BANDS:
        mask = labelled & (truth > lo) & (truth <= hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.abs(np.log1p(-0.1 / truth[mask]) - np.log1p(-0.1 / prediction[mask])) * 1e4
        usable = np.isfinite(values)
        bands[name] = {
            "rows": int(mask.sum()),
            "invalid_phase_rows": int((~usable).sum()),
            "MiD_finite_rows": _number(float(values[usable].mean())) if usable.any() else None,
        }
    positive = paired & (truth > 0)
    return {
        "rows": len(truth),
        "labelled": int(labelled.sum()),
        "finite_paired": int(paired.sum()),
        "missing_prediction_on_label": int((labelled & ~np.isfinite(prediction)).sum()),
        "MAE_s": _number(float(np.abs(error).mean())) if error.size else None,
        "mean_bias_s": _number(float(error.mean())) if error.size else None,
        "median_bias_s": _number(float(np.median(error))) if error.size else None,
        "RTE_pct": _number(
            float(
                np.mean(np.abs(prediction[nonzero] - truth[nonzero]) / np.abs(truth[nonzero])) * 100
            )
        )
        if nonzero.any()
        else None,
        "positive_overestimate_pct": float(np.mean(prediction[positive] > truth[positive]) * 100)
        if positive.any()
        else None,
        "positive_wrong_sign": int((positive & (prediction <= 0)).sum()),
        "abs_prediction_at_least_59_999": int((paired & (np.abs(prediction) >= 59.999)).sum()),
        "bands": bands,
    }


def build(seal_path: Path, scored_csv: Path, output: Path, title: str) -> dict[str, Any]:
    """Verify the prediction seal and compare identical rows in each reported view."""
    seal = read_json_shared(seal_path)
    prediction_path = Path(seal["predictions_path"])
    if seal["status"] != "COMPLETE" or seal["labels_read"] is not False:
        raise ValueError("Complete target-free prediction seal required")
    if sha256_file(prediction_path) != seal["predictions_sha256"]:
        raise ValueError("Prediction bytes changed after sealing")
    own = _read(prediction_path)
    rows = _read(scored_csv)
    original = {row["query_id"]: row for row in own}
    if (
        len(original) != len(own)
        or len(rows) != len(own)
        or {row["query_id"] for row in rows} != set(original)
    ):
        raise ValueError("Scored population differs from sealed predictions")
    for row in rows:
        for key in (*OWN, "anchor_us", "sequence_id", "rgb_available", "rgb_unavailable_reason"):
            if row[key] != original[row["query_id"]][key]:
                raise ValueError(f"Scoring modified sealed field {key}")
    methods = [name for name in (*OWN, *REFERENCES) if name in rows[0]]
    truth = _numbers(rows, "truth_ttc_seconds")
    available = np.asarray([r["rgb_available"].lower() in {"true", "1"} for r in rows])
    groups = np.asarray([r["sequence_id"] for r in rows])
    masks = {
        "all_queries": np.ones(len(rows), bool),
        "same_RGB_available_queries": available,
        "RGB_unavailable_queries": ~available,
    }
    masks.update({"sequence_" + name: groups == name for name in sorted(set(groups))})
    results = {
        scope: {name: metrics(truth[mask], _numbers(rows, name)[mask]) for name in methods}
        for scope, mask in masks.items()
    }
    receipt = {
        "status": "COMPLETE",
        "title": title,
        "rows": len(rows),
        "rgb_available": int(available.sum()),
        "prediction_seal_sha256": sha256_file(seal_path),
        "scored_csv_sha256": sha256_file(scored_csv),
        "views": results,
        "optimizer_updates": 0,
        "metric_definition": "MiD=1e4*abs(log1p(-0.1/GT)-log1p(-0.1/pred)); pooled by band",
        "limitations": [
            "Exposed exploratory datasets",
            "RGB availability selects a different cohort",
            "MiD on finite phases must be read with invalid counts",
            "No official SOTA comparison",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "DIAGNOSTICS.json", receipt)
    lines = [
        f"# {title}",
        "",
        f"{len(rows)} consultas; {int(available.sum())} con RGB disponible. Cero updates.",
        "",
        "Resultados exploratorios sobre conjuntos expuestos. "
        "No son una clasificación oficial ni evidencia SOTA.",
        "",
        "MiD es la media agrupada de errores de fase por banda; "
        "no es la media por secuencia de V13. "
        "Se muestran los casos de fase no válida y no se imputa una banda ausente.",
        "",
    ]

    def fmt(x: float | None) -> str:
        return "—" if x is None else f"{x:.3f}"

    for scope in ("all_queries", "same_RGB_available_queries", "RGB_unavailable_queries"):
        lines += [
            f"## {scope}",
            "",
            "| Modelo | Pares finitos / etiquetas | MAE s | RTE % | Sesgo medio s | "
            "MiDc | MiDs | MiDl | MiDn | Fases inválidas c/s/l/n |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for name in methods:
            m = results[scope][name]
            cells = [
                name,
                f"{m['finite_paired']}/{m['labelled']}",
                fmt(m["MAE_s"]),
                fmt(m["RTE_pct"]),
                fmt(m["mean_bias_s"]),
            ]
            cells += [fmt(m["bands"][b]["MiD_finite_rows"]) for b, _, _ in BANDS]
            cells += ["/".join(str(m["bands"][b]["invalid_phase_rows"]) for b, _, _ in BANDS)]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Diagnóstico por secuencia",
        "",
        "Las métricas, sesgos, signos y predicciones en el límite de ±60 s "
        "están en `DIAGNOSTICS.json`.",
        "El subconjunto RGB se fija por entradas y calidad del registro, nunca por el error TTC. "
        "Comparar modelos dentro de una misma tabla; "
        "las cabezas RGB sin fallback tienen menor cobertura en la tabla completa.",
        "",
        "Regeneración: `python -m operational.external_transfer_report --seal SEAL "
        "--scored SCORED_PREDICTIONS.csv --output DIRECTORIO --title TITULO`.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    build(args.seal, args.scored, args.output, args.title)


if __name__ == "__main__":
    main()
