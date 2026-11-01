"""Describe full-population platform drift using predictions only, without native targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from operational.release_research.protocol import ResearchRun, write_json
from scripts.release_research import read, verify_complete


def describe(parent: Path, output: Path) -> None:
    """Stratify existing scalar parity differences by local predicted magnitude, not truth."""
    verify_complete(parent)
    records = read(parent / "ORIGINAL_PARITY.json")
    if len(records) != 5805:
        raise ValueError("Require the full physical population")
    field = "PHASE_ONLY_EVENT_S7_TRAIN40.output_ttc"
    tokens = sorted(records)
    predictions = np.array([records[t]["original"] for t in tokens], dtype=float)
    differences = np.array([records[t]["fields"][field]["max_abs"] for t in tokens], dtype=float)
    failures = np.array([not records[t]["fields"][field]["passed"] for t in tokens])
    if not np.isfinite(predictions).all() or not np.isfinite(differences).all():
        raise ValueError("Undefined scalar parity values")
    run = ResearchRun(
        output,
        {
            "parent": parent / "COMPLETE.json",
            "parity": parent / "ORIGINAL_PARITY.json",
            "source:scripts/release_native_parity_distribution.py": Path(__file__),
        },
        {
            "experiment_id": "full-native-parity-distribution-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "scope": "descriptive platform drift; no accuracy metric",
        },
    )
    try:
        groups = []
        magnitude = np.abs(predictions)
        for low, high in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, float("inf"))):
            mask = (magnitude >= low) & (magnitude < high)
            groups.append(
                {
                    "predicted_magnitude_min_s": low,
                    "predicted_magnitude_max_s": high if np.isfinite(high) else None,
                    "queries": int(mask.sum()),
                    "failed": int(failures[mask].sum()),
                    "absolute_difference_p50_p95_p99_max_s": np.quantile(
                        differences[mask], [0.5, 0.95, 0.99, 1]
                    ).tolist()
                    if mask.any()
                    else None,
                }
            )
        result = {
            "queries": len(tokens),
            "failed": int(failures.sum()),
            "absolute_difference_p50_p95_p99_max_s": np.quantile(
                differences, [0.5, 0.95, 0.99, 1]
            ).tolist(),
            "by_predicted_magnitude": groups,
            "targets_read": False,
            "prediction_error_against_truth": False,
            "interpretation": "Magnitude uses local prediction; "
            "groups are not official scoring categories.",
        }
        write_json(output / "RESULTS.json", result)
        lines = [
            "# Deriva nativa respecto al archivo histórico",
            "",
            "Comparación de predicciones, sin etiquetas: "
            "estas diferencias no son error TTC frente a verdad.",
            f"Consultas: {len(tokens)}. "
            f"Fallan la tolerancia histórica de TTC: {int(failures.sum())}.",
            "Cuantiles p50/p95/p99/máximo de diferencia absoluta (s):",
            str(result["absolute_difference_p50_p95_p99_max_s"]),
            "",
            "| Magnitud de TTC local (s) | Consultas | Fallos de tolerancia | p95 diferencia (s) |",
            "|---|---:|---:|---:|",
        ]
        for group in groups:
            q = group["absolute_difference_p50_p95_p99_max_s"]
            lines.append(
                f"| {group['predicted_magnitude_min_s']}–{group['predicted_magnitude_max_s']} | "
                f"{group['queries']} | {group['failed']} | {q[1] if q else 'sin casos'} |"
            )
        lines += [
            "",
            "El último intervalo no tiene cota superior. Las categorías se definen por",
            "la predicción local, no por el TTC verdadero ni por categorías del scorer.",
            "No identifican la causa de la deriva ni recuperan la paridad histórica.",
            "",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    describe(args.parent, args.output)


if __name__ == "__main__":
    main()
