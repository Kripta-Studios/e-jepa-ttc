"""Regenerate C1 tables and interpretation from its completed admitted evidence."""

from __future__ import annotations

from operational.range_transfer.common import read, sha, write

from .common import ROOT, RUN
from .geometry_evaluate import BASELINE
from .geometry_train import CONTROL, DIRECTORY, FIT


def build() -> dict:
    """Verify prediction/metric bindings before reporting positive and negative outcomes."""
    evidence = read(DIRECTORY / "evidence/COMPLETE.json")
    selection = read(DIRECTORY / "SELECTION_FINAL.json")
    if evidence["status"] != "C1_READY_FOR_INTERPRETATION" or evidence["selection"] != selection:
        raise ValueError("Complete C1 evidence and frozen selection required")
    rows, bindings = [], {}
    for role in ("P", "H"):
        for name in (BASELINE, CONTROL, FIT):
            folder = (DIRECTORY if name == FIT else RUN) / "predictions" / role / name
            receipt = read(folder / "COMPLETE.json")
            metrics = read(folder / "METRICS.json")
            key = f"{role}/{name}"
            proof = evidence["verification"][key]
            predictions, metric_hash = (
                sha(folder / "predictions.parquet"),
                sha(folder / "METRICS.json"),
            )
            if not (
                receipt["status"] == "COMPLETE"
                and proof["status"] == "PASS"
                and predictions
                == receipt["sha256"]
                == evidence["cohorts"][key]
                == proof["prediction_sha256"]
                and metric_hash == proof["metrics_sha256"]
                and metrics["num_rows"] == receipt["rows"] == proof["rows"]
            ):
                raise ValueError(f"C1 report evidence changed: {key}")
            bindings[key] = {"predictions": predictions, "metrics": metric_hash}
            rows.append(
                {
                    "role": role,
                    "model": name,
                    "rows": metrics["num_rows"],
                    "overall_MiD": metrics["overall_MiD"],
                    **{band: value["group_macro_mid"] for band, value in metrics["bins"].items()},
                    "mae_s": metrics["diagnostics"]["mae_s"],
                    "sign_error_rate": metrics["diagnostics"]["sign_error_rate"],
                    "p95_ae_s": metrics["diagnostics"]["p95_ae_s"],
                }
            )
    bootstrap = read(DIRECTORY / "bootstrap/H" / CONTROL / "SUMMARY.json")
    if (
        bootstrap["binding"]["reference_sha256"] != bindings[f"H/{CONTROL}"]["predictions"]
        or bootstrap["binding"]["candidate_sha256"] != bindings[f"H/{FIT}"]["predictions"]
    ):
        raise ValueError("Bootstrap belongs to another contrast")
    journal = read(DIRECTORY / "fits" / FIT / "UPDATE_JOURNAL.json")
    endpoint = read(DIRECTORY / "fits" / FIT / "CHECKPOINT_RECEIPT.json")
    if journal["durable_updates"] != 12000 or endpoint["status"] != "COMPLETE":
        raise ValueError("C1 endpoint must remain complete")
    ledger = read(RUN / "GPU_BUDGET.json")
    training = [
        entry for entry in ledger["entries"] if entry["scope"] == "geometry_scientific_training"
    ]
    result = {
        "status": selection["status"],
        "rows": rows,
        "bindings": bindings,
        "selection": selection,
        "endpoint_sha256": endpoint["checkpoint_sha256"],
        "scientific_updates": 12000,
        "recovery_upper": journal["recovery_upper"],
        "training_scope_charged_upper_s": sum(entry["charged_s"] for entry in training),
        "unsettled_scopes_count": sum(entry["status"] != "SETTLED" for entry in training),
        "own_prediction_rows": evidence["own_prediction_rows"],
        "bootstrap_H_vs_free": bootstrap,
        "full_objective_complete": False,
        "system_promotion": False,
        "official_comparison": False,
    }
    write(DIRECTORY / "RESULTS.json", result)
    lines = [
        "# V18 C1: medición relativa geométrica",
        "",
        f"Estado: **{selection['status']}**. Sin promoción de sistema ni afirmación SOTA.",
        "",
        "Endpoint fijo de 12.000 updates, seed7, encoder entrenable "
        "y readout de escala/traslación.",
        "PAIR-CORR es el control libre de la matriz primaria: se reutiliza sin reentrenarlo.",
        "",
        "| Población | Modelo | Agregado | MiDc | Small | Large | Negativo | "
        "MAE (s) | Error de signo | p95 AE (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | {row['model']} | {row['overall_MiD']:.6f} | "
            f"{row['crucial']:.6f} | {row['small']:.6f} | {row['large']:.6f} | "
            f"{row['negative']:.6f} | {row['mae_s']:.6f} | "
            f"{100 * row['sign_error_rate']:.4f}% | {row['p95_ae_s']:.6f} |"
        )
    lines += [
        "",
        "P es entrenamiento; H es desarrollo del productor, separado de P.",
        "Las métricas menores son mejores. La mejora del agregado "
        "no implica dominancia en cada medida.",
        "",
        "C1 mejora el agregado y todas las bandas MiD en H respecto al control libre.",
        "Su MAE y cola de error absoluto empeoran frente al control y A5. En P tampoco resuelve",
        "la banda negativa o el signo frente a A5: el fallo no es sólo transferencia P/H.",
        "Frente a A5, los gates de banda y signo siguen fallando. No se abre V para rescatarlo.",
        "",
        "Intervalos descriptivos pareados por grupo de adquisición (C1 menos PAIR-CORR en H):",
        "",
    ]
    for key, interval in bootstrap["intervals"].items():
        lines.append(
            f"- {key}: IC95 {interval['ci95_if_all_defined']}; "
            f"réplicas indefinidas: {interval['undefined_replicates']}."
        )
    lines += [
        "",
        "El intervalo crucial incluye cero; no se establece una ganancia crucial",
        "frente al control libre. La mejora del agregado aporta una señal",
        "mecanística para una fusión separada con anchor RGB.",
        "Usar C1 como complemento experimental no lo convierte en candidato standalone admisible.",
        "",
        f"Predicciones propias: {evidence['own_prediction_rows']:,}; "
        "referencias reutilizadas: cuatro tablas.",
        "Se conservan recálculo NumPy, bootstrap por grupos y verificación",
        "de 384.000 IDs del sampler, estados Adam/scheduler y cambio de los",
        "18 tensores del encoder; export TorchScript y replay.",
        f"Checkpoint: `{endpoint['checkpoint_sha256']}`.",
        "",
        f"Updates científicos: 12.000; recuperación no durable: {journal['recovery_upper']}.",
        f"Scopes de entrenamiento: {result['training_scope_charged_upper_s']:.3f} "
        "s contabilizados;",
        f"{result['unsettled_scopes_count']} reservas conservadoras no liquidadas incluidas.",
        "Estos scopes incluyen carga/guardado, no sólo kernels;",
        "no incluyen toda la QA y análisis C1.",
        "El coste de selección mide tensores preparados en GPU, no el pipeline completo.",
        "",
        "Réplica/refit no se activan por este resultado standalone. Continúa la prueba autorizada",
        "de complemento event a R_CTX con control ZERO. R_CTX fue supervisado en H: entrenar",
        "otra cabeza en H será ajuste secuencial, no OOF ni evaluación H independiente.",
        "",
        "Regeneración: `python -B -m operational.joint_measurement.geometry_report`.",
        "",
    ]
    output = ROOT / "docs/joint_measurement_20261012/C1_RESULTS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return {"status": result["status"], "tables": len(rows), "report": str(output)}


if __name__ == "__main__":
    print(build())
