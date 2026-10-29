"""Regenerate the primary milestone conclusion without closing the V18 program."""

from __future__ import annotations

from operational.range_transfer.common import read, sha, utc, write

from .common import RUN


def build() -> dict:
    """Bind the completed system comparison to its unchanged prospective selection."""
    comparison = read(RUN / "system_compare/RESULT.json")
    selection = read(RUN / "PRIMARY_SELECTION_FINAL.json")
    decision = read(RUN / "NEXT_DECISION.json")
    cost = read(RUN / "cost/primary_raw_e2e/COMPLETE.json")
    budget = read(RUN / "GPU_BUDGET.json")
    counts = read(RUN / "ACCOUNTING.json")
    if (
        comparison["status"] != "INTERNAL_SYSTEM_COMPARISON_COMPLETE"
        or comparison["selection_sha256"] != sha(RUN / "PRIMARY_SELECTION_FINAL.json")
        or comparison["cost_sha256"] != sha(RUN / "cost/primary_raw_e2e/COMPLETE.json")
        or comparison["contract_sha256"] != sha(RUN / "system_compare/REGISTRATION.json")
        or set(comparison["candidates"]) != set(selection["ranked"])
        or counts["pending_upper"] != 0
    ):
        raise ValueError("Primary comparison is incomplete, changed or not durable")
    if any(row["gates"]["eligible"] for row in comparison["candidates"].values()):
        raise ValueError("This negative milestone requires a fresh interpretation if gates pass")
    bindings = {
        name: sha(RUN / name)
        for name in (
            "system_compare/RESULT.json",
            "system_compare/REGISTRATION.json",
            "PRIMARY_SELECTION_FINAL.json",
            "PRIMARY_ENDPOINTS.json",
            "primary_evidence/COMPLETE.json",
            "NEXT_DECISION.json",
            "rgb/REGISTRATION.json",
            "rgb/EXECUTION_HOLD.json",
            "BUDGET_AUTHORIZATION_20261013.json",
            "cost/primary_raw_e2e/COMPLETE.json",
            "cost/primary_raw_e2e/RAW_PARITY.json",
        )
    }
    rows = []
    metrics = {comparison["reference"]: comparison["reference_metrics"]}
    for name, result in comparison["candidates"].items():
        directory = RUN / "predictions/V" / name
        verified = read(RUN / "verification/predictions/V" / name / "NUMPY_RECALC.json")
        if (
            sha(directory / "predictions.parquet") != result["predictions_sha256"]
            or verified["status"] != "PASS"
            or sha(directory / "METRICS.json") != verified["metrics_sha256"]
            or read(directory / "METRICS.json") != result["metrics"]
        ):
            raise ValueError("Candidate metrics differ from the independently checked table")
        metrics[name] = result["metrics"]
    for name, metric in metrics.items():
        rows.append(
            {
                "model": name,
                "rows": metric["num_rows"],
                "overall_MiD": metric["overall_MiD"],
                "MiDc": metric["bins"]["crucial"]["group_macro_mid"],
                "MAE_s": metric["diagnostics"]["mae_s"],
                "sign_error_percent": 100 * metric["diagnostics"]["sign_error_rate"],
                "raw_E2E_p50_ms": 1000 * cost["results"][name]["p50_seconds"],
            }
        )
    conclusion = {
        "schema": "v18_primary_milestone_conclusion_v1",
        "utc": utc(),
        "research_status": "RESEARCH_ONGOING",
        "primary_status": "NO_SYSTEM_PROMOTION",
        "full_objective_complete": False,
        "system_promotion": False,
        "SOTA_confirmation": False,
        "scope": "Completed seed7 primary matrix; V is exposed internal development",
        "H_development_candidates_preserved": selection["ranked"],
        "V_rows": rows,
        "failed_system_gates": {
            name: [key for key, passed in result["gates"]["checks"].items() if not passed]
            for name, result in comparison["candidates"].items()
        },
        "evidence": bindings,
        "accounting_snapshot": counts,
        "budget_snapshot": {
            "cap_s": budget["cap_s"],
            "charged_total_s": budget["charged_total_s"],
            "remaining_s": budget["cap_s"] - budget["charged_total_s"],
            "scope": "Four existing primary arms and their evaluation only",
            "RGB_execution": "AWAIT_BUDGET_SCOPE_AUTHORIZATION",
        },
        "remaining_work": decision["remaining_program"],
        "next_registered_experiment": decision["next_registered_experiment"],
        "resume_after_scope_approval": decision["resume_after_scope_approval"],
        "prospective_decision_preserved": "NEXT_DECISION.json predates V interpretation",
        "limitations": [
            "One seed; group bootstrap does not estimate variability across training seeds",
            "E_CTX retains native history and experts; P2 uses less temporal context",
            "Raw E2E covers 64 paired queries; resident weights and uncontrolled OS page cache",
            "This milestone is not completion of the adaptive/RGB/replication/refit program",
            "Portable replay is not full retraining or official benchmark evaluation",
        ],
    }
    lines = [
        "# Resultado de la fase primaria V18",
        "",
        "**NO_SYSTEM_PROMOTION. La investigación V18 sigue abierta.**",
        "",
        "Cuatro fits de 12.000 updates, seed7, terminados. Los cinco modelos, incluido",
        "A5-P2 congelado, tienen cohortes P/H completas. NATIVE fue el único candidato",
        "admitido en H; su selección queda conservada y se fijó antes de evaluar V.",
        "Las variantes PAIR incumplen las protecciones de signo y bandas, también con",
        "debilidades en P. El resultado no demuestra que la correlación sea inútil.",
        "",
        "## Comparación del sistema en V",
        "",
        "V es desarrollo interno expuesto. Menor MiD y MAE es mejor.",
        "",
        "| Modelo | Filas | Agregado MiD | MiDc | MAE s | Error signo % | E2E p50 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {r['model']} | {r['rows']} | {r['overall_MiD']:.3f} | {r['MiDc']:.3f} | "
            f"{r['MAE_s']:.3f} | {r['sign_error_percent']:.3f} | {r['raw_E2E_p50_ms']:.3f} |"
            for r in rows
        ],
        "",
        "El ahorro de latencia no compensa el incumplimiento de los gates de precisión.",
        "NATIVE conserva su señal de mejora en H frente a A5-P2; eso no implica superar",
        "el sistema completo E_CTX. No se promocionan otros brazos tras mirar V.",
        "",
        "El coste E2E incluye HDF5, ROI común, transferencia, modelo y TTC. Se midieron",
        "64 consultas emparejadas, ocho por grupo, con pesos residentes. La caché del",
        "sistema operativo no se controla. E_CTX mantiene su historial nativo y expertos;",
        "los modelos P2 usan menos contexto. No es una comparación arquitectónica aislada.",
        "",
        "## Estado y continuación",
        "",
        f"Updates físicos: {counts['physical_upper']:,}; pendientes: {counts['pending_upper']}.",
        f"Scopes GPU propios: {budget['charged_total_s'] / 3600:.3f} h de "
        f"{budget['cap_s'] / 3600:.0f} h. Incluyen carga y guardado dentro del scope.",
        "",
        "La ampliación a 48 horas cubre la matriz primaria y su evaluación. RGB está",
        "registrado, con cero updates, pendiente de ampliar el alcance de ese mismo techo.",
        "La decisión RGB se conserva tal como se registró antes de conocer el resultado V.",
        "Un resultado negativo primario no cierra las demás hipótesis del programa.",
        "",
        "[Resultados P/H](../primary_analysis/README.md), "
        "[comparación y bootstrap](../system_compare/RESULT.json), "
        "[conclusión y trabajo restante](../CAMPAIGN_CONCLUSION.json), "
        "[decisión prospectiva](../NEXT_DECISION.json).",
        "",
        "## Límites de la entrega",
        "",
        *[f"- {item}" for item in conclusion["limitations"]],
        "",
    ]
    write(RUN / "CAMPAIGN_CONCLUSION.json", conclusion)
    write(RUN / "NEXT_DECISION_POST_PRIMARY.json", conclusion)
    directory = RUN / "primary_milestone"
    directory.mkdir(exist_ok=True)
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "primary_status": conclusion["primary_status"],
        "full_objective_complete": False,
        "report": str(directory / "README.md"),
    }


if __name__ == "__main__":
    print(build())
