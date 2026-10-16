"""Refresh the campaign conclusion while retaining the closed primary milestone."""

from __future__ import annotations

from operational.range_transfer.common import freeze, read, sha, utc, write

from .common import ROOT, RUN


def build() -> dict:
    """Describe completed evidence and outstanding work; never infer campaign success."""
    path = RUN / "CAMPAIGN_CONCLUSION.json"
    primary = RUN / "verification/PRIMARY_MILESTONE_CONCLUSION_20261016.json"
    if not primary.exists():
        old = read(path)
        if old["schema"] != "v18_primary_milestone_conclusion_v1":
            raise ValueError("Preserve the original primary conclusion before refreshing it")
        freeze(primary, old)
    original = read(primary)
    decision = read(RUN / "NEXT_DECISION.json")
    accounting = read(RUN / "ACCOUNTING.json")
    budget = read(RUN / "GPU_BUDGET.json")
    evidence = {"verification/PRIMARY_MILESTONE_CONCLUSION_20261016.json": sha(primary)}
    runtime_evidence = {}
    for label, relative in {
        "A1_optimized_training": "runtime/A1_parallel_persistent_v1/ADMISSION.json",
        "A1_scope_continuity": "runtime/A1_parallel_persistent_v1/PRODUCTION_SCOPE_CONTINUITY.json",
        "A1_evaluation_registered": "adaptive/A1_adaptation_depth/evaluation/REGISTRATION.json",
        "F_TRUE_head_CPU": "fusion/cost/F_TRUE_head_diagnostic_v1/CPU_RESULT.json",
        "fusion_cost_amended_acceptance": (
            "fusion/cost/raw_e2e_millisecond_v1/SAVED_OUTPUT_AUDIT.json"
        ),
    }.items():
        source = RUN / relative
        if source.exists():
            runtime_evidence[label] = {"path": relative, "sha256": sha(source)}
            evidence[relative] = sha(source)
    stages = {}
    for label, relative in {
        "primary": "primary_evidence/COMPLETE.json",
        "RGB": "rgb/evidence/COMPLETE.json",
        "C1": "adaptive/C1_geometry/evidence/COMPLETE.json",
        "fusion_seed7": "fusion/evaluation/RESULT.json",
        "fusion_three_seeds": "replication/fusion_C1/evaluation/RESULT.json",
        "D1": "adaptive/D1_predictor_width/evidence/COMPLETE.json",
        "B1": "adaptive/B1_correspondence/evidence/COMPLETE.json",
        "A1": "adaptive/A1_adaptation_depth/evidence/COMPLETE.json",
    }.items():
        source = RUN / relative
        if source.exists():
            value = read(source)
            stages[label] = {
                "evidence_complete": True,
                "status": value["status"],
                "path": relative,
                "sha256": sha(source),
            }
            if "selection" in value:
                stages[label]["selection"] = value["selection"]
            evidence[relative] = sha(source)
        else:
            stages[label] = {"evidence_complete": False, "status": "PENDING"}
    replica = read(RUN / "replication/fusion_C1/evaluation/RESULT.json")
    remaining = []
    for label in ("D1", "B1"):
        if not stages[label]["evidence_complete"]:
            remaining.append(
                f"Completar {label}: endpoint, P/H, coste, export e interpretación registrados"
            )
    remaining += [
        (
            "Completar las 64 consultas de coste de fusión con aceptación autorizada de 1 ms; "
            "conservar en paralelo los fallos del criterio estricto"
            if "fusion_cost_amended_acceptance" in runtime_evidence
            else "Resolver/documentar la paridad antes de admitir el coste sensorial de fusión"
        ),
        "Congelar cualquier export test admisible; el envío oficial sigue sin autorización",
        "Publicar informes y verificar independientemente el ZIP final <=100000000 bytes",
    ]
    if (RUN / "adaptive/A1_adaptation_depth/REGISTRATION.json").exists():
        if not stages["A1"]["evidence_complete"]:
            remaining.append("Completar A1 registrada: dos endpoints, P/H, coste y exports")
        remaining.append("Resolver refit/export condicionado por A1 y los gates originales")
    else:
        remaining.append("Decidir con P/H si procede familia A, datos o refit")
    garl_cost = RUN / "cost/garl_routes/RESULT.json"
    if not garl_cost.exists() or read(garl_cost)["status"] != "COMPLETE":
        remaining.append(
            "Medir Garl completo: rutas nativa/local emparejadas, CPU FP32 y coste raw"
        )
    value = {
        "schema": "v18_campaign_progress_conclusion_v2",
        "utc": utc(),
        "research_status": "RESEARCH_ONGOING",
        "full_objective_complete": False,
        "primary_status": original["primary_status"],
        "system_promotion": False,
        "SOTA_confirmation": False,
        "stages": stages,
        "primary_V_rows": original["V_rows"],
        "fusion_three_seed_means": replica["means"],
        "fusion_three_seed_gates": replica["gates"],
        "accounting_snapshot": accounting,
        "accounting_snapshot_scope": (
            "Non-atomic live snapshot; journals and GPU_BUDGET remain authoritative"
        ),
        "budget_snapshot": {
            "cap_s": budget["cap_s"],
            "charged_or_reserved_s": sum(row["charged_s"] for row in budget["entries"]),
            "scope": "Own wall-clock GPU scopes including loading/saving, not kernel time",
            "authorization_sha256": sha(RUN / "BUDGET_AUTHORIZATION_20261014.json"),
        },
        "active_driver_snapshot": decision.get("active_driver"),
        "adaptive_D1_snapshot": decision.get("adaptive_D1"),
        "adaptive_B1_snapshot": decision.get("adaptive_B1"),
        "full_fusion_cost": decision.get("fusion_cost"),
        "A1_runtime_optimization": decision.get("runtime_optimization"),
        "A1_evaluation_prepared": decision.get("A1_evaluation_prepared"),
        "F_TRUE_head_CPU_diagnostic": decision.get("F_TRUE_head_CPU_diagnostic"),
        "runtime_evidence": runtime_evidence,
        "evidence": evidence,
        "remaining_work": remaining,
        "limitations": [
            "V is exposed internal development, not an untouched external benchmark",
            "RGB anchor was H-supervised; sequential fusion training is not OOF",
            "Three adaptation seeds share historical A5/R_CTX initialization",
            "No full-system promotion or interchangeable comparison to official Garl test12",
            "Portable exports do not alone reproduce raw-sensor pipelines or training",
        ],
    }
    if path.exists():
        archive = RUN / "conclusion_history" / f"{sha(path)}.json"
        if not archive.exists():
            freeze(archive, read(path))
    write(path, value)
    lines = [
        "# V18: conclusión actual y trabajo pendiente",
        "",
        "**La campaña sigue abierta; no hay promoción del sistema ni confirmación SOTA.** "
        "La fusión presenta una señal controlada en las tres semillas, pero no cumple "
        "el gate crucial original del10%. Los negativos y las regresiones se conservan.",
        "",
        f"Corte UTC: `{value['utc']}`. Fuente: `CAMPAIGN_CONCLUSION.json`.",
        "",
        "| Etapa | Evidencia completa | Estado |",
        "|---|---|---|",
    ]
    for name, stage in stages.items():
        lines.append(f"| {name} | {stage['evidence_complete']} | {stage['status']} |")
    lines += [
        "",
        "| Fusión, media de3 seeds | Agregado V | MiDc V | MAE V, s |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in replica["means"].items():
        lines.append(
            f"| {name} | {metric['overall_MiD']:.3f} | "
            f"{metric['bins']['crucial']['group_macro_mid']:.3f} | "
            f"{metric['diagnostics']['mae_s']:.3f} |"
        )
    lines += [
        "",
        "El agregado mejora frente al control ZERO de igual capacidad. "
        "Esto no demuestra superioridad frente a V16: el intervalo de la diferencia "
        "agregada entre sus medias incluye cero. Véase "
        "[el informe de réplica](FUSION_REPLICATION_RESULTS.md).",
        "",
        "## Contabilidad del corte",
        "",
        f"- Updates científicos: {accounting['scientific']:,}.",
        f"- Técnicos: {accounting['technical']:,}; recuperación superior: "
        f"{accounting['recovery_upper']:,}; "
        f"pendientes superiores: {accounting['pending_upper']:,}.",
        f"- Físicos superiores: {accounting['physical_upper']:,}.",
        f"- Scopes GPU contabilizados/reservados: "
        f"{value['budget_snapshot']['charged_or_reserved_s']:.1f} / {budget['cap_s']:.0f} s.",
        "",
        "Es una instantánea no atómica mientras hay trabajo vivo. No son kernels "
        "GPU puros; las reservas activas pueden cambiar al liquidarse. No se recontaron "
        "los controles históricos reutilizados.",
        "",
        "## Pendiente",
        "",
    ]
    lines += [f"- {item}" for item in remaining]
    if runtime_evidence:
        lines += [
            "",
            "## Ejecución y comprobaciones recientes",
            "",
            "A1 mantiene su receta científica; el lector paralelo, la caché y los grafos "
            "persistentes tienen admisión y prueba de continuidad. "
            "[Detalle del runtime](A1_PERSISTENT_RUNTIME_20261016.md).",
            "",
            "La evaluación A1 está registrada antes de los endpoints: "
            "[contrato P/H](A1_EVALUATION_CONTRACT_20261016.md). "
            "Su implementación y pruebas no equivalen a resultados del modelo.",
            "",
            "La cabeza F_TRUE reproduce las 13 consultas diagnósticas en CPU con el "
            "batch histórico; el coste raw completo sigue pendiente de paridad. "
            "[Contrastes y límites](FUSION_COST_DIAGNOSTIC.md).",
        ]
    lines += [
        "",
        "Las ramas condicionadas se justifican con evidencia; un resultado negativo "
        "cierra su experimento. La evaluación oficial requiere autorización separada "
        "y no se utiliza como bucle de ajuste.",
        "",
    ]
    report = ROOT / "docs/joint_measurement_20261012/CURRENT_CONCLUSION.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return {"conclusion_sha256": sha(path), "report_sha256": sha(report), "stages": stages}


if __name__ == "__main__":
    result = build()
    print({key: value for key, value in result.items() if key != "stages"})
