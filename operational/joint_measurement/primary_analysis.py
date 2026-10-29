"""Regenerate the primary P/H interpretation from independently checked evidence."""

from __future__ import annotations

from operational.range_transfer.common import read, sha, utc, write

from .common import RUN, config


def build() -> dict:
    """Describe the fixed primary matrix without using V or changing selection."""
    evidence_path = RUN / "primary_evidence/COMPLETE.json"
    evidence = read(evidence_path)
    if evidence["status"] != "PRIMARY_EVIDENCE_COMPLETE_RESEARCH_ONGOING":
        raise ValueError("Finish complete primary evidence before interpreting it")
    reference = "A5-P2-FROZEN"
    names = [reference, *config()["fits"]]
    selection = read(RUN / "PRIMARY_SELECTION_FINAL.json")
    cost = read(RUN / "cost/primary_prepared_forward/COMPLETE.json")
    if cost["status"] != "COMPLETE":
        raise ValueError("Complete matched cost required")
    rows = []
    bindings = {}
    for role in ("P", "H"):
        baseline = read(RUN / "predictions" / role / reference / "METRICS.json")
        for name in names:
            directory = RUN / "predictions" / role / name
            verification = evidence["verification"][f"{role}/{name}"]
            metric_hash = sha(directory / "METRICS.json")
            if (
                metric_hash != verification["metrics_sha256"]
                or sha(directory / "predictions.parquet")
                != evidence["binding"]["cohorts"][f"{role}/{name}"]
            ):
                raise ValueError(
                    "Primary metrics or predictions differ from independent verification"
                )
            metric = read(directory / "METRICS.json")
            bindings[f"{role}/{name}"] = metric_hash
            rows.append(
                {
                    "role": role,
                    "fit": name,
                    "rows": metric["num_rows"],
                    "overall": metric["overall_MiD"],
                    "crucial": metric["bins"]["crucial"]["group_macro_mid"],
                    "negative": metric["bins"]["negative"]["group_macro_mid"],
                    "large": metric["bins"]["large"]["group_macro_mid"],
                    "sign_error_percent": 100 * metric["diagnostics"]["sign_error_rate"],
                    "MAE_seconds": metric["diagnostics"]["mae_s"],
                    "aggregate_improvement_percent": 100
                    * (1 - metric["overall_MiD"] / baseline["overall_MiD"]),
                    "crucial_improvement_percent": 100
                    * (
                        1
                        - metric["bins"]["crucial"]["group_macro_mid"]
                        / baseline["bins"]["crucial"]["group_macro_mid"]
                    ),
                    "H_eligible": selection["decisions"].get(name, {}).get("eligible"),
                    "prepared_forward_p50_ms": 1000 * cost["results"][name]["p50_seconds"],
                }
            )
    result = {
        "utc": utc(),
        "status": "PRIMARY_INTERPRETATION_RESEARCH_ONGOING",
        "primary_evidence_sha256": sha(evidence_path),
        "metric_bindings": bindings,
        "selection_sha256": sha(RUN / "PRIMARY_SELECTION_FINAL.json"),
        "prepared_cost_sha256": sha(RUN / "cost/primary_prepared_forward/COMPLETE.json"),
        "ranked": selection["ranked"],
        "rows": rows,
        "V_used_for_this_interpretation": False,
        "system_promotion": False,
        "official_confirmation": False,
    }
    lines = [
        "# Interpretación de la matriz primaria V18",
        "",
        f"Generado: {result['utc']}. Estado: RESEARCH_ONGOING.",
        "",
        "Cuatro endpoints de 12.000 updates, seed7. P es entrenamiento; H selecciona.",
        "Todas las cifras proceden de las diez tablas verificadas. Menor MiD es mejor.",
        "La media por grupos y bandas es un agregado interno, no un score oficial.",
        "",
    ]
    for role in ("P", "H"):
        lines += [
            f"## {role}",
            "",
            "| Modelo | Agregado | MiDc | MiD negativo | MiD grande | Signo % | MAE s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {r['fit']} | {r['overall']:.3f} | {r['crucial']:.3f} | "
            f"{r['negative']:.3f} | {r['large']:.3f} | {r['sign_error_percent']:.3f} | "
            f"{r['MAE_seconds']:.3f} |"
            for r in rows
            if r["role"] == role
        ]
        lines.append("")
    native = next(r for r in rows if r["role"] == "H" and r["fit"] == "NATIVE-P2-FT-S7")
    lines += [
        "## Lectura y límites",
        "",
        f"NATIVE mejora el agregado H un {native['aggregate_improvement_percent']:.3f}% y "
        f"MiDc un {native['crucial_improvement_percent']:.3f}% frente a A5-P2 congelado. "
        "Cumple los límites registrados de cobertura, signo y otras bandas. "
        "Es candidato de desarrollo; no una sustitución demostrada de E_CTX ni SOTA.",
        "",
        "Las variantes PAIR mejoran especialmente MiDc, pero fallan los límites de signo "
        "y otras bandas. La debilidad negativa aparece también en P: no puede atribuirse "
        "sólo a transferencia P→H. La pequeña diferencia CORR/DIRECT no establece superioridad "
        "de correspondencias sin más contraste. Ninguna variante se rescata por velocidad.",
        "",
        "NATIVE conserva el objetivo nativo; los tres PAIR comparten otro objetivo. "
        "Por ello NATIVE/PAIR no aísla exclusivamente arquitectura. Todos usan P2; "
        "la comparación pendiente contra E_CTX conserva su historial nativo más largo.",
        "",
        "El bootstrap conservado en ../bootstrap usa grupos de adquisición, no ventanas "
        "independientes. Sus intervalos son descriptivos; no sustituyen las réplicas de semillas.",
        "",
        "## Coste preparado",
        "",
        "128 consultas P emparejadas; batch1, FP32, orden intercalado, 20 warmups por modelo. "
        "Incluye tensor residente en GPU hasta TTC; excluye sensores, transferencia y carga. "
        "No es coste E2E ni rendimiento del trainer con CUDA Graphs.",
        "",
        "| Modelo | Mediana forward preparado, ms |",
        "|---|---:|",
        *[f"| {r['fit']} | {r['prepared_forward_p50_ms']:.3f} |" for r in rows if r["role"] == "H"],
        "",
        "## Siguiente decisión",
        "",
        "[NEXT_DECISION](../NEXT_DECISION.json) fija la interpretación P/H y el estado de "
        "presupuesto. La receta RGB-CORR entrenable/congelada se ha registrado antes de "
        "updates. Su registro no amplía el presupuesto. Las otras familias adaptativas "
        "requieren sus observaciones habilitantes; no se improvisa un sweep tras mirar V.",
        "",
        "[Datos y hashes de este informe](RESULTS.json). Regenerar con "
        "`python -B -m operational.joint_measurement.primary_analysis`.",
        "",
    ]
    directory = RUN / "primary_analysis"
    write(directory / "RESULTS.json", result)
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": result["status"], "rows": len(rows), "ranked": result["ranked"]}


if __name__ == "__main__":
    print(build())
