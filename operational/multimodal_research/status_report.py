"""Regenerate a recoverable research report from actual sealed evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from operational.multimodal_validation.common import read, sha, write

from .common import OUT, accounting


def table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main() -> None:
    ledger = accounting()
    now = datetime.now(UTC).isoformat()
    sources = {}
    results: list[tuple[str, int | str, Path]] = [
        ("BASELINE", 7, OUT),
        ("PHASE_UNITS", 7, OUT / "studies/PHASE_UNITS"),
        ("UNIT_REPLICATION", 13, OUT / "studies/UNIT_REPLICATION/S13"),
        ("UNIT_REPLICATION", 23, OUT / "studies/UNIT_REPLICATION/S23"),
        ("RELATIVE_PHASE", 7, OUT / "studies/RELATIVE_PHASE/S7"),
        ("FUSION_STATE", 7, OUT / "studies/FUSION_STATE/S7"),
        ("RGB_REGRET", 7, OUT / "studies/RGB_REGRET/S7"),
        ("PHASE_ENSEMBLE", "7+13+23", OUT / "studies/PHASE_ENSEMBLE"),
        ("FUSION_PRESERVING", 7, OUT / "studies/FUSION_PRESERVING/S7"),
        ("PHASE_ONLY", 7, OUT / "studies/PHASE_ONLY/S7"),
        ("PHASE_ONLY_REPLICATION", 13, OUT / "studies/PHASE_ONLY_REPLICATION/S13"),
        ("PHASE_ONLY_REPLICATION", 23, OUT / "studies/PHASE_ONLY_REPLICATION/S23"),
        ("GROUP_CROSSFIT", 7, OUT / "studies/GROUP_CROSSFIT/S7"),
    ]
    rows, external = [], []
    for study, seed, directory in results:
        result_path = directory / "V_RESULTS.json"
        if not result_path.exists():
            rows.append([study, seed, "pendiente", "—", "—", "—", "—"])
            continue
        data = read(result_path)
        sources[str(result_path.relative_to(OUT))] = sha(result_path)
        arms = ("FUSED", "ZERO_FUSED") if study == "FUSION_STATE" else ("EVENT", "ZERO")
        contrast_key = "fused_minus_zero_fused" if study == "FUSION_STATE" else "event_minus_zero"
        prefix = "R_ANCHOR_"
        if study == "FUSION_PRESERVING":
            arms = ("REFINED_RGB", "ZERO_RGB")
            contrast_key = "refined_rgb_minus_zero_rgb"
            prefix = "F_ANCHOR_"
        for arm in arms:
            name = prefix + arm
            score = data["scores"][name]
            failed = [key for key, ok in data["gates"][name]["checks"].items() if not ok]
            rows.append(
                [
                    study,
                    seed,
                    arm,
                    f"{score['group_macro_bucket_MiD']:.6f}",
                    f"{score['diagnostics']['mae_s']:.6f}",
                    ", ".join(failed) or "ninguno",
                    str(data[contrast_key]["ci95"]) if arm == arms[0] else "—",
                ]
            )
        for route in ("cpu", "native_gpu"):
            for cohort in ("FCWD", "DEV32"):
                metric_path = directory / "external_evaluation_v2" / route / cohort / "METRICS.csv"
                if not metric_path.exists() and route == "cpu":
                    metric_path = directory / "external_evaluation" / cohort / "METRICS.csv"
                if not metric_path.exists():
                    continue
                sources[str(metric_path.relative_to(OUT))] = sha(metric_path)
                metrics = pd.read_csv(metric_path)
                for arm in arms:
                    full = metrics.query(
                        "scope == 'all_queries' and band == 'all' and method == @name",
                        local_dict={"name": prefix + arm},
                    ).iloc[0]
                    crucial = metrics.query(
                        "scope == 'all_queries' and band == 'crucial' and method == @name",
                        local_dict={"name": prefix + arm},
                    ).iloc[0]
                    external.append(
                        [
                            study,
                            seed,
                            arm,
                            cohort,
                            route,
                            int(full["labelled"]),
                            f"{full['strict_mean_MiD']:.6f}",
                            f"{full['MAE_finite_s']:.6f}",
                            f"{crucial['strict_mean_MiD']:.6f}",
                        ]
                    )
    text = [
        "# Continuación multimodal: estado recuperable",
        f"Generado {now}. Investigación en curso; SOTA no demostrado.",
        f"Updates científicos: {ledger['scientific']}; recuperación superior: "
        f"{ledger['recovery_upper']}; límite físico consumido: {ledger['physical_upper']}.",
        "Pasar del 2 % al 4 % cambia la tolerancia de degradación del tramo TTC largo; "
        "no cambia las predicciones ni demuestra superioridad sobre otro sistema. "
        "PHASE_ONLY ya cumple los criterios originales del 2 % en las semillas 7, 13 y 23, "
        "incluido el contraste con el control sin información predictiva event. "
        "Sus ventajas externas dependen de la métrica y la cohorte: F_TRUE conserva mejor "
        "agregado/MAE FCWD y Dev32 no confirma una mejora consistente de MiD. "
        "No se declara una victoria universal ni un resultado oficial de test12.",
        "## Evidencia interna V",
        "V es desarrollo expuesto. Cada par de correctores conserva 2.500 updates por brazo. "
        "Los estados H originales proceden de refinadores supervisados en H y no son OOF. "
        "GROUP_CROSSFIT prepara contextos y normalizadores excluyendo el grupo predicho; "
        "sus ocho fits de contexto tienen contabilidad propia. Las semillas adicionales "
        "de PHASE_ONLY replican las cabezas con productores congelados, no el sistema completo.",
        table(
            [
                "Estudio",
                "Semilla",
                "Brazo",
                "MiD agregado",
                "MAE s",
                "Gates fallidos",
                "IC brazo−control",
            ],
            rows,
        ),
        "El gate principal conserva 2 % de tolerancia por bucket. El 4 % en TTC largo es "
        "una sensibilidad solicitada después de ver la semilla 7. Su análisis secundario en "
        "semillas 13/23 se registró antes de sus nuevas predicciones V; no cambia el principal.",
        "PHASE_ENSEMBLE promedia fases emitidas de todas las semillas 7/13/23 con pesos iguales "
        "y cero updates nuevos. FUSION_STATE compara presencia/ausencia de estados F_TRUE sobre "
        "el ancla RGB. FUSION_PRESERVING conserva F_TRUE como ancla y compara presencia/ausencia "
        "de estados adicionales R_CTX. Esos dos últimos controles permanecen multimodales; "
        "sus nombres no significan ausencia de una modalidad en el sistema completo.",
        "PHASE_ONLY conserva el ancla R_CTX y usa un corrector de 1.955 parámetros basado "
        "en fases y relojes, sin estados ocultos. Los estudios sucesivos forman una búsqueda "
        "adaptativa en V; sus intervalos por contraste no corrigen la selección entre recetas.",
        "## Transferencia congelada",
        "Todas las consultas permanecen. Sin RGB, los nuevos brazos usan exactamente E_CTX "
        "de la misma ruta. FCWD tiene registro estimado, no calibración extrínseca oficial. "
        "Dev32 conserva 1.024 consultas/946 etiquetas. Los CSV también muestran soporte común, "
        "cobertura, todos los comparadores y contrastes descriptivos por secuencia. "
        "Dev32 y FCWD ya estaban expuestos durante el desarrollo: esta transferencia es "
        "descriptiva y no constituye un nuevo test ciego de la búsqueda completa.",
        table(
            [
                "Estudio",
                "Semilla",
                "Brazo",
                "Cohorte",
                "Ruta padres",
                "Etiquetas",
                "mean MiD",
                "MAE s",
                "MiDc",
            ],
            external,
        ),
        "La ruta histórica Dev32 era CPU; la de FCWD era GPU. La ruta FCWD conserva "
        "matmul TF32 desactivado y cuDNN TF32 activado. El replay global "
        "de padres tiene sus recibos separados en parent_replay. Los readouts externos "
        "se ejecutan en CPU FP32 y esa precisión se declara en cada sello.",
    ]
    path = OUT / "H_RECIPE_DIAGNOSTICS.json"
    if path.exists():
        sources[str(path.relative_to(OUT))] = sha(path)
        diagnostics = read(path)
        text += [
            "## Diagnóstico del ajuste sobre H",
            "Pérdidas in-sample, sin updates nuevos. No son OOF ni una medida de generalización. "
            "La mejora H de FUSED acompañada por su deterioro V es compatible con sobreajuste; "
            "no demuestra una causa única.",
            table(
                ["Estudio", "Brazo", "Error absoluto fase H", "Pérdida nativa H"],
                [
                    [
                        r["study"],
                        r["arm"],
                        f"{r['mean_abs_phase_error']:.8f}",
                        f"{r['native_fusion_loss']:.8f}",
                    ]
                    for r in diagnostics["rows"]
                ],
            ),
        ]
    path = OUT / "PHASE_ONLY_ATTRIBUTION.json"
    if path.exists():
        sources[str(path.relative_to(OUT))] = sha(path)
        attribution = read(path)
        examples = {
            "CCRs-1-medium-100-overlap-100:01",
            "CCRs-1-low-100-overlap-100:24",
        }
        text += [
            "## Atribución en los episodios ya publicados",
            f"{len(attribution['rows'])} trazas de las dos cabezas congeladas sobre los "
            "32 controles H seleccionados originalmente por hash y los 20 episodios "
            "externos ya expuestos. Instrumentación con paridad bit a bit; las salidas "
            "externas coinciden exactamente con las predicciones selladas. Se reutilizan "
            "estados de padres sellados; no es una nueva inferencia sensorial.",
            table(
                ["Consulta", "Brazo", "GT s", "R_CTX s", "F_TRUE s", "Nueva s"],
                [
                    [
                        r["query_id"],
                        r["arm"],
                        f"{r['truth_ttc']:.6f}",
                        f"{r['R_CTX']:.6f}",
                        f"{r['F_TRUE']:.6f}",
                        f"{r['emitted_ttc']:.6f}",
                    ]
                    for r in attribution["rows"]
                    if r["query_id"] in examples
                ],
            ),
            "La nueva cabeza evita el extremo de F_TRUE en estos dos ejemplos, pero "
            "no corrige el signo del caso medium:01 y empeora respecto a R_CTX en ambos. "
            "El control ZERO resulta mejor aquí. Son diagnósticos de casos seleccionados "
            "por fallos anteriores, no una estimación representativa de la contribución event.",
        ]
    path = OUT / "PHASE_ONLY_STRATA.csv"
    if path.exists():
        sources[str(path.relative_to(OUT))] = sha(path)
        strata = pd.read_csv(path).query("arm == 'EVENT' and comparator == 'R_CTX'")
        text += [
            "## Desacuerdo de signo entre los padres",
            "Diagnóstico posterior de todas las consultas externas del candidato S7. "
            "Los estratos usan signos de E_CTX/R_CTX, sin ajustar umbrales al error TTC. "
            "Delta negativo favorece al candidato; la comparación con RGB usa sólo "
            "etiquetas con RGB disponible. No es una nueva validación independiente.",
            table(
                ["Cohorte", "Estrato", "Etiquetas comunes", "Delta MiD", "Delta MAE s"],
                [
                    [
                        r["cohort"],
                        r["stratum"],
                        r["labelled_common"],
                        f"{r['delta_MiD']:.6f}",
                        f"{r['delta_MAE']:.6f}",
                    ]
                    for r in strata.to_dict("records")
                ],
            ),
            "En Dev32 el desacuerdo concentra un empeoramiento respecto a RGB, aunque "
            "la nueva cabeza reduce el error de F_TRUE en ese mismo estrato. "
            "Las 13 predicciones negativas de R_CTX entre las 946 etiquetas positivas "
            "siguen siendo negativas. Esto localiza una limitación, pero no demuestra "
            "causalidad ni autoriza una compuerta ajustada con estas etiquetas.",
        ]
    text += ["## Coste y paridad"]
    path = OUT / "EXACT_EVENT_ROUTE_SUMMARY.json"
    if path.exists():
        sources[str(path.relative_to(OUT))] = sha(path)
        cost_rows = [
            [
                r["cohort"],
                r["comparisons"],
                f"{r['native_mean_ms']:.3f}",
                f"{r['reuse_mean_ms']:.3f}",
                f"{100 * (1 - r['ratio_total_ms']):.2f} %",
            ]
            for r in read(path)
        ]
        text += [table(["Cohorte", "Comparaciones", "Nativo ms", "Reuso ms", "Ahorro"], cost_rows)]
        text += [
            "CPU, un hilo, carga concurrente; HDF5 hasta E_CTX con pesos residentes. Incluye "
            "reuso exacto de ventanas, mapa y frames del encoder. No incluye adquisición de ROI, "
            "carga inicial de pesos, RGB ni registro. Las 128 comparaciones preservan inputs, "
            "features, expertos, punto y cuantiles bit a bit. No es una cifra de streaming GPU.",
        ]
    path = OUT / "PHASE_ONLY_GARL_POSTHOC.csv"
    if path.exists():
        sources[str(path.relative_to(OUT))] = sha(path)
        comparisons = pd.read_csv(path).query("seed == 7")
        text += [
            "## PHASE_ONLY frente a Garl y F_TRUE",
            "Contrastes descriptivos posteriores a la congelación del candidato semilla 7. "
            "Delta negativo favorece PHASE_ONLY. Bootstrap de secuencias completas; "
            "FCWD sólo tiene tres secuencias. No corrige la búsqueda adaptativa ni autoriza "
            "escoger otra receta según estos resultados externos.",
            table(
                ["Cohorte", "Comparador", "Métrica", "Delta", "IC95 %"],
                [
                    [
                        r["cohort"],
                        r["comparator"],
                        r["metric"],
                        f"{r['delta']:.6f}",
                        f"[{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]",
                    ]
                    for r in comparisons.to_dict("records")
                ],
            ),
        ]
    path = OUT / "PHASE_ONLY_ROUTE_COST.json"
    if path.exists():
        sources[str(path.relative_to(OUT))] = sha(path)
        cost = read(path)
        text += [
            "## Coste sensorial del candidato",
            cost["scope"],
            table(
                ["Cohorte", "Etapa", "p50 ms", "p95 ms"],
                [
                    [cohort, stage, f"{v['p50']:.3f}", f"{v['p95']:.3f}"]
                    for cohort, stages in cost["summary"].items()
                    for stage, v in stages.items()
                ],
            ),
            f"{cost['rows']} ejecuciones con paridad exacta de estados y predicción. "
            "El coste de registro FCWD debe añadirse; esta medición no es la latencia "
            "completa de un sistema con detector y registro en línea.",
        ]
    path = OUT / "test12_phase_only/MODEL_INPUT_FREEZE.json"
    if path.exists():
        sources[str(path.relative_to(OUT))] = sha(path)
        frozen = read(path)
        text += [
            "## Preparación oficial test12",
            f"Candidato {frozen['candidate']} congelado en {frozen['utc']}. "
            "Se mantienen los 6.762 tokens oficiales. Las 957 repeticiones por clips "
            "solapados se resuelven en 5.805 observaciones físicas tras comprobar igualdad "
            "de ventanas, ROI y relojes. Cada observación aporta evidencia una sola vez "
            "al contexto; todos los tokens reciben su predicción. No se leen etiquetas privadas.",
            "Los productores/PAIR y refinadores siguen siendo los padres V13 originales. "
            "No se presenta esta inferencia como un refit nuevo de TRAIN40. Las fuentes RGB "
            "están verificadas contra los hashes de la revisión fijada de eAP. "
            "El candidato usa historias causales de observaciones del mismo objeto, con "
            "soporte total de hasta 650 ms; Garl full utiliza su pareja temporal nativa.",
        ]
        package_path = OUT / "test12_phase_only/PACKAGE_VERIFICATION.json"
        if package_path.exists():
            package = read(package_path)
            sources[str(package_path.relative_to(OUT))] = sha(package_path)
            text += [
                f"Estado: {package['status']}; {package['sample_count']} predicciones. "
                f"SHA-256 del ZIP: `{package['zip_sha256']}`. No enviado, sin score oficial."
            ]
        else:
            text += ["Inferencia en curso; el ZIP se declarará completo tras su verificación."]
    verification_path = OUT / "METRICS_REPLAY_VERIFICATION.json"
    if verification_path.exists():
        verification = read(verification_path)
        sources[str(verification_path.relative_to(OUT))] = sha(verification_path)
        counts = verification["counts"]
        text += [
            "## Reproducción de las tablas",
            f"{counts['scalar_checks']} comprobaciones escalares aprobadas sobre "
            f"{counts['internal_tables']} tablas V y {counts['external_tables']} tablas externas. "
            f"Se verifican los endpoints de los {counts['verified_fits']} fits. "
            "Es un recálculo de CSV y una verificación de hashes; no se presenta como "
            "una repetición de las inferencias desde los sensores.",
        ]
    text += [
        "## Qué falta para una afirmación de SOTA",
        "Hay que demostrar superioridad frente a métodos comparables bajo el mismo protocolo "
        "y declarar modalidades y métricas. Ganar una métrica oficial no equivale a ganar "
        "todos los regímenes. Las réplicas completas de productores/PAIR/cabezas y la "
        "transferencia son evidencia pendiente para afirmaciones amplias. El ranking "
        "oficial test12 requiere scoring oficial: sus etiquetas TTC no son públicas. "
        "Los ZIP de formato no son mediciones de precisión. No se han enviado submissions. "
        "[Protocolo del release oficial](https://github.com/NAIL-HNU/Garl-TTC).",
        "El registro geométrico conserva pendientes las correspondencias independientes; "
        "NCC no sustituye esa referencia. V15/V17 y los resultados históricos se preservan.",
    ]
    report_path = OUT / "CURRENT_RESEARCH_REVIEW.md"
    report_path.write_text("\n\n".join(text) + "\n", encoding="utf-8")
    if (OUT / "test12_phase_only/PACKAGE_VERIFICATION.json").exists():
        (OUT / "MULTIMODAL_REVIEW_FINAL.md").write_bytes(report_path.read_bytes())
    write(
        OUT / "CURRENT_RESEARCH_REVIEW_SOURCES.json",
        dict(
            utc=now,
            sources=sources,
            report_sha256=sha(report_path),
            source_sha256=sha(__file__),
            status="IN_PROGRESS",
            official_SOTA=False,
        ),
    )
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
