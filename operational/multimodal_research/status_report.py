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
        "## Evidencia interna V",
        "V es desarrollo expuesto. Cada par entrenado conserva 2.500 updates por brazo y estados H "
        "de refinadores previamente supervisados en H: no son OOF. Las semillas adicionales "
        "replican las cabezas, no el sistema completo.",
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
        "cobertura, todos los comparadores y contrastes descriptivos por secuencia.",
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
        "La ruta histórica Dev32 era CPU; la de FCWD era GPU. En diez casos FCWD, E_CTX "
        "se reprodujo con matmul TF32 desactivado y cuDNN TF32 activado. El replay global "
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
        "## Qué falta para una afirmación de SOTA",
        "Superar métodos comparables con las mismas modalidades, cohortes y métricas, "
        "replicar productores/PAIR/cabezas completos y verificar transferencia. El ranking "
        "oficial test12 requiere scoring oficial: sus etiquetas TTC no son públicas. "
        "Los ZIP de formato no son mediciones de precisión. No se han enviado submissions. "
        "[Protocolo del release oficial](https://github.com/NAIL-HNU/Garl-TTC).",
        "El registro geométrico conserva pendientes las correspondencias independientes; "
        "NCC no sustituye esa referencia. V15/V17 y los resultados históricos se preservan.",
    ]
    report_path = OUT / "CURRENT_RESEARCH_REVIEW.md"
    report_path.write_text("\n\n".join(text) + "\n", encoding="utf-8")
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
