"""Generate detailed V14 reports from frozen results and a verified delivery ZIP.

This command never trains, selects a candidate or modifies the scientific runner.
"""
# ruff: noqa: E501 -- Markdown prose paragraphs retain their natural line boundaries.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTROL = "RT-M0-L0-S7"
BASE = "da068ebdeb2d967ee10b2afa348e1317b8720ad0"


def sha(path: Path) -> str:
    """Hash a file without loading checkpoints into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read an existing evidence document."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path: Path) -> list[dict[str, str]]:
    """Read a saved CSV without recomputing predictions."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: object) -> None:
    """Write finite publication metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def fmt(value: float | str | None, places: int = 3) -> str:
    """Format display values while retaining full precision in source files."""
    return "ausente" if value is None or value == "" else f"{float(value):.{places}f}"


def table(headers: list[str], values: list[list[Any]]) -> str:
    """Render a Markdown table with explicit blank-line separation."""
    body = [headers, ["---"] * len(headers), *values]
    return (
        "\n"
        + "\n".join(
            "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in body
        )
        + "\n\n"
    )


def generate(run: Path, output: Path) -> None:
    """Render the closed campaign, preserving the frozen selection and all failures."""
    output.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, str] = {}

    def evidence(relative: str) -> Path:
        source = run / relative
        target = output / "evidence" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs[relative] = sha(source)
        return source

    fixed = [
        "NEXT_DECISION.json",
        "SELECTION_FREEZE.json",
        "SCREEN_ENDPOINTS.json",
        "ACCOUNTING.json",
        "ADMISSION.json",
        "SOURCE_BINDINGS.json",
        "ROLE_AUDIT.json",
        "H_METADATA_AUDIT.json",
        "H_FINAL_CACHE_AUDIT.json",
        "MASS_AUDIT.csv",
        "GRADIENT_AUDIT.csv",
        "GRADIENT_AUDIT.json",
        "SCIENTIFIC_DELIVERY_AUDIT.json",
        "FACTORIAL_RESULTS.csv",
        "COST_SCOPE.json",
        "N1_STATUS.json",
        "DIAGNOSTIC_DEPENDENCIES.json",
        "COMMANDS_EXECUTED.md",
        "qa/RESUME_TEST.json",
        "qa/TESTS_PREFIT_FINAL_20261011.txt",
        "qa/PYRIGHT_PREFIT_FINAL_20261011.txt",
        "qa/TEST_MATCHED_PAIR_ADAPTER.txt",
        "qa/PUBLICATION_CHECKS.json",
        "public_analysis/ANALYSIS.json",
        "v12_raw_attribution/RESULT.json",
        "v12_cached_attribution/RESULT.json",
        "matched_H_controls/RESULT.json",
        "matched_H_raw/FAILURE.json",
        "matched_H_raw_producers/ATTRIBUTION_BY_RANGE.csv",
        "matched_H_raw_adapted/RESULT.json",
        "matched_H_raw_adapted/BINDING.json",
        "matched_H_raw_adapted/EXPERT_PHASE_TRACES.csv",
    ]
    for relative in fixed:
        evidence(relative)
    selection = read(run / "SELECTION_FREEZE.json")
    assert selection["status"] == "NO_CANDIDATE" and selection["candidate"] is None
    ledger = read(run / "ACCOUNTING.json")
    assert (
        ledger["scientific"],
        ledger["technical"],
        ledger["recovery_upper"],
        ledger["pending_upper"],
    ) == (15000, 20, 0, 0)
    endpoints = read(run / "SCREEN_ENDPOINTS.json")
    assert len(endpoints) == 6
    source_bindings = read(run / "SOURCE_BINDINGS.json")["runner_sources"]
    for relative, expected in source_bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Executed scientific source changed: {relative}")
    metrics, details, groups, provenance, bootstrap = {}, {}, {}, {}, {}
    for name, endpoint in sorted(endpoints.items()):
        checkpoint = run / "fits" / name / "checkpoint_last.pt"
        assert sha(checkpoint) == endpoint["checkpoint_sha256"]
        assert endpoint["completed_updates"] == 2500
        for file in (
            "CHECKPOINT_RECEIPT.json",
            "RUN_PROVENANCE.json",
            "UPDATE_JOURNAL.json",
            "CURVE.csv",
        ):
            evidence(f"fits/{name}/{file}")
        for file in (
            "METRICS.json",
            "GROUP_METRICS.json",
            "SUBRANGE_RESULTS.csv",
            "SUBRANGE_GROUP_RESULTS.csv",
            "COST.json",
            "BINDING.json",
            "PREDICTIONS_RECEIPT.json",
        ):
            evidence(f"evaluation/screen/{name}/{file}")
        prediction = run / "evaluation/screen" / name / "PREDICTIONS.csv"
        assert sha(prediction) == read(prediction.with_name("PREDICTIONS_RECEIPT.json"))["sha256"]
        inputs[str(prediction.relative_to(run)).replace("\\", "/")] = sha(prediction)
        metrics[name] = read(run / "evaluation/screen" / name / "METRICS.json")
        details[name] = rows(run / "evaluation/screen" / name / "SUBRANGE_RESULTS.csv")
        groups[name] = read(run / "evaluation/screen" / name / "GROUP_METRICS.json")
        provenance[name] = read(run / "fits" / name / "RUN_PROVENANCE.json")
        if name != CONTROL:
            bootstrap[name] = read(evidence(f"bootstrap/screen/{name}/SUMMARY.json"))
    assert len({v["draw_sequence_sha256"] for v in provenance.values()}) == 1
    assert all(
        m["num_rows"] == 16432 and m["complete_prediction_and_mid_coverage"]
        for m in metrics.values()
    )
    for source in (run / "figures").glob("*.png"):
        target = output / "figures" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        inputs[str(source.relative_to(run)).replace("\\", "/")] = sha(source)

    def md(name: str, content: str) -> None:
        (output / name).write_text(content.strip() + "\n", encoding="utf-8")

    names = sorted(metrics)
    reference = metrics[CONTROL]
    main_table = table(
        ["Receta seed7", "MiDc", "MiDs", "MiDl", "MiDn", "Overall MiD", "mean_MiD", "MAE s"],
        [
            [
                name,
                *[
                    fmt(metrics[name]["bins"][b]["group_macro_mid"])
                    for b in ("crucial", "small", "large", "negative")
                ],
                fmt(metrics[name]["overall_MiD"]),
                fmt(metrics[name]["mean_MiD"]),
                fmt(metrics[name]["diagnostics"]["mae_s"]),
            ]
            for name in names
        ],
    )
    md(
        "README.md",
        """# V14 RANGE-TRANSFER: entrega detallada

Fecha de esta entrega: 12 de octubre de 2026, Europe/Madrid. Los recibos usan UTC; el cierre alrededor de las 22:08 UTC del día 11 corresponde al día 12 en Madrid.

**Resultado: NO_CANDIDATE.** Ejecuté seis cabezas event-only de 2.500 updates, congelé sus endpoints y evalué las mismas 16.432 consultas V. Las variantes que redistribuyen masa reducen el error crucial, pero incumplen las salvaguardas de otros rangos. Ninguna receta habilita las réplicas seeds13/23. Conservo los seis pesos y sus resultados.

## Informes

1. [Trabajo realizado, identidades y método](01_EJECUCION_Y_METODO.md).
2. [Diagnósticos: masa, gradientes y origen del sesgo](02_DIAGNOSTICOS.md).
3. [Resultados completos, gates, grupos y bootstrap](03_RESULTADOS_Y_DECISION.md).
4. [Pruebas, coste, contabilidad y fallos](04_PRUEBAS_COSTE_Y_CONTABILIDAD.md).
5. [Reproducción, rutas y contenido del bundle](05_REPRODUCCION_Y_ENTREGA.md).

## Resultado principal
"""
        + main_table
        + """Menor es mejor en las columnas de error. Overall MiD y mean_MiD tienen agregaciones distintas; el informe 03 explica ambas. Estos resultados corresponden a validación interna V.

Ejecuté 15.000 updates científicos y 20 técnicos sintéticos. La recuperación fue cero. Mantengo el presupuesto no consumido sin reasignarlo. Dejé V13/R1, sus entrenamientos y sus resultados intactos.

El usuario autorizó el push de estos informes y del código en su petición del 12 de octubre. Esa autorización posterior permite publicar la rama V14; el handoff original conserva su prohibición histórica de push como evidencia del alcance anterior. La publicación no autoriza nuevos fits, cambios en V13 ni merge.

Las tablas se regeneran con `scripts/build_range_transfer_reports.py`. [REPORT_INPUTS.json](REPORT_INPUTS.json) registra los hashes de sus fuentes. Los JSON/CSV ligeros están en `evidence/`; las predicciones por consulta, checkpoints y trazas completas están en el ZIP.
""",
    )

    roles = read(run / "ROLE_AUDIT.json")["roles"]
    parents = read(run / "ADMISSION.json")["parents"]
    md(
        "01_EJECUCION_Y_METODO.md",
        """# 01. Trabajo realizado y método

## Pregunta y alcance

La campaña intenta reducir la sobreestimación del TTC corto conservando precisión media/larga, negativos, cobertura y coste. Aislé dos factores en una cabeza temporal: la masa de las muestras en la pérdida y un término relativo adicional. No entrené nuevos encoders ni PAIR. Los seis fits usan los mismos productores event-only finales de V13.

## Identidades y preparación

- Evidencia histórica V12: `1a686e338a1145c6794bec0704bdae4cda95f06d`.
- Base auditada V13 y base del worktree V14: `da068ebdeb2d967ee10b2afa348e1317b8720ad0`.
- Rama: `scientific-recovery-v14-range-transfer`.
- ZIP de handoff: `E_JEPA_TTC_RANGE_TRANSFER_20261010.zip`, SHA256 `96a95c1ad8d63cbe5097e96b004c7f0d5f67e6a7707deb1ef6a7189e832ebb78`.

Leí el handoff y readmenow.txt, comprobé sus identidades, creé el worktree separado e integré el patch de referencia tras `git apply --check`. Inventarié procesos, recursos y finales disponibles. V14 no tomó la propiedad de la cola V13. Cuando V13 liberó OWNER.json tras terminar, registré su ausencia; no fabriqué un propietario ni reinicié la cola.

Verifiqué los padres, sus recibos finales, la procedencia P-only y los hashes de caché/normalizador H. El control V13 existente se entrenó en CUDA. Por eso entrené el control M0-L0 en CPU con la misma receta del factorial, en vez de tratar ambos endpoints como idénticos.
"""
        + table(
            ["Padre congelado", "Updates", "SHA256 del checkpoint"],
            [[p, 6840 if p == "PAIR_E_MATCHED" else 30330, f"`{h}`"] for p, h in parents.items()],
        )
        + """## Separación de roles
"""
        + table(
            ["Rol", "Uso", "Filas", "Grupos"],
            [
                [r, usage, roles[r]["rows"], len(roles[r]["groups"])]
                for r, usage in [
                    ("P", "Entrenamiento previo de productores y PAIR"),
                    ("H", "Normalizador y entrenamiento de cabezas"),
                    ("V", "Desarrollo interno: gates y selección"),
                ]
            ],
        )
        + """Comprobé grupos disjuntos y la correspondencia de tokens con manifiestos. H contiene features `[18422,8,17]`, timing `[18422,8,4]`, máscara `[18422,8]` y tres fases de expertos por consulta. Ocho slots es la capacidad de la cabeza: las máscaras y tiempos conservan las observaciones reales, sin densificar el historial.

El dominio admitido de H es `-10 < T < 0` o `0.1 < T <= 10`. La auditoría final encontró extremos aproximados −9,9993 y 9,9983 segundos. No eliminé targets para hacer viable el entrenamiento. La reutilización de la caché sensorial TRAIN40 no equivale a reutilizar sus productores: los pesos del factorial pertenecen a los padres matched entrenados en P.

## Matriz cerrada y factores

Entrené `M0/M1/M2 × L0/L1`, seed7, seis endpoints de 2.500 updates.

- **M0:** copié el vector nativo de masa sin renormalizarlo.
- **M1:** conservé la masa de cada grupo y repartí 0,5/0,3/0,1/0,1 entre crucial, media, larga y negativa presentes. Dentro de cada grupo y banda repartí la masa entre sus filas.
- **M2:** conservé la masa grupo×banda de M1. Busqué equilibrio entre (0,1;1], (1;2], (2;3] y entre (6;8], (8;10], acotando cada peso a [0,25;4] veces M1. El límite impide alcanzar igualdad perfecta cuando un subrango es escaso.

Las masas multiplican la contribución a la pérdida. El sampler sigue siendo uniforme con reemplazo, con los mismos draws por semilla. Los seis hashes de draws coinciden. Ninguna variante crea más episodios independientes mediante sus pesos.

**L0** conserva `simplex_t.model.training_loss`: fase, cuantiles y costes auxiliares. **L1** añade:

```text
z_i = (T_pred_i - T_target_i) / max(abs(T_target_i), 0.5 segundos)
L1 = L0 + 0.25 * mean_batch(N * mass_i * SmoothL1(z_i, 0; beta=1))
```

La predicción usa el mapping TTC nativo de `raw_location`. El target se reconstruye de su fase sin aplicarle el cap de salida. Conservé la política de emisión, la saturación y las regiones de gradiente plano; no introduje straight-through. La cabeza sigue siendo un corrector residual, no un router que seleccione el experto de menor coste auxiliar.

## Receta fija

CPU FP32, un thread numérico en el runner, batch128, GRU160 de dos capas, 17 features, salida residual, normalizador H congelado, AdamW (betas 0,9/0,999; eps 1e−8; weight decay 1e−3; foreach false), warmup100, pico LR 3e−4, final 3e−5, cosine y clipping de norma 1. La semilla fija inicialización y sampler. Todas las celdas parten de una inicialización nueva; ninguna parte del ganador de otra celda.

## Orden de ejecución y continuidad

1. N0: comprobación del paquete, estado vivo y recálculos públicos históricos.
2. N1: trazas históricas V12; controles H por hash y trazas matched. El informe 02 distingue la traza adicional terminada durante el factorial de los diagnósticos prefits.
3. N2: auditoría de masa, exposición y gradientes H sin updates.
4. Prueba sintética de continuidad: 10 updates frente a 5+restore+5.
5. Seis fits en el orden M0-L0, M1-L0, M2-L0, M0-L1, M1-L1, M2-L1. Guardado cada 100 updates y en pausa.
6. Congelación conjunta en SCREEN_ENDPOINTS.json, antes de inferir o puntuar esta matriz V.
7. Predicción V, métricas, bootstrap por grupos y selección con gates fijados.
8. Cierre NO_CANDIDATE, sin cuatro fits de réplica ni nueva evaluación de transferencia.

Cada checkpoint conserva pesos, AdamW, scheduler, cursor, RNG Python/NumPy/Torch, estado del sampler y contadores. El runner valida las fuentes y la admisión en fronteras durables. La publicación de informes de esta entrega no altera esos archivos científicos.

Fuentes: [admisión](evidence/ADMISSION.json), [roles](evidence/ROLE_AUDIT.json), [caché H final](evidence/H_FINAL_CACHE_AUDIT.json), [fuentes ejecutadas](evidence/SOURCE_BINDINGS.json), [endpoints](evidence/SCREEN_ENDPOINTS.json), [protocolo original](../range_transfer_handoff_20261010/docs/03_EXPERIMENT.md).
""",
    )

    mass = [r for r in rows(run / "MASS_AUDIT.csv") if r["group"] == "ALL"]
    public = read(run / "public_analysis/ANALYSIS.json")
    attribution = rows(run / "matched_H_raw_producers/ATTRIBUTION_BY_RANGE.csv")
    md(
        "02_DIAGNOSTICOS.md",
        """# 02. Masa, gradientes y atribución del sesgo

## Cuántos ejemplos cortos existen en H

H reúne 18.422 filas y ocho grupos. Sólo 52 filas están en (0,1;1] segundos. La tabla cuenta tracks y episodios descriptivos, definidos como tramos del track separados por más de un segundo. Esos episodios no garantizan escenas independientes.
"""
        + table(
            [
                "Subrango",
                "Filas",
                "Tracks",
                "Episodios",
                "Masa M0 %",
                "Masa M1 %",
                "Masa M2 %",
                "Draws esperados",
            ],
            [
                [
                    r["range"],
                    r["rows"],
                    r["tracks"],
                    r["episodes_gap_gt_1s_descriptive"],
                    *[fmt(100 * float(r[k])) for k in ("native_mass", "M1_mass", "M2_mass")],
                    fmt(r["expected_uniform_draws"], 1),
                ]
                for r in mass
            ],
        )
        + """En el conjunto crucial completo, M0 asigna cerca del 10 % de la masa; M1 y M2 asignan el 50 %. Para ≤1 s, el cap de M2 limita el aumento a cuatro veces M1. La mayor presión no añade información ni cambia la frecuencia de muestreo.

En 2.500×128 draws uniformes, cada fila aparece 17,37 veces en esperanza. El draw realizado seed7 cubrió todas las filas; el máximo observado fue 36 apariciones para una fila. H_EXPOSURE.csv, incluido en el bundle, contiene los conteos por token.

MASS_AUDIT.csv y H_METADATA_AUDIT.json describen la auditoría inicial de metadatos. El segundo conserva `FEATURE_GRADIENTS_PENDING` y `native_final_cache_audited=false` por su scope original. Para el estado final hay que consultar H_FINAL_CACHE_AUDIT.json, GRADIENT_AUDIT.json y la admisión, que sí documentan la caché final y los gradientes. Conservo ambos momentos sin sobrescribir su significado.

## Gradientes prefits

Medí por separado fase, cuantiles, costes y el término relativo añadido sobre batches H fijados por hash, con la inicialización seed7. No ejecuté optimizer.step en ese diagnóstico. GRADIENT_AUDIT.csv conserva 84 filas: siete rangos, tres masas y cuatro términos, además de fracciones de cap y clamp.

En ≤1 s, la norma de gradiente del término de fase pasa aproximadamente de 1,681 (M0) a 7,285 (M1) y 29,139 (M2). Para el término relativo añadido pasa de 0,416 a 2,029 y 8,118. Estas normas miden presión local al inicio, no el efecto final en V. No cambié el coeficiente 0,25 después de medirlas.

![Gradientes iniciales H](figures/H_GRADIENTS_BY_RANGE.png)

El gráfico muestra log10 de la norma; el suelo 1e−12 sirve sólo para dibujar. Las cifras sin esa transformación están en [GRADIENT_AUDIT.csv](evidence/GRADIENT_AUDIT.csv).

## Controles H y trazas por productor

Seleccioné 32 consultas H por hash, cubriendo los rangos disponibles. Ejecuté la cabeza nativa final V13 sobre su caché, y comparé su salida con la mediana de expertos al suprimir el residual. Durante el factorial completé además las trazas crudas de A5/C2F/PAIR sobre 118 observaciones reales de esos controles. Esas trazas adicionales no cambiaron el código, las features ni los hiperparámetros del factorial.
"""
        + table(
            [
                "Rango",
                "Productor",
                "Consultas",
                "Sesgo productor s",
                "Analítico log-ratio",
                "Residual log-ratio",
                "Sesgo mediana s",
                "Sesgo cabeza nativa s",
            ],
            [
                [
                    r["range"],
                    r["producer"],
                    r["queries"],
                    fmt(r["mean_producer_bias_s"]),
                    fmt(r["mean_analytic_log_ratio"], 6),
                    fmt(r["mean_residual_log_ratio"], 6),
                    fmt(r["mean_suppressed_head_bias_s"]),
                    fmt(r["mean_native_head_bias_s"]),
                ]
                for r in attribution
            ],
        )
        + """Un sesgo positivo significa sobreestimación. En los cuatro controles ≤1 s, A5 y C2F ya sobreestiman antes de la cabeza; el residual de la cabeza reduce el sesgo de la mediana, pero no lo elimina. En otros rangos aparecen cambios de signo o deterioros. Esta muestra fija permite localizar comportamientos; no estima causalidad en toda H ni demuestra que la cabeza sea la única causa del problema.

El selector interno fino/grueso de C2F, su flujo, alturas, log-ratios, soporte y máscaras quedan en los fragmentos y CSV del bundle. Los costes auxiliares de la cabeza no son pesos de un router H8.

## Fallo de forma known y adaptación diagnóstica

La primera traza matched falló con `producer known mask differs from support clock`. El productor emite known booleano por consulta, forma `[B]`, y soporte temporal `[B,3]`. El adaptador nativo exigía que ambas formas coincidieran. Reproduje el error y guardé la evidencia; no modifiqué V13.

En un script diagnóstico separado validé la forma real de known y construí las mismas 133 features PAIR mediante la función nativa pair133: token128, dt, log(dt), 1/dt, soporte actual y mínimo anterior/actual. La prueba compara esos campos y rechaza máscaras expandidas e intervalos inválidos. No añadí known a las features ni fabriqué timestamps.

La ruta adaptada completó las 118 observaciones. El máximo delta de fase CPU frente a la caché GPU fue 6,704e−5; no afirmo igualdad bit a bit de toda la ruta de productores. La adaptación no se usó para entrenar las seis cabezas y no resuelve ni publica una corrección de Dev32 en V13.

## Evidencia V12/FCWD histórica

"""
        + f"El recálculo público conservó {public['eligible']} consultas elegibles de {public['population']}, {public['excluded']} exclusiones y {public['independent_sequences']} secuencias. mean_MiD H8 fue {public['h8_mid']:.6f}, frente a {public['garl_mid']:.6f} del comparador Garl. No había targets ≤1 s. overall_MiD permanece ausente por soporte de bandas incompleto.\n\n"
        + """Ejecuté 96 forwards de cabezas históricas en 32 consultas publicadas y vecinos, además de 512 observaciones de productores. Los pesos TRAIN40 aparecen sólo en esta explicación histórica V12. Los pesos del factorial V14 son matched P-only.

FCWD y Dev32 son desarrollo expuesto. No usé sus etiquetas para ajustar o seleccionar las recetas V14. La evidencia histórica no constituye confirmación externa, una nueva evaluación de los seis endpoints ni una afirmación de SOTA universal.

Fuentes: [masa](evidence/MASS_AUDIT.csv), [atribución H](evidence/matched_H_raw_producers/ATTRIBUTION_BY_RANGE.csv), [traza adaptada](evidence/matched_H_raw_adapted/RESULT.json), [fallo original](evidence/matched_H_raw/FAILURE.json), [recálculo FCWD](evidence/public_analysis/ANALYSIS.json).
""",
    )

    gate_table = table(
        ["Candidato", "Mejora MiDc %", "Gates incumplidos", "Elegible"],
        [
            [
                name,
                fmt(
                    100
                    * (
                        1
                        - metrics[name]["bins"]["crucial"]["group_macro_mid"]
                        / reference["bins"]["crucial"]["group_macro_mid"]
                    )
                ),
                ", ".join(k for k, v in selection["gates"][name]["checks"].items() if not v),
                "No",
            ]
            for name in names
            if name != CONTROL
        ],
    )
    warning_table = table(
        [
            "Receta",
            "Error signo %",
            "p95 sobreestimación crucial s",
            "Aviso tardío % (n=76)",
            "Falsa alerta % (n=16356)",
        ],
        [
            [
                n,
                fmt(100 * metrics[n]["diagnostics"]["sign_error_rate"]),
                fmt(metrics[n]["p95_positive_crucial_overestimate_s"]),
                fmt(100 * metrics[n]["late_warning"]["rate"]),
                fmt(100 * metrics[n]["false_alert"]["rate"]),
            ]
            for n in names
        ],
    )
    ci_table = table(
        ["Candidato menos control", "Δ MiDc", "IC95 % bootstrap", "Grupos", "Remuestreos"],
        [
            [
                n,
                fmt(bootstrap[n]["delta_group_macro_MiDc"]),
                f"[{fmt(bootstrap[n]['CI95'][0])}; {fmt(bootstrap[n]['CI95'][1])}]",
                bootstrap[n]["groups"],
                bootstrap[n]["replicates"],
            ]
            for n in names
            if n != CONTROL
        ],
    )
    group_table = table(
        ["Grupo V", *[n.replace("RT-", "").replace("-S7", "") for n in names]],
        [
            [g, *[fmt(groups[n][g]["bins"]["crucial"]["group_macro_mid"]) for n in names]]
            for g in sorted(groups[CONTROL])
        ],
    )
    sub_table = table(
        ["Subrango", "Consultas", *[n.replace("RT-", "").replace("-S7", "") for n in names]],
        [
            [
                r["range"],
                r["queries"],
                *[
                    fmt(next(x["group_macro_MiD"] for x in details[n] if x["range"] == r["range"]))
                    for n in names
                ],
            ]
            for r in details[CONTROL]
        ],
    )
    md(
        "03_RESULTADOS_Y_DECISION.md",
        """# 03. Resultados, gates y decisión

## Población y métricas

Evalué 16.432 consultas V de ocho grupos, idénticas para las seis celdas. Todas produjeron predicciones finitas y MiD válido en el soporte evaluado. Los seis endpoints ya estaban congelados. V es desarrollo interno; estos números no son test oficial de GarlTTC.

Por consulta, con delta=0,1 segundos:

```text
MiD_i = 10000 * abs(log(1 - 0.1 / T_pred_i) - log(1 - 0.1 / T_true_i))
```

MiDc cubre (0,1;3], MiDs (3;6], MiDl (6;10] y MiDn negativos del protocolo. Primero calculo la media de MiD dentro de cada grupo y banda; después promedio grupos. El agregado nativo es `0.5*MiDc + 0.3*MiDs + 0.1*MiDl + 0.1*MiDn`. mean_MiD es la media por consulta de toda la población y da más peso a los rangos con más filas. MAE está en segundos y también agrupa consultas. No sustituyo una agregación por otra según favorezca una receta.

Si falta una banda no completo overall_MiD. En esta V están presentes las cuatro: 1.845 cruciales, 4.284 medias, 3.283 largas y 7.020 negativas.

## Factorial completo
"""
        + main_table
        + """Las diferencias de últimas cifras entre tablas de subrangos y métricas originales proceden del CSV y de las formas numéricas log/log1p del recálculo descriptivo. La selección utiliza METRICS.json congelado, no las tablas redondeadas de este informe.

## Gates fijados antes de ver V

1. MiDc debe bajar al menos 10 % y mejorar en al menos 60 % de los grupos con cruciales.
2. El agregado nativo no puede empeorar.
3. Cada banda media/larga/negativa puede empeorar como máximo `max(2 MiD, 5 % del control)`; el MAE global no puede superar 1,02 veces el control.
4. Cobertura completa; error de signo como máximo +0,5 puntos porcentuales; p95 de la parte positiva del error crucial como máximo 1,05 veces el control.
5. Aviso tardío y falsa alerta pueden aumentar como máximo un punto porcentual cuando cada denominador tenga al menos 30 casos. El soporte insuficiente impide promover una receta.

Con el control observado, los máximos de MiDs/MiDl/MiDn son aproximadamente 57,514 / 54,828 / 64,451. Todas las variantes superan el límite largo. Los denominadores de aviso tienen soporte suficiente; NO_CANDIDATE se debe a incumplimientos, no a una banda ausente.
"""
        + gate_table
        + """## Interpretación de los factores

**M1-L0** obtiene el menor MiDc (134,470 frente a 156,832; mejora del 14,26 %). El rango largo pasa de 52,217 a 76,397 y el negativo de 61,382 a 84,208; MAE sube de 2,512 a 2,979 s. La redistribución mejora lo corto a costa de otras exigencias del protocolo.

**M0-L1** ofrece el menor agregado y MAE, pero sólo reduce MiDc un 4,62 %. Empeora el rango largo y el aviso tardío. No cumple el objetivo por elegir una métrica favorable.

**L1 con M1/M2** reduce MAE respecto a sus L0 correspondientes y mantiene parte de la mejora crucial. Aun así incumple las bandas media, larga y negativa. **M2** no establece una mejora global frente a M1 en esta semilla. No hay réplica que permita estimar la variación entre semillas.

Esta matriz aporta evidencia de un compromiso entre rangos dentro de la receta ensayada. No demuestra que una ponderación distinta, otro decoder o un productor nuevo resuelvan el problema; esas pruebas requerirían otro protocolo.

## Signo, sobreestimación y avisos
"""
        + warning_table
        + """La frontera de aviso urgente es 1 s. Un TTC negativo predicho no cuenta como aviso. El denominador de aviso tardío son las 76 consultas con target (0,1;1]; el de falsa alerta reúne las restantes 16.356 consultas. Los porcentajes altos de aviso tardío describen una limitación seria de esta cohorte. No interpreto tasas bajas de falsa alerta como seguridad vehicular.

El p95 anterior calcula `quantile(max(pred-target,0),0.95)` sobre todos los cruciales. Es distinto de un percentil calculado sólo sobre los errores positivos.

## Bootstrap pareado por grupos
"""
        + ci_table
        + """Cada contraste usa 1.000 remuestreos de ocho grupos completos, en diez fragmentos reproducibles de 100 draws. Los índices se comparten por construcción entre control/candidato dentro de cada contraste. No trato las ventanas correlacionadas como observaciones independientes. Los intervalos describen la diferencia MiDc de esta seed7; no incorporan variación de entrenamiento entre semillas ni convierten el análisis en test externo. Que un intervalo crucial sea favorable no elimina los otros gates.

## MiDc por grupo
"""
        + group_table
        + """Los JSON GROUP_METRICS contienen las demás bandas y diagnósticos por grupo. SUBRANGE_GROUP_RESULTS.csv conserva el detalle por grupo y subrango. Las celdas ausentes se mantienen ausentes.

## Subrangos descriptivos: MiD macro por grupos
"""
        + sub_table
        + """Estas tablas permiten localizar el comportamiento dentro de la banda crucial. No añadí gates ni reajusté la selección después de obtenerlas.

## Decisión cerrada

`SELECTION_FREEZE.json` registra candidato null y NO_CANDIDATE. No ejecuté las cuatro réplicas candidato/control de seeds13/23 ni una evaluación de transferencia de endpoints nuevos. Conservo todos los resultados, incluidos los brazos que empeoran. Los 10.000 updates científicos no consumidos no financian otro experimento.

Fuentes: [resultados CSV](evidence/FACTORIAL_RESULTS.csv), [gates congelados](evidence/SELECTION_FREEZE.json), [decisión final](evidence/NEXT_DECISION.json), carpetas `evidence/evaluation/screen/` y `evidence/bootstrap/screen/`.
""",
    )

    fit_seconds = sum(p["elapsed_fit_s"] for p in provenance.values())
    cost_table = table(
        ["Fit", "CPU fit + persistencia s", "Forward de 16432 consultas desde caché s"],
        [
            [
                n,
                fmt(provenance[n]["elapsed_fit_s"]),
                fmt(read(run / "evaluation/screen" / n / "COST.json")["seconds"]),
            ]
            for n in names
        ],
    )
    resume = read(run / "qa/RESUME_TEST.json")
    md(
        "04_PRUEBAS_COSTE_Y_CONTABILIDAD.md",
        """# 04. Pruebas, coste, contabilidad y límites

## Verificaciones realizadas

- 41 tests y 17 subtests de los módulos nuevos antes de iniciar los fits: masa, objetivo, paridad L0, dominio, gates, separación de roles, presupuesto y contratos de checkpoint. Conservo el stdout; su configuración quiet reduce el resumen a puntos.
- Ruff y Pyright sobre la integración; Pyright informó cero errores y advertencias del analizador. Su aviso de `.venv` local ausente no desencadenó una instalación: usé el entorno existente del proyecto.
- Prueba real CPU FP32 de continuidad: 10 updates sintéticos continuos frente a 5+restauración+5. Su consumo total fue 20 updates técnicos.
- Prueba independiente del adaptador diagnóstico PAIR: forma known por consulta, 133 features exactas y rechazo de intervalos/formas inválidos. Cero updates.
- Auditoría final de seis endpoints: SHA256, 2.500 entradas de curva, pesos finitos, fuentes científicas sin cambios, cohortes alineadas y hashes de predicciones.
- Inspección visual de curvas y gradientes; verificación SHA256 de las entradas del bundle.

No presento esto como ejecución de la suite completa del repositorio, una prueba de robustez bajo corrupciones, calibración probabilística ni validación de seguridad. No hubo un corte real durante un fit científico: la prueba de resume usa el fixture sintético con la misma maquinaria de entrenamiento/checkpoint.

En la preparación del push también ejecuté Ruff sobre todo el repositorio: informó 1.780 errores heredados y el formateador señaló 370 archivos. Comprobé que las rutas con errores pertenecen a archivos ya versionados que esta entrega no modifica. Los checks dirigidos a V14 y al generador pasan. No afirmo que la CI global esté verde ni reformateo el histórico para publicar estos resultados. [PUBLICATION_CHECKS.json](evidence/qa/PUBLICATION_CHECKS.json) registra la comprobación; los logs completos están en el bundle.

## Resume: comprobaciones exactas
"""
        + table(
            ["Estado comparado", "Igualdad exacta"],
            [[k, "PASS" if v else "FAIL"] for k, v in resume["checks"].items()],
        )
        + """El modelo, AdamW, scheduler, sampler y RNG coinciden. Los hashes completos de los dos archivos checkpoint pueden diferir por identidad y metadatos de los runs; la prueba compara sus estados relevantes, draws y pérdidas.

## Contabilidad física
"""
        + table(
            ["Concepto", "Consumido", "Máximo"],
            [
                ["Científicos", ledger["scientific"], 25000],
                ["Técnicos sintéticos", ledger["technical"], 500],
                ["Recuperación superior", ledger["recovery_upper"], 4500],
                ["Pendientes inciertos", ledger["pending_upper"], "incluidos en el límite"],
                ["Físicos superiores", ledger["physical_upper"], 30000],
            ],
        )
        + """Cada fit tiene UPDATE_JOURNAL y checkpoint durable. La cota física suma científicos, técnicos, recuperación e incertidumbre. Los seis fits completaron su endpoint sin recuperación. La selección negativa cerró la cola; el presupuesto libre no se reasigna.

## Costes con scope
"""
        + cost_table
        + f"La suma de los seis tiempos de fit es {fit_seconds:.3f} s ({fit_seconds / 60:.3f} min). Incluye trabajo de la cabeza en H y persistencia; no es el coste de entrenar productores. El reloj de la invocación principal registró {read(run / 'COST_SCOPE.json')['seconds']:.3f} s para diagnósticos y trabajo con cabezas desde caché. No sumo tiempos de procesos concurrentes como si fueran tiempo total de pared.\n\n"
        + """Los forwards V duran aproximadamente 2,44–2,49 s por 16.432 consultas en batch128. Son medidas de esa ejecución, con conversión de salidas. Excluyen productores, lectura sensorial, construcción del historial, preparación de features y servicios de streaming. Dividir por el número de consultas no produce una latencia E2E ni una latencia batch1 comparable con Garl.

La traza H adaptada registra 32,581 s de CPU para caché sensorial, productores e instrumentación. La traza histórica V12 registra 679,829 s con preparación cruda y captura. Tienen scopes distintos; no se usan para una clasificación de velocidad entre sistemas.

## Recursos y protección de V13

V14 usa CPU FP32 y un thread numérico en el runner. El guard comprueba memoria disponible, memoria comprometida Windows, RSS del proyecto, árbol de procesos propio y disco. La política fija reserva dura de 2 GiB, reentrada con 3 GiB, 23 GB de RSS de proyecto, 4 GiB para V14 y 10 GB libres después de reservas. Los snapshots y fallos previos por recursos quedan en el bundle. No cambié drivers, paging, energía ni afinidades.

## Fallos conservados y alcance de las correcciones

Conservo reintentos históricos por recursos/imports, fallos transitorios de escritura Windows y el rechazo de forma known. Para los JSON de V14 reutilicé las primitivas nativas de lectura compartida y reemplazo atómico con reintentos limitados. La prueba sintética y los journals permiten auditar la continuidad; ninguno de esos diagnósticos agregó updates científicos.

El adaptador H separado resuelve la extracción diagnóstica V14 y deja el error nativo V13 registrado. No afirmo haber reparado el Dev32 de otra campaña. La ausencia de réplicas y transferencia nueva obedece al gate científico, no a ese error.

Fuentes: [contabilidad](evidence/ACCOUNTING.json), [resume](evidence/qa/RESUME_TEST.json), [auditoría final](evidence/SCIENTIFIC_DELIVERY_AUDIT.json), [coste](evidence/COST_SCOPE.json), [comandos ejecutados](evidence/COMMANDS_EXECUTED.md).
""",
    )

    md(
        "05_REPRODUCCION_Y_ENTREGA.md",
        """# 05. Reproducción, rutas y entrega

## Rutas del host original

```text
V14:
C:\\Users\\Álvaro Schwiedop\\Desktop\\KriptaStudios\\EVOCON_JEPA_Codex_Handoff\\e-jepa-ttc-v14-range-transfer

Resultados V14, relativos a esa raíz:
artifacts/range_transfer_20261010/

Informes de esta entrega:
docs/range_transfer_results_20261012/

V13, dependencia de sólo lectura:
../e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008/

Evidencia V12 de sólo lectura:
../e-jepa-ttc-v12-efficient-context/
```

## Qué contiene cada nivel

| Ruta relativa a la campaña V14 | Contenido |
|---|---|
| fits/RT-M*-L*-S7/checkpoint_last.pt | Seis pesos nuevos, estado de optimizador y continuidad |
| fits/*/CHECKPOINT_RECEIPT.json | Endpoint, contador y hash de checkpoint |
| fits/*/CURVE.csv | 2.500 updates por fit: loss, norma previa a clipping y LR |
| evaluation/screen/*/PREDICTIONS.csv | 16.432 predicciones por celda sobre la misma V |
| evaluation/screen/*/LOSSES_PER_QUERY.csv | MiD, error absoluto/relativo, sesgo y cobertura por consulta |
| evaluation/screen/*/METRICS.json | Métricas usadas en los gates |
| evaluation/screen/*/GROUP_METRICS.json | Todas las métricas por grupo |
| evaluation/screen/*/SUBRANGE*.csv | Desglose descriptivo por subrango y grupo |
| bootstrap/screen/ | Cinco contrastes, fragmentos y sus intervalos |
| MASS_AUDIT.csv, H_EXPOSURE.csv, GRADIENT_AUDIT.csv | Masa, draws y presión del objetivo |
| matched_H_controls/, matched_H_raw_adapted/ | Trazas de cabezas y productores H |
| v12_cached_attribution/, v12_raw_attribution/ | Diagnóstico histórico; pesos TRAIN40 fuera del factorial |
| qa/ | Pruebas, scripts auxiliares, errores y checkpoints sintéticos |
| NEXT_DECISION.json, SELECTION_FREEZE.json | Cierre NO_CANDIDATE y gates |

GitHub contiene la implementación, el handoff, informes, figuras y evidencia ligera por banda/grupo. El ZIP añade pesos, predicciones y pérdidas por consulta, trazas y logs. No subo datos sensoriales crudos al repositorio ni incluyo las cachés H/V completas de V13 en este bundle.

## Regenerar estos informes sin entrenar

Desde la raíz V14, con el Python existente:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B scripts/build_range_transfer_reports.py
```

El generador verifica fuentes científicas, pesos, recibos y predicciones, copia la evidencia ligera y genera las tablas. Requiere la campaña completa del ZIP en `artifacts/range_transfer_20261010`. No ejecuta modelos ni recalcula la selección.

Para crear un nuevo ZIP con esos informes y la campaña:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B scripts/build_range_transfer_reports.py --bundle
```

El generador verifica los hashes del ZIP y escribe un `.sha256` y un `.verification.json` externos. El manifiesto interno identifica cada archivo incluido. Los informes históricos en artifacts describen el momento de ejecución anterior al push; los informes de esta carpeta documentan la publicación autorizada después.

## Verificar el ZIP sin Torch ni datos

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B scripts/build_range_transfer_reports.py --verify-zip 'artifacts/RANGE_TRANSFER_V14_REPORTS_RESULTS_20261012.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath 'artifacts/RANGE_TRANSFER_V14_REPORTS_RESULTS_20261012.zip'
```

Compare el hash externo con el entregado y compruebe el manifiesto interno. Una coincidencia de hashes acredita integridad de los archivos, no validez científica por sí sola.

## Continuidad de entrenamiento: campaña cerrada

El comando real usado fue `pwsh -NoProfile -File scripts/Resume-RangeTransfer.ps1`. Conserva el checkpoint y no acorta endpoints. No hace falta invocarlo para leer o regenerar informes. La cola terminó NO_CANDIDATE y no debe ampliarse con el presupuesto sobrante.

Un replay completo del entrenamiento requiere las fuentes históricas, los tres padres matched, los manifiestos y las cachés H/V de V13 con los hashes admitidos. Los paths absolutos de los recibos pertenecen al host original; moverlos a otra máquina exige resolver esas dependencias sin alterar la identidad científica. El ZIP no pretende sustituir los datasets ni esos padres.

## Reproducción de diagnósticos y pruebas

Los scripts de `artifacts/range_transfer_20261010/qa/` conservan los comandos auxiliares. TRACE_MATCHED_H_ADAPTED.py usa matched_pair_adapter.py y requiere los padres/inputs locales; no necesita V ni optimizer.step. VERIFY_SCIENTIFIC_DELIVERY.py verifica resultados y genera desglose por consulta/subrango. Los scripts antiguos de reconstrucción tienen nombres históricos: use el generador de informes de esta entrega para empaquetar el cierre final.

Las pruebas de integración publicadas se ejecutan así:

```powershell
$env:PYTHONPATH='src;.'
$env:OMP_NUM_THREADS='1'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m pytest tests/test_range_transfer.py tests/test_range_transfer_integration.py -q --override-ini addopts=
```

## Decisión y trabajo no ejecutado

El usuario puede revisar qué rango empeora y dónde empieza el sesgo antes de autorizar otro estudio. Esta entrega no elige nuevos coeficientes, encoders, semillas, calibraciones o splits. No hay resultado de réplica, transferencia nueva, test oficial, robustez global ni latencia E2E V14. Una investigación posterior necesita definir su propia hipótesis y sus gates antes de ejecutarla.
""",
    )
    write_json(
        output / "REPORT_INPUTS.json",
        {
            "schema": "v14_detailed_reports_v1",
            "scientific_source_base": BASE,
            "scientific_sources_unchanged": True,
            "selection_changed": False,
            "new_optimizer_updates": 0,
            "inputs_sha256": dict(sorted(inputs.items())),
        },
    )


def verify_bundle(path: Path) -> dict[str, Any]:
    """Verify every manifest entry, membership and absence of duplicate ZIP names."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        if set(names) != set(manifest) | {"BUNDLE_MANIFEST.json"}:
            raise ValueError("ZIP membership differs from manifest")
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"ZIP hash mismatch: {name}")
    return {
        "status": "PASS",
        "entries_verified": len(manifest),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def bundle(run: Path, output: Path) -> Path:
    """Package scientific evidence and reports without changing the original ZIP."""
    destination = ROOT / "artifacts/RANGE_TRANSFER_V14_REPORTS_RESULTS_20261012.zip"
    files: dict[str, Path] = {}
    for folder in (
        run,
        output,
        ROOT / "docs/range_transfer_handoff_20261010",
        ROOT / "operational/range_transfer",
        ROOT / "src/e_jepa_ttc/range_transfer",
    ):
        for path in folder.rglob("*"):
            if (
                path.is_file()
                and path.suffix not in (".zip", ".lock", ".pyc")
                and ".tmp" not in path.name
                and "__pycache__" not in path.parts
            ):
                files[path.relative_to(ROOT).as_posix()] = path
    for path in [
        *ROOT.glob("tests/test_range_transfer*.py"),
        ROOT / "scripts/Resume-RangeTransfer.ps1",
        ROOT / "scripts/range_transfer_v12_trace.py",
        Path(__file__),
        ROOT / "docs/range_transfer_integration_20261010.md",
        ROOT / "AGENTS.md",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]:
        files[path.relative_to(ROOT).as_posix()] = path
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for relative in read(run / "SOURCE_BINDINGS.json")["runner_sources"]:
        files[relative] = ROOT / relative
    extra = {
        "README_BUNDLE.md": (
            "# V14: informes y resultados\n\nEmpiece por [los informes detallados](docs/range_transfer_results_20261012/README.md).\n\nEl árbol conserva rutas relativas al repositorio. Contiene seis endpoints, predicciones, pérdidas, curvas, diagnósticos, pruebas e informes. Los datos sensoriales y cachés V13 no se incluyen.\n\nEstado: NO_CANDIDATE; 15000 científicos + 20 técnicos; cero recuperación.\n"
        ).encode(),
        "BUNDLE_PROVENANCE.json": json.dumps(
            {
                "git_head_at_packaging": commit,
                "scientific_base": BASE,
                "scientific_sources_unchanged": True,
                "optimizer_updates_this_delivery": 0,
                "repository": "Kripta-Studios/e-jepa-ttc",
                "branch": "scientific-recovery-v14-range-transfer",
            },
            indent=2,
        ).encode(),
    }
    manifest = {name: sha(path) for name, path in sorted(files.items())}
    manifest.update({name: hashlib.sha256(data).hexdigest() for name, data in extra.items()})
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        for name, data in extra.items():
            archive.writestr(name, data)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    verification = verify_bundle(destination)
    destination.with_suffix(".zip.sha256").write_text(
        f"{verification['sha256']}  {destination.name}\n", encoding="ascii"
    )
    write_json(destination.with_suffix(".zip.verification.json"), verification)
    print(json.dumps({"bundle": str(destination), **verification}))
    return destination


def main() -> None:
    """Generate reports, optionally package, or independently verify an archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/range_transfer_20261010")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/range_transfer_results_20261012"
    )
    parser.add_argument("--bundle", action="store_true")
    parser.add_argument("--verify-zip", type=Path)
    args = parser.parse_args()
    if args.verify_zip:
        print(json.dumps(verify_bundle(args.verify_zip)))
        return
    generate(args.run.resolve(strict=True), args.output.resolve())
    print(f"Detailed reports: {args.output}")
    if args.bundle:
        bundle(args.run.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
