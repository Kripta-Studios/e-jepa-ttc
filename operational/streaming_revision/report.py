"""Regenerate streaming research tables and conclusions from recorded measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from operational.efficient_context.common import digest
from operational.train40_system.durable_io import atomic_json


def table(frame: pd.DataFrame) -> str:
    """Render a small table without an optional tabulate dependency."""
    rows = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = [
            f"{v:.3f}"
            if isinstance(v, float) and np.isfinite(v)
            else "N/D"
            if pd.isna(v)
            else str(v)
            for v in values
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def run(root: Path, document: Path | None = None) -> None:
    """Never pool latency across input scopes or partially completed systems."""
    lines = [
        "# Optimización experimental de streaming — 2026-10-10",
        "Implementación: `operational/streaming_revision`. Los modelos y las tres cabezas H8 "
        "conservan sus pesos. Los datos y resultados de esta investigación están en E:.",
        "## Condiciones de interpretación",
        "Los pilotos usan 12 consultas de dos secuencias TRAIN40, reservadas para validar "
        "el estudiante pero vistas por el profesor. Son pruebas de ingeniería, no una "
        "evaluación externa ni evidencia SOTA. Se mantiene el mismo conjunto de consultas "
        "entre variantes. MiD es la media por muestra, no el overall_MiD ponderado oficial.",
        "Se informan las tres cabezas H8. La mediana y p95 excluyen la primera consulta "
        "de cada secuencia, pero pueden incluir otras recargas y compilaciones. Son solo "
        "diez observaciones de latencia; no constituyen una garantía de tiempo real.",
        "HDF5 y streaming residente son rutas diferentes. El segundo reproduce eventos "
        "grabados en un búfer causal: excluye la lectura de disco para ambos modelos e "
        "incluye la ingestión del paquete. Los saltos grandes entre consultas se recargan "
        "como arranques fríos; no se ha medido un sensor físico ni ingestión continua.",
    ]
    bindings, stage_rows, parity_rows, cold_rows = {}, [], [], []
    for name in (
        "gpu_warp_pilot",
        "live_roi_pilot",
        "isolated_roi_graph",
        "isolated_fixed",
        "isolated_fixed_repeat2",
        "isolated_fixed_repeat3",
    ):
        directory = root / name
        if not (directory / "SUMMARY.csv").exists():
            continue
        summary = pd.read_csv(directory / "SUMMARY.csv")
        predictions = pd.read_csv(directory / "PREDICTIONS.csv")
        receipt = json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
        complete = summary.loc[summary.complete.astype(bool)]
        bindings[name] = {
            file: digest(directory / file)
            for file in ("SUMMARY.csv", "PREDICTIONS.csv", "RESULT.json", "ROWS.jsonl")
        }
        lines.extend(
            [
                f"## {name}",
                f"Estado: {receipt['status']}.",
                table(
                    complete[["variant", "n", "warm_median_ms", "warm_p95_ms", "mean_MiD", "MAE_s"]]
                ),
            ]
        )
        if "h8_reference" in set(complete.variant):
            reference = complete.set_index("variant").loc["h8_reference"]
            for _, variant in complete.loc[
                complete.variant.isin(["h8_warp", "h8_warp_graph_b2", "h8_warp_graph_fixed"])
            ].iterrows():
                reduction = 100 * (1 - variant.warm_median_ms / reference.warm_median_ms)
                lines.append(
                    f"{variant.variant}: {reduction:.1f}% menos mediana total "
                    "que H8_reference dentro de este mismo piloto."
                )
        timing_columns = [
            c
            for c in (
                "input_ingest_ms",
                "prep_lookup_warp_ms",
                "prep_read_ms",
                "prep_crop_map_ms",
                "prep_encode_cache_ms",
                "prep_stack_ms",
                "producer_ms",
                "head_and_commit_ms",
                "total_ms",
            )
            if c in predictions
        ]
        for variant, group in predictions.groupby("variant"):
            for _, row in group.loc[group.groupby("sequence").cumcount() == 0].iterrows():
                cold_rows.append(
                    {
                        "pilot": name,
                        "variant": variant,
                        "sequence": row.sequence,
                        "total_ms": row.total_ms,
                    }
                )
            warm = group.loc[group.groupby("sequence").cumcount() > 0]
            stage_rows.append(
                {
                    "pilot": name,
                    "variant": variant,
                    "n": len(warm),
                    **{c: float(warm[c].mean()) for c in timing_columns},
                }
            )
        pivot = predictions.pivot(index="sample_token", columns="variant", values="ttc")
        for left, right in (
            ("h8_warp", "h8_warp_b2"),
            ("h8_warp_b2", "h8_warp_graph_b2"),
            ("h8_warp", "h8_warp_graph_fixed"),
            ("garl_event", "garl_event_graph"),
            ("garl_event", "garl_event_native_input"),
        ):
            if left in pivot and right in pivot:
                pair = pivot[[left, right]].dropna()
                parity_rows.append(
                    {
                        "pilot": name,
                        "reference": left,
                        "variant": right,
                        "n": len(pair),
                        "max_ttc_difference_s": float(
                            np.max(np.abs(np.asarray(pair[left]) - np.asarray(pair[right])))
                        ),
                    }
                )
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(root / "STAGES.csv", index=False)
    pd.DataFrame(parity_rows).to_csv(root / "NUMERICAL_PARITY.csv", index=False)
    cold = pd.DataFrame(cold_rows)
    cold.to_csv(root / "COLD_STARTS.csv", index=False)
    if "isolated_fixed" in bindings:
        first_sequence = cold.loc[cold.pilot == "isolated_fixed", "sequence"].iloc[0]
        starts = cold.loc[(cold.pilot == "isolated_fixed") & (cold.sequence == first_sequence)]
        lines.extend(
            [
                "## Primera consulta, incluida la compilación",
                table(cast(pd.DataFrame, starts[["variant", "total_ms"]])),
                "Estos arranques se conservan y no se confunden con la mediana de "
                "consultas posteriores. El calentamiento previo a uso en vivo aún "
                "requiere un protocolo explícito.",
            ]
        )
    isolated = stages.loc[stages.pilot == "isolated_fixed"] if len(stages) else stages
    if len(isolated):
        lines.extend(
            [
                "## Coste medio por etapa en consultas posteriores al arranque",
                "La preparación test12 se suspendió temporalmente y se reanudó "
                "automáticamente; véase CPU_ISOLATION_V2.json. El escritorio WDDM "
                "sigue activo. La etiqueta genérica de contención del runner se "
                "complementa con ese recibo de aislamiento.",
                table(cast(pd.DataFrame, isolated)),
            ]
        )
    external = pd.read_csv(root / "external/METRICS.csv")
    packet_path = root / "packet_blocks/cpu_trial/RESULT.json"
    if packet_path.exists():
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        bindings["packet_cpu_profile"] = {"RESULT.json": digest(packet_path)}
        packet_frame = pd.DataFrame(packet["summary"])
        lines.extend(
            [
                "## Ingestión compacta CPU",
                table(packet_frame),
                "Doce consultas TRAIN40, una ingestión por consulta y tres lecturas "
                "ROI por consulta en orden aleatorio. La inferencia test12 seguía activa: "
                "estos tiempos no se combinan con el piloto GPU aislado. Las repeticiones "
                "de lectura no son ejecuciones independientes de extremo a extremo.",
                f"Igualdad exacta de eventos y voxels: "
                f"{sum(p['raw_and_voxel_exact'] for p in packet['parity'])}/"
                f"{len(packet['parity'])} comprobaciones. Cero segundos GPU.",
                "La aritmética compacta elimina copias temporales de timestamps y "
                "exploraciones duplicadas; está activa por defecto en PacketRing. "
                "La división adicional en bloques de 50 ms es opcional: empeora "
                "ligeramente la ingestión frente a la aritmética compacta, aunque "
                "mejora algunas lecturas. La reducción de ingestión no demuestra "
                "la misma reducción de latencia total.",
            ]
        )
    stream_path = root / "fcwd_stream_cpu/RESULT.json"
    stream_complete = False
    if stream_path.exists():
        stream = json.loads(stream_path.read_text(encoding="utf-8"))
        if stream["status"] in ("COMPLETE", "BASELINE_PARITY_FAILED"):
            stream_complete = stream["status"] == "COMPLETE"
            stream_metrics = pd.read_csv(stream_path.parent / "METRICS.csv")
            bindings["fcwd_raw_stream_cpu"] = {
                name: digest(stream_path.parent / name)
                for name in ("RESULT.json", "METRICS.csv", "PREDICTIONS.csv", "ROWS.jsonl")
            }
            lines.extend(
                [
                    "## FCWD completo desde eventos crudos",
                    f"Estado: {stream['status']}. 630 consultas cronológicas, tres "
                    "secuencias, sin ajustes de pesos ni selección por etiquetas. "
                    "Las etiquetas se incorporan después de todas las predicciones. "
                    "La inferencia se ejecuta en CPU; no mide latencia GPU.",
                    table(
                        cast(
                            pd.DataFrame,
                            stream_metrics.loc[
                                stream_metrics.sequence == "ALL",
                                [
                                    "variant",
                                    "eligible_n",
                                    "mean_MiD",
                                    "MiDc",
                                    "MAE_s",
                                    "signed_bias_s",
                                ],
                            ],
                        )
                    ),
                    "El CSV incluye resultados por secuencia y FR por banda. "
                    f"Paridad del H8 de referencia con las predicciones congeladas: "
                    f"{stream['baseline_cpu_parity_admitted']}; diferencia máxima "
                    f"{stream['baseline_max_difference_s']:.6g} s. "
                    "Los intervalos usan agrupación por secuencia; solo hay tres grupos. "
                    "FCWD ya es un conjunto expuesto y no sustituye test12.",
                ]
            )
            diagnostic_path = stream_path.parent / "ERROR_BANDS.csv"
            if diagnostic_path.exists():
                errors = pd.read_csv(diagnostic_path)
                reuse = pd.read_csv(stream_path.parent / "REUSE.csv")
                bindings["fcwd_diagnostics"] = {
                    name: digest(stream_path.parent / name)
                    for name in ("ERROR_BANDS.csv", "REUSE.csv", "DIAGNOSTICS.json")
                }
                lines.extend(
                    [
                        "### Error en TTC crucial y reutilización",
                        table(cast(pd.DataFrame, errors.loc[errors.band == "c"])),
                        table(reuse),
                        "El principal déficit frente a Garl se concentra en (0,3] s. "
                        "Una MAE global menor no implica mejor MiD ni mejor resultado "
                        "en el rango crucial. FR es cero para todas estas variantes "
                        "según el scorer utilizado. No hay banda negativa y no se "
                        "calcula un overall_MiD con pesos redistribuidos.",
                    ]
                )
    lines.extend(
        [
            "## Precisión externa disponible",
            "Se evaluaron todas las filas guardadas de DEV32 y FCWD. Estas tablas prueban "
            "truncamiento de historia y destilación sobre características preparadas; "
            "no prueban la reproyección de voxels ni su combinación con precisión reducida.",
            table(
                cast(
                    pd.DataFrame,
                    external[
                        [
                            "dataset",
                            "variant",
                            "eligible_n",
                            "mean_MiD",
                            "MiDc",
                            "MAE_s",
                            "signed_bias_s",
                        ]
                    ],
                )
            ),
            "No hay banda negativa etiquetada en estos conjuntos: overall_MiD permanece N/D; "
            "no se redistribuyen los pesos oficiales. Dev32 ya es un conjunto expuesto.",
            "## Cambios y límites",
            "- Búfer de paquetes con compresión sin pérdida: 9 bytes por evento típico frente "
            "a 17; coordenadas fuera de int16 usan fallback exacto. Descarte parcial de paquetes "
            "caducados, límite de memoria y errores explícitos ante huecos o retrocesos.",
            "- ROI recortada antes de descomprimir columnas; conserva eventos, orden y voxels. "
            "Garl recibe la misma optimización preservando el primer evento del sensor como "
            "origen temporal. El piloto exige igualdad exacta de sus tensores nativos.",
            "- Caché de características por objeto, secuencia, ROI y antigüedad; ocho "
            "observaciones con timestamps reales. Reproyección bilineal solo de voxels originales, "
            "sin cadenas de interpolaciones. No recupera eventos fuera de la ROI original y "
            "aproxima conteos, normalización y pequeños desfases temporales.",
            "- La caché de 17 características apenas ocupa memoria. Las prioridades de "
            "compresión son eventos y voxels. INT8 dinámico del estudiante en CPU fue más "
            "lento que FP32; no se promueve ni equivale a cuantización GPU.",
            "- El estudiante suprime C2F/PAIR en inferencia; mejora FCWD frente a H8 y empeora "
            "DEV32. H4/H2 son truncamientos de los pesos H8, no modelos reentrenados.",
            "- La captura completa GPU dinámica recompila con nuevas formas; el piloto "
            "live_graph_pilot_v2 agotó su presupuesto. La variante b2 fija el tamaño de los "
            "productores, incluye padding descartado y conserva las tres cabezas. "
            "graph_fixed añade padding con máscara a las cabezas para conservar ocho "
            "posiciones incluso en el arranque. Las búsquedas de timestamps usan uint32 "
            "para evitar que NumPy copie una columna completa al promover su tipo.",
            "- garl_event_native_input usa la preparación nativa desde la ventana completa "
            "del sensor, con el mismo lector persistente. garl_event añade recorte temprano "
            "exacto; garl_event_graph añade captura GPU. Ninguna variante Garl aplica "
            "reproyección ni aproximación. Son reproducciones locales, no los tiempos "
            "del hardware del artículo.",
            "## Fallos conservados",
            "distill_seed7 falló antes del primer update por una operación in-place; "
            "distill_seed7_v2 registra el entrenamiento CPU corregido. live_graph_pilot falló "
            "por límite de memoria. live_graph_pilot_v2 es parcial y se reserva íntegramente "
            "su presupuesto de 300 s. Ninguno se presenta como piloto completo.",
            "## Pendiente",
            (
                "La reproyección CPU ya se evaluó en todo FCWD. Queda validarla en DEV32 "
                "y medir el runtime GPU optimizado en secuencias externas completas; "
                if stream_complete
                else "Validar reproyección y runtime optimizado "
                "sobre secuencias externas completas; "
            )
            + "examinar MiDc, FR y colas por secuencia; ampliar repetición temporal y memoria "
            "pico. La selección final requiere datos independientes, tres semillas para "
            "modelos entrenados y evaluación oficial test12. No hay puntuación privada local.",
            "La repetición adicional de latencia quedó pendiente al terminar la preparación "
            "de las 6.762 entradas test12 y comenzar su inferencia GPU congelada. "
            "No se ejecutan nuevos pilotos GPU en paralelo con esa campaña.",
            "## Reproducción",
            "`python -m operational.streaming_revision.benchmark --help` describe las rutas. "
            "Cada nuevo piloto conserva sus fuentes en source_archive, pesos vinculados por "
            "hash, consultas, políticas y ROWS.jsonl. La reserva GPU se registra antes de "
            "ejecutar, con watchdog y contabilidad de tiempo de pared conservadora.",
            "`python -m operational.streaming_revision.report --root <resultados> "
            "--document <informe.md>` regenera este informe. STAGES.csv contiene los costes "
            "medios y NUMERICAL_PARITY.csv las diferencias numéricas medidas.",
        ]
    )
    text = "\n\n".join(lines) + "\n"
    (root / "REPORT.md").write_text(text, encoding="utf-8")
    if document is not None:
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(text, encoding="utf-8")
    atomic_json(
        root / "REPORT_BINDINGS.json",
        {
            "source_sha256": digest(Path(__file__)),
            "inputs": bindings,
            "external_sha256": digest(root / "external/METRICS.csv"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--document", type=Path)
    args = parser.parse_args()
    run(args.root, args.document)
