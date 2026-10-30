"""Regenerate the seed-7 comparison and decision report from verified JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.evidence / "comparison.json").read_text())
    diagnostics = json.loads((args.evidence / "diagnostics.json").read_text())
    rare = json.loads((args.evidence / "rare_full.json").read_text())
    curves = json.loads((args.evidence / "monitor_curves.json").read_text())
    coverage = json.loads(args.coverage.read_text())
    consumed = json.loads((args.evidence / "consumed_train_parity.json").read_text())
    if not consumed["same_ordered_train_tokens_every_update"]:
        raise ValueError("Training exposure parity did not pass")
    runs = comparison["runs"]
    final = {name: runs[name]["full"]["8192"]["metrics"] for name in ("r0", "r2")}
    delta = final["r2"]["AP"] - final["r0"]["AP"]
    if not comparison["same_full_population_and_protocol"]:
        raise ValueError("Comparison population gate did not pass")
    if coverage["manifest_file_sha256"] != runs["r2"]["config"]["data"]["expected_manifest_sha256"]:
        raise ValueError("Training coverage belongs to a different manifest")
    for name in ("r0", "r2"):
        for category in ("bicycle", "other_vehicle"):
            if (
                rare[name]["class_agnostic_pairs_2m"][category]["eligible"]
                != final[name]["per_class"][category]["eligible_ground_truth_count"]
            ):
                raise ValueError("Rare diagnostic omitted eligible annotations")
    lines = [
        "# R0 frente a R2, semilla 7 — cierre y siguiente intervención",
        "",
        "Fecha: 2026-10-31. Informe regenerado desde JSON verificados.",
        "",
        f"R2 completa 8.192 updates y 32.768 exposiciones. En selección completa, "
        f"la AP pasa de {final['r0']['AP']:.8f} a {final['r2']['AP']:.8f}: "
        f"+{delta:.8f} absoluto, +{100 * delta / final['r0']['AP']:.2f} % relativo.",
        "",
        "Se conserva R2 como baseline de trabajo. Las semillas 13 y 23 de R0/R2 quedan "
        "aplazadas; no queda ninguna ejecución automática pendiente. La siguiente prueba "
        "propuesta es R2 con muestreo limitado por identidad para clases raras, aislado "
        "de cambios de arquitectura, pérdidas y augmentation fotométrica.",
        "",
        "## Comparación con presupuesto fijo",
        "",
        "Ambos modelos usan D4096, batch 4, FP32, R50, ocho clases, 50 queries, "
        "11 grupos y la semilla 7. Se comparan checkpoints finales al mismo update. "
        "R0 usa LR constante; R2 combina warmup 5 %, cosine hasta 0,05× y LR del "
        "backbone 0,1×. Por tanto, la mejora no se puede atribuir exclusivamente al "
        "LR diferencial. R0 fue reanudado a 4.096 con linaje verificado; R2 comenzó frío.",
        "",
        f"Los logs confirman los mismos tokens en el mismo orden en los "
        f"{consumed['updates']:,} updates: {consumed['exposures_per_run']:,} "
        f"exposiciones por run y {consumed['unique_frames']:,} frames únicos. "
        "El recibo `consumed_train_parity.json` fija el hash de ese calendario.",
        "",
        "| Update | AP R0 completa | AP R2 completa | Diferencia |",
        "|---:|---:|---:|---:|",
    ]
    for update in ("4096", "8192"):
        a, b = (runs[name]["full"][update]["metrics"]["AP"] for name in ("r0", "r2"))
        lines.append(f"| {update} | {a:.8f} | {b:.8f} | {b - a:+.8f} |")
    population = final["r2"]["population"]
    lines += [
        "",
        f"Población idéntica: {final['r2']['sample_count']:,} frames, "
        f"{population['raw_ground_truth_count']:,} anotaciones raw, "
        f"{population['eligible_ground_truth_count']:,} evaluables y "
        f"{population['ignored_ground_truth_count']:,} ignoradas por profundidad. "
        "Se verificaron los hashes del contenido de cada GT, el protocolo y los checkpoints.",
        "",
        "| Clase | GT evaluable completo | AP R0 | AP R2 | Diferencia |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, row in final["r2"]["per_class"].items():
        a = final["r0"]["per_class"][category]["AP101_mean_thresholds"]
        b = row["AP101_mean_thresholds"]
        lines.append(
            f"| {category} | {row['eligible_ground_truth_count']} | "
            f"{a:.8f} | {b:.8f} | {b - a:+.8f} |"
        )
    lines += ["", "| Umbral XY | Recall sin exigir clase R0 | R2 |", "|---:|---:|---:|"]
    for threshold in ("0.5", "1.0", "2.0", "4.0"):
        a, b = (
            final[name]["class_agnostic_recall"]["thresholds"][threshold]["recall"]
            for name in ("r0", "r2")
        )
        lines.append(f"| {threshold} m | {a:.6f} | {b:.6f} |")
    lines += [
        "",
        "R2 mejora AP en las siete clases con AP no nula, pero no recupera "
        "other_vehicle. El recall a 4 m disminuye; la ganancia de AP no implica "
        "mejora de todas las propiedades de las propuestas.",
        "",
        "## Panel monitor y selección de checkpoint",
        "",
        "| Update | AP R0, 1.024 frames | AP R2, 1.024 frames |",
        "|---:|---:|---:|",
    ]
    for a, b in zip(curves["r0"], curves["r2"], strict=True):
        if a["update"] != b["update"]:
            raise ValueError("Monitor update grids differ")
        lines.append(f"| {a['update']} | {a['AP']:.8f} | {b['AP']:.8f} |")
    lines += [
        "",
        "R0 seleccionó update 6.144 y R2 update 8.192 por AP del panel. "
        "La tabla completa anterior compara ambos finales a 8.192; no es una "
        "comparación de los dos checkpoints seleccionados sobre selección completa.",
        "",
        "El panel contiene solo 26 bicicletas. Su AP final cae en R2, mientras "
        "que la AP sobre las 687 anotaciones de bicicleta de selección completa "
        "aumenta. La impresión inicial de colapso global queda descartada. No se "
        "debe elegir una intervención exclusivamente por la curva rara de este panel.",
        "",
        "## Diagnóstico de todas las anotaciones raras",
        "",
        "Correspondencia greedy, una a una, sin exigir clase, a distancia XY < 2 m. "
        "Se incluyen todos los objetos y predicciones de cada frame para conservar "
        "la competencia por asignaciones; después se agregan las dos clases raras. "
        "Una propuesta cercana no demuestra identidad correcta.",
        "",
        "| Run | Clase | GT | Propuestas asignadas a <2 m | Clase correcta | Confusiones |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name in ("r0", "r2"):
        for category, row in rare[name]["class_agnostic_pairs_2m"].items():
            confusion = (
                ", ".join(f"{k}: {v}" for k, v in row["predicted_classes"].items() if k != category)
                or "—"
            )
            lines.append(
                f"| {name} | {category} | {row['eligible']} | {row['matched_agnostic_2m']} | "
                f"{row['predicted_classes'].get(category, 0)} | {confusion} |"
            )
    lines += [
        "",
        "## Oracles diagnósticos: solo panel de 1.024 frames",
        "",
        "| Transformación GT | AP R0 | AP R2 |",
        "|---|---:|---:|",
    ]
    for kind in ("class", "center", "geometry", "ranking"):
        a, b = (diagnostics[name]["oracles"][kind]["AP"] for name in ("r0", "r2"))
        lines.append(f"| {kind} | {a:.8f} | {b:.8f} |")
    lines += [
        "",
        "Son intervenciones con GT sobre correspondencias limitadas a 2 m, "
        "con rematching posterior; no son rendimiento alcanzable ni cotas superiores "
        "globales. El oracle de ranking puede empeorar una clase y no demuestra qué "
        "ganaría un calibrador. El oracle de dimensiones/yaw no cambia la métrica "
        "si no altera la admisión geométrica: AP aquí se basa en distancia de centros. "
        "La reevaluación normal del panel tuvo paridad exacta con los resultados guardados.",
        "",
        "## Decisión: R2 + muestreo raro limitado por identidad",
        "",
        "La evidencia favorece conservar la receta R2 y estudiar la escasa supervisión "
        "de las clases raras. Esto es una hipótesis de intervención, no una demostración "
        "causal de que el desbalance explique todos los errores.",
        "",
        "| Clase train | Anotaciones entrenables en D4096 | Pares (secuencia, track_id) |",
        "|---|---:|---:|",
    ]
    for category in ("car", "bicycle", "other_vehicle", "pedestrian", "tricycle", "bus"):
        lines.append(
            f"| {category} | "
            f"{coverage['train']['trainable_annotations_legacy_profile'][category]} | "
            f"{coverage['train']['trainable_identity_count'][category]} |"
        )
    lines += [
        "",
        "Prueba propuesta `R2-rare-v1`: mismo D4096, 8.192 updates, batch 4, semilla 7, "
        "receta R2 y evaluación natural sin modificaciones. Único factor: sampler "
        "ponderado usando exclusivamente etiquetas de train, empezando por bicicleta "
        "y other_vehicle. Construir un calendario determinista con exposición por "
        "frame y por identidad rara limitada a 4× la correspondiente al calendario "
        "uniforme; registrar exposiciones efectivas, frames únicos e identidades. "
        "El límite se aplica al calendario materializado, no solo a pesos que luego "
        "podrían producir repeticiones excesivas. Mantener el presupuesto total y "
        "documentar qué exposiciones comunes se desplazan. Esto adelanta el factor de "
        "muestreo de R4 sin mezclarlo con augmentation R3.",
        "",
        "Antes de lanzarlo: comprobar que el calendario cumple los límites, se "
        "reanuda exactamente y conserva el protocolo, split y geometría. Para la "
        "decisión de screening, comparar el final a 8.192 con este R2 sobre los "
        "23.958 frames: promover solo si mejora AP macro y el promedio AP de las "
        "dos clases objetivo, y ninguna otra clase cae más de 0,01 AP absoluto. "
        "Este umbral es una regla operacional propuesta, no significación estadística. "
        "Informar todas las clases y los fallos aunque no pase. Las semillas adicionales "
        "se reservan para confirmar una intervención prometedora.",
        "",
        "Si el sampler no aporta, conservar R2 y pasar a augmentation fotométrica "
        "R3 como factor aislado. Si other_vehicle sigue sin soporte eficaz, ampliar "
        "diversidad de datos antes de atribuir el fallo a arquitectura: repetir seis "
        "identidades anotadas no crea seis adquisiciones independientes ni nuevos objetos. "
        "La siguiente intervención queda determinada; "
        "este cierre no ha lanzado otro entrenamiento.",
        "",
        "## Alcance y reproducibilidad",
        "",
        "Resultado exploratorio de una semilla, sobre desarrollo ya expuesto. "
        "No es puntuación oficial eAP, SOTA, test independiente ni evidencia nueva "
        "de precisión TTC. No se han usado test ni calibración para esta decisión.",
        "",
        "Evidencia: [comparison.json](evidence/r0_r2_seed7_20261031/comparison.json), "
        "[diagnostics.json](evidence/r0_r2_seed7_20261031/diagnostics.json), "
        "[rare_full.json](evidence/r0_r2_seed7_20261031/rare_full.json), "
        "[curvas](evidence/r0_r2_seed7_20261031/monitor_curves.json). "
        "Incluyen hashes de checkpoints, evaluaciones y fuentes. El script ejecutado "
        "para la verificación inicial se conserva como `analysis_executed.py`; "
        "`scripts/compare_detection_r0_r2.py` añade el diagnóstico raro completo y "
        "una anotación de tipo sin cambiar las reglas de verificación.",
        "",
        "Regenerar este informe desde la raíz del worktree:",
        "",
        "```text",
        "python scripts/report_detection_r0_r2.py "
        "--evidence docs/detection_sota_20261029/evidence/r0_r2_seed7_20261031 "
        "--coverage docs/detection_sota_20261029/evidence/"
        "r0_r2_seed7_20261031/d4096_coverage.json "
        "--output docs/detection_sota_20261029/R0_R2_SEED7_DECISION.md",
        "```",
        "",
        "Comprobaciones: verificador congelado de summaries/checkpoints y cuatro "
        "evaluaciones completas; igualdad de contenido GT entre runs; paridad exacta "
        "de métricas al reevaluar los dos paneles; ocho tests de diagnósticos; Ruff y "
        "Pyright. Tras retirar el controlador original no se capturó el código de "
        "salida del hijo: el cierre se acredita por artefactos completos y verificados, "
        "sin inventar un exit code.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
