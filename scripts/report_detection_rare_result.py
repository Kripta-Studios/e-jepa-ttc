"""Regenerate the R2-rare closure and Dfull decision from verified evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Render measured comparison tables and the explicit next-stage decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text(encoding="utf-8"))
    data = json.loads(args.readiness.read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE" or result["calendar"]["status"] != "PASS":
        raise ValueError("Analysis or committed training ledger is incomplete")
    passed = result["screening_pass"]
    selected = "R2-rare" if passed else "R2"
    final = result["full"]["8192"]
    before, after = final["r2"], final["r2_rare"]
    replay = result["replayed_update_parity"]
    base, full = data["d4096"], data["full_train_readable"]
    lines = [
        "# R2-rare: cierre recuperado y decisión de escala",
        "",
        "Fecha: 2026-10-31. Tablas regeneradas desde evidencia JSON.",
        "",
        f"R2-rare termina 8.192 updates y 32.768 exposiciones sobre 4.096 frames únicos. "
        f"La AP completa final es {after['AP']:.8f}, frente a {before['AP']:.8f} de R2 "
        f"({after['AP'] - before['AP']:+.8f} absoluto). "
        f"El screening prerregistrado {'se supera' if passed else 'no se supera'}; "
        f"se conserva **{selected}** como receta de trabajo para la ampliación.",
        "",
        "## Comparación con presupuesto y población fijados",
        "",
        "Mismos D4096, R50, ocho clases, batch cuatro, FP32, 50 queries y 11 grupos. "
        "El cambio declarado es el calendario de muestreo raro. No es fine-tuning del "
        "R2 final: la intervención empezó desde la misma inicialización del screening.",
        "",
        "| Update | AP R2 completa | AP R2-rare completa | Diferencia |",
        "|---:|---:|---:|---:|",
    ]
    for update, row in result["full"].items():
        lines.append(
            f"| {update} | {row['r2']['AP']:.8f} | {row['r2_rare']['AP']:.8f} | "
            f"{row['AP_delta']:+.8f} |"
        )
    lines += [
        "",
        f"Selección completa: {after['sample_count']:,} frames. Se verificaron hashes "
        "de archivos y contenido de todos los GT; las poblaciones son idénticas. "
        "Esta selección está expuesta a desarrollo, no es test ni scorer oficial.",
        "",
        "| Clase | GT evaluable | AP R2 | AP R2-rare | Diferencia |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, row in after["per_class"].items():
        a = before["per_class"][category]["AP101_mean_thresholds"]
        b = row["AP101_mean_thresholds"]
        lines.append(
            f"| {category} | {row['eligible_ground_truth_count']} | {a:.8f} | "
            f"{b:.8f} | {b - a:+.8f} |"
        )
    lines += [
        "",
        "| Distancia XY | Recall sin exigir clase R2 | R2-rare |",
        "|---:|---:|---:|",
    ]
    for threshold in ("0.5", "1.0", "2.0", "4.0"):
        a = before["class_agnostic_recall"]["thresholds"][threshold]["recall"]
        b = after["class_agnostic_recall"]["thresholds"][threshold]["recall"]
        lines.append(f"| {threshold} m | {a:.6f} | {b:.6f} |")
    lines += [
        "",
        "| Clase rara, umbral 2 m | TP R2 | FN R2 | TP R2-rare | FN R2-rare |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in ("bicycle", "other_vehicle"):
        a = before["per_class"][category]["thresholds"]["2.0"]
        b = after["per_class"][category]["thresholds"]["2.0"]
        lines.append(f"| {category} | {a['TP']} | {a['FN']} | {b['TP']} | {b['FN']} |")
    lines += [
        "",
        "Los TP/FN usan el pool del protocolo y exigen clase correcta. No representan "
        "un umbral de score operacional calibrado. El recall agnóstico mide otro "
        "aspecto de las propuestas y no sustituye la AP.",
        "",
        "Criterios fijados antes del resultado:",
        "",
    ]
    labels = {
        "macro_improves": "Mejora la AP macro completa",
        "target_average_improves": "Mejora el promedio AP de bicicleta y other_vehicle",
        "no_other_class_drops_over_001": "Ninguna otra clase pierde más de 0,01 AP absoluto",
    }
    for key, value in result["acceptance_checks"].items():
        lines.append(f"- {labels[key]}: **{'PASS' if value else 'FAIL'}**.")
    lines += [
        "",
        "La regla es operacional, no un contraste estadístico. Una única trayectoria "
        "no demuestra una contribución causal reproducible ni liderazgo SOTA. "
        "Las semillas adicionales continúan aplazadas.",
        "",
        "## Curva de desarrollo",
        "",
        "| Update | AP R2, panel 1.024 | AP R2-rare, panel 1.024 |",
        "|---:|---:|---:|",
    ]
    curves = {
        name: {r["update"]: r["AP"] for r in rows}
        for name, rows in result["monitor"].items()
    }
    if set(curves["r2"]) != set(curves["r2_rare"]):
        raise ValueError("Monitor grids differ")
    for update in sorted(curves["r2"]):
        lines.append(
            f"| {update} | {curves['r2'][update]:.8f} | "
            f"{curves['r2_rare'][update]:.8f} |"
        )
    lines += [
        "",
        "La comparación principal usa checkpoints finales al mismo presupuesto. "
        "Los checkpoints seleccionados por el panel corto se conservan por separado.",
        "",
        "## Recuperación y límite de reproducibilidad",
        "",
        "El checkpoint recuperado estaba en update 3.840. Se repitieron los 256 updates "
        "registrados después de ese checkpoint y se continuó hasta 8.192. "
        "Se verificaron configuración, productor congelado, estado de optimizador, "
        "scheduler, sampler, RNG y calendario consumido. El ledger final cuenta una "
        "sola vez cada update comprometido; no suma la repetición a las exposiciones.",
        "",
        f"La repetición no es idéntica bit a bit: en el panel 4.096, AP "
        f"{replay['original_monitor_AP']:.8f} antes del corte y "
        f"{replay['resumed_monitor_AP']:.8f} tras reanudar. "
        "El primer loss reanudado coincide exactamente y la siguiente diferencia "
        "es pequeña, pero crece durante los 256 updates. Lotes y LR coinciden. "
        "El productor usa grid_sample y una prueba estricta en el pod confirma "
        "que su backward CUDA no dispone de implementación determinista. Esto "
        "identifica una fuente posible, sin demostrar que sea la causa exclusiva. "
        "Se documenta la trayectoria reanudada y no se mezclan sus resultados "
        "con los monitores no comprometidos anteriores al corte.",
        "",
        "## Decisión sobre Dfull",
        "",
        f"**Sí se justifica ampliar al train completo legible, partiendo de {selected}.** "
        "La justificación es la cobertura de datos que falta en D4096; no una promesa "
        "de que entrenar más resolverá todas las clases. No es necesario lanzar más "
        "semillas del screening antes de esa fase de datos. La ejecución Dfull "
        "se documenta por separado para no confundir su progreso con este cierre.",
        "",
        f"La auditoría local admite **{full['frames']:,} frames de {full['sequences']} "
        "secuencias train**, tras excluir tres medios defectuosos. Se mantienen "
        "separadas selección y calibración. Los tres defectuosos no tienen anotaciones "
        "y no eliminan identidades entrenables.",
        "",
        "| Clase | Anotaciones D4096 | Dfull legible | IDs D4096 | IDs Dfull |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, count in base["trainable_annotations"].items():
        lines.append(
            f"| {category} | {count} | {full['trainable_annotations'][category]} | "
            f"{base['trainable_sequence_track_identities'][category]} | "
            f"{full['trainable_sequence_track_identities'][category]} |"
        )
    lines += [
        "",
        "IDs significa (secuencia, track_id), no independencia física o de adquisición. "
        "Bicicleta y other_vehicle no ganan identidades; sus nuevas vistas y exposiciones "
        "no equivalen a nueva diversidad. El límite de soporte raro sigue vigente.",
        "",
        "El lanzamiento Dfull requiere manifest y configuración versionados con las "
        "exclusiones, perfil de memoria acotada y presupuesto de validación explícito. "
        "Sus imágenes preparadas FP32 ocuparían unos 193 GiB. La continuación usa "
        "una representación compacta reversible y precarga CPU, verificadas contra "
        "los tensores originales; la velocidad debe medirse en esa ejecución.",
        "",
        "## Alcance de entrega",
        "",
        "Se cierra este experimento RGB de detección. Dfull sería otra etapa evaluada, "
        "y no convierte automáticamente el detector en un modelo de producción. "
        "Siguen pendientes la validación final por dominio/clase, rendimiento operativo "
        "y evaluación integral de detección, asociación y PHASE. La independencia "
        "respecto a TRAIN40 y las comparaciones temporales/eventos del plan requieren "
        "su propia evidencia; este resultado no acredita nueva precisión TTC.",
        "",
        "Regeneración: `python scripts/report_detection_rare_result.py --analysis "
        "<analysis.json> --readiness <dfull_readiness.json> --output <informe.md>`.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
