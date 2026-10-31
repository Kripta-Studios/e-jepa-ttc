"""Regenerate the bounded Dfull compute investigation from measured JSON receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Compare only within each fixed-checkpoint probe; never infer quality from timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probes = {}
    for version in (3, 4, 5):
        path = args.evidence / f"probe_v{version}.json"
        value = json.loads(path.read_text())
        if value["status"] != "COMPLETE" or not value["training_resumed"]:
            raise ValueError(f"Incomplete diagnostic: {path}")
        probes[version] = value
    a, b, c = (probes[k]["variants"] for k in (3, 4, 5))

    def seconds(value: dict) -> float:
        return value["median_last6_seconds"]

    comparisons = [
        ("Un hilo CPU, entrada residente", a["baseline"], a["threads1"]),
        ("Comprobaciones finitas agrupadas", a["baseline"], a["fused_checks"]),
        ("TF32 y comprobaciones agrupadas", a["baseline"], a["tf32_fused"]),
        ("Un hilo CPU con preparación concurrente", b["cpu4_prefetch"], b["cpu1_prefetch"]),
        ("BF16 + pérdidas FP32, con preparación", b["cpu4_prefetch"], b["bf16_fp32loss_prefetch"]),
    ]
    lines = [
        "# Dfull: investigación de cómputo en la A40",
        "",
        "Fecha: 2026-10-31. La campaña principal continúa con su productor v2. "
        "Estas pruebas no modificaron sus pesos, configuración ni ledger.",
        "",
        "## Método y alcance",
        "",
        "Cada variante restaura pesos, optimizador y RNG desde el mismo checkpoint "
        "dentro de su prueba. Usa ocho actualizaciones sobre un lote fijo de frames "
        "reales; el tiempo es la mediana de las seis últimas. No acredita convergencia "
        "ni AP. El sampler de la campaña principal no se modifica. Las pausas de "
        "aislamiento conservan el proceso y su caché en memoria, con finally y un "
        "watchdog independiente que envían SIGCONT.",
        "",
        f"Pausa total medida: {sum(p['pause_seconds'] for p in probes.values()):.2f} s. "
        "Los intentos iniciales v1/v2 fallaron al adaptar la API del optimizador, "
        "antes de pausar el entrenamiento. Sus logs permanecen en el pod.",
        "",
        "v3 usa checkpoint 1.024; v4/v5 usan 2.048. No se comparan pérdidas entre "
        "checkpoints ni entre batches diferentes. La divergencia de trayectorias "
        "posteriores al primer paso también aparece entre repeticiones FP32 por "
        "el backward CUDA no determinista del productor.",
        "",
        "## Etapas",
        "",
        "Medianas de tres pasos v3 con sincronización entre etapas; instrumentar "
        "estas fronteras añade sobrecoste. Los tiempos no son porcentajes de un "
        "perfil sin instrumentación.",
        "",
        "| Etapa | ms |",
        "|---|---:|",
    ]
    for name, value in probes[3]["synchronized_stage_median_seconds"].items():
        lines.append(f"| {name} | {value * 1000:.2f} |")
    lines += [
        "",
        "La traza registra 7.290 lanzamientos CUDA y 668 llamadas de sincronización "
        "de stream en un paso instrumentado. Esos contadores no significan que todo "
        "ese trabajo pueda eliminarse. Hay convoluciones, GEMM, grid_sample/backward "
        "y muchos kernels elementales. Los rangos anidados del profiler se solapan: "
        "no deben sumarse como fracciones exclusivas del tiempo GPU.",
        "",
        "## Variantes medidas",
        "",
        "| Variante | Referencia ms/update | Candidato ms/update | Reducción temporal |",
        "|---|---:|---:|---:|",
    ]
    for name, reference, candidate in comparisons:
        before, after = seconds(reference), seconds(candidate)
        lines.append(
            f"| {name} | {1000 * before:.2f} | {1000 * after:.2f} | "
            f"{100 * (1 - after / before):.2f}% |"
        )
    low, high = sorted([seconds(c["batch4"]), seconds(c["batch4_repeat"])])
    batch8 = seconds(c["batch8"])
    lines += [
        "",
        f"Batch 8: {batch8 * 1000:.2f} ms/update, {8 / batch8:.2f} frames/s. "
        f"Los controles batch 4 dan {4 / high:.2f}–{4 / low:.2f} frames/s. "
        f"Para las mismas exposiciones, la reducción de tiempo de cómputo sería "
        f"{100 * (1 - batch8 / (2 * low)):.1f}–"
        f"{100 * (1 - batch8 / (2 * high)):.1f}%, antes de lectura, migración y evaluación.",
        "",
        "El campo first_outputs_exact de v5 compara cada variante consigo misma "
        "porque el harness reinicia la referencia al cambiar de batch. **No es "
        "evidencia de paridad batch 4/8 ni TF32/FP32.** Batch 8 cambia la frecuencia "
        "de los pasos AdamW y requiere una receta y calendario de exposiciones nuevos. "
        "No basta con cambiar batch_size conservando los 70.279 updates.",
        "",
        "## Decisiones",
        "",
        "- Caché compacta sin pérdida y precarga: ya activas. En el run real la espera "
        "de datos ronda 2 ms por lote. La preparación concurrente eleva el cómputo "
        "de aproximadamente 248 a 296 ms en v4; el control con un hilo no lo corrige. "
        "Después de llenar la caché desaparece la preparación de primeras visitas.",
        "- Checks agrupados / menos hilos: ganancia pequeña o nula; no justifican "
        "reiniciar y volver a llenar la caché de esta ejecución.",
        "- TF32: ganancia moderada, con diferencias de salidas. No equivale al productor "
        "FP32 histórico y no se ha demostrado igualdad de calidad.",
        "- BF16 directo: falla en cdist del matching. Convertir outputs a FP32 para "
        "pérdidas permite ejecutar, pero la ganancia con precarga es pequeña. "
        "La diferencia máxima observada de un elemento de output es 51,10, sin "
        "asignarle unidades físicas ni interpretarla como error de profundidad.",
        "- Batch 8: mejor candidato de throughput probado. Es una intervención de "
        "entrenamiento, aún sin evidencia de calidad ni calendario migrado; no se "
        "aplicó a la campaña principal.",
        "- CUDA Graphs completos: la ruta inspeccionada usa targets variables, "
        "matching SciPy en CPU y decisiones host. Necesita separar regiones "
        "capturables y comprobaciones; no es un interruptor de configuración. "
        "No se ha medido un speedup de CUDA Graphs ni de torch.compile.",
        "- Más entrenamientos en paralelo en la misma A40 no reduce el trabajo "
        "de esta trayectoria. No se han lanzado otras semillas.",
        "",
        "Las cuatro pasadas siguen siendo el presupuesto registrado, no una prueba "
        "de que sean necesarias u óptimas. Reducirlas sería una decisión de "
        "presupuesto/calidad distinta de acelerar las mismas actualizaciones.",
        "",
        "Documentación oficial consultada para la versión instalada: "
        "[CUDA y Graphs](https://docs.pytorch.org/docs/2.11/notes/cuda.html), "
        "[precisión matmul](https://docs.pytorch.org/docs/2.11/generated/"
        "torch.set_float32_matmul_precision.html), "
        "[profiler](https://docs.pytorch.org/docs/2.11/profiler.html).",
        "",
        "## Fuentes",
        "",
    ]
    for path in sorted(args.evidence.iterdir()):
        if path.is_file():
            lines.append(f"- `{path.name}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`")
    lines += [
        "",
        "Regenerar con `python scripts/report_dfull_compute_probe.py --evidence "
        "docs/detection_sota_20261029/evidence/dfull_compute_20261031 --output "
        "docs/detection_sota_20261029/DFULL_COMPUTE_INVESTIGATION_20261031.md`.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
