"""Generate the retained raw-fusion numerical failure report from actual diagnostic receipts."""

from __future__ import annotations

from operational.range_transfer.common import read, sha, write

from .common import ROOT, RUN


def build() -> dict:
    """Separate input identity, batch arithmetic and unproven backend explanations."""
    source = RUN / "fusion/cost/parity_diagnostic_v1/GPU_RESULT.json"
    result = read(source)
    if result["status"] != "COMPLETE_DIAGNOSTIC" or result["cost_admitted"]:
        raise ValueError("Expected retained unqualified cost diagnostic")
    rows = result["rows"]
    exact_inputs = all(row["raw_pixels_exact"] and row["raw_delta_exact"] for row in rows)
    exact_batch = all(
        row["event_batch8_vs_stored"]["exact"] and row["event_batch8_ttc_vs_stored"]["exact"]
        for row in rows
    )
    lines = [
        "# Fusión: diagnóstico de la paridad sensorial",
        "",
        "**El coste sensorial completo sigue sin admisión.** Se conserva el fallo de "
        "paridad frente a la cohorte almacenada y no se amplían las tolerancias. "
        "Los pesos, predicciones históricas y resultados de los tres seeds no se modifican.",
        "",
        "El primer intento se detuvo en la admisión de fuentes históricas. La siguiente "
        "ruta cargó las definiciones nativas originales, pero encontró una mediana CUDA "
        "sin implementación determinista. Una política explícita y acotada permitió "
        "ejecutar esa operación conservando el warning. La prueba posterior falló "
        "en la primera consulta cronometrada: ZERO y F_TRUE no cumplen "
        "`rtol=1e-5, atol=1e-4 s`. Sus tiempos parciales no forman una medición completa.",
        "",
        "## Localización sobre cuatro consultas fijadas",
        "",
        "Se usan las primeras cuatro consultas del orden original de coste, sin selección "
        "por etiquetas ni cambio de la población. El diagnóstico no ejecuta optimizador.",
        "",
        f"Pixels P2 y separación temporal raw/preparados idénticos: **{exact_inputs}**. "
        f"C1 con el batch8 original reproduce features y TTC almacenados exactamente: "
        f"**{exact_batch}**. Por tanto, en estos casos el cache sensorial P2 conserva "
        "los datos y la ruta de cohorte puede reproducirse.",
        "",
        "| Índice V | Δ máx. features C1 batch1/cohorte | Δ máx. R_HISTORY | "
        "Δ máx. E_HISTORY | Δ anchor RGB, s | Δ F_TRUE raw/cache, s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        numbers = [
            row["event_batch1_vs_stored"]["maximum_absolute_difference"],
            row["R_HISTORY"]["features"]["maximum_absolute_difference"],
            row["E_HISTORY"]["features"]["maximum_absolute_difference"],
            row["raw_rgb_anchor_vs_stored"]["maximum_absolute_difference"],
            row["F_TRUE_cached_vs_raw_contexts"]["maximum_absolute_difference"],
        ]
        lines.append(f"| {row['index']} | " + " | ".join(f"{n:.9g}" for n in numbers) + " |")
    cpu = read(source.parent / "CPU_RESULT.json")
    maximum = max(row["batch1_vs_stored_ttc"]["maximum_absolute_difference"] for row in cpu["rows"])
    lines += [
        "",
        "Los relojes y máscaras históricas coinciden; las features y fases de los "
        "productores raw presentan diferencias. C1 también cambia ligeramente al "
        "pasar de batch8 a batch1. Estas son diferencias numéricas observadas, no una "
        "demostración de fuga de datos ni una licencia para aceptar el coste.",
        "",
        f"El control CPU del anchor RGB reproduce la cohorte con batch128. "
        f"Cambiar únicamente a batch1 introduce como máximo {maximum:.9g} s en esas "
        "cuatro consultas; por sí solo no explica las diferencias de la ruta raw.",
        "",
        "## Hipótesis de backend registrada",
        "",
        "El extractor histórico y su constructor fijan determinismo, pero no fijan "
        "`cudnn.allow_tf32`; el setup V18 lo desactiva explícitamente. Se registra "
        "un diagnóstico de ambos valores sólo en las convoluciones de los productores "
        "históricos, manteniendo matmul TF32 desactivado y restaurando la política "
        "al salir. No cambia las convoluciones C1 ni los pesos. El flag efectivo "
        "histórico no quedó registrado: esta es una hipótesis verificable, no una "
        "causa ya demostrada por inspección del código.",
    ]
    backend = RUN / "fusion/cost/backend_diagnostic_v1/RESULT.json"
    if backend.exists():
        probe = read(backend)
        lines += [
            "",
            "| Índice V | cuDNN TF32 productores | Δ R_HISTORY | Δ E_HISTORY | "
            "Δ F_TRUE raw/cache, s |",
            "|---:|---|---:|---:|---:|",
        ]
        for row in probe["measurements"]:
            values = [
                row["R_HISTORY"]["features"]["maximum_absolute_difference"],
                row["E_HISTORY"]["features"]["maximum_absolute_difference"],
                row["F_TRUE_raw_vs_cached_contexts"]["maximum_absolute_difference"],
            ]
            lines.append(
                f"| {row['index']} | {row['producer_cudnn_tf32']} | "
                + " | ".join(f"{v:.9g}" for v in values)
                + " |"
            )
        maximum_by_flag = {
            flag: max(
                row["F_TRUE_raw_vs_cached_contexts"]["maximum_absolute_difference"]
                for row in probe["measurements"]
                if row["producer_cudnn_tf32"] == flag
            )
            for flag in (False, True)
        }
        failed = [
            row["index"]
            for row in probe["measurements"]
            if row["producer_cudnn_tf32"]
            and abs(row["raw_ttc_s"] - row["cached_ttc_s"]) > 1e-4 + 1e-5 * abs(row["cached_ttc_s"])
        ]
        lines += [
            "",
            f"Máximo error F_TRUE raw/cache en estas consultas: "
            f"{maximum_by_flag[False]:.9g} s con TF32 desactivado y "
            f"{maximum_by_flag[True]:.9g} s al activarlo sólo en los productores. "
            f"Consultas que todavía fallan la tolerancia original: `{failed}`. "
            "El backend influye, pero este único cambio no resuelve toda la paridad. "
            "No se promueve la ruta ni se continúa una búsqueda de tolerancias.",
        ]
        lines += [
            "",
            f"Recibo del contraste: `{sha(backend)}`. El diagnóstico no admite "
            "automáticamente la ruta: una nueva medición necesita su propio "
            "registro y pasar la paridad original completa.",
        ]
    else:
        lines += ["", "Estado del contraste de backend: registrado, pendiente de GPU exclusiva."]
    lines += [
        "",
        f"Diagnóstico GPU: `{sha(source)}`.",
        f"Diagnóstico CPU: `{sha(source.parent / 'CPU_RESULT.json')}`.",
        "",
        "No se infiere precisión TTC de estos cuatro casos ni se utilizan "
        "para elegir un modelo. La inferencia preparada y los exports siguen "
        "teniendo un alcance distinto del coste sensorial completo.",
        "",
    ]
    destination = ROOT / "docs/joint_measurement_20261012/FUSION_COST_DIAGNOSTIC.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    receipt = {"diagnostic_sha256": sha(source), "report_sha256": sha(destination)}
    write(source.parent / "REPORT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(build())
