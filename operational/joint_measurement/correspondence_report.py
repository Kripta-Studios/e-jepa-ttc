"""Regenerate the complete B1 paired report without choosing a seed or dropping failures."""

from __future__ import annotations

import numpy as np

from operational.range_transfer.common import read, sha, write

from .common import ROOT
from .correspondence_evaluate import CONTROL, DIRECTORY, FIT
from .correspondence_train import REGISTRATION


def build() -> dict:
    """Render registered outcomes, every band, stage proofs and the known-texture diagnostic."""
    source = DIRECTORY / "evidence/COMPLETE.json"
    result = read(source)
    if result["status"] != "B1_READY_FOR_INTERPRETATION":
        raise ValueError("Complete B1 evidence required")
    selection = result["selection"]
    lines = [
        "# B1: correspondencias conocidas y TTC posterior",
        "",
        f"Estado: **{selection['status']}**. No implica promoción del sistema ni SOTA.",
        "",
        "Ambos brazos parten del mismo C1 seed7 entrenado sólo en P. En la primera "
        "etapa se entrena únicamente el encoder durante2.000 updates: correspondencia "
        "de texturas P con transformaciones conocidas frente al objetivo TTC real. "
        "Después se entrenan todos los parámetros con TTC real durante10.000 updates "
        "en cada brazo, reiniciando Adam/scheduler/RNG en ambos al cambiar de etapa. "
        "Total nuevo:24.000 updates científicos. Los12.000 del padre se reutilizan; "
        "no se vuelven a contar. El control iguala updates, no FLOPs ni segundos GPU.",
        "",
        "El padre C1 ya recibió supervisión TTC; sólo la fase auxiliar de "
        "correspondencia prescinde de esas etiquetas. No es preentrenamiento "
        "independiente desde cero. No se sintetizan etiquetas TTC ni se simula la física "
        "de una cámara. "
        "El régimen de correspondencia utiliza el último tensor event, cuatro warps "
        "fijos, volúmenes nativos32/16 y ambos sentidos. Las cuatro etapas completas "
        "se congelaron antes de cualquier inferencia H de esta comparación.",
        "",
        "## Cohortes completas",
        "",
        "| Población/modelo | n | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Signo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, metric in result["cohorts"].items():
        values = [metric["overall_MiD"]] + [
            metric["bins"][band]["group_macro_mid"]
            for band in ("crucial", "small", "large", "negative")
        ]
        values.append(metric["diagnostics"]["mae_s"])
        lines.append(
            f"| {key} | {metric['num_rows']} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + f" | {100 * metric['diagnostics']['sign_error_rate']:.3f}% |"
        )
    paired = selection["paired_mechanism"]
    development = selection["decisions"][FIT]
    lines += [
        "",
        "## Decisiones separadas",
        "",
        f"Frente al control con igual número de updates, la fase auxiliar reduce "
        f"el agregado H un {100 * (1 - paired['overall_ratio']):.2f}% y MiDc un "
        f"{100 * (1 - paired['ratios']['crucial']):.2f}%. Las protecciones "
        "emparejadas de bandas, signo y cobertura se cumplen. Es una señal del "
        "régimen de aprendizaje de correspondencia, con una sola semilla y sobre "
        "H de desarrollo; no demuestra una causa única ni una mejora universal.",
        "",
        f"Frente a A5-P2, el agregado H se reduce un "
        f"{100 * (1 - development['overall_ratio']):.2f}% y MiDc un "
        f"{100 * (1 - development['ratios']['crucial']):.2f}%, pero MiDn aumenta "
        f"un {100 * (development['ratios']['negative'] - 1):.2f}%. "
        "El signo pasa la protección registrada; el rechazo procede de la banda "
        "negativa. No debe describirse como rechazo por signo. Se conserva "
        "NO_CANDIDATE_B1 y no se abre V para buscar un rescate.",
        "",
        "El control TTC ajusta mejor el agregado P y transfiere peor a H. "
        "La correspondencia conocida mejora durante la fase auxiliar y pierde "
        "parte de esa mejora durante el ajuste TTC posterior, aunque conserva "
        "ventaja sobre el control. Estas observaciones justifican estudiar la "
        "interacción entre representación y objetivo; no prueban por sí solas "
        "que congelar el encoder o cambiar la pérdida vaya a resolverla.",
        "",
        f"Señal de mecanismo H frente al control TTC: **{paired['eligible']}**. "
        f"Razón agregado: {paired['overall_ratio']:.6f}. "
        f"Protecciones: `{paired['checks']}`. Rutas de mejora: `{paired['signals']}`.",
        "",
        "La admisión de desarrollo conserva las protecciones originales frente a "
        "A5-P2 y añade el coste preparado. No es una relajación de los gates de "
        "promoción del sistema. Ambos brazos y sus resultados negativos se conservan.",
        "",
    ]
    for name, decision in selection["decisions"].items():
        lines.append(f"- {name}: `{decision}`.")
    lines += ["", "## Variabilidad por grupos", ""]
    for key, comparison in result["bootstrap"].items():
        interval = comparison["intervals"]["delta_overall"]["ci95_if_all_defined"]
        lines.append(f"- {key}: IC95 agregado CORRESPONDENCE − referencia: `{interval}`.")
    lines += [
        "",
        "Son2.000 resamples de grupos completos, no ventanas IID. Esta comparación "
        "tiene una semilla y no establece variabilidad entre inicializaciones.",
        "",
        "## Correspondencia conocida en P",
        "",
        "Mismas24 consultas y escalas0,9/1,1 que el diagnóstico previo. Exactitud "
        "estricta de desplazamiento no nulo, promediada sobre consulta/escala definida. "
        "El soporte de caja es sólo de diagnóstico. No mide TTC ni movimiento real.",
        "",
        "| Etapa/modelo | Soporte | Casos definidos | Exactitud | Null permutado |",
        "|---|---|---:|---:|---:|",
    ]
    observations = result["known_scale_diagnostic"]["measurements"]
    for name in sorted({row["model"] for row in observations}):
        for support in ("interior", "weak_bbox"):
            values = {}
            for kind in ("warped", "permuted"):
                raw = [
                    row["supports"][support][kind]["strict_nonzero_accuracy"]
                    for row in observations
                    if row["model"] == name
                ]
                valid = [value for value in raw if value is not None]
                values[kind] = (len(valid), float(np.mean(valid)) if valid else None)
            lines.append(
                f"| {name} | {support} | {values['warped'][0]} | "
                f"{values['warped'][1]} | {values['permuted'][1]} |"
            )
    lines += [
        "",
        "## Coste y verificación",
        "",
        "| Ruta | Mediana preparada ms | p95 ms |",
        "|---|---:|---:|",
    ]
    for name, timing in result["prepared_cost"]["results"].items():
        lines.append(
            f"| {name} | {1000 * timing['p50_seconds']:.3f} | {1000 * timing['p95_seconds']:.3f} |"
        )
    lines += [
        "",
        "El coste empieza con tensores preparados en GPU y termina en TTC firmado; "
        "excluye preparación sensorial, carga y transferencias. No es E2E. Se ejecutan "
        "las tres rutas en orden emparejado sobre128 consultas P fijas.",
        "",
        "La verificación independiente regenera768.000 IDs de muestra nuevos entre "
        "las cuatro etapas e inspecciona los pasos Adam, scheduler, cambio efectivo "
        "del encoder y conservación de las cabezas durante la primera etapa. "
        "Las cuatro tablas P/H se recalculan con NumPy y ambos endpoints se exportan "
        "con pruebas CPU sobre ejemplos P. No se infiere una reproducción sensorial "
        "completa a partir de estos exports.",
        "",
        f"Registro científico SHA-256: `{sha(REGISTRATION)}`.",
        f"Evidencia completa SHA-256: `{sha(source)}`.",
        "",
        f"Brazos retenidos: `{FIT}` y `{CONTROL}`.",
        "",
    ]
    path = ROOT / "docs/joint_measurement_20261012/B1_RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    receipt = {"source_sha256": sha(source), "report_sha256": sha(path), "path": str(path)}
    write(DIRECTORY / "evidence/REPORT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(build())
