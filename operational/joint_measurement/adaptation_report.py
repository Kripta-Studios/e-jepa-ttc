"""Render the complete A1 evidence without presuming a favourable outcome."""

from __future__ import annotations

import json
from statistics import mean

from operational.range_transfer.common import read, sha, write

from .common import ROOT, RUN

DIRECTORY = RUN / "adaptive/A1_adaptation_depth"
FIT = "A1-LAST-BLOCK-S7"
CONTROL = "A1-FULL-S7"
BASELINE = "A5-P2-FROZEN"
BANDS = ("crucial", "small", "large", "negative")


def render(result: dict) -> str:
    """Describe recorded decisions and every cohort; reject incomplete report inputs."""
    names = (FIT, CONTROL)
    expected = {f"{role}/{name}" for role in ("P", "H") for name in (*names, BASELINE)}
    if result.get("status") != "A1_READY_FOR_INTERPRETATION":
        raise ValueError("Complete A1 evidence required")
    if set(result["cohorts"]) != expected:
        raise ValueError("All six A1 and reference P/H cohorts are required")
    selection = result["selection"]
    for key in ("endpoint_proofs", "exports"):
        if set(result[key]) != set(names):
            raise ValueError(f"Both A1 arms required in {key}")
        required_status = "PASS" if key == "endpoint_proofs" else "COMPLETE"
        if any(value["status"] != required_status for value in result[key].values()):
            raise ValueError(f"Incomplete A1 {key}")
    expected_proofs = {f"{role}/{name}" for role in ("P", "H") for name in names}
    if set(result["metric_proofs"]) != expected_proofs or any(
        value["status"] != "PASS" for value in result["metric_proofs"].values()
    ):
        raise ValueError("All four independent metric proofs required")
    expected_bootstrap = {
        f"{role}/{candidate}_vs_{reference}"
        for role in ("P", "H")
        for candidate, reference in ((FIT, CONTROL), (FIT, BASELINE), (CONTROL, BASELINE))
    }
    if set(result["bootstrap"]) != expected_bootstrap or any(
        value["status"] != "COMPLETE" for value in result["bootstrap"].values()
    ):
        raise ValueError("All six paired bootstrap results required")
    if result["prepared_cost"]["status"] != "COMPLETE" or set(
        result["prepared_cost"]["results"]
    ) != {FIT, CONTROL, BASELINE}:
        raise ValueError("All three prepared cost routes required")
    if result["known_scale_diagnostic"]["status"] != "COMPLETE_DIAGNOSTIC":
        raise ValueError("Complete known-scale diagnostic required")
    if set(selection["decisions"]) != set(names):
        raise ValueError("Both A1 admission decisions required")
    if result.get("system_promotion") or result.get("official_comparison"):
        raise ValueError("A1 P/H evidence cannot establish system or official promotion")
    if any(
        name not in names or not selection["decisions"][name]["eligible"]
        for name in selection["ranked"]
    ):
        raise ValueError("Ranked candidates must have recorded development admission")
    expected_status = "DEVELOPMENT_CANDIDATE" if selection["ranked"] else "NO_CANDIDATE_A1"
    if selection["status"] != expected_status:
        raise ValueError("Selection status and ranked candidates disagree")

    lines = [
        "# A1: profundidad de adaptación del encoder",
        "",
        f"Estado: **{selection['status']}**. No demuestra promoción del sistema ni SOTA.",
        "",
        "LAST-BLOCK y FULL parten del mismo endpoint B1 de correspondencia, seed7. "
        "Cada brazo añade 12.000 updates TTC en P. LAST-BLOCK actualiza el último "
        "bloque del encoder y el predictor; FULL actualiza el encoder completo y "
        "el mismo predictor. Se conserva objetivo, sampler, batch y presupuesto. "
        "Son 24.000 updates científicos nuevos; los 14.000 históricos del padre "
        "se reutilizan y no se cuentan otra vez. Igualar updates no iguala FLOPs.",
        "",
        "Ambos endpoints se congelaron antes de evaluar H. La comparación tiene "
        "una sola semilla, un padre compartido y H de desarrollo; no mide "
        "variabilidad entre inicializaciones independientes ni confirma una "
        "causa única. El padre ya recibió supervisión TTC en su etapa C1.",
        "",
        "## Poblaciones completas",
        "",
        "| Población/modelo | n | Agregado | MiDc | MiDs | MiDl | MiDn | "
        "MAE s | Signo | Cobertura |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for role in ("P", "H"):
        for name in (BASELINE, FIT, CONTROL):
            metric = result["cohorts"][f"{role}/{name}"]
            values = (
                [metric["overall_MiD"]]
                + [metric["bins"][band]["group_macro_mid"] for band in BANDS]
                + [metric["diagnostics"]["mae_s"]]
            )
            lines.append(
                f"| {role}/{name} | {metric['num_rows']} | "
                + " | ".join(f"{v:.3f}" for v in values)
                + f" | {100 * metric['diagnostics']['sign_error_rate']:.3f}%"
                + f" | {metric['complete_prediction_and_mid_coverage']} |"
            )
    lines += [
        "",
        "## Señal del mecanismo y admisión de desarrollo",
        "",
        "Las razones siguientes son candidato/referencia; valores menores que 1 "
        "indican menor error. La señal de LAST-BLOCK frente a FULL y la admisión "
        "de cada brazo frente a A5 son preguntas distintas. Una banda favorable "
        "no elimina las protecciones de las otras bandas, signo o cobertura.",
        "",
        "| Contraste H | Agregado | Crítico | Pequeño | Largo | Negativo | Elegible |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    comparisons = {f"{FIT} / {CONTROL}": selection["paired_mechanism"]}
    comparisons.update({f"{name} / {BASELINE}": selection["decisions"][name] for name in names})
    for label, decision in comparisons.items():
        ratios = [decision["overall_ratio"]] + [decision["ratios"][band] for band in BANDS]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{v:.6f}" for v in ratios)
            + f" | {decision['eligible']} |"
        )
    lines += ["", "Decisiones completas, incluidas las protecciones que fallan:", ""]
    for label, decision in comparisons.items():
        lines += [f"### {label}", "", "```json", json.dumps(decision, indent=2), "```", ""]
    lines += [
        f"Orden de admisión registrado: `{selection['ranked']}`.",
        "",
        "La admisión incorpora el coste preparado. Ninguna decisión de esta tabla "
        "es una promoción del pipeline completo: ésta requiere las comparaciones "
        "registradas frente a E_CTX, coste E2E comparable y decisión posterior. "
        "No se ha evaluado V en este informe.",
        "",
        "## Intervalos por grupos",
        "",
        "2.000 remuestreos emparejados de grupos de adquisición. Cada diferencia "
        "es candidato menos referencia, según los nombres del contraste; no es "
        "un bootstrap de ventanas independientes ni un intervalo entre semillas.",
        "",
        "| Contraste | Métrica | IC95 si todos los remuestreos están definidos |",
        "|---|---|---|",
    ]
    for label, comparison in sorted(result["bootstrap"].items()):
        for metric, interval in sorted(comparison["intervals"].items()):
            lines.append(f"| {label} | {metric} | {interval['ci95_if_all_defined']} |")
    lines += [
        "",
        "## Correspondencia conocida",
        "",
        "Las mismas 24 consultas P y transformaciones de textura conocidas se "
        "usan para padre y endpoints. No son etiquetas de movimiento real ni "
        "de TTC. La media excluye únicamente valores matemáticamente indefinidos "
        "y muestra su denominador; el soporte de caja es sólo diagnóstico.",
        "",
        "| Modelo | Soporte | Tipo | Casos definidos | Exactitud no nula media |",
        "|---|---|---|---:|---:|",
    ]
    observations = result["known_scale_diagnostic"]["measurements"]
    for name in sorted({row["model"] for row in observations}):
        for support in ("interior", "weak_bbox"):
            for kind in ("warped", "permuted"):
                values = [
                    row["supports"][support][kind]["strict_nonzero_accuracy"]
                    for row in observations
                    if row["model"] == name
                ]
                defined = [v for v in values if v is not None]
                value = f"{mean(defined):.6f}" if defined else "indefinida"
                lines.append(f"| {name} | {support} | {kind} | {len(defined)} | {value} |")
    lines += [
        "",
        "## Coste y trazabilidad",
        "",
        "| Ruta | Mediana ms | p95 ms |",
        "|---|---:|---:|",
    ]
    for name, timing in sorted(result["prepared_cost"]["results"].items()):
        lines.append(
            f"| {name} | {1000 * timing['p50_seconds']:.3f} | {1000 * timing['p95_seconds']:.3f} |"
        )
    lines += [
        "",
        "Coste preparado: 128 consultas P emparejadas, 20 warmups, FP32 y batch 1. "
        "Empieza con tensores residentes en GPU; excluye preparación sensorial "
        "y transferencias. No es E2E. La aceleración del entrenamiento mediante "
        "grafos CUDA es otra medición, descrita en A1_LOSS_GRAPH_20261016.md.",
        "",
        "Los recibos de endpoints verifican los estados de pesos, Adam, scheduler "
        "y muestreo; la política congelada se contrasta con el padre. Las tablas "
        "P/H tienen recálculo independiente y métricas por grupo. Se exportan "
        "ambos modelos, incluidos los rechazados, sobre cinco ejemplos P. "
        "Los exports empiezan en tensores P2 y no incluyen el sensor crudo.",
        "",
        "Fuentes: `adaptive/A1_adaptation_depth/evidence/COMPLETE.json`, "
        "`predictions/{P,H}/{fit}/GROUP_METRICS.json`, `bootstrap/`, `exports/` "
        "y los recibos durables de `fits/`. La contabilidad acumulada y el coste "
        "de entrenamiento, con su alcance de scopes, están en CURRENT_CONCLUSION.md.",
        "",
    ]
    return "\n".join(lines)


def build() -> dict:
    """Generate only after complete evidence exists, retaining source hashes."""
    source = DIRECTORY / "evidence/COMPLETE.json"
    result = read(source)
    evaluation = DIRECTORY / "evaluation/REGISTRATION.json"
    if result["evaluation_registration_sha256"] != sha(evaluation):
        raise ValueError("A1 evaluation registration changed")
    content = render(result)
    content += f"\nRegistro científico SHA-256: `{sha(DIRECTORY / 'REGISTRATION.json')}`.\n"
    content += f"\nEvidencia completa SHA-256: `{sha(source)}`.\n"
    path = ROOT / "docs/joint_measurement_20261012/A1_RESULTS.md"
    path.write_text(content, encoding="utf-8")
    receipt = {"source_sha256": sha(source), "report_sha256": sha(path), "path": str(path)}
    write(DIRECTORY / "evidence/REPORT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(build())
