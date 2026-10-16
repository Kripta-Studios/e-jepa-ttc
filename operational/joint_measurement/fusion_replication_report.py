"""Render all fixed fusion seeds, safeguards and uncertainty without selecting a winner."""

from __future__ import annotations

from operational.range_transfer.common import read, sha, write

from .common import ROOT, RUN

DIRECTORY = RUN / "replication/fusion_C1"


def row(name: str, metrics: dict) -> str:
    """Format the same native metrics for individual seeds and scalar means."""
    bands = metrics["bins"]
    values = [metrics["overall_MiD"]] + [
        bands[band]["group_macro_mid"] for band in ("crucial", "small", "large", "negative")
    ]
    values.append(metrics["diagnostics"]["mae_s"])
    return (
        "| "
        + name
        + " | "
        + " | ".join(f"{value:.3f}" for value in values)
        + f" | {100 * metrics['diagnostics']['sign_error_rate']:.3f}% |"
    )


def build() -> dict:
    """Bind report tables to complete predictions and their independent metric proofs."""
    result_path = DIRECTORY / "evaluation/RESULT.json"
    result = read(result_path)
    if result["status"] != "REPLICA_METRICS_READY_FOR_INTERPRETATION":
        raise ValueError("Complete registered replica evidence is required")
    evidence = {"replica_result": sha(result_path)}
    for name, metrics in result["metrics"].items():
        seed7_or_reference = name in {"R_CTX", "F_TRUE"} or name.endswith("-S7")
        base = RUN / "fusion" if seed7_or_reference else DIRECTORY
        folder = base / "predictions/V" / name
        proof = read(RUN / "verification/predictions/V" / name / "NUMPY_RECALC.json")
        if not (
            proof["status"] == "PASS"
            and proof["prediction_sha256"] == sha(folder / "predictions.parquet")
            and proof["metrics_sha256"] == sha(folder / "METRICS.json")
            and metrics == read(folder / "METRICS.json")
        ):
            raise ValueError(f"Report evidence changed for {name}")
        evidence[name] = proof["prediction_sha256"]
    lines = [
        "# Réplicas de la fusión C1 + R_CTX",
        "",
        "Se publican las tres semillas registradas (7, 13, 23) y sus controles ZERO. "
        "La réplica mide la variabilidad de la adaptación C1 y de las nuevas cabezas. "
        "A5 de inicialización y R_CTX histórico son compartidos: no son tres "
        "pretrainings independientes. No se elige una semilla favorable.",
        "",
        "Por semilla adicional: 12.000 updates P del encoder y lectura C1, extracción "
        "nueva de H y normalizador event propio, y 2.500 updates H por cada cabeza "
        "TRUE/ZERO. Los seis endpoints nuevos se congelan antes de cualquier "
        "inferencia V de las réplicas. R_CTX ya está supervisado en H; la nueva "
        "cabeza es ajuste secuencial, no OOF. V sigue siendo desarrollo expuesto.",
        "",
        "## Cohorte V completa",
        "",
        "| Modelo | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Error signo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (7, 13, 23):
        for mode in ("TRUE", "ZERO"):
            name = f"FUSION-{mode}-S{seed}"
            lines.append(row(name, result["metrics"][name]))
    for mode in ("TRUE", "ZERO"):
        lines.append(row(f"Media {mode}", result["means"][mode]))
    for name in ("R_CTX", "F_TRUE"):
        lines.append(row(name, result["metrics"][name]))
    lines += ["", "## Variabilidad y gates originales", ""]
    for mode, deviation in result["metric_sample_std"].items():
        lines.append(
            f"- {mode}: desviación entre semillas (ddof=1), agregado "
            f"{deviation['aggregate']:.4f}; MiDc {deviation['crucial']:.4f}; "
            f"MAE {deviation['mae']:.4f} s."
        )
    lines += [
        "",
        "Las medias son de métricas escalares, no un ensemble de TTC. La protección "
        "p95 usa la media de los p95 por semilla, tal como se registró antes de los "
        "resultados. Se conservan también los gates de cada semilla.",
        "",
        "| Contraste | Comprobaciones fallidas |",
        "|---|---|",
    ]
    for name, gate in result["gates"].items():
        failed = [key for key, passed in gate["checks"].items() if not passed]
        lines.append(f"| {name} | {', '.join(failed) or 'Ninguna en este contraste'} |")
    lines += ["", "## Incertidumbre por grupos", ""]
    for name, bootstrap in result["group_bootstrap_seed_mean_loss"].items():
        ci = bootstrap["intervals"]["overall"]["ci95_if_all_defined"]
        text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "no definido en todos los draws"
        lines.append(f"- TRUE − {name}, IC95 del cambio agregado: {text}.")
    lines += [
        "",
        "Son 2.000 resamples de grupos completos, después de promediar pérdidas "
        "MiD por grupo entre las tres semillas fijas. No se tratan ventanas "
        "correlacionadas como independientes. Este bootstrap no estima por sí "
        "solo la variabilidad de entrenamiento: ésta se presenta por separado.",
        "",
        "## Portabilidad y límites",
        "",
        "Las tablas H se conservan como diagnóstico del ajuste, no generalización. "
        "Los exports de cabeza incorporan sus normalizadores y fallback event "
        "dinámico, pero necesitan features de sus productores. Los exports C1 "
        "empiezan en los dos tensores sensoriales preparados. Ninguno de esos "
        "artefactos aislados equivale al pipeline RGB/event completo.",
        "",
        "Esta tabla no concede promoción ni confirma SOTA. La decisión del sistema "
        "requiere además el comparador V16 conservado, coste completo y las "
        "condiciones del protocolo. Una cifra interna no es una evaluación oficial.",
        "",
        "Fuentes: `replication/fusion_C1/evaluation/RESULT.json`, "
        "`predictions/{H,V}`, `ENDPOINTS.json`, `exports` y "
        "`verification/predictions`. Regenerar con "
        "`python -B -m operational.joint_measurement.fusion_replication_report`.",
    ]
    v16_path = RUN / "fusion/v16_compare/THREE_SEEDS.json"
    if v16_path.exists():
        comparison = read(v16_path)
        if comparison["status"] != "COMPLETE_DESCRIPTIVE_THREE_SEEDS":
            raise ValueError("Complete fixed-seed V16 comparison required")
        evidence["V16_three_seeds"] = sha(v16_path)
        failed = [
            name for name, passed in comparison["gate_seed_mean"]["checks"].items() if not passed
        ]
        interval = comparison["group_bootstrap_seed_mean_loss"]["intervals"]["overall"][
            "ci95_if_all_defined"
        ]
        ci_text = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "no definido"
        lines += [
            "",
            "## Comparador V16 conservado",
            "",
            "Comparación descriptiva posterior a la selección de la familia de réplica. "
            "No se reajustan cabezas, normalizadores ni gates con este resultado. "
            "Se conservan las tres semillas de V16 y las tres de V18.",
            "",
            "| Modelo | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Error signo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            row("Media V16 PHASE_ONLY_EVENT", comparison["reference_mean"]),
            row("Media V18 TRUE", comparison["candidate_mean"]),
            "",
            f"Gates fallidos de la media: {', '.join(failed) or 'ninguno'}. "
            f"IC95 del cambio agregado V18 − V16: {ci_text}. "
            "Una menor media agregada no demuestra por sí sola superioridad sobre V16; "
            "deben conservarse el intervalo, cada banda y los controles de promoción.",
        ]
    output = ROOT / "docs/joint_measurement_20261012/FUSION_REPLICATION_RESULTS.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {"sources": evidence, "report_sha256": sha(output), "automatic_promotion": False}
    write(DIRECTORY / "evaluation/REPORT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(build())
