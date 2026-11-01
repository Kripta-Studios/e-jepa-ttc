"""Publish available-data findings, keeping local sensitivity distinct from official score."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_research import read, sources, verify_complete


def report(
    continuation: Path, backend: Path, native: Path, latency: Path, followups: Path, output: Path
) -> None:
    """Render the fixed conclusions and all primary quantities from authenticated results."""
    folders = {
        "continuation": continuation,
        "backend": backend,
        "native": native,
        "latency": latency,
        "followups": followups,
    }
    for folder in folders.values():
        verify_complete(folder)
    population = read(continuation / "population_evaluation/REPORT.json")
    sensitivity = read(native / "paired_replay/RESULTS.json")
    runtime = read(latency / "RESULTS.json")
    policies = read(backend / "RESULTS.json")
    if (
        population["population_size"] != 2608
        or not population["routes"]["F0"]["eligible_for_eap_validation"]
        or sensitivity["queries"] != 5805
        or not sensitivity["local_original_exact"]
        or not runtime["all_predictions_exact_to_local_replay"]
    ):
        raise ValueError("Incomplete scientific population or local identity")
    documents = {
        "POPULATION.md": continuation / "population_report/README.md",
        "POPULATION.json": continuation / "population_evaluation/REPORT.json",
        "NATIVE_SENSITIVITY.json": native / "paired_replay/RESULTS.json",
        "NATIVE_PHYSICAL_PREDICTIONS.json": native / "paired_replay/PREDICTIONS.json",
        "NATIVE_ORIGINAL_PARITY.json": native / "original_replay/RESULTS.json",
        "BACKEND_DIAGNOSIS.json": backend / "RESULTS.json",
        "LATENCY.json": latency / "RESULTS.json",
        "LATENCY_MEASUREMENTS.json": latency / "MEASUREMENTS.json",
        "REFERENCE_COMPARISONS.json": followups / "REFERENCE_COMPARISONS.json",
    }
    capture_freeze = read(native / "capture/FREEZE.json")
    public = Path(capture_freeze["input_paths"]["public_inputs"])
    if sha256(public) != capture_freeze["inputs"]["public_inputs"]:
        raise ValueError("Public native aliases changed")
    inputs = {
        **sources(),
        "source:scripts/report_release_research_campaign.py": Path(__file__),
        **{name + ":complete": path / "COMPLETE.json" for name, path in folders.items()},
        **documents,
        "native_public_inputs": public,
    }
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "release-research-campaign-report-v1",
            "scope": "available exposed data; no independent confirmation or official promotion",
            "optimizer_updates": 0,
            "official_release_modified": False,
        },
    )
    try:
        for name, path in documents.items():
            shutil.copyfile(path, output / name)
        physical = {
            r["token"]: r for r in read(documents["NATIVE_PHYSICAL_PREDICTIONS.json"])["records"]
        }
        aliases = read(public)["physical_aliases"]
        if set(aliases.values()) != set(physical):
            raise ValueError("Logical aliases differ from the complete physical population")
        write_json(
            output / "NATIVE_ALIASED_PREDICTIONS.json",
            {
                "scope": "local diagnostic, no native targets or official promotion",
                "records": [
                    {
                        "token": token,
                        "physical": identity,
                        "original": physical[identity]["original"],
                        "canonical": physical[identity]["canonical"],
                    }
                    for token, identity in aliases.items()
                ],
            },
        )
        route = population["routes"]["F0"]
        lines = [
            "# Resultado de la investigación con los datos disponibles",
            "",
            "La corrección de cadencia supera los criterios fijados de desarrollo en 2608",
            "consultas. La comparación nativa local y la latencia están ejecutadas. Esto no",
            "demuestra conservación del score oficial: la réplica histórica del original",
            "falla su criterio intermedio y no se han abierto targets privados.",
            "",
            "El usuario confirma que no existen adquisiciones reservadas ni calibración",
            "fotométrica adicional. Las conclusiones mantienen su exposición de desarrollo.",
            "",
            "## Cadencia: ampliación completa",
            "",
            "| Dominio | Pares/población | MAE original (s) | MAE canónico (s) | "
            "Cambio relativo MAE |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, domain in route["domains"].items():
            old, new = domain["original"]["mae_seconds"], domain["canonical"]["mae_seconds"]
            lines.append(
                f"| {name} | {domain['paired']}/{domain['population']} | {old:.9f} | "
                f"{new:.9f} | {100 * (new / old - 1):.4f}% |"
            )
        interval = route["road_bootstrap"]["mae_seconds"]["ci95"]
        lines += [
            "",
            f"Admisión F0: `{route['eligible_for_eap_validation']}`. "
            f"CI95 del delta MAE de carretera: `{interval}` s.",
            "Slider es una sola adquisición, reutilizada del screen; no constituye una",
            "confirmación nueva independiente. La ampliación añade 2459 consultas de carretera.",
            "La mejora media de carretera es pequeña y su intervalo incluye cero.",
            "Cobertura, error relativo, signos y todas las grabaciones: "
            "[informe completo](POPULATION.md).",
            "",
            "## Conservación histórica y comparación local",
            "",
            "La ruta histórica original se conserva cerrada por fallo de paridad. Las",
            "cuatro políticas numéricas se probaron exclusivamente en el original:",
            "",
            "| Política | Consultas históricas que pasan |",
            "|---|---:|",
        ]
        for name, policy in policies["policies"].items():
            values = list(policy["queries"].values())
            lines.append(
                f"| {name} | {sum(v['historical_parity'] for v in values)}/{len(values)} |"
            )
        lines += [
            "",
            "El ensayo posterior responde a una pregunta distinta: cuánto cambia el",
            "candidato respecto al original ejecutado en este mismo equipo. Primero exigió",
            "repetibilidad exacta de todos los PAIR/PHASE17, contextos y cabeza originales.",
            "",
            f"- Consultas físicas: {sensitivity['queries']}; emparejadas: {sensitivity['paired']}.",
            f"- Predicciones idénticas: {sensitivity['exact_predictions']}.",
            f"- Nuevas ausencias: {sensitivity['newly_missing']}; "
            f"cambios de signo: {sensitivity['sign_changes']}.",
            f"- Máximo cambio absoluto: {sensitivity['maximum_absolute_change_seconds']:.9g} s.",
            f"- Percentiles 50/95/99 del cambio absoluto: "
            f"`{sensitivity['absolute_change_p50_p95_p99_seconds']}` s.",
            f"- Contexto de eventos idéntico: `{sensitivity['event_context_exact']}`.",
            "- Original local frente a histórico: "
            f"{sensitivity['historical_original_failed_queries']} consultas fallan algún campo; "
            f"{sensitivity['historical_original_ttc_failed_queries']} "
            "fallan la tolerancia TTC.",
            f"- Aliases lógicos resueltos: {len(aliases)}; no son observaciones independientes.",
            "",
            "Una diferencia pequeña no acredita conservación de Overall/Crucial/Negative.",
            "El release oficial permanece intacto. "
            "[Datos de sensibilidad](NATIVE_SENSITIVITY.json).",
            "",
            "## Latencia en el equipo local",
            "",
            "Primera pasada sin hooks de medida, desde preparación/lectura hasta salida host,",
            "con sincronización CUDA. Pesos calientes y cachés independientes por brazo.",
            "La selección utiliza los primeros diez ejemplos físicos por secuencia; no",
            "representa toda la distribución de despliegue. No equivale al servidor histórico.",
            "",
            "| Brazo | Consultas | p50 (ms) | p95 (ms) | p99 (ms) |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, value in runtime["service"].items():
            p50, p95, p99 = value["p50_p95_p99_ms"]
            lines.append(f"| {arm} | {value['queries']} | {p50:.3f} | {p95:.3f} | {p99:.3f} |")
        lines += [
            "",
            f"Cociente p95 candidato/original: {runtime['candidate_over_original_p95']:.5f}. "
            "Dentro del presupuesto descriptivo +10%: "
            f"`{runtime['within_descriptive_10pct_budget']}`.",
            "Memoria pico CUDA asignada: "
            f"{runtime['cuda_peak_allocated_bytes'] / 1024**2:.2f} MiB; "
            f"pico RAM del proceso: {runtime['process_peak_rss_bytes'] / 1024**2:.2f} MiB.",
            "La carga de pesos, importación, admisión de medios y adquisición física previa",
            "no forman parte del tiempo predict. Las colas son simulaciones de servidor único;",
            "no prueban servicio sostenido a 50 Hz. "
            "[Mediciones completas](LATENCY_MEASUREMENTS.json).",
            "",
            "La segunda pasada instrumentada separa costes y añade sincronizaciones.",
            "Sus medianas no deben sumarse ni confundirse con la primera pasada:",
            "",
            "| Brazo | Preparación/caché (ms) | Productores (ms) | Contextos/fusión/host (ms) |",
            "|---|---:|---:|---:|",
        ]
        measured = read(latency / "MEASUREMENTS.json")
        for arm in ("original", "canonical"):
            rows = [r for r in measured if r["instrumented"] and r["arm"] == arm]
            medians = [
                np.median([r[k] for r in rows])
                for k in ("preparation_and_cache_ms", "producers_ms", "context_fusion_host_ms")
            ]
            lines.append(f"| {arm} | {medians[0]:.3f} | {medians[1]:.3f} | {medians[2]:.3f} |")
        lines += [
            "",
            "## Qué queda de las demás propuestas",
            "",
            "Los diez contrastes previos están en "
            "[el informe de mecanismos](../followup_results_v2/RESULTS.md).",
            "O0 no supera los controles; la lectura robusta mantiene poca cobertura. Los",
            "filtros del reloj tampoco superan al contexto tras corregir la cadencia.",
            "S1 encuentra información de movimiento en RGB observado, pero su recuperación",
            "de q empeora TTC; no basta mejorar el residuo sensorial para entrenar ese auxiliar.",
            "La medición de superficies disponible tampoco mejora TTC.",
            "",
            "Por ello no se han entrenado O2, D4, S2/S3, V2/V3 ni JEPA condicional. Sus",
            "bloques y controles implementados no se presentan como modelos entrenados.",
            "Los resultados negativos delimitan estas implementaciones y esta población;",
            "no demuestran que toda variante posible sea inútil.",
            "",
            "No quedan extracciones de esta ejecución declaradas completas sin recibo. La",
            "confirmación independiente y la conservación de precisión oficial siguen sin",
            "demostrarse. La variante de cadencia se conserva como candidato de desarrollo.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(
            output / "DECISION.json",
            {
                "cadence_development_gate": route["eligible_for_eap_validation"],
                "native_local_sensitivity_complete": True,
                "latency_complete": True,
                "independent_confirmation": False,
                "official_score_conservation_demonstrated": False,
                "official_promotion_admitted": False,
                "learned_conditional_families_trained": False,
                "release_modified": False,
            },
        )
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("output/release_research_20261031")
    parser.add_argument("--continuation", type=Path, default=root / "continuation_v1")
    parser.add_argument("--backend", type=Path, default=root / "native_backend_diagnosis_v1")
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument(
        "--followups",
        type=Path,
        default=Path("docs/release_research_implementation_20261031/followup_results_v2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.continuation, args.backend, args.native, args.latency, args.followups, args.output)


if __name__ == "__main__":
    main()
