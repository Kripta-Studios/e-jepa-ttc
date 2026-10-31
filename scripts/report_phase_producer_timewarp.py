"""Regenerate a two-query temporal diagnostic from hash-bound executed evidence."""

import argparse
import json
import math
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read


def interval(values: list) -> str:
    """Display only measured finite ratios; undefined entries remain counted."""
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return f"sin cociente válido (0/{len(values)})"
    return f"{min(finite):.6g}–{max(finite):.6g} ({len(finite)}/{len(values)})"


def main() -> None:
    """Publish diagnostic ratios without converting consistency into accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("output/phase_timewarp/pilot_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research_log_20261031_producer_timewarp")
    )
    parser.add_argument("--failed-attempt", type=Path)
    args = parser.parse_args()
    receipt = read(args.pilot / "COMPLETE.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["queries"] != 2
        or not receipt["identity_exact"]
        or receipt["targets_read"]
        or receipt["optimizer_updates"] != 0
    ):
        raise ValueError("Require complete two-query identity-verified zero-update pilot")
    if file_sha(args.pilot / "FREEZE.json") != receipt["freeze_sha256"]:
        raise ValueError("Pilot freeze changed")
    root = args.pilot.resolve()
    for relative, digest in receipt["artifacts"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_sha(path) != digest:
            raise ValueError(f"Changed pilot evidence: {relative}")
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("COMPLETE", "FREEZE", "PREFLIGHT"):
        source = args.pilot / f"{name}.json"
        (evidence / f"{name.lower()}.json").write_bytes(
            source.read_bytes().replace(b"\r\n", b"\n")
        )
    verification = {
        "status": "PASS",
        "complete_sha256": file_sha(args.pilot / "COMPLETE.json"),
        "freeze_sha256": receipt["freeze_sha256"],
        "verified_artifacts": len(receipt["artifacts"]),
        "identity_arrays": {
            name: {
                "producer": query["compared_producer_arrays"],
                "context": query["compared_context_arrays"],
            }
            for name, query in receipt["reports"].items()
        },
        "accuracy_evaluated": False,
        "checkpoint_promoted": False,
    }
    (evidence / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    lines = [
        "# PHASE-TTC: diagnóstico temporal de productores y fusión",
        "",
        "Este piloto ejecuta los nueve componentes originales congelados sobre dos consultas "
        "de desarrollo fijadas antes de la prueba. No utiliza targets, no actualiza pesos y "
        "no evalúa la precisión del router adaptado que acompaña a la entrega actual.",
        "",
        "## Intervención y controles",
        "",
        "Se conservan las imágenes RGB. En eventos se mantienen los canales 0–10 de la "
        "representación y se transforma la tasa del canal 11 según la duración real de cada "
        "ventana. Todos los relojes se transforman alrededor de la consulta con factores "
        "0,5 y 2; la fase canónica sigue siendo de 100 ms. Se aplica el presupuesto causal "
        "de 650 ms sin incorporar observaciones nuevas. Esto transforma representaciones "
        "ya preparadas: no reproduce una nueva selección de ventanas de 100 ms sobre un "
        "stream crudo acelerado. El redondeo de eventos sobre límites semiabiertos puede "
        "cambiar su pertenencia; las pruebas sintéticas conservan ese contraejemplo.",
        "",
        "El control a=1 exige igualdad exacta de entradas, salidas de productores y "
        "contextos/cabeza, con los mismos lotes de la extracción original. Si cambia el "
        "conjunto retenido, se vuelve a ejecutar un control limpio con esas identidades "
        "y tamaños de lote. Los rechazos posteriores del productor se registran aparte.",
        "",
        f"Tiempo medido del piloto, incluida carga/extracción/replay: "
        f"{receipt['elapsed_seconds']:.3f} s. Actualizaciones de optimizador: **0**.",
        "",
        "## Resultados por consulta",
        "",
        "Un cociente a·T_transformado/T_control igual a 1 representa la respuesta temporal "
        "ideal para esta intervención. No expresa error respecto al TTC real. Los "
        "cocientes de contextos/cabeza solo aíslan la intervención sobre las mismas "
        "observaciones admitidas cuando la columna correspondiente lo confirma.",
        "",
        "| Consulta | Factor | Retenidas/admitidas | Mismas admitidas que control | "
        "TTC F0 (s) | Cocientes F0 / contexto RGB / contexto eventos |",
        "|---|---:|---:|---|---:|---|",
    ]
    for name, query in receipt["reports"].items():
        if not query["identity_exact"]:
            raise ValueError("Per-query identity gate missing")
        for factor in query["factors"].values():
            ratios = factor["native_ttc_ratio_against_matched_control"]
            ratio_text = " / ".join("indefinido" if v is None else f"{v:.6g}" for v in ratios)
            value = factor["frozen_F0_seconds"]
            text_value = "ausente" if value is None else f"{value:.6g}"
            lines.append(
                f"| {name} | {factor['speed']} | {len(factor['retained_physical_ids'])}/"
                f"{len(factor['accepted_producer_ids'])} | "
                f"{factor['head_comparison_isolating']} | {text_value} | {ratio_text} |"
            )
    lines += [
        "",
        "## Respuesta por bloque",
        "",
        "Los intervalos siguientes abarcan las observaciones con cocientes definidos. "
        "Se conserva el número de cocientes finitos sobre el total. Las salidas individuales, "
        "máscaras, soporte, saturación y rechazos permanecen en la evidencia JSON/NPZ.",
        "",
        "| Consulta | Factor | Bloque | q_transformado/(a·q_control) | "
        "a·T_transformado/T_control | Cociente por par corregido por dt real | "
        "Máximo cambio absoluto del log-ratio |",
        "|---|---:|---|---|---|---|---:|",
    ]
    for name, query in receipt["reports"].items():
        for factor in query["factors"].values():
            grouped: dict[str, list[dict]] = {}
            for stages in factor["stage_by_observation"].values():
                for stage, values in stages.items():
                    if isinstance(values, dict):
                        grouped.setdefault(stage, []).append(values)
            for stage, values in sorted(grouped.items()):
                inverse = [v for row in values for v in row.get("inverse_ratio", [])]
                ttc = [v for row in values for v in row.get("ttc_ratio", [])]
                measured = [
                    v for row in values for v in row.get("pair_inverse_ratio_measured_dt", [])
                ]
                changes = [
                    row["pair_log_ratio_max_abs_difference"]
                    for row in values if "pair_log_ratio_max_abs_difference" in row
                ]
                valid_changes = [v for v in changes if v is not None and math.isfinite(v)]
                change = (
                    f"{max(valid_changes):.6g} ({len(valid_changes)}/{len(changes)})"
                    if valid_changes else "no definido/no aplica"
                )
                lines.append(
                    f"| {name} | {factor['speed']} | {stage} | {interval(inverse)} | "
                    f"{interval(ttc)} | {interval(measured)} | {change} |"
                )
    lines += [
        "",
        "## Alcance de la evidencia",
        "",
        "El cociente corregido por dt es (q_transformado·dt_transformado)/"
        "(q_control·dt_control) por par. Utiliza los deltas ejecutados y separa su "
        "cuantización del efecto de cambiar la representación. Las salidas PAIR no "
        "disponen de ese diagnóstico analítico por par.",
        "",
        "Dos consultas no caracterizan una población ni demuestran el origen del error "
        "de Slider. La falta de equivariancia puede localizar una sensibilidad, pero "
        "corregirla solo se justifica como mejora tras una evaluación agrupada de precisión "
        "y cobertura. Una modificación de productores exige recalcular los estados afectados "
        "para comprobar eAP; las cachés del modelo previo no certifican su conservación.",
        "",
        "No se promueve un checkpoint ni se afirma SOTA. Este documento se regenera con "
        "`scripts/report_phase_producer_timewarp.py`; los tensores y pesos permanecen fuera "
        "de Git. Los hashes de los archivos publicados pueden diferir de los originales "
        "solo por normalización CRLF→LF; los recibos mantienen los hashes de ejecución.",
        "",
    ]
    if args.failed_attempt is not None:
        failure = read(args.failed_attempt / "FAILED.json")
        for name in ("FAILED", "FREEZE"):
            source = args.failed_attempt / f"{name}.json"
            (evidence / f"previous_attempt_{name.lower()}.json").write_bytes(
                source.read_bytes().replace(b"\r\n", b"\n")
            )
        lines += [
            "## Intento anterior conservado",
            "",
            f"El intento `{args.failed_attempt.name}` terminó con `{failure['error']}`. "
            "Se conservan su recibo de fallo y su configuración; no se presentan como "
            "una ejecución completa. El piloto posterior usa una carpeta nueva.",
            "",
        ]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Report regenerated: {args.output}")


if __name__ == "__main__":
    main()
