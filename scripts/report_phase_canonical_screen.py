"""Regenerate the fixed canonical screen report and publish small bound evidence."""

import argparse
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_campaign.data import read
from scripts.build_phase_native_context_augmentations import verify_artifacts


def number(value: float | None) -> str:
    """Render undefined metrics explicitly."""
    return "no disponible" if value is None else f"{value:.6f}"


def main() -> None:
    """Write tables exclusively from sealed inference and evaluation JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--compact-audit", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research_log_20261031_canonical_screen149")
    )
    args = parser.parse_args()
    complete = read(args.screen / "COMPLETE.json")
    report = read(args.evaluation / "REPORT.json")
    verify_artifacts(args.screen, complete["artifacts"])
    if report["screen_complete_sha256"] != file_sha(args.screen / "COMPLETE.json"):
        raise ValueError("Evaluation belongs to another screen")
    if report["scored_sha256"] != file_sha(args.evaluation / "SCORED.npz"):
        raise ValueError("Scored arrays changed")
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (args.screen / "FREEZE.json", "freeze.json"),
        (args.screen / "COMPLETE.json", "complete.json"),
        (args.screen / "PREDICTIONS.json", "predictions.json"),
        (args.evaluation / "REPORT.json", "report.json"),
    ):
        (evidence / name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    compact = None
    if args.compact_audit:
        compact = read(args.compact_audit)
        if (
            compact["status"] != "PASS"
            or compact["screen_complete_sha256"] != report["screen_complete_sha256"]
        ):
            raise ValueError("Compact parity audit differs")
        (evidence / "compact_parity.json").write_bytes(
            args.compact_audit.read_bytes().replace(b"\r\n", b"\n")
        )
    lines = [
        "# PHASE-TTC: screen fijo de PAIR canónico, 149 consultas",
        "",
        "Desarrollo con exposición histórica: 72 consultas de carretera y 77 de Slider-500. "
        "No es evaluación oficial, confirmación independiente ni evidencia de SOTA.",
        "",
        f"La extracción y los replays tardaron **{complete['elapsed_seconds']:.3f} s**. "
        "Las 149 consultas reprodujeron exactamente los arrays originales disponibles de "
        "productores, contextos y cabeza antes de aplicar la corrección. Se conservaron "
        "los lotes, los relojes, los normalizadores y los pesos; no hubo entrenamiento.",
        "",
        "Se capturó PAIR133 y se reconstruyó raw17 completo, incluidos sus desacuerdos, "
        "antes de recalcular contextos y cabeza. El router es el adaptado de FINAL_MODEL, "
        "con sus mismas entradas de geometría/calidad ligadas al hash de la población. "
        "F0 y router se evalúan por separado. Predicciones y ausencias se sellaron antes "
        "de que otro proceso abriera targets.",
        "",
        "## Precisión y cobertura",
        "",
        "MAE y error relativo son medias macro por grabación en el conjunto emparejado. "
        "Se muestran todos los denominadores; una pérdida de cobertura invalida el gate "
        "aunque mejoren esas métricas.",
        "",
        "| Ruta | Dominio | Disponibles original → canónico | Población | "
        "MAE original → canónico (s) | Error relativo original → canónico (%) | No empeora |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for route, data in report["routes"].items():
        for domain, metric in data["domains"].items():
            old, new = metric["original"], metric["canonical"]
            lines.append(
                f"| {route} | {domain} | {metric['original_available']} → "
                f"{metric['canonical_available']} | {metric['population']} | "
                f"{number(old['mae_seconds'])} → {number(new['mae_seconds'])} | "
                f"{number(old['relative_error_pct'])} → {number(new['relative_error_pct'])} | "
                f"{metric['nonworsening']} |"
            )
    lines += [
        "",
        "## Rechazos y comprobaciones",
        "",
    ]
    for name, data in report["producer_counts"].items():
        accepted = data["accepted_observations"]
        lines.append(
            f"- {name}: {accepted['event']} observaciones de eventos y "
            f"{accepted['rgb']} RGB aceptadas; {data['rejected_calls']} lotes rechazados."
        )
    for route, data in report["routes"].items():
        for domain, metric in data["domains"].items():
            lines.append(
                f"- {route}/{domain}: {metric['newly_missing']} nuevas ausencias; "
                f"{metric['sign_changes']} cambios de signo."
            )
    if compact is not None:
        lines += [
            "",
            "En las dos consultas comunes, el replay compacto reproduce exactamente "
            "los raw17/expert3 y los contextos/cabeza canónicos del piloto anterior, "
            "que sí ejecutaba los productores completos. La auditoría compara "
            "artefactos sellados; no repite los pilotos ni abre targets.",
        ]
    lines += [
        "",
        "## Criterio fijado y resultado",
        "",
        "Antes de abrir targets se exigió conservar cobertura, no empeorar ninguno "
        "de los cuatro criterios (MAE/error relativo × carretera/Slider) y mejorar "
        "al menos uno, por cada ruta. No se cambia d0 ni se elige otra corrección "
        "a partir de estos resultados.",
        "",
    ]
    for route, data in report["routes"].items():
        decision = (
            "justifica evaluar la población completa"
            if data["continue_full_population"]
            else "no supera el screen; no promover ni ampliar esta ruta como candidato aprobado"
        )
        lines.append(f"- {route}: {decision}.")
    lines += [
        "",
        "## Resultados por grabación",
        "",
        "| Ruta | Grabación | Pares/población | MAE original → canónico (s) | "
        "Error relativo original → canónico (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for route, data in report["routes"].items():
        for domain in data["domains"].values():
            for group, metric in domain["by_recording"].items():
                old, new = metric["original"], metric["canonical"]
                lines.append(
                    f"| {route} | {group} | {metric['paired']}/{metric['population']} | "
                    f"{number(old['mae_seconds'])} → {number(new['mae_seconds'])} | "
                    f"{number(old['relative_error_pct'])} → {number(new['relative_error_pct'])} |"
                )
    lines += [
        "",
        "## Límites y reproducción",
        "",
        "La consulta Slider-500:76 conserva su ausencia nativa por fin de cobertura "
        "de eventos. La disponibilidad del router puede diferir porque dispone de "
        "geometría. No se ha modificado ni sustituido el entregable.",
        "",
        "El piloto anterior conserva la regresión del contexto de eventos de carretera "
        "al ralentizar el tiempo. La auditoría eAP de observaciones actuales no equivale "
        "a conservación de historias, contextos y cabeza completos. La discrepancia "
        "oficial de Slider-750 sigue abierta; no se ha enviado correo ni otra entrega.",
        "",
        f"Inferencia local sellada: `{args.screen}`. Evaluación: `{args.evaluation}`. "
        "Los NPZ y pesos siguen fuera de Git. Los JSON pequeños publicados normalizan "
        "CRLF a LF; los hashes de recibos describen los bytes locales originales.",
        "",
        "```powershell",
        ".venv-transfer/Scripts/python.exe scripts/report_phase_canonical_screen.py "
        f"--screen {args.screen} --evaluation {args.evaluation}"
        + (f" --compact-audit {args.compact_audit}" if args.compact_audit else ""),
        "```",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
