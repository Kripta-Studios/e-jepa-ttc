"""Publish the sealed 2608-query canonical PAIR result without changing its gate."""

import argparse
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from scripts.build_phase_native_context_augmentations import verify_artifacts


def number(value: float | None) -> str:
    """Keep missing metrics explicit rather than substituting zero."""
    return "no disponible" if value is None else f"{value:.9f}"


def verified_inputs(screen: Path, evaluation: Path) -> tuple[dict, dict]:
    """Reject incomplete, altered, or incorrectly scoped evidence before publishing."""
    complete = read(screen / "COMPLETE.json")
    freeze = read(screen / "FREEZE.json")
    if (
        complete["status"] != "COMPLETE"
        or complete["queries"] != 2608
        or not complete["baseline_exact"]
        or complete["targets_read"]
        or complete["optimizer_updates"] != 0
        or freeze["schema"] != "phase_canonical_population2608_v1"
        or complete["freeze_sha256"] != file_sha(screen / "FREEZE.json")
        or not {"FREEZE.json", "PREDICTIONS.json"} <= complete["artifacts"].keys()
    ):
        raise ValueError("Require the sealed target-blind 2608-query population")
    verify_artifacts(screen, complete["artifacts"])
    report = read(evaluation / "REPORT.json")
    if (
        report["population_size"] != 2608
        or report["primary_route"] != "F0"
        or report["router_prior_screen"] != "failed; diagnostic only"
        or report["screen_complete_sha256"] != file_sha(screen / "COMPLETE.json")
        or report["scored_sha256"] != file_sha(evaluation / "SCORED.npz")
        or report["routes"]["router"]["eligible_for_eap_validation"]
    ):
        raise ValueError("Evaluation scope or binding differs")
    for route in ("F0", "router"):
        domains = report["routes"][route]["domains"]
        if domains["road"]["population"] != 2531 or domains["slider"]["population"] != 77:
            raise ValueError("Domain population differs from the fixed plan")
    return complete, report


def render(complete: dict, report: dict, screen: Path, evaluation: Path) -> str:
    """Render all route decisions, denominators, and recording-level changes."""
    f0_passes = report["routes"]["F0"]["eligible_for_eap_validation"]
    decision = (
        "F0 supera los criterios fijados; procede comprobar eAP completo antes de promoverlo."
        if f0_passes
        else "F0 no supera los criterios fijados; no procede promover esta intervención."
    )
    lines = [
        "# PHASE-TTC: PAIR canónico en la población de 2.608 consultas",
        "",
        decision,
        "",
        "Desarrollo con exposición histórica: 2.531 consultas de carretera en 24 grabaciones "
        "y 77 de Slider-500. No es evaluación oficial ni confirmación independiente de SOTA. "
        "El router ya falló el screen previo y sus resultados aquí son solo diagnósticos.",
        "",
        f"Tiempo de ejecución: {complete['elapsed_seconds']:.3f} s. Se reutilizaron las 149 "
        "consultas selladas del screen y se extrajeron 2.459 nuevas. Cada consulta nueva "
        "reprodujo exactamente su ejecución original en PAIR, raw17, contextos y cabeza. "
        "El control es la ejecución original emparejada; las características nativas de "
        "la antigua POPULATION.npz no son el control de este experimento.",
        "",
        "Se conservaron pesos, normalizadores, relojes y lotes. Se reconstruyó raw17 "
        "completo a partir de PAIR133 y se recalcularon contextos/cabeza. No hubo "
        "actualizaciones del optimizador. Las predicciones y ausencias se sellaron "
        "antes de unir targets en otro proceso.",
        "",
        "## Precisión y cobertura",
        "",
        "Métricas macro por grabación en el conjunto emparejado, con salidas firmadas "
        "sin recorte y denominador relativo max(abs(target), 1e-6). Los criterios exigen "
        "conservar cobertura, no empeorar MAE ni error relativo en ninguno de los dos "
        "dominios y mejorar al menos uno. No se reajustan umbrales tras ver el resultado.",
        "",
        "| Ruta | Dominio | Original → canónico | Pares/targets/población | "
        "MAE original → canónico (s) | Error relativo original → canónico (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for route, data in report["routes"].items():
        for domain, metric in data["domains"].items():
            old, new = metric["original"], metric["canonical"]
            lines.append(
                f"| {route} | {domain} | {metric['original_available']} → "
                f"{metric['canonical_available']} | {metric['paired']}/{metric['targets']}/"
                f"{metric['population']} | {number(old['mae_seconds'])} → "
                f"{number(new['mae_seconds'])} | {number(old['relative_error_pct'])} → "
                f"{number(new['relative_error_pct'])} |"
            )
    lines += ["", "## Cambios por grabación y bootstrap", ""]
    lines.append(
        "Bootstrap pareado de grabaciones de carretera (10.000 remuestreos, semilla 7). "
        "Diferencia canónico − original: negativa indica menor error. Es descriptivo; "
        "no trata ventanas correlacionadas como independientes. Slider tiene una sola "
        "grabación y no recibe un intervalo entre grabaciones."
    )
    lines += ["", "| Ruta | Métrica | Cambio medio | IC 95% |", "|---|---|---:|---|"]
    for route, data in report["routes"].items():
        bootstrap = data["road_bootstrap"]
        if not bootstrap["available"]:
            lines.append(f"| {route} | No disponible | — | {bootstrap['reason']} |")
            continue
        for key in ("mae_seconds", "relative_error_pct"):
            metric = bootstrap[key]
            low, high = metric["ci95"]
            lines.append(
                f"| {route} | {key} | {number(metric['mean_change'])} | "
                f"[{number(low)}, {number(high)}] |"
            )
    lines += [
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
    lines += ["", "## Disponibilidad y decisiones", ""]
    for name, data in report["producer_counts"].items():
        accepted = data["accepted_observations"]
        lines.append(
            f"- {name}: {accepted['event']} observaciones de eventos, {accepted['rgb']} RGB; "
            f"{data['rejected_calls']} lotes rechazados."
        )
    for route, data in report["routes"].items():
        lines.append(
            f"- {route}: cumple criterios numéricos fijos = {data['passes_fixed_criteria']}."
        )
        for domain, metric in data["domains"].items():
            lines.append(
                f"- {route}/{domain}: {metric['newly_missing']} nuevas ausencias, "
                f"{metric['newly_available']} nuevas disponibilidades y "
                f"{metric['sign_changes']} cambios de signo."
            )
    lines += [
        "",
        "## Límites y reproducción",
        "",
        "El entregable sigue intacto. El resultado del router no revoca su fallo previo. "
        "La auditoría parcial eAP de consultas actuales no demuestra conservación de "
        "historias, contextos y cabeza; sus cachés OOF tampoco prueban preservación de "
        "despliegue. Se conserva la regresión temporal del contexto de eventos del "
        "piloto. Sigue pendiente la discrepancia oficial de Slider-750.",
        "",
        f"Inferencia: `{screen}`. Evaluación: `{evaluation}`. Se publican el informe "
        "JSON, el protocolo y un recibo compacto con hashes; los arrays, pesos y el "
        "listado completo de hashes por consulta permanecen localmente. Los JSON "
        "copiados normalizan CRLF a LF; los recibos conservan hashes de los bytes locales.",
        "",
        "```powershell",
        ".venv-transfer/Scripts/python.exe scripts/report_phase_canonical_population.py "
        f"--screen {screen} --evaluation {evaluation} --output DIRECTORIO_NUEVO",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write a fresh report only from verified completed evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh report directory; preserve existing reports")
    complete, report = verified_inputs(args.screen, args.evaluation)
    rendered = render(complete, report, args.screen, args.evaluation)
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True)
    for source, name in (
        (args.screen / "FREEZE.json", "freeze.json"),
        (args.evaluation / "REPORT.json", "report.json"),
    ):
        (evidence / name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    atomic_receipt(
        evidence / "verification.json",
        {
            "status": "VERIFIED",
            "complete_sha256": file_sha(args.screen / "COMPLETE.json"),
            "predictions_sha256": file_sha(args.screen / "PREDICTIONS.json"),
            "report_sha256": file_sha(args.evaluation / "REPORT.json"),
            "artifact_count_verified": len(complete["artifacts"]),
            "complete_summary": {k: v for k, v in complete.items() if k != "artifacts"},
            "generator_sha256": file_sha(Path(__file__)),
        },
    )
    (args.output / "README.md").write_text(rendered, encoding="utf-8", newline="\n")
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
