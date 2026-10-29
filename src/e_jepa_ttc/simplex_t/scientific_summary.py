"""Render verified canonical development comparisons without selecting a new winner."""

from __future__ import annotations

import math

from .registry import registered_graph


def render_scientific_summary(coverage: dict, accounting: dict) -> str:
    """Render one report section from already reverified graph and work evidence.

    This pure renderer does not establish source authority. The configured
    postprocessor verifies all inputs before and after writing its outputs.
    Uncertainty, all controls and detailed diagnostics remain in their complete
    analysis exports; this section cannot replace the final delivery report.
    """
    flags = coverage["resolved_availability"]
    graph = registered_graph(**flags)
    count, updates = len(graph), sum(spec.updates for spec in graph)
    if (
        coverage.get("status") != "SCIENTIFIC_GRAPH_COVERAGE_VERIFIED_NOT_FINAL_DELIVERY"
        or accounting.get("status")
        != "COMPLETE_REQUIRED_FITS_AND_WORK_ACCOUNTING_VERIFIED_NOT_FINAL_DELIVERY"
        or coverage["freeze_sha256"] != accounting["freeze_sha256"]
        or coverage["fits_completed"] != count
        or accounting["scientific_fits_completed"] != count
        or coverage["scientific_updates_completed"] != updates
        or accounting["scientific_saved_updates"] != updates
        or coverage.get("holdout_opened") is not False
        or accounting.get("campaign_complete") is not False
    ):
        raise ValueError("summary requires consistent verified graph and work accounting")
    primary = "D1" if flags["d1"] else "D0"
    families = {"TPR", "LATENT"} if flags["latent"] else {"TPR"}
    if set(coverage["decisions"]) != families:
        raise ValueError("summary omits or adds a canonical family")
    lines = [
        "# SIMPLEX-T: resumen científico de desarrollo",
        "",
        "Evaluación OLD reutilizada: 8.192 consultas de nueve secuencias. "
        "No es confirmación independiente y no se ha abierto ningún holdout.",
        "",
        "El contexto es retrospectivo con ROI actual; no acredita seguimiento del objeto. "
        "sequence_id tampoco acredita adquisiciones independientes.",
        "",
        "## Comparaciones canónicas, semilla 7",
        "",
        "Delta = candidato menos referencia; un valor negativo indica menor pérdida MID. "
        "Los controles no sustituyen retrospectivamente al candidato canónico.",
        "",
        "| Candidato | MID candidato | MID H1 | Delta H1 | MID RISK17 | Delta RISK17 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def number(value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("finite measured scalar required in scientific summary")
        return float(value)

    for family in sorted(families):
        decision = coverage["decisions"][family]
        name = f"{family}-{primary}-H8-C160"
        if (
            decision["canonical_candidate"] != name
            or decision["h1_control"] != f"{family}-{primary}-H1-C160"
            or decision["seed"] != 7
        ):
            raise ValueError("summary changes the canonical comparison")
        current, risk = decision["versus_h1"], decision["versus_risk17"]
        values = [
            number(current[key]) for key in ("candidate_score", "reference_score", "point_delta")
        ]
        values += [number(risk[key]) for key in ("reference_score", "point_delta")]
        if number(risk["candidate_score"]) != values[0] or not (
            math.isclose(values[0] - values[1], values[2], rel_tol=1e-10, abs_tol=1e-8)
            and math.isclose(values[0] - values[3], values[4], rel_tol=1e-10, abs_tol=1e-8)
        ):
            raise ValueError("summary point estimates disagree across paired comparisons")
        lines.append(f"| {name} | " + " | ".join(format(value, ".10g") for value in values) + " |")
    technical = accounting["technical"]
    lower, upper = (accounting[f"recorded_total_updates_{bound}"] for bound in ("lower", "upper"))
    if not updates <= lower <= upper <= 250000:
        raise ValueError("summary work bounds differ from campaign limits")
    lines += [
        "",
        "## Trabajo y evidencia",
        "",
        f"Fits científicos terminados: {count}; updates científicos guardados: {updates}.",
        f"Updates técnicos registrados: {technical['recorded_executed_updates']}. "
        f"Total registrado: [{lower}, {upper}]; el extremo superior incluye posibles pérdidas, "
        "no ejecución observada adicional.",
        f"De los updates técnicos, {technical['historical_noninstrumented_updates']} "
        "tienen evidencia histórica no instrumentada; esa limitación se conserva.",
        "",
        "## Análisis completos que acompañan este resumen",
        "",
        "[Cobertura, disponibilidad y puertas prácticas](SCIENTIFIC_GRAPH_COVERAGE.json). "
        "Las puertas no requieren p < 0,05; T4 depende de disponibilidad técnica independiente.",
        "",
        "[T2: todos los brazos, controles, interacciones, incertidumbre y diagnósticos]"
        "(analyses/T2/T2_ANALYSIS.json). Los intervalos proceden del remuestreo agrupado "
        "documentado; este resumen de estimaciones puntuales no los sustituye.",
    ]
    for stage in sorted(coverage["phase_fit_counts"]):
        if stage != "T2":
            lines.append(f"[Análisis {stage}](analyses/{stage}/FOLLOWUP_ANALYSIS.json).")
    for family, flag in (("TPR", "replicate_scalar"), ("LATENT", "replicate_latent")):
        if flags[flag]:
            lines.append(
                f"[Tres semillas {family}](three_seed/{family}/THREE_SEED_ANALYSIS.json). "
                "Se agregan pérdidas, nunca TTC firmados."
            )
    lines += [
        "",
        "Los exports incluyen escape-hull gain/harm, cold-start, disponibilidad y edad del "
        "contexto. Los diagnósticos temporales son de endpoints muestreados, no de trayectorias "
        "completas ni de retardos de transición acreditados.",
        "",
        "[Contabilidad](CAMPAIGN_ACCOUNTING.json) y "
        "[interfaz futura](TEMPORAL_CANDIDATE_FREEZE.json). "
        "La interfaz no autoriza abrir confirmación.",
        "",
        "Este archivo es una sección del informe: no certifica por sí solo el empaquetado, "
        "la evidencia de recursos ni la entrega final de T6.",
        "",
    ]
    return "\n".join(lines)
