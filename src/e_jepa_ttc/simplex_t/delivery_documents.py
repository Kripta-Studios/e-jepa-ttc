"""Render required delivery documents from independently verified development results."""

from __future__ import annotations

from .scientific_summary import render_scientific_summary


def render_delivery_documents(
    coverage: dict, accounting: dict, *, analysis_commit: str
) -> tuple[str, dict]:
    """Compose reports, not authority: callers must verify sources and archive coverage.

    The decision preserves canonical candidates and practical gates. It does not
    choose a winner, authorize confirmation or certify that a ZIP was delivered.
    No fit or filesystem access occurs here.
    """
    if len(analysis_commit) != 40 or set(analysis_commit) - set("0123456789abcdef"):
        raise ValueError("full analysis commit required")
    summary = render_scientific_summary(coverage, accounting)
    # This renderer emits known relative links. The required report lives one
    # directory above the postprocessing publication in the transport archive.
    for prefix in (
        "SCIENTIFIC_GRAPH_COVERAGE.json",
        "analyses/",
        "three_seed/",
        "CAMPAIGN_ACCOUNTING.json",
        "TEMPORAL_CANDIDATE_FREEZE.json",
    ):
        summary = summary.replace(f"]({prefix}", f"](postprocessing/{prefix}")
    report = (
        "# Informe SIMPLEX-T de desarrollo\n\n"
        f"Commit de análisis: `{analysis_commit}`.\n\n"
        "Conservamos Stage70–76 y los expertos históricos. Evaluamos el contexto "
        "retrospectivo con ROI actual sobre OLD; no acreditamos seguimiento del objeto.\n\n"
        "Consulta el inventario y el recibo de entrega para comprobar la cobertura y "
        "el SHA256 del ZIP. Este texto no prueba que la transferencia haya terminado.\n\n"
        + summary
        + "\n## Recursos y siguiente decisión\n\n"
        "Incluimos los recibos de recursos declarados por intento. Sus muestras no "
        "miden picos continuos ni garantizan tiempos completos tras una terminación "
        "externa. No deducimos fits del número de intentos.\n\n"
        "Conservamos la interfaz futura para una comparación con autorización propia. "
        "No abrimos confirmación ni elegimos otro candidato a partir de controles.\n"
    )
    decision = {
        "schema": "simplex_t_next_decision_v1",
        "status": "DEVELOPMENT_GRAPH_VERIFIED_DELIVERY_VERIFICATION_REQUIRED",
        "analysis_commit": analysis_commit,
        "freeze_sha256": coverage["freeze_sha256"],
        "canonical_decisions": coverage["decisions"],
        "resolved_availability": coverage["resolved_availability"],
        "scientific_fits_completed": coverage["fits_completed"],
        "scientific_updates_completed": coverage["scientific_updates_completed"],
        "recorded_total_updates_lower": accounting["recorded_total_updates_lower"],
        "recorded_total_updates_upper": accounting["recorded_total_updates_upper"],
        "next_action": "RETAIN_FUTURE_INTERFACE_REQUEST_SEPARATE_COMPARISON_AUTHORIZATION",
        "future_interface": "postprocessing/TEMPORAL_CANDIDATE_FREEZE.json",
        "holdout_authorized": False,
        "transport_completion_inferred_from_rendering": False,
    }
    return report, decision
