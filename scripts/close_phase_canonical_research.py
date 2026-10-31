"""Close the three requested research checks without relabelling incomplete SOTA evidence."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_submission.runtime import load_submission_model
from scripts.audit_phase_eap_router_sensitivity import verified_parent
from scripts.build_phase_native_context_augmentations import verify_artifacts
from scripts.report_phase_canonical_population import number, verified_inputs


def main() -> None:
    """Generate a bounded closing decision from complete local checks and retained blockers."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "router", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--population", type=Path, default=Path("output/phase_timewarp/population2608_v1")
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("output/phase_timewarp/population2608_evaluation_v1"),
    )
    parser.add_argument("--screen", type=Path, default=Path("output/phase_timewarp/screen149_v1"))
    parser.add_argument(
        "--screen-evaluation",
        type=Path,
        default=Path("output/phase_timewarp/screen149_evaluation_v1"),
    )
    parser.add_argument(
        "--official-audit", type=Path, default=Path("output/phase_safe_fusion/evaluator_audit")
    )
    parser.add_argument(
        "--web-snapshot", type=Path, default=Path("output/phase_timewarp/closure_sources_20261101")
    )
    parser.add_argument("--bundle", type=Path, default=Path("output/phase_submission/FINAL_MODEL"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh closure directory")
    _, population = verified_inputs(args.population, args.evaluation)
    verified_parent(args.native, {"FREEZE.json", "DECISION.json"})
    verified_parent(
        args.native / "paired_replay", {"FREEZE.json", "RESULTS.json", "PREDICTIONS.json"}
    )
    verified_parent(args.router, {"FREEZE.json", "REPORT.json", "PREDICTIONS.json"})
    native = read(args.native / "paired_replay/RESULTS.json")
    router = read(args.router / "REPORT.json")
    router_freeze = read(args.router / "FREEZE.json")
    if (
        native["queries"] != 5805
        or router["physical"]["queries"] != 5805
        or router["logical_aliases"]["queries"] != 6762
        or router["accuracy_evaluated"]
        or router["promotion_admitted"]
        or router_freeze["parents"][str(args.native / "paired_replay/COMPLETE.json")]
        != file_sha(args.native / "paired_replay/COMPLETE.json")
    ):
        raise ValueError("Incomplete or mismatched native/router closure")
    screen = read(args.screen_evaluation / "REPORT.json")
    verified_parent(args.screen, {"FREEZE.json", "PREDICTIONS.json"})
    if (
        any(
            sum(row["population"] for row in route["domains"].values()) != 149
            for route in screen["routes"].values()
        )
        or screen["scored_sha256"] != file_sha(args.screen_evaluation / "SCORED.npz")
        or screen["screen_complete_sha256"] != file_sha(args.screen / "COMPLETE.json")
    ):
        raise ValueError("Screen evidence differs")
    audit_path = args.official_audit / "AUDIT.json"
    audit = read(audit_path)
    manifest = read(Path("docs/research_log_20261031_sign_safe/evidence_manifest.json"))
    if file_sha(audit_path) != manifest["receipts"]["evaluator_audit.json"]:
        raise ValueError("Evaluator audit differs from published evidence")
    verify_artifacts(Path.cwd(), audit["input_sha256"])
    verify_artifacts(args.official_audit, audit["output_sha256"])
    web = read(args.web_snapshot / "RECEIPT.json")
    for row in web["sources"]:
        if file_sha(args.web_snapshot / (row["name"] + ".html")) != row["sha256"]:
            raise ValueError("Primary-source snapshot changed")
    expected = read(args.population / "FREEZE.json")["inputs"]
    model_key = next(k for k in expected if k.replace("\\", "/").endswith("FINAL_MODEL/MODEL.json"))
    if file_sha(args.bundle / "MODEL.json") != expected[model_key]:
        raise ValueError("Delivered model identity changed since the population freeze")
    load_submission_model(args.bundle)
    mismatches = {
        k: v for k, v in audit["original_vs_official"].items() if not v["matches_pdf_rounding"]
    }
    if (
        screen["routes"]["router"]["continue_full_population"]
        or audit["official_equivalence"]
        or set(mismatches) != {"Slider-750"}
    ):
        raise ValueError("Evidence changed: reassess closure rather than reuse its old decision")
    sources = {
        "native": args.native / "paired_replay/RESULTS.json",
        "router": args.router / "REPORT.json",
        "population": args.evaluation / "REPORT.json",
        "screen": args.screen_evaluation / "REPORT.json",
        "evaluator": audit_path,
        "primary_sources": args.web_snapshot / "RECEIPT.json",
        "model": args.bundle / "MODEL.json",
    }
    decision = {
        "created_utc": datetime.now(UTC).isoformat(),
        "bounded_closure_complete": True,
        "sota_demonstrated": False,
        "research_objective_achieved": False,
        "stop_after_closure_requested": True,
        "eap": {
            "local_diagnostic_complete": True,
            "physical_queries": 5805,
            "logical_aliases": 6762,
            "precision_preservation_demonstrated": False,
            "historical_failed_queries": native["historical_original_failed_queries"],
            "historical_ttc_failed_queries": native["historical_original_ttc_failed_queries"],
            "missing_evidence": "Authorized Test12 targets or external evaluation",
        },
        "candidate": {
            "decision": "DO_NOT_PROMOTE",
            "delivered_model_unchanged": True,
            "F0_population_gate_passes": population["routes"]["F0"]["passes_fixed_criteria"],
            "router_screen_gate_passes": False,
        },
        "comparability": {
            "official_equivalence": False,
            "mismatches": mismatches,
            "remaining_blocker": audit["remaining_blocker"],
            "independent_comparable_sota_comparison_demonstrated": False,
        },
        "inputs": {
            name: {"path": str(path), "sha256": file_sha(path)} for name, path in sources.items()
        },
        "generator_sha256": file_sha(Path(__file__)),
    }
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True)
    for name, path in sources.items():
        (evidence / (name + ".json")).write_bytes(path.read_bytes())
    atomic_receipt(args.output / "DECISION.json", decision)
    lines = [
        "# Cierre solicitado de los tres puntos",
        "",
        "Se cierran las comprobaciones disponibles y se detiene esta continuación por petición "
        "del usuario. No se ha demostrado SOTA ni se promueve un modelo nuevo. Cerrar este "
        "trabajo no convierte requisitos no acreditados en resultados positivos.",
        "",
        "## 1. Validación eAP",
        "",
        "Se completó el diagnóstico local de las 5805 consultas físicas y se conservó la vista "
        "de 6762 aliases. Las consultas duplicadas no son muestras independientes. Los "
        "productores, PAIR, historias, contextos y cabeza del original local se reconstruyen "
        "antes de medir la intervención. El router se mide por separado y sigue siendo "
        "diagnóstico.",
        "",
        f"Fallos de paridad histórica del original: {native['historical_original_failed_queries']} "
        f"consultas; en TTC: {native['historical_original_ttc_failed_queries']}. La repetibilidad "
        "local no revoca estos fallos. No hay targets Test12 para demostrar conservación "
        "de precisión.",
        "",
        "| Ruta local | Consultas | Nuevas ausencias | Cambios de signo | "
        "Cambio absoluto máximo (s) |",
        "|---|---:|---:|---:|---:|",
        f"| F0 | {native['queries']} | {native['newly_missing']} | {native['sign_changes']} | "
        f"{number(native['maximum_absolute_change_seconds'])} |",
        f"| Router | {router['physical']['queries']} | {router['physical']['newly_missing']} | "
        f"{router['physical']['sign_changes']} | {number(router['physical']['max_abs_seconds'])} |",
        "",
        "## 2. Decisión del candidato",
        "",
        "**No promover.** F0 supera el gate de desarrollo de 2608; el router falló el screen149 "
        "y no se revoca esa decisión por un agregado posterior. La precisión eAP no está "
        "acreditada. Se verificó que MODEL.json y los componentes del entregable conservan "
        "su identidad. No se entrenan nuevas variantes ni se abre una búsqueda adicional.",
        "",
        "| Dominio F0 | MAE original (s) | MAE canónico (s) |",
        "|---|---:|---:|",
    ]
    for domain, row in population["routes"]["F0"]["domains"].items():
        lines.append(
            f"| {domain} | {number(row['original']['mae_seconds'])} | "
            f"{number(row['canonical']['mae_seconds'])} |"
        )
    slider = mismatches["Slider-750"]
    lines += [
        "",
        "En carretera, el intervalo descriptivo del bootstrap del cambio incluye cero. "
        "Esta población tiene exposición histórica y no es confirmación independiente.",
        "",
        "## 3. Comparabilidad y SOTA",
        "",
        "La auditoría reproduce el redondeo del PDF en nueve de diez secuencias; esto "
        "no demuestra igualdad de todos los targets privados. Slider-750 sigue sin "
        f"equivalencia: {slider['local_original_rte_pct']:.6f}% local frente a "
        f"{slider['official_pdf']['rte_pct']:.2f}% en el PDF original. No se ajustaron offsets "
        "ni targets para forzar coincidencia.",
        "",
        "La [página oficial](https://nail-hnu.github.io/EvTTC/competition/) y el "
        "[formato de datos](https://nail-hnu.github.io/EvTTC/download/data_format/) se "
        "comprobaron de nuevo; sus snapshots y hashes constan en la evidencia. Los "
        "scores publicados de otros métodos no se presentan como reproducciones propias. "
        "No se acredita una comparación SOTA equivalente completa, ni se envió otra entrega "
        "o correo. El borrador para resolver Slider-750 permanece sin enviar.",
        "",
        "## Evidencia y parada",
        "",
        "DECISION.json contiene los hashes de todos los informes consumidos. Los JSON "
        "pequeños se publican en evidence; medios, pesos, NPZ y runs completos se conservan "
        "localmente. Se preservan el intento interrumpido y los fallos anteriores. "
        "El objetivo científico SOTA sigue sin alcanzarse; no queda una promesa de "
        "continuación automática tras este cierre.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
