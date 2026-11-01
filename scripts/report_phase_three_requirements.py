"""Report completed experiments against the three still-distinct scientific requirements."""

import argparse
import json
from pathlib import Path

from scripts.audit_phase_external_requirements import digest


def read(path: Path) -> dict:
    """Read a local evidence JSON without altering its bytes."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sealed_report(root: Path, name: str) -> dict:
    """Check a report against its complete parent receipt; do not rerun the experiment."""
    receipt = read(root / "COMPLETE.json")
    if (
        receipt["status"] != "COMPLETE"
        or (root / "FAILED.json").exists()
        or digest(root / name) != receipt["artifacts"][name]
    ):
        raise ValueError("Incomplete or altered report")
    return read(root / name)


def main() -> None:
    """Publish results without turning local sensitivity or a passing screen into SOTA."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "router", "refit", "external", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh report directory")
    native = sealed_report(args.native / "paired_replay", "RESULTS.json")
    router = sealed_report(args.router, "REPORT.json")
    router_freeze = sealed_report(args.router, "FREEZE.json")
    refit = sealed_report(args.refit, "REPORT.json")
    if (
        native["queries"] != 5805
        or not native["local_original_exact"]
        or not native["event_context_exact"]
        or native["native_accuracy_evaluated"]
        or router["accuracy_evaluated"]
        or router_freeze["parents"][str(args.native / "paired_replay/COMPLETE.json")]
        != digest(args.native / "paired_replay/COMPLETE.json")
    ):
        raise ValueError("Native/router scope or ancestry differs")
    external = read(args.external)
    if not external["exact_logical_token_set_match"] or external["test12_accuracy_evaluated"]:
        raise ValueError("External evidence scope changed")
    sources = {
        "native_result": args.native / "paired_replay/RESULTS.json",
        "native_decision": args.native / "DECISION.json",
        "router_result": args.router / "REPORT.json",
        "router_freeze": args.router / "FREEZE.json",
        "router_complete": args.router / "COMPLETE.json",
        "refit_result": args.refit / "REPORT.json",
        "refit_freeze": args.refit / "FREEZE.json",
        "refit_complete": args.refit / "COMPLETE.json",
        "external": args.external,
    }
    status = {
        "eap_local_sensitivity_complete": True,
        "eap_test12_accuracy_validated": False,
        "new_router_development_gate_passes": refit["admit_further_validation"],
        "new_model_approved": False,
        "sota_comparable_superiority_demonstrated": False,
        "goal_achieved": False,
        "next_dependencies": [
            "Authorized official Test12 evaluation; public targets are unavailable",
            "Slider-750 aligned targets/evaluator and equivalent baseline reproduction",
            "Full candidate validation before any delivered-model replacement",
        ],
        "sources": {
            name: {"path": str(path), "sha256": digest(path)} for name, path in sources.items()
        },
        "generator_sha256": digest(Path(__file__)),
    }
    evidence = args.output / "evidence"
    evidence.mkdir(parents=True)
    for name, source in sources.items():
        (evidence / (name + ".json")).write_bytes(source.read_bytes())
    (args.output / "STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# Avance sobre los tres requisitos científicos",
        "",
        "El objetivo sigue sin estar alcanzado. Se conserva el entregable anterior. "
        "Este informe no es un cierre SOTA ni sustituye evaluaciones ausentes.",
        "",
        "## 1. eAP",
        "",
        "Se reconstruyó la cadena local de 5805 consultas físicas / 6762 aliases. "
        "El original local se repite exactamente y el contexto de eventos conserva "
        "identidad. El router congelado se auditó por separado.",
        "",
        f"El original conserva {native['historical_original_failed_queries']} fallos "
        f"de paridad histórica; {native['historical_original_ttc_failed_queries']} afectan "
        "al control de TTC. No se sustituyen por la repetibilidad del nuevo equipo.",
        "",
        "| Ruta | Nuevas ausencias | Cambios de signo | Máximo cambio absoluto (s) |",
        "|---|---:|---:|---:|",
        f"| F0 | {native['newly_missing']} | {native['sign_changes']} | "
        f"{native['maximum_absolute_change_seconds']} |",
        f"| Router congelado | {router['physical']['newly_missing']} | "
        f"{router['physical']['sign_changes']} | {router['physical']['max_abs_seconds']} |",
        "",
        "La precisión Test12 sigue sin medirse. El paquete oficial no distribuye "
        "targets; requiere evaluación en CodaBench. Su plantilla actual coincide "
        "exactamente con nuestros 6762 tokens. No se ha enviado una nueva entrega.",
        "",
        "## 2. Nuevo reajuste emparejado",
        "",
        "Se realizaron 48 ajustes: dos brazos, ocho familias excluidas y tres semillas. "
        "Arquitectura, pérdida, aumentos y presupuesto se conservaron. Cada brazo usa "
        "sus estados originales o canónicos correspondientes. Son resultados OOF de "
        "desarrollo con exposición histórica; no confirmación independiente.",
        "",
        "| Vista | Dominio | MAE original → canónico (s) | RAE original → canónico (%) |",
        "|---|---|---:|---:|",
    ]
    for view in ("population2608", "screen149"):
        for domain, row in refit[view]["domains"].items():
            old, new = row["original"], row["canonical"]
            lines.append(
                f"| {view} | {domain} | {old['mae_seconds']} → {new['mae_seconds']} | "
                f"{old['relative_error_pct']} → {new['relative_error_pct']} |"
            )
    decision = (
        "Supera el filtro de desarrollo; faltan los demás controles."
        if refit["admit_further_validation"]
        else "No supera el filtro de desarrollo; no se realiza ajuste final ni exportación."
    )
    lines += [
        "",
        decision + " El fallo previo del router congelado permanece intacto.",
        "",
        "## 3. Comparabilidad y superioridad",
        "",
        "Slider-750 sigue sin equivalencia establecida entre el evaluador local y el "
        "resultado recibido. La auditoría anterior se conserva; no se ajustan offsets "
        "ni targets para forzar coincidencia. Tampoco se acredita una reproducción "
        "completa equivalente de los baselines SOTA.",
        "",
        "La validación local no elimina estas dependencias externas. La consulta a los "
        "organizadores permanece pendiente de autorización explícita; no se presupone "
        "el envío por haber preparado un borrador.",
        "",
        "## Reproducción y evidencia",
        "",
        "STATUS.json enlaza por hash los informes exactos copiados en evidence. "
        "Los runs completos, cachés y pesos permanecen fuera de Git. Los protocolos "
        "y runners conservan los controles previos y las genealogías de recuperación.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
