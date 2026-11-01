"""Regenerate the handoff's numerical tables from preserved evaluation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/detection_sota_20261029"
TARGET = DOCS / "ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261101.md"
BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"


def read(path: Path) -> Any:
    """Read a UTF-8 result file without changing its historical bytes."""
    return json.loads(path.read_text(encoding="utf-8"))


def render() -> str:
    """Build official, local, per-class and arithmetic-control tables."""
    evidence = DOCS / "evidence"
    official = read(evidence / "official_benchmark_20261101/metrics_summary.json")
    folder = evidence / "benchmark_local_alignment_20261101"
    comparison = read(folder / "COMPARISON.json")
    local = {name: read(folder / f"{name}.json") for name in comparison["variants"]}
    ranked = local["ranked"]
    historical = comparison["historical_comparison"]["results"]
    metrics = (
        ("mAP", official["mean_ap"]),
        ("ATE / trans_err", official["tp_errors"]["trans_err"]),
        ("ASE / scale_err", official["tp_errors"]["scale_err"]),
        ("AOE / orient_err", official["tp_errors"]["orient_err"]),
    )
    score = (
        5 * official["mean_ap"]
        + sum(
            max(0, 1 - official["tp_errors"][name])
            for name in ("trans_err", "scale_err", "orient_err")
        )
    ) / 8
    lines = [
        "### Resultado oficial recibido (test privado)",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        *[f"| {name} | {value:.8f} |" for name, value in metrics],
        f"| eAP compuesto | {score:.8f} |",
        f"| Muestras | {official['num_samples']} |",
        f"| Objetos GT | {official['num_gt_boxes']} |",
        f"| Predicciones | {official['num_pred_boxes']} |",
        "",
        "Método enviado: `PHASE-TTC MonoDGP-R50 RGB + Quality Ranking`; "
        "`uses_event=false`, `uses_image=true`.",
        "",
        "### Selección completa con el perfil alineado",
        "",
        "| Variante | mAP | ATE | ASE | AOE | eAP compuesto |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "scalar_uncertainty", "ranked"):
        r = local[name]
        e = r["tp_errors"]
        lines.append(
            f"| {name} | {r['mAP']:.8f} | {e['trans_err']:.8f} | "
            f"{e['scale_err']:.8f} | {e['orient_err']:.8f} | {r['eAP_score']:.8f} |"
        )
    gain = ranked["mAP"] - local["baseline"]["mAP"]
    eligible = sum(comparison["historical_eligible_by_class"].values())
    ignored = sum(comparison["historical_ignored_by_class"].values())
    lines += [
        "",
        f"Población: **{ranked['samples']} frames, {ranked['gt_boxes']} GT raw y "
        f"{ranked['pred_boxes']} propuestas** por variante.",
        f"Antes: {eligible} GT elegibles y {ignored} ignorados "
        f"({100 * ignored / ranked['gt_boxes']:.2f} %).",
        f"Ganancia del ranking: **{gain:.8f} mAP absoluta; "
        f"{100 * gain / local['baseline']['mAP']:.2f} % relativa**.",
        "",
        "### Diagnóstico por clase del candidato con ranking",
        "",
        "AP en escala 0–1. Los recalls son fracciones en selección local raw.",
        "",
        "| Clase | GT local | AP local | Recall local 2 m | Recall local 4 m | AP oficial |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, c in ranked["classes"].items():
        ap_official = sum(official["label_aps"][name].values()) / 4
        lines.append(
            f"| {name} | {c['gt_count']} | {c['AP']:.6f} | "
            f"{c['thresholds']['2.0']['recall']:.6f} | "
            f"{c['thresholds']['4.0']['recall']:.6f} | {ap_official:.6f} |"
        )
    lines += [
        "",
        "### Separar fórmula y población",
        "",
        "| Cálculo | Baseline | Ranking |",
        "|---|---:|---:|",
        f"| Histórico v2, dominio antiguo | {historical['baseline']['AP']:.8f} | "
        f"{historical['learned']['AP']:.8f} |",
    ]
    key = "diagnostic_unclipped_envelope_AP101_same_population"
    lines += [
        f"| Control AP101 con envolvente, mismos matches raw nuevos | "
        f"{local['baseline'][key]:.8f} | {ranked[key]:.8f} |",
        f"| Perfil alineado, población raw | {local['baseline']['mAP']:.8f} | "
        f"{ranked['mAP']:.8f} |",
        "",
        "El control de la segunda fila cambia la fórmula sobre los mismos matches de la",
        "tercera. La primera también cambia población, filtros y reglas de empates.",
        "No atribuir toda la diferencia histórica–oficial a una única causa.",
        "",
        f"Hash del perfil alineado: `{comparison['profile_sha256']}`.",
    ]
    return "\n".join(lines)


def main() -> None:
    """Check table freshness, or explicitly write the generated block."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    original = TARGET.read_text(encoding="utf-8")
    if original.count(BEGIN) != 1 or original.count(END) != 1:
        raise ValueError("Expected one numerical-results block")
    before, rest = original.split(BEGIN)
    _, after = rest.split(END)
    updated = before + BEGIN + "\n" + render() + "\n" + END + after
    if args.write:
        TARGET.write_bytes(updated.encode("utf-8"))
    elif updated != original:
        raise SystemExit("Stale result tables; run with --write")
    print("Handoff numerical tables match preserved JSON evidence.")


if __name__ == "__main__":
    main()
