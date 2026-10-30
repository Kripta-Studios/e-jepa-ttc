"""Regenerate diagnostic baseline tables from saved ledgers and aligned targets."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from operational.evttc_transfer_audit.baselines import saved_predictions
from operational.evttc_transfer_audit.common import file_sha, read_document, write_receipt
from operational.evttc_transfer_audit.scoring import score_rows


def metric_text(value: float | None) -> str:
    """Render absent metrics without losing zero-coverage baseline rows."""
    return "N/A" if value is None else f"{value:.6f}"


def main() -> None:
    """Write full-population coverage and paired comparisons without imputing outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phase-run", "garl-run", "targets", "policy", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    targets = read_document(args.targets)["rows"]
    policy = read_document(args.policy)
    phase_path, garl_path = args.phase_run / "all_rows.jsonl", args.garl_run / "all_rows.jsonl"
    phase = [json.loads(line) for line in phase_path.read_text(encoding="utf-8").splitlines()]
    garl = [json.loads(line) for line in garl_path.read_text(encoding="utf-8").splitlines()]
    methods = ("PHASE", "PHASE_event", "PHASE_rgb", "Garl_full", "Garl_visual", "Garl_event")
    predictions = {
        m: saved_predictions(phase if m.startswith("PHASE") else garl, targets, m) for m in methods
    }
    scores = {m: score_rows(predictions[m], targets, policy) for m in methods}
    paired = {}
    for method in methods[1:]:
        ids = {
            r["query_id"]
            for r in predictions[method]
            if r["prediction"] is not None and math.isfinite(r["prediction"])
        }
        ids &= {
            r["query_id"]
            for r in predictions["PHASE"]
            if r["prediction"] is not None and math.isfinite(r["prediction"])
        }
        shared = [t for t in targets if t["query_id"] in ids and t["target_status"] == "valid"]
        ids = {t["query_id"] for t in shared}
        paired[method] = {
            m: score_rows([r for r in predictions[m] if r["query_id"] in ids], shared, policy)
            for m in ("PHASE", method)
        }
    result = {
        "scope": "diagnostic_partial",
        "official_equivalence": "unverified",
        "source_sha256": {
            "phase_ledger": file_sha(phase_path),
            "garl_ledger": file_sha(garl_path),
            "targets": file_sha(args.targets),
            "policy": file_sha(args.policy),
        },
        "scores": scores,
        "paired": paired,
    }
    write_receipt(args.output / "report.json", result)
    lines = [
        "# Baselines: diagnóstico local expuesto",
        "",
        "Targets CSV locales con geometría no verificada equivalente al scorer oficial. "
        "Los subconjuntos compartidos pueden ser más fáciles. "
        "Ninguna fila sustituye el score oficial de 434 consultas.",
        "",
        "| Método | Predicción finita / solicitadas | Puntuadas | MAE micro (s) | "
        "Error relativo micro (%) | Error relativo macro (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, report in scores.items():
        counts = report["counts"]
        micro, macro = report["metrics"]["micro"], report["metrics"]["macro"]
        lines.append(
            f"| {method} | {counts['finite_prediction']}/{counts['requested']} | "
            f"{counts['scored']} | "
            f"{metric_text(micro['mae_seconds'])} | "
            f"{metric_text(micro['relative_absolute_error_pct'])} | "
            f"{metric_text(macro['relative_absolute_error_pct'])} |"
        )
    lines += [
        "",
        "## Comparación en consultas compartidas",
        "",
        "| Comparador | Consultas | PHASE error relativo micro (%) | Comparador (%) |",
        "|---|---:|---:|---:|",
    ]
    for method, reports in paired.items():
        lines.append(
            f"| {method} | {reports[method]['counts']['scored']} | "
            f"{metric_text(reports['PHASE']['metrics']['micro']['relative_absolute_error_pct'])} | "
            f"{metric_text(reports[method]['metrics']['micro']['relative_absolute_error_pct'])} |"
        )
    lines += [
        "",
        "PHASE: RGB+eventos, rama RGB o rama eventos del mismo ledger sellado. "
        "Garl: variantes full/visual/event con su preparación nativa declarada. "
        "Las diferencias de cámara, ROI, historia, endpoint y preparación siguen las guías "
        "`docs/evttc_competition.md` y `docs/garl_evttc_competition.md`.",
        "",
        "No se ha ejecutado entrenamiento, interpolación de predicciones ni corrección del signo.",
        "",
    ]
    path = args.output / "BASELINES.md"
    content = "\n".join(lines)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError("Refusing to replace different baseline evidence")
    path.write_text(content, encoding="utf-8")
    print(f"Generated {len(scores)} baselines and {len(paired)} paired comparisons")


if __name__ == "__main__":
    main()
