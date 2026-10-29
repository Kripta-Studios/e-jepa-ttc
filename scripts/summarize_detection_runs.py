"""Archive aggregate run receipts and regenerate the local detection result table.

No images, annotations, per-object predictions or checkpoint tensors are copied.
Use --run repeatedly to collect completed runs; omit it to render archived receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Hash a file without loading a checkpoint into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(run: Path) -> dict[str, Any]:
    """Validate a finished run and retain its aggregate metrics and provenance."""
    report_path = run / "report.json"
    prereg_path = run / "preregistration.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if report["status"] != "COMPLETED":
        raise ValueError(f"Run is not completed: {run}")
    if report["preregistration_sha256"] != sha256(prereg_path):
        raise ValueError(f"Preregistration hash mismatch: {run}")
    if report["checkpoint_sha256"] != sha256(run / "checkpoint.pt"):
        raise ValueError(f"Checkpoint hash mismatch: {run}")
    for key in ("initial_train", "final_train", "selection"):
        if key in report and report[key]["official_eap_score"] is not False:
            raise ValueError("Expected explicitly nonofficial development metrics.")
    # These are token lists and source hashes, not object annotations.
    return {
        "run_name": run.name,
        "report_sha256": sha256(report_path),
        "preregistration_sha256": sha256(prereg_path),
        "checkpoint_sha256_verified": True,
        "preregistration": prereg,
        "report": report,
    }


def number(value: float | None, digits: int = 4) -> str:
    """Render undefined metrics explicitly rather than replacing them with zero."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def render(receipts: list[dict[str, Any]]) -> str:
    """Build all numerical table cells from archived JSON receipts."""
    lines = [
        "# Resultados medidos del detector RGB",
        "",
        "Generado por `scripts/summarize_detection_runs.py` desde "
        "`evidence/training_runs/*.json`. AP101 es de desarrollo, no oficial: "
        "matching XY estricto a 0,5/1/2/4 m y macro sobre clases con GT admitido. "
        "La columna de ocho clases asigna cero a las ausentes. "
        "ATE/ASE/AOE usan matches a 2 m. AP se expresa entre 0 y 1.",
        "",
        "| Run | Población | Frames | Clases GT | AP101 soportadas | AP101 ocho | "
        "ATE m | ASE | AOE rad |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for receipt in receipts:
        report = receipt["report"]
        for key, label in (("final_train", "train"), ("selection", "selección")):
            if key not in report:
                continue
            metrics = report[key]
            cells = [
                receipt["run_name"],
                label,
                str(metrics["sample_count"]),
                str(metrics["supported_class_count"]),
                number(metrics["dev_AP101_supported_classes"]),
                number(metrics["dev_AP101_all_eight_absent_zero"]),
                number(metrics["dev_ATE_m"]),
                number(metrics["dev_ASE"]),
                number(metrics["dev_AOE_rad"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Run | Updates etapa / acumulados | Batch | Entrenamiento s | "
        "Mediana s/update | Pico GPU GB | Razón pérdida centro | Gate overfit |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for receipt in receipts:
        report, prereg = receipt["report"], receipt["preregistration"]
        stage = report["training_steps"]
        total = report.get("total_training_steps", stage)
        cells = [
            receipt["run_name"],
            f"{stage} / {total}",
            str(prereg.get("batch_size", 1)),
            number(report["training_seconds"], 2),
            number(report["step_seconds_median"]),
            number(report["peak_allocated_bytes"] / 1e9, 3),
            number(report["mean_center_loss_ratio"]),
            report["overfit_acceptance"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Los dos runs `overfit16` usan las mismas imágenes; el segundo continúa "
        "el checkpoint y optimizador del primero. No son dos semillas independientes. "
        "La razón de pérdida de esa continuación conserva el denominador inicial "
        "del primer run. El piloto es un modelo nuevo con backbone ImageNet y cabeza "
        "de velocidad; no es una comparación controlada con el overfit.",
        "",
        "El tiempo de entrenamiento excluye preparación, evaluaciones y guardado. "
        "El pico GPU es memoria asignada por Torch, no consumo total del dispositivo. "
        "La tabla no acredita robustez, ocho clases aprendidas, calidad TTC ni "
        "equivalencia con CodaBench. Los recibos conservan pérdidas, métricas por "
        "clase, configuración, hashes y resultados negativos.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Collect optional local runs and regenerate Markdown from aggregate receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.docs / "evidence" / "training_runs"
    evidence.mkdir(parents=True, exist_ok=True)
    for run in args.run:
        receipt = collect(run)
        (evidence / f"{run.name}.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    receipts = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(evidence.glob("*.json"))
    ]
    if not receipts:
        raise ValueError("No archived run receipts to render.")
    (args.docs / "TRAINING_TABLES.md").write_text(render(receipts), encoding="utf-8")
    print(json.dumps({"archived_runs": len(receipts), "table": "TRAINING_TABLES.md"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
