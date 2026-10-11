"""Audit published CSV metrics and regenerate handoff tables without ML dependencies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BANDS = (("c", 0, 3), ("s", 3, 6), ("l", 6, 10), ("n", -10, 0))


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: str) -> float:
    return float(value) if value else math.nan


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def metrics(rows: list[dict[str, str]], method: str) -> dict:
    """Compute independently with Python scalars, retaining failure denominators."""
    labelled = [(number(r["truth_ttc_seconds"]), number(r[method])) for r in rows]
    labelled = [(t, p) for t, p in labelled if math.isfinite(t)]
    paired = [(t, p) for t, p in labelled if math.isfinite(p)]
    errors = [p - t for t, p in paired]
    bands = {}
    for name, low, high in BANDS:
        selected = [(t, p) for t, p in labelled if low < t <= high]
        valid = []
        for truth, pred in selected:
            try:
                value = abs(math.log1p(-0.1 / truth) - math.log1p(-0.1 / pred)) * 1e4
            except (ValueError, ZeroDivisionError):
                continue
            if math.isfinite(value):
                valid.append(value)
        bands[name] = {
            "rows": len(selected),
            "invalid_phase_rows": len(selected) - len(valid),
            "MiD_finite_rows": mean(valid),
        }
    return {
        "rows": len(rows),
        "labelled": len(labelled),
        "finite_paired": len(paired),
        "missing_prediction_on_label": len(labelled) - len(paired),
        "MAE_s": mean([abs(e) for e in errors]),
        "mean_bias_s": mean(errors),
        "RTE_pct": mean([abs(p - t) / abs(t) * 100 for t, p in paired if t != 0]),
        "positive_wrong_sign": sum(t > 0 and p <= 0 for t, p in paired),
        "abs_prediction_at_least_59_999": sum(abs(p) >= 59.999 for _, p in paired),
        "bands": bands,
    }


def compare(actual: dict, expected: dict) -> int:
    count = 0
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, dict):
            count += compare(value, other)
        else:
            if isinstance(value, float):
                if other is None or not math.isclose(value, other, rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError(f"Metric mismatch {key}: {value} != {other}")
            elif value != other:
                raise ValueError(f"Count/status mismatch {key}: {value} != {other}")
            count += 1
    return count


def run(output: Path) -> dict:
    manifest = read(HERE / "SOURCE_MANIFEST.json")
    for entry in manifest["files"]:
        content = (ROOT / entry["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source bytes changed: {entry['path']}")
    lines = [
        "# 04 — Resultados regenerados desde evidencia",
        "",
        "Generado por `replay.py`. No incluye nuevas inferencias ni updates.",
        "",
        "## Endpoints V13",
        "",
        "| Fit | Estado | Updates |",
        "|---|---|---:|",
    ]
    for filename in ("PRODUCERS_PAIR_ENDPOINT_FREEZE.json", "HEADS_V_ROUND_FREEZE.json"):
        for name, item in read(HERE / "evidence" / filename)["endpoints"].items():
            # PAIR may be referenced again by the head freeze; report each once.
            row = f"| {name} | {item['status']} | {item['completed_updates']} |"
            if row not in lines:
                lines.append(row)
    lines += [
        "",
        "## Validación V: macro por grupo y banda",
        "",
        "No es overall_MiD pooled oficial. 16.432 consultas de ocho grupos.",
        "",
        "| Modelo | group_macro_bucket_MiD | MAE s | Fallos |",
        "|---|---:|---:|---:|",
    ]
    validation = read(HERE / "evidence/V_RESULTS.json")
    for name, result in validation["scores"].items():
        m = result["native"]
        lines.append(
            f"| {name} | {fmt(m['group_macro_bucket_MiD'])} | "
            f"{fmt(m['diagnostics']['mae_s'])} | {m['num_failed_predictions']} |"
        )
    lines += ["", "| Gate V | Aprobado | IC95 diferencia A−B |", "|---|---|---|"]
    for name, gate in validation["screening_gates"].items():
        lo, hi = gate["bootstrap"]["ci95"]
        lines.append(f"| {name} | {gate['passed']} | [{lo:.3f}, {hi:.3f}] |")
    lines += [
        "",
        "## V12 eAP TRAIN40: diagnóstico sobre entrenamiento",
        "",
        "88.744 consultas. No es holdout ni puntuación CodaBench.",
        "",
        "| Método | overall_MiD | mean_MiD | MiDc | MiDs | MiDl | MiDn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    with (HERE / "evidence/V12_TRAIN40_METRICS.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            keys = ("overall_MiD", "mean_MiD", "MiDc", "MiDs", "MiDl", "MiDn")
            lines.append(
                "| " + " | ".join([row["method"], *[fmt(float(row[k])) for k in keys]]) + " |"
            )
    checks = 0
    for folder, label in (
        ("fcwd_rgb_registration_20261012", "FCWD"),
        ("dev32_transfer_20261012", "Dev32"),
    ):
        base = ROOT / "docs" / folder
        diagnostics = read(base / "DIAGNOSTICS.json")
        with (base / "SCORED_PREDICTIONS.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        def available(row: dict[str, str]) -> bool:
            return row["rgb_available"].lower() in {"true", "1"}

        scopes = {
            "all_queries": rows,
            "same_RGB_available_queries": [r for r in rows if available(r)],
            "RGB_unavailable_queries": [r for r in rows if not available(r)],
        }
        scopes.update(
            {
                "sequence_" + seq: [r for r in rows if r["sequence_id"] == seq]
                for seq in sorted({r["sequence_id"] for r in rows})
            }
        )
        lines += [
            "",
            f"## {label}",
            "",
            f"{len(rows)} predicciones. Sin banda negativa: overall_MiD oficial no disponible.",
            "",
        ]
        for scope, selected in scopes.items():
            table = scope in {
                "all_queries",
                "same_RGB_available_queries",
                "RGB_unavailable_queries",
            }
            if table:
                lines += [
                    f"### {scope}",
                    "",
                    "| Modelo | Pares / GT | MAE s | RTE % | Sesgo s | "
                    "MiDc | MiDs | MiDl | MiDn | Inválidos c/s/l/n |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            for method, expected in diagnostics["views"][scope].items():
                m = metrics(selected, method)
                checks += compare(m, expected)
                if table:
                    cells = [
                        method,
                        f"{m['finite_paired']}/{m['labelled']}",
                        fmt(m["MAE_s"]),
                        fmt(m["RTE_pct"]),
                        fmt(m["mean_bias_s"]),
                    ]
                    cells += [fmt(m["bands"][b]["MiD_finite_rows"]) for b, _, _ in BANDS]
                    cells += [
                        "/".join(str(m["bands"][b]["invalid_phase_rows"]) for b, _, _ in BANDS)
                    ]
                    lines.append("| " + " | ".join(cells) + " |")
            if table:
                lines.append("")
        lines += [
            "### F_TRUE por secuencia",
            "",
            "| Secuencia | GT | MAE s | Sesgo s | Signos erróneos GT positivo | "
            "Salidas abs≥59,999 s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for scope, selected in scopes.items():
            if scope.startswith("sequence_"):
                m = metrics(selected, "F_TRUE")
                lines.append(
                    f"| {scope[9:]} | {m['labelled']} | {fmt(m['MAE_s'])} | "
                    f"{fmt(m['mean_bias_s'])} | {m['positive_wrong_sign']} | "
                    f"{m['abs_prediction_at_least_59_999']} |"
                )
    lines += [
        "",
        "## Costes V13: host con carga CPU concurrente",
        "",
        "Rutas distintas; no sumar medianas. No es el piloto streaming V12.",
        "",
        "| Ruta | Mediana caliente ms | p95 ms |",
        "|---|---:|---:|",
    ]
    costs = read(ROOT / "docs/external_mask_20261012/COST_SUMMARY.json")
    for route, m in costs["routes"].items():
        lines.append(f"| {route} | {fmt(m['warm_ms']['median'])} | {fmt(m['warm_ms']['p95'])} |")
    lines += [
        "",
        "Fuentes y hashes: `SOURCE_MANIFEST.json`. Las comprobaciones por secuencia "
        "incluyen también las vistas no impresas de los demás modelos.",
        "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "04_RESULTADOS_GENERADOS.md").write_text("\n".join(lines), encoding="utf-8")
    receipt = {
        "status": "PASSED",
        "source_files_verified": len(manifest["files"]),
        "scalar_metric_checks": checks,
        "optimizer_updates": 0,
        "new_inferences": 0,
        "dependencies": "Python standard library",
    }
    (output / "REPLAY_RESULT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
