"""Recompute review tables from saved public evidence; never run training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    i = int(pos)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i)


def main() -> None:
    process = subprocess.run([sys.executable, "-S", "paper/scripts/verify_public_data.py"],
                             cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=True)
    checks = {"paper_verifier": json.loads(process.stdout)}
    with (ROOT / "paper/data/transfer_queries.csv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["cohort"] == "extended_new" and not r["target_exclusion"]]
    expected = read("paper/revision/derived_transfer_analysis.json")
    models = {"phase": "PHASE_ONLY_TRAIN40_S7", "rgb": "R_CTX_TRAIN40",
              "event": "E_CTX_TRAIN40", "garl": "Garl_full"}
    calculated = {}
    for name, field in models.items():
        errors = [abs(float(r[field]) - float(r["target_ttc"])) for r in rows]
        metrics = dict(mean=statistics.mean(errors), median=statistics.median(errors),
                       p95=quantile(errors, 0.95), max=max(errors),
                       rte=statistics.mean(100 * e / abs(float(r["target_ttc"])) for e, r in zip(errors, rows)),
                       counts_above_s={str(t): sum(e > t for e in errors) for t in [1, 3, 5, 10]})
        for key in ["mean", "median", "p95", "max", "rte"]:
            assert math.isclose(metrics[key], expected["metrics"][name][key], rel_tol=1e-12, abs_tol=1e-12)
        assert metrics["counts_above_s"] == expected["metrics"][name]["counts_above_s"]
        calculated[name] = metrics
    checks["transfer"] = dict(n=len(rows), groups=len({r["group_id"] for r in rows}), metrics=calculated,
                              note="Recomputed from predictions; bootstrap intervals below are archived, not new resampling.")
    factorial = [r for r in read("paper/data/early_results.json") if r["evidence_id"] == "E21"]
    checks["v8_saved_scalar_projection"] = factorial
    closure = read("docs/jepa_closure_20261031/closure_analysis.json")
    checks["jepa_saved_analysis"] = {k: closure[k] for k in ["decision", "primary_benefit_gate_passed", "primary_mae", "primary_comparisons"]}
    checks["math"] = {}
    for q in [-0.5, -0.1, 0.0, 0.1, 0.5]:
        f = lambda x, dt: x / (1 - dt * x)
        assert math.isclose(f(f(q, 0.1), 0.2), f(q, 0.3), abs_tol=1e-15)
    checks["math"]["clock_composition"] = "passes 5 signed/zero scalar examples; arithmetic check, not sensor validation"
    checks["math"]["endpoint_example"] = dict(dt_s=0.1, height_ratio=1.1, current_ttc=0.1 / math.expm1(math.log(1.1)), previous_ttc=0.1 / (1 - 1 / 1.1))
    files = ["paper/data/transfer_queries.csv", "paper/data/early_results.json", "paper/data/public_evidence_registry.json",
             "paper/revision/derived_transfer_analysis.json", "docs/jepa_closure_20261031/closure_analysis.json"]
    checks["source_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
    (OUT / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Resultados verificados y procedencia", "", "Generado por verify_evidence.py. No se entrenan modelos ni se ejecuta inferencia de sensores.", "",
             "## Artículo", "", "13 hashes públicos, 8 resúmenes de transferencia y 12 campos del scorer comprobados por el verificador original.", "",
             f"Overall MiD recalculado: {checks['paper_verifier']['official_weighted_MiD']:.12f}. Ranking: snapshot histórico, no consulta actual al leaderboard.", "",
             "## Transferencia Extended-new", "", f"{len(rows)} consultas; {checks['transfer']['groups']} secuencias. Cálculo nuevo desde CSV congelado.", "",
             "| Modelo | MAE s | Mediana s | p95 s | Máximo s | RTE % | Errores >10 s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in calculated.items():
        lines.append(f"| {name} | {m['mean']:.9f} | {m['median']:.9f} | {m['p95']:.9f} | {m['max']:.9f} | {m['rte']:.6f} | {m['counts_above_s']['10']} |")
    lines.extend(["", "## Intervalos históricos PHASE menos RGB", "", "No se han vuelto a generar las 10.000 réplicas; se transcribe el análisis archivado con hash.", "",
                  "| Estadístico | Delta s | IC95% inferior | Superior |", "|---|---:|---:|---:|"])
    b = expected["bootstrap"]
    for i, key in enumerate(["media", "mediana", "p95"]):
        lines.append(f"| {key} | {b['delta'][i]:.9f} | {b['ci_low'][i]:.9f} | {b['ci_high'][i]:.9f} |")
    lines.extend(["", "## V8: escalares archivados, no reejecución", "", "| Ruta | MiD |", "|---|---:|"])
    for row in factorial:
        lines.append(f"| {row['model']} | {row['value']:.9f} |")
    lines.extend(["", "## Piloto JEPA: análisis archivado", "", "MAE de cambio de log-altura, sin unidades; no MAE de TTC.", "",
                  "| Brazo | MAE |", "|---|---:|"])
    for name, value in closure["primary_mae"].items():
        lines.append(f"| {name} | {value:.9f} |")
    lines.extend(["", "La puerta principal falló. Comparaciones, límites y hashes completos: [verification.json](verification.json).", ""])
    (OUT / "RESULTS_VERIFIED.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(public=checks["paper_verifier"], transfer_rows=len(rows), transfer_groups=checks["transfer"]["groups"], math="passed")))


if __name__ == "__main__":
    main()
