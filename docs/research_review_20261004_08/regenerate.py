"""Verify published evidence and regenerate review tables without models or raw data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def require(condition: object, message: str) -> None:
    """Enforce integrity even when Python runs with optimization enabled."""
    if not condition:
        raise ValueError(message)


def read(name: str) -> dict:
    """Read a frozen evidence snapshot."""
    return json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))


def table(headers: list[str], rows: list[list]) -> str:
    """Render a deterministic Markdown table."""
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    )


def main() -> None:
    """Check hashes, recompute EvTTC metrics and write deterministic tables."""
    manifest = ROOT / "SHA256SUMS.txt"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Delivery file changed: {name}")
    inventory = json.loads((ROOT / "SOURCE_INVENTORY.json").read_text(encoding="utf-8"))
    for item in inventory:
        actual = hashlib.sha256((ROOT / item["snapshot"]).read_bytes()).hexdigest()
        if actual != item["snapshot_sha256"]:
            raise ValueError(f"Evidence changed: {item['snapshot']}")

    result = read("rgb_event/RESULT.json")
    old = read("event_only/RESULT.json")
    with (ROOT / "evidence/rgb_event/SCORED_PREDICTIONS.csv").open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    with (ROOT / "evidence/event_only/SCORED_PREDICTIONS.csv").open(encoding="utf-8") as h:
        inherited = list(csv.DictReader(h))
    require(len(rows) == len(inherited) == 1024, "Expected 1024 inherited rows")
    require(len({r["query_id"] for r in rows}) == len(rows), "Duplicate query IDs")
    truth = np.array([float(r["truth_ttc_seconds"] or "nan") for r in rows])
    seq = np.array([r["sequence_id"] for r in rows])
    valid = np.isfinite(truth)
    require(valid.sum() == 946 and len(np.unique(seq)) == 32, "Cohort coverage changed")
    models = list(result["metrics"])
    pred = {m: np.array([float(r[m] or "nan") for r in rows]) for m in models}
    for r, prior in zip(rows, inherited, strict=True):
        for key in ("query_id", "sequence_id", "anchor_us", "truth_ttc_seconds", *models[:-1]):
            require(r[key] == prior[key], f"Inherited value changed: {r['query_id']} {key}")
    metric_rows = []
    for model in models:
        require(np.isfinite(pred[model][valid]).all(), f"Non-finite prediction: {model}")
        err = pred[model][valid] - truth[valid]
        calculated = [
            np.abs(err).mean(),
            np.median(np.abs(err)),
            np.sqrt((err**2).mean()),
            err.mean(),
        ]
        names = [
            "mae_seconds",
            "median_absolute_error_seconds",
            "rmse_seconds",
            "signed_bias_seconds",
        ]
        expected = result["metrics"][model]
        np.testing.assert_allclose(calculated, [expected[k] for k in names], rtol=1e-12, atol=1e-12)
        if model in old["metrics"]:
            for key in names:
                require(expected[key] == old["metrics"][model][key], "Inherited metric changed")
        metric_rows.append([model, 946, *[f"{v:.6f}" for v in calculated]])
    paired_rows = []
    for model, expected in result["paired_sequence_comparison"].items():
        delta = []
        for s in np.unique(seq):
            mask = valid & (seq == s)
            delta.append(
                (
                    np.abs(pred[model][mask] - truth[mask])
                    - np.abs(pred[models[-1]][mask] - truth[mask])
                ).mean()
            )
        d = np.array(delta)
        rng = np.random.default_rng(20261008)
        boot = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
        ci = np.quantile(boot, [0.025, 0.975])
        np.testing.assert_allclose(d.mean(), expected["difference_seconds"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            ci, expected["sequence_bootstrap_95pct_seconds"], rtol=1e-12, atol=1e-12
        )
        paired_rows.append([model, f"{d.mean():.6f}", f"[{ci[0]:.6f}, {ci[1]:.6f}]"])
    train = read("train40/REPORT.json")
    account = train["accounting"]
    require(
        sum(v["committed_updates"] for v in account["fits"].values()) == 114204,
        "Scientific update accounting mismatch",
    )
    require(
        40748 + 114204 + 91 + 35 == account["physical_upper"] == 155078,
        "Combined physical upper mismatch",
    )
    require(
        22500 + 17948 + 300 == account["previous_campaign_physical_upper"],
        "Previous campaign upper mismatch",
    )
    train_rows = [
        [m, v["population"], f"{v['MAE_seconds_finite']:.6f}", f"{v['RMSE_seconds_finite']:.6f}"]
        for m, v in train["TRAIN_fit_metrics"].items()
    ]
    engineering = read("engineering_audit.json")
    rates = engineering["training_efficiency"]["host_pipeline_and_collate"]["windows"]
    rate_rows = [
        [
            v["name"],
            v["updates"],
            f"{v['range'][0]}–{v['range'][1]}",
            f"{v['upm']:.3f}",
            v.get("selection_eligible", "véase análisis"),
        ]
        for v in rates
    ]
    wide = read("e0_e3/WIDE_REPLICATION_RESULTS.json")["results"]["all_three_seeds"]
    wide_rows = [[m, f"{score:.6f}"] for m, score in wide["scores"].items()]
    with (ROOT / "commits.csv").open(encoding="utf-8") as h:
        commits = list(csv.DictReader(h))
    commit_rows = [[r["author_datetime"], f"`{r['commit'][:7]}`", r["subject"]] for r in commits]
    docs = {
        "EVTTC_METRICS.md": "# EvTTC: métricas regeneradas\n\n"
        "Mismo soporte finito: 946 consultas, 32 secuencias. No es el benchmark oficial.\n\n"
        + table(["Modelo", "N", "MAE s", "Mediana AE s", "RMSE s", "Sesgo s"], metric_rows)
        + "\n\n## Diferencia macro por secuencia respecto a Garl RGB+eventos\n\n"
        "Bootstrap por secuencia, 2.000 remuestreos, semilla 20261008. "
        "Negativo favorece al modelo de la fila. "
        "No confundir con diferencia de MAE agrupados.\n\n"
        + table(["Modelo", "Delta s", "IC 95% s"], paired_rows),
        "TRAIN_METRICS.md": "# TRAIN40: ajuste sobre entrenamiento\n\n"
        "No son métricas de generalización.\n\n"
        + table(["Modelo", "N", "MAE s", "RMSE s"], train_rows),
        "PERFORMANCE.md": "# Ventanas de rendimiento observadas\n\n"
        "Ventanas secuenciales con carga y datos cambiantes. No son ensayos aleatorizados. "
        "La fila fast_4 está invalidada por desconexión de disco.\n\n"
        + table(["Ventana", "Updates", "Rango", "Updates/min", "Elegible"], rate_rows),
        "WIDE.md": "# WIDE: OLD_DEV reutilizado\n\n"
        "Media de pérdidas pareadas por consulta sobre tres semillas. "
        "MiD es el score local PHASE17, no segundos.\n\n"
        + table(["Modelo", "MiD"], wide_rows)
        + "\n\n"
        + table(
            ["Referencia", "Delta WIDE", "IC jerárquico 95%"],
            [
                [m, f"{v['point_delta']:.6f}", str(v["hierarchical_ci95"])]
                for m, v in wide["contrasts"].items()
            ],
        ),
        "ACCOUNTING.md": "# Contabilidad al cierre de TRAIN40\n\n"
        + table(
            ["Componente", "Updates / cota"],
            [
                ["WIDE", 22500],
                ["Garl nativo conservado", 17948],
                ["Recuperación original, cota", 300],
                ["A5", 49932],
                ["C2F", 49932],
                ["PAIR", 6840],
                ["Tres cabezas H8", 7500],
                ["Recuperación TRAIN40, cota", 91],
                ["Técnica sintética", 35],
                ["Total físico superior", account["physical_upper"]],
                ["Techo conjunto", account["physical_cap"]],
            ],
        )
        + "\n\nLos 40.748 originales ya contienen WIDE. "
        "Las 1.728 mediciones históricas no son updates. "
        "La cota 155.078 no afirma ejecución exacta ni autoriza consumir el saldo.",
        "COMMITS.md": "# Historial de cambios del 4 al 8 de octubre de 2026\n\n"
        "77 commits anteriores a estos informes; fechas de autor con zona horaria. "
        "Un commit no prueba por sí solo una ejecución experimental.\n\n"
        + table(["Fecha", "Commit", "Cambio"], commit_rows),
    }
    target = ROOT / "tables"
    target.mkdir(exist_ok=True)
    for name, content in docs.items():
        (target / name).write_text(content + "\n", encoding="utf-8", newline="\n")
    print(
        f"PASS: {len(inventory)} evidence hashes; 1024 inherited rows; "
        f"5 model metrics; 4 sequence bootstraps; accounting; {len(commits)} commits"
    )


if __name__ == "__main__":
    main()
