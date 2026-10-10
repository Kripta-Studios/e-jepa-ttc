"""Recompute FCWD diagnostics from public CSVs, without models, data mounts or GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "docs/sota_evidence_20261010/evidence/streaming/fcwd_stream_cpu"
SCORED_SHA256 = "3c69670ab689d0f58245b8433d54d4b6b16f5ee5d7ab201bb477e3bebef38df2"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 public evidence table."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    """Hash exact bytes, including original line endings."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mid(truth: float, prediction: float) -> float:
    """MiD at dt=0.1; reject rather than silently drop invalid values in this replay."""
    a, b = 1 - 0.1 / truth, 1 - 0.1 / prediction
    if not all(math.isfinite(x) and x > 0 for x in (a, b)):
        raise ValueError("Invalid logarithm domain: strict replay cannot average this row")
    return 10000 * abs(math.log(a) - math.log(b))


def band(truth: float) -> str:
    """Assign official TTC bands, retaining out-of-band queries."""
    if 0 < truth <= 3:
        return "c"
    if 3 < truth <= 6:
        return "s"
    if 6 < truth <= 10:
        return "l"
    if -10 < truth <= 0:
        return "n"
    return "outside"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write deterministic, portable result tables."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def average(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def table(rows: list[dict], fields: list[str]) -> str:
    """Render a table directly from result records."""

    def fmt(value: object) -> str:
        if value is None or value == "":
            return "N/D"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")

    return "\n".join(
        [
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |",
            *["| " + " | ".join(fmt(row.get(field)) for field in fields) + " |" for row in rows],
        ]
    )


def run(output: Path) -> None:
    """Verify all FCWD pairs and regenerate summaries, contributions and worst cases."""
    output.mkdir(parents=True, exist_ok=True)
    scored = SOURCE / "SCORED_PREDICTIONS.csv"
    if digest(scored) != SCORED_SHA256:
        raise ValueError("Public scored CSV byte identity changed")
    records = read_csv(scored)
    grouped = defaultdict(list)
    paired = defaultdict(dict)
    maximum_delta = 0.0
    for row in records:
        truth, pred = float(row["truth_ttc_seconds"]), float(row["ttc"])
        error = pred - truth
        value = mid(truth, pred)
        maximum_delta = max(maximum_delta, abs(value - float(row["MiD"])))
        if not math.isclose(value, float(row["MiD"]), abs_tol=1e-7, rel_tol=1e-10):
            raise ValueError(f"MiD mismatch: {row['query_id']}")
        if band(truth) != row["band"]:
            raise ValueError("Band mismatch")
        item = {**row, "truth": truth, "prediction": pred, "mid": value, "error": error}
        query = row["query_id"]
        variant = row["variant"]
        if variant in paired[query]:
            raise ValueError("Duplicate query/variant")
        paired[query][variant] = item
        for sequence in ("ALL", row["sequence_id"]):
            for bucket in ("ALL", row["band"]):
                grouped[(sequence, variant, bucket)].append(item)
    variants = {"H8_reference", "Garl_event", "H8_warp", "H8_warp_student"}
    if len(paired) != 597 or len(records) != 2388:
        raise ValueError("Unexpected eligible population")
    for values in paired.values():
        if set(values) != variants or len({x["truth"] for x in values.values()}) != 1:
            raise ValueError("Unpaired or inconsistent ground truth")
    summary = []
    for (sequence, variant, bucket), rows in sorted(grouped.items()):
        summary.append(
            {
                "sequence": sequence,
                "variant": variant,
                "band": bucket,
                "n": len(rows),
                "mean_MiD": average([x["mid"] for x in rows]),
                "MAE_s": average([abs(x["error"]) for x in rows]),
                "bias_s": average([x["error"] for x in rows]),
                "overestimate_fraction": sum(x["error"] > 0 for x in rows) / len(rows),
            }
        )
    lookup = {(x["sequence"], x["variant"], x["band"]): x for x in summary}
    for original in read_csv(SOURCE / "METRICS.csv"):
        rebuilt = lookup[(original["sequence"], original["variant"], "ALL")]
        for field in ("mean_MiD", "MAE_s"):
            if not math.isclose(rebuilt[field], float(original[field]), abs_tol=1e-7):
                raise ValueError(f"Aggregate mismatch: {original['sequence']} {field}")
    contributions = []
    for variant in sorted(variants - {"Garl_event"}):
        subtotal = 0.0
        for bucket in ("c", "s", "l", "outside"):
            ours = lookup[("ALL", variant, bucket)]
            comparator = lookup[("ALL", "Garl_event", bucket)]
            delta = ours["mean_MiD"] - comparator["mean_MiD"]
            contribution = ours["n"] / 597 * delta
            subtotal += contribution
            contributions.append(
                {
                    "variant": variant,
                    "band": bucket,
                    "n": ours["n"],
                    "delta_band_MiD": delta,
                    "contribution_to_mean_gap": contribution,
                }
            )
        expected = (
            lookup[("ALL", variant, "ALL")]["mean_MiD"]
            - lookup[("ALL", "Garl_event", "ALL")]["mean_MiD"]
        )
        if not math.isclose(subtotal, expected, abs_tol=1e-10):
            raise ValueError("Band contributions do not reconstruct the mean gap")
    worst = []
    for query, values in paired.items():
        h, g = values["H8_reference"], values["Garl_event"]
        worst.append(
            {
                "query_id": query,
                "sequence": h["sequence_id"],
                "band": h["band"],
                "truth_s": h["truth"],
                "H8_ttc_s": h["prediction"],
                "Garl_ttc_s": g["prediction"],
                "H8_MiD": h["mid"],
                "Garl_MiD": g["mid"],
                "delta_MiD": h["mid"] - g["mid"],
                "warp_ttc_s": values["H8_warp"]["prediction"],
                "student_ttc_s": values["H8_warp_student"]["prediction"],
            }
        )
    worst.sort(key=lambda x: (-x["delta_MiD"], x["query_id"]))
    write_csv(output / "FCWD_SUMMARY.csv", summary)
    write_csv(output / "FCWD_GAP_CONTRIBUTIONS.csv", contributions)
    write_csv(output / "FCWD_PAIRED_QUERIES.csv", worst)
    report = {
        "status": "VERIFIED_PUBLIC_CSV_REPLAY",
        "optimizer_updates": 0,
        "gpu_seconds": 0,
        "eligible_queries": 597,
        "variant_rows": len(records),
        "sequences": 3,
        "overall_MiD": None,
        "overall_unavailable_reason": "No negative-TTC ground truth",
        "maximum_row_MiD_difference": maximum_delta,
        "sources": {
            str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
            for p in (scored, SOURCE / "METRICS.csv", Path(__file__))
        },
        "output_sha256": {p.name: digest(p) for p in sorted(output.glob("FCWD_*.csv"))},
    }
    (output / "REPLAY_RESULT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    appendix = [
        "# Tablas regeneradas desde evidencia pública",
        "",
        "Generadas por `replay_public.py`. No hay inferencia ni entrenamiento nuevos. "
        "MiD con dt=0,1 s; overall_MiD no disponible en FCWD por ausencia del rango negativo.",
        "",
        "## FCWD: resultado global",
        "",
        table(
            [x for x in summary if x["sequence"] == "ALL" and x["band"] == "ALL"],
            ["variant", "n", "mean_MiD", "MAE_s", "bias_s", "overestimate_fraction"],
        ),
        "",
        "## FCWD: bandas",
        "",
        table(
            [x for x in summary if x["sequence"] == "ALL" and x["band"] != "ALL"],
            ["variant", "band", "n", "mean_MiD", "MAE_s", "bias_s", "overestimate_fraction"],
        ),
        "",
        "## FCWD: contribución de cada banda a la diferencia con Garl",
        "",
        "Positivo significa que H8 empeora. La suma por variante reconstruye la diferencia "
        "de mean_MiD; no es el overall ponderado oficial.",
        "",
        table(contributions, list(contributions[0])),
        "",
        "## FCWD: secuencias",
        "",
        table(
            [x for x in summary if x["sequence"] != "ALL" and x["band"] == "ALL"],
            ["sequence", "variant", "n", "mean_MiD", "MAE_s", "bias_s"],
        ),
        "",
        "## Veinte consultas con mayor desventaja H8 frente a Garl",
        "",
        "Selección descriptiva posterior; no usar estas filas para elegir un checkpoint.",
        "",
        table(worst[:20], list(worst[0])),
        "",
    ]
    sources = [
        (
            "eAP TRAIN40: diagnóstico de entrenamiento",
            HERE / "evidence/train40/METRICS.csv",
            None,
            ["method", "mean_MiD", "overall_MiD", "MiDc", "MiDs", "MiDl", "MiDn"],
        ),
        (
            "Dev32 y FCWD: scorer fijado, predicción nativa",
            ROOT / "docs/sota_evidence_20261010/evidence/garl/MID_OFFICIAL_SCORER.csv",
            "native",
            [
                "dataset",
                "method",
                "mean_MiD",
                "strict_mean_MiD",
                "MiDc",
                "MiDs",
                "MiDl",
                "invalid_mid_total",
                "overall_MiD",
            ],
        ),
        (
            "Ablation sobre características guardadas",
            ROOT / "docs/sota_evidence_20261010/evidence/streaming/external/METRICS.csv",
            None,
            ["dataset", "variant", "MAE_s", "mean_MiD", "MiDc"],
        ),
        (
            "Piloto aislado final: incluye medias con arranque; warm es distinto",
            ROOT / "docs/sota_evidence_20261010/evidence/streaming/isolated_fixed/SUMMARY.csv",
            None,
            ["variant", "n", "mean_total_ms", "warm_median_ms", "warm_p95_ms", "mean_MiD"],
        ),
    ]
    for title, path, policy, fields in sources:
        rows = read_csv(path)
        if policy:
            rows = [x for x in rows if x.get("policy") == policy]
        appendix += [
            f"## {title}",
            "",
            f"Fuente: `{path.relative_to(ROOT).as_posix()}`.",
            "",
            table(rows, fields),
            "",
        ]
    (output / "TABLES.md").write_text("\n".join(appendix), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
