"""Score sealed author-interval predictions using nearest TTC timestamps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from operational.external_transfer_report import metrics
from operational.rgb_port.accounting import atomic_write_json, sha256_file

from .inputs import SCENARIOS
from .run import read

MODELS = (
    "E_H1_MATCHED",
    "E_CTX_MATCHED",
    "R_H1",
    "R_CTX",
    "F_TRUE",
    "F_ZERO",
    "H8_seed7",
    "H8_seed13",
    "H8_seed23",
    "H8_median3",
    "Garl_event",
    "Garl_full",
)


def nearest_truth(table: np.ndarray, timestamp_s: float) -> tuple[float | None, str, float]:
    """Match the nearest row, never interpolate or search past an invalid nearest GT."""
    if table.ndim != 2 or table.shape[1] != 5 or not len(table):
        raise ValueError("Expected five-column TTC table")
    clock = table[:, 1]
    if not np.isfinite(clock).all() or np.any(np.diff(clock) < 0):
        raise ValueError("TTC timestamps must be finite and monotonic")
    pos = int(np.searchsorted(clock, timestamp_s))
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(clock)]
    index = min(candidates, key=lambda i: (abs(float(clock[i]) - timestamp_s), i))
    delta = abs(float(clock[index]) - timestamp_s)
    if delta > 0.05 + 1e-9:
        return None, "GT_TIMESTAMP_MISMATCH", delta
    truth = float(table[index, 4])
    if not math.isfinite(truth) or truth == 0:
        return None, "INVALID_NEAREST_GT", delta
    return truth, "", delta


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict:
    """Retain failures and report per-model, all-model-common and pairwise coverage."""
    truth = np.array(
        [r["truth_ttc_seconds"] if r["truth_ttc_seconds"] is not None else np.nan for r in rows]
    )
    predictions = {
        name: np.array([r[name] if r[name] is not None else np.nan for r in rows])
        for name in MODELS
    }
    common = np.isfinite(truth)
    for values in predictions.values():
        common &= np.isfinite(values)
    scopes = {"own_available": np.ones(len(rows), bool), "common_all_models": common}
    scopes.update(
        {
            "sequence_" + name: np.array([r["scenario"] == name for r in rows])
            for name in sorted({r["scenario"] for r in rows})
        }
    )
    result = {
        scope: {name: metrics(truth[mask], values[mask]) for name, values in predictions.items()}
        for scope, mask in scopes.items()
    }
    for name, values in predictions.items():
        pair = np.isfinite(truth) & np.isfinite(values) & np.isfinite(predictions["Garl_full"])
        result["pair_" + name + "_Garl_full"] = {
            name: metrics(truth[pair], values[pair]),
            "Garl_full": metrics(truth[pair], predictions["Garl_full"][pair]),
        }
    return result


def score(output: Path) -> dict:
    manifest = read(output / "QUERY_MANIFEST.json")
    baseline = read(output / "BASELINES_SEALED.json")
    own = read(output / "v13/PREDICTIONS_SEALED.json")
    if baseline["status"] != "COMPLETE" or own["status"] != "COMPLETE":
        raise ValueError("Both prediction campaigns must be sealed before reading TTC")
    if baseline["labels_read"] or own["labels_read"]:
        raise ValueError("Prediction seal must be label-free")
    binding_path = output / "BASELINES_BINDING.json"
    if sha256_file(binding_path) != baseline["binding_sha256"]:
        raise ValueError("Baseline binding changed after seal")
    if read(binding_path)["manifest_sha256"] != sha256_file(output / "QUERY_MANIFEST.json"):
        raise ValueError("Baseline population changed")
    if own["query_manifest_sha256"] != sha256_file(output / "QUERY_MANIFEST.json"):
        raise ValueError("V13 population changed")
    own_path = output / "v13/PREDICTIONS.csv"
    if sha256_file(own_path) != own["predictions_sha256"]:
        raise ValueError("V13 predictions changed")
    with own_path.open(encoding="utf-8", newline="") as handle:
        own_rows = list(csv.DictReader(handle))
    own_by_id = {r["query_id"]: r for r in own_rows}
    baseline_by_id = {}
    for name, digest in baseline["fragments"].items():
        path = output / name
        if sha256_file(path) != digest:
            raise ValueError("Baseline predictions changed")
        row = read(path)
        if row["query_id"] in baseline_by_id or row["binding_sha256"] != baseline["binding_sha256"]:
            raise ValueError("Baseline fragment identity mismatch")
        baseline_by_id[row["query_id"]] = row
    ids = {r["query_id"] for r in manifest["rows"]}
    if len(own_rows) != len(ids) or set(own_by_id) != ids or set(baseline_by_id) != ids:
        raise ValueError("Exact query-set equality required")
    # Only now open TTC files. They never enter any model input or query selection.
    labels, label_hashes = {}, {}
    for row in manifest["rows"]:
        sequence = row["sequence_id"]
        if sequence not in labels:
            path = Path(row["raw_path"]).parent / "ttc.csv"
            labels[sequence] = np.genfromtxt(path, dtype=np.float64, ndmin=2)
            label_hashes[sequence] = {"path": str(path), "sha256": sha256_file(path)}
    scored = []
    for row in manifest["rows"]:
        gt = [nearest_truth(labels[row["sequence_id"]], t / 1e6) for t in row["rgb_timestamps_us"]]
        reason = next((reason for _, reason, _ in gt if reason), "")
        value = {
            "query_id": row["query_id"],
            "scenario": row["scenario"],
            "sequence_id": row["sequence_id"],
            "anchor_us": row["anchor_us"],
            "truth_ttc_seconds": gt[-1][0] if not reason else None,
            "gt_reason": reason,
            "gt_max_offset_s": max(x[2] for x in gt),
        }
        for name in MODELS:
            raw = own_by_id[row["query_id"]].get(name)
            if raw is None:
                raw = baseline_by_id[row["query_id"]]["predictions"].get(name)
            prediction = float(raw) if raw is not None and raw != "" else math.nan
            value[name] = prediction if math.isfinite(prediction) else None
        scored.append(value)
    result = {
        "status": "COMPLETE",
        "protocol": manifest["protocol"],
        "manifest_sha256": sha256_file(output / "QUERY_MANIFEST.json"),
        "baseline_seal_sha256": sha256_file(output / "BASELINES_SEALED.json"),
        "v13_seal_sha256": sha256_file(output / "v13/PREDICTIONS_SEALED.json"),
        "labels": label_hashes,
        "population": len(scored),
        "candidate_status": dict(
            Counter(r["reason"] or "ADMITTED" for r in manifest["candidates"])
        ),
        "GT_status": dict(Counter(r["gt_reason"] or "VALID" for r in scored)),
        "metrics": summarize(scored),
        "optimizer_updates": 0,
        "official_comparison_admitted": False,
    }
    write_csv(output / "SCORED_PREDICTIONS.csv", scored)
    write_csv(output / "CANDIDATE_AUDIT.csv", manifest["candidates"])
    result["scored_csv_sha256"] = sha256_file(output / "SCORED_PREDICTIONS.csv")
    atomic_write_json(output / "RESULT.json", result)
    lines = [
        "# EvTTC: reconstrucción de los intervalos indicados por los autores",
        "",
        "No es una reproducción certificada de la Tabla VI: faltan IDs oficiales de consultas, "
        "confirmación del checkpoint exacto y detalles completos del mapping.",
        "",
        f"Candidatos: {len(manifest['candidates'])}. Tripletas admitidas: {len(scored)}.",
        "",
        "Métrica primaria RTE %, predicciones nativas sin nuevo clipping. "
        "MiD es diagnóstico adicional. "
        "V13/V12 mantienen contexto más largo que Garl; las consultas y etiquetas sí son comunes.",
        "",
    ]
    for scope in (
        "own_available",
        "common_all_models",
        *sorted(k for k in result["metrics"] if k.startswith("sequence_")),
    ):
        lines += [
            "## " + scope,
            "",
            "| Modelo | Pares / GT | RTE % | MAE s | MiDc | MiDs | MiDl |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]

        def fmt(x: float | None) -> str:
            return "—" if x is None else f"{x:.3f}"

        for name, m in result["metrics"][scope].items():
            cells = [
                name,
                f"{m['finite_paired']}/{m['labelled']}",
                fmt(m["RTE_pct"]),
                fmt(m["MAE_s"]),
            ]
            cells += [fmt(m["bands"][b]["MiD_finite_rows"]) for b in ("c", "s", "l")]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Promedio macro de RTE entre los tres escenarios",
        "",
        "| Modelo | RTE macro % |",
        "|---|---:|",
    ]
    for name in MODELS:
        values = [result["metrics"]["sequence_" + s][name]["RTE_pct"] for s, _, _, _ in SCENARIOS]
        macro = float(np.mean(values)) if all(v is not None for v in values) else None
        lines.append(f"| {name} | {fmt(macro)} |")
    lines += [
        "",
        "Los tiempos por consulta están en los fragmentos: CPU/FP32 con carga concurrente. "
        "El temporizador V13 incluye las seis cabezas; no compararlo con un forward Garl "
        "o los milisegundos del paper como si tuvieran el mismo perímetro.",
        "",
        "Tabla VI publicada: RTE Garl 8,31 / 10,56 / 12,93 %, promedio 10,60 %. "
        "Estas cifras pertenecen al paper y no a esta ejecución. "
        "Fuente: https://arxiv.org/html/2603.16303v1#S7.T6",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    answer = score(parser.parse_args().output)
    print(
        json.dumps(
            {
                "status": answer["status"],
                "population": answer["population"],
                "GT": answer["GT_status"],
            }
        )
    )
