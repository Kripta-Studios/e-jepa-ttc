"""Post-seal FCWD scoring; registration and weights cannot use these targets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.evaluate import common_cap60_metrics, score_signed_ttc
from operational.rgb_port.transfer import _write_csv


def score(prediction_dir: Path, comparator: Path) -> dict[str, Any]:
    """Retain every row, with separate whole-population and paired RGB views."""
    seal_path = prediction_dir / "PREDICTIONS_SEALED.json"
    seal = read_json_shared(seal_path)
    path = prediction_dir / "PREDICTIONS.csv"
    if (
        seal["status"] != "COMPLETE"
        or seal["labels_read"]
        or sha256_file(path) != seal["predictions_sha256"]
    ):
        raise ValueError("Complete target-free sealed predictions required before reading labels")
    with path.open(encoding="utf-8", newline="") as handle:
        own = list(csv.DictReader(handle))
    with comparator.open(encoding="utf-8", newline="") as handle:
        reference = list(csv.DictReader(handle))
    by_id = {r["query_id"]: r for r in reference}
    if (
        len(by_id) != len(reference)
        or len(own) != len(by_id)
        or {r["query_id"] for r in own} != set(by_id)
    ):
        raise ValueError("Comparator population must match every frozen query exactly")
    for row in own:
        other = by_id[row["query_id"]]
        if any(row[key] != other[key] for key in ("anchor_us", "sequence_id")):
            raise ValueError("Comparator query clock/sequence changed")
    joined = [{**by_id[r["query_id"]], **r} for r in own]
    _write_csv(prediction_dir / "SCORED_PREDICTIONS.csv", joined, list(joined[0]))
    truth = np.asarray([float(r["truth_ttc_seconds"]) for r in joined])
    groups = np.asarray([r["sequence_id"] for r in joined])
    rgb_valid = np.asarray([r["rgb_available"].lower() in {"1", "true"} for r in joined])
    methods = [
        "E_H1_MATCHED",
        "E_CTX_MATCHED",
        "R_H1",
        "R_CTX",
        "F_TRUE",
        "F_ZERO",
        "H8_seed7",
        "H8_seed13",
        "H8_seed23",
        "public_Garl_event_lhr",
    ]
    views = {}
    for scope, mask in (
        ("all_queries", np.ones(len(joined), bool)),
        ("same_RGB_available_queries", rgb_valid),
    ):
        metrics = {}
        for method in methods:
            pred = np.asarray([float(r[method]) if r[method] else np.nan for r in joined])
            labelled = mask & np.isfinite(truth)
            native = score_signed_ttc(truth[labelled], pred[labelled], groups[labelled])
            metrics[method] = {
                "native": native,
                "common_cap60": common_cap60_metrics(
                    truth[labelled], pred[labelled], groups[labelled]
                ),
            }
        views[scope] = {
            "rows": int(mask.sum()),
            "labelled_rows": int((mask & np.isfinite(truth)).sum()),
            "metrics": metrics,
        }
    result = {
        "status": "COMPLETE",
        "scope": "FCWD_exploratory_registered_RGB_not_official_leaderboard",
        "population": len(joined),
        "rows_omitted": 0,
        "rgb_available": int(rgb_valid.sum()),
        "prediction_seal_sha256": sha256_file(seal_path),
        "comparator_sha256": sha256_file(comparator),
        "views": views,
        "optimizer_updates": 0,
        "official_comparison_admitted": False,
        "metric_note": (
            "group_macro_bucket_MiD is not official pooled overall_MiD; "
            "absent bands remain unavailable"
        ),
        "Garl_RGB_event": "unavailable historical comparator; no value imputed",
    }
    atomic_write_json(prediction_dir / "RESULT.json", result)
    return result
