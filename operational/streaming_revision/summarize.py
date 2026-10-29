"""Regenerate pilot tables from immutable measurement journals, retaining failed trials."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from operational.efficient_context.common import digest
from operational.sota_evidence.metrics import observations
from operational.sota_evidence.run import json_safe
from operational.train40_system.durable_io import atomic_json


def run(output: Path) -> None:
    """Extract scalar Garl TTC from its native return envelope, never alter predictions."""
    receipt = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
    journal = output / "ROWS.jsonl"
    records, failures, repairs = {}, [], 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        row = event["last_measurement"]
        failures = event["failures"]
        if row is None or row["variant"] != event["variant"] or row["ordinal"] != event["ordinal"]:
            continue
        if isinstance(row["ttc"], dict):
            if row["variant"] != "garl_event" or set(row["ttc"]) != {"ttc", "heights"}:
                raise ValueError("unrecognized TTC envelope")
            row["ttc"] = row["ttc"]["ttc"]
            repairs += 1
        records[(row["variant"], row["ordinal"])] = row
    frame = pd.DataFrame(records.values())
    original = output / "PREDICTIONS.csv"
    if repairs and original.exists() and not (output / "PREDICTIONS_RAW.csv").exists():
        shutil.copy2(original, output / "PREDICTIONS_RAW.csv")
    frame.to_csv(original, index=False)
    summary = []
    for name, group in frame.groupby("variant"):
        truth, prediction = group.truth_ttc.to_numpy(float), group.ttc.to_numpy(float)
        mid = observations(truth, prediction)["garl_mid"]
        warm = group.loc[group.groupby("sequence").cumcount() > 0]
        summary.append(
            {
                "variant": name,
                "n": len(group),
                "complete": len(group) == len(receipt["queries"]),
                "mean_total_ms": float(group.total_ms.mean()),
                "warm_median_ms": float(warm.total_ms.median()),
                "warm_p95_ms": float(warm.total_ms.quantile(0.95)),
                "mean_cpu_ms": float(group.cpu_ms.mean()),
                "mean_producer_ms": float(group.producer_ms.mean()),
                "mean_reused": float(group.reused.mean()),
                "mean_MiD": float(np.mean(mid)),
                "MAE_s": float(np.abs(prediction - truth).mean()),
                "signed_bias_s": float((prediction - truth).mean()),
            }
        )
    pd.DataFrame(summary).to_csv(output / "SUMMARY.csv", index=False)
    atomic_json(
        output / "REPORT.json",
        cast(
            dict,
            json_safe(
                {
                    "status": "COMPLETE_PILOT"
                    if len(records)
                    == (
                        len(receipt["policies"])
                        + len(receipt.get("baseline_variants", ["garl_event"]))
                    )
                    * len(receipt["queries"])
                    else "PARTIAL_PILOT",
                    "summary": summary,
                    "failures": failures,
                    "native_garl_envelopes_unwrapped": repairs,
                    "journal_sha256": digest(journal),
                    "source_sha256": digest(Path(__file__)),
                    "gpu_seconds": 0,
                    "predictions_recomputed": False,
                    "timing_caveat": (
                        "Concurrent dataset download then campaign CPU inference/preparation"
                    ),
                }
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
