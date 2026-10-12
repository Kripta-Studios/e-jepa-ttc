"""Independently audit the published author-interval CSV using the standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(output: Path) -> dict:
    """Reuse the independent scalar auditor and check exact prediction joins."""
    with (HERE / "SCORED_PREDICTIONS.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = json.loads((HERE / "RESULT.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((HERE / "SCORED_PREDICTIONS.csv").read_bytes()).hexdigest()
    if digest != result["scored_csv_sha256"]:
        raise ValueError("Scored prediction bytes changed")
    for entry in json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))["files"]:
        if hashlib.sha256((HERE / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Published evidence bytes changed: " + entry["path"])
    scalar = runpy.run_path(str(HERE.parent / "agent_handoff_20261012/replay.py"))
    models = list(result["metrics"]["own_available"])

    def labelled(row: dict[str, str]) -> bool:
        return bool(row["truth_ttc_seconds"]) and math.isfinite(float(row["truth_ttc_seconds"]))

    common = [
        r for r in rows if labelled(r) and all(r[m] and math.isfinite(float(r[m])) for m in models)
    ]
    scopes = {"own_available": rows, "common_all_models": common}
    scopes.update(
        {
            "sequence_" + s: [r for r in rows if r["scenario"] == s]
            for s in sorted({r["scenario"] for r in rows})
        }
    )
    checks = 0
    for scope, selected in scopes.items():
        for model in models:
            checks += scalar["compare"](
                scalar["metrics"](selected, model), result["metrics"][scope][model]
            )
    own_path = HERE / "evidence/V13_PREDICTIONS.csv"
    own_seal = json.loads((HERE / "evidence/V13_PREDICTIONS_SEALED.json").read_text())
    if hashlib.sha256(own_path.read_bytes()).hexdigest() != own_seal["predictions_sha256"]:
        raise ValueError("Own prediction seal differs")
    with own_path.open(encoding="utf-8", newline="") as handle:
        own = {r["query_id"]: r for r in csv.DictReader(handle)}
    base_seal = json.loads((HERE / "evidence/BASELINES_SEALED.json").read_text())
    baselines = {}
    for name, expected in base_seal["fragments"].items():
        p = HERE / "evidence" / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError("Baseline fragment changed")
        fragment = json.loads(p.read_text())
        baselines[fragment["query_id"]] = fragment["predictions"]
    if set(own) != set(baselines) or set(own) != {r["query_id"] for r in rows}:
        raise ValueError("Prediction populations differ")
    for row in rows:
        for model in models:
            source = own[row["query_id"]].get(model, baselines[row["query_id"]].get(model))
            if row[model] and (source is None or float(row[model]) != float(source)):
                raise ValueError("Scoring changed a prediction")
    receipt = {
        "status": "PASSED",
        "queries": len(rows),
        "scalar_metric_checks": checks,
        "prediction_join_checks": len(rows) * len(models),
        "optimizer_updates": 0,
        "scope": "CSV statistics and sealed prediction identity; no neural inference",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPLAY_RESULT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(main(parser.parse_args().output), indent=2))
