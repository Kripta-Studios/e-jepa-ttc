"""Reproduce all packaged evaluation metrics without any original feature cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from operational.range_transfer.analyze import metrics
from operational.range_transfer.common import read, sha, write


def replay(campaign: Path) -> dict:
    results = []
    for pattern in ("evaluation/*/PREDICTIONS.csv", "*_evaluation/*/*/PREDICTIONS.csv"):
        for path in sorted(campaign.glob(pattern)):
            frame = pd.read_csv(
                path, converters={"target_ttc": float, "prediction_ttc": np.float32}
            )
            receipt = read(path.parent / "RECEIPT.json")
            if sha(path) != receipt["predictions_sha256"]:
                raise ValueError("Changed prediction table")
            actual = metrics(
                frame.target_ttc.to_numpy(np.float64),
                frame.prediction_ttc.to_numpy(np.float32),
                frame.group_id.to_numpy(str),
            )
            if actual != read(path.parent / "METRICS.json"):
                raise AssertionError("Metrics do not reproduce: " + str(path))
            results.append(
                {"table": str(path.relative_to(campaign)), "rows": len(frame), "exact": True}
            )
    probe = campaign / "group_probe"
    if (probe / "RESULTS.json").exists():
        folds = pd.read_csv(probe / "FOLDS.csv", converters={"phase_mae": float})
        for name, expected in read(probe / "RESULTS.json").items():
            path = probe / f"{name}_OOF.csv"
            frame = pd.read_csv(path, converters={"target_ttc": float, "prediction_ttc": float})
            actual = metrics(
                frame.target_ttc.to_numpy(np.float64),
                frame.prediction_ttc.to_numpy(np.float64),
                frame.group_id.to_numpy(str),
            )
            actual["macro_phase_mae"] = float(
                folds.loc[folds.representation == name, "phase_mae"].to_numpy().mean()
            )
            if actual != expected:
                raise AssertionError("OOF diagnostic metrics do not reproduce: " + name)
            results.append(
                {
                    "table": str(path.relative_to(campaign)),
                    "rows": len(frame),
                    "exact": True,
                    "scope": "H leave-group-out diagnostic; not H8 model evaluation",
                }
            )
    if not results:
        raise ValueError("No evaluation tables found")
    result = {
        "status": "PASS",
        "tables": results,
        "rows_including_reused_controls": sum(r["rows"] for r in results),
        "new_inference": False,
        "original_feature_caches_used": False,
    }
    write(campaign / "BUNDLE_METRIC_REPLAY.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    result = replay(parser.parse_args().campaign)
    print(f"{result['rows_including_reused_controls']} prediction rows reproduced exactly")
