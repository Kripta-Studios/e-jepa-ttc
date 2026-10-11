"""Post-seal compatibility for FCWD rows outside measured ground-truth support.

The historical comparator serializes missing TTC as an empty CSV cell. Treat
that cell as NaN, never zero. Retain the immutable inference source and outputs.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.fcwd_rgb_registration import score as original
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file


def normalize_missing_truth(rows: Iterator[dict[str, str]]) -> Iterator[dict[str, str]]:
    """Only normalize explicitly empty TTC cells; malformed numeric text still fails."""
    for row in rows:
        if "truth_ttc_seconds" in row and row["truth_ttc_seconds"] == "":
            row = {**row, "truth_ttc_seconds": "nan"}
        yield row


def score(predictions: Path, comparator: Path) -> dict[str, Any]:
    """Reuse all original scoring gates, changing only missing-label decoding."""
    seal_path = predictions / "PREDICTIONS_SEALED.json"
    seal_before = sha256_file(seal_path)
    csv_before = sha256_file(predictions / "PREDICTIONS.csv")
    native_reader = csv.DictReader

    def reader(handle: Iterable[str]) -> Iterator[dict[str, str]]:
        return normalize_missing_truth(native_reader(handle))

    with patch.object(original.csv, "DictReader", reader):
        result = original.score(predictions, comparator)
    if (
        sha256_file(seal_path) != seal_before
        or sha256_file(predictions / "PREDICTIONS.csv") != csv_before
    ):
        raise ValueError("Scoring changed sealed predictions")
    result["missing_truth_decode"] = (
        "empty_CSV_cell_to_NaN; no_zero_imputation; all_queries_retained"
    )
    result["scoring_adapter_sha256"] = sha256_file(Path(__file__))
    result["unlabelled_queries"] = (
        result["population"] - result["views"]["all_queries"]["labelled_rows"]
    )
    atomic_write_json(predictions / "RESULT.json", result)
    state = predictions.parent / "STATE.json"
    previous = read_json_shared(state) if state.exists() else {}
    atomic_write_json(
        predictions / "SCORING_RECOVERY.json",
        {
            "status": "COMPLETE",
            "previous_state": previous,
            "prediction_seal_sha256_unchanged": seal_before,
            "predictions_sha256_unchanged": csv_before,
            "scoring_adapter_sha256": result["scoring_adapter_sha256"],
            "unlabelled_queries": result["unlabelled_queries"],
            "optimizer_updates": 0,
        },
    )
    atomic_write_json(
        state, {"status": "COMPLETE", "scoring_adapter": "operational.fcwd_rgb_score_fix"}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    args = parser.parse_args()
    score(args.predictions, args.comparator)


if __name__ == "__main__":
    main()
