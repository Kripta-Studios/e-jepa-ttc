"""Score all sealed Dev32 queries while measuring only rows with public GT."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from operational.fcwd_rgb_score_fix import normalize_missing_truth
from operational.rgb_port import transfer
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port.evaluate import common_cap60_metrics, score_signed_ttc


def labelled_arrays(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select on target availability only; retain every failed model prediction."""
    valid = np.isfinite(truth)
    if truth.shape != prediction.shape or groups.shape != truth.shape:
        raise ValueError("Metric arrays differ in shape")
    if not valid.any():
        raise ValueError("No public TTC labels available for scoring")
    return truth[valid], prediction[valid], groups[valid]


def score(seal: Path, comparator: Path, output: Path) -> dict[str, Any]:
    """Keep the original population join and coverage accounting; decode missing GT."""
    native_reader = csv.DictReader

    def reader(handle: Iterable[str]) -> Iterator[dict[str, str]]:
        return normalize_missing_truth(native_reader(handle))

    def signed(truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
        return score_signed_ttc(*labelled_arrays(truth, prediction, groups))

    def capped(truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
        return common_cap60_metrics(*labelled_arrays(truth, prediction, groups))

    seal_before = sha256_file(seal)
    with (
        patch.object(transfer.csv, "DictReader", reader),
        patch.object(transfer, "score_signed_ttc", signed),
        patch.object(transfer, "common_cap60_metrics", capped),
    ):
        result = transfer.score_dev32(seal_path=seal, comparator_csv=comparator, output=output)
    if sha256_file(seal) != seal_before:
        raise ValueError("Scoring changed prediction seal")
    labelled = result["metrics"]["E_CTX_MATCHED"]["legacy_native"]["labeled_queries"]
    result["row_accounting"].update(
        labelled_queries=labelled,
        unlabelled_queries=result["row_accounting"]["population"] - labelled,
        missing_truth_policy="empty_to_NaN; retained_in_CSV; omitted_only_from_target_metrics",
    )
    result["scoring_adapter_sha256"] = sha256_file(Path(__file__))
    result["missing_truth_parser_sha256"] = sha256_file(Path("operational/fcwd_rgb_score_fix.py"))
    atomic_write_json(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.seal, args.comparator, args.output)


if __name__ == "__main__":
    main()
