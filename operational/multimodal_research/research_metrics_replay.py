"""Recompute persisted development/transfer metrics and verify all new fit endpoints offline."""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from operational.multimodal_validation.common import ROOT, read, sha, write
from operational.rgb_port.evaluate import score_signed_ttc

from .common import OUT


def compare(actual: object, expected: object, path: str, counts: dict) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key, value in expected.items():
            compare(actual[key], value, path + "/" + key, counts)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, path + "/" + str(i), counts)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float, np.integer, np.floating)), path
        assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-7), (
            path,
            actual,
            expected,
        )
        counts["scalar_checks"] += 1
    else:
        assert actual == expected, (path, actual, expected)


def main() -> None:
    counts = dict(scalar_checks=0, internal_tables=0, external_tables=0, verified_fits=0)
    sources = {}
    for receipt_path in sorted((OUT / "fits").glob("*/CHECKPOINT_RECEIPT.json")):
        receipt = read(receipt_path)
        journal = read(receipt_path.parent / "UPDATE_JOURNAL.json")
        assert receipt["status"] == "COMPLETE" and receipt["completed_updates"] == 2500
        assert journal["completed_updates"] == 2500 and journal["pending_update_upper"] == 0
        assert receipt["checkpoint_sha256"] == sha(receipt_path.parent / "checkpoint_last.pt")
        sources[str(receipt_path.relative_to(OUT))] = sha(receipt_path)
        counts["verified_fits"] += 1
    paths = [OUT / "V_RESULTS.json", *sorted((OUT / "studies").rglob("V_RESULTS.json"))]
    for path in paths:
        expected = read(path)["scores"]
        csv_path = path.parent / "V_SCORED_PREDICTIONS.csv"
        frame = pd.read_csv(csv_path, float_precision="round_trip")
        assert len(frame) == 16432 and not frame.sample_token.duplicated().any()
        for method, score in expected.items():
            values = frame[method].to_numpy()
            if method.startswith(("R_ANCHOR_", "F_ANCHOR_")) or method == "R_CTX_SAME_CACHE":
                if "PHASE_ENSEMBLE" not in str(path):
                    values = values.astype(np.float32)
            actual = score_signed_ttc(
                frame.target_ttc.to_numpy(), values, frame.group_id.to_numpy()
            )
            compare(actual, score, str(path) + "/" + method, counts)
        sources[str(path.relative_to(OUT))] = sha(path)
        sources[str(csv_path.relative_to(OUT))] = sha(csv_path)
        counts["internal_tables"] += 1
    sys.path.insert(0, str(ROOT / "docs/multimodal_next_handoff_20261012/scripts"))
    from metrics_core import summarize

    for path in sorted(OUT.rglob("METRICS.csv")):
        predictions = path.parent / "SCORED_PREDICTIONS.csv"
        if not predictions.exists():
            continue
        frame = pd.read_csv(predictions, float_precision="round_trip")
        metrics = pd.read_csv(path, float_precision="round_trip")
        for row in metrics.to_dict("records"):
            common = frame.rgb_available.to_numpy(bool, copy=True)
            if "garl_full_available" in frame:
                common &= frame.garl_full_available.to_numpy(bool)
            elif "public_Garl_rgb_event_full" in frame:
                common &= frame.public_Garl_rgb_event_full.notna().to_numpy()
            mask = common if row["scope"] == "common_support" else np.ones(len(frame), bool)
            truth = frame.truth_ttc_seconds.to_numpy()
            band = row["band"]
            if band != "all":
                lo, hi = {
                    "crucial": (0, 3),
                    "small": (3, 6),
                    "large": (6, 10),
                    "gt10": (10, np.inf),
                }[band]
                mask &= (truth > lo) & (truth <= hi)
            assert int(mask.sum()) == row["queries"]
            actual = summarize(zip(truth[mask], frame[row["method"]].to_numpy()[mask], strict=True))
            for name, value in actual.items():
                recorded = row[name]
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    assert pd.isna(recorded)
                else:
                    compare(
                        value, recorded, str(path) + "/" + str(row["method"]) + "/" + name, counts
                    )
        sources[str(path.relative_to(OUT))] = sha(path)
        sources[str(predictions.relative_to(OUT))] = sha(predictions)
        counts["external_tables"] += 1
    write(
        OUT / "METRICS_REPLAY_VERIFICATION.json",
        dict(
            utc=datetime.now(UTC).isoformat(),
            status="PASS",
            counts=counts,
            sources=sources,
            source_sha256=sha(__file__),
            optimizer_updates=0,
            test_labels_read=False,
            scope="Offline recomputation from published prediction CSVs; no checkpoint inference. "
            "FP32 prediction columns are restored to their declared native dtype; "
            "phase-ensemble predictions retain FP64. Numeric tolerance abs1e-7/rel1e-9.",
        ),
    )
    print(counts, flush=True)


if __name__ == "__main__":
    main()
