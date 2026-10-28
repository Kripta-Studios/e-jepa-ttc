"""Regenerate paired TRAIN40 transport tables from recorded case/binding files.

This reads evidence only. It does not load candidate models or access CUDA.
All repetitions are retained; separate descriptive views expose first updates
and subsequent repeats without replacing the complete latency distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def distribution(values: list[float]) -> dict[str, Any]:
    """Use linear empirical quantiles and preserve every measured sample."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Latency samples must be nonempty, finite and nonnegative")
    return {
        "n": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.5)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "under_20ms": int((array < 20).sum()),
        "quantile_method": "linear",
        "values_ms": values,
    }


def summarize(runs: list[Path]) -> dict[str, Any]:
    """Aggregate valid and failed cases without discarding adverse evidence."""
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    run_receipts = []
    source_hashes: dict[str, str] = {}
    source_drift = []
    configuration_drift = []
    comparison_configuration = None
    input_hashes = {}
    for folder in runs:
        paths = [folder / "binding.json", folder / "summary.json", folder / "startup.json"]
        binding, summary, startup = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        mode = "uint8" if binding["rgb_uint8"] else "PIL"
        configuration = {
            key: binding.get(key)
            for key in (
                "components",
                "manifest_sha256",
                "precision",
                "seed",
                "torch",
                "cuda",
                "gpu",
                "indices",
                "counts",
                "repeats",
                "representation_backend",
                "shared_rgb_images",
                "rgb_object_chunk_size",
                "parallel_preprocessing",
                "event_query_graphs",
                "event_normalization",
                "rgb_feature_cache",
                "reuse_event_encodings_override",
            )
        }
        if comparison_configuration is None:
            comparison_configuration = configuration
        elif configuration != comparison_configuration:
            configuration_drift.append(folder.name)
        cases = sorted(folder.glob("case_*.json"))
        observed = 0
        for path in [*paths, *cases]:
            input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in cases:
            for row in json.loads(path.read_text(encoding="utf-8"))["updates"]:
                admitted = (
                    row["parity"]["passed"]
                    and row["preserved_primary_parity"]["passed"]
                    and all(row["history_receipts_exact"])
                    and row["same_inputs_as_tensor_checked_repeat"]
                    and all(
                        check["event_passed"] and check["rgb_exact"] and check["rgb_delta_exact"]
                        for check in row["input_parity"]
                    )
                )
                groups[(mode, row["object_count"], row["index"])].append(
                    {
                        "run": folder.name,
                        "repetition": row["repetition"],
                        "ms": row["complete_wall_ms"],
                        "passed": admitted,
                        "first_update_for_count": row["repetition"] == 0
                        and row["index"] == binding["indices"][0],
                        "gpu_pixel_deltas_exact": all(
                            check["rgb_exact"] and check["rgb_delta_exact"]
                            for check in row["input_parity"]
                        ),
                    }
                )
                observed += 1
        for name, checksum in binding["sources"].items():
            if name in source_hashes and source_hashes[name] != checksum:
                source_drift.append({"source": name, "run": folder.name})
            source_hashes[name] = checksum
        run_receipts.append(
            {
                "run": folder.name,
                "mode": mode,
                "passed": summary["passed"] and observed == summary["expected_updates"],
                "updates": observed,
                "expected_updates": summary["expected_updates"],
                "preservation": summary["originals"],
                "startup": startup,
            }
        )
    rows = []
    for (mode, count, index), observations in sorted(groups.items()):
        first = [row["ms"] for row in observations if row["repetition"] == 0]
        later = [row["ms"] for row in observations if row["repetition"] > 0]
        rows.append(
            {
                "mode": mode,
                "objects": count,
                "index": index,
                "all": distribution([row["ms"] for row in observations]),
                "first_repeat_each_case": distribution(first) if first else None,
                "later_repeats": distribution(later) if later else None,
                "first_real_update_for_count_ms": [
                    row["ms"] for row in observations if row["first_update_for_count"]
                ],
                "parity_passed": all(row["passed"] for row in observations),
                "exact_rgb_and_delta": all(row["gpu_pixel_deltas_exact"] for row in observations),
                "observations": observations,
            }
        )
    return {
        "scope": "ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE",
        "run_order": [path.name for path in runs],
        "runs": run_receipts,
        "groups": rows,
        "input_evidence_sha256": input_hashes,
        "shared_source_drift": source_drift,
        "comparison_configuration": comparison_configuration,
        "configuration_drift": configuration_drift,
        "passed": bool(rows)
        and not source_drift
        and not configuration_drift
        and all(row["passed"] for row in run_receipts)
        and all(row["parity_passed"] for row in rows),
        "goal_50hz_under_20ms_demonstrated": False,
        "causal_speedup_admitted": False,
        "environment_limitations": [
            "Latency drift across ABBA; CPU contention and commit pressure were observed.",
            "Read-only host/GPU telemetry was sampled while the experiment was active.",
            "Synthetic priming, RAW disk loading and initial event history are outside the clock.",
            "Every complete-update sample is retained; no sustained arrival-rate test was run.",
        ],
    }


def markdown(result: dict[str, Any]) -> str:
    """Render the numerical table directly from the JSON aggregation."""
    lines = [
        "# TRAIN40: paired PIL / uint8, one and two objects",
        "",
        "All reported times are complete updates in milliseconds, including current ROI work, "
        "transfers, producers, context and all host TTC outputs. Sample counts appear in n; "
        "quantiles use linear interpolation. The second ROI is artificial, not an annotated track.",
        "",
        "| Mode | Objects | Index | n | Mean | Median | p95 | p99 | Max | Parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["groups"]:
        values = row["all"]
        lines.append(
            f"| {row['mode']} | {row['objects']} | {row['index']} | {values['n']} | "
            f"{values['mean_ms']:.3f} | {values['p50_ms']:.3f} | {values['p95_ms']:.3f} | "
            f"{values['p99_ms']:.3f} | {values['max_ms']:.3f} | "
            f"{'PASS' if row['parity_passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "The JSON retains every sample, first-repeat and later-repeat distributions, actual "
        "first updates per object count, startup receipts, and hashes of all input evidence.",
        "",
        "PIL ROI-first remains the conservative default. Uint8 is numerically admitted for "
        "this pilot but does not show a consistent latency benefit. Neither route demonstrates "
        "50 Hz or under 20 ms. ABBA timing drift and host commit pressure prevent a clean "
        "causal speedup claim; the independent parity checks remain valid.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write fresh JSON and Markdown evidence without running inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output / "REPORT.md").write_text(markdown(result), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(
            "Paired evidence failed a recorded parity, coverage or preservation gate"
        )


if __name__ == "__main__":
    main()
