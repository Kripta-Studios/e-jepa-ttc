"""Independent post-run Garl competition audit; never executes a model or reads TTC labels."""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/verify_evttc_competition_run.py"
spec = importlib.util.spec_from_file_location("independent_phase_audit_helpers", HELPER_PATH)
assert spec is not None and spec.loader is not None
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
require = helpers.require
finite = helpers.finite
read_json = helpers.read_json
file_sha = helpers.file_sha
canonical_sha = helpers.canonical_sha
describe = helpers.describe
SEQUENCES = helpers.SEQUENCES
VARIANTS = ("full", "event", "visual")


def event_plan(query: int, start: int) -> list[list[int]] | None:
    available = query - start
    if available < 2:
        return None
    if available >= 200_000:
        return [[query - 200_000, query - 100_000], [query - 100_000, query]]
    middle = start + available // 2
    return [[start, middle], [middle, query]]


def rgb_plan(clock: Sequence[int], query: int) -> list[int] | None:
    current = bisect.bisect_right(clock, query) - 1
    if current < 1:
        return None
    target = clock[current] - 100_000
    previous = min(range(current), key=lambda index: (abs(clock[index] - target), index))
    return [previous, current]


def square(boxes: Sequence[Sequence[float]], index: int) -> list[int]:
    integer_boxes = [[int(value) for value in box] for box in boxes]
    edge = max(max(b[2] - b[0], b[3] - b[1]) for b in integer_boxes)
    b = integer_boxes[index]
    cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
    return [
        math.ceil(cx - edge / 2),
        math.ceil(cy - edge / 2),
        math.ceil(cx + edge / 2),
        math.ceil(cy + edge / 2),
    ]


def inspect_prediction(row: Mapping[str, Any], delta: float | None) -> None:
    prediction = row["prediction"]
    if prediction is None:
        require(
            row["status"] == "unavailable" and row["prediction_ttc_seconds"] is None,
            "Unexecuted model must remain unavailable/null",
        )
        require(row["model_wall_seconds"] == 0, "Unexecuted model has nonzero model cost")
        return
    require(delta is not None and delta > 0, "Executed model lacks actual positive cadence")
    require(prediction["variant"] == row["variant"], "Prediction variant changed")
    require(prediction["native_delta_t_s"] == 0.1, "Native graph dT changed")
    require(prediction["conversion_delta_t_s"] == delta, "Conversion uses wrong actual cadence")
    require(
        prediction["status"] == row["status"] and prediction["reason"] == row["reason"],
        "Prediction status/reason differs from ledger",
    )
    require(prediction["model_ttc"] == row["prediction_ttc_seconds"], "Prediction value differs")
    heights = prediction["heights"]
    require(len(heights) == 2, "Expected exactly two raw heights")
    special = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}
    values = []
    for value in heights:
        values.append(
            special[value]
            if isinstance(value, str) and value in special
            else finite(value, "raw height")
        )
    diagnostics = prediction["diagnostics"]
    require(
        [diagnostics["height_t0"], diagnostics["height_t1"]] == heights,
        "Raw height diagnostics disagree",
    )
    h = np.asarray(values, dtype=np.float32)
    if not np.isfinite(h).all() or h[1] == 0:
        require(
            row["status"] == "unavailable"
            and row["prediction_ttc_seconds"] is None
            and row["reason"] == "nonfinite_or_zero_height_output",
            "Nonfinite/zero second height must remain unavailable",
        )
        return
    with np.errstate(all="ignore"):
        denominator = np.float32(1) - np.float32(h[0] / h[1])
        converted = np.float32(np.float32(delta) / denominator)
    require(
        float(denominator) == diagnostics["height_ratio_denominator"],
        "Height-ratio denominator differs from native FP32 arithmetic",
    )
    if np.isfinite(converted):
        require(
            row["status"] == "ok" and row["reason"] == "available",
            "Finite native conversion was dropped",
        )
        require(
            float(converted) == finite(row["prediction_ttc_seconds"], "signed TTC"),
            "Signed TTC was clipped, changed or converted using wrong cadence",
        )
    else:
        require(
            row["status"] == "unavailable"
            and row["prediction_ttc_seconds"] is None
            and row["reason"] == "nonfinite_height_ratio_ttc",
            "Singular native conversion must remain unavailable/null",
        )
        expected = "nan" if np.isnan(converted) else "+inf" if converted > 0 else "-inf"
        require(diagnostics.get("raw_ttc") == expected, "Nonfinite TTC diagnostic changed")


def inspect_temporal(
    row: Mapping[str, Any],
    clock: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> dict:
    d = row["input_details"]
    variant = row["variant"]
    q = row["identity"]["anchor_us"]
    start, end = clock["event_start_us"], clock["event_end_us"]
    require(
        d["variant"] == variant and d["query_us"] == q and d["available_us"] == q,
        "Input identity/availability mismatch",
    )
    require(
        d["native_timevolume_range_s"] == 0.1
        and d["native_timevolume_bins"] == 20
        and d["native_bin_width_s"] == 0.005
        and d["polarity_consumed"] is False,
        "Native event representation metadata changed",
    )
    result = {
        "max_span_us": 0,
        "bbox_fallbacks": 0,
        "zero_roi_windows": 0,
        "observed_prefix": False,
        "variable_cadence": False,
        "full_gap_us": None,
    }
    windows = None
    if variant == "event":
        windows = event_plan(q, start)
        if windows is None or q > end:
            require(
                row["prediction"] is None and row["reason"] == "insufficient_real_event_history",
                "Absent event coverage was fabricated",
            )
            inspect_prediction(row, None)
            return result
        endpoints = [w[1] for w in windows]
        prefix = q - start < 200_000
    else:
        indices = rgb_plan(clock["rgb_us"], q)
        if indices is None:
            require(
                row["prediction"] is None
                and row["reason"] == "fewer_than_two_real_causal_rgb_frames",
                "Missing RGB history was fabricated",
            )
            require(
                "bbox_sources" not in d and "rgb" not in d, "Unexpected missing-RGB preparation"
            )
            inspect_prediction(row, None)
            return result
        endpoints = [clock["rgb_us"][i] for i in indices]
        require(
            d["rgb"]["frame_indices"] == indices and d["rgb"]["timestamps_us"] == endpoints,
            "RGB pair is not the two distinct causal native-selected real frames",
        )
        prefix = False
        if variant == "full":
            spacing = endpoints[1] - endpoints[0]
            support = min(spacing, 100_000)
            windows = [[max(start, t - support), t] for t in endpoints]
            prefix = any(b - a != support for a, b in windows)
            require(
                d["event_support_cap_us"] == 100_000
                and d["event_support_us"] == support
                and d["actual_rgb_delta_minus_event_support_us"] == spacing - support
                and d["event_support_ends_at_each_rgb_endpoint"] is True,
                "Full event support cap/real RGB gap changed",
            )
            result["full_gap_us"] = spacing - support
    require(endpoints[0] < endpoints[1] <= q, "Future/unordered endpoint")
    delta = (endpoints[1] - endpoints[0]) / 1e6
    require(delta > 0, "Nonpositive actual endpoint delta")
    sources = d["bbox_sources"]
    require(len(sources) == 2, "Wrong bbox-source count")
    causal = [r for r in identities if r["anchor_us"] <= q]
    event_boxes, rgb_boxes = [], []
    width, height = calibration["event_resolution"]
    for endpoint, record in zip(endpoints, sources, strict=True):
        eligible = [r for r in causal if r["anchor_us"] <= endpoint]
        expected = eligible[-1] if eligible else causal[0]
        fallback = not eligible
        require(
            record["endpoint_us"] == endpoint
            and record["row_index"] == expected["row_index"]
            and record["index"] == expected["index"]
            and record["anchor_us"] == expected["anchor_us"],
            "Wrong or future bbox source identity",
        )
        require(record["raw_box"] == [float(x) for x in expected["box_text"]], "Raw bbox changed")
        require(
            record["fallback"] is fallback
            and record["selection"]
            == ("earliest_query_causal_fallback" if fallback else "latest_at_or_before_endpoint")
            and record["available_us"] == (q if fallback else expected["anchor_us"]),
            "BBox fallback availability/label is not truthful",
        )
        require(record["available_us"] <= q, "Future bbox availability")
        raw = record["raw_box"]
        bounded = [
            min(max(raw[0], 0), width),
            min(max(raw[1], 0), height),
            min(max(raw[2], 0), width),
            min(max(raw[3], 0), height),
        ]
        require(record["event_box"] == bounded, "Event-plane bbox normalization differs")
        for key in ("event_box", "rgb_box"):
            box = record[key]
            require(
                len(box) == 4
                and all(math.isfinite(v) for v in box)
                and box[2] > box[0]
                and box[3] > box[1],
                "Malformed projected bbox",
            )
        event_boxes.append(record["event_box"])
        rgb_boxes.append(record["rgb_box"])
        result["bbox_fallbacks"] += int(fallback)
    require(
        d["same_box_used_for_both_endpoints"]
        is (sources[0]["row_index"] == sources[1]["row_index"]),
        "Same-bbox diagnostic changed",
    )
    if variant != "event":
        require(
            d["rgb"]["square_xyxy"] == square(rgb_boxes, 1), "Native RGB common-square ROI changed"
        )
    if windows is not None:
        recorded = d["event_windows"]
        require(len(recorded) <= 2, "Too many event windows")
        for i, item in enumerate(recorded):
            a, b = windows[i]
            require(
                item["window_us"] == [a, b] and item["duration_us"] == b - a,
                "Window is not exact independently reconstructed support",
            )
            require(
                start <= a < b <= end and b <= q and b - a <= 100_000,
                "Future/out-of-recording/over-100ms event support",
            )
            require(
                item["square_xyxy"] == square(event_boxes, i), "Native event square ROI changed"
            )
            raw_count, roi_count = item["raw_event_count"], item["roi_event_count"]
            require(
                isinstance(raw_count, int)
                and raw_count > 0
                and isinstance(roi_count, int)
                and 0 <= roi_count <= raw_count,
                "Invalid raw/ROI event counts",
            )
            require(item["roi_empty"] is (roi_count == 0), "ROI-empty diagnostic changed")
            result["zero_roi_windows"] += int(roi_count == 0)
        if len(recorded) != 2:
            require(
                row["prediction"] is None
                and row["reason"]
                in {"raw_event_window_empty", "event_window_out_of_recorded_coverage"},
                "Incomplete preparation must remain explicitly unavailable",
            )
            if row["reason"] == "event_window_out_of_recorded_coverage":
                a, b = windows[len(recorded)]
                require(a < start or b > end or b <= a, "Claimed coverage failure is false")
            inspect_prediction(row, None)
            return result
    require(d["observed_prefix_adaptation"] is prefix, "Observed-prefix diagnostic changed")
    require(
        d["endpoint_timestamps_us"] == endpoints
        and d["conversion_delta_t_s"] == delta
        and d["native_training_delta_t_s"] == 0.1,
        "Native/actual cadence mismatch",
    )
    variable = not math.isclose(delta, 0.1, rel_tol=0, abs_tol=1e-9)
    require(d["variable_cadence_transfer"] is variable, "Variable-cadence diagnostic changed")
    require(
        d["sensor_shape"] == [{"full": 46, "event": 40, "visual": 6}[variant], 128, 128],
        "Sensor shape mismatch",
    )
    require(
        len(d["sensor_sha256"]) == 64 and all(c in "0123456789abcdef" for c in d["sensor_sha256"]),
        "Missing prepared tensor byte hash",
    )
    first = min(endpoints[0], *(w[0] for w in windows)) if windows else endpoints[0]
    result.update(max_span_us=q - first, observed_prefix=prefix, variable_cadence=variable)
    inspect_prediction(row, delta)
    return result


def verify_integrity(freeze: Mapping[str, Any], data_root: Path) -> dict:
    checked = {}

    def check(path: Path, expected: str, size: int | None = None) -> None:
        resolved = path.resolve(strict=True)
        if size is not None:
            require(resolved.stat().st_size == size, f"File size changed: {resolved}")
        key = str(resolved)
        if key not in checked:
            checked[key] = file_sha(resolved)
        require(checked[key] == expected, f"Frozen SHA changed: {resolved}")

    reference = freeze["phase_reference"]
    phase_root = Path(reference["path_resolved"])
    for name, digest in reference["artifact_sha256"].items():
        check(helpers.inside(phase_root, name), digest)
    phase_audit = read_json(phase_root / "independent_audit.json")
    for name, digest in phase_audit["artifact_sha256"].items():
        check(helpers.inside(phase_root, name), digest)
    phase_freeze = read_json(phase_root / "RUN_FREEZE.json")
    for key in ("query_manifest", "raw_sources", "calibrations"):
        require(freeze[key] == phase_freeze[key], f"Garl/PHASE input mismatch: {key}")
    raw_count, raw_bytes = 0, 0
    for sequence, entry in freeze["raw_sources"].items():
        for name, record in entry["files"].items():
            require(Path(name).name == name, "Raw filename is not local")
            check(
                helpers.inside(data_root, f"{sequence}/{name}"), record["sha256"], record["bytes"]
            )
            raw_count += 1
            raw_bytes += record["bytes"]
    require(raw_count == 22, "Wrong raw file population")
    for relative, digest in freeze["executed_source_sha256"].items():
        check(helpers.inside(ROOT, relative), digest)
    phase_sources = dict(phase_freeze["executed_source_sha256"])
    phase_sources.update(phase_freeze["frozen_runtime"]["source_sha256"])
    for relative, digest in phase_sources.items():
        check(helpers.inside(ROOT, relative), digest)
    inventory = freeze["inventory"]
    for record in inventory["upstream"]["native_import_graph"]:
        check(Path(record["path"]), record["sha256"], record["bytes"])
    for variant in VARIANTS:
        item = inventory["variants"][variant]
        for name in ("checkpoint", "config"):
            record = item[name]
            check(Path(record["path"]), record["sha256"], record["bytes"])
        binding = freeze["model_bindings"][variant]
        require(
            binding["strict_state_load"] is True and binding["native_delta_t_s"] == 0.1,
            "Model strict load/native cadence not retained",
        )
        for key in ("checkpoint", "config"):
            require(
                binding[f"{key}_sha256"] == item[key]["sha256"]
                and Path(binding[f"{key}_path"]).resolve() == Path(item[key]["path"]).resolve(),
                "Model binding differs from official inventory",
            )
    inventory_path = ROOT / "tmp/garl_evttc_readiness/inventory.json"
    check(inventory_path, freeze["inventory_sha256"])
    require(read_json(inventory_path) == inventory, "Embedded inventory differs")
    gt = freeze["ground_truth_identity_audit"]
    check(Path(gt["path_resolved"]), gt["sha256"])
    require(gt["status"] == "NO_VERIFIED_EXACT_QUERY_GROUND_TRUTH", "GT status changed")
    return {
        "raw_files": raw_count,
        "raw_bytes": raw_bytes,
        "garl_native_files": 5,
        "official_checkpoints": 3,
        "official_configs": 3,
        "phase_artifacts_unchanged": True,
        "phase_source_files_unchanged": len(phase_sources),
        "calibrations_identical_to_phase": True,
        "sha256": checked,
    }


def variant_stats(rows: Sequence[Mapping[str, Any]], temporal: Sequence[Mapping[str, Any]]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    return {
        "rows": len(rows),
        "available": len(ok),
        "unavailable": len(rows) - len(ok),
        "unavailable_reasons": dict(Counter(r["reason"] for r in rows if r["status"] != "ok")),
        "signed_ttc_seconds": describe([r["prediction_ttc_seconds"] for r in ok]),
        "negative_ttc_rows": sum(r["prediction_ttc_seconds"] < 0 for r in ok),
        "zero_ttc_rows": sum(r["prediction_ttc_seconds"] == 0 for r in ok),
        "runtime_seconds": {
            key: describe([r[key] for r in rows])
            for key in ("preprocess_seconds", "model_wall_seconds", "cost_time_seconds")
        },
        "max_sensor_span_us": max((t["max_span_us"] for t in temporal), default=0),
        "bbox_fallback_uses": sum(t["bbox_fallbacks"] for t in temporal),
        "zero_roi_windows": sum(t["zero_roi_windows"] for t in temporal),
        "prefix_rows": sum(t["observed_prefix"] for t in temporal),
        "variable_cadence_rows": sum(t["variable_cadence"] for t in temporal),
        "full_rgb_gap_beyond_event_cap_us": describe(
            [t["full_gap_us"] for t in temporal if t["full_gap_us"] is not None]
        ),
    }


def audit(run: Path, data_root: Path) -> dict:
    started = time.perf_counter()
    run = run.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    summary = read_json(run / "summary.json")
    require(
        summary["status"] in {"COMPLETE", "PARTIAL"}
        and summary["terminal_population_accounted"] is True,
        "Run is not terminal",
    )
    require(
        summary["expected_records"] == 1302 and summary["recorded_records"] == 1302,
        "Run does not account for all 1302 records",
    )
    require(summary["official_accuracy_evaluated"] is False, "Unexpected accuracy claim")
    freeze = read_json(run / "RUN_FREEZE.json")
    freeze_sha = file_sha(run / "RUN_FREEZE.json")
    require(summary["freeze_sha256"] == freeze_sha, "Summary freeze hash mismatch")
    require(summary["ledger_sha256"] == file_sha(run / "all_rows.jsonl"), "Ledger SHA differs")
    require(
        freeze["variants"] == list(VARIANTS)
        and freeze["policy"]["version"] == "garl_evttc_competition_transfer_v1",
        "Wrong variant population/policy",
    )
    require(
        freeze["targets_read"] is False
        and freeze["training_updates"] == 0
        and freeze["checkpoint_selection_from_test"] is False,
        "Blind frozen contract changed",
    )
    require(
        freeze["timing"]["cross_variant_latency_ranking_supported"] is False,
        "Measured run cannot support fair cross-variant latency ranking",
    )
    clocks = {s: helpers.sensor_clocks(data_root / s) for s in SEQUENCES}
    identities = {
        s: helpers.original_queries(data_root / s, clocks[s]["rgb_us"]) for s in SEQUENCES
    }
    require(sum(map(len, identities.values())) == 434, "Wrong CSV population")
    require(
        identities == freeze["query_manifest"]
        and canonical_sha(identities) == freeze["query_manifest_sha256"],
        "Freeze/raw identities differ",
    )
    rows = [
        json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        for line in (run / "all_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    require(len(rows) == 1302, "Wrong ledger length")
    expected = [(q, v) for s in SEQUENCES for q in identities[s] for v in VARIANTS]
    temporal = []
    for row, (identity, variant) in zip(rows, expected, strict=True):
        require(
            row["schema"] == "garl_evttc_competition_row_v1" and row["freeze_sha256"] == freeze_sha,
            "Row freeze/schema mismatch",
        )
        require(
            row["identity"] == identity and row["variant"] == variant,
            "Dropped/reordered/foreign identity",
        )
        require(row["status"] in {"ok", "unavailable"}, "Nonterminal row status")
        if row["status"] == "ok":
            finite(row["prediction_ttc_seconds"], "TTC")
        else:
            require(row["prediction_ttc_seconds"] is None, "Unavailable TTC must be null")
        costs = [
            finite(row[k], k)
            for k in ("preprocess_seconds", "model_wall_seconds", "cost_time_seconds")
        ]
        require(
            all(v >= 0 for v in costs) and costs[2] == costs[0] + costs[1], "Runtime sum mismatch"
        )
        s = identity["sequence_id"]
        temporal.append(inspect_temporal(row, clocks[s], identities[s], freeze["calibrations"][s]))
    require(
        not (run / "errors.jsonl").exists() or (run / "errors.jsonl").stat().st_size == 0,
        "Terminal run contains execution errors",
    )
    variants = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        ts = [t for r, t in zip(rows, temporal, strict=True) if r["variant"] == variant]
        entry = summary["variants"][variant]
        require(read_json(run / variant / "summary.json") == entry, "Variant summary differs")
        available = sum(r["status"] == "ok" for r in subset)
        partial = available != 434
        require(
            entry["recorded_rows"] == 434
            and entry["ok_rows"] == available
            and entry["null_rows"] == 434 - available
            and entry["status"] == ("PARTIAL" if partial else "COMPLETE"),
            "Variant coverage mismatch",
        )
        key = "partial_results_zip" if partial else "results_zip"
        basename = "partial_results.zip" if partial else "results.zip"
        require(entry[key] == f"{variant}/{basename}", "Wrong variant ZIP path")
        normalized = dict(entry)
        normalized[key] = basename
        if partial:
            normalized.update(
                partial_results="partial_results",
                partial_available_rows=available,
                partial_unavailable_rows=434 - available,
            )
        by_sequence = {
            s: [r for r in subset if r["identity"]["sequence_id"] == s] for s in SEQUENCES
        }
        package = helpers.verify_package(
            run / variant, normalized, normalized, by_sequence, partial=partial
        )
        variants[variant] = {
            "package": package,
            "aggregate": variant_stats(subset, ts),
            "per_sequence": {
                s: variant_stats(
                    [r for r in subset if r["identity"]["sequence_id"] == s],
                    [
                        t
                        for r, t in zip(subset, ts, strict=True)
                        if r["identity"]["sequence_id"] == s
                    ],
                )
                for s in SEQUENCES
            },
        }
    all_complete = all(v["package"]["submission_complete"] for v in variants.values())
    require(
        summary["status"] == ("COMPLETE" if all_complete else "PARTIAL"),
        "Overall completeness mismatch",
    )
    integrity = verify_integrity(freeze, data_root)
    return {
        "schema": "garl_evttc_independent_audit_v1",
        "audit_status": "PASS",
        "run_status": summary["status"],
        "status": "PASS" if all_complete else "PARTIAL",
        "records_verified": 1302,
        "queries_per_variant": 434,
        "variants": variants,
        "integrity": integrity,
        "audited_utc": datetime.now(UTC).isoformat(),
        "timing_scope": freeze["timing"],
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_sha256": {
            name: file_sha(run / name)
            for name in ("RUN_FREEZE.json", "all_rows.jsonl", "summary.json")
        },
        "verifier_sha256": file_sha(Path(__file__)),
        "helper_sha256": file_sha(HELPER_PATH),
        "limitations": [
            "No model execution, TTC labels, PDF digitization, accuracy metric or winner "
            "selection.",
            "Raw sensor clocks and frozen bytes independently checked; tensor values, "
            "counts and geometric projection are not recomputed.",
            "Partial variants retain every unavailable identity and are not complete "
            "official submissions.",
            "Actual-delta transfer, observed-prefix windows and bbox fallbacks are "
            "explicit adaptations, not a reproduction of the paper protocol.",
            "Per-query timings include preprocessing and synchronized model execution but "
            "exclude initialization and share cache/GPU warm state; "
            "they are not a controlled latency ranking.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.run, args.data_root)
    except (helpers.AuditError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {
            "schema": "garl_evttc_independent_audit_v1",
            "audit_status": "FAIL",
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    output = args.output or args.run / "independent_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "audit_status", "run_status", "records_verified", "error")
                if k in result
            }
        )
    )
    return 0 if result["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
