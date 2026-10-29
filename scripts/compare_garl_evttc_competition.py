"""Independent label-blind comparison of terminal Garl and PHASE EvTTC runs."""

# ruff: noqa: E402 -- bind this checkout before importing plotting dependencies

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

SEQUENCES = (
    "CCRs1-low",
    "CCRs1-medium",
    "CCRs1-high",
    "CCRs2-low",
    "CCRs2-medium",
    "CCRs2-high",
    "CCRm-low",
    "CCRm-medium",
    "Slider-750",
    "Slider-1000",
)
VARIANTS = ("full", "event", "visual")
PHASE_ROWS = 434
GARL_ROWS = PHASE_ROWS * len(VARIANTS)
EXPECTED_PHASE_RUN = ROOT / "output" / "evttc_competition_20261028_v2"
LITERATURE_RECEIPT = ROOT / "tmp" / "garl_evttc_literature" / "literature_receipt.json"
PHASE_ARTIFACTS = (
    "RUN_FREEZE.json",
    "all_rows.jsonl",
    "summary.json",
    "independent_audit.json",
    "results.zip",
)
INK, MUTED, GRID, LIGHT = "#172335", "#667386", "#D5DCE7", "#F5F7FA"
COLORS = {"phase": "#268B86", "full": "#3769DB", "event": "#AD534B", "visual": "#A88B32"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "pdf.fonttype": 42,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": LIGHT,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.7,
        "legend.frameon": False,
    }
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def identity_key(identity: Mapping[str, Any]) -> tuple[str, int]:
    return str(identity["sequence_id"]), int(identity["row_index"])


def validate_terminal_summary(run: Path) -> dict[str, Any]:
    """Read only the summary until the run proves that all records are terminal."""
    summary = read_json(run / "summary.json")
    if summary.get("status") not in {"COMPLETE", "PARTIAL"}:
        raise ValueError("Garl run is not terminal")
    if summary.get("terminal_population_accounted") is not True:
        raise ValueError("Garl terminal population is not fully accounted")
    if summary.get("expected_records") != GARL_ROWS or summary.get("recorded_records") != GARL_ROWS:
        raise ValueError("Garl terminal run must account for exactly 1,302 records")
    if sha256_file(run / "RUN_FREEZE.json") != summary.get("freeze_sha256"):
        raise ValueError("Garl summary freeze hash differs")
    if sha256_file(run / "all_rows.jsonl") != summary.get("ledger_sha256"):
        raise ValueError("Garl summary ledger hash differs")
    variants = summary.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise ValueError("Garl summary must contain full, event, and visual variants")
    for variant in VARIANTS:
        item = variants[variant]
        count = item.get("count", item.get("recorded_rows"))
        if count != PHASE_ROWS:
            raise ValueError(f"Variant {variant} does not account for 434 rows")
        artifact = item.get("results_zip") or item.get("partial_results_zip")
        digest = item.get("results_zip_sha256") or item.get("partial_results_zip_sha256")
        if not artifact or not digest:
            raise ValueError(f"Variant {variant} lacks a terminal result artifact binding")
        path = run / str(artifact)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Variant {variant} result artifact hash differs")
    total_nulls = sum(
        int(variants[variant].get("nulls", variants[variant].get("null_rows", -1)))
        for variant in VARIANTS
    )
    if total_nulls < 0:
        raise ValueError("Garl variant summaries must declare null counts")
    if (summary["status"] == "COMPLETE") != (total_nulls == 0):
        raise ValueError("Garl global status disagrees with declared variant null counts")
    return summary


def validate_phase_reference(
    run: Path, freeze: Mapping[str, Any], expected_phase: Path
) -> tuple[Path, dict[str, Any]]:
    reference = freeze.get("phase_reference")
    if not isinstance(reference, dict):
        raise ValueError("Garl freeze lacks phase_reference")
    phase = Path(str(reference.get("path_resolved", ""))).resolve(strict=True)
    if phase != expected_phase.resolve(strict=True):
        raise ValueError("Garl freeze does not reference the required immutable PHASE run")
    hashes = reference.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(PHASE_ARTIFACTS):
        raise ValueError("PHASE reference must bind all five terminal artifacts")
    for name in PHASE_ARTIFACTS:
        path = phase / name
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ValueError(f"PHASE reference hash differs: {name}")
    phase_summary = read_json(phase / "summary.json")
    if (
        phase_summary.get("status") != "COMPLETE"
        or phase_summary.get("terminal_population_accounted") is not True
        or phase_summary.get("recorded_rows") != PHASE_ROWS
        or phase_summary.get("null_rows") != 0
    ):
        raise ValueError("PHASE reference is not a complete 434-row terminal run")
    audit = read_json(phase / "independent_audit.json")
    audit_passed = (
        audit.get("valid") is True
        or audit.get("status") in {"PASS", "VALID"}
        or (audit.get("audit_status") == "PASS" and audit.get("rows_verified") == PHASE_ROWS)
    )
    if not audit_passed:
        raise ValueError("PHASE independent audit is not valid")
    return phase, phase_summary


def validate_executed_sources(freeze: Mapping[str, Any]) -> None:
    sources = freeze.get("executed_source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Garl freeze lacks executed source hashes")
    for relative, expected in sources.items():
        path = (ROOT / str(relative)).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Executed Garl source differs: {relative}")


def validate_ground_truth_audit(freeze: Mapping[str, Any]) -> dict[str, Any]:
    record = freeze.get("ground_truth_identity_audit")
    if not isinstance(record, dict):
        raise ValueError("Garl freeze lacks its exact-query ground-truth identity audit")
    path = Path(str(record.get("path_resolved", ""))).resolve(strict=True)
    if sha256_file(path) != record.get("sha256"):
        raise ValueError("Ground-truth identity audit hash differs")
    audit = read_json(path)
    if (
        record.get("schema") != "evttc_competition_gt_identity_inventory_v1"
        or record.get("status") != "NO_VERIFIED_EXACT_QUERY_GROUND_TRUTH"
        or audit.get("schema") != record.get("schema")
        or audit.get("status") != record.get("status")
        or audit.get("query_count") != PHASE_ROWS
    ):
        raise ValueError("Ground-truth identity audit does not prove the descriptive-only scope")
    return record


def validate_inventory_assets(inventory: Mapping[str, Any]) -> int:
    checked = 0

    def visit(value: Any) -> None:
        nonlocal checked
        if isinstance(value, dict):
            if {"path", "sha256"}.issubset(value):
                path = Path(str(value["path"])).resolve(strict=True)
                if sha256_file(path) != value["sha256"]:
                    raise ValueError(f"Pinned Garl inventory asset changed: {path}")
                if "bytes" in value and path.stat().st_size != int(value["bytes"]):
                    raise ValueError(f"Pinned Garl inventory size changed: {path}")
                checked += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(inventory)
    if checked == 0:
        raise ValueError("Garl inventory contains no pinned local assets")
    return checked


def validate_timing_policy(freeze: Mapping[str, Any]) -> dict[str, Any]:
    timing = freeze.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Garl freeze lacks its timing policy")
    if (
        timing.get("fixed_variant_order") != list(VARIANTS)
        or timing.get("no_explicit_tensor_or_preprocessing_reuse") is not True
        or timing.get("cross_variant_latency_ranking_supported") is not False
        or "page_cache" not in str(timing.get("shared_state", ""))
        or timing.get("not_optimized_streaming_latency") is not True
    ):
        raise ValueError("Garl timing policy does not preserve the declared descriptive scope")
    return dict(timing)


def validate_phase_rows(phase: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(phase / "all_rows.jsonl")
    if len(rows) != PHASE_ROWS:
        raise ValueError("PHASE row ledger does not contain 434 rows")
    seen = set()
    for row in rows:
        identity = row.get("identity")
        value = row.get("prediction_ttc_seconds")
        if (
            row.get("schema") != "evttc_competition_row_v1"
            or row.get("status") != "ok"
            or not isinstance(identity, dict)
            or value is None
            or not math.isfinite(float(value))
        ):
            raise ValueError("PHASE ledger contains a non-finite or noncanonical row")
        key = identity_key(identity)
        if key in seen:
            raise ValueError(f"Duplicate PHASE identity: {key}")
        seen.add(key)
    return rows


def validate_garl_rows(
    run: Path,
    phase_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_jsonl(run / "all_rows.jsonl")
    if len(rows) != GARL_ROWS:
        raise ValueError("Garl row ledger does not contain 1,302 rows")
    expected = [
        (identity_key(phase["identity"]), variant) for phase in phase_rows for variant in VARIANTS
    ]
    actual = []
    phase_identity = {identity_key(row["identity"]): row["identity"] for row in phase_rows}
    for row in rows:
        if row.get("schema") != "garl_evttc_competition_row_v1":
            raise ValueError("Unknown Garl row schema")
        variant = row.get("variant")
        identity = row.get("identity")
        if variant not in VARIANTS or not isinstance(identity, dict):
            raise ValueError("Garl row lacks a valid variant or identity")
        key = identity_key(identity)
        if identity != phase_identity.get(key):
            raise ValueError(f"Garl identity differs from PHASE: {key}")
        actual.append((key, variant))
        status, value = row.get("status"), row.get("prediction_ttc_seconds")
        if status == "ok":
            if value is None or not math.isfinite(float(value)):
                raise ValueError("Garl ok row lacks a finite prediction")
        elif status == "unavailable":
            if value is not None:
                raise ValueError("Garl unavailable row contains a prediction")
        else:
            raise ValueError(f"Nonterminal Garl row status: {status}")
        preprocessing = float(row.get("preprocess_seconds", math.nan))
        model = float(row.get("model_wall_seconds", math.nan))
        cost = float(row.get("cost_time_seconds", math.nan))
        if not all(math.isfinite(item) and item >= 0 for item in (preprocessing, model, cost)):
            raise ValueError("Garl row contains invalid runtime values")
        if not math.isclose(cost, preprocessing + model, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("Garl full cost differs from preprocessing plus model wall time")
        if not isinstance(row.get("input_details"), dict):
            raise ValueError("Garl row lacks input history details")
        if status == "ok" and not isinstance(row.get("prediction"), dict):
            raise ValueError("Garl ok row lacks runtime prediction details")
    if actual != expected:
        raise ValueError("Garl all_rows order is not sequence/row/variant aligned to PHASE")
    return rows


def history_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delta_values: list[float] = []
    window_counts: list[int] = []

    def visit(value: Any, key: str = "", *, collect_deltas: bool = True) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key), collect_deltas=collect_deltas)
        elif isinstance(value, list):
            if collect_deltas and key in {"delta_t_s", "deltas_s", "deltas"}:
                for item in value:
                    if isinstance(item, (int, float)) and math.isfinite(float(item)):
                        delta_values.append(float(item))
            if key in {"windows", "windows_us", "event_windows"}:
                window_counts.append(len(value))
            for item in value:
                visit(item, key, collect_deltas=collect_deltas)

    for row in rows:
        details = row.get("input_details", {})
        conversion_delta = (
            details.get("conversion_delta_t_s") if isinstance(details, dict) else None
        )
        if isinstance(conversion_delta, (int, float)) and math.isfinite(conversion_delta):
            delta_values.append(float(conversion_delta))
            visit(details, collect_deltas=False)
        else:
            visit(details)
    return {
        "delta_semantics": (
            "one input conversion_delta_t_s per available query"
            if any(
                isinstance(row.get("input_details"), dict)
                and row["input_details"].get("conversion_delta_t_s") is not None
                for row in rows
            )
            else "all declared sensor interval deltas in the PHASE input audit"
        ),
        "delta_count": len(delta_values),
        "delta_seconds_min": min(delta_values) if delta_values else None,
        "delta_seconds_median": statistics.median(delta_values) if delta_values else None,
        "delta_seconds_max": max(delta_values) if delta_values else None,
        "window_group_count": len(window_counts),
        "windows_per_group_min": min(window_counts) if window_counts else None,
        "windows_per_group_max": max(window_counts) if window_counts else None,
    }


def model_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finite = [float(row["prediction_ttc_seconds"]) for row in rows if row["status"] == "ok"]
    model = [float(row["model_wall_seconds"]) for row in rows]
    full = [float(row["cost_time_seconds"]) for row in rows]
    reasons = Counter(
        str(row.get("reason", "unspecified")) for row in rows if row["status"] != "ok"
    )
    p01, p99 = percentile(finite, 1), percentile(finite, 99)
    outside = (
        sum(value < p01 or value > p99 for value in finite)
        if p01 is not None and p99 is not None
        else 0
    )
    return {
        "records": len(rows),
        "finite_predictions": len(finite),
        "null_predictions": len(rows) - len(finite),
        "unavailable_reasons": dict(sorted(reasons.items())),
        "negative_predictions": sum(value < 0 for value in finite),
        "prediction_min_s": min(finite) if finite else None,
        "prediction_max_s": max(finite) if finite else None,
        "prediction_p01_s": p01,
        "prediction_p99_s": p99,
        "outside_empirical_p01_p99": outside,
        "model_wall_median_s": statistics.median(model),
        "model_wall_p95_s": percentile(model, 95),
        "full_cost_median_s": statistics.median(full),
        "full_cost_p95_s": percentile(full, 95),
        "history_and_cadence": history_summary(rows),
    }


def comparison_rows(
    phase_rows: Sequence[Mapping[str, Any]],
    garl_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    phase = {identity_key(row["identity"]): row for row in phase_rows}
    result = []
    for row in garl_rows:
        identity = row["identity"]
        reference = phase[identity_key(identity)]
        garl_value = row.get("prediction_ttc_seconds")
        phase_value = float(reference["prediction_ttc_seconds"])
        shared = row["status"] == "ok" and garl_value is not None
        if shared and garl_value is not None:
            garl_finite = float(garl_value)
            disagreement = garl_finite - phase_value
            absolute_disagreement = abs(disagreement)
        else:
            garl_finite = None
            disagreement = None
            absolute_disagreement = None
        result.append(
            {
                "sequence_id": identity["sequence_id"],
                "row_index": identity["row_index"],
                "index": identity["index"],
                "timestamp_text": identity["timestamp_text"],
                "variant": row["variant"],
                "phase_ttc_s": phase_value,
                "garl_ttc_s": garl_finite,
                "shared_valid": shared,
                "prediction_disagreement_garl_minus_phase_s": disagreement,
                "absolute_prediction_disagreement_s": absolute_disagreement,
                "garl_status": row["status"],
                "garl_reason": row.get("reason"),
                "garl_model_wall_s": row["model_wall_seconds"],
                "garl_full_cost_s": row["cost_time_seconds"],
                "phase_model_wall_s": reference.get("model_wall_seconds"),
                "phase_full_cost_s": reference.get("cost_time_seconds"),
            }
        )
    return result


def disagreement_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["prediction_disagreement_garl_minus_phase_s"])
        for row in rows
        if row["shared_valid"]
    ]
    absolute = [abs(value) for value in values]
    return {
        "shared_valid_queries": len(values),
        "null_gap_queries": len(rows) - len(values),
        "signed_disagreement_median_s": statistics.median(values) if values else None,
        "signed_disagreement_min_s": min(values) if values else None,
        "signed_disagreement_max_s": max(values) if values else None,
        "absolute_disagreement_median_s": statistics.median(absolute) if absolute else None,
        "absolute_disagreement_p95_s": percentile(absolute, 95),
    }


def per_sequence_stats(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for sequence_id in SEQUENCES:
        for variant in VARIANTS:
            selected = [
                row
                for row in rows
                if row["sequence_id"] == sequence_id and row["variant"] == variant
            ]
            absolute = [
                float(row["absolute_prediction_disagreement_s"])
                for row in selected
                if row["shared_valid"]
            ]
            result.append(
                {
                    "sequence_id": sequence_id,
                    "variant": variant,
                    "queries": len(selected),
                    "shared_valid_queries": len(absolute),
                    "null_gap_queries": len(selected) - len(absolute),
                    "coverage_fraction": len(absolute) / len(selected),
                    "absolute_disagreement_median_s": (
                        statistics.median(absolute) if absolute else None
                    ),
                    "absolute_disagreement_p95_s": percentile(absolute, 95),
                }
            )
    if len(result) != len(SEQUENCES) * len(VARIANTS):
        raise RuntimeError("Per-sequence comparison table must contain exactly 30 rows")
    return result


def report_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def report_cadence(stats: Mapping[str, Any]) -> str:
    return (
        f"{stats['delta_count']} deltas; min/median/max "
        f"{report_number(stats['delta_seconds_min'], 6)}/"
        f"{report_number(stats['delta_seconds_median'], 6)}/"
        f"{report_number(stats['delta_seconds_max'], 6)} s; "
        f"semantics: {stats['delta_semantics']}"
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: Figure, output: Path, stem: str) -> list[Path]:
    paths = [output / f"{stem}.pdf", output / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths[1], dpi=190, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return paths


def plot_all_predictions(
    phase_rows: Sequence[Mapping[str, Any]],
    garl_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    by_variant = {
        variant: [row for row in garl_rows if row["variant"] == variant] for variant in VARIANTS
    }
    series = {"phase": [float(row["prediction_ttc_seconds"]) for row in phase_rows]}
    for variant, rows in by_variant.items():
        series[variant] = [
            float(row["prediction_ttc_seconds"]) if row["status"] == "ok" else math.nan
            for row in rows
        ]
    finite = [value for values in series.values() for value in values if math.isfinite(value)]
    low, high = min(finite), max(finite)
    gap_y = low - max((high - low) * 0.04, 0.1)
    fig, axes = plt.subplots(4, 1, figsize=(11.0, 8.4), sharex=True, sharey=True)
    x = np.arange(PHASE_ROWS)
    for ax, (name, values) in zip(axes, series.items(), strict=True):
        array = np.asarray(values, dtype=float)
        ax.plot(x, array, color=COLORS[name], lw=0.9, label=name.upper())
        missing = np.flatnonzero(~np.isfinite(array))
        if len(missing):
            ax.scatter(
                missing,
                np.full(len(missing), gap_y),
                marker="x",
                color="#111827",
                s=18,
                label="null gap",
            )
        ax.set_ylabel("TTC prediction (s)")
        ax.legend(loc="upper right", ncol=2)
        ax.grid(axis="y")
    axes[-1].set_xlabel("Frozen query order (434 rows)")
    fig.suptitle("Unscored predictions / nulls marked as gaps", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, output, "all_predictions_with_null_gaps")


def plot_sequences(
    phase_rows: Sequence[Mapping[str, Any]],
    garl_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    phase_by_sequence = defaultdict(list)
    garl_by_sequence: dict[str, dict[str, list[float]]] = {
        variant: defaultdict(list) for variant in VARIANTS
    }
    for row in phase_rows:
        phase_by_sequence[row["identity"]["sequence_id"]].append(
            float(row["prediction_ttc_seconds"])
        )
    for row in garl_rows:
        garl_by_sequence[row["variant"]][row["identity"]["sequence_id"]].append(
            float(row["prediction_ttc_seconds"]) if row["status"] == "ok" else math.nan
        )
    all_values = [value for values in phase_by_sequence.values() for value in values]
    for variant in VARIANTS:
        all_values.extend(
            value
            for values in garl_by_sequence[variant].values()
            for value in values
            if math.isfinite(value)
        )
    low, high = min(all_values), max(all_values)
    gap_y = low - max((high - low) * 0.04, 0.1)
    fig, axes = plt.subplots(5, 2, figsize=(12.0, 12.2), sharey=True)
    for ax, sequence_id in zip(axes.flat, SEQUENCES, strict=True):
        x = np.arange(len(phase_by_sequence[sequence_id]))
        ax.plot(x, phase_by_sequence[sequence_id], color=COLORS["phase"], lw=1.0, label="PHASE")
        for variant in VARIANTS:
            values = np.asarray(garl_by_sequence[variant][sequence_id], dtype=float)
            ax.plot(x, values, color=COLORS[variant], lw=0.85, label=f"Garl {variant}")
            missing = np.flatnonzero(~np.isfinite(values))
            if len(missing):
                ax.scatter(
                    missing, np.full(len(missing), gap_y), color=COLORS[variant], marker="x", s=15
                )
        ax.set_title(sequence_id, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_ylim(gap_y - max((high - low) * 0.02, 0.05), high + max((high - low) * 0.05, 0.1))
        ax.grid(axis="y")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.972))
    fig.suptitle(
        "Common-scale per-sequence predictions (unscored)", y=0.995, fontsize=13, fontweight="bold"
    )
    fig.supxlabel("Within-sequence frozen query order")
    fig.supylabel("TTC prediction (s)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.93))
    return save_figure(fig, output, "predictions_by_sequence_common_scale")


def plot_sequences_symlog(
    phase_rows: Sequence[Mapping[str, Any]],
    garl_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    """Supplement the full linear view without dropping signed outliers or null gaps."""
    phase_by_sequence = defaultdict(list)
    garl_by_sequence: dict[str, dict[str, list[float]]] = {
        variant: defaultdict(list) for variant in VARIANTS
    }
    for row in phase_rows:
        phase_by_sequence[row["identity"]["sequence_id"]].append(
            float(row["prediction_ttc_seconds"])
        )
    for row in garl_rows:
        garl_by_sequence[row["variant"]][row["identity"]["sequence_id"]].append(
            float(row["prediction_ttc_seconds"]) if row["status"] == "ok" else math.nan
        )
    fig, axes = plt.subplots(5, 2, figsize=(12.0, 12.2), sharey=True)
    for ax, sequence_id in zip(axes.flat, SEQUENCES, strict=True):
        x = np.arange(len(phase_by_sequence[sequence_id]))
        ax.set_yscale("symlog", linthresh=1.0)
        ax.plot(
            x,
            phase_by_sequence[sequence_id],
            color=COLORS["phase"],
            lw=1.0,
            label="PHASE",
        )
        for variant in VARIANTS:
            values = np.asarray(garl_by_sequence[variant][sequence_id], dtype=float)
            ax.plot(x, values, color=COLORS[variant], lw=0.85, label=f"Garl {variant}")
            missing = np.flatnonzero(~np.isfinite(values))
            if len(missing):
                ax.scatter(
                    missing,
                    np.full(len(missing), 0.035),
                    color=COLORS[variant],
                    marker="x",
                    s=16,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                )
        ax.set_title(sequence_id, loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="y")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.972))
    fig.suptitle(
        "Signed symlog per-sequence predictions / linthresh = 1 s / nulls at axis baseline",
        y=0.995,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.supxlabel("Within-sequence frozen query order")
    fig.supylabel("TTC prediction (s; signed symlog scale)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.93))
    return save_figure(fig, output, "predictions_by_sequence_signed_symlog")


def plot_runtime(runtime_stats: Mapping[str, Mapping[str, Any]], output: Path) -> list[Path]:
    methods = ("phase", *VARIANTS)
    x = np.arange(len(methods))
    width = 0.19
    fields = (
        ("model_wall_median_s", "Model median", "#3769DB"),
        ("model_wall_p95_s", "Model P95", "#7895E5"),
        ("full_cost_median_s", "Full median", "#268B86"),
        ("full_cost_p95_s", "Full P95", "#65AAA6"),
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for index, (field, label, color) in enumerate(fields):
        values = [float(runtime_stats[method][field]) for method in methods]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.set_xticks(x, ["PHASE", *(f"Garl {variant}" for variant in VARIANTS)])
    ax.set_ylabel("Seconds per query")
    ax.set_title("Offline runtime telemetry / descriptive only, no ranking", fontweight="bold")
    ax.grid(axis="y")
    ax.legend(ncol=2)
    fig.tight_layout()
    return save_figure(fig, output, "offline_runtime_profile")


def build_comparison(
    run: Path,
    output: Path | None = None,
    expected_phase: Path = EXPECTED_PHASE_RUN,
) -> dict[str, Any]:
    run = run.resolve(strict=True)
    summary = validate_terminal_summary(run)
    freeze = read_json(run / "RUN_FREEZE.json")
    if freeze.get("schema") != "garl_evttc_competition_freeze_v1":
        raise ValueError("Unknown Garl run freeze schema")
    if not isinstance(freeze.get("inventory"), dict) or not freeze["inventory"]:
        raise ValueError("Garl freeze lacks its pinned checkpoint/source inventory")
    pinned_asset_count = validate_inventory_assets(freeze["inventory"])
    timing_policy = validate_timing_policy(freeze)
    ground_truth_audit = validate_ground_truth_audit(freeze)
    literature = read_json(LITERATURE_RECEIPT.resolve(strict=True))
    garl_papers = [
        paper
        for paper in literature.get("papers", [])
        if isinstance(paper, dict) and "evttc_result" in paper
    ]
    if (
        len(garl_papers) != 1
        or garl_papers[0]["evttc_result"]["unweighted_three_sequence_average"]["rte_percent"]
        != 10.60
    ):
        raise ValueError("Literature receipt does not support the separately scoped 10.60% RTE")
    literature_sha256 = sha256_file(LITERATURE_RECEIPT)
    validate_executed_sources(freeze)
    phase, _phase_summary = validate_phase_reference(run, freeze, expected_phase)
    phase_freeze = read_json(phase / "RUN_FREEZE.json")
    if freeze.get("query_manifest") != phase_freeze.get("query_manifest"):
        raise ValueError("Garl and PHASE query manifests differ")
    phase_rows = validate_phase_rows(phase)
    garl_rows = validate_garl_rows(run, phase_rows)

    output = (output or run / "comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparisons = comparison_rows(phase_rows, garl_rows)
    write_csv(output / "per_query_prediction_disagreement.csv", comparisons)
    sequence_stats = per_sequence_stats(comparisons)
    write_csv(output / "per_sequence_prediction_disagreement.csv", sequence_stats)
    grouped = {
        variant: [row for row in garl_rows if row["variant"] == variant] for variant in VARIANTS
    }
    for variant, rows in grouped.items():
        actual_nulls = sum(row["status"] == "unavailable" for row in rows)
        declared = summary["variants"][variant]
        declared_nulls = declared.get("nulls", declared.get("null_rows"))
        if actual_nulls != int(declared_nulls):
            raise ValueError(f"Variant {variant} null count disagrees with summary")
    variant_stats = {variant: model_stats(rows) for variant, rows in grouped.items()}
    phase_stats = model_stats(phase_rows)
    disagreement = {
        variant: disagreement_stats([row for row in comparisons if row["variant"] == variant])
        for variant in VARIANTS
    }
    figures = [
        *plot_all_predictions(phase_rows, garl_rows, output),
        *plot_sequences(phase_rows, garl_rows, output),
        *plot_sequences_symlog(phase_rows, garl_rows, output),
        *plot_runtime({"phase": phase_stats, **variant_stats}, output),
    ]
    result = {
        "schema": "garl_phase_evttc_prediction_comparison_v1",
        "status": "COMPLETE_LABEL_BLIND_COMPARISON",
        "ground_truth_available_for_matching_434_queries": False,
        "ground_truth_identity_audit": ground_truth_audit,
        "pinned_inventory_assets_verified": pinned_asset_count,
        "timing_policy": timing_policy,
        "metric_semantics": "prediction disagreement, not error, MAE, RTE, or accuracy",
        "garl_terminal_status": summary["status"],
        "phase_rows": len(phase_rows),
        "garl_records": len(garl_rows),
        "phase_stats": phase_stats,
        "variant_stats": variant_stats,
        "prediction_disagreement": disagreement,
        "per_sequence_prediction_disagreement": sequence_stats,
        "literature_context": {
            "reported_value": "10.60% RTE",
            "scope": "Garl-TTC paper, three-sequence zero-shot EvTTC comparison",
            "comparability": (
                "separate literature result; not compared numerically to these "
                "unscored 434-query predictions"
            ),
            "source_receipt": str(LITERATURE_RECEIPT.relative_to(ROOT)).replace("\\", "/"),
            "source_receipt_sha256": literature_sha256,
        },
        "caveats": [
            "No proven matching ground truth is available for these exact 434 query identities.",
            "Pointwise differences measure model disagreement only.",
            "Runtime values come from independent local implementations and are not an "
            "official benchmark latency comparison.",
            "Garl variants ran in fixed full/event/visual order; OS/HDF5 page cache and "
            "GPU warm state may be shared despite independent tensor preparation.",
            "The single pass cannot support cross-variant latency ranking or an efficiency "
            "conclusion, and PHASE used a different history and recomputed offline adapter.",
            "Null predictions remain explicit gaps and are not imputed.",
        ],
    }
    atomic_json(output / "comparison.json", result)
    report = [
        "# Garl-TTC and PHASE-TTC on the frozen EvTTC queries",
        "",
        "This is a label-blind prediction-disagreement audit, not an accuracy evaluation.",
        "No proven matching ground truth exists for these exact 434 query identities; "
        "therefore the report does not compute MAE, RTE, error, or a winner.",
        "",
        "## Terminal accounting",
        "",
        f"- PHASE: {len(phase_rows)} finite predictions.",
        f"- Garl: {len(garl_rows)} terminal records across full, event-only, and "
        "visual-only checkpoints.",
        "- Nulls are retained as gaps and are never filled.",
        (
            f"- PHASE signed range: {report_number(phase_stats['prediction_min_s'])} to "
            f"{report_number(phase_stats['prediction_max_s'])} s; negative predictions "
            f"{phase_stats['negative_predictions']}."
        ),
        "",
        "## Prediction disagreement",
        "",
    ]
    for variant in VARIANTS:
        stats = disagreement[variant]
        runtime = variant_stats[variant]
        report.append(
            f"- **Garl {variant}:** {stats['shared_valid_queries']} shared-valid; "
            f"{stats['null_gap_queries']} null gaps; median absolute disagreement "
            f"{report_number(stats['absolute_disagreement_median_s'])} s; P95 "
            f"{report_number(stats['absolute_disagreement_p95_s'])} s; "
            f"negative predictions {runtime['negative_predictions']}; signed prediction "
            f"range [{report_number(runtime['prediction_min_s'])}, "
            f"{report_number(runtime['prediction_max_s'])}] s; "
            f"outside empirical P01--P99: {runtime['outside_empirical_p01_p99']}; "
            f"unavailable reasons {runtime['unavailable_reasons']}."
        )
    report.extend(
        [
            "",
            "## Per-sequence coverage and disagreement",
            "",
            "The table reports prediction coverage and median absolute model disagreement; "
            "it is not an accuracy table.",
            "",
            "| Sequence | Variant | Coverage | Median absolute disagreement (s) |",
            "|---|---:|---:|---:|",
            *(
                f"| {row['sequence_id']} | {row['variant']} | "
                f"{row['shared_valid_queries']}/{row['queries']} "
                f"({100 * row['coverage_fraction']:.1f}%) | "
                f"{report_number(row['absolute_disagreement_median_s'])} |"
                for row in sequence_stats
            ),
        ]
    )
    report.extend(
        [
            "",
            "## Runtime scope",
            "",
            "Model-wall and full-cost medians/P95 are reported separately. They describe "
            "independent offline local implementations, not official latency and not accuracy.",
            "Garl used fixed full/event/visual order with independent preparation, but OS/HDF5 "
            "page cache and GPU warm state may be shared. This one pass does not support "
            "cross-variant latency ranking or an efficiency conclusion. PHASE is also not a "
            "fair latency comparator because it uses a different history and recomputed "
            "offline adapter.",
            *(
                f"- {name}: model median/P95 "
                f"{report_number(stats['model_wall_median_s'])}/"
                f"{report_number(stats['model_wall_p95_s'])} s; full median/P95 "
                f"{report_number(stats['full_cost_median_s'])}/"
                f"{report_number(stats['full_cost_p95_s'])} s; cadence "
                f"{report_cadence(stats['history_and_cadence'])}."
                for name, stats in (("PHASE", phase_stats), *variant_stats.items())
            ),
            "",
            "## Literature context",
            "",
            "The Garl-TTC paper reports 10.60% RTE for a separate three-sequence "
            "zero-shot EvTTC comparison. That literature number is not matched to this "
            "ten-sequence, 434-query identity set and is not compared to these unscored "
            "predictions.",
        ]
    )
    atomic_text(output / "REPORT.md", "\n".join(report) + "\n")
    generated = [
        output / "comparison.json",
        output / "per_query_prediction_disagreement.csv",
        output / "per_sequence_prediction_disagreement.csv",
        output / "REPORT.md",
        *figures,
    ]
    manifest = {
        "schema": "garl_phase_evttc_comparison_manifest_v1",
        "inputs": {
            "garl_RUN_FREEZE.json": sha256_file(run / "RUN_FREEZE.json"),
            "garl_all_rows.jsonl": sha256_file(run / "all_rows.jsonl"),
            "garl_summary.json": sha256_file(run / "summary.json"),
            "ground_truth_identity_audit": ground_truth_audit["sha256"],
            "literature_receipt": literature_sha256,
            **{f"phase_{name}": sha256_file(phase / name) for name in PHASE_ARTIFACTS},
        },
        "generator": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {path.name: sha256_file(path) for path in generated},
    }
    atomic_json(output / "comparison_manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", type=Path, default=EXPECTED_PHASE_RUN)
    args = parser.parse_args()
    result = build_comparison(args.run, args.output, args.phase)
    print(json.dumps({"status": result["status"], "records": result["garl_records"]}))


if __name__ == "__main__":
    main()
