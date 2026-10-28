"""Plot sealed EvTTC competition predictions without reading target data.

The input ledger is opened only after ``summary.json`` reports a complete run.
All recorded rows are plotted; the figures are descriptive diagnostics, not an
accuracy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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
STARTUP_MODES = frozenset({"adaptive_observed_prefix", "native_100ms_windows"})
MODEL_REASONS = frozenset({"fused_event_rgb", "event_only_fallback"})
ROW_SCHEMA = "evttc_competition_row_v1"


@dataclass(frozen=True)
class PlotRow:
    """Validated fields used by the diagnostic plots."""

    sequence_id: str
    timestamp_text: str
    timestamp_seconds: float
    index: int
    prediction_seconds: float | None
    cost_seconds: float
    status: str
    startup_mode: str
    zero_support: bool
    model_reason: str
    rejected_event_observations: int
    rejected_rgb_observations: int
    source_ordinal: int


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _load_terminal_run(run: Path) -> tuple[Mapping[str, Any], list[PlotRow], Path, Path]:
    """Validate terminal status before opening the prediction ledger."""

    summary_path = run / "summary.json"
    rows_path = run / "all_rows.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    summary = _object(json.loads(summary_path.read_text(encoding="utf-8")), "summary.json")
    status = summary.get("status")
    terminal_partial = status == "PARTIAL" and summary.get("terminal_population_accounted") is True
    if status != "COMPLETE" and not terminal_partial:
        raise RuntimeError(
            "Refusing to read all_rows.jsonl before summary.json reports COMPLETE or a "
            "terminal population-accounted PARTIAL run"
        )
    if not rows_path.is_file():
        raise FileNotFoundError(f"Missing completed prediction ledger: {rows_path}")

    rows: list[PlotRow] = []
    with rows_path.open("r", encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at line {ordinal}")
            record = _object(json.loads(line), f"all_rows.jsonl line {ordinal}")
            if record.get("schema") != ROW_SCHEMA:
                raise ValueError(f"Unexpected row schema at line {ordinal}")
            row_status = str(record.get("status"))
            if row_status not in {"ok", "unavailable"}:
                raise ValueError(f"Terminal run contains unsupported row status at line {ordinal}")

            identity = _object(record.get("identity"), f"line {ordinal}.identity")
            input_details = _object(record.get("input_details"), f"line {ordinal}.input_details")
            prediction = _object(record.get("prediction"), f"line {ordinal}.prediction")
            sequence_id = str(identity.get("sequence_id"))
            timestamp_text = str(identity.get("timestamp_text"))
            timestamp_seconds = _finite_float(
                identity.get("timestamp_text"), f"line {ordinal}.identity.timestamp_text"
            )
            index = _integer(identity.get("index"), f"line {ordinal}.identity.index")
            raw_prediction = record.get("prediction_ttc_seconds")
            prediction_seconds = (
                None
                if raw_prediction is None
                else _finite_float(
                    raw_prediction,
                    f"line {ordinal}.prediction_ttc_seconds",
                )
            )
            if (row_status == "ok") != (prediction_seconds is not None):
                raise ValueError(
                    f"line {ordinal} status/prediction availability is internally inconsistent"
                )
            cost_seconds = _finite_float(
                record.get("cost_time_seconds"), f"line {ordinal}.cost_time_seconds"
            )
            if cost_seconds < 0:
                raise ValueError(f"Negative cost_time_seconds at line {ordinal}")

            startup_mode = str(input_details.get("startup_mode"))
            if startup_mode not in STARTUP_MODES:
                raise ValueError(f"Unexpected startup_mode at line {ordinal}: {startup_mode}")
            observations = input_details.get("event_observations")
            if not isinstance(observations, list):
                raise ValueError(f"line {ordinal}.input_details.event_observations must be a list")
            zero_support = False
            for observation_index, observation_value in enumerate(observations):
                observation = _object(
                    observation_value,
                    f"line {ordinal}.event_observations[{observation_index}]",
                )
                flag = observation.get("zero_support")
                if not isinstance(flag, bool):
                    raise ValueError(
                        f"line {ordinal}.event_observations[{observation_index}].zero_support "
                        "must be boolean"
                    )
                zero_support |= flag

            model_reason = str(prediction.get("model_reason"))
            if prediction_seconds is not None and model_reason not in MODEL_REASONS:
                raise ValueError(f"Unexpected prediction.model_reason at line {ordinal}")
            if not model_reason or model_reason == "None":
                raise ValueError(f"Missing prediction.model_reason at line {ordinal}")
            runtime_internal = _object(
                record.get("runtime_internal"), f"line {ordinal}.runtime_internal"
            )
            diagnostics = _object(
                runtime_internal.get("producer_diagnostics"),
                f"line {ordinal}.runtime_internal.producer_diagnostics",
            )
            rejected_by_modality: dict[str, int] = {}
            for modality in ("event", "rgb"):
                raw_modality = diagnostics.get(modality, {})
                modality_diagnostics = _object(
                    raw_modality, f"line {ordinal}.producer_diagnostics.{modality}"
                )
                rejected = modality_diagnostics.get("observations_rejected_unrepresentable", 0)
                rejected_by_modality[modality] = _integer(
                    rejected, f"line {ordinal}.{modality}.observations_rejected_unrepresentable"
                )
                if rejected_by_modality[modality] < 0:
                    raise ValueError(f"Negative rejected-observation count at line {ordinal}")
            rows.append(
                PlotRow(
                    sequence_id=sequence_id,
                    timestamp_text=timestamp_text,
                    timestamp_seconds=timestamp_seconds,
                    index=index,
                    prediction_seconds=prediction_seconds,
                    cost_seconds=cost_seconds,
                    status=row_status,
                    startup_mode=startup_mode,
                    zero_support=zero_support,
                    model_reason=model_reason,
                    rejected_event_observations=rejected_by_modality["event"],
                    rejected_rgb_observations=rejected_by_modality["rgb"],
                    source_ordinal=ordinal,
                )
            )

    expected_rows = _integer(summary.get("expected_rows"), "summary.expected_rows")
    recorded_rows = _integer(summary.get("recorded_rows"), "summary.recorded_rows")
    if len(rows) != expected_rows or len(rows) != recorded_rows:
        raise ValueError(
            f"Ledger count {len(rows)} differs from summary counts "
            f"(expected={expected_rows}, recorded={recorded_rows})"
        )
    if set(row.sequence_id for row in rows) != set(SEQUENCES):
        raise ValueError("Completed ledger does not contain exactly the ten expected sequences")
    return summary, rows, summary_path, rows_path


def _shared_y_limits(rows: Sequence[PlotRow]) -> tuple[float, float]:
    values = np.asarray(
        [row.prediction_seconds for row in rows if row.prediction_seconds is not None],
        dtype=np.float64,
    )
    if not values.size:
        return -1.0, 1.0
    lower = min(float(values.min()), 0.0)
    upper = max(float(values.max()), 0.0)
    span = upper - lower
    pad = max(span * 0.06, 0.05)
    return lower - pad, upper + pad


def _plot_predictions(rows: Sequence[PlotRow], output: Path) -> tuple[float, float]:
    by_sequence: dict[str, list[PlotRow]] = defaultdict(list)
    for row in rows:
        by_sequence[row.sequence_id].append(row)
    y_limits = _shared_y_limits(rows)
    figure, axes = plt.subplots(
        5,
        2,
        figsize=(14.2, 17.8),
        sharey=True,
    )
    figure.subplots_adjust(
        top=0.89, bottom=0.045, left=0.065, right=0.985, hspace=0.38, wspace=0.06
    )
    for axis, sequence_id in zip(axes.flat, SEQUENCES, strict=True):
        sequence_rows = sorted(
            by_sequence[sequence_id],
            key=lambda row: (row.timestamp_seconds, row.index, row.source_ordinal),
        )
        x = np.asarray([row.timestamp_seconds for row in sequence_rows])
        y = np.asarray(
            [
                np.nan if row.prediction_seconds is None else row.prediction_seconds
                for row in sequence_rows
            ]
        )
        available = np.isfinite(y)
        adaptive = np.asarray(
            [row.startup_mode == "adaptive_observed_prefix" for row in sequence_rows]
        )
        zero = np.asarray([row.zero_support for row in sequence_rows])
        rejected_event = np.asarray([row.rejected_event_observations > 0 for row in sequence_rows])
        rejected_rgb = np.asarray([row.rejected_rgb_observations > 0 for row in sequence_rows])
        axis.plot(
            x,
            y,
            color="#1768AC",
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            markeredgewidth=0,
            zorder=2,
        )
        if (adaptive & available).any():
            axis.scatter(
                x[adaptive & available],
                y[adaptive & available],
                s=43,
                marker="s",
                facecolors="none",
                edgecolors="#E68A00",
                linewidths=1.25,
                zorder=3,
            )
        if (zero & available).any():
            axis.scatter(
                x[zero & available],
                y[zero & available],
                s=45,
                marker="x",
                color="#C33C54",
                linewidths=1.45,
                zorder=4,
            )
        if (rejected_event & available).any():
            axis.scatter(
                x[rejected_event & available],
                y[rejected_event & available],
                s=58,
                marker="D",
                facecolors="none",
                edgecolors="#8E44AD",
                linewidths=1.25,
                zorder=5,
            )
        if (rejected_rgb & available).any():
            axis.scatter(
                x[rejected_rgb & available],
                y[rejected_rgb & available],
                s=58,
                marker="+",
                color="#16836B",
                linewidths=1.45,
                zorder=5,
            )
        missing = ~available
        if missing.any():
            missing_y = y_limits[0] + 0.025 * (y_limits[1] - y_limits[0])
            axis.scatter(
                x[missing],
                np.full(int(missing.sum()), missing_y),
                s=50,
                marker="v",
                facecolors="none",
                edgecolors="#344054",
                linewidths=1.25,
                zorder=6,
            )
            axis.vlines(
                x[missing],
                y_limits[0],
                missing_y,
                color="#98A2B3",
                linewidth=0.7,
                linestyle=":",
                zorder=1,
            )
        axis.axhline(0.0, color="#667085", linewidth=0.8, linestyle="--", zorder=1)
        axis.set_ylim(y_limits)
        axis.grid(True, color="#D6DEE8", linewidth=0.65, alpha=0.7)
        axis.set_title(
            f"{sequence_id}  |  n={len(sequence_rows)}, adaptive={int(adaptive.sum())}, "
            f"empty={int(zero.sum())}, unavailable={int(missing.sum())}",
            fontsize=10.5,
            loc="left",
        )
        axis.set_xlabel("Provided clip timestamp (s)")
        axis.set_ylabel("Predicted signed TTC (s)")

    handles = (
        Line2D([0], [0], color="#1768AC", marker="o", linewidth=1.35, label="Model prediction"),
        Line2D(
            [0],
            [0],
            color="#E68A00",
            marker="s",
            markerfacecolor="none",
            linewidth=0,
            label="Adaptive-startup prediction",
        ),
        Line2D(
            [0],
            [0],
            color="#C33C54",
            marker="x",
            linewidth=0,
            label="Contains empty event ROI",
        ),
        Line2D(
            [0],
            [0],
            color="#8E44AD",
            marker="D",
            markerfacecolor="none",
            linewidth=0,
            label="Event observation rejected",
        ),
        Line2D(
            [0],
            [0],
            color="#16836B",
            marker="+",
            linewidth=0,
            label="RGB observation rejected",
        ),
        Line2D(
            [0],
            [0],
            color="#344054",
            marker="v",
            markerfacecolor="none",
            linewidth=0,
            label="Unavailable query (gap)",
        ),
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "PHASE-TTC EvTTC predictions (unscored)",
        fontsize=17,
        fontweight="bold",
        y=0.992,
    )
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)
    return y_limits


def _cost_statistics(rows: Sequence[PlotRow]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.sequence_id].append(row.cost_seconds)
    return {
        sequence_id: {
            "n": len(grouped[sequence_id]),
            "median_seconds": float(np.median(grouped[sequence_id])),
            "p95_seconds": float(np.quantile(grouped[sequence_id], 0.95)),
            "minimum_seconds": float(np.min(grouped[sequence_id])),
            "maximum_seconds": float(np.max(grouped[sequence_id])),
        }
        for sequence_id in SEQUENCES
    }


def _plot_costs(statistics: Mapping[str, Mapping[str, float | int]], output: Path) -> None:
    medians = np.asarray([statistics[name]["median_seconds"] for name in SEQUENCES])
    p95 = np.asarray([statistics[name]["p95_seconds"] for name in SEQUENCES])
    positions = np.arange(len(SEQUENCES), dtype=np.float64)
    width = 0.38
    figure, axis = plt.subplots(figsize=(14.2, 7.4))
    figure.subplots_adjust(top=0.86, bottom=0.2, left=0.07, right=0.985)
    axis.bar(
        positions - width / 2,
        medians,
        width,
        color="#1768AC",
        label="Median measured cost",
    )
    axis.bar(
        positions + width / 2,
        p95,
        width,
        color="#35A79C",
        label="95th percentile measured cost",
    )
    axis.set_xticks(positions, SEQUENCES, rotation=30, ha="right")
    axis.set_ylabel("Measured per-row wall time (s)")
    figure.suptitle(
        "Observed end-to-end prediction cost by sequence",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.07,
        0.925,
        "cost_time_seconds = preprocessing + synchronized model wall time; all completed rows",
        color="#475467",
        fontsize=10.5,
    )
    axis.grid(axis="y", color="#D6DEE8", linewidth=0.7, alpha=0.75)
    axis.legend(frameon=False)
    ceiling = max(float(p95.max()) * 1.12, 1e-6)
    axis.set_ylim(0.0, ceiling)
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def build_diagnostics(run: Path, output: Path) -> dict[str, Any]:
    """Build plots and provenance records for one sealed complete run."""

    summary, rows, summary_path, rows_path = _load_terminal_run(run)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "ttc_predictions_by_sequence.png"
    costs_path = output / "measured_cost_by_sequence.png"
    metadata_path = output / "plot_metadata.json"
    ledger_path = output / "plot_ledger.json"

    y_limits = _plot_predictions(rows, predictions_path)
    costs = _cost_statistics(rows)
    _plot_costs(costs, costs_path)

    sequence_counts = Counter(row.sequence_id for row in rows)
    startup_counts = Counter(row.startup_mode for row in rows)
    reason_counts = Counter(row.model_reason for row in rows)
    per_sequence = {
        name: {
            "rows": sequence_counts[name],
            "adaptive_startup_rows": sum(
                row.sequence_id == name and row.startup_mode == "adaptive_observed_prefix"
                for row in rows
            ),
            "rows_containing_empty_event_roi": sum(
                row.sequence_id == name and row.zero_support for row in rows
            ),
            "unavailable_rows": sum(
                row.sequence_id == name and row.prediction_seconds is None for row in rows
            ),
            "rows_with_rejected_event_observations": sum(
                row.sequence_id == name and row.rejected_event_observations > 0 for row in rows
            ),
            "rows_with_rejected_rgb_observations": sum(
                row.sequence_id == name and row.rejected_rgb_observations > 0 for row in rows
            ),
            "model_reason_counts": dict(
                sorted(Counter(row.model_reason for row in rows if row.sequence_id == name).items())
            ),
            "cost_time_seconds": costs[name],
        }
        for name in SEQUENCES
    }
    metadata: dict[str, Any] = {
        "schema": "evttc_competition_prediction_plots_v1",
        "run": str(run.resolve()),
        "summary_status": summary["status"],
        "expected_rows": summary["expected_rows"],
        "plotted_rows": len(rows),
        "scope": (
            "Descriptive unscored EvTTC predictions only; no ground truth or accuracy curves "
            "were read or plotted by this postprocessor."
        ),
        "point_policy": (
            "All terminal ledger rows included; no filtering or outlier removal. Unavailable "
            "queries are explicit gap markers and are never interpolated."
        ),
        "prediction_axis": {
            "scale": "linear",
            "signed": True,
            "shared_across_panels": True,
            "limits_seconds": list(y_limits),
        },
        "timestamp_policy": (
            "identity.timestamp_text preserved in the ledger and parsed as float only for plotting"
        ),
        "cost_scope": (
            "cost_time_seconds from the sealed row: preprocessing_seconds plus synchronized "
            "model wall time; not model-only latency"
        ),
        "startup_mode_counts": dict(sorted(startup_counts.items())),
        "rows_containing_empty_event_roi": sum(row.zero_support for row in rows),
        "empty_event_roi_definition": (
            "A counted row contains at least one entirely empty event-ROI observation; this does "
            "not imply that the full prediction lacked event support."
        ),
        "unavailable_rows": sum(row.prediction_seconds is None for row in rows),
        "rows_with_rejected_event_observations": sum(
            row.rejected_event_observations > 0 for row in rows
        ),
        "rows_with_rejected_rgb_observations": sum(
            row.rejected_rgb_observations > 0 for row in rows
        ),
        "model_reason_counts": dict(sorted(reason_counts.items())),
        "per_sequence": per_sequence,
        "figures": [predictions_path.name, costs_path.name],
    }
    _atomic_json(metadata_path, metadata)

    script_path = Path(__file__).resolve()
    ledger: dict[str, Any] = {
        "schema": "evttc_competition_prediction_plot_ledger_v1",
        "sources": {
            "summary.json": {"path": str(summary_path.resolve()), "sha256": _sha256(summary_path)},
            "all_rows.jsonl": {"path": str(rows_path.resolve()), "sha256": _sha256(rows_path)},
            "plot_script": {"path": str(script_path), "sha256": _sha256(script_path)},
        },
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in (predictions_path, costs_path, metadata_path)
        },
    }
    _atomic_json(ledger_path, ledger)
    return {
        "status": summary["status"],
        "plotted_rows": len(rows),
        "output": str(output.resolve()),
        "ledger_sha256": _sha256(ledger_path),
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot diagnostics from a sealed terminal EvTTC competition run."
    )
    parser.add_argument("--run", type=Path, required=True, help="Completed run directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: RUN/diagnostics)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    run = arguments.run.resolve()
    output = (arguments.output or (run / "diagnostics")).resolve()
    result = build_diagnostics(run, output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
