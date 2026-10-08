"""Explicit signed-TTC scoring with GT-only cohorts and paired cluster bootstrap.

This is a new evaluation contract, not an alias for historical TableVI/eAP MiD.
RTE is abs(prediction - truth) / abs(truth) * 100, in percentage points.
All input rows, including missing-GT rows, are retained in SCORED_ROWS.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOOTSTRAP_METRICS = (
    "micro_rte_percent",
    "micro_mae_seconds",
    "micro_rmse_seconds",
    "macro_sequence_rte_percent",
    "macro_sequence_mae_seconds",
    "macro_sequence_rmse_seconds",
)
DEFAULT_SEED = 20261008


@dataclass(frozen=True)
class Row:
    """A source row, retained even if its target or prediction is unavailable."""

    source: dict[str, str]
    query_id: str
    sequence_id: str
    group_id: str
    truth: float
    predictions: dict[str, float]

    @property
    def gt_status(self) -> str:
        """Classify truth without consulting any prediction."""
        if not math.isfinite(self.truth):
            return "EXCLUDED_GT_NONFINITE"
        if self.truth == 0:
            return "EXCLUDED_GT_ZERO"
        return "ELIGIBLE"


@dataclass(frozen=True)
class Loss:
    """Per-query losses in the new, explicit metric definition."""

    absolute: float
    squared: float
    rte: float


@dataclass(frozen=True)
class Cluster:
    """Sufficient statistics for a cluster of complete, GT-eligible sequences."""

    queries: int
    sequences: int
    absolute: float
    squared: float
    rte: float
    sequence_mae_sum: float
    sequence_rmse_sum: float
    sequence_rte_sum: float


def _number(raw: str, *, column: str, query_id: str) -> float:
    if raw.strip().lower() in {"", "none", "null", "na", "n/a"}:
        return math.nan
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in {column}, query {query_id}: {raw!r}") from exc


def read_rows(
    path: Path, methods: Sequence[str], group_column: str | None = None
) -> tuple[list[str], list[Row]]:
    """Validate unique query IDs and a single bootstrap group per sequence."""
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Specify at least one distinct, explicit model column")
    reserved = {"query_id", "sequence_id", "truth_ttc_seconds"}
    if reserved.intersection(methods):
        raise ValueError("Model columns must not be ID or truth columns")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None or len(fields) != len(set(fields)):
            raise ValueError("CSV header missing or contains duplicate columns")
        required = reserved | set(methods) | ({group_column} if group_column else set())
        if missing := required.difference(fields):
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        seen: set[str] = set()
        groups: dict[str, str] = {}
        rows: list[Row] = []
        for source in reader:
            if None in source or any(value is None for value in source.values()):
                raise ValueError("Malformed CSV row: field count differs from header")
            query_id, sequence_id = source["query_id"], source["sequence_id"]
            group_id = source[group_column] if group_column else sequence_id
            if not query_id.strip() or not sequence_id.strip() or not group_id.strip():
                raise ValueError("Query, sequence and bootstrap group IDs must be nonempty")
            if query_id in seen:
                raise ValueError(f"Duplicate query_id: {query_id}")
            if sequence_id in groups and groups[sequence_id] != group_id:
                raise ValueError(f"Sequence {sequence_id} spans multiple bootstrap groups")
            groups[sequence_id] = group_id
            seen.add(query_id)
            rows.append(
                Row(
                    source=source,
                    query_id=query_id,
                    sequence_id=sequence_id,
                    group_id=group_id,
                    truth=_number(
                        source["truth_ttc_seconds"], column="truth_ttc_seconds", query_id=query_id
                    ),
                    predictions={
                        method: _number(source[method], column=method, query_id=query_id)
                        for method in methods
                    },
                )
            )
    if not rows:
        raise ValueError("CSV has no query rows")
    return list(fields), rows


def query_loss(row: Row, method: str) -> Loss | None:
    """Return losses only for eligible GT and finite prediction/arithmetic."""
    prediction = row.predictions[method]
    if row.gt_status != "ELIGIBLE" or not math.isfinite(prediction):
        return None
    absolute = abs(prediction - row.truth)
    squared = absolute * absolute
    rte = absolute / abs(row.truth) * 100
    if not all(math.isfinite(value) for value in (absolute, squared, rte)):
        return None
    return Loss(absolute, squared, rte)


def _metrics(losses: Sequence[Loss]) -> dict[str, float] | None:
    if not losses:
        return None
    count = len(losses)
    return {
        "rte_percent": math.fsum(loss.rte / count for loss in losses),
        "mae_seconds": math.fsum(loss.absolute / count for loss in losses),
        "median_absolute_error_seconds": statistics.median(loss.absolute for loss in losses),
        "rmse_seconds": math.sqrt(math.fsum(loss.squared / count for loss in losses)),
    }


def _coverage(rows: Sequence[Row], method: str) -> dict[str, Any]:
    eligible = [row for row in rows if row.gt_status == "ELIGIBLE"]
    losses = [loss for row in eligible if (loss := query_loss(row, method)) is not None]
    prediction_failures = sum(not math.isfinite(row.predictions[method]) for row in eligible)
    arithmetic_failures = len(eligible) - len(losses) - prediction_failures
    return {
        "total_rows": len(rows),
        "gt_eligible": len(eligible),
        "gt_nonfinite": sum(row.gt_status == "EXCLUDED_GT_NONFINITE" for row in rows),
        "gt_zero": sum(row.gt_status == "EXCLUDED_GT_ZERO" for row in rows),
        "prediction_nonfinite_on_eligible_gt": prediction_failures,
        "arithmetic_nonfinite_on_eligible_gt": arithmetic_failures,
        "scorable_predictions": len(losses),
        "coverage_on_eligible_gt": len(losses) / len(eligible) if eligible else None,
        "failure_rate_on_eligible_gt": 1 - len(losses) / len(eligible) if eligible else None,
        "status": (
            "NO_ELIGIBLE_GT"
            if not eligible
            else "COMPLETE_COHORT_AVAILABLE"
            if len(losses) == len(eligible)
            else "COMPLETE_COHORT_UNAVAILABLE_PREDICTION_FAILURE"
        ),
        "complete_cohort_metrics": _metrics(losses) if len(losses) == len(eligible) else None,
        "conditional_on_scorable_predictions_metrics": _metrics(losses),
    }


def _cluster_metrics(clusters: Iterable[Cluster]) -> dict[str, float]:
    items = list(clusters)
    queries = sum(item.queries for item in items)
    sequences = sum(item.sequences for item in items)
    return {
        "micro_rte_percent": math.fsum(item.rte / queries for item in items),
        "micro_mae_seconds": math.fsum(item.absolute / queries for item in items),
        "micro_rmse_seconds": math.sqrt(math.fsum(item.squared / queries for item in items)),
        "macro_sequence_rte_percent": math.fsum(
            item.sequence_rte_sum / sequences for item in items
        ),
        "macro_sequence_mae_seconds": math.fsum(
            item.sequence_mae_sum / sequences for item in items
        ),
        "macro_sequence_rmse_seconds": math.fsum(
            item.sequence_rmse_sum / sequences for item in items
        ),
    }


def _clusters(rows: Sequence[Row], method: str) -> dict[str, Cluster]:
    by_sequence: dict[tuple[str, str], list[Loss]] = defaultdict(list)
    for row in rows:
        if row.gt_status != "ELIGIBLE":
            continue
        loss = query_loss(row, method)
        if loss is None:
            raise ValueError("Cannot construct complete-cohort bootstrap with missing predictions")
        by_sequence[(row.group_id, row.sequence_id)].append(loss)
    by_group: dict[str, list[list[Loss]]] = defaultdict(list)
    for (group_id, _), losses in by_sequence.items():
        by_group[group_id].append(losses)
    result = {}
    for group_id, sequences in by_group.items():
        losses = [loss for sequence in sequences for loss in sequence]
        result[group_id] = Cluster(
            queries=len(losses),
            sequences=len(sequences),
            absolute=math.fsum(loss.absolute for loss in losses),
            squared=math.fsum(loss.squared for loss in losses),
            rte=math.fsum(loss.rte for loss in losses),
            sequence_mae_sum=math.fsum(
                math.fsum(loss.absolute / len(sequence) for loss in sequence)
                for sequence in sequences
            ),
            sequence_rmse_sum=math.fsum(
                math.sqrt(math.fsum(loss.squared / len(sequence) for loss in sequence))
                for sequence in sequences
            ),
            sequence_rte_sum=math.fsum(
                math.fsum(loss.rte / len(sequence) for loss in sequence) for sequence in sequences
            ),
        )
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def score(
    rows: Sequence[Row], methods: Sequence[str], *, bootstrap_draws: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score fixed GT cohorts and all method pairs without prediction-based cohort selection."""
    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    sequence_ids = sorted({row.sequence_id for row in rows})
    summaries: dict[str, dict[str, Any]] = {}
    per_sequence: list[dict[str, Any]] = []
    complete: dict[str, dict[str, Cluster]] = {}
    for method in methods:
        summary = _coverage(rows, method)
        for sequence_id in sequence_ids:
            selected = [row for row in rows if row.sequence_id == sequence_id]
            record = {
                "method": method,
                "sequence_id": sequence_id,
                "group_id": selected[0].group_id,
                **_coverage(selected, method),
            }
            per_sequence.append(record)
        if summary["status"] == "COMPLETE_COHORT_AVAILABLE":
            clusters = _clusters(rows, method)
            complete[method] = clusters
            aggregated = _cluster_metrics(clusters.values())
            sequence_medians = []
            for sequence_id in sequence_ids:
                losses = [
                    loss
                    for row in rows
                    if row.sequence_id == sequence_id
                    and (loss := query_loss(row, method)) is not None
                ]
                if losses:
                    sequence_medians.append(statistics.median(loss.absolute for loss in losses))
            all_losses = [loss for row in rows if (loss := query_loss(row, method)) is not None]
            aggregated["micro_median_absolute_error_seconds"] = statistics.median(
                loss.absolute for loss in all_losses
            )
            aggregated["macro_sequence_median_absolute_error_seconds"] = statistics.mean(
                sequence_medians
            )
            summary["complete_cohort_aggregate"] = aggregated
        else:
            summary["complete_cohort_aggregate"] = None
        summaries[method] = summary

    group_ids = sorted({row.group_id for row in rows if row.gt_status == "ELIGIBLE"})
    rng = random.Random(seed)
    replicates: dict[str, dict[str, list[float]]] = {
        method: {metric: [] for metric in BOOTSTRAP_METRICS} for method in complete
    }
    draw_digest = hashlib.sha256()
    if len(group_ids) >= 2:
        for _ in range(bootstrap_draws):
            indices = [rng.randrange(len(group_ids)) for _ in group_ids]
            draw_digest.update((",".join(map(str, indices)) + "\n").encode("ascii"))
            for method, clusters in complete.items():
                metrics = _cluster_metrics(clusters[group_ids[index]] for index in indices)
                for metric in BOOTSTRAP_METRICS:
                    replicates[method][metric].append(metrics[metric])

    paired: list[dict[str, Any]] = []
    for first, second in itertools.combinations(methods, 2):
        comparison: dict[str, Any] = {
            "first": first,
            "second": second,
            "delta_direction": "first_minus_second; negative favors first",
            "cohort": "ALL_GT_ELIGIBLE_ROWS_NO_PREDICTION_FILTER",
            "gt_eligible": sum(row.gt_status == "ELIGIBLE" for row in rows),
            "bootstrap_groups": len(group_ids),
        }
        if first not in complete or second not in complete:
            comparison.update(status="UNAVAILABLE_INCOMPLETE_COHORT", metrics=None)
        else:
            first_point = _cluster_metrics(complete[first].values())
            second_point = _cluster_metrics(complete[second].values())
            differences = {}
            for metric in BOOTSTRAP_METRICS:
                delta = first_point[metric] - second_point[metric]
                draws = [
                    a - b
                    for a, b in zip(
                        replicates[first][metric], replicates[second][metric], strict=True
                    )
                ]
                differences[metric] = {
                    "point_delta": delta,
                    "percentile_ci95": [_quantile(draws, 0.025), _quantile(draws, 0.975)]
                    if draws
                    else None,
                    "draws": len(draws),
                }
            comparison.update(
                status="AVAILABLE" if len(group_ids) >= 2 else "POINT_ONLY_FEWER_THAN_TWO_GROUPS",
                metrics=differences,
            )
        paired.append(comparison)
    report: dict[str, Any] = {
        "schema": "explicit_signed_ttc_rte_v1",
        "metric_contract": {
            "rte_percent": "abs(prediction-truth)/abs(truth)*100",
            "micro": "Equal weight per GT-eligible query",
            "macro_sequence": "Equal weight per sequence with eligible GT; average sequence metric",
            "gt_eligibility": "finite truth and truth != 0; signed negative targets retained",
            "prediction_failure": (
                "Nonfinite prediction/arithmetic counts as failure, never removes GT row"
            ),
            "complete_metrics": "Unavailable if any eligible query fails for that method",
            "conditional_metrics": (
                "Descriptive only, conditional on finite computable prediction losses"
            ),
            "historical_compatibility": (
                "Not historical TableVI inverse-TTC-relative alias or eAP MiD"
            ),
            "bins": "No eAP bins or benchmark mixture weights",
        },
        "cohort": {
            "total_rows": len(rows),
            "sequences": len(sequence_ids),
            "gt_eligible": sum(row.gt_status == "ELIGIBLE" for row in rows),
            "gt_nonfinite": sum(row.gt_status == "EXCLUDED_GT_NONFINITE" for row in rows),
            "gt_zero": sum(row.gt_status == "EXCLUDED_GT_ZERO" for row in rows),
            "all_rows_retained": True,
            "selection_uses_predictions": False,
        },
        "methods": summaries,
        "bootstrap": {
            "seed": seed,
            "requested_draws": bootstrap_draws,
            "groups": group_ids,
            "unit": "Whole sequence or explicit group containing complete sequences",
            "scheme": "Paired cluster percentile bootstrap; same sampled clusters for every method",
            "draw_indices_sha256": draw_digest.hexdigest() if len(group_ids) >= 2 else None,
            "metrics": BOOTSTRAP_METRICS,
            "median_intervals": "Not computed; medians are descriptive point metrics",
            "limitations": (
                "Descriptive intervals, no multiplicity correction or independence certification"
            ),
        },
        "paired": paired,
    }
    return report, per_sequence, paired


def _write_csv(path: Path, records: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            result.update({f"{key}__{nested}": item for nested, item in value.items()})
        else:
            result[key] = value
    return result


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(
    input_csv: Path,
    output_dir: Path,
    methods: Sequence[str],
    *,
    bootstrap_draws: int = 2000,
    seed: int = DEFAULT_SEED,
    group_column: str | None = None,
) -> dict[str, Any]:
    """Write reproducible scores, all source rows, paired comparisons and hashes."""
    names = {
        "REPORT.json",
        "METHOD_METRICS.csv",
        "PER_SEQUENCE.csv",
        "PAIRED.csv",
        "SCORED_ROWS.csv",
        "METADATA.json",
        "SHA256.json",
    }
    if input_csv.resolve() in {(output_dir / name).resolve() for name in names}:
        raise ValueError("Output would overwrite the input CSV")
    input_sha = hashlib.sha256(input_csv.read_bytes()).hexdigest()
    fields, rows = read_rows(input_csv, methods, group_column)
    extra = ["scorer_gt_status"] + [
        f"{method}__{suffix}"
        for method in methods
        for suffix in ["status", "absolute_error_seconds", "rte_percent"]
    ]
    if set(extra).intersection(fields):
        raise ValueError("Input already contains scorer output columns")
    report, per_sequence, paired = score(rows, methods, bootstrap_draws=bootstrap_draws, seed=seed)
    if hashlib.sha256(input_csv.read_bytes()).hexdigest() != input_sha:
        raise RuntimeError("Input CSV changed during scoring")
    config = {
        "methods": list(methods),
        "bootstrap_draws": bootstrap_draws,
        "seed": seed,
        "group_column": group_column,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    _json(output_dir / "REPORT.json", report)
    methods_report = report["methods"]
    assert isinstance(methods_report, dict)
    method_rows = [
        _flatten({"method": name, **summary}) for name, summary in methods_report.items()
    ]
    sequence_rows = [_flatten(record) for record in per_sequence]
    for name, records in [("METHOD_METRICS.csv", method_rows), ("PER_SEQUENCE.csv", sequence_rows)]:
        columns = list(dict.fromkeys(key for record in records for key in record))
        _write_csv(output_dir / name, records, columns)
    paired_rows = []
    for comparison in paired:
        metrics = comparison["metrics"]
        if isinstance(metrics, dict):
            for metric, values in metrics.items():
                interval = values["percentile_ci95"]
                paired_rows.append(
                    {
                        "first": comparison["first"],
                        "second": comparison["second"],
                        "status": comparison["status"],
                        "metric": metric,
                        "first_minus_second": values["point_delta"],
                        "ci95_lower": interval[0] if interval else None,
                        "ci95_upper": interval[1] if interval else None,
                        "draws": values["draws"],
                    }
                )
        else:
            paired_rows.append(
                {
                    "first": comparison["first"],
                    "second": comparison["second"],
                    "status": comparison["status"],
                }
            )
    _write_csv(
        output_dir / "PAIRED.csv",
        paired_rows,
        [
            "first",
            "second",
            "status",
            "metric",
            "first_minus_second",
            "ci95_lower",
            "ci95_upper",
            "draws",
        ],
    )
    retained = []
    for row in rows:
        record: dict[str, Any] = {**row.source, "scorer_gt_status": row.gt_status}
        for method in methods:
            loss = query_loss(row, method)
            record[f"{method}__status"] = (
                row.gt_status
                if row.gt_status != "ELIGIBLE"
                else "SCORED"
                if loss is not None
                else "PREDICTION_OR_ARITHMETIC_NONFINITE"
            )
            record[f"{method}__absolute_error_seconds"] = loss.absolute if loss else None
            record[f"{method}__rte_percent"] = loss.rte if loss else None
        retained.append(record)
    _write_csv(output_dir / "SCORED_ROWS.csv", retained, fields + extra)
    _json(
        output_dir / "METADATA.json",
        {
            "input_filename": input_csv.name,
            "input_sha256": input_sha,
            "input_bytes": input_csv.stat().st_size,
            "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "configuration": config,
            "configuration_sha256": config_hash,
            "optimizer_updates": 0,
            "inference_runs": 0,
            "source_mutated": False,
            "no_timestamp": (
                "Outputs intentionally omit wall-clock time for deterministic regeneration"
            ),
        },
    )
    _json(
        output_dir / "SHA256.json",
        {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in sorted(names - {"SHA256.json"})
        },
    )
    return report


def main() -> None:
    """Score an existing CSV, requiring explicit prediction columns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--group-column", help="Optional group column; one group per whole sequence"
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = run(
        args.input,
        args.output_dir,
        args.models,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        group_column=args.group_column,
    )
    print(json.dumps({"cohort": result["cohort"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
