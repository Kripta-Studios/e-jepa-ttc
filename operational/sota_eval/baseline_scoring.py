"""Score fixed-query geometric pilots against already-exposed development truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "sota_geometric_baseline_pilot_scoring_v1"
LEGACY_METHODS = (
    "H8_seed7",
    "H8_seed13",
    "H8_seed23",
    "public_Garl_event_lhr",
    "public_Garl_rgb_event_full",
)
ALL_METHODS = ("cmax_reference", "strttc_adapted", *LEGACY_METHODS)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_sha256(path: Path) -> str:
    """Hash one file or a directory's relative names and physical file hashes."""

    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot bind empty input directory: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_signed(document: Mapping[str, Any]) -> None:
    payload = dict(document)
    expected = payload.pop("artifact_sha256", None)
    if not isinstance(expected, str) or _canonical_sha256(payload) != expected:
        raise ValueError("signed fragment canonical SHA-256 mismatch")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _first_per_sequence(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("expanded manifest lacks rows")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("manifest row must be an object")
        sequence_id = str(raw.get("sequence_id"))
        if sequence_id not in seen:
            seen.add(sequence_id)
            selected.append(raw)
    return selected


def _validate_query_identity(
    expanded_manifest: Path,
    historical_manifest: Path,
) -> list[dict[str, Any]]:
    expanded = _read_json(expanded_manifest)
    historical = _read_json(historical_manifest)
    if expanded.get("status") != "LABEL_FREE_MANIFEST":
        raise ValueError("expanded queries are not a label-free manifest")
    if historical.get("status") != "LABEL_FREE_MANIFEST":
        raise ValueError("historical queries are not a label-free manifest")
    selected = _first_per_sequence(expanded)
    if len(selected) != 32:
        raise ValueError(f"expected 32 fixed sequence queries, found {len(selected)}")
    historical_rows = historical.get("rows")
    if not isinstance(historical_rows, list):
        raise ValueError("historical manifest lacks rows")
    historical_by_id = {
        str(row["query_id"]): row for row in historical_rows if isinstance(row, dict)
    }
    for row in selected:
        query_id = str(row["query_id"])
        old = historical_by_id.get(query_id)
        if old is None:
            raise ValueError(f"fixed query is absent from historical manifest: {query_id}")
        for field in ("query_id", "sequence_id", "anchor_us", "metadata_sha256"):
            if row.get(field) != old.get(field):
                raise ValueError(f"{query_id}: historical identity mismatch for {field}")
    return selected


def _read_fragments(directory: Path, expected_schema: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        document = _read_json(path)
        _verify_signed(document)
        if document.get("schema") != expected_schema:
            raise ValueError(f"unexpected fragment schema in {path}")
        query_id = str(document.get("query_id"))
        if query_id in result:
            raise ValueError(f"duplicate fragment query_id: {query_id}")
        result[query_id] = document
    return result


def _read_exposed_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "query_id",
            "sequence_id",
            "anchor_us",
            "truth_ttc_seconds",
            *LEGACY_METHODS,
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("historical scored predictions lack required columns")
        rows = list(reader)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        query_id = row["query_id"]
        if query_id in result:
            raise ValueError(f"duplicate historical score row: {query_id}")
        result[query_id] = row
    return result


def _finite_or_none(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(str(value))
    return parsed if math.isfinite(parsed) else None


def _metrics(truth: list[float], prediction: list[float]) -> dict[str, float]:
    errors = [predicted - expected for expected, predicted in zip(truth, prediction, strict=True)]
    absolute = [abs(value) for value in errors]
    return {
        "mae_seconds": statistics.fmean(absolute),
        "rmse_seconds": math.sqrt(statistics.fmean(value * value for value in errors)),
        "median_absolute_error_seconds": statistics.median(absolute),
        "signed_bias_seconds": statistics.fmean(errors),
    }


def _score_method(rows: list[dict[str, object]], method: str) -> dict[str, Any]:
    truth_exposed = [row for row in rows if row["truth_ttc_seconds"] != ""]
    predicted_all = [row for row in rows if row[method] != ""]
    conditional = [row for row in truth_exposed if row[method] != ""]
    conditional_truth = [float(str(row["truth_ttc_seconds"])) for row in conditional]
    conditional_prediction = [float(str(row[method])) for row in conditional]
    complete = len(conditional) == len(truth_exposed)
    return {
        "cohort_queries": len(rows),
        "truth_exposed_queries": len(truth_exposed),
        "predictions_all_queries": len(predicted_all),
        "prediction_coverage_all_queries": len(predicted_all) / len(rows),
        "predictions_on_exposed_truth": len(conditional),
        "prediction_coverage_on_exposed_truth": (
            len(conditional) / len(truth_exposed) if truth_exposed else None
        ),
        "complete_exposed_cohort_metrics": (
            _metrics(conditional_truth, conditional_prediction) if complete else None
        ),
        "complete_exposed_cohort_status": (
            "AVAILABLE" if complete else "N/A_INCOMPLETE_PREDICTION_COVERAGE"
        ),
        "conditional_success_metrics": (
            _metrics(conditional_truth, conditional_prediction) if conditional else None
        ),
        "conditional_support": len(conditional),
        "conditional_metrics_not_rankable": not complete,
    }


def score_pilot(
    *,
    expanded_manifest: str | Path,
    historical_manifest: str | Path,
    exposed_predictions: str | Path,
    cmax_fragments: str | Path,
    strttc_fragments: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Join exact query identities and score without opening any new labels."""

    paths = {
        "expanded_manifest": Path(expanded_manifest).resolve(strict=True),
        "historical_manifest": Path(historical_manifest).resolve(strict=True),
        "exposed_predictions": Path(exposed_predictions).resolve(strict=True),
        "cmax_fragments": Path(cmax_fragments).resolve(strict=True),
        "strttc_fragments": Path(strttc_fragments).resolve(strict=True),
    }
    selected = _validate_query_identity(paths["expanded_manifest"], paths["historical_manifest"])
    exposed = _read_exposed_rows(paths["exposed_predictions"])
    cmax = _read_fragments(
        paths["cmax_fragments"], "evttc_cmax_reference_equation_v1"
    )
    strttc = _read_fragments(
        paths["strttc_fragments"], "evttc_fixed_query_geometric_baselines_v1"
    )
    output_rows: list[dict[str, object]] = []
    for query in selected:
        query_id = str(query["query_id"])
        source = exposed.get(query_id)
        if source is None:
            raise ValueError(f"fixed query absent from exposed historical scores: {query_id}")
        if source["sequence_id"] != query["sequence_id"]:
            raise ValueError(f"{query_id}: sequence identity differs from exposed scores")
        if int(source["anchor_us"]) != int(query["anchor_us"]):
            raise ValueError(f"{query_id}: anchor differs from exposed scores")
        cmax_row = cmax.get(query_id)
        strttc_row = strttc.get(query_id)
        if cmax_row is None or strttc_row is None:
            raise ValueError(f"{query_id}: geometric fragment coverage is incomplete")
        for fragment in (cmax_row, strttc_row):
            if fragment["sequence_id"] != query["sequence_id"]:
                raise ValueError(f"{query_id}: fragment sequence mismatch")
            if int(fragment["anchor_us"]) != int(query["anchor_us"]):
                raise ValueError(f"{query_id}: fragment anchor mismatch")
            if fragment["query_metadata_sha256"] != query["metadata_sha256"]:
                raise ValueError(f"{query_id}: fragment metadata identity mismatch")
        output_row: dict[str, object] = {
            "query_id": query_id,
            "sequence_id": str(query["sequence_id"]),
            "anchor_us": int(query["anchor_us"]),
            "truth_ttc_seconds": source["truth_ttc_seconds"],
            "truth_status": "EXPOSED" if source["truth_ttc_seconds"] else "NOT_EXPOSED",
            "cmax_reference": cmax_row["prediction_ttc_s"] or "",
            "cmax_reference_status": cmax_row["status"],
            "cmax_reference_failure": cmax_row["failure"] or "",
            "strttc_adapted": strttc_row["prediction_ttc_s"] or "",
            "strttc_adapted_status": strttc_row["status"],
            "strttc_adapted_failure": strttc_row["failure"] or "",
        }
        for method in LEGACY_METHODS:
            parsed = _finite_or_none(source[method])
            output_row[method] = parsed if parsed is not None else ""
        output_rows.append(output_row)
    metrics = {method: _score_method(output_rows, method) for method in ALL_METHODS}
    output = Path(output_dir)
    row_fields = (
        "query_id",
        "sequence_id",
        "anchor_us",
        "truth_ttc_seconds",
        "truth_status",
        "cmax_reference",
        "cmax_reference_status",
        "cmax_reference_failure",
        "strttc_adapted",
        "strttc_adapted_status",
        "strttc_adapted_failure",
        *LEGACY_METHODS,
    )
    _atomic_csv(output / "SCORED_ROWS.csv", output_rows, row_fields)
    metric_rows = []
    for method, values in metrics.items():
        conditional = values["conditional_success_metrics"] or {}
        complete = values["complete_exposed_cohort_metrics"] or {}
        metric_rows.append(
            {
                "method": method,
                "cohort_queries": values["cohort_queries"],
                "truth_exposed_queries": values["truth_exposed_queries"],
                "prediction_coverage_all_queries": values["prediction_coverage_all_queries"],
                "prediction_coverage_on_exposed_truth": values[
                    "prediction_coverage_on_exposed_truth"
                ],
                "complete_exposed_cohort_status": values[
                    "complete_exposed_cohort_status"
                ],
                "complete_cohort_mae_seconds": complete.get("mae_seconds", ""),
                "conditional_support": values["conditional_support"],
                "conditional_mae_seconds": conditional.get("mae_seconds", ""),
                "conditional_rmse_seconds": conditional.get("rmse_seconds", ""),
                "conditional_median_absolute_error_seconds": conditional.get(
                    "median_absolute_error_seconds", ""
                ),
                "conditional_signed_bias_seconds": conditional.get(
                    "signed_bias_seconds", ""
                ),
                "conditional_metrics_not_rankable": values[
                    "conditional_metrics_not_rankable"
                ],
            }
        )
    _atomic_csv(output / "METHOD_METRICS.csv", metric_rows, tuple(metric_rows[0]))
    report = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scope": "32 fixed historical development queries, one per sequence",
        "query_identity": {
            "count": len(output_rows),
            "sequence_count": len({str(row["sequence_id"]) for row in output_rows}),
            "historical_manifest_identity_exact": True,
            "anchor_identity_exact": True,
            "metadata_sha256_identity_exact": True,
        },
        "truth": {
            "source": paths["exposed_predictions"].as_posix(),
            "already_exposed_only": True,
            "exposed_queries": sum(row["truth_status"] == "EXPOSED" for row in output_rows),
            "not_exposed_queries": sum(
                row["truth_status"] == "NOT_EXPOSED" for row in output_rows
            ),
            "new_test_labels_opened": False,
        },
        "methods": metrics,
        "geometric_contracts": {
            "cmax_reference": (
                "Corrected affine reference-time equation; 15/32 raw coverage. Local "
                "EvTTC observed-box adaptation, not a full paper reproduction."
            ),
            "strttc_adapted": (
                "Nonlinear source-port adaptation; 11/32 raw coverage. It does not "
                "reproduce MATLAB sequential lastOptimizedResult state, exact 0.4 s epochs, "
                "undistortion, bilateral filtering, or official 1000-iteration RANSAC."
            ),
        },
        "interpretation": (
            "Diagnostic pilot, not a SOTA ranking. Incomplete methods have N/A complete-cohort "
            "metrics; conditional metrics describe only successful predictions and cannot be "
            "ranked against complete-coverage methods."
        ),
        "inputs": {
            name: {
                "path": path.as_posix(),
                "sha256": _path_sha256(path),
            }
            for name, path in paths.items()
        },
        "outputs": {
            "scored_rows": "SCORED_ROWS.csv",
            "method_metrics": "METHOD_METRICS.csv",
        },
        "scientific_optimizer_updates": 0,
        "runner_source_sha256": _sha256(__file__),
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    _atomic_json(output / "REPORT.json", report)
    checksums = {
        name: _sha256(output / name)
        for name in ("SCORED_ROWS.csv", "METHOD_METRICS.csv", "REPORT.json")
    }
    _atomic_json(output / "SHA256.json", checksums)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--exposed-predictions", type=Path, required=True)
    parser.add_argument("--cmax-fragments", type=Path, required=True)
    parser.add_argument("--strttc-fragments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = score_pilot(
        expanded_manifest=args.expanded_manifest,
        historical_manifest=args.historical_manifest,
        exposed_predictions=args.exposed_predictions,
        cmax_fragments=args.cmax_fragments,
        strttc_fragments=args.strttc_fragments,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["score_pilot"]
