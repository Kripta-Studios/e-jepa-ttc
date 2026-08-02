"""Aggregate existing high-resolution evidence without running experiments.

The input files are immutable evidence.  This script only reads the existing
S3/S4/S5 smoke, temporal sweep, token-scaling benchmark, and real screen
artifacts.  It deliberately keeps the real-screen decision as the source of
truth: S5/KDA is rejected when that artifact reports a regression.  Historical
K1 Object-KDA evidence is not an input to this matrix and is never included in
its statistics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_ARMS = {
    "architecture_smoke": (
        "S3_R4_WINDOW_TEMPORAL",
        "S4_R4_WINDOW_MERGE_TEMPORAL",
        "S5_R4_WINDOW_MERGE_KDA",
    ),
    "temporal_sweep": (
        "S4_R4_WINDOW_MERGE_TEMPORAL",
        "S5_R4_WINDOW_MERGE_KDA",
    ),
    "real_screen": (
        "S3_R4_WINDOW_TEMPORAL",
        "S4_R4_WINDOW_MERGE_TEMPORAL",
        "S5_R4_WINDOW_MERGE_KDA",
    ),
    "token_scaling": ("R1", "R2", "R3", "R4"),
}
EXPECTED_TEMPORAL_STEPS = (2, 5, 8, 16, 32)

_REAL_COMPARISON_METRICS = (
    "paper_MiD_overall",
    "weighted_RTE_pct",
    "failure_rate_pct",
)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """Return the artifact hash convention used by the high-res runners."""

    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"High-resolution evidence artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in high-resolution artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"High-resolution artifact must contain a JSON object: {path}")
    return value


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _source_record(path: Path, payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    declared_hash = payload.get("artifact_sha256")
    canonical_hash = _canonical_sha256(payload)
    row_selection_disabled = (
        payload.get("artifact_type") == "highres_token_scaling_benchmark_v1"
        and isinstance(payload.get("results"), list)
        and all(
            isinstance(row, dict) and row.get("selection_allowed") is not True
            for row in payload["results"]
        )
    )
    return {
        "path": _display_path(path, repo_root),
        "file_sha256": _file_sha256(path),
        "declared_artifact_sha256": declared_hash,
        "canonical_artifact_sha256": canonical_hash,
        "artifact_sha256_valid": declared_hash == canonical_hash,
        "artifact_type": payload.get("artifact_type"),
        "schema_version": payload.get("schema_version"),
        "code_commit": payload.get("code_commit"),
        "protocol_version": payload.get("protocol_version"),
        "status": payload.get("status"),
        "selection_allowed": payload.get("selection_allowed"),
        "selection_disabled_in_rows": row_selection_disabled,
    }


def _validate_source(name: str, payload: dict[str, Any], record: dict[str, Any]) -> None:
    errors: list[str] = []
    if payload.get("status") != "pass":
        errors.append(f"status={payload.get('status')!r}")
    selection_disabled_in_rows = record["selection_disabled_in_rows"] is True
    if payload.get("selection_allowed") is not False and not selection_disabled_in_rows:
        errors.append("selection_allowed must be false")
    if record["artifact_sha256_valid"] is not True:
        errors.append("artifact_sha256 does not match the canonical payload")
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"Source contract failed for {name}: {joined}")


def _rows(payload: dict[str, Any], field: str, name: str) -> list[dict[str, Any]]:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"{name}.{field} must be a list of JSON objects")
    return [copy.deepcopy(row) for row in value]


def _arm_key(row: dict[str, Any], name: str) -> str:
    key_field = "name" if name == "token_scaling" else "arm"
    value = row.get(key_field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} row has no non-empty {key_field}")
    return value


def _select_rows(
    name: str, rows: list[dict[str, Any]], expected: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    excluded: list[str] = []
    seen: set[tuple[str, str]] = set()
    seen_arms: set[str] = set()
    expected_set = set(expected)
    for row in rows:
        key = _arm_key(row, name)
        if key in expected_set:
            # The temporal sweep intentionally contains one row per arm and
            # temporal length.  The other evidence tables contain one row per
            # arm, so their duplicate contract remains strict.
            identity = (
                key,
                str(row.get("temporal_steps")) if name == "temporal_sweep" else "single",
            )
            if identity in seen:
                raise ValueError(f"Duplicate {name} row for {key} at {identity[1]}")
            selected.append(row)
            seen.add(identity)
            seen_arms.add(key)
        else:
            excluded.append(key)
    missing = [key for key in expected if key not in seen_arms]
    if missing:
        raise ValueError(f"Missing {name} rows: {', '.join(missing)}")
    if name == "temporal_sweep":
        present_steps = {
            (_arm_key(row, name), int(row["temporal_steps"]))
            for row in selected
            if isinstance(row.get("temporal_steps"), int)
        }
        missing_steps = [
            f"{arm}@T{steps}"
            for arm in expected
            for steps in EXPECTED_TEMPORAL_STEPS
            if (arm, steps) not in present_steps
        ]
        if missing_steps:
            raise ValueError(f"Missing temporal sweep rows: {', '.join(missing_steps)}")
    return selected, excluded


def _find_text_tokens(value: object, tokens: tuple[str, ...]) -> list[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.update(_find_text_tokens(key, tokens))
            found.update(_find_text_tokens(child, tokens))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_text_tokens(child, tokens))
    elif isinstance(value, str):
        lowered = value.casefold()
        for token in tokens:
            if token.casefold() in lowered:
                found.add(token)
    return sorted(found)


def _real_metric_view(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"Real-screen row {row.get('arm')!r} has no validation_metrics object")
    missing = [metric for metric in _REAL_COMPARISON_METRICS if metric not in metrics]
    if missing:
        raise ValueError(
            f"Real-screen row {row.get('arm')!r} is missing signed metrics: {', '.join(missing)}"
        )
    return {metric: metrics[metric] for metric in _REAL_COMPARISON_METRICS}


def _real_screen_decision(rows: list[dict[str, Any]], source_comparison: object) -> dict[str, Any]:
    if not isinstance(source_comparison, dict):
        raise TypeError("Real screen must contain an s4_vs_s5 comparison object")
    comparison: dict[str, Any] = source_comparison
    decision = comparison.get("decision")
    if decision != "regression_in_short_screen":
        raise ValueError(
            "The real high-resolution screen does not report the required "
            f"regression_in_short_screen decision: {decision!r}"
        )
    by_arm = {row["arm"]: row for row in rows}
    s4_metrics = _real_metric_view(by_arm["S4_R4_WINDOW_MERGE_TEMPORAL"])
    s5_metrics = _real_metric_view(by_arm["S5_R4_WINDOW_MERGE_KDA"])
    gate_axes: dict[str, dict[str, Any]] = {}
    regressed_metrics: list[str] = []
    for metric in _REAL_COMPARISON_METRICS:
        s4_value = s4_metrics[metric]
        s5_value = s5_metrics[metric]
        if not isinstance(s4_value, (int, float)) or not isinstance(s5_value, (int, float)):
            raise TypeError(f"Real-screen metric {metric} must be numeric")
        not_worse = s5_value <= s4_value
        gate_axes[metric] = {
            "s4": s4_value,
            "s5": s5_value,
            "s5_not_worse": not_worse,
        }
        if not not_worse:
            regressed_metrics.append(metric)
    if not regressed_metrics:
        raise ValueError(
            "The real screen reports a regression but no signed metric worsens; "
            "refusing to manufacture a rejection reason."
        )
    return {
        "source_decision": decision,
        "status": "rejected",
        "promotion_allowed": False,
        "regressed_metrics": regressed_metrics,
        "gate_axes": gate_axes,
        "source_comparison": copy.deepcopy(comparison),
        "reason": (
            "S5_R4_WINDOW_MERGE_KDA is rejected because the real short screen "
            "reports regression_in_short_screen under its predeclared S4 comparison gate."
        ),
    }


def _r4_safety(token_rows: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in token_rows}
    r4 = by_name["R4"]
    global_allocated = payload.get("global_attention_over_r4_allocated")
    required = r4.get("theoretical_oom_guard_required")
    triggered = r4.get("theoretical_oom_guard_triggered")
    if global_allocated is not False or required is not True or triggered is not True:
        raise ValueError(
            "R4 token-scaling evidence does not prove the required pre-allocation OOM guard"
        )
    return {
        "global_attention_over_r4_allocated": global_allocated,
        "r4_theoretical_oom_guard_required": required,
        "r4_theoretical_oom_guard_triggered": triggered,
        "r4_global_guard_error": r4.get("global_guard_error"),
        "all_rows_oom_diagnostics_consistent": all(
            row.get("theoretical_oom_guard_required") == row.get("theoretical_oom_guard_triggered")
            for row in token_rows
        ),
    }


def aggregate_highres_matrix(
    *,
    architecture_screen_path: Path,
    temporal_sweep_path: Path,
    token_scaling_path: Path,
    real_screen_path: Path,
    repo_root: Path,
    code_commit: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable Phase 7B summary from existing artifacts only."""

    source_paths = {
        "architecture_smoke": architecture_screen_path,
        "temporal_sweep": temporal_sweep_path,
        "token_scaling": token_scaling_path,
        "real_screen": real_screen_path,
    }
    payloads = {name: _load_json(path) for name, path in source_paths.items()}
    records = {
        name: _source_record(source_paths[name], payloads[name], repo_root) for name in source_paths
    }
    for name, payload in payloads.items():
        _validate_source(name, payload, records[name])

    architecture_rows, architecture_excluded = _select_rows(
        "architecture_smoke",
        _rows(payloads["architecture_smoke"], "results", "architecture_smoke"),
        EXPECTED_ARMS["architecture_smoke"],
    )
    temporal_rows, temporal_excluded = _select_rows(
        "temporal_sweep",
        _rows(payloads["temporal_sweep"], "results", "temporal_sweep"),
        EXPECTED_ARMS["temporal_sweep"],
    )
    token_rows, token_excluded = _select_rows(
        "token_scaling",
        _rows(payloads["token_scaling"], "results", "token_scaling"),
        EXPECTED_ARMS["token_scaling"],
    )
    real_rows, real_excluded = _select_rows(
        "real_screen",
        _rows(payloads["real_screen"], "arms", "real_screen"),
        EXPECTED_ARMS["real_screen"],
    )

    historical_tokens = _find_text_tokens(
        payloads,
        ("K1_OBJECT_KDA", "K2_ALIGNED_PATCH_KDA", "object_kda"),
    )
    excluded_arms = sorted(
        set(architecture_excluded + temporal_excluded + token_excluded + real_excluded)
    )
    if "K1_OBJECT_KDA" in excluded_arms:
        historical_tokens = sorted(set(historical_tokens + ["K1_OBJECT_KDA"]))

    real_comparison = _real_screen_decision(real_rows, payloads["real_screen"].get("s4_vs_s5"))
    result: dict[str, Any] = {
        "artifact_type": "highres_matrix_aggregate_v1",
        "schema_version": "v1",
        "evidence_type": "existing_highres_artifacts_only",
        "code_commit": code_commit if code_commit is not None else _git_commit(repo_root),
        "protocol_version": "highres_s3_s4_s5_matrix_v1",
        "created_at": generated_at if generated_at is not None else datetime.now(UTC).isoformat(),
        "selection_allowed": False,
        "training_executed": False,
        "inputs": records,
        "matrix": {
            "architecture_smoke": {
                "metrics_scope": payloads["architecture_smoke"].get("metrics_scope"),
                "rows": architecture_rows,
            },
            "temporal_sweep": {
                "metrics_scope": payloads["temporal_sweep"].get("metrics_scope"),
                "rows": temporal_rows,
            },
            "token_scaling": {
                "metrics_scope": payloads["token_scaling"].get("evidence_type"),
                "rows": token_rows,
            },
            "real_screen": {
                "metrics_scope": payloads["real_screen"].get("arms", [{}])[0].get("metrics_scope"),
                "rows": real_rows,
                "s4_vs_s5": copy.deepcopy(payloads["real_screen"].get("s4_vs_s5")),
            },
        },
        "decisions": {
            "s5_kda": real_comparison,
            "historical_k1": {
                "arm": "K1_OBJECT_KDA",
                "included_in_matrix": False,
                "mixed_with_s3_s4_s5": False,
                "source_read": False,
                "detected_only_as_excluded_text": "K1_OBJECT_KDA" in historical_tokens,
                "note": (
                    "Historical K1 Object-KDA remains a separate negative result and is not "
                    "read, aggregated, or combined statistically with this high-resolution matrix."
                ),
            },
        },
        "safety_gates": {
            "source_status_and_hash_contracts_green": True,
            "selection_disabled_in_all_sources": all(
                record["selection_allowed"] is False or record["selection_disabled_in_rows"] is True
                for record in records.values()
            ),
            "r4_global_attention_not_allocated": _r4_safety(token_rows, payloads["token_scaling"]),
            "k1_historical_separate": True,
            "s5_kda_promoted": False,
        },
        "excluded_arm_ids": excluded_arms,
        "historical_tokens_detected_in_read_sources": historical_tokens,
        "status": "pass",
    }
    result["artifact_sha256"] = _canonical_sha256(result)
    return result


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main() -> int:
    """Run the read-only aggregation CLI."""

    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture-screen",
        type=Path,
        default=Path("artifacts/benchmarks/highres_architecture_screen_v1.json"),
    )
    parser.add_argument(
        "--temporal-sweep",
        type=Path,
        default=Path("artifacts/benchmarks/highres_temporal_sweep_v1.json"),
    )
    parser.add_argument(
        "--token-scaling",
        type=Path,
        default=Path("artifacts/benchmarks/highres_token_scaling_v1.json"),
    )
    parser.add_argument(
        "--real-screen",
        type=Path,
        default=Path("artifacts/benchmarks/highres_real_screen_v1_gpu_workers4/screen.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Without it, the summary is printed only.",
    )
    args = parser.parse_args()
    try:
        result = aggregate_highres_matrix(
            architecture_screen_path=_resolve(args.architecture_screen, repo_root),
            temporal_sweep_path=_resolve(args.temporal_sweep, repo_root),
            token_scaling_path=_resolve(args.token_scaling, repo_root),
            real_screen_path=_resolve(args.real_screen, repo_root),
            repo_root=repo_root,
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"high-resolution aggregation failed: {exc}", file=sys.stderr)
        return 1

    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        output = _resolve(args.output, repo_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
