"""Verify evidence, rerun every published scorer, and rebuild all tables."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys, tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
SCORE_OUTPUTS = ("METADATA.json", "METHOD_METRICS.csv", "PAIRED.csv",
                 "PER_SEQUENCE.csv", "REPORT.json", "SCORED_ROWS.csv", "SHA256.json")
METRIC_COLUMNS = ("branch", "method", "total_rows", "gt_eligible",
    "coverage_on_eligible_gt", "failure_rate_on_eligible_gt", "status",
    "complete_cohort_metrics__rte_percent", "complete_cohort_metrics__mae_seconds",
    "complete_cohort_metrics__median_absolute_error_seconds",
    "complete_cohort_metrics__rmse_seconds",
    "conditional_on_scorable_predictions_metrics__rte_percent",
    "conditional_on_scorable_predictions_metrics__mae_seconds",
    "conditional_on_scorable_predictions_metrics__median_absolute_error_seconds",
    "conditional_on_scorable_predictions_metrics__rmse_seconds")
COST_COLUMNS = ("system", "stage", "count", "mean_ms", "p50_ms", "p95_ms",
                "minimum_ms", "maximum_ms", "samples_per_second_from_mean")
BASELINE_COVERAGE_COLUMNS = ("variant", "equation", "requested", "successful",
    "failed_retained", "success_fraction", "interpretation")
PILOT_CONDITIONAL_COLUMNS = ("method", "cohort_queries", "truth_exposed_queries",
    "prediction_coverage_all_queries", "prediction_coverage_on_exposed_truth",
    "complete_exposed_cohort_status", "conditional_support", "conditional_mae_seconds",
    "conditional_rmse_seconds", "conditional_median_absolute_error_seconds",
    "conditional_signed_bias_seconds", "conditional_metrics_not_rankable")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]

def write_csv(path: Path, values, columns) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(values)

def verify_snapshot() -> None:
    manifest = ROOT / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise SystemExit("missing top-level SHA256SUMS.txt")
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise SystemExit("malformed top-level SHA256SUMS.txt")
        expected, name = fields
        relative = PurePosixPath(name)
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or "\\" in name or name in seen):
            raise SystemExit(f"unsafe or duplicate snapshot path: {name}")
        seen.add(name)
        path = (ROOT / Path(*relative.parts)).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file() or digest(path) != expected:
            raise SystemExit(f"snapshot hash mismatch: {name}")

def rerun(job, scorer: Path) -> None:
    reference, input_csv = ROOT / job["reference"], ROOT / job["input"]
    config = read_json(reference / "METADATA.json")["configuration"]
    with tempfile.TemporaryDirectory(prefix="sota_scoring_verify_") as temporary:
        command = [sys.executable, str(scorer), "--input", str(input_csv),
                   "--output-dir", temporary, "--models", *config["methods"],
                   "--bootstrap-draws", str(config["bootstrap_draws"]),
                   "--seed", str(config["seed"])]
        if config.get("group_column"):
            command += ["--group-column", config["group_column"]]
        run = subprocess.run(command, check=False, capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(f"scoring regeneration failed for {job['branch']}: {run.stderr}")
        for name in SCORE_OUTPUTS:
            if digest(Path(temporary) / name) != digest(reference / name):
                raise SystemExit(f"scoring regeneration differs for {job['branch']}: {name}")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify_snapshot()
    inventory = read_json(ROOT / "SOURCE_INVENTORY.json")
    for item in inventory["files"]:
        path = ROOT / item["published_path"]
        if not path.is_file() or digest(path) != item["published_sha256"]:
            raise SystemExit(f"hash mismatch: {path}")
    regeneration = read_json(ROOT / "tables" / "REGENERATION.json")
    scorer = ROOT / regeneration["scorer"]
    for job in regeneration["scoring_jobs"]:
        rerun(job, scorer)
    temporary_tables = tempfile.TemporaryDirectory(prefix="sota_tables_verify_")
    generated = Path(temporary_tables.name)
    state = read_json(ROOT / "NEXT_DECISION.json")
    branch_rows = state["branches"]
    target = generated / "branch_status.csv"
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("branch", "status", "complete", "detail"))
        for row in branch_rows:
            writer.writerow((row["name"], row["status"],
                             str(row["complete"]).lower(), row["detail"]))
    metric_rows = []
    for job in regeneration["scoring_jobs"]:
        for source in rows(ROOT / job["reference"] / "METHOD_METRICS.csv"):
            metric_rows.append({
                column: job["branch"] if column == "branch" else source.get(column, "")
                for column in METRIC_COLUMNS})
    write_csv(generated / "metrics.csv", metric_rows, METRIC_COLUMNS)
    cost_rows = []
    cost_path = ROOT / "evidence" / "cost" / "SYSTEM_COST_SUMMARY.json"
    if cost_path.is_file():
        for system, stages in read_json(cost_path)["metrics"].items():
            for stage in ("cpu_prepare", "gpu_inference", "sequential_end_to_end"):
                values = stages.get(stage, {})
                cost_rows.append({
                    column: system if column == "system" else
                    stage if column == "stage" else values.get(column, "")
                    for column in COST_COLUMNS})
    write_csv(generated / "system_cost.csv", cost_rows, COST_COLUMNS)
    failure_rows = []
    failure_path = ROOT / "evidence" / "baselines" / "FAILURE_COUNTS.json"
    if failure_path.is_file():
        failure_source = read_json(failure_path)
        for method, reasons in failure_source.get("failure_counts", {}).items():
            for reason, count in reasons.items():
                failure_rows.append({"method": method, "reason": reason, "count": count})
    write_csv(generated / "baseline_failures.csv", failure_rows,
              ("method", "reason", "count"))
    coverage_rows, conditional_rows = [], []
    diagnosis_path = ROOT / "evidence" / "baselines" / "CMAX_REFERENCE_DIAGNOSIS.json"
    pilot_path = ROOT / "evidence" / "baselines" / "pilot_scored" / "METHOD_METRICS.csv"
    if diagnosis_path.is_file():
        diagnosis = read_json(diagnosis_path)
        old, corrected = diagnosis["old_variant"], diagnosis["corrected_variant"]
        coverage_rows = [{
            "variant": "initial_local_cmax", "equation": old["equation"],
            "requested": old["requested"], "successful": old["successful"],
            "failed_retained": old["requested"] - old["successful"],
            "success_fraction": old["successful"] / old["requested"],
            "interpretation": "Incorrect exponential/latest-event warp in the initial local "
                              "adapter; not a failure of the published CMax method"}, {
            "variant": "affine_corrected_cmax_reference",
            "equation": corrected["equation"], "requested": corrected["requested"],
            "successful": corrected["successful"],
            "failed_retained": corrected["failed_retained"],
            "success_fraction": corrected["success_fraction"],
            "interpretation": "Corrected affine reference-time local adaptation; diagnostic, "
                              "not a paper-result reproduction"}]
    if pilot_path.is_file():
        conditional_rows = [
            {column: row.get(column, "") for column in PILOT_CONDITIONAL_COLUMNS}
            for row in rows(pilot_path)
            if row.get("method") in {"cmax_reference", "strttc_adapted"}]
    write_csv(generated / "baseline_coverage.csv", coverage_rows,
              BASELINE_COVERAGE_COLUMNS)
    write_csv(generated / "baseline_conditional_metrics.csv", conditional_rows,
              PILOT_CONDITIONAL_COLUMNS)
    results = {"metrics": metric_rows, "system_cost": cost_rows,
               "baseline_failures": failure_rows, "baseline_coverage": coverage_rows,
               "baseline_conditional_metrics": conditional_rows}
    (generated / "RESULTS.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, expected in state["generated_tables"].items():
        if name == "REGENERATION.json":
            candidate = ROOT / "tables" / name
        else:
            candidate = generated / name
        if not candidate.is_file() or digest(candidate) != expected:
            raise SystemExit(f"regenerated table differs: {name}")
        published = ROOT / "tables" / name
        if not published.is_file() or digest(published) != expected:
            raise SystemExit(f"published table hash mismatch: {name}")
    temporary_tables.cleanup()
    if args.verify:
        print(f"verified {len(inventory['files'])} evidence files and "
              f"{len(regeneration['scoring_jobs'])} exact scoring runs")

if __name__ == "__main__":
    main()
