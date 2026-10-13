"""Regenerate V18 status, result tables and curves from durable campaign evidence."""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
from pathlib import Path

from operational.range_transfer.common import read, sha, utc, write

from .common import ROOT, RUN, config


def atomic_text(path: Path, value: str) -> None:
    """Publish a complete report file without exposing partial text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def table(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write a header even when no completed scientific result exists."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def collect() -> dict:
    """Read complete evidence only; never execute models or select recipes."""
    bound_files = {}

    def evidence(path: Path) -> dict:
        value = read(path)
        bound_files[path.relative_to(ROOT).as_posix()] = sha(path)
        return value

    counts = {"scientific": 0, "technical": 0, "recovery_upper": 0, "pending_upper": 0}
    for path in RUN.glob("**/UPDATE_JOURNAL.json"):
        if "report" in path.relative_to(RUN).parts:
            continue
        record = evidence(path)
        kind = "technical" if "qa" in path.relative_to(RUN).parts else "scientific"
        counts[kind] += record["completed_updates"]
        counts["recovery_upper"] += record["recovery_upper"]
        counts["pending_upper"] += record["pending_update_upper"]
    counts["physical_upper"] = sum(counts.values())
    budget_path = RUN / "GPU_BUDGET.json"
    budget = evidence(budget_path) if budget_path.exists() else {"entries": []}
    budget_cap = budget.get("cap_s", 64800)
    budget_authorizations = []
    for path in sorted(RUN.glob("BUDGET_AUTHORIZATION_*.json")):
        authorization = evidence(path)
        if (
            authorization.get("status") == "APPLIED"
            and authorization.get("authorized_cap_seconds") == budget_cap
        ):
            budget_authorizations.append(
                {"path": path.relative_to(RUN).as_posix(), **authorization}
            )
    settled = sum(r["charged_s"] for r in budget["entries"] if r["status"] == "SETTLED")
    reserved = sum(r["charged_s"] for r in budget["entries"] if r["status"] == "RESERVED")
    fits = []
    for fit_id in config()["fits"]:
        folder = RUN / "fits" / fit_id
        receipt = folder / "CHECKPOINT_RECEIPT.json"
        state = (
            evidence(receipt)
            if receipt.exists()
            else {
                "status": "NOT_STARTED",
                "completed_updates": 0,
                "checkpoint_sha256": None,
            }
        )
        fits.append({"fit_id": fit_id, **state})
    results = []
    for path in sorted((RUN / "predictions").glob("*/*/COMPLETE.json")):
        receipt = evidence(path)
        if receipt["status"] != "COMPLETE":
            continue
        metric = evidence(path.with_name("METRICS.json"))
        row = {
            "role": path.parent.parent.name,
            "fit_id": path.parent.name,
            "rows": metric["num_rows"],
            "complete_coverage": metric["complete_prediction_and_mid_coverage"],
            "internal_group_macro_MiD": metric["overall_MiD"],
            "pooled_mean_MiD": metric["mean_MiD"],
            "MAE_s": metric["diagnostics"]["mae_s"],
            "sign_error_rate": metric["diagnostics"]["sign_error_rate"],
            "invalid_predictions": metric["num_failed_predictions"],
            "prediction_sha256": receipt["sha256"],
        }
        row.update(
            {f"MiD_{name}": value["group_macro_mid"] for name, value in metric["bins"].items()}
        )
        results.append(row)
    decisions = {}
    for name in ("PRIMARY_SELECTION_FINAL.json", "NEXT_DECISION.json"):
        if (RUN / name).exists():
            decisions[name] = evidence(RUN / name)
    preparation = {}
    for role in ("P", "H", "V"):
        if (RUN / f"PREPARE_{role}_PROGRESS.json").exists():
            preparation[role] = evidence(RUN / f"PREPARE_{role}_PROGRESS.json")
    return {
        "generated_utc": utc(),
        "report_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "state": "RESEARCH_ONGOING",
        "counts": counts,
        "fits": fits,
        "gpu_settled_scope_seconds": settled,
        "gpu_reserved_upper_seconds": reserved,
        "gpu_charged_upper_seconds": settled + reserved,
        "gpu_cap_seconds": budget_cap,
        "gpu_budget_authorizations": budget_authorizations,
        "results": results,
        "decisions": decisions,
        "preparation": preparation,
        "source_files": bound_files,
        "SOTA_confirmed": False,
        "scope": "Internal exposed development. Official test has no authorized submission.",
    }


def figures(directory: Path) -> list[str]:
    """Draw separate objective curves; native and pair loss scales are not pooled."""
    curves = sorted((RUN / "fits").glob("*/CURVE.csv"))
    if not curves:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output = []
    for path in curves:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        fit_id = path.parent.name
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout="constrained")
        axes[0].plot(frame["update"], frame["loss"], alpha=0.3, linewidth=0.6)
        axes[0].plot(frame["update"], frame["loss"].rolling(100, min_periods=1).mean())
        axes[0].set_ylabel("Native objective" if fit_id.startswith("NATIVE") else "Phase objective")
        axes[0].set_title(fit_id + " — P training, fixed endpoint")
        axes[1].plot(frame["update"], frame["grad_norm"], linewidth=0.7)
        axes[1].set_ylabel("Gradient norm before clipping")
        axes[1].set_xlabel("Scientific optimizer updates")
        for suffix in ("png", "pdf"):
            relative = f"plots/{fit_id}.{suffix}"
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=200)
            output.append(relative)
        plt.close(fig)
    return output


def build() -> dict:
    """Publish reproducible tables without turning incomplete work into a conclusion."""
    result = collect()
    directory = RUN / "report"
    write(directory / "SNAPSHOT.json", result)
    table(
        directory / "FITS.csv",
        result["fits"],
        ["fit_id", "status", "completed_updates", "checkpoint_sha256"],
    )
    table(
        directory / "METRICS.csv",
        result["results"],
        [
            "role",
            "fit_id",
            "rows",
            "complete_coverage",
            "internal_group_macro_MiD",
            "pooled_mean_MiD",
            "MiD_crucial",
            "MiD_small",
            "MiD_large",
            "MiD_negative",
            "MAE_s",
            "sign_error_rate",
            "invalid_predictions",
            "prediction_sha256",
        ],
    )
    plots = figures(directory)
    counts = result["counts"]
    lines = [
        "# V18 JOINT-MEASUREMENT",
        "",
        f"Generated: {result['generated_utc']}. State: RESEARCH_ONGOING.",
        "",
        "This report does not establish SOTA or official benchmark confirmation.",
        "A complete endpoint is not by itself a promoted system.",
        "",
        f"Scientific updates: {counts['scientific']}; technical: {counts['technical']}; "
        f"recovery upper: {counts['recovery_upper']}; pending upper: {counts['pending_upper']}.",
        f"Total physical upper bound: {counts['physical_upper']} / 1,000,000.",
        f"Settled own GPU scopes: {result['gpu_settled_scope_seconds']:.3f} s; "
        f"reserved upper bound: {result['gpu_reserved_upper_seconds']:.3f} s; "
        f"ledger cap: {result['gpu_cap_seconds']:,.0f} s.",
        *[
            f"Recorded budget amendment: [{row['path']}](../{row['path']}). "
            f"Scope: {row['scope']}. Read later applied amendments for any scope expansion."
            for row in result["gpu_budget_authorizations"]
        ],
        *(
            ["The changed ledger cap has no matching applied authorization in this snapshot."]
            if result["gpu_cap_seconds"] != 64800 and not result["gpu_budget_authorizations"]
            else []
        ),
        "Scopes include loading/saving inside the timer; these are not GPU kernel-only seconds.",
        "",
        "## Primary fixed endpoints",
        "",
        "| Fit | Durable status | Completed updates |",
        "| --- | --- | ---: |",
        *[
            f"| {row['fit_id']} | {row['status']} | {row['completed_updates']} |"
            for row in result["fits"]
        ],
        "",
        "## Results",
        "",
        "[Machine-readable metrics](METRICS.csv). Empty means no completed scientific table.",
        "Group-macro weighted MiD and pooled mean MiD are distinct internal statistics.",
        "P is training; H selects recipes; V is exposed development. None is official test12.",
        "Negative results and invalid outputs remain in their full-cohort evidence.",
        "",
        "## Provenance and remaining work",
        "",
        "[Snapshot and source hashes](SNAPSHOT.json). Checkpoint hashes are taken from durable",
        "receipts; the report itself does not independently replay their weights or predictions.",
        "NEXT_DECISION and registered experiment records determine which adaptive/RGB/replication/",
        "refit branches remain enabled. No empty table closes the authorized program.",
        "",
        *[f"[{name}]({name})" for name in plots],
        "",
    ]
    atomic_text(directory / "README.md", "\n".join(lines))
    return {
        "report": str(directory / "README.md"),
        "completed_result_tables": len(result["results"]),
        "counts": counts,
    }


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(build())
