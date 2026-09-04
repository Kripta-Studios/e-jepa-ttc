#!/usr/bin/env python
"""Recompute X0 campaign tables from physical CSV/JSON and write the final report."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_protocol import production_sequence_macro_metrics
from e_jepa_ttc.evaluation.garl_ttc_protocol import BUCKETS

ARMS = ("X0-A5-REPLAY", "X0-BASE-U", "X0-DYN-U", "X0-PAIR-U")


def _load_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"signed artifact verification failed: {path}")
    return value


def _optional_signed(path: Path) -> dict[str, Any] | None:
    return _load_signed(path) if path.is_file() else None


def _master_qa(campaign: Path) -> list[dict[str, Any]]:
    path = campaign / "state/master_state.jsonl"
    if not path.is_file():
        return []
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line
    ]
    starts: dict[str, datetime] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        stage = str(record.get("stage", ""))
        status = str(record.get("status", ""))
        timestamp = datetime.fromisoformat(str(record["utc"]))
        if status == "running":
            starts[stage] = timestamp
        elif status in {"complete", "fatal"} and stage in starts:
            rows.append(
                {
                    "stage": stage,
                    "status": status,
                    "exit_code": record.get("exit_code"),
                    "duration_seconds": (timestamp - starts.pop(stage)).total_seconds(),
                    "log": record.get("log"),
                }
            )
    return rows


def _checked_rows(arm_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for fold in (0, 1, 2):
        root = arm_root / f"fold-{fold}"
        summary = _load_signed(root / "fold_summary.json")
        path = root / "oof_predictions.csv"
        if summary["oof_file_sha256"] != compute_file_hash(str(path)):
            raise ValueError(f"OOF SHA mismatch: {path}")
        frame = pd.read_csv(path, float_precision="round_trip")
        if set(frame["outer_fold"].astype(int)) != {fold}:
            raise ValueError(f"mixed fold rows: {path}")
        frames.append(frame)
        summaries.append(summary)
    rows = pd.concat(frames, ignore_index=True)
    if len(rows) != 8192 or rows["sample_token"].duplicated().any():
        raise ValueError(f"arm row universe is not canonical: {arm_root.name}")
    target = rows["target_ttc_s"].to_numpy(dtype=np.float64)
    buckets = np.full(len(rows), "", dtype=object)
    for name, lower, upper in BUCKETS:
        buckets[(target > lower) & (target <= upper)] = name
    if np.any(buckets == ""):
        raise ValueError("analysis encountered a target outside signed buckets")
    rows["ttc_bucket"] = buckets.astype(str)
    return rows, summaries


def _telemetry(campaign: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    gpu = campaign / "telemetry/gpu.csv"
    if gpu.is_file() and gpu.stat().st_size:
        columns = [
            "timestamp",
            "index",
            "name",
            "driver",
            "temperature",
            "gpu_util",
            "memory_util",
            "memory_used_mib",
            "memory_total_mib",
            "power_w",
            "sm_clock",
            "memory_clock",
            "pstate",
        ]
        frame = pd.read_csv(gpu, names=columns, skipinitialspace=True)
        for column in ("temperature", "gpu_util", "memory_util", "memory_used_mib", "power_w"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        gpu_util = frame["gpu_util"].to_numpy(dtype=np.float64)
        memory_used = frame["memory_used_mib"].to_numpy(dtype=np.float64)
        temperature = frame["temperature"].to_numpy(dtype=np.float64)
        power = frame["power_w"].to_numpy(dtype=np.float64)
        result["gpu"] = {
            "samples": len(frame),
            "util_mean": float(np.nanmean(gpu_util)),
            "util_p50": float(np.nanquantile(gpu_util, 0.5)),
            "util_p95": float(np.nanquantile(gpu_util, 0.95)),
            "vram_peak_mib": float(np.nanmax(memory_used)),
            "temperature_peak_c": float(np.nanmax(temperature)),
            "power_mean_w": float(np.nanmean(power)),
            "power_peak_w": float(np.nanmax(power)),
            "starvation_fraction_util_below_30": float(np.mean(gpu_util < 30)),
        }
    host = campaign / "telemetry/host.jsonl"
    if host.is_file() and host.stat().st_size:
        records = [
            json.loads(line) for line in host.read_text(encoding="utf-8").splitlines() if line
        ]
        rss = [
            float(x["campaign_process_rss_bytes"])
            for x in records
            if x.get("campaign_process_rss_bytes")
        ]
        free = [float(x["ram_free_bytes"]) for x in records if x.get("ram_free_bytes")]
        result["host"] = {
            "samples": len(records),
            "campaign_rss_peak_bytes": max(rss) if rss else None,
            "ram_free_min_bytes": min(free) if free else None,
        }
    summary = sign_artifact(
        {"artifact_type": "eclock_x0_telemetry_summary_v1", "telemetry": result}
    )
    path = campaign / "telemetry/summaries/summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    arm_rows: dict[str, pd.DataFrame] = {}
    arm_metrics: dict[str, dict[str, Any]] = {}
    runtime_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in ARMS:
        arm_root = campaign / arm
        aggregate_path = arm_root / "aggregate.json"
        if not aggregate_path.is_file():
            missing.append(arm)
            continue
        aggregate = _load_signed(aggregate_path)
        rows, summaries = _checked_rows(arm_root)
        recomputed = production_sequence_macro_metrics(rows)
        signed_mid = float(aggregate["metrics"]["sequence_macro_paper_MiD_overall"])
        observed_mid = float(recomputed["sequence_macro_paper_MiD_overall"])
        if not math.isclose(signed_mid, observed_mid, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{arm} aggregate differs from CSV recomputation")
        arm_rows[arm] = rows
        failures = rows["scientific_failure"].to_numpy(dtype=np.float64)
        clipped = rows["is_clip_saturated"].to_numpy(dtype=np.float64)
        arm_metrics[arm] = {
            "mid": observed_mid,
            "delta_vs_a5": float(aggregate["delta_mid_vs_official_a5_oof"]),
            "failure_rate": float(np.mean(failures)),
            "finite_fraction": float(
                np.isfinite(rows["predicted_benchmark_phase"].to_numpy(float)).mean()
            ),
            "clip_fraction": float(np.mean(clipped)),
            "gate": aggregate["gate_decision"].get("decision"),
            "per_sequence": recomputed["per_sequence"],
            "per_fold": {
                str(fold): production_sequence_macro_metrics(rows.loc[rows["outer_fold"] == fold])[
                    "sequence_macro_paper_MiD_overall"
                ]
                for fold in (0, 1, 2)
            },
            "phase_distribution": rows["predicted_benchmark_phase"].describe().to_dict(),
            "ttc_raw_distribution": rows["predicted_ttc_raw"].describe().to_dict(),
            "clipping_diagnostics": aggregate.get("clipping_diagnostics"),
            "bootstrap_vs_official_a5": aggregate.get("bootstrap"),
            "aggregate_sha256": aggregate["artifact_sha256"],
        }
        for fold, summary in enumerate(summaries):
            fold_rows = rows.loc[rows["outer_fold"] == fold]
            progress = arm_root / f"fold-{fold}/progress.jsonl"
            losses: list[float] = []
            if progress.is_file():
                progress_records = [
                    json.loads(line)
                    for line in progress.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                update_records = [
                    record for record in progress_records if record.get("event") == "update"
                ]
                losses = [float(record["loss"]) for record in update_records]
            else:
                progress_records = []
                update_records = []
            checkpoint = Path(str(summary["checkpoint_path"]))
            grad_norms = np.asarray(
                [record["grad_norm"] for record in update_records], dtype=np.float64
            )
            rates = np.asarray(
                [record["samples_per_second"] for record in update_records], dtype=np.float64
            )
            batch_load = np.asarray(
                [record["batch_load_ms"] for record in update_records], dtype=np.float64
            )
            final_update = update_records[-1] if update_records else {}
            runtime_rows.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "train_updates": summary.get("updates_completed", 0),
                    "train_rows": 8192 - len(fold_rows),
                    "dev_rows": len(fold_rows),
                    "wall_seconds": summary.get("fold_wall_seconds"),
                    "start_utc": progress_records[0].get("utc") if progress_records else None,
                    "end_utc": final_update.get("utc"),
                    "samples_per_second_mean": float(np.mean(rates)) if rates.size else None,
                    "initial_loss": losses[0] if losses else None,
                    "final_loss": losses[-1] if losses else None,
                    "minimum_train_loss_diagnostic": min(losses) if losses else None,
                    "rolling_loss_100_final": final_update.get("loss_rolling_100"),
                    "grad_norm_p50": (
                        float(np.quantile(grad_norms, 0.5)) if grad_norms.size else None
                    ),
                    "grad_norm_p95": (
                        float(np.quantile(grad_norms, 0.95)) if grad_norms.size else None
                    ),
                    "grad_norm_max": float(np.max(grad_norms)) if grad_norms.size else None,
                    "phase_mean_final": final_update.get("phase_mean"),
                    "phase_std_final": final_update.get("phase_std"),
                    "batch_load_ms_p50": (
                        float(np.quantile(batch_load, 0.5)) if batch_load.size else None
                    ),
                    "batch_load_ms_p95": (
                        float(np.quantile(batch_load, 0.95)) if batch_load.size else None
                    ),
                    "cpu_rss_peak_bytes": (
                        max(record["cpu_rss_bytes"] for record in update_records)
                        if update_records
                        else None
                    ),
                    "gpu_peak_allocated_bytes": (
                        max(record["gpu_peak_allocated_bytes"] for record in update_records)
                        if update_records
                        else None
                    ),
                    "resume_count": sum(
                        record.get("event") == "training_resume" for record in progress_records
                    ),
                    "checkpoint": summary["checkpoint_path"],
                    "checkpoint_sha256": summary["checkpoint_file_sha256"],
                    "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
                    "cache_mode": summary.get("cache_mode"),
                }
            )
    telemetry = _telemetry(campaign)
    environment = _optional_signed(campaign / "environment.json")
    preflight = _optional_signed(campaign / "preflight/preflight_summary.json")
    cache_decision = _optional_signed(campaign / "preflight/cache_engineering_decision.json")
    smoke = _optional_signed(campaign / "smoke/smoke_summary.json")
    provenance = _optional_signed(campaign / "provenance_exception.json")
    qa_rows = _master_qa(campaign)
    comparison_path = campaign / "comparisons/x0_dyn_vs_base.json"
    comparison = _load_signed(comparison_path) if comparison_path.is_file() else None
    if comparison is not None:
        base = arm_rows["X0-BASE-U"].sort_values("sample_token")
        dyn = arm_rows["X0-DYN-U"].sort_values("sample_token")
        if base["sample_token"].tolist() != dyn["sample_token"].tolist():
            raise ValueError("analysis BASE/DYN token order mismatch")
        observed_delta = arm_metrics["X0-DYN-U"]["mid"] - arm_metrics["X0-BASE-U"]["mid"]
        if not math.isclose(
            observed_delta, float(comparison["delta_mid_dyn_minus_base"]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("primary comparison differs from CSV recomputation")
    fatal = campaign / "failure/fatal.json"
    if fatal.is_file():
        decision = "X0_CAMPAIGN_FATAL_INCOMPLETE"
    elif comparison is None:
        decision = "X0_SCREEN_COMPLETE_BUT_PRIMARY_GATE_INCOMPLETE"
    elif comparison["gate_decision"].get("passed") is True and not missing:
        decision = "X0_PRIMARY_SUPPORTED_SCREEN_COMPLETE"
    elif not missing:
        decision = "X0_PRIMARY_NOT_SUPPORTED_SCREEN_COMPLETE"
    else:
        decision = "X0_SCREEN_COMPLETE_BUT_PRIMARY_GATE_INCOMPLETE"
    repo = Path(__file__).resolve().parents[1]
    hardening_commits = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "log",
            "--format=%H %s",
            "af66f2c8ca2017059d7765b5f171e1cda866ab07..HEAD",
        ],
        text=True,
    ).strip()
    lines = [
        "# E-Clock X0 final seed-7 campaign report",
        "",
        f"Campaign status: `{'complete' if not missing else 'incomplete'}`",
        "",
        "## Executive summary",
        "",
        "All values below were recomputed from physical CSV/JSON artifacts; "
        "stdout was not used as a metric source.",
        "",
        "## Git and environment",
        "",
        f"Environment: `{json.dumps(environment, sort_keys=True)}`",
        "",
        "Hardening commits since the required starting HEAD:",
        "",
        "```text",
        hardening_commits,
        "```",
        "",
        "## Cross-commit provenance",
        "",
        (
            f"Signed provenance exception: `{json.dumps(provenance, sort_keys=True)}`"
            if provenance is not None
            else "No cross-commit artifact reuse was declared."
        ),
        "",
        "## Preflight QA, cache and smoke",
        "",
        f"Preflight: `{json.dumps(preflight, sort_keys=True)}`",
        "",
        f"Cache engineering: `{json.dumps(cache_decision, sort_keys=True)}`",
        "",
        f"Real outer-train smoke: `{json.dumps(smoke, sort_keys=True)}`",
        "",
        "| Stage | Status | Exit | Seconds | Log |",
        "|---|---|---:|---:|---|",
    ]
    for row in qa_rows:
        lines.append(
            f"| {row['stage']} | {row['status']} | {row['exit_code']} | "
            f"{row['duration_seconds']:.3f} | {row['log']} |"
        )
    lines += [
        "",
        "## Arm metrics",
        "",
        "| Arm | Rows | MiD | Δ vs official A5 | failure | finite | clip | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        if arm not in arm_metrics:
            lines.append(f"| {arm} | missing | — | — | — | — | — | incomplete |")
            continue
        m = arm_metrics[arm]
        lines.append(
            f"| {arm} | 8192 | {m['mid']:.9f} | {m['delta_vs_a5']:.9f} | "
            f"{m['failure_rate']:.6f} | {m['finite_fraction']:.6f} | "
            f"{m['clip_fraction']:.6f} | {m['gate']} |"
        )
        lines += [
            "",
            f"{arm} folds: `{json.dumps(m['per_fold'], sort_keys=True)}`",
            "",
            f"{arm} sequences/buckets: `{json.dumps(m['per_sequence'], sort_keys=True)}`",
            "",
            f"{arm} phase distribution: `{json.dumps(m['phase_distribution'], sort_keys=True)}`",
            "",
            f"{arm} raw TTC distribution: "
            f"`{json.dumps(m['ttc_raw_distribution'], sort_keys=True)}`",
            "",
            f"{arm} clipping diagnostics: "
            f"`{json.dumps(m['clipping_diagnostics'], sort_keys=True)}`",
            "",
            f"{arm} bootstrap versus official A5: "
            f"`{json.dumps(m['bootstrap_vs_official_a5'], sort_keys=True)}`",
            "",
            f"{arm} signed aggregate SHA256: `{m['aggregate_sha256']}`",
        ]
    lines += ["", "## Primary DYN-U versus BASE-U", ""]
    if comparison is None:
        lines.append("Primary comparison artifact is missing.")
    else:
        boot = comparison["bootstrap"]["delta_candidate_minus_reference"]
        comparison_header = (
            "| BASE MiD | DYN MiD | Δ DYN−BASE | bootstrap mean | CI95 low | "
            "CI95 high | P(Δ<0) | finite draws | decision |"
        )
        comparison_row = (
            f"| {comparison['base_mid']:.9f} | {comparison['dyn_mid']:.9f} | "
            f"{comparison['delta_mid_dyn_minus_base']:.9f} | {boot['mean']:.9f} | "
            f"{boot['ci95_low']:.9f} | {boot['ci95_high']:.9f} | "
            f"{boot['probability_delta_lt_zero']:.6f} | "
            f"{boot['finite_draw_fraction']:.6f} | "
            f"{comparison['gate_decision']['decision']} |"
        )
        lines += [
            comparison_header,
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            comparison_row,
            "",
            f"Fold deltas: `{json.dumps(comparison['fold_deltas'], sort_keys=True)}`",
            "",
            f"Sequence deltas: `{json.dumps(comparison['sequence_deltas'], sort_keys=True)}`",
            "",
            f"Bucket deltas: `{json.dumps(comparison['bucket_deltas'], sort_keys=True)}`",
            "",
            f"Track deltas: `{json.dumps(comparison['track_deltas'], sort_keys=True)}`",
            "",
            f"Row win rate={comparison['row_win_rate']:.6f}; "
            f"track win rate={comparison['track_win_rate']:.6f}; "
            f"error correlation={comparison['error_correlation']:.6f}; "
            "oracle MiD diagnostic="
            f"{comparison['oracle_base_dyn_mid_diagnostic']:.9f}.",
        ]
    lines += [
        "",
        "## Runtime and hardware",
        "",
        "| Arm | Fold | Train/dev | Updates | Wall s | samples/s | initial/final/min | "
        "roll100 | grad p50/p95/max | phase final mean/std | checkpoint bytes | resumes | "
        "batch load p50/p95 ms | CPU/GPU peak bytes | Cache |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---|---|---|",
    ]
    for row in runtime_rows:
        lines.append(
            f"| {row['arm']} | {row['fold']} | {row['train_rows']}/{row['dev_rows']} | "
            f"{row['train_updates']}/6840 | {row['wall_seconds']} | "
            f"{row['samples_per_second_mean']} | {row['initial_loss']}/{row['final_loss']}/"
            f"{row['minimum_train_loss_diagnostic']} | {row['rolling_loss_100_final']} | "
            f"{row['grad_norm_p50']}/{row['grad_norm_p95']}/{row['grad_norm_max']} | "
            f"{row['phase_mean_final']}/{row['phase_std_final']} | "
            f"{row['checkpoint_bytes']} | {row['resume_count']} | "
            f"{row['batch_load_ms_p50']}/{row['batch_load_ms_p95']} | "
            f"{row['cpu_rss_peak_bytes']}/{row['gpu_peak_allocated_bytes']} | "
            f"{row['cache_mode']} |"
        )
        lines.append(
            f"Checkpoint `{row['checkpoint']}` SHA256 `{row['checkpoint_sha256']}`; "
            f"start `{row['start_utc']}`, end `{row['end_utc']}`."
        )
    lines += [
        "",
        f"Telemetry summary: `{json.dumps(telemetry, sort_keys=True)}`",
        "",
        "## Integrity, limitations and negative evidence",
        "",
        "PAIR-U is a geometry-infused readout diagnostic and cannot establish height "
        "bypass. BASE/DYN consume a box-conditioned upstream ROI, so this campaign is "
        "not geometry-free, bbox-free or detector-free. Seed 7 is a preregistered screen, "
        "not multiseed or external confirmation. Training minima are diagnostic only; "
        "every scientific checkpoint is fixed update 6840.",
        "",
        "Bootstrap semantics were frozen before results: a draw is discarded in full when "
        "any resampled sequence replica loses a TTC bucket; the finite-draw fraction is "
        "disclosed.",
        "",
        "Signed facts are the artifact identities, row counts, MiD values, bootstrap, gates and "
        "checkpoint hashes above. Loss curves, clipping, oracle selection, row/track wins and "
        "resource telemetry are diagnostics. Any statement about concentration is an inference, "
        "not a new gate. Failures and incomplete arms remain visible rather than imputed.",
        "",
        "## Sealed evaluation",
        "",
        "```text",
        "public_validation_opened=false",
        "private_test_opened=false",
        "evttc_test_opened=false",
        "codabench_opened=false",
        "```",
        "",
        decision,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    analysis = sign_artifact(
        {
            "artifact_type": "eclock_x0_final_analysis_v1",
            "decision": decision,
            "missing_arms": missing,
            "arm_metrics": arm_metrics,
            "runtime": runtime_rows,
            "provenance_exception_sha256": (
                provenance.get("artifact_sha256") if provenance is not None else None
            ),
            "report_sha256": compute_file_hash(str(args.output)),
        }
    )
    (campaign / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "report": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
