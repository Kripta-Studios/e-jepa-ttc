"""Regenerate engineering comparison tables from already-recorded local windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize(root: Path) -> dict[str, Any]:
    single_path = root / "SINGLE_6MIN_RESULT.json"
    single = load(single_path)
    rows = [
        {
            "name": "single",
            "workers": 1,
            "composition": "1 RGB A5 producer",
            "wall_seconds": single["window_seconds"],
            "total_updates": single["updates"],
            "aggregate_updates_per_minute": single["updates_per_minute"],
            "started_utc": single["first_utc"],
            "ended_utc": single["last_utc"],
            "gpu_utilization_mean_percent": single["gpu_util_mean_pct"],
            "gpu_memory_peak_mib": single["gpu_memory_max_MiB"],
            "gpu_power_mean_watts": single["gpu_power_mean_W"],
            "gpu_temperature_peak_c": single["gpu_temp_max_C"],
            "cgroup_memory_peak_bytes": single["memory_cgroup_max_bytes"],
        }
    ]
    sources = [single_path]
    for directory, workers, composition in (
        ("DUAL_6MIN", 2, "2 RGB A5 producers"),
        ("TRIPLE_6MIN", 3, "3 RGB A5 producers"),
        ("SIX_6MIN", 6, "3 RGB A5 + 3 RGB C2F producers; unequal work per update"),
    ):
        result_path, sample_path = (
            root / directory / "RESULT.json",
            root / directory / "samples.jsonl",
        )
        result = load(result_path)
        samples = [json.loads(line) for line in sample_path.read_text().splitlines() if line]
        seconds = result["actual_seconds"]
        updates = sum(fit["updates"] for fit in result["fits"])
        gpu = [
            dict(
                zip(sample["gpu_fields"].split(","), sample["gpu_values"].split(", "), strict=True)
            )
            for sample in samples
        ]
        rows.append(
            {
                "name": directory,
                "workers": workers,
                "composition": composition,
                "wall_seconds": seconds,
                "total_updates": updates,
                "aggregate_updates_per_minute": updates * 60 / seconds,
                "started_utc": samples[0]["utc"],
                "ended_utc": samples[-1]["utc"],
                "gpu_utilization_mean_percent": statistics.mean(
                    float(x["utilization.gpu"]) for x in gpu
                ),
                "gpu_memory_peak_mib": max(float(x["memory.used"]) for x in gpu),
                "gpu_power_mean_watts": statistics.mean(float(x["power.draw"]) for x in gpu),
                "gpu_temperature_peak_c": max(float(x["temperature.gpu"]) for x in gpu),
                "cgroup_memory_peak_bytes": max(
                    sample["cgroup_memory_bytes"] for sample in samples
                ),
            }
        )
        sources.extend((result_path, sample_path))
    for row in rows:
        row["average_worker_updates_per_minute"] = (
            row["aggregate_updates_per_minute"] / row["workers"]
        )
        row["observed_aggregate_ratio_to_single"] = (
            row["aggregate_updates_per_minute"] / rows[0]["aggregate_updates_per_minute"]
        )
    return {
        "measurement_repeated": False,
        "rows": rows,
        "sources": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sources
        ],
        "scope": "Recorded Pod windows; sequential, not randomized timing experiments.",
        "limitations": [
            "Six-worker updates mix A5 and C2F workloads; ratio is not equal-work speedup.",
            "Utilization is sampled device occupancy, not kernel time or latency p95.",
            "Concurrent aggregate denominator is one shared wall-clock interval.",
            "These measurements do not establish prediction accuracy or SOTA.",
            "Exported-head parity at batch1/7 and phase edges is numerical compatibility, "
            "not full-sensor latency.",
            "No training/data-loading/inference decomposition without profiler evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing historical throughput windows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, help="Local preserved *.tar directory")
    parser.add_argument("--live-root", type=Path, help="Live /workspace directory; metadata only")
    parser.add_argument("--mode", choices=("partial", "final"), default="partial")
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("artifacts/final_diagnostics_20261027")
    )
    parser.add_argument(
        "--candidate-root", type=Path, help="Verified portable nine-checkpoint candidate"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    code_output = args.output / "report_code"
    code_output.mkdir(exist_ok=True)
    for script in sorted(Path(__file__).parent.glob("runpod_final_diagnostics*.py")):
        target = code_output / script.name
        if target.resolve() != script.resolve():
            target.write_bytes(script.read_bytes())
    if args.archive_root or args.live_root:
        try:
            from runpod_final_diagnostics_tables import build_report
        except ModuleNotFoundError:
            from scripts.runpod_final_diagnostics_tables import build_report

        result = build_report(
            archive_root=args.archive_root,
            live_root=args.live_root,
            output=args.output,
            mode=args.mode,
            evidence_root=args.evidence_root,
        )
        print(json.dumps(result))
    candidate = args.candidate_root
    if candidate is None and args.archive_root is not None:
        proposed = args.archive_root / "FINAL_CANDIDATE"
        if (proposed / "PORTABLE_PATHS.json").is_file():
            candidate = proposed
    if candidate is not None:
        try:
            from runpod_final_diagnostics_losses import extract
        except ModuleNotFoundError:
            from scripts.runpod_final_diagnostics_losses import extract
        receipt = extract(candidate, args.output)
        print(json.dumps({"stored_training_loss_rows": receipt["loss_rows"]}))
    if args.source is None:
        if not (args.archive_root or args.live_root):
            parser.error("Supply --source, --archive-root or --live-root")
        return
    report = summarize(args.source)
    for source in report["sources"]:
        path = Path(source["path"])
        target = args.output / "evidence/HISTORICAL_WINDOWS" / path.relative_to(args.source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != path.resolve():
            target.write_bytes(path.read_bytes())
    (args.output / "HISTORICAL_THROUGHPUT.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Historical measured throughput",
        "",
        report["scope"],
        "",
        "Recomputed from existing records; no training or measurement rerun.",
        "",
        "| Workers | Composition | Updates/min, whole Pod | Mean GPU % | Peak VRAM MiB |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['workers']} | {row['composition']} | "
            f"{row['aggregate_updates_per_minute']:.2f} | "
            f"{row['gpu_utilization_mean_percent']:.2f} | {row['gpu_memory_peak_mib']:.0f} |"
        )
    lines.extend(["", *[f"- {value}" for value in report["limitations"]], ""])
    (args.output / "HISTORICAL_THROUGHPUT.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "rows": len(report["rows"])}))


if __name__ == "__main__":
    main()
