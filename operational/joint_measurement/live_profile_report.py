"""Summarize passive Python stack samples without treating them as CUDA kernel timings."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from operational.range_transfer.common import read, sha, write

from .common import ROOT, RUN


def summarize(directory: Path) -> dict:
    """Attribute main-thread time once; keep concurrent reader/decoder time separate."""
    trace, resource = read(directory / "STACKS.json"), read(directory / "RESOURCES.json")
    frames = trace["shared"]["frames"]
    main = [p for p in trace["profiles"] if '"MainThread"' in p["name"]]
    loader = [p for p in trace["profiles"] if '"v18-fast-input_0"' in p["name"]]
    workers = [p for p in trace["profiles"] if '"p2-random_' in p["name"]]
    if len(main) != 1 or len(loader) != 1:
        raise ValueError("Expected one main thread and one prefetch coordinator")
    stage = Counter()
    forward = Counter()
    leaves = Counter()
    worker_leaves = Counter()
    replay = Counter()
    for profile in trace["profiles"]:
        for stack, weight in zip(profile["samples"], profile["weights"], strict=True):
            entries = [frames[i] for i in stack]
            if any(
                e["name"] == "replay" and e.get("file", "").endswith("graphs.py") for e in entries
            ):
                replay[profile["name"]] += weight
            if profile is loader[0] and entries:
                leaf = entries[-1]
                leaves[
                    f"{leaf['name']} ({Path(leaf.get('file', '')).name}:{leaf.get('line', 0)})"
                ] += weight
            if any(profile is worker for worker in workers) and entries:
                leaf = entries[-1]
                worker_leaves[
                    f"{leaf['name']} ({Path(leaf.get('file', '')).name}:{leaf.get('line', 0)})"
                ] += weight
            if profile is not main[0]:
                continue
            names = {e["name"] for e in entries}
            if any(
                e["name"] == "batch"
                and e.get("file", "").endswith(("fast_runtime.py", "parallel_training_sensor.py"))
                for e in entries
            ):
                category = "input_batch_wait_or_submit"
            elif "step_backward" in names or any(
                e["name"] == "backward" and e.get("file", "").endswith("phase_transport_loss.py")
                for e in entries
            ):
                category = "forward_loss_backward_including_host_sync"
                if "_engine_run_backward" in names:
                    substage = "autograd_engine_including_captured_backward"
                elif "checked_phase" in names:
                    substage = "captured_forward_domain_guard"
                elif "phase_terms" in names:
                    substage = "loss_tensor_arithmetic"
                elif any(e.get("file", "").endswith("graphs.py") for e in entries):
                    substage = "captured_forward_replay"
                else:
                    substage = "loss_scaling_guard_and_diagnostics_or_other_forward"
                forward[substage] += weight
            elif "step" in names:
                category = "optimizer_or_scheduler"
            elif "begin" in names or "commit" in names:
                category = "update_journal"
            elif any(
                e["name"] == "to" and e.get("file", "").endswith("object_event_v4.py")
                for e in entries
            ):
                category = "batch_transfer"
            elif "clip_grad_norm_" in names:
                category = "gradient_clip"
            else:
                category = "other"
            stage[category] += weight
    samples = resource["samples"]
    duration = samples[-1]["t"] - samples[0]["t"]
    updates = (
        samples[-1]["journal"]["completed_updates"] - samples[0]["journal"]["completed_updates"]
    )
    gpu = [[float(v.strip()) for v in s["gpu"].split(",")] for s in samples]
    total = sum(main[0]["weights"])
    if abs(sum(stage.values()) - total) > 1e-8 or not replay:
        raise ValueError("Incomplete exclusive main-thread attribution or missing graph replay")
    result = {
        "scope": (
            "Passive nonblocking py-spy 50Hz, idle threads included; "
            "no optimizer or process mutation"
        ),
        "limitations": (
            "Sampling estimates, not GPU kernel timing; overlapping threads must not be added. "
            "Nonblocking stack failures retained in SAMPLING.json. Process read bytes include "
            "OS-cached reads, not physical SSD bandwidth."
        ),
        "pid": resource["pid"],
        "create_time": resource["create_time"],
        "inputs": {
            n: sha(directory / n) for n in ("STACKS.json", "RESOURCES.json", "SAMPLING.json")
        },
        "source_binding": {
            n: sha(ROOT / "operational/joint_measurement" / n)
            for n in (
                "adaptation_depth.py",
                "correspondence_train.py",
                "correspondence_model.py",
                "geometry_capture.py",
                "fast_runtime.py",
                "data.py",
                "train.py",
                "phase_transport_loss.py",
                "persistent_adaptation.py",
                "parallel_training_sensor.py",
                "live_profile_report.py",
            )
        },
        "main_sampled_seconds": total,
        "main_stage_seconds": dict(stage),
        "main_stage_percent": {k: 100 * v / total for k, v in stage.items()},
        "loader_leaf_seconds": dict(leaves.most_common()),
        "forward_loss_backward_breakdown_seconds": dict(forward),
        "parallel_decoder_threads": len(workers),
        "parallel_decoder_leaf_seconds": dict(worker_leaves.most_common()),
        "parallel_decoder_scope": (
            "Concurrent thread-seconds; never add to main-thread wall-time percentages. "
            "Includes idle threads because the capture used --idle."
        ),
        "graph_replay_sampled_seconds_by_thread": dict(replay),
        "resource_window_seconds": duration,
        "completed_updates_in_resource_window": updates,
        "seconds_per_update": duration / updates,
        "updates_per_hour": 3600 * updates / duration,
        "gpu_utilization_percent_mean": statistics.mean(v[0] for v in gpu),
        "gpu_utilization_percent_range": [min(v[0] for v in gpu), max(v[0] for v in gpu)],
        "gpu_memory_mib_range": [min(v[2] for v in gpu), max(v[2] for v in gpu)],
        "gpu_temperature_c_range": [min(v[3] for v in gpu), max(v[3] for v in gpu)],
        "process_cpu_percent_mean_one_core_100": statistics.mean(
            s["cpu_percent"] for s in samples[1:]
        ),
        "host_available_gib_min": min(s["available"] for s in samples) / 1024**3,
        "process_rss_gib_range": [
            min(s["rss"] for s in samples) / 1024**3,
            max(s["rss"] for s in samples) / 1024**3,
        ],
        "process_logical_read_mb_per_s": (
            samples[-1]["io"]["read_bytes"] - samples[0]["io"]["read_bytes"]
        )
        / duration
        / 1e6,
        "optimizer_updates_from_diagnostic": 0,
    }
    write(directory / "SUMMARY.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RUN / "runtime/A1_live_profile_20261016")
    args = parser.parse_args()
    result = summarize(args.directory)
    print(
        {
            key: result[key]
            for key in ("main_stage_percent", "seconds_per_update", "gpu_utilization_percent_mean")
        }
    )
