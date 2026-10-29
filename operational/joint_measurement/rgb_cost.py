"""Paired RGB prepared-forward cost and prospective development selection."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha, utc

from .common import ROOT, RUN, gpu_charge, require_resources
from .export import TTCOnly
from .rgb_data import RGBPairDataset
from .rgb_evaluate import BASELINE, freeze_endpoints, load_model, select_h
from .rgb_train import REGISTRATION, registered
from .train import setup

DIRECTORY = RUN / "rgb/cost/prepared_forward"


def sources() -> dict:
    """Bind timing, model loading and the common signed-output wrapper."""
    return {
        name: sha(ROOT / "operational/joint_measurement" / name)
        for name in ("rgb_cost.py", "rgb_evaluate.py", "export.py", "rgb_data.py")
    }


def combine_decision(metric: dict, ratio: float, routes: dict) -> dict:
    """Require the same coverage/sign/band constraints for a cost-only signal."""
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Paired cost ratio must be finite and positive")
    checks = metric.get("checks", {})
    cost_pass = (
        set(checks) == {"coverage", "sign", "bands"}
        and all(checks.values())
        and metric.get("overall_ratio", math.inf) <= routes["cost_route_overall_ratio_max"]
        and ratio <= routes["cost_ratio_max"]
    )
    return {
        **metric,
        "paired_cost_ratio": ratio,
        "cost_route_pass": cost_pass,
        "eligible": bool(metric["eligible"] or cost_pass),
        "cost_scope": "prepared RGB tensor resident on GPU through signed TTC output",
    }


def validate_row(row: dict, binding: dict, ordinal: int) -> None:
    """Reject partial or mismatched cached timings before reusing them."""
    if (
        row["binding"] != digest(binding)
        or row["P_index"] != binding["indices"][ordinal]
        or set(row["timings"]) != set(binding["endpoints"])
    ):
        raise ValueError("RGB timing fragment binding, query or route changed")
    for value in row["timings"].values():
        if (
            not math.isfinite(value["seconds"])
            or value["seconds"] <= 0
            or not math.isfinite(value["ttc_seconds"])
            or value["peak_extra_allocated_bytes"] < 0
        ):
            raise ValueError("Invalid RGB timing fragment")


def measure() -> dict:
    """Measure three frozen RGB routes on the same 128 label-blind P inputs."""
    endpoints = freeze_endpoints()["endpoints"]
    record = registered()
    ids = [BASELINE, *record["fits"]]
    dataset = RGBPairDataset("P")
    try:
        generator = np.random.default_rng(20261012)
        indices = generator.choice(dataset.population, 128, replace=False).tolist()
        binding = {
            "endpoints": endpoints,
            "registration_sha256": sha(REGISTRATION),
            "sources": sources(),
            "P_cache_manifest_sha256": dataset.source.prepared_cache.manifest_sha256,
            "population": "P",
            "indices": indices,
            "paired_orders": [generator.permutation(ids).tolist() for _ in indices],
            "warmup_per_model": 20,
            "batch_size": 1,
            "dtype": "FP32",
            "TF32": False,
            "scope": "prepared RGB tensor resident on GPU through signed TTC output",
            "excludes": ["sensor preprocessing", "host-to-device transfer", "weight loading"],
            "exploration_cost_statistic": "paired population warm median",
        }
        freeze(DIRECTORY / "REGISTRATION.json", binding)
        if (DIRECTORY / "COMPLETE.json").exists():
            result = read(DIRECTORY / "COMPLETE.json")
            if result["binding"] != digest(binding):
                raise ValueError("Completed RGB timing identity changed")
            return result
        require_resources(starting=True)
        setup(7)
        with gpu_charge("rgb_paired_forward_cost", reserve=900) as deadline:
            models = {
                fit_id: TTCOnly(load_model(fit_id)).cuda().eval().requires_grad_(False)
                for fit_id in ids
            }
            sample = dataset.batch([indices[0]])
            example = (sample.events.cuda(), sample.delta_t_s.cuda().reshape(1, 1))
            with torch.inference_mode():
                cold = {}
                for fit_id, model in models.items():
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    model(*example)
                    torch.cuda.synchronize()
                    cold[fit_id] = time.perf_counter() - start
                    for _ in range(binding["warmup_per_model"]):
                        model(*example)
                freeze(
                    DIRECTORY / f"COLD_{time.time_ns()}.json",
                    {"meaning": "first inference after process load", "seconds": cold},
                )
                for ordinal, index in enumerate(indices):
                    path = DIRECTORY / f"row_{ordinal:03d}.json"
                    if path.exists():
                        validate_row(read(path), binding, ordinal)
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "next_row": ordinal}
                    require_resources()
                    sample = dataset.batch([index])
                    example = (sample.events.cuda(), sample.delta_t_s.cuda().reshape(1, 1))
                    timings = {}
                    for fit_id in binding["paired_orders"][ordinal]:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        baseline = torch.cuda.memory_allocated()
                        start = time.perf_counter()
                        output = models[fit_id](*example)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        timings[fit_id] = {
                            "seconds": elapsed,
                            "ttc_seconds": float(output[0]),
                            "peak_extra_allocated_bytes": (
                                torch.cuda.max_memory_allocated() - baseline
                            ),
                        }
                    row = {
                        "binding": digest(binding),
                        "P_index": index,
                        "timings": timings,
                        "utc": utc(),
                    }
                    validate_row(row, binding, ordinal)
                    freeze(path, row)
        frames = [read(DIRECTORY / f"row_{i:03d}.json") for i in range(len(indices))]
        summary = {}
        for fit_id in ids:
            seconds = np.array([row["timings"][fit_id]["seconds"] for row in frames])
            summary[fit_id] = {
                "p50_seconds": float(np.quantile(seconds, 0.5)),
                "p95_seconds": float(np.quantile(seconds, 0.95)),
                "p99_seconds": float(np.quantile(seconds, 0.99)),
                "batch1_windows_per_second": float(1 / seconds.mean()),
                "n": len(seconds),
                "max_extra_allocated_bytes": max(
                    row["timings"][fit_id]["peak_extra_allocated_bytes"] for row in frames
                ),
            }
        result = {
            "status": "COMPLETE",
            "binding": digest(binding),
            "results": summary,
            "E2E": False,
            "official_comparison": False,
        }
        freeze(DIRECTORY / "COMPLETE.json", result)
        return result
    finally:
        dataset.close()


def finalize_selection() -> dict:
    """Combine H metrics and admitted paired cost without consulting V outcomes."""
    endpoints = freeze_endpoints()["endpoints"]
    record = registered()
    metric_selection = select_h()
    binding = read(DIRECTORY / "REGISTRATION.json")
    cost = read(DIRECTORY / "COMPLETE.json")
    if (
        binding["endpoints"] != endpoints
        or binding["registration_sha256"] != sha(REGISTRATION)
        or binding["sources"] != sources()
        or cost["status"] != "COMPLETE"
        or cost["binding"] != digest(binding)
        or set(cost["results"]) != set(endpoints)
    ):
        raise ValueError("RGB cost evidence is incomplete or belongs to different endpoints")
    decisions, outcomes = {}, {}
    for fit_id in record["fits"]:
        ratio = cost["results"][fit_id]["p50_seconds"] / cost["results"][BASELINE]["p50_seconds"]
        decisions[fit_id] = combine_decision(
            metric_selection["decisions"][fit_id], ratio, record["development_gates"]["routes"]
        )
        outcomes[fit_id] = read(RUN / "rgb/predictions/H" / fit_id / "METRICS.json")
    ranked = sorted(
        [name for name, decision in decisions.items() if decision["eligible"]],
        key=lambda name: (
            outcomes[name]["overall_MiD"],
            outcomes[name]["bins"]["crucial"]["group_macro_mid"],
            name,
        ),
    )[: record["development_gates"]["max_integrations"]]
    result = {
        "status": "DEVELOPMENT_CANDIDATE" if ranked else "NO_CANDIDATE_RGB_PAIR",
        "ranked": ranked,
        "decisions": decisions,
        "reference": BASELINE,
        "metric_evidence_sha256": sha(RUN / "rgb/H_METRIC_SELECTION.json"),
        "cost_evidence_sha256": sha(DIRECTORY / "COMPLETE.json"),
        "system_promotion": False,
        "official_comparison": False,
        "next": "interpret P/H, register RGB-anchor fusion and ZERO control, compare full systems",
    }
    freeze(RUN / "rgb/SELECTION_FINAL.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["measure", "select"])
    args = parser.parse_args()
    print(measure() if args.action == "measure" else finalize_selection())
