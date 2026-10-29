"""Paired prepared-input timing for the geometry endpoint and its unchanged controls."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha, utc

from .common import ROOT, RUN, gpu_charge, require_resources
from .data import SensorDataset, cache_root
from .export import TTCOnly
from .geometry_evaluate import BASELINE, freeze_endpoint, load_model, select_metrics
from .geometry_train import CONTROL, DIRECTORY, FIT, REGISTRATION, registered
from .rgb_cost import combine_decision, validate_row
from .train import initialize, setup

COST = DIRECTORY / "cost/prepared_forward"


def sources() -> dict:
    """Bind timing, model admission, conversions and shared cost validation."""
    return {
        name: sha(ROOT / "operational/joint_measurement" / name)
        for name in (
            "geometry_cost.py",
            "geometry_evaluate.py",
            "geometry_model.py",
            "export.py",
            "rgb_cost.py",
        )
    }


def measure() -> dict:
    """Interleave all three routes on identical label-blind P queries."""
    endpoints = freeze_endpoint()["endpoints"]
    require_resources(starting=True)
    setup(7)
    dataset = SensorDataset("P")
    names = [BASELINE, CONTROL, FIT]
    try:
        generator = np.random.default_rng(20261012)
        indices = generator.choice(dataset.population, 128, replace=False).tolist()
        binding = {
            "endpoints": endpoints,
            "registration_sha256": sha(REGISTRATION),
            "sources": sources(),
            "P_sensor_binding": sha(cache_root("P") / "COMPLETE.json"),
            "population": "P",
            "indices": indices,
            "paired_orders": [generator.permutation(names).tolist() for _ in indices],
            "warmup_per_model": 20,
            "batch_size": 1,
            "precision": "FP32_no_TF32",
            "scope": "prepared event tensors resident on GPU through signed TTC output",
            "excludes": ["sensor preparation", "host-to-device copy", "weight loading"],
            "exploration_cost_statistic": "paired population warm median",
        }
        freeze(COST / "REGISTRATION.json", binding)
        key = digest(binding)
        if (COST / "COMPLETE.json").exists():
            result = read(COST / "COMPLETE.json")
            if result["binding"] != key:
                raise ValueError("Geometry timing binding changed")
            for ordinal in range(len(indices)):
                validate_row(read(COST / f"row_{ordinal:03d}.json"), binding, ordinal)
            return result
        with gpu_charge("geometry_paired_forward_cost", reserve=900) as deadline:
            models = {}
            for name in names:
                if name == FIT:
                    model = load_model()
                else:
                    model = initialize(name)
                    if name == CONTROL:
                        payload = torch.load(
                            RUN / "fits" / name / "checkpoint_last.pt",
                            map_location="cpu",
                            weights_only=False,
                        )
                        model.load_state_dict(payload["model_state_dict"], strict=True)
                        del payload
                models[name] = TTCOnly(model).cuda().eval().requires_grad_(False)
            batch = dataset.batch([indices[0]])
            example = (batch.events.cuda(), batch.delta_t_s.cuda().reshape(1, 1))
            with torch.inference_mode():
                cold = {}
                for name in names:
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    models[name](*example)
                    torch.cuda.synchronize()
                    cold[name] = time.perf_counter() - tick
                    for _ in range(20):
                        models[name](*example)
                freeze(
                    COST / f"COLD_{time.time_ns()}.json",
                    {
                        "meaning": "first inference after process load; not disk-cold E2E",
                        "seconds": cold,
                        "binding": key,
                    },
                )
                for ordinal, (index, order) in enumerate(
                    zip(indices, binding["paired_orders"], strict=True)
                ):
                    path = COST / f"row_{ordinal:03d}.json"
                    if path.exists():
                        validate_row(read(path), binding, ordinal)
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "next_row": ordinal}
                    require_resources()
                    batch = dataset.batch([index])
                    example = (batch.events.cuda(), batch.delta_t_s.cuda().reshape(1, 1))
                    timings = {}
                    for name in order:
                        torch.cuda.reset_peak_memory_stats()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        output = models[name](*example)
                        torch.cuda.synchronize()
                        timings[name] = {
                            "seconds": time.perf_counter() - tick,
                            "ttc_seconds": float(output[0]),
                            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated()
                            - before,
                        }
                    row = {"binding": key, "P_index": index, "timings": timings, "utc": utc()}
                    validate_row(row, binding, ordinal)
                    freeze(path, row)
            gpu_name = torch.cuda.get_device_name(0)
        rows = [read(COST / f"row_{i:03d}.json") for i in range(len(indices))]
        summary = {}
        for name in names:
            elapsed = np.array([row["timings"][name]["seconds"] for row in rows])
            summary[name] = {
                **{f"p{q}_seconds": float(np.quantile(elapsed, q / 100)) for q in (50, 95, 99)},
                "batch1_windows_per_second": float(1 / elapsed.mean()),
                "n": len(rows),
                "max_extra_allocated_bytes": max(
                    row["timings"][name]["peak_extra_allocated_bytes"] for row in rows
                ),
            }
        result = {
            "status": "COMPLETE",
            "binding": key,
            "results": summary,
            "gpu_name": gpu_name,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "E2E": False,
            "official_comparison": False,
        }
        freeze(COST / "COMPLETE.json", result)
        return result
    finally:
        dataset.close()


def finalize_selection() -> dict:
    """Apply the unchanged coverage/sign/band protections to metric and cost routes."""
    endpoints = freeze_endpoint()["endpoints"]
    record = registered()
    metric = select_metrics()
    binding, cost = read(COST / "REGISTRATION.json"), read(COST / "COMPLETE.json")
    if (
        binding["endpoints"] != endpoints
        or binding["sources"] != sources()
        or binding["registration_sha256"] != sha(REGISTRATION)
        or cost["status"] != "COMPLETE"
        or cost["binding"] != digest(binding)
        or set(cost["results"]) != set(endpoints)
    ):
        raise ValueError("Geometry cost evidence is incomplete or has changed")
    ratio = cost["results"][FIT]["p50_seconds"] / cost["results"][BASELINE]["p50_seconds"]
    decision = combine_decision(metric["decision"], ratio, record["development_gates"]["routes"])
    decision["cost_scope"] = "prepared event tensor resident on GPU through signed TTC output"
    result = {
        "status": "DEVELOPMENT_CANDIDATE" if decision["eligible"] else "NO_CANDIDATE_C1",
        "ranked": [FIT] if decision["eligible"] else [],
        "decision": decision,
        "reference": BASELINE,
        "reused_control": CONTROL,
        "metric_evidence_sha256": sha(DIRECTORY / "H_METRIC_SELECTION.json"),
        "cost_evidence_sha256": sha(COST / "COMPLETE.json"),
        "system_promotion": False,
        "official_comparison": False,
        "next": "Interpret paired P/H; system promotion and replication remain separate",
    }
    freeze(DIRECTORY / "SELECTION_FINAL.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["measure", "select"])
    args = parser.parse_args()
    print(measure() if args.action == "measure" else finalize_selection())
