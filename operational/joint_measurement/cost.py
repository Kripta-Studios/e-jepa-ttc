"""Paired batch-one forward timing, separate from raw sensor-to-output cost."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from .common import ROOT, RUN, config, gpu_charge, require_resources
from .data import SensorDataset
from .evaluate import development_decision, freeze_primary
from .export import TTCOnly
from .train import initialize, setup


def measure() -> dict:
    """Interleave frozen models on identical P inputs, with native TTC conversion."""
    endpoints = freeze_primary()["endpoints"]
    ids = ["A5-P2-FROZEN", *config()["fits"]]
    generator = np.random.default_rng(20261012)
    indices = generator.choice(53890, 128, replace=False).tolist()
    orders = [generator.permutation(ids).tolist() for _ in indices]
    directory = RUN / "cost/primary_prepared_forward"
    binding = {
        "endpoints": endpoints,
        "source": sha(ROOT / "operational/joint_measurement/cost.py"),
        "population": "P",
        "indices": indices,
        "paired_orders": orders,
        "batch_size": 1,
        "warmup_per_model": 20,
        "dtype": "FP32",
        "TF32": False,
        "scope": "prepared tensor resident on GPU through signed TTC output",
        "excludes": ["sensor preprocessing", "host-to-device transfer", "weight loading"],
        "exploration_cost_statistic": "paired population warm median",
    }
    freeze(directory / "REGISTRATION.json", binding)
    if (directory / "COMPLETE.json").exists():
        return read(directory / "COMPLETE.json")
    require_resources(starting=True)
    setup(7)
    dataset = SensorDataset("P")
    try:
        with gpu_charge("primary_paired_forward_cost", reserve=900) as deadline:
            models = {}
            for fit_id in ids:
                model = initialize(fit_id)
                if fit_id != "A5-P2-FROZEN":
                    payload = torch.load(
                        RUN / "fits" / fit_id / "checkpoint_last.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    model.load_state_dict(payload["model_state_dict"], strict=True)
                    del payload
                models[fit_id] = TTCOnly(model).cuda().eval().requires_grad_(False)
            sample = dataset.batch([indices[0]])
            example = (sample.events.cuda(), sample.delta_t_s.cuda().reshape(1, 1))
            cold = {}
            with torch.inference_mode():
                for fit_id in ids:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    models[fit_id](*example)
                    torch.cuda.synchronize()
                    cold[fit_id] = time.perf_counter() - start
                    for _ in range(binding["warmup_per_model"]):
                        models[fit_id](*example)
                for row_number, (index, order) in enumerate(zip(indices, orders, strict=True)):
                    path = directory / f"row_{row_number:03d}.json"
                    if path.exists():
                        if read(path)["binding"] != digest(binding):
                            raise ValueError("Timing fragment binding changed")
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "next_row": row_number}
                    sample = dataset.batch([index])
                    example = (sample.events.cuda(), sample.delta_t_s.cuda().reshape(1, 1))
                    record = {}
                    for fit_id in order:
                        torch.cuda.reset_peak_memory_stats()
                        baseline = torch.cuda.memory_allocated()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = models[fit_id](*example)
                        torch.cuda.synchronize()
                        record[fit_id] = {
                            "seconds": time.perf_counter() - start,
                            "ttc_seconds": float(output[0]),
                            "peak_extra_allocated_bytes": (
                                torch.cuda.max_memory_allocated() - baseline
                            ),
                        }
                    freeze(
                        path,
                        {
                            "binding": digest(binding),
                            "P_index": index,
                            "timings": record,
                            "utc": utc(),
                        },
                    )
            write(
                directory / f"COLD_{time.time_ns()}.json",
                {"meaning": "first inference after process load", "seconds": cold},
            )
        frames = [read(directory / f"row_{i:03d}.json") for i in range(len(indices))]
        summary = {}
        for fit_id in ids:
            seconds = np.array([r["timings"][fit_id]["seconds"] for r in frames])
            summary[fit_id] = {
                "p50_seconds": float(np.quantile(seconds, 0.5)),
                "p95_seconds": float(np.quantile(seconds, 0.95)),
                "p99_seconds": float(np.quantile(seconds, 0.99)),
                "batch1_windows_per_second": float(1 / seconds.mean()),
                "n": len(seconds),
                "max_extra_allocated_bytes": max(
                    r["timings"][fit_id]["peak_extra_allocated_bytes"] for r in frames
                ),
            }
        result = {
            "status": "COMPLETE",
            "binding": digest(binding),
            "results": summary,
            "E2E": False,
            "official_comparison": False,
        }
        freeze(directory / "COMPLETE.json", result)
        return result
    finally:
        dataset.close()


def finalize_selection() -> dict:
    """Combine registered metric and cost routes before V or adaptive decisions."""
    freeze_primary()
    cost = read(RUN / "cost/primary_prepared_forward/COMPLETE.json")
    if cost["status"] != "COMPLETE":
        raise ValueError("Cost route requires the complete paired comparison")
    baseline = "A5-P2-FROZEN"
    for name in [baseline, *config()["fits"]]:
        folder = RUN / "predictions/H" / name
        receipt = read(folder / "COMPLETE.json")
        if (
            receipt["status"] != "COMPLETE"
            or sha(folder / "predictions.parquet") != receipt["sha256"]
        ):
            raise ValueError("Selection requires immutable complete H predictions")
    reference = read(RUN / "predictions/H" / baseline / "METRICS.json")
    decisions, outcomes = {}, {}
    routes = config()["selection"]["routes"]
    for fit_id in config()["fits"]:
        outcome = read(RUN / "predictions/H" / fit_id / "METRICS.json")
        decision = development_decision(reference, outcome)
        ratio = cost["results"][fit_id]["p50_seconds"] / cost["results"][baseline]["p50_seconds"]
        cost_route = (
            all(decision.get("checks", {}).values())
            and "overall_ratio" in decision
            and decision["overall_ratio"] <= routes["cost_route_overall_ratio_max"]
            and ratio <= routes["cost_ratio_max"]
        )
        decisions[fit_id] = {
            **decision,
            "paired_cost_ratio": ratio,
            "cost_route_pass": cost_route,
            "eligible": decision["eligible"] or cost_route,
        }
        outcomes[fit_id] = outcome
    eligible = [name for name, decision in decisions.items() if decision["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda name: (
            outcomes[name]["overall_MiD"],
            outcomes[name]["bins"]["crucial"]["group_macro_mid"],
            name,
        ),
    )[: config()["selection"]["max_integrations"]]
    result = {
        "status": "DEVELOPMENT_CANDIDATE" if ranked else "NO_CANDIDATE_PRIMARY",
        "ranked": ranked,
        "decisions": decisions,
        "system_promotion": False,
        "next": "observe P/H mechanisms, register enabled continuation, then frozen V comparison",
    }
    freeze(RUN / "PRIMARY_SELECTION_FINAL.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["measure", "select"])
    args = parser.parse_args()
    print(measure() if args.action == "measure" else finalize_selection())
