"""A1-only parity admission for the existing captured inference and lossless prefetch."""

from __future__ import annotations

import statistics
import time
from unittest.mock import patch

import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import freeze, read, sha, write

from . import adaptation_evaluate as a1
from . import geometry_evaluate as evaluation
from .common import gpu_charge
from .correspondence_inference_graph import InferenceGraph
from .data import SensorDataset
from .prefetched_sensor import PrefetchedSensor
from .train import setup


def probe() -> dict:
    """Check new weights on P only after both endpoints freeze; retain any failed probe."""
    endpoints = a1.freeze_endpoints()
    out = a1.DIRECTORY / "evaluation/inference_runtime"
    binding = {
        "evaluation_registration_sha256": sha(a1.EVALUATION),
        "endpoints": endpoints,
        "sources": a1.source_binding(),
        "P_indices": list(range(64)) + [53888, 53889],
        "parity": "Exact native pixels, clocks, phase, TTC; model and RNG unchanged",
        "maximum_median_warm_graph_to_eager_ratio": 0.9,
        "optimizer_updates": 0,
    }
    freeze(out / "REGISTRATION.json", binding)
    if (out / "RESULT.json").exists():
        result = read(out / "RESULT.json")
        if result["registration_sha256"] != sha(out / "REGISTRATION.json"):
            raise ValueError("A1 inference proof binding changed")
        return result
    measurements, checks = [], []
    result = {"registration_sha256": sha(out / "REGISTRATION.json"), "optimizer_updates": 0}
    try:
        setup(7)
        with gpu_charge("A1_exact_inference_graph_probe", reserve=900), torch.inference_mode():
            for name in (a1.FIT, a1.CONTROL):
                with a1.predict_context(name):
                    model = evaluation.load_model().cuda().eval().requires_grad_(False)
                candidate = InferenceGraph(model)
                original, prefetched = SensorDataset("P"), PrefetchedSensor("P", ahead=False)
                weights = {key: value.clone() for key, value in model.state_dict().items()}
                cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                try:
                    indices = [list(range(start, start + 4)) for start in range(0, 64, 4)] + [
                        [53888, 53889]
                    ]
                    for ordinal, ids in enumerate(indices):
                        left, right = original.batch(ids), prefetched.batch(ids)
                        if not torch.equal(left.events, right.events) or not torch.equal(
                            left.delta_t_s.reshape(-1), right.delta_t_s.reshape(-1)
                        ):
                            raise ValueError("A1 exact pixel/clock parity failed")
                        pixels, dt = right.events.cuda(), right.delta_t_s.cuda().reshape(-1)
                        # Capture/warm this exact signature before the timed comparison.
                        candidate(pixels, dt)
                        actual, elapsed = {}, {}
                        order = ("eager", "graph") if ordinal % 2 == 0 else ("graph", "eager")
                        for route in order:
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            raw = (model if route == "eager" else candidate)(pixels, dt)[
                                "raw_phase"
                            ]
                            ttc = phase_to_ttc(raw)
                            torch.cuda.synchronize()
                            elapsed[route] = time.perf_counter() - start
                            actual[route] = (raw.clone(), ttc.clone())
                        if not all(
                            torch.equal(x, y)
                            for x, y in zip(actual["eager"], actual["graph"], strict=True)
                        ):
                            raise ValueError("A1 phase/TTC graph parity failed")
                        measurements.append(
                            {"fit": name, "ids": ids, "seconds": elapsed, "order": order}
                        )
                    invalid = dt.clone()
                    invalid[0] = 0
                    rejected = False
                    try:
                        candidate(pixels, invalid)
                    except ValueError as error:
                        rejected = "guard failed" in str(error)
                    check = {
                        "fit": name,
                        "weights_unchanged": all(
                            torch.equal(value, model.state_dict()[key])
                            for key, value in weights.items()
                        ),
                        "cpu_rng_unchanged": torch.equal(cpu_rng, torch.get_rng_state()),
                        "cuda_rng_unchanged": all(
                            torch.equal(x, y)
                            for x, y in zip(gpu_rng, torch.cuda.get_rng_state_all(), strict=True)
                        ),
                        "invalid_clock_rejected": rejected,
                    }
                    if not all(value for key, value in check.items() if key != "fit"):
                        raise ValueError(f"A1 inference integrity failed: {check}")
                    checks.append(check)
                finally:
                    original.close()
                    prefetched.close()
                del candidate, model, weights
                torch.cuda.empty_cache()
        ratio = statistics.median(
            row["seconds"]["graph"] / row["seconds"]["eager"] for row in measurements
        )
        result.update(
            status="ADMITTED" if ratio <= 0.9 else "REJECTED_SPEED",
            paired_median_ratio=ratio,
            checks=checks,
            measurements=measurements,
            exact_pixels_clocks_phase_ttc=True,
        )
    except Exception as error:
        result.update(
            status="REJECTED_ERROR", error=repr(error), checks=checks, measurements=measurements
        )
    freeze(out / "RESULT.json", result)
    return result


def predict(name: str, role: str) -> dict:
    """Preserve complete original cohorts; use a rejected optimization's eager fallback."""
    if role not in ("P", "H") or name not in (a1.FIT, a1.CONTROL):
        raise ValueError("Only the two A1 endpoints and P/H are admitted")
    endpoints = a1.freeze_endpoints()
    proof_path = a1.DIRECTORY / "evaluation/inference_runtime/RESULT.json"
    proof = read(proof_path)
    binding = read(proof_path.parent / "REGISTRATION.json")
    if proof["registration_sha256"] != sha(proof_path.parent / "REGISTRATION.json"):
        raise ValueError("A1 inference proof changed")
    if (
        binding["sources"] != a1.source_binding()
        or binding["endpoints"] != endpoints
        or binding["evaluation_registration_sha256"] != sha(a1.EVALUATION)
        or proof["status"] not in ("ADMITTED", "REJECTED_SPEED", "REJECTED_ERROR")
    ):
        raise ValueError("A1 inference admission sources or endpoints changed")
    enabled = proof["status"] == "ADMITTED"
    directory = a1.DIRECTORY / "predictions" / role / name
    freeze(
        directory / "INFERENCE_RUNTIME.json",
        {
            "proof_sha256": sha(proof_path),
            "sources": a1.source_binding(),
            "route": "graph_and_lossless_prefetch" if enabled else "original_eager_fallback",
            "batch_precision_population_unchanged": True,
        },
    )
    original_load = evaluation.load_model
    readers = []

    def reader(cohort: str) -> PrefetchedSensor:
        data = PrefetchedSensor(cohort)
        readers.append(data)
        return data

    started = time.perf_counter()
    with a1.predict_context(name):
        if enabled:
            with (
                patch.object(evaluation, "SensorDataset", reader),
                patch.object(evaluation, "load_model", lambda: InferenceGraph(original_load())),
            ):
                result = evaluation.predict(role)
        else:
            result = evaluation.predict(role)
    write(
        directory / "RUNTIME_LATEST.json",
        {
            "elapsed_s": time.perf_counter() - started,
            "status": result["status"],
            "packs": [row for data in readers for row in data.timings],
            "scope": "Prediction throughput including read/startup/drain; not model cost benchmark",
        },
    )
    return result
