"""Synthetic-only objective parity and actual CUDA graph timing; no integration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch

from operational.phase_event_expansion.model import temporal_agreement, warp_endpoint
from operational.phase_event_expansion.synthetic import make_fixture
from operational.phase_event_expansion.torch_objective import (
    CachedTorchObjective,
    CudaGraphObjective,
)


def compare_objectives(reference: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    """Report absolute tolerance and ordering, identifying near-ties explicitly."""
    tolerance = 1e-6
    gap = reference[:, None] - reference[None, :]
    actual_gap = actual[:, None] - actual[None, :]
    material = abs(gap) > 2 * tolerance
    return {
        "max_absolute_error": float(np.max(abs(reference - actual))),
        "within_tolerance": bool(
            np.isfinite(actual).all() and np.max(abs(reference - actual)) <= tolerance
        ),
        "exact_order_matches": bool(np.array_equal(np.argsort(reference), np.argsort(actual))),
        "material_order_matches": bool(
            np.all(np.sign(gap[material]) == np.sign(actual_gap[material]))
        ),
        "argmin_matches": bool(np.argmin(reference) == np.argmin(actual)),
        "reference": reference.tolist(),
        "actual": actual.tolist(),
    }


def numpy_objective(
    points: np.ndarray, age: np.ndarray, polarity: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """Evaluate the frozen solver's objective, using its actual numerical helpers."""
    early = age >= (age.min() + age.max()) / 2
    result = []
    for candidate in parameters:
        scale = 1 + candidate[0] * age
        if np.any(scale <= 0):
            result.append(1000 + float(np.maximum(0, -scale).sum()))
        else:
            score, lost = temporal_agreement(
                warp_endpoint(points, age, candidate), polarity, early, 128, 1.0, 0.8
            )
            result.append(-score + 100 * lost)
    return np.asarray(result)


def measure(call: Callable[[], object], repetitions: int, cuda: bool = False) -> dict[str, Any]:
    """Measure synchronized wall latency, retaining every sample in seconds."""
    samples = []
    for _ in range(repetitions):
        if cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        if cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return {
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.quantile(samples, 0.95)),
        "repetitions": repetitions,
    }


def main() -> None:
    """Preload synthetic observations; compare float64 and explicitly experimental float32."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh immutable profile directory")
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires actual CUDA hardware")
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()
    paths = [
        Path(__file__),
        Path("operational/phase_event_expansion/torch_objective.py"),
        Path("operational/phase_event_expansion/model.py"),
        Path("operational/phase_event_expansion/synthetic.py"),
        Path("tests/unit/test_phase_event_expansion_torch.py"),
    ]
    config = {
        "schema": "phase_event_objective_profile_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "events": 20_000,
        "seed": 7,
        "canvas": [128, 128],
        "channels": 4,
        "gaussian": {"sigma": 0.8, "truncate": 4, "radius": 3, "padding": "half_sample_symmetric"},
        "precision": ["float64", "float32_experimental"],
        "batch_sizes": [1, 81],
        "tolerance_absolute": 1e-6,
        "memory_limit_bytes": 1_000_000_000,
        "timing": "2 torch CPU threads; CUDA synchronize around every sample; "
        "resident timings exclude host result transfer; callback includes it",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
            "host": platform.node(),
            "tf32": False,
        },
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "scope": "synthetic events only; no fixture labels read; no solver or deployment changes",
    }
    args.output.mkdir(parents=True)
    (args.output / "CONFIG.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for path in paths:
        destination = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    fixture = make_fixture(7, "expansion", count=20_000)
    points = (fixture.events[:, :2] - 128) / 256
    age = (fixture.query_us - fixture.events[:, 2]) / 1e6
    polarity = fixture.events[:, 3]
    grid = np.array(list(product(np.linspace(-10, 10, 9), (-2.5, 0.0, 2.5), (-2.5, 0.0, 2.5))))
    rng = np.random.default_rng(7)
    stress_points = rng.uniform(-1.1, 1.1, (512, 2))
    stress_points[:8] = [
        [-1, -1],
        [1, 1],
        [1 - 1e-12, 0],
        [0, 1 - 1e-12],
        [-1 + 1e-12, 0],
        [0, -1 + 1e-12],
        [-0.99, 0.99],
        [0.99, -0.99],
    ]
    stress_age = np.linspace(0, 0.2, len(stress_points))
    stress_polarity = rng.choice([-1, 1], len(stress_points))
    stress_parameters = np.array(
        [[0.0, 0.0, 0.0], [-10.0, -5.0, 5.0], [-5.0, 0.0, 0.0], [10.0, 5.0, -5.0]]
    )
    stress_reference = numpy_objective(
        stress_points, stress_age, stress_polarity, stress_parameters
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for dtype in (torch.float64, torch.float32):
            precision = str(dtype).split(".")[-1]
            stress_model = CachedTorchObjective(
                stress_points, stress_age, stress_polarity, dtype=dtype, device="cuda"
            )
            stress_actual = stress_model(
                torch.tensor(stress_parameters, dtype=dtype, device="cuda")
            )
            stress = compare_objectives(stress_reference, stress_actual.cpu().numpy())
            del stress_model, stress_actual
            for batch in (1, 81):
                parameters = grid[40:41].copy() if batch == 1 else grid.copy()
                reference = numpy_objective(points, age, polarity, parameters)
                cpu_numpy = measure(partial(numpy_objective, points, age, polarity, parameters), 10)
                cpu_model = CachedTorchObjective(points, age, polarity, dtype=dtype)
                cpu_parameters = torch.tensor(parameters, dtype=dtype)
                cpu_actual = cpu_model(cpu_parameters).numpy()
                cpu_torch = measure(partial(cpu_model, cpu_parameters), 10)
                del cpu_model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                model = CachedTorchObjective(points, age, polarity, dtype=dtype, device="cuda")
                candidates = torch.tensor(parameters, dtype=dtype, device="cuda")
                torch.cuda.synchronize()
                preload_seconds = time.perf_counter() - started
                started = time.perf_counter()
                for _ in range(5):
                    model(candidates)
                torch.cuda.synchronize()
                warmup_seconds = time.perf_counter() - started
                actual = model(candidates).cpu().numpy()
                cuda_eager = measure(partial(model, candidates), 30, cuda=True)
                captured = CudaGraphObjective(model, batch)
                graph_actual = captured(candidates).cpu().numpy().copy()
                cuda_graph = measure(partial(captured, candidates), 30, cuda=True)
                callback = measure(
                    lambda runner=captured, values=cpu_parameters: runner(values).cpu().numpy(),
                    30,
                    cuda=True,
                )
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
                comparisons = {
                    "cpu_torch": compare_objectives(reference, cpu_actual),
                    "cuda_eager": compare_objectives(reference, actual),
                    "cuda_graph": compare_objectives(reference, graph_actual),
                    "boundary_and_scale_stress": stress,
                }
                parity_ok = all(
                    r["within_tolerance"] and r["material_order_matches"]
                    for r in comparisons.values()
                )
                speedup = cpu_numpy["median_seconds"] / callback["median_seconds"]
                memory_ok = max(peak_allocated, peak_reserved) < config["memory_limit_bytes"]
                row = {
                    "precision": precision,
                    "batch": batch,
                    "comparisons": comparisons,
                    "timings": {
                        "numpy_scipy": cpu_numpy,
                        "cpu_torch": cpu_torch,
                        "cuda_eager_resident": cuda_eager,
                        "cuda_graph_resident": cuda_graph,
                        "cuda_graph_host_callback": callback,
                    },
                    "setup": {
                        "preload_seconds": preload_seconds,
                        "eager_warmup5_seconds": warmup_seconds,
                        "capture_warmup3_seconds": captured.warmup_seconds,
                        "capture_seconds": captured.capture_seconds,
                    },
                    "capture_count": captured.capture_count,
                    "replay_count": captured.replay_count,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "parity_pass": bool(parity_ok),
                    "memory_pass": bool(memory_ok),
                    "callback_speedup_vs_numpy": speedup,
                    "eligible_for_separate_integration_review": bool(
                        parity_ok and memory_ok and speedup > 1
                    ),
                    "production_integrated": False,
                }
                rows.append(row)
                (args.output / f"{precision}_batch{batch}.json").write_text(
                    json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                print(
                    f"{precision} B{batch}: parity={parity_ok} memory={peak_reserved / 1e6:.1f}MB "
                    f"callback_speedup={speedup:.2f}x capture={captured.capture_count} "
                    f"replay={captured.replay_count}",
                    flush=True,
                )
                del captured, model, candidates
                gc.collect()
                torch.cuda.empty_cache()
    (args.output / "SUMMARY.json").write_text(
        json.dumps({"config": config, "rows": rows}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
