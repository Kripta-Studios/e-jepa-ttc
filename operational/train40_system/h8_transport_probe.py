"""Bounded synthetic transport replay probe; never trains or modifies production."""

from __future__ import annotations

import statistics
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from operational.efficient_context.common import ROOT, digest, read
from operational.train40_system.durable_io import atomic_json


def tensor_bytes(value: torch.Tensor) -> bytes:
    """Preserve dtype bits, including signed zero, in parity comparisons."""
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def fingerprint(result: tuple[Any, ...]) -> list[bytes | None]:
    """Include all correlation fields as well as physical summaries."""
    flattened: list[bytes | None] = []
    for item in result:
        values = (
            [item]
            if isinstance(item, torch.Tensor)
            else [getattr(item, field.name) for field in fields(item)]
        )
        flattened.extend(None if value is None else tensor_bytes(value) for value in values)
    return flattened


def main() -> None:
    """Measure only function-level feasibility while retaining the live H8 process."""
    from e_jepa_ttc.models import causal_scale_ttc as model_module
    from operational.train40_system.h8_transport_graph import H8TransportGraph

    output = ROOT / "artifacts/train40_system_20261005"
    directory = output / "h8_bottleneck_20261007"
    report: dict[str, Any] = {
        "status": "RUNNING",
        "started_utc": datetime.now(UTC).isoformat(),
        "optimizer_updates": 0,
        "synthetic_inputs_only": True,
        "production_interrupted": False,
        "production_admission": False,
        "hardware_settings_changed": False,
        "limitation": "Function microbenchmark concurrent with production; not end-to-end speedup.",
        "source_sha256": digest(Path(__file__)),
        "graph_source_sha256": digest(Path(__file__).with_name("h8_transport_graph.py")),
        "production_before": read(output / "H8_FEATURE_PROGRESS.json"),
    }
    destination = directory / "TRANSPORT_GRAPH_PROBE.json"
    atomic_json(destination, report)
    start = time.perf_counter()
    graph = None
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.25)
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError("Insufficient spare VRAM for bounded isolated probe")
        generator = torch.Generator(device="cuda").manual_seed(719)
        inputs = []
        for size, radius in ((32, 1), (16, 2)):
            cases = []
            for _ in range(2):
                cases.append(
                    (
                        torch.randn((32, 64, size, size), device="cuda", generator=generator),
                        torch.randn((32, 64, size, size), device="cuda", generator=generator),
                        torch.rand((32, 1, size, size), device="cuda", generator=generator),
                    )
                )
            inputs.append((radius, cases))
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
        original_match = model_module.local_correlation_match
        original_physical = model_module.transport_physical_features

        def workload(case: int) -> tuple[Any, ...]:
            results = []
            for radius, cases in inputs:
                previous, current, weight = cases[case]
                forward = model_module.local_correlation_match(
                    previous,
                    current,
                    radius=radius,
                    temperature=0.1,
                )
                reverse = model_module.local_correlation_match(
                    current,
                    previous,
                    radius=radius,
                    temperature=0.1,
                )
                summary = model_module.transport_physical_features(
                    forward,
                    reverse,
                    foreground_weight=weight,
                    radius=radius,
                )
                back = model_module.transport_physical_features(
                    reverse,
                    forward,
                    foreground_weight=weight,
                    radius=radius,
                )
                results.extend((forward, reverse, summary, back))
            return tuple(results)

        with torch.inference_mode():
            expected = [fingerprint(workload(case)) for case in (0, 1)]
            graph = H8TransportGraph(max_entries=8, warmup_calls=3)
            with graph.install():
                retained = workload(0)
                first = fingerprint(retained)
                second = fingerprint(workload(1))
                last = fingerprint(workload(0))
                if first != expected[0] or second != expected[1] or last != first:
                    raise ValueError("Synthetic exact-byte A-B-A parity failed")
                if fingerprint(retained) != first:
                    raise ValueError("Later replay overwrote previously returned outputs")
            eager_times, graph_times = [], []
            for iteration in range(6):
                order = (False, True) if iteration % 2 == 0 else (True, False)
                for candidate in order:
                    torch.cuda.synchronize()
                    begun = time.perf_counter()
                    if candidate:
                        with graph.install():
                            workload(iteration % 2)
                    else:
                        workload(iteration % 2)
                    torch.cuda.synchronize()
                    (graph_times if candidate else eager_times).append(time.perf_counter() - begun)
            if not torch.equal(cpu_rng, torch.get_rng_state()) or any(
                not torch.equal(left, right)
                for left, right in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
            ):
                raise ValueError("Probe changed global RNG state")
            if model_module.local_correlation_match is not original_match or (
                model_module.transport_physical_features is not original_physical
            ):
                raise ValueError("Function aliases were not restored")
            report.update(
                {
                    "status": "PASSED_SYNTHETIC_ONLY",
                    "exact_bytes_A_B_A": True,
                    "retained_outputs_unchanged": True,
                    "global_rng_unchanged": True,
                    "eager_seconds": eager_times,
                    "graph_seconds": graph_times,
                    "median_function_speed_ratio": statistics.median(eager_times)
                    / statistics.median(graph_times),
                    "graph": graph.snapshot(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if graph is not None:
            graph.close()
        report["elapsed_seconds"] = time.perf_counter() - start
        report["production_after"] = read(output / "H8_FEATURE_PROGRESS.json")
        report["finished_utc"] = datetime.now(UTC).isoformat()
        atomic_json(destination, report)


if __name__ == "__main__":
    main()
