"""Compare exact sort and fixed-tail top-k event normalization on prepared voxels.

This is a synthetic voxel-normalization microbenchmark, not raw sensor -> TTC.
Each measured call includes copying identical prepared input into reusable scratch
and normalizing all six windows per object. It excludes ingest, ROI projection,
scalar count channels, transfers and every neural model. CUDA requires an exclusive
GPU measurement slot assigned by the coordinating agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from operational.train40_runtime.measurement import latency_distribution  # noqa: E402
from operational.train40_runtime.normalization import ExactTopKNormalizer  # noqa: E402


class SortNormalizer:
    """The frozen ring's sorted-quantile arithmetic with identical out buffers."""

    def __init__(self, rows: int, columns: int, device: str) -> None:
        self.occupied = torch.empty((rows, columns), dtype=torch.bool, device=device)
        self.values = torch.empty((rows, columns), dtype=torch.float64, device=device)
        self.indices = torch.empty((rows, columns), dtype=torch.int64, device=device)

    def normalize_(self, voxel: Tensor) -> Tensor:
        """Apply the unchanged P95 rank and arithmetic in place."""
        occupied, values = self.occupied, self.values
        torch.gt(voxel, 1e-6, out=occupied)
        nonzero = occupied.sum(1)
        torch.sort(voxel, dim=1, out=(values, self.indices))
        rank = (nonzero - 1).clamp_min(0).double() * 0.95
        lower, upper = rank.floor().long(), rank.ceil().long()
        offset = voxel.shape[1] - nonzero
        a = values.gather(1, (offset + lower).clamp_max(voxel.shape[1] - 1)[:, None])[:, 0]
        b = values.gather(1, (offset + upper).clamp_max(voxel.shape[1] - 1)[:, None])[:, 0]
        fraction = rank - lower
        scale = torch.where(fraction >= 0.5, b - (b - a) * (1.0 - fraction), a + (b - a) * fraction)
        voxel.div_(scale.clamp_min(1e-6)[:, None])
        voxel.masked_fill_(~occupied, 0.0)
        return scale


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _case(objects: int, occupancy: float, args: argparse.Namespace) -> dict[str, Any]:
    rows, columns = objects * 6, 10 * args.roi_size**2
    generator = torch.Generator().manual_seed(args.seed)
    # Fixed CPU generation; no preparation is misreported as model latency.
    raw = torch.rand((rows, columns), dtype=torch.float64, generator=generator)
    raw[raw < 1.0 - occupancy] = 0.0
    source = raw.to(args.device)
    normalizers = {
        "sort": SortNormalizer(rows, columns, args.device),
        "topk": ExactTopKNormalizer(rows, columns, device=args.device),
    }
    work = {name: torch.empty_like(source) for name in normalizers}

    def execute(name: str) -> Tensor:
        work[name].copy_(source)
        return normalizers[name].normalize_(work[name])

    scales = {name: execute(name) for name in normalizers}
    _sync(args.device)
    parity = {
        "normalized_float64_exact": torch.equal(work["sort"], work["topk"]),
        "normalized_float32_exact": torch.equal(work["sort"].float(), work["topk"].float()),
        "scale_float64_exact": torch.equal(scales["sort"], scales["topk"]),
        "max_abs": float((work["sort"] - work["topk"]).abs().max().cpu()),
    }
    for _ in range(args.warmup):
        for name in normalizers:
            execute(name)
    _sync(args.device)
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    if args.graphs:
        for name in normalizers:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                execute(name)
            graphs[name] = graph
        _sync(args.device)
    samples: dict[str, list[float]] = {name: [] for name in normalizers}
    for index in range(args.repeats):
        order = list(normalizers)
        if index % 2:
            order.reverse()
        for name in order:
            _sync(args.device)
            begin = time.perf_counter_ns()
            if args.graphs:
                graphs[name].replay()
            else:
                execute(name)
            _sync(args.device)
            samples[name].append((time.perf_counter_ns() - begin) / 1e6)
    parity["after_repetitions_exact"] = torch.equal(work["sort"], work["topk"])
    topk = normalizers["topk"]
    assert isinstance(topk, ExactTopKNormalizer)
    return {
        "objects": objects,
        "windows": rows,
        "columns": columns,
        "requested_occupancy": occupancy,
        "actual_occupied_values": int((raw > 1e-6).sum()),
        "topk_size": topk.tail_size,
        "parity": parity,
        "passed": all(value for key, value in parity.items() if key != "max_abs"),
        "latency_ms": {name: latency_distribution(values) for name, values in samples.items()},
        "samples_ms": samples,
        "topk_retained_scratch_bytes": topk.allocated_bytes,
        "sort_retained_scratch_bytes": rows * columns * 17,
    }


def run(args: argparse.Namespace) -> None:
    """Keep fresh, source-bound evidence even if a parity or execution check fails."""
    if (
        args.roi_size < 1
        or args.repeats < 1
        or args.warmup < 0
        or any(number < 1 or number > 20 for number in args.objects)
        or len(set(args.objects)) != len(args.objects)
        or any(not 0 <= fraction <= 1 for fraction in args.occupancy)
        or (args.graphs and torch.device(args.device).type != "cuda")
    ):
        raise ValueError(
            "Invalid dimensions, repetitions, populations, occupancy, or CUDA graph mode"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = [Path(__file__), ROOT / "operational/train40_runtime/normalization.py"]
    sources = {str(path.relative_to(ROOT)): _sha(path) for path in paths}
    binding = {
        "scope": "SYNTHETIC_PREPARED_VOXEL_NORMALIZATION_WITH_RESET_COPY_NO_SENSOR_NO_MODELS",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if torch.device(args.device).type == "cuda" else None,
        "seed": args.seed,
        "precision": "FLOAT64_NORMALIZATION_FLOAT32_OUTPUT_TF32_DISABLED",
        "cuda_graphs": args.graphs,
        "ordering": "alternate sort/topk order each paired repetition",
        "warmup": args.warmup,
        "sources": sources,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "candidate_loaded": False,
        "full_pipeline_claim": False,
    }
    _write(args.output / "binding.json", binding)
    results = []
    error = None
    try:
        with torch.inference_mode():
            for objects in args.objects:
                for occupancy in args.occupancy:
                    result = _case(objects, occupancy, args)
                    results.append(result)
                    _write(args.output / "cases.json", results)
                    print(json.dumps(result, allow_nan=False), flush=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        unchanged = {str(path.relative_to(ROOT)): _sha(path) for path in paths} == sources
        _write(
            args.output / "summary.json",
            {
                "passed": bool(results)
                and len(results) == len(args.objects) * len(args.occupancy)
                and all(row["passed"] for row in results)
                and error is None
                and unchanged,
                "error": error,
                "sources_unchanged": unchanged,
                "results": results,
                "full_pipeline_claim": False,
            },
        )
    if not unchanged or not all(row["passed"] for row in results):
        raise RuntimeError("Normalization probe failed parity/source binding")


def main() -> None:
    """Parse an explicitly scoped microbenchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--objects", nargs="+", type=int, default=[1, 20])
    parser.add_argument("--occupancy", nargs="+", type=float, default=[0.0, 0.01, 0.5, 1.0])
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--graphs", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
