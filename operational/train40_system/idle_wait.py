"""CPU-only collator-shaped idle-wait benchmark for fresh baseline/passive processes."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path

ITERATIONS = 30
SLEEP_SECONDS = 0.100
PROFILED_GPU_ARITHMETIC_SECONDS = 0.1093


def _blocktime(torch_root: Path) -> dict[str, object]:
    """Read the loaded Intel OpenMP blocktime when its documented symbol is available."""
    candidates = sorted(torch_root.rglob("libiomp5md.dll"))
    if not candidates:
        return {"available": False, "value": None, "library": None}
    library = ctypes.WinDLL(str(candidates[0]))
    getter = getattr(library, "kmp_get_blocktime", None)
    if getter is None:
        return {"available": False, "value": None, "library": str(candidates[0])}
    getter.argtypes = []
    getter.restype = ctypes.c_int
    return {
        "available": True,
        "value": int(getter()),
        "library": str(candidates[0]),
    }


def _tensor_sha256(tensors: Sequence[object]) -> str:
    import torch

    result = hashlib.sha256()
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("CPU benchmark digest accepts tensors only")
        array = tensor.detach().cpu().contiguous().numpy()
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def run(output: Path, mode: str) -> None:
    """Measure collator CPU and wall cost without touching CUDA or an optimizer."""
    import psutil
    import torch

    from operational.train40_system.durable_io import atomic_json

    if mode not in {"baseline", "passive"}:
        raise ValueError("mode must be baseline or passive")
    if mode == "passive" and (
        os.environ.get("OMP_WAIT_POLICY") != "PASSIVE"
        or os.environ.get("KMP_BLOCKTIME") != "0"
    ):
        raise ValueError("Passive benchmark requires the admitted launcher environment")
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(401)
    records = [
        {
            "events": torch.randn((3, 10, 64, 64), generator=generator),
            "boxes": torch.randn((3, 4), generator=generator),
            "delta": torch.tensor(0.050, dtype=torch.float32),
            "motion": torch.randn((18,), generator=generator),
            "heights": torch.rand((2,), generator=generator) + 1.0,
            "target": torch.tensor(float(index + 1), dtype=torch.float32),
            "square": torch.tensor([0.0, 0.0, 64.0, 64.0]),
        }
        for index in range(32)
    ]

    def collate() -> list[torch.Tensor]:
        return [
            torch.stack([record["events"] for record in records]),
            torch.stack([record["boxes"] for record in records]),
            torch.stack([record["delta"] for record in records]),
            torch.stack([record["motion"] for record in records]),
            torch.stack([record["heights"] for record in records]),
            torch.stack([record["target"] for record in records]),
            torch.stack([record["square"] for record in records]),
        ]

    torch_root = Path(torch.__file__).resolve().parent
    blocktime_before = _blocktime(torch_root)
    process = psutil.Process()
    process_threads_before = process.num_threads()
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    batch: list[torch.Tensor] = []
    for _ in range(ITERATIONS):
        batch = collate()
        time.sleep(SLEEP_SECONDS)
    process_cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    blocktime_after = _blocktime(torch_root)
    tensor_sha256 = _tensor_sha256(batch)
    atomic_json(
        output,
        {
            "schema": "train40_idle_wait_cpu_benchmark_v1",
            "status": "COMPLETE",
            "mode": mode,
            "iterations": ITERATIONS,
            "sleep_seconds_per_iteration": SLEEP_SECONDS,
            "calibration": {
                "basis": "measured full-step GPU active-union arithmetic",
                "measured_seconds": PROFILED_GPU_ARITHMETIC_SECONDS,
                "benchmark_sleep_seconds": SLEEP_SECONDS,
                "earlier_20ms_pilot_was_not_representative": True,
            },
            "batch_shapes": [list(tensor.shape) for tensor in batch],
            "process_cpu_seconds": process_cpu_seconds,
            "wall_seconds": wall_seconds,
            "process_cpu_ms_per_iteration": process_cpu_seconds * 1000 / ITERATIONS,
            "wall_ms_per_iteration": wall_seconds * 1000 / ITERATIONS,
            "tensor_sha256": tensor_sha256,
            "environment": {
                "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"),
                "KMP_BLOCKTIME": os.environ.get("KMP_BLOCKTIME"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            },
            "torch_num_threads": torch.get_num_threads(),
            "process_threads_before": process_threads_before,
            "process_threads_after": process.num_threads(),
            "kmp_blocktime_before": blocktime_before,
            "kmp_blocktime_after": blocktime_after,
            "cuda_apis_called": False,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "passive"), required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.mode)
