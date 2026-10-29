"""Benchmark batch-1 model-only latency for a supervised TTC checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity
from e_jepa_ttc.models import build_regressor
from e_jepa_ttc.utils.io import write_structured


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_supervised_latency(
    *,
    cache_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str = "auto",
    warmup_iterations: int = 50,
    measured_iterations: int = 200,
) -> dict[str, Any]:
    """Measure synchronous model-only latency using one real cached input."""

    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("Warmup must be non-negative and measured iterations positive.")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )

    with np.load(cache_path, allow_pickle=False) as cache:
        x = torch.from_numpy(cache["x"][0:1].astype(np.float32, copy=False)).to(device)
        cache_declared_sha256 = (
            str(np.asarray(cache["cache_sha256"]).item()) if "cache_sha256" in cache.files else None
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = str(checkpoint["model_name"])
    model = build_regressor(model_name, in_channels=int(x.shape[1])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(x)
        _synchronize(device)
        latency_ms: list[float] = []
        for _ in range(measured_iterations):
            start = time.perf_counter_ns()
            model(x)
            _synchronize(device)
            latency_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    latency = np.asarray(latency_ms, dtype=np.float64)
    cache_sha256 = compute_file_hash(str(cache_path))
    if cache_declared_sha256 is not None and cache_declared_sha256 != cache_sha256:
        raise ValueError("Physical cache hash differs from its declared SHA-256.")
    protocol_version, protocol_sha256 = get_current_protocol_identity()
    payload: dict[str, Any] = {
        "artifact_type": "supervised_ttc_latency_benchmark",
        "schema_version": "1.0",
        "evidence_type": "validation_pilot",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "cache_path": cache_path.as_posix(),
        "cache_sha256": cache_sha256,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": compute_file_hash(str(checkpoint_path)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_name": model_name,
        "parameter_count": int(parameter_count),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "batch_size": 1,
        "model_only": True,
        "includes_event_voxelization": False,
        "synchronous_timing": True,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "latency_ms": {
            "mean": float(latency.mean()),
            "std": float(latency.std()),
            "median": float(np.median(latency)),
            "p95": float(np.quantile(latency, 0.95)),
            "p99": float(np.quantile(latency, 0.99)),
            "minimum": float(latency.min()),
            "maximum": float(latency.max()),
        },
        "windows_per_second_from_mean": float(1000.0 / latency.mean()),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "comparison_limitations": [
            "model-only latency excludes event reading and voxelization",
            (
                "not directly comparable across hardware, precision, runtime, "
                "or synchronization policy"
            ),
            "not an official Garl-TTC reproduction",
        ],
    }
    sign_artifact(payload)
    write_structured(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--measured-iterations", type=int, default=200)
    args = parser.parse_args()
    payload = benchmark_supervised_latency(
        cache_path=args.cache,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
