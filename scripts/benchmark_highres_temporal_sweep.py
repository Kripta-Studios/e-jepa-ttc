"""Benchmark S4/S5 temporal scaling without selecting an architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig

TEMPORAL_STEPS = (2, 5, 8, 16, 32)
ARMS = (
    ("S4_R4_WINDOW_MERGE_TEMPORAL", "block_causal"),
    ("S5_R4_WINDOW_MERGE_KDA", "kda"),
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(*, device_name: str, output: Path, warmups: int, repeats: int) -> dict[str, Any]:
    device = torch.device(device_name)
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive.")
    rows: list[dict[str, Any]] = []
    for temporal_steps in TEMPORAL_STEPS:
        torch.manual_seed(23)
        inputs = torch.randn(1, temporal_steps, 21, 192, 320, device=device)
        for arm, mixer in ARMS:
            torch.manual_seed(7)
            model = (
                EJEPATubeletLHR(
                    EJEPATubeletLHRConfig(
                        in_channels=21,
                        embed_dim=32,
                        patch_size=8,
                        spatial_window=8,
                        heads=4,
                        spatial_depth=1,
                        temporal_depth=1,
                        temporal_mixer=mixer,
                        merge_2x2=True,
                        pooling="query",
                        global_attention=False,
                        memory_budget_gb=12.0,
                    )
                )
                .to(device)
                .eval()
            )
            if device.type == "cuda":
                torch.cuda.set_device(device)
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for _ in range(warmups):
                    model(inputs)
                _sync(device)
                timings: list[float] = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    model(inputs)
                    _sync(device)
                    timings.append((time.perf_counter() - start) * 1000.0)
            output_value = model(inputs)
            _sync(device)
            rows.append(
                {
                    "arm": arm,
                    "temporal_mixer": mixer,
                    "readout": "query",
                    "temporal_steps": temporal_steps,
                    "input_resolution": [320, 192],
                    "patch_size": 8,
                    "merge_2x2": True,
                    "tokens_after_merge": int(
                        output_value.tokens.shape[1] * output_value.tokens.shape[2]
                    ),
                    "temporal_attention_pairs": int(
                        temporal_steps * temporal_steps * output_value.tokens.shape[2]
                    ),
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "forward_p50_ms": float(torch.tensor(timings).quantile(0.50)),
                    "forward_p95_ms": float(torch.tensor(timings).quantile(0.95)),
                    "peak_vram_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else None
                    ),
                    "global_attention_used": False,
                    "selection_allowed": False,
                }
            )
            del model, output_value
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del inputs
    result: dict[str, Any] = {
        "artifact_type": "highres_temporal_sweep_v1",
        "schema_version": "v1",
        "evidence_type": "synthetic_forward_scaling_only",
        "code_commit": _git_commit(),
        "protocol_version": "highres_s4_s5_temporal_sweep_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "device": str(device),
        "warmups": warmups,
        "repeats": repeats,
        "temporal_steps": list(TEMPORAL_STEPS),
        "arms": [arm for arm, _ in ARMS],
        "selection_allowed": False,
        "metrics_scope": "forward_latency_and_theoretical_scaling_only",
        "results": rows,
        "status": "pass",
    }
    result["artifact_sha256"] = _canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/highres_temporal_sweep_v1.json"),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    try:
        result = benchmark(
            device_name=args.device,
            output=args.output,
            warmups=args.warmups,
            repeats=args.repeats,
        )
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "highres_temporal_sweep_failure_v1",
            "status": "failed",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_path = args.output.with_name(f"{args.output.stem}.FAILURE.json")
        failure_path.write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
