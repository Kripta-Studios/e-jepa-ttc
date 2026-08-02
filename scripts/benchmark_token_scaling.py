"""Benchmark factorized token scaling without allocating global R4 attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from e_jepa_ttc.models.highres_factorized import (
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
    theoretical_attention_bytes,
    theoretical_oom_guard,
)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    config_path: Path, *, device: str, output: Path, memory_budget_gb: float
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = torch.device(device)
    results: list[dict[str, Any]] = []
    for item in config["resolutions"]:
        width = int(item["width"])
        height = int(item["height"])
        steps = int(config["temporal_steps"])
        model = (
            EJEPATubeletLHR(
                EJEPATubeletLHRConfig(
                    in_channels=int(config["input_channels"]),
                    embed_dim=int(config["embed_dim"]),
                    patch_size=int(item["patch_size"]),
                    spatial_window=int(config["spatial_window"]),
                    heads=int(config["heads"]),
                    spatial_depth=1,
                    temporal_depth=int(config["temporal_depth"]),
                    merge_2x2=bool(item.get("merge_2x2", False)),
                    pooling="query",
                    memory_budget_gb=memory_budget_gb,
                )
            )
            .to(target)
            .eval()
        )
        inputs = torch.randn(
            1,
            steps,
            int(config["input_channels"]),
            height,
            width,
            device=target,
        )
        with torch.inference_mode():
            for _ in range(2):
                model.forward_features(inputs)
            _sync(target)
            start = time.perf_counter()
            output_value = model.forward_features(inputs)
            _sync(target)
            elapsed = time.perf_counter() - start
        patches = int(output_value.tokens.shape[2])
        total_tokens = steps * patches
        estimated_global_bytes = theoretical_attention_bytes(
            1, steps, patches, int(config["heads"])
        )
        guard_required = total_tokens >= 4800 or estimated_global_bytes > int(
            memory_budget_gb * (1024**3) // 2
        )
        guard_triggered = False
        global_guard_error = ""
        try:
            theoretical_oom_guard(
                batch=1,
                steps=steps,
                patches=patches,
                heads=int(config["heads"]),
                memory_budget_gb=memory_budget_gb,
                global_attention=True,
            )
        except RuntimeError as error:
            guard_triggered = True
            global_guard_error = str(error)
        results.append(
            {
                "name": str(item["name"]),
                "device": str(target),
                "source_resolution": [width, height],
                "patch_size": int(item["patch_size"]),
                "merge_2x2": bool(item.get("merge_2x2", False)),
                "tokens": total_tokens,
                "valid_tokens": int(output_value.valid_patch_mask.sum()),
                "padding_tokens": int((~output_value.valid_patch_mask).sum()),
                "forward_ms": elapsed * 1000.0,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "temporal_attention_pairs": int(
                    output_value.diagnostics["temporal_attention_pairs"]
                ),
                "spatial_attention_pairs": int(output_value.diagnostics["spatial_attention_pairs"]),
                "global_attention_pairs": total_tokens * total_tokens,
                "theoretical_oom_guard_required": guard_required,
                "theoretical_oom_guard_triggered": guard_triggered,
                "global_guard_error": global_guard_error,
            }
        )
        del model, inputs, output_value
        if target.type == "cuda":
            torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "artifact_type": "highres_token_scaling_benchmark_v1",
        "schema_version": "v1",
        "evidence_type": "synthetic_forward_scaling",
        "code_commit": _git_commit(),
        "protocol_version": "patch_resolution_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "device": str(target),
        "memory_budget_gb": memory_budget_gb,
        "global_attention_over_r4_allocated": False,
        "results": results,
        "status": "pass"
        if all(
            row["theoretical_oom_guard_required"] == row["theoretical_oom_guard_triggered"]
            for row in results
        )
        else "fail",
    }
    result["protocol_sha256"] = _canonical_sha256(config)
    result["artifact_sha256"] = _canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memory-budget-gb", type=float, default=12.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/highres_token_scaling_v1.json"),
    )
    args = parser.parse_args()
    result = benchmark(
        args.config,
        device=args.device,
        output=args.output,
        memory_budget_gb=args.memory_budget_gb,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
