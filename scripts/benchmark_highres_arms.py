"""Run comparable S3/S4/S5 forward-backward and causal smoke screens."""

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

from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig

ARMS: tuple[tuple[str, bool, str], ...] = (
    ("S3_R4_WINDOW_TEMPORAL", False, "block_causal"),
    ("S4_R4_WINDOW_MERGE_TEMPORAL", True, "block_causal"),
    ("S5_R4_WINDOW_MERGE_KDA", True, "kda"),
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _causal_smoke(model: EJEPATubeletLHR, inputs: torch.Tensor) -> bool:
    changed = inputs.clone()
    changed[:, -1] += 100.0
    with torch.inference_mode():
        first = model.forward_features(inputs).tokens
        second = model.forward_features(changed).tokens
    return bool(torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5, rtol=1e-5))


def benchmark(*, device: str, output: Path) -> dict[str, Any]:
    target = torch.device(device)
    torch.manual_seed(23)
    inputs = torch.randn(1, 5, 4, 192, 320, device=target)
    rows: list[dict[str, Any]] = []
    for name, merge, temporal_mixer in ARMS:
        torch.manual_seed(7)
        model = EJEPATubeletLHR(
            EJEPATubeletLHRConfig(
                in_channels=4,
                embed_dim=32,
                patch_size=8,
                spatial_window=8,
                heads=4,
                spatial_depth=1,
                temporal_depth=1,
                temporal_mixer=temporal_mixer,
                merge_2x2=merge,
                pooling="query",
            )
        ).to(target)
        model.train()
        start = time.perf_counter()
        output_value = model(inputs)
        loss = output_value.ttc_mean_seconds.mean() + output_value.collision_logits.square().mean()
        loss.backward()
        elapsed = time.perf_counter() - start
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        rows.append(
            {
                "arm": name,
                "merge_2x2": merge,
                "temporal_mixer": temporal_mixer,
                "readout": "query",
                "tokens": int(output_value.tokens.shape[1] * output_value.tokens.shape[2]),
                "valid_tokens": int(output_value.valid_patch_mask.sum()),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "loss": float(loss.detach()),
                "forward_backward_ms": elapsed * 1000.0,
                "finite_gradients": finite_gradients,
                "causal_smoke": _causal_smoke(model.eval(), inputs),
                "global_attention_used": False,
                "metrics_scope": "architecture_forward_backward_smoke",
            }
        )
        del model
        if target.type == "cuda":
            torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "artifact_type": "highres_architecture_screen_v1",
        "schema_version": "v1",
        "evidence_type": "synthetic_architecture_smoke",
        "code_commit": _git_commit(),
        "protocol_version": "highres_s3_s4_s5_v1",
        "protocol_sha256": _canonical_sha256({"arms": ARMS, "seed": 7, "input_seed": 23}),
        "created_at": datetime.now(UTC).isoformat(),
        "device": str(target),
        "selection_allowed": False,
        "metrics_scope": "forward_backward_smoke_only",
        "results": rows,
        "status": "pass"
        if all(row["finite_gradients"] and row["causal_smoke"] for row in rows)
        else "fail",
    }
    result["artifact_sha256"] = _canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/highres_architecture_screen_v1.json"),
    )
    args = parser.parse_args()
    result = benchmark(device=args.device, output=args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
