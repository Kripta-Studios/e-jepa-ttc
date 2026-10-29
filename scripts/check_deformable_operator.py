"""Assert real CUDA deformable-attention parity; never certify a detector.

Run after installing a reviewed port of the pinned candidate in its own venv.
The upstream test prints some failures without failing the process; this wrapper
uses assertions and writes a structured result. It performs no installation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def run_operator_check(source_root: Path, family: str, *, seed: int = 7) -> dict[str, Any]:
    """Execute forward/backward parity against the upstream grid-sample reference."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "deformable_operator_only_not_detector",
        "status": "BLOCKED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "family": family,
        "source_root": str(source_root.resolve()),
        "seed": seed,
        "dtype_results": [],
        "g1_passed": False,
    }
    try:
        import torch
    except ImportError as exc:
        report["reason"] = f"torch unavailable: {exc}"
        return report
    report["torch_version"] = torch.__version__
    report["cuda_build"] = torch.version.cuda
    if not torch.cuda.is_available():
        report["reason"] = "no accessible CUDA device"
        return report
    wrapper = source_root / "lib" / "models" / family / "ops/functions/ms_deform_attn_func.py"
    if not wrapper.is_file():
        report["reason"] = f"upstream wrapper missing: {wrapper}"
        return report
    report["source_wrapper_sha256"] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report["source_revision"] = head.stdout.strip() if head.returncode == 0 else None
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report["source_diff_sha256"] = hashlib.sha256(diff.stdout.encode()).hexdigest()
    try:
        spec = importlib.util.spec_from_file_location("phase_detection_msda_probe", wrapper)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load upstream operator wrapper")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        extension_path = Path(module.MSDA.__file__)
        report["extension_sha256"] = hashlib.sha256(extension_path.read_bytes()).hexdigest()
        report["gpu_name"] = torch.cuda.get_device_name()
        report["gpu_capability"] = list(torch.cuda.get_device_capability())
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        shapes = torch.tensor([[6, 4], [3, 2]], dtype=torch.long, device="cuda")
        starts = torch.tensor([0, 24], dtype=torch.long, device="cuda")
        for dtype, atol, rtol in ((torch.float32, 1e-4, 1e-3), (torch.float64, 1e-7, 1e-5)):
            value = torch.rand(1, 30, 2, 4, device="cuda", dtype=dtype, requires_grad=True)
            locations = torch.rand(1, 2, 2, 2, 2, 2, device="cuda", dtype=dtype)
            locations.requires_grad_()
            weights = torch.rand(1, 2, 2, 2, 2, device="cuda", dtype=dtype)
            weights = (weights / weights.sum((-1, -2), keepdim=True)).detach().requires_grad_()
            reference_inputs = [
                tensor.detach().clone().requires_grad_() for tensor in (value, locations, weights)
            ]
            output = module.MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, 1)
            reference = module.ms_deform_attn_core_pytorch(
                reference_inputs[0], shapes, reference_inputs[1], reference_inputs[2]
            )
            torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
            output.square().sum().backward()
            reference.square().sum().backward()
            if not torch.isfinite(output).all().item():
                raise AssertionError("non-finite operator output")
            for actual, expected in zip((value, locations, weights), reference_inputs, strict=True):
                if actual.grad is None or not torch.isfinite(actual.grad).all().item():
                    raise AssertionError("missing or non-finite operator gradient")
                torch.testing.assert_close(actual.grad, expected.grad, atol=atol, rtol=rtol)
            report["dtype_results"].append(
                {
                    "dtype": str(dtype),
                    "forward": "PASS",
                    "backward": "PASS",
                    "atol": atol,
                    "rtol": rtol,
                    "max_abs_forward_error": float((output - reference).abs().max().item()),
                }
            )
        torch.cuda.synchronize()
        report["operator_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["operator_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "PASS"
        report["reason"] = "operator parity only; full detector forward/loss/decode still required"
    except Exception as exc:
        report["status"] = "FAIL"
        report["reason"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> int:
    """Exit zero only for actual operator parity; GPU absence is exit two."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--family", choices=("monodgp", "monodetr"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_operator_check(args.source_root, args.family, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "reason": report["reason"]}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
