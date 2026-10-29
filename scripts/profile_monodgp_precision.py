"""Measure full real-image optimizer steps under explicit FP32/TF32/BF16 profiles."""

# ruff: noqa: E402 -- allow direct execution in the isolated detector environment.

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch
from smoke_monodgp_portable import build_detector, prepare_source, require_finite, sha256
from train_monodgp_portable import inputs_on_gpu, write_json

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample


def memory_snapshot() -> dict[str, int]:
    """Read this process and host memory counters on Windows without extra tools."""
    if sys.platform != "win32":
        return {}

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong)
            for name in (
                "total_phys",
                "avail_phys",
                "total_page",
                "avail_page",
                "total_virtual",
                "avail_virtual",
                "avail_extended",
            )
        ]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_ws",
                "ws",
                "peak_pool_paged",
                "pool_paged",
                "peak_pool_nonpaged",
                "pool_nonpaged",
                "pagefile",
                "peak_pagefile",
                "private",
            )
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    process = ProcessMemory()
    process.cb = ctypes.sizeof(process)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(-1), ctypes.byref(process), ctypes.sizeof(process)
    )
    return {
        "host_available_physical_bytes": status.avail_phys,
        "host_available_commit_bytes": status.avail_page,
        "process_working_set_bytes": process.ws,
        "process_private_bytes": process.private,
    }


def main() -> int:
    """Run three bounded precision profiles on the same training sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "output/detector_sources/MonoDGP"
    )
    parser.add_argument(
        "--portable-root", type=Path, default=ROOT / "output/detector_sources/MonoDGP-portable"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=10)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("fp32_highest", "fp32_high", "bf16_high"),
        default=["fp32_highest", "fp32_high", "bf16_high"],
    )
    args = parser.parse_args()
    tokens = json.loads(args.manifest.read_text(encoding="utf-8"))["overfit16"]["sample_tokens"]
    with EAPRGBDataset(args.data_root, tokens[:1]) as dataset:
        prepared = prepare_sample(dataset[0], width=640, height=384)
    images, calibration, targets, sizes = inputs_on_gpu(prepared)
    report = {
        "scope": "single_real_train_frame_precision_feasibility_not_accuracy_comparison",
        "sample_token": tokens[0],
        "manifest_sha256": sha256(args.manifest),
        "script_sha256": sha256(Path(__file__)),
        "torch": torch.__version__,
        "profiles": [],
        "warmup_steps": args.warmup,
        "measured_steps": args.measured,
        "seed": 7,
        "batch_size": 1,
        "image_hw": [384, 640],
        "group_num": 11,
        "memory_before_cuda_profile": memory_snapshot(),
        "source": prepare_source(
            args.source_root, args.portable_root, args.output.with_suffix(".patch")
        ),
        "operator_sha256": sha256(ROOT / "src/e_jepa_ttc/detection/portable_attention.py"),
        "reader_sha256": sha256(ROOT / "src/e_jepa_ttc/detection/eap_rgb_data.py"),
        "adapter_sha256": sha256(ROOT / "src/e_jepa_ttc/detection/monodgp_eap_adapter.py"),
        "geometry_sha256": sha256(ROOT / "src/e_jepa_ttc/detection/geometry_contract.py"),
    }
    torch.set_num_threads(4)
    reference = None
    for name, matmul, amp in (
        ("fp32_highest", "highest", False),
        ("fp32_high", "high", False),
        ("bf16_high", "high", True),
    ):
        if name not in args.profiles:
            continue
        item = {"name": name, "matmul_precision": matmul, "autocast_bf16": amp, "status": "FAIL"}
        try:
            if amp and not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA BF16 is not supported on the measured device.")
            torch.manual_seed(7)
            torch.set_float32_matmul_precision(matmul)
            model, criterion, _ = build_detector(args.portable_root, group_num=11)
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                probe = model(images, calibration, None, sizes)
            require_finite(probe)
            current = {
                key: value.detach().float().cpu()
                for key, value in probe.items()
                if isinstance(value, torch.Tensor)
            }
            if reference is None:
                reference = current
            item["eval_output_max_abs_difference_from_fp32_highest"] = {
                key: float((value - reference[key]).abs().max()) for key, value in current.items()
            }
            del probe
            model.train()
            criterion.train()
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), lr=2e-4
            )
            torch.cuda.reset_peak_memory_stats()
            rows = []
            for step in range(args.warmup + args.measured):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                event_start, event_end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                tick = time.perf_counter()
                event_start.record()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    outputs = model(images, calibration, targets, sizes)
                    require_finite(outputs)
                    losses = criterion(outputs, targets)
                    require_finite(losses, "losses")
                    loss = sum(
                        value * criterion.weight_dict[key]
                        for key, value in losses.items()
                        if key in criterion.weight_dict
                    )
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                event_end.record()
                torch.cuda.synchronize()
                rows.append(
                    {
                        "step": step,
                        "seconds": time.perf_counter() - tick,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                        "cuda_stream_span_ms": event_start.elapsed_time(event_end),
                        "memory": memory_snapshot(),
                    }
                )
                del outputs, losses, loss
            item.update(
                status="PASS",
                steps=rows,
                median_measured_seconds=statistics.median(
                    row["seconds"] for row in rows[args.warmup :]
                ),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            del model, criterion, optimizer
            torch.cuda.empty_cache()
        except Exception as exc:
            item.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        report["profiles"].append(item)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in item.items()
                    if key in ("name", "status", "median_measured_seconds", "error")
                }
            ),
            flush=True,
        )
        write_json(args.output, report)
    return 0 if all(item["status"] == "PASS" for item in report["profiles"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
