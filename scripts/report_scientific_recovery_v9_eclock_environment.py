#!/usr/bin/env python
"""Capture the immutable environment identity for one E-Clock X0 campaign."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact


def _command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--authorized-commit", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    head = _command("git", "-C", str(repo), "rev-parse", "HEAD")
    if head != args.authorized_commit:
        raise ValueError("environment report HEAD differs from authorized commit")
    disk = shutil.disk_usage(args.output.resolve().anchor)
    vm = psutil.virtual_memory()
    payload = sign_artifact(
        {
            "artifact_type": "eclock_x0_environment_v1",
            "git_commit": head,
            "git_branch": _command("git", "-C", str(repo), "branch", "--show-current"),
            "git_tracked_clean": not bool(
                _command("git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no")
            ),
            "platform": platform.platform(),
            "windows_version": platform.win32_ver(),
            "powershell_version": _command(
                "pwsh", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"
            ),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "uv_version": _command("uv", "--version"),
            "torch_version": torch.__version__,
            "torchvision_version": importlib.metadata.version("torchvision"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_bytes": (
                torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            ),
            "nvidia_smi": _command(
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            "cpu": platform.processor(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "ram_total_bytes": vm.total,
            "ram_available_bytes": vm.available,
            "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_MAX_THREADS",
                )
            },
            "cuda_determinism_environment": {
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "cache_root": str(args.cache_root.resolve()),
            "reference_root": str(args.reference_root.resolve()),
            "uv_lock_sha256": compute_file_hash(str(repo / "uv.lock")),
            "sealed_evaluation": {
                "public_validation_opened": False,
                "private_test_opened": False,
                "evttc_test_opened": False,
                "codabench_opened": False,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
