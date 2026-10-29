"""Capture package versions and code identities without importing Torch or exposing URLs."""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from runpod_final_diagnostics_telemetry import cgroup

PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "h5py",
    "hdf5plugin",
    "psutil",
    "pillow",
    "einops",
    "opencv-python",
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "triton",
    "huggingface-hub",
    "safetensors",
)


def identity(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        if stat.st_size > 8 * 1024**2:
            return {"path": str(path), "error": "EXCEEDS_CODE_OR_RECEIPT_LIMIT"}
        return {
            "path": str(path),
            "sha256_on_disk": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": stat.st_size,
            "mtime_unix_seconds": stat.st_mtime,
        }
    except OSError as error:
        return {"path": str(path), "error": type(error).__name__}


def capture() -> dict[str, Any]:
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    compiled_cuda = None
    torch_version_file = None
    try:
        torch_version_file = Path(metadata.distribution("torch").locate_file("torch/version.py"))
        for node in ast.parse(torch_version_file.read_text()).body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else ([node.target] if isinstance(node, ast.AnnAssign) else [])
            )
            if any(isinstance(target, ast.Name) and target.id == "cuda" for target in targets):
                compiled_cuda = ast.literal_eval(node.value)
    except (OSError, ValueError, metadata.PackageNotFoundError):
        pass
    cudnn: dict[str, Any] = {"torch_imported": False, "gpu_context_created": False}
    try:
        library = Path(
            metadata.distribution("nvidia-cudnn-cu12").locate_file("nvidia/cudnn/lib/libcudnn.so.9")
        )
        handle = ctypes.CDLL(str(library))
        handle.cudnnGetVersion.restype = ctypes.c_size_t
        cudnn.update(library_path=str(library), runtime_version_integer=handle.cudnnGetVersion())
    except (OSError, AttributeError, metadata.PackageNotFoundError) as error:
        cudnn["error"] = str(error)
    ops = Path("/workspace/ejepa_final_operations")
    paths = [
        ops / name
        for name in (
            "runpod_final_bootstrap.py",
            "runpod_final_queue.py",
            "runpod_test12_staged.py",
            "runpod_final_export_compat.py",
            "runpod_final_path_compat.py",
            "runpod_final_cache_compat.py",
            "runpod_final_lineage_compat.py",
            "runpod_final_phase_bridge.py",
            "runpod_final_diagnostics_telemetry.py",
        )
    ]
    base = Path("/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement")
    paths += [
        base / name
        for name in (
            "operational/final_refit/scientific_oof_final.py",
            "operational/final_refit/training.py",
            "operational/final_refit/oof_lineage.py",
            "operational/rgb_port/train_producers.py",
        )
    ]
    active = []
    for process in psutil.process_iter():
        try:
            # Only selected script paths leave the process inventory; no env or arbitrary argv.
            scripts = [
                Path(arg)
                for arg in process.cmdline()
                if arg.startswith(("/workspace/ejepa", "/opt/ejepa")) and arg.endswith(".py")
            ]
            if not scripts:
                continue
            record = {
                "pid": process.pid,
                "created_utc": datetime.fromtimestamp(process.create_time(), UTC).isoformat(),
                "scripts": [],
            }
            for script in scripts:
                item = identity(script)
                item["disk_newer_than_process_start"] = (
                    item.get("mtime_unix_seconds", 0) > process.create_time()
                )
                item["active_loaded_hash_proven"] = False
                record["scripts"].append(item)
                paths.append(script)
            active.append(record)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    export = Path("/workspace/ejepa_final_runs/final/PHASE_ONLY_HEAD.json")
    export_receipt = json.loads(export.read_text()) if export.is_file() else None
    return {
        "schema": "ejepa_environment_code_evidence_v1",
        "captured_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "platform": platform.platform(),
        "os_release": platform.freedesktop_os_release(),
        "packages": packages,
        "full_distribution_versions": sorted(
            (
                {
                    "name": distribution.metadata.get("Name", "UNKNOWN"),
                    "version": distribution.version,
                }
                for distribution in metadata.distributions()
            ),
            key=lambda item: (item["name"].casefold(), item["version"]),
        ),
        "torch_compiled_cuda": compiled_cuda,
        "torch_version_file": identity(torch_version_file) if torch_version_file else None,
        "cudnn_runtime": cudnn,
        "gpu_csv": gpu.stdout.strip(),
        "gpu_error": gpu.stderr.strip(),
        "visible_logical_cpus": psutil.cpu_count(),
        "visible_physical_cpus": psutil.cpu_count(logical=False),
        "process_cpu_affinity": psutil.Process().cpu_affinity(),
        "cgroup": cgroup(),
        "code_on_disk": [identity(path) for path in sorted(set(paths))],
        "active_process_scripts": active,
        "export_receipt": export_receipt,
        "export_receipt_identity": identity(export),
        "limitations": [
            "On-disk SHA does not prove code already loaded by a running interpreter.",
            "cuDNN version call allocates no training/model/GPU tensors.",
            "Head parity is not whole-pipeline latency or official evaluation.",
            "No private test targets, leaderboard measurements or official SOTA claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(capture(), indent=2))
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
